"""Shared dense/MoE layer scans and optional recurrent depth.

Attention implementations supply one residual block; this executor owns the
layer order, routing statistics and auxiliary-loss reduction for every tower.
"""
import jax
import jax.numpy as jnp


def rms_norm(x, gain):
    a = x.astype(jnp.float32)
    return (a * jax.lax.rsqrt(jnp.mean(a * a, -1, keepdims=True) + 1e-6) * gain).astype(x.dtype)


def run_layers(tokens, mask, params, ffn_cfg, bias, attention, loop_cfg=None):
    """Execute P -> core**R -> C without copying or resampling layer weights.

    A zero initial state preserves deterministic, permutation-equivariant set
    predictions. Variance-scaled addition reinjects the prelude at every pass,
    avoiding an extra learned projection. Only the final state is normalized.
    """
    import strategize_moe as moe
    p = dict(params)
    cfg = None if ffn_cfg is None else dict(ffn_cfg)
    loop = {} if loop_cfg is None else dict(loop_cfg)
    depth = p["W_q_layers"].shape[0]
    prefix = depth if cfg is None else int(cfg["first_k_dense"])
    ctx = None if cfg is None else moe._context.get()
    output_dtype = tokens.dtype
    if cfg is not None:
        tokens = tokens.astype(moe.compute_dtype(cfg, tokens.dtype))
        bias = ctx.bias if ctx is not None else bias
        if bias is None:
            raise ValueError("A saved MoE model must include frozen router biases")
        bias = jnp.asarray(bias, jnp.float32)
        if bias.shape != (depth - prefix, int(cfg["n_routed_experts"])):
            raise ValueError("MoE router bias shape does not match layer/expert counts")
    common = {name[:-7]: value for name, value in p.items()
              if name.endswith("_layers") and not name.startswith(("W_ff", "W_moe"))}
    remat = cfg is None or cfg.get("activation_checkpointing", True)
    if remat:
        attention = jax.checkpoint(attention, prevent_cse=False)

    def dense(x, layer):
        h, z, alpha, loss, count = attention(x, layer)
        y = moe.swiglu(z, layer["W_ff1"].astype(z.dtype), layer["W_ff2"].astype(z.dtype))
        return h + alpha.astype(h.dtype) * y, (loss, count)

    dense_step = jax.checkpoint(dense, prevent_cse=False) if remat else dense

    def routed(carry, layer):
        x, chain = carry
        h, z, alpha, loss, count = attention(x, layer)
        weights = [layer["W_moe_" + name] for name in ("router", "expert1", "expert2", "shared1", "shared2")]
        y, stats, chain = moe.dispatch(z, mask, *weights, layer["bias"], cfg,
            training=ctx is not None, row_mask=None if ctx is None else ctx.row_mask,
            runtime=None if ctx is None else ctx.runtime, chain=chain)
        return (h + alpha.astype(h.dtype) * y, chain), (stats, loss, count)

    def segment(carry, start, end):
        x, chain, stats, loss, count = carry
        stop = min(end, prefix)
        if start < stop:
            layers = {name: value[start:stop] for name, value in common.items()}
            layers.update({name: p[name + "_layers"][start:stop] for name in ("W_ff1", "W_ff2")})
            x, (ll, nn) = jax.lax.scan(dense_step, x, layers)
            loss, count = loss + ll.sum(), count + nn.sum()
        begin = max(start, prefix)
        if begin < end:
            lo, hi = begin - prefix, end - prefix
            layers = {name: value[begin:end] for name, value in common.items()}
            layers.update({name: value[lo:hi] for name, value in p.items()
                           if name.startswith("W_moe_") and name.endswith("_layers")})
            layers = {name.removesuffix("_layers"): value for name, value in layers.items()}
            layers["bias"] = bias[lo:hi]
            (x, chain), (ss, ll, nn) = jax.lax.scan(routed, (x, chain), layers)
            stats = stats.at[lo:hi].add(ss)
            loss, count = loss + ll.sum(), count + nn.sum()
        return x, chain, stats, loss, count

    stats = jnp.zeros((depth - prefix, 0 if cfg is None else 3 * int(cfg["n_routed_experts"]) + 3), jnp.float32)
    carry = (tokens, jnp.float32(0) if ctx is None else ctx.chain, stats, jnp.float32(0), jnp.float32(0))
    if not loop.get("enabled", False):
        carry = segment(carry, 0, depth)
    else:
        prelude, coda, iterations = (int(loop[k]) for k in ("prelude_layers", "coda_layers", "iterations"))
        if iterations < 1 or min(prelude, coda) < 0 or prelude + coda >= depth:
            raise ValueError("Recurrent depth requires positive iterations and a nonempty core")
        carry = segment(carry, 0, prelude)
        embedded = carry[0]
        if mask is not None:
            embedded = jnp.where(jnp.asarray(mask)[..., None] > 0, embedded, 0)
        prelude_loss, prelude_count = carry[3:]
        carry = (jnp.zeros_like(embedded), *carry[1:3], jnp.float32(0), jnp.float32(0))

        def recur(state, _):
            injected = (state[0] + embedded) * jnp.asarray(2 ** -.5, embedded.dtype)
            return segment((injected, *state[1:]), prelude, depth - coda), None

        backprop = int(loop.get("backprop_iterations", 0)) if loop.get("training", False) else 0
        if backprop < 0:
            raise ValueError("backprop_iterations must be nonnegative (0 means all)")
        burnin = max(0, iterations - backprop) if backprop else 0
        if burnin:
            carry, _ = jax.lax.scan(recur, carry, None, length=burnin)
            # Drop the early recurrent graph, retaining the live prelude input
            # in later injections. Loss/count values still cover every pass.
            carry = jax.tree.map(jax.lax.stop_gradient, carry)
        carry, _ = jax.lax.scan(recur, carry, None, length=iterations - burnin)
        carry = (*carry[:3], carry[3] + prelude_loss, carry[4] + prelude_count)
        carry = segment(carry, depth - coda, depth)
    tokens, chain, stats, loss, count = carry
    if ctx is not None:
        ctx.add(stats, chain)
    return rms_norm(tokens, p["RMS_final"]).astype(output_dtype), loss, count
