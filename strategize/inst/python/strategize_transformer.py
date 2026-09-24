"""Shared dense/MoE layer scans and optional recurrent depth.

Attention implementations supply one residual block; this executor owns the
layer order, routing statistics and auxiliary-loss reduction for every tower.
"""
import jax
import jax.numpy as jnp


def rms_norm(x, gain):
    a = x.astype(jnp.float32)
    return (a * jax.lax.rsqrt(jnp.mean(a * a, -1, keepdims=True) + 1e-6) * gain).astype(x.dtype)


def _mean_over_valid(values, mask):
    """Mean token statistic while ignoring padding."""
    if mask is None:
        return jnp.mean(values)
    valid = jnp.asarray(mask, jnp.float32)
    return jnp.sum(values * valid) / jnp.maximum(jnp.sum(valid), 1)


def attention_dtype(label, backend):
    """'auto' is float32 except for cuDNN, which only runs in half precision."""
    named = {"bf16": jnp.bfloat16, "bfloat16": jnp.bfloat16, "fp16": jnp.float16,
             "float16": jnp.float16, "f16": jnp.float16, "half": jnp.float16,
             "fp32": jnp.float32, "float32": jnp.float32, "f32": jnp.float32}
    return named.get(str(label or "auto").lower(), jnp.bfloat16 if backend == "cudnn" else jnp.float32)


def self_attention(Qh, Kh, Vh, token_mask, model_dims, n_heads, head_dim,
                   attention_backend, attention_dtype_label, attention_padding_multiple,
                   scale=None):
    """Key-masked bidirectional attention over [batch, token, head, width].

    'pallas' is the Triton flash kernel (CUDA and ROCm); padding forms its own
    segment, so valid queries see exactly the valid keys. Padded query rows stay
    finite but differ across backends; no caller reads them.
    """
    backend = str(attention_backend or "xla").lower()
    scale = float(head_dim) ** -.5 if scale is None else float(scale)
    if backend not in ("xla", "cudnn", "pallas") or not hasattr(jax.nn, "dot_product_attention"):
        scores = jnp.einsum("nqhd,nkhd->nhqk", Qh, Kh) * jnp.asarray(scale, Qh.dtype)
        if token_mask is not None:
            scores = jnp.where((token_mask > 0)[:, None, None, :], scores,
                               jnp.asarray(jnp.finfo(scores.dtype).min, scores.dtype))
        return jnp.einsum("nhqk,nkhd->nqhd", jax.nn.softmax(scores, axis=-1), Vh)
    n_batch, seq_len = Qh.shape[:2]
    dtype = attention_dtype(attention_dtype_label, backend)
    Q, K, V = (x.astype(dtype) for x in (Qh, Kh, Vh))
    mask = jnp.ones((n_batch, seq_len), jnp.float32) if token_mask is None else token_mask
    block = 16 if seq_len <= 16 else 32
    seq_use = seq_len
    if backend != "xla":
        multiple = block if backend == "pallas" else max(int(attention_padding_multiple), 1)
        seq_use = -(-seq_len // multiple) * multiple
        pad = ((0, 0), (0, seq_use - seq_len))
        Q, K, V = (jnp.pad(x, pad + ((0, 0), (0, 0))) for x in (Q, K, V))
        mask = jnp.pad(mask, pad)
    if backend == "pallas":
        from jax.experimental.pallas.ops.gpu import attention as pallas_attention
        context = pallas_attention.mha(Q, K, V, (mask > 0).astype(jnp.int32), sm_scale=scale,
                                       block_sizes=pallas_attention.BlockSizes(*[block] * 6))
    else:
        key_mask = (mask > 0)[:, None, None, :]
        if backend == "cudnn":
            key_mask = jnp.broadcast_to(key_mask, (n_batch, 1, seq_use, seq_use))
        context = jax.nn.dot_product_attention(Q, K, V, mask=key_mask, scale=scale,
                                               implementation=backend)
    return context[:, :seq_len].astype(Qh.dtype)


_pallas_status = {}


def pallas_status(head_dim):
    """'' when the flash kernel compiles and matches XLA at this head width.

    Probes once per width in a fresh thread: callers may be inside a JAX trace
    (trace state is thread-local), and the probe needs concrete values.
    """
    key = (int(head_dim), jax.default_backend())
    if key not in _pallas_status:
        def probe():
            try:
                x = jax.random.normal(jax.random.PRNGKey(0), (2, 20, 2, int(head_dim)), jnp.float32)
                mask = jnp.ones((2, 20)).at[1, 13:].set(0)

                def loss(q, backend):
                    out = self_attention(q, q, q, mask, 0, 2, head_dim, backend, "float32", 8)
                    return jnp.sum(jnp.where(mask[..., None, None] > 0, out, 0) ** 2)
                got = jax.jit(jax.value_and_grad(lambda q: loss(q, "pallas")))(x)
                want = jax.value_and_grad(lambda q: loss(q, "xla"))(x)
                ok = all(bool(jnp.allclose(a, b, rtol=2e-2, atol=2e-2)) for a, b in zip(got, want))
                return "" if ok else "pallas_probe_mismatch"
            except Exception as error:  # no Triton, unsupported GPU, shared-memory limit
                return "pallas_probe_failed: " + (str(error).splitlines() or [type(error).__name__])[0][:160]
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(1) as pool:
            _pallas_status[key] = pool.submit(probe).result()
    return _pallas_status[key]


def _state_diagnostics(states, initial, mask, active_iterations):
    previous = jnp.concatenate((initial[None], states[:-1]), axis=0)
    norm = jnp.linalg.norm(states.astype(jnp.float32), axis=-1)
    delta = jnp.linalg.norm((states - previous).astype(jnp.float32), axis=-1)
    state_rms = jnp.sqrt(jnp.mean(states.astype(jnp.float32) ** 2, axis=-1))
    dot = jnp.sum(states.astype(jnp.float32) * previous.astype(jnp.float32), axis=-1)
    previous_norm = jnp.linalg.norm(previous.astype(jnp.float32), axis=-1)
    cosine = dot / jnp.maximum(norm * previous_norm, 1e-12)
    means = lambda x: jax.vmap(lambda row: _mean_over_valid(row, mask))(x)
    iteration = jnp.arange(states.shape[0], dtype=jnp.int32)
    active = iteration < active_iterations
    nan = jnp.asarray(jnp.nan, jnp.float32)
    cosine_mean = means(cosine).at[0].set(nan)
    return {
        "state_l2_mean": jnp.where(active, means(norm), nan),
        "state_rms_mean": jnp.where(active, means(state_rms), nan),
        "state_delta_l2_mean": jnp.where(active, means(delta), nan),
        "state_cosine_previous": jnp.where(active, cosine_mean, nan),
    }


def _routing_diagnostics(pass_stats, routes, mask, active_iterations, experts):
    """Summarize expert load and exact top-k assignment changes per pass."""
    count = pass_stats.shape[0]
    active = jnp.arange(count, dtype=jnp.int32) < active_iterations
    nan = jnp.asarray(jnp.nan, jnp.float32)
    if experts <= 0 or pass_stats.shape[1] == 0:
        return {
            "moe_load_entropy": jnp.full((count,), nan),
            "moe_route_change_fraction": jnp.full((count,), nan),
        }
    attempts = pass_stats[:, :, :experts].sum(axis=1)
    probability = attempts / jnp.maximum(attempts.sum(axis=-1, keepdims=True), 1)
    entropy = -jnp.sum(jnp.where(probability > 0, probability * jnp.log(probability), 0), axis=-1)
    if experts > 1:
        entropy = entropy / jnp.log(jnp.asarray(experts, jnp.float32))
    previous = jnp.concatenate((routes[:1], routes[:-1]), axis=0)
    comparable = (routes >= 0) & (previous >= 0)
    if mask is not None:
        valid = jnp.asarray(mask) > 0
        # routes: recurrence, routed-layer, batch, token, top-k
        comparable = comparable & valid[None, None, :, :, None]
    changed = (routes != previous) & comparable
    axes = tuple(range(1, routes.ndim))
    route_change = changed.sum(axis=axes) / jnp.maximum(comparable.sum(axis=axes), 1)
    route_change = route_change.at[0].set(nan)
    return {
        "moe_load_entropy": jnp.where(active, entropy, nan),
        "moe_route_change_fraction": jnp.where(active, route_change, nan),
    }


def run_layers(tokens, mask, params, ffn_cfg, bias, attention, loop_cfg=None,
               return_details=False):
    """Execute P -> core**R -> C without copying or resampling weights.

    ``active_iterations`` may be a traced scalar sampled once per optimizer
    update. ``max_iterations`` remains static, so one compiled scan supports
    every sampled recurrent depth. The first gated pass receives the prelude
    representation directly; later passes use learned positive state/input
    gains followed by RMS normalization.
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
        h, z, alpha, loss, n = attention(x, layer)
        y = moe.swiglu(z, layer["W_ff1"].astype(z.dtype), layer["W_ff2"].astype(z.dtype))
        return h + alpha.astype(h.dtype) * y, (loss, n)

    dense_step = jax.checkpoint(dense, prevent_cse=False) if remat else dense

    def routed(carry, layer):
        x, chain = carry
        h, z, alpha, loss, n = attention(x, layer)
        weights = [layer["W_moe_" + name] for name in ("router", "expert1", "expert2", "shared1", "shared2")]
        y, stats, chain, routes = moe.dispatch(
            z, mask, *weights, layer["bias"], cfg,
            training=ctx is not None, row_mask=None if ctx is None else ctx.row_mask,
            runtime=None if ctx is None else ctx.runtime, chain=chain,
            return_routes=True)
        return (h + alpha.astype(h.dtype) * y, chain), (stats, loss, n, routes)

    route_k = 0 if cfg is None else int(cfg["n_experts_per_tok"])

    def empty_routes(x):
        return jnp.empty((0, *x.shape[:-1], route_k), dtype=jnp.int32)

    def segment(carry, start, end):
        x, chain, stats, loss, n = carry
        stop = min(end, prefix)
        if start < stop:
            layers = {name: value[start:stop] for name, value in common.items()}
            layers.update({name: p[name + "_layers"][start:stop] for name in ("W_ff1", "W_ff2")})
            x, (ll, nn) = jax.lax.scan(dense_step, x, layers)
            loss, n = loss + ll.sum(), n + nn.sum()
        routes = empty_routes(x)
        begin = max(start, prefix)
        if begin < end:
            lo, hi = begin - prefix, end - prefix
            layers = {name: value[begin:end] for name, value in common.items()}
            layers.update({name: value[lo:hi] for name, value in p.items()
                           if name.startswith("W_moe_") and name.endswith("_layers")})
            layers = {name.removesuffix("_layers"): value for name, value in layers.items()}
            layers["bias"] = bias[lo:hi]
            (x, chain), (ss, ll, nn, routes) = jax.lax.scan(routed, (x, chain), layers)
            stats = stats.at[lo:hi].add(ss)
            loss, n = loss + ll.sum(), n + nn.sum()
        return (x, chain, stats, loss, n), routes

    stats = jnp.zeros((depth - prefix, 0 if cfg is None else 3 * int(cfg["n_routed_experts"]) + 3), jnp.float32)
    carry = (tokens, jnp.float32(0) if ctx is None else ctx.chain, stats, jnp.float32(0), jnp.float32(0))
    details = None
    if not loop.get("enabled", False):
        carry, _ = segment(carry, 0, depth)
    else:
        prelude, coda = (int(loop[k]) for k in ("prelude_layers", "coda_layers"))
        max_iterations = int(loop.get("max_iterations", loop["iterations"]))
        active_iterations = jnp.asarray(loop.get("active_iterations", loop["iterations"]), jnp.int32)
        if max_iterations < 1 or min(prelude, coda) < 0 or prelude + coda >= depth:
            raise ValueError("Recurrent depth requires positive iterations and a nonempty core")
        carry, _ = segment(carry, 0, prelude)
        embedded = carry[0]
        if mask is not None:
            embedded = jnp.where(jnp.asarray(mask)[..., None] > 0, embedded, 0)
        prelude_loss, prelude_count = carry[3:]
        carry = (jnp.zeros_like(embedded), *carry[1:3], jnp.float32(0), jnp.float32(0))
        reinjection = str(loop.get("reinjection", "scaled_add"))
        if reinjection not in ("scaled_add", "rms_gated"):
            raise ValueError("reinjection must be 'scaled_add' or 'rms_gated'")
        state_gain = jnp.asarray(p.get("loop_state_gain", jnp.ones((tokens.shape[-1],), jnp.float32)))
        input_gain = jnp.asarray(p.get("loop_input_gain", jnp.ones((tokens.shape[-1],), jnp.float32)))
        one = jnp.ones((tokens.shape[-1],), jnp.float32)

        def inject(state, iteration):
            if reinjection == "scaled_add":
                return (state + embedded) * jnp.asarray(2 ** -.5, embedded.dtype)
            combined = rms_norm(
                state * state_gain.astype(state.dtype) + embedded * input_gain.astype(embedded.dtype),
                one)
            return jax.lax.cond(iteration == 0, lambda _: embedded, lambda _: combined, operand=None)

        backprop = int(loop.get("backprop_iterations", 0)) if loop.get("training", False) else 0
        if backprop < 0:
            raise ValueError("backprop_iterations must be nonnegative (0 means all)")
        detach_at = jnp.maximum(active_iterations - backprop, 0)
        routed_core_layers = max(0, (depth - coda) - max(prelude, prefix))
        route_shape = (routed_core_layers, *tokens.shape[:-1], route_k)

        def recur(state, iteration):
            if backprop:
                should_detach = (active_iterations > backprop) & (iteration == detach_at)
                state = jax.lax.cond(
                    should_detach,
                    lambda value: jax.tree.map(jax.lax.stop_gradient, value),
                    lambda value: value,
                    state)

            def active(value):
                before_stats = value[2]
                result, routes = segment(
                    (inject(value[0], iteration), *value[1:]), prelude, depth - coda)
                return result, (result[0], routes, result[2] - before_stats)

            def inactive(value):
                routes = jnp.full(route_shape, -1, dtype=jnp.int32)
                return value, (value[0], routes, jnp.zeros_like(value[2]))

            return jax.lax.cond(iteration < active_iterations, active, inactive, state)

        carry, (states, routes, pass_stats) = jax.lax.scan(
            recur, carry, jnp.arange(max_iterations, dtype=jnp.int32))
        carry = (*carry[:3], carry[3] + prelude_loss, carry[4] + prelude_count)
        carry, _ = segment(carry, depth - coda, depth)

        if return_details:
            details = _state_diagnostics(states, jnp.zeros_like(embedded), mask, active_iterations)
            details.update(_routing_diagnostics(
                pass_stats, routes, mask, active_iterations,
                0 if cfg is None else int(cfg["n_routed_experts"])))
            details.update({
                "active_iterations": active_iterations,
                "max_iterations": jnp.asarray(max_iterations, jnp.int32),
                "states": states,
                "moe_pass_stats": pass_stats,
            })

            power_steps = int(loop.get("jacobian_power_iterations", 0))
            if power_steps > 0:
                final_state = states[-1]
                base_chain = jnp.float32(0) if ctx is None else ctx.chain

                def core_map(state):
                    base = (inject(state, jnp.maximum(active_iterations, 1)), base_chain,
                            jnp.zeros_like(stats), jnp.float32(0), jnp.float32(0))
                    mapped, _ = segment(base, prelude, depth - coda)
                    return mapped[0]

                vector = jnp.ones_like(final_state, jnp.float32)
                vector = vector / jnp.maximum(jnp.linalg.norm(vector), 1e-12)
                for _ in range(power_steps):
                    _, jv = jax.jvp(core_map, (final_state,), (vector.astype(final_state.dtype),))
                    _, pullback = jax.vjp(core_map, final_state)
                    vector = pullback(jv)[0].astype(jnp.float32)
                    vector = vector / jnp.maximum(jnp.linalg.norm(vector), 1e-12)
                _, jv = jax.jvp(core_map, (final_state,), (vector.astype(final_state.dtype),))
                details["core_jacobian_spectral_norm"] = jnp.linalg.norm(jv.astype(jnp.float32))

    tokens, chain, stats, loss, n = carry
    if ctx is not None:
        ctx.add(stats, chain)
    output = rms_norm(tokens, p["RMS_final"]).astype(output_dtype)
    if return_details:
        result = {} if details is None else details
        result.update({"output": output, "loss": loss, "count": n})
        return result
    return output, loss, n
