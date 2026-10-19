"""Transformer MoE: local expert computation, global capacity, explicit SVI state.

The trace context only connects the R model's transformer calls while tracing.
All changing numerical state is an input/output of SVI, never a cached Python
array. Training traces on accelerators compute experts with grouped matmuls;
prediction uses the same routing with exact, bounded-memory block dispatch.
"""
from __future__ import annotations

from contextvars import ContextVar
import functools
import os

import jax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
import numpyro
from numpyro.primitives import mutable as mutable_site
from numpyro.infer import SVI
from numpyro.infer.elbo import ELBO
from numpyro.infer.svi import SVIState, _make_loss_fn

STATE = "_strategize_transformer_moe"
# Routing decisions (selected experts and capacity acceptance). A recurrence
# checkpoint saves them, so its backward recompute never reruns top-k or the
# distributed capacity all_gather.
ROUTING_DECISION = "strategize_moe_routing"
_context = ContextVar("strategize_moe_trace", default=None)
# Set while a training trace is batched (vectorized particles) or differentiated
# in forward mode; native ragged_dot supports neither with shared weights.
_blocks_only = ContextVar("strategize_moe_blocks_only", default=False)
EXPERT_KERNELS = ("auto", "grouped", "blocks")


def route(x, router, bias, top_k, scale):
    scores = jax.nn.sigmoid(x.astype(jnp.float32) @ router.astype(jnp.float32))
    _, indices = jax.lax.top_k(scores + jax.lax.stop_gradient(bias), top_k)
    indices = checkpoint_name(indices, ROUTING_DECISION)
    gates = jnp.take_along_axis(scores, indices, axis=-1)
    gates = gates / jnp.maximum(gates.sum(-1, keepdims=True), 1e-9) * scale
    return indices, gates


def swiglu(x, w1, w2):
    gate, value = jnp.split(x @ w1, 2, axis=-1)
    return (jax.nn.silu(gate) * value) @ w2


def selected_experts(x, w1, w2, indices, keep=None):
    """Compute accepted assignments in compact, single-expert matrix blocks.

    Capacity is decided by the caller, before sorting. Each block gathers one
    expert matrix, not a copy per token. Rematerialize the block computation so
    reverse-mode AD retains neither gathered weights nor expanded FFN values.
    The loop bound allows at most one partial block per expert; it is independent
    of the global routing capacity and works under grad, jvp, vmap and scan.
    Blocks return their rows and one scatter follows the loop, so no iteration
    carries (or, in reverse mode, differentiates) the whole output.
    """
    n, k = indices.shape
    if not n:
        return jnp.zeros((n, k, x.shape[-1]), x.dtype)
    experts, assignments = w1.shape[0], n * k
    keep = jnp.ones_like(indices, bool) if keep is None else keep
    expert_ids = jnp.where(keep.reshape(-1), indices.reshape(-1), experts)
    order = jnp.argsort(expert_ids, stable=True)
    counts = jnp.bincount(expert_ids, length=experts + 1)[:experts]
    starts = jnp.cumsum(counts) - counts
    block_size = min(64, assignments)
    blocks = (counts + block_size - 1) // block_size
    block_ends = jnp.cumsum(blocks)
    # ceil(sum(counts)/B) + E-1 bounds sum(ceil(counts[e]/B)).
    max_blocks = (assignments + block_size - 1) // block_size + experts - 1
    offsets = jnp.arange(block_size)

    def compute(expert, positions, valid):
        rows = jnp.where(valid[:, None], x[positions // k], 0)
        values = swiglu(rows, w1[expert].astype(x.dtype), w2[expert].astype(x.dtype))
        return jnp.where(valid[:, None], values, 0)

    # The conditional belongs INSIDE remat. Otherwise its backward residuals
    # save all closed-over weights on every iteration, even though the
    # active branch checkpoints its matrix multiplies.
    @functools.partial(jax.checkpoint, prevent_cse=False)
    def block(i):
        # Padding and idle blocks target distinct rows past the end, which the
        # scatter drops, so every index is unique.
        unused = assignments + i * block_size + offsets
        def active(_):
            expert = jnp.sum(i >= block_ends)
            within = (i - (block_ends[expert] - blocks[expert])) * block_size + offsets
            valid = within < counts[expert]
            positions = order[jnp.minimum(starts[expert] + within, assignments - 1)]
            return compute(expert, positions, valid), jnp.where(valid, positions, unused)
        def idle(_):
            return jnp.zeros((block_size, x.shape[-1]), x.dtype), unused
        return jax.lax.cond(i < block_ends[-1], active, idle, None)

    # Carrying the output through the loop made each backward iteration touch
    # every assignment, quadratic in the routing-group size.
    values, positions = jax.lax.map(block, jnp.arange(max_blocks))
    out = jnp.zeros((assignments, x.shape[-1]), x.dtype).at[positions.reshape(-1)].set(
        values.reshape(-1, x.shape[-1]), mode="drop", unique_indices=True)
    return out.reshape(n, k, x.shape[-1])


def grouped_experts(x, w1, w2, indices, keep=None):
    """Compute accepted assignments with grouped (ragged) expert matmuls.

    Same values as ``selected_experts``, without its block loop: each block's
    backward there materializes full-size weight and token cotangents. Dropped
    assignments sort after every expert group, where ragged_dot returns zero
    rows, and are masked again. Native ragged_dot only batches when every operand
    is batched, so vmapped callers and forward mode keep ``selected_experts``.
    """
    n, k = indices.shape
    if not n:
        return jnp.zeros((n, k, x.shape[-1]), x.dtype)
    experts = w1.shape[0]
    keep = jnp.ones_like(indices, bool) if keep is None else keep
    accepted = keep.reshape(-1)
    expert_ids = jnp.where(accepted, indices.reshape(-1), experts)
    order = jnp.argsort(expert_ids, stable=True)
    sizes = jnp.bincount(expert_ids, length=experts + 1)[:experts].astype(jnp.int32)
    rows = jnp.where(accepted[order, None], x[order // k], 0)
    gate, value = jnp.split(jax.lax.ragged_dot(rows, w1.astype(x.dtype), sizes), 2, axis=-1)
    values = jax.lax.ragged_dot(jax.nn.silu(gate) * value, w2.astype(x.dtype), sizes)
    # order is a permutation. Without the declaration, differentiating the
    # scatter uses JAX's general rule and builds an operand-sized ID buffer.
    out = jnp.zeros_like(values).at[order].set(values, unique_indices=True)
    return jnp.where(keep[..., None], out.reshape(n, k, x.shape[-1]), 0)


def use_grouped_experts(training):
    """Grouped matmuls for the training trace on accelerators; blocks otherwise.

    STRATEGIZE_MOE_EXPERT_KERNEL ('auto', 'grouped' or 'blocks') overrides the
    choice, except where grouped matmuls cannot run (outside training, batched
    particles, forward mode). CPU lowers ragged_dot to one dense product per
    expert, so 'auto' keeps the blocks there.
    """
    kernel = os.environ.get("STRATEGIZE_MOE_EXPERT_KERNEL", "auto")
    if kernel not in EXPERT_KERNELS:
        raise ValueError(f"STRATEGIZE_MOE_EXPERT_KERNEL must be one of {EXPERT_KERNELS}")
    if not training or _blocks_only.get():
        return False
    if kernel != "auto":
        return kernel == "grouped"
    return jax.default_backend() in ("gpu", "tpu")


def compute_dtype(cfg, fallback):
    # Missing metadata belongs to an older FP32 model. New fits explicitly save
    # their compute dtype; inference never chooses precision from its hardware.
    return jnp.bfloat16 if cfg.get("compute_dtype", "float32") == "bfloat16" else fallback


def expert_output(x, w1, w2, shared1, shared2, indices, gates, keep, grouped=False):
    out = (grouped_experts if grouped else selected_experts)(x, w1, w2, indices, keep)
    return (out.astype(jnp.float32) * gates[..., None]).sum(1).astype(x.dtype) + swiglu(
        x, shared1.astype(x.dtype), shared2.astype(x.dtype))


def _depend(value, chain):
    return jax.lax.optimization_barrier(
        jnp.where(jnp.isfinite(chain), value, jnp.full_like(value, jnp.nan)))


def dispatch(x, mask, router, w1, w2, shared1, shared2, bias, cfg,
             *, training=False, row_mask=None, runtime=None, chain=None,
             return_routes=False):
    """Return output, local detached statistics, and collective dependency.

    A routing group follows [repeat, global observation, token] order. Repeats
    account for R's concatenated left/right candidate batches. Device padding
    and zero-weight logical dummy observations do not consume capacity.
    """
    shape, d = x.shape, x.shape[-1]
    x2 = x.reshape(-1, d)
    n, experts, k = x2.shape[0], int(cfg["n_routed_experts"]), int(cfg["n_experts_per_tok"])
    if not n:
        result = (x, jnp.zeros((3 * experts + 3,), jnp.float32), jnp.float32(0))
        if return_routes:
            return (*result, jnp.empty((*shape[:-1], k), dtype=jnp.int32))
        return result
    valid = jnp.ones(shape[:-1], bool) if mask is None else jnp.asarray(mask) > 0
    rows = shape[0] if row_mask is None else row_mask.shape[0]
    if shape[0] % rows:
        raise ValueError("MoE rows must be complete repetitions of the observation layout")
    repeats = shape[0] // rows
    if row_mask is not None:
        valid = valid & jnp.tile(row_mask > 0, repeats)[:, None]
    valid = valid.reshape(-1)
    indices, gates = route(x2, router, bias, k, float(cfg["routed_scaling_factor"]))
    assignments = jax.nn.one_hot(indices.reshape(-1), experts, dtype=jnp.int32)
    assignments = assignments * jnp.repeat(valid, k)[:, None]
    chain = jnp.float32(0) if chain is None else chain
    distributed = runtime is not None and runtime.in_svi_shard

    if training:
        total_bound = n * (runtime.device_count if distributed else 1)
        if total_bound * k > 2**24:
            raise ValueError("MoE supports at most 2**24 assignments per global routing group")
        counts = assignments.reshape(repeats, -1, experts).sum(1).astype(jnp.float32)
        token_counts = valid.reshape(repeats, -1).sum(1).astype(jnp.float32)
        stats = jnp.concatenate((counts, token_counts[:, None]), axis=1)
        if distributed:
            gathered = jax.lax.all_gather(_depend(stats, chain), "data", axis=0)
            chain = gathered.sum()
            # Every preceding repeat on all devices, then preceding devices
            # within this repeat. Rank-major ordering would break pair layouts.
            all_counts = gathered[:, :, :experts]
            repeat_counts = all_counts.sum(0)
            prefix_repeat = jnp.cumsum(repeat_counts, axis=0) - repeat_counts
            preceding = jnp.arange(runtime.device_count) < jax.lax.axis_index("data")
            prefix_device = jnp.where(preceding[:, None, None], all_counts, 0).sum(0)
            prefix = (prefix_repeat + prefix_device).astype(jnp.int32)
            tokens = gathered[:, :, experts].sum()
        else:
            prefix = (jnp.cumsum(counts, axis=0) - counts).astype(jnp.int32)
            tokens = token_counts.sum()
        cf = float(cfg["capacity_factor"])
        cutoff = jnp.maximum(jnp.floor(cf * tokens * k / experts), jnp.minimum(tokens, 256)).astype(jnp.int32)
        grouped = assignments.reshape(repeats, -1, experts)
        ranks = (jnp.cumsum(grouped, axis=1) * grouped).sum(-1) - 1
        flat_e = indices.reshape(repeats, -1)
        global_rank = ranks + jnp.take_along_axis(prefix, flat_e, axis=1)
        keep = ((global_rank < cutoff) & (ranks >= 0)).reshape(n, k) & valid[:, None]
    else:
        keep = jnp.broadcast_to(valid[:, None], (n, k))
    keep = checkpoint_name(keep, ROUTING_DECISION)
    # This remat boundary contains no collectives. Routing/all_gather above runs
    # exactly once in the forward pass, in the qualified global order; an
    # enclosing recurrence checkpoint saves ROUTING_DECISION rather than rerun it.
    compute = functools.partial(expert_output, grouped=use_grouped_experts(training))
    if cfg.get("activation_checkpointing", True):
        compute = jax.checkpoint(compute, prevent_cse=False)
    y = compute(x2.astype(compute_dtype(cfg, x2.dtype)), w1, w2, shared1, shared2, indices, gates, keep)
    y = y.astype(x.dtype)
    y = jnp.where(valid[:, None], y, 0).reshape(shape)
    attempted = assignments.astype(jnp.float32).sum(0)
    accepted = (assignments * keep.reshape(-1, 1)).astype(jnp.float32).sum(0)
    detail = jnp.concatenate((attempted, accepted, attempted - accepted,
        jnp.stack((valid.astype(jnp.float32).sum(), (gates * valid[:, None]).sum(),
                   (gates * (valid[:, None] & ~keep)).sum()))))
    result = (y, jax.lax.stop_gradient(detail), jax.lax.stop_gradient(chain))
    if return_routes:
        routed = jnp.where(valid[:, None], indices, -1).reshape(*shape[:-1], k)
        return (*result, jax.lax.stop_gradient(routed))
    return result


class TraceContext:
    def __init__(self, cfg, bias, runtime):
        self.cfg, self.bias, self.runtime = cfg, bias, runtime
        self.row_mask = None
        self.stats = jnp.zeros((bias.shape[0], 3 * bias.shape[1] + 3), jnp.float32)

    @property
    def chain(self):
        return self.runtime.collective_chain if self.runtime is not None and self.runtime.in_svi_shard else jnp.float32(0)

    def add(self, stats, chain):
        self.stats = self.stats + stats
        if self.runtime is not None and self.runtime.in_svi_shard:
            self.runtime.collective_chain = chain

    def finish(self):
        stats = self.stats
        if self.runtime is not None and self.runtime.in_svi_shard:
            stats = jax.lax.psum(_depend(stats, self.chain), "data")
            self.runtime.collective_chain = stats.sum()
        return jax.lax.stop_gradient(stats)


def set_branch(branch, scale, rows):
    ctx = _context.get()
    if ctx is None:
        return
    rows = int(rows)
    mask = jnp.ones((rows,), bool) if scale is None else jnp.broadcast_to(jnp.asarray(scale) > 0, (rows,))
    if ctx.runtime is not None and ctx.runtime.in_svi_shard:
        ctx.runtime.set_branch(str(branch))
        n, local_n = ctx.runtime._branch_rows()
        if rows != local_n:
            raise ValueError("MoE branch does not match the distributed observation layout")
        mask = mask & (jnp.arange(rows) + jax.lax.axis_index("data") * local_n < n)
    ctx.row_mask = mask


def current_config():
    ctx = _context.get()
    return None if ctx is None else ctx.cfg


def wrap_model(model, cfg, initial_bias=None, runtime=None):
    cfg = dict(cfg)
    shape = (int(cfg["n_moe_layers"]), int(cfg["n_routed_experts"]))
    bias = jnp.zeros(shape, jnp.float32) if initial_bias is None else jnp.asarray(initial_bias, jnp.float32)
    if bias.shape != shape:
        raise ValueError("MoE initial router bias does not match the architecture")
    @functools.wraps(model)
    def wrapped(*args, **kwargs):
        state = mutable_site(STATE, {"bias": bias, "stats": jnp.zeros((shape[0], 3 * shape[1] + 3), jnp.float32),
                                        "updates": jnp.int32(0)})
        ctx = TraceContext(cfg, state["bias"], runtime)
        token = _context.set(ctx)
        try:
            result = model(*args, **kwargs)
            state["stats"] = ctx.finish()
            return result
        finally:
            _context.reset(token)
    return wrapped


def unwrap_model(model):
    return model.__wrapped__


class MoEELBO(ELBO):
    """Preserve the chosen ELBO and its RNG convention, including all particles."""
    def __init__(self, base):
        super().__init__(num_particles=base.num_particles, vectorize_particles=base.vectorize_particles)
        import copy
        self.base = copy.copy(base)
        self.base.num_particles = 1

    def loss_with_mutable_state(self, rng_key, param_map, model, guide, *args, **kwargs):
        def one(key):
            # Mutable-site dictionaries are private to this particle trace.
            params = jax.tree.map(lambda x: x, param_map)
            return self.base.loss_with_mutable_state(key, params, model, guide, *args, **kwargs)
        if self.num_particles == 1:
            return one(rng_key)
        # Only False selects jax.lax.map; True and custom mappers batch the trace.
        token = _blocks_only.set(_blocks_only.get() or self.vectorize_particles is not False)
        try:
            result = self.vectorize_particles_fn(one, jax.random.split(rng_key, self.num_particles))
        finally:
            _blocks_only.reset(token)
        states = result["mutable_state"]
        state = jax.tree.map(lambda x: x[0], states)
        state[STATE]["stats"] = states[STATE]["stats"].sum(0)
        return {"loss": jax.tree.map(lambda x: x.mean(0), result["loss"]), "mutable_state": state}


def commit_state(old, candidate, accepted, cfg):
    if set(old or {}) != {STATE} or set(candidate or {}) != {STATE}:
        raise ValueError("MoE SVI only supports its explicit transformer routing state")
    previous, current = old[STATE], dict(candidate[STATE])
    counts = current["stats"][:, :int(cfg["n_routed_experts"])]
    delta = float(cfg["router_bias_rate"]) * jnp.sign(counts.mean(-1, keepdims=True) - counts)
    current["bias"] = previous["bias"] + jnp.where(counts.sum(-1, keepdims=True) > 0, delta, 0)
    current["updates"] = previous["updates"] + jnp.int32(1)
    return jax.tree.map(lambda a, b: jnp.where(accepted, a, b), {STATE: current}, old)


class MoESVI(SVI):
    def __init__(self, model, guide, optim, loss, cfg):
        super().__init__(model, guide, optim, MoEELBO(loss))
        self.moe_config = dict(cfg)

    def init(self, *args, **kwargs):
        init_params = kwargs.get("init_params")
        frozen_bias = None
        if init_params is not None and "_transformer_moe_bias" in init_params:
            init_params = dict(init_params)
            frozen_bias = init_params.pop("_transformer_moe_bias")
            kwargs["init_params"] = init_params
        state = super().init(*args, **kwargs)
        mutable = jax.tree.map(lambda x: x, state.mutable_state)
        if frozen_bias is not None:
            if frozen_bias.shape != mutable[STATE]["bias"].shape:
                raise ValueError("Warm-start router bias does not match the MoE architecture")
            mutable[STATE]["bias"] = jnp.asarray(frozen_bias, jnp.float32)
        mutable[STATE]["stats"] = jnp.zeros_like(mutable[STATE]["stats"])
        mutable[STATE]["updates"] = jnp.int32(0)
        return SVIState(state.optim_state, mutable, state.rng_key)

    def get_params(self, state):
        # Carry the matching frozen routing state through existing median,
        # validation, best-parameter and parameter-only checkpoint workflows.
        # This key is never part of the optimizer or theta schema.
        return {**super().get_params(state), "_transformer_moe_bias": router_bias(state)}

    def stable_update(self, state, *args, forward_mode_differentiation=False, **kwargs):
        rng, step_rng = jax.random.split(state.rng_key)
        loss_fn = _make_loss_fn(self.loss, step_rng, self.constrain_fn, self.model, self.guide,
                               args, kwargs, self.static_kwargs, mutable_state=state.mutable_state)
        derivative = jax.jacfwd if forward_mode_differentiation else jax.value_and_grad
        if forward_mode_differentiation:
            # jacfwd batches tangents over shared weights: keep the expert blocks.
            token = _blocks_only.set(True)
            try:
                (loss, mutable) = loss_fn(self.optim.get_params(state.optim_state))
                grads = derivative(lambda p: loss_fn(p)[0])(self.optim.get_params(state.optim_state))
            finally:
                _blocks_only.reset(token)
        else:
            (loss, mutable), grads = derivative(loss_fn, has_aux=True)(self.optim.get_params(state.optim_state))
        candidate = self.optim.update(grads, state.optim_state, value=loss)
        finite = jnp.all(jnp.stack([jnp.isfinite(a).all() for a in jax.tree.leaves((loss, grads, candidate, mutable))]))
        optim = jax.tree.map(lambda a, b: jnp.where(finite, a, b), candidate, state.optim_state)
        mutable = commit_state(state.mutable_state, mutable, finite, self.moe_config)
        return SVIState(optim, mutable, rng), jnp.where(finite, loss, jnp.nan)

    update = stable_update

    def evaluate(self, state, *args, **kwargs):
        # NumPyro's default evaluate only passes optimizer parameters. Supply
        # the current frozen mutable state too, without committing new stats.
        _, rng = jax.random.split(state.rng_key)
        loss_fn = _make_loss_fn(self.loss, rng, self.constrain_fn, self.model, self.guide,
                               args, kwargs, self.static_kwargs, mutable_state=state.mutable_state)
        return loss_fn(self.optim.get_params(state.optim_state))[0]


def router_bias(state):
    return state.mutable_state[STATE]["bias"]


def diagnostics(state):
    s = state.mutable_state[STATE]
    e = s["bias"].shape[1]
    stats = s["stats"]
    attempts = stats[:, :e]
    drops = stats[:, 2*e:3*e]
    return {"updates": s["updates"], "attempted": attempts, "accepted": stats[:, e:2*e],
            "dropped": drops, "drop_fraction": drops.sum(-1) / jnp.maximum(attempts.sum(-1), 1),
            "load_imbalance": attempts.max(-1) / jnp.maximum(attempts.mean(-1), 1e-9),
            "tokens": stats[:, -3], "selected_gate_weight": stats[:, -2], "lost_gate_weight": stats[:, -1]}


def ffn(x, mask, params, cfg, layer, bias=None):
    """Unrolled/full-attention-residual counterpart of the scanned FFN."""
    layer, cfg = int(layer), dict(cfg)
    index = layer - int(cfg["first_k_dense"]) - 1
    ctx = _context.get()
    if ctx is not None:
        bias = ctx.bias[index]
    elif bias is None:
        raise ValueError("A saved MoE model must include frozen router biases")
    else:
        bias = jnp.asarray(bias, jnp.float32)[index]
    names = ("router", "expert1", "expert2", "shared1", "shared2")
    weights = [params[f"W_moe_{name}_l{layer}"] for name in names]
    y, stats, chain = dispatch(x, mask, *weights, bias, cfg, training=ctx is not None,
                              row_mask=None if ctx is None else ctx.row_mask,
                              runtime=None if ctx is None else ctx.runtime,
                              chain=None if ctx is None else ctx.chain)
    if ctx is not None:
        ctx.add(jnp.zeros_like(ctx.stats).at[index].set(stats), chain)
    return y


def transformer_scan(tokens, mask, params, cfg, bias, n_heads, head_dim,
                     attention_fn, attention_backend, attention_dtype, padding_multiple,
                     loop_cfg=None, return_details=False):
    """MHA with the shared dense/MoE and recurrent-depth executor."""
    from strategize_transformer import rms_norm, run_layers
    params = dict(params)
    dims = tokens.shape[-1]
    use_qk = "RMS_q_layers" in params and "RMS_k_layers" in params

    def attention(x, layer):
        z = rms_norm(x, layer["RMS_attn"])
        q, k, v = [(z @ layer[name].astype(z.dtype)).reshape(*x.shape[:-1], int(n_heads), int(head_dim))
                   for name in ("W_q", "W_k", "W_v")]
        if use_qk:
            q, k = rms_norm(q, layer["RMS_q"]), rms_norm(k, layer["RMS_k"])
        a = attention_fn(q, k, v, mask, dims, int(n_heads), int(head_dim),
                         attention_backend, attention_dtype, int(padding_multiple)).reshape(x.shape)
        h = x + layer["alpha_attn"].astype(x.dtype) * (a.astype(x.dtype) @ layer["W_o"].astype(x.dtype))
        return h, rms_norm(h, layer["RMS_ff"]), layer["alpha_ff"], jnp.float32(0), jnp.float32(0)

    result = run_layers(tokens, mask, params, cfg, bias, attention, loop_cfg,
                        return_details=return_details)
    return result if return_details else result[0]
