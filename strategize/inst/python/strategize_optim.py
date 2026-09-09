"""Full-objective gradient normalization and diagnostics for SVI optimizers."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

_tree_norm = getattr(getattr(optax, "tree", None), "norm", optax.global_norm)


class UpdateDiagnosticsState(NamedTuple):
    inner_state: object
    count: object
    clipped_count: object
    clip_factor_sum: object
    raw_gradient_norm: object
    normalized_gradient_norm: object
    update_parameter_ratio: object
    max_update_parameter_ratio: object


def normalized_optimizer(optimizer, objective_scale=1.0, clip_global_norm=None):
    """Scale the entire ELBO gradient, then clip, then apply the optimizer.

    NumPyro continues reporting the original ELBO. Scaling all gradient leaves
    by one fixed constant preserves the likelihood/KL/auxiliary-loss ratios.
    With data parallelism this transform runs after global gradient reduction.
    Diagnostics describe actual accepted optimizer updates, not extra gradient
    evaluations. They are part of the checkpointed optimizer state.
    """
    objective_scale = float(objective_scale)
    if not 0 < objective_scale <= 1:
        raise ValueError("objective_scale must be in (0, 1]")
    threshold = float(clip_global_norm) if clip_global_norm is not None else None
    if threshold is not None and not threshold > 0:
        raise ValueError("clip_global_norm must be positive")
    optimizer = optax.with_extra_args_support(optimizer)

    def init(params):
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        return UpdateDiagnosticsState(optimizer.init(params), jnp.int32(0), jnp.int32(0),
                                      zero, zero, zero, zero, zero)

    def update(grads, state, params=None, **extra_args):
        raw_norm = _tree_norm(grads)
        scaled = jax.tree.map(lambda g: g * objective_scale, grads)
        norm = _tree_norm(scaled)
        factor = jnp.asarray(1.0, dtype=norm.dtype)
        if threshold is not None:
            factor = jnp.minimum(1.0, threshold / jnp.maximum(norm, jnp.finfo(norm.dtype).tiny))
        clipped = jax.tree.map(lambda g: g * factor, scaled)
        updates, inner_state = optimizer.update(clipped, state.inner_state, params, **extra_args)
        ratio = jnp.asarray(jnp.nan, dtype=jnp.float32)
        if params is not None:
            ratio = _tree_norm(updates) / jnp.maximum(_tree_norm(params), 1e-12)
        return updates, UpdateDiagnosticsState(
            inner_state, optax.safe_int32_increment(state.count),
            state.clipped_count + (factor < 1).astype(jnp.int32),
            state.clip_factor_sum + factor, raw_norm, norm, ratio,
            jnp.maximum(state.max_update_parameter_ratio, ratio),
        )

    return optax.GradientTransformationExtraArgs(init, update)


def update_diagnostics(optim_state):
    """Find our state inside the NumPyro optimizer wrapper, on the host."""
    leaves = jax.tree.leaves(optim_state, is_leaf=lambda x: isinstance(x, UpdateDiagnosticsState))
    states = [x for x in leaves if isinstance(x, UpdateDiagnosticsState)]
    if not states:
        return {"update_diagnostics_status": "unavailable"}
    if len(states) != 1:
        raise ValueError("Expected one SVI optimizer diagnostics state")
    s = states[0]
    count = int(s.count)
    clipped = int(s.clipped_count)
    return {
        "update_diagnostics_status": "actual_optimizer_updates",
        "update_count": count,
        "clipped_update_count": clipped,
        "clipped_update_fraction": clipped / count if count else 0.0,
        "mean_clip_factor": float(s.clip_factor_sum) / count if count else 1.0,
        "last_raw_gradient_norm": float(s.raw_gradient_norm),
        "last_normalized_gradient_norm": float(s.normalized_gradient_norm),
        "last_update_parameter_ratio": float(s.update_parameter_ratio),
        "max_update_parameter_ratio": float(s.max_update_parameter_ratio),
    }
