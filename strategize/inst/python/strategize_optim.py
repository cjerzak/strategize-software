"""Full-objective gradient normalization and diagnostics for SVI optimizers."""

import math
import re
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

_tree_norm = getattr(getattr(optax, "tree", None), "norm", optax.global_norm)


def _states(tree, name):
    """Inspect named Optax states without transferring parameter arrays to host."""
    match = lambda x: type(x).__name__ == name
    return [x for x in jax.tree.leaves(tree, is_leaf=match) if match(x)]


def muon_partition_diagnostics(state, params=None, dimension_numbers=None):
    """Verify actual momentum leaves against the explicit hidden-weight partition."""
    muon = _states(state, "MuonState")
    adam = _states(state, "ScaleByAdamState")
    if len(muon) != 1 or len(adam) != 1:
        raise ValueError("Strict Muon requires one Muon momentum state and one auxiliary Adam state")

    def shapes(tree):
        return {path: tuple(x.shape) for path, x in jax.tree_util.tree_flatten_with_path(tree)[0]}

    actual_muon, actual_adam = shapes(muon[0].mu), shapes(adam[0].mu)
    if sum(math.prod(s) for s in actual_muon.values()) <= 0:
        raise ValueError("Strict Muon found no eligible hidden matrices in the optimizer state")
    if actual_adam != shapes(adam[0].nu):
        raise ValueError("Strict Muon auxiliary Adam moment trees disagree")
    if params is not None:
        dims = dimension_numbers(params)
        labels = jax.tree.map(lambda d: d is not None, dims,
                             is_leaf=lambda d: d is None or isinstance(d, optax.contrib.MuonDimensionNumbers))
        labels = dict(jax.tree_util.tree_flatten_with_path(labels)[0])
        expected = shapes(params)
        if (actual_muon != {p: s for p, s in expected.items() if labels[p]} or
                actual_adam != {p: s for p, s in expected.items() if not labels[p]}):
            raise ValueError("Strict Muon momentum state does not match the requested weight partition")
    return {"muon_partition_status": "verified",
            "muon_parameter_count": sum(math.prod(s) for s in actual_muon.values()),
            "auxiliary_adam_parameter_count": sum(math.prod(s) for s in actual_adam.values())}


def strict_muon_optimizer(learning_rate, dimension_numbers):
    """Construct Muon with fixed scientific settings; never substitute an optimizer."""
    if not callable(dimension_numbers):
        raise ValueError("Strict Muon requires an explicit weight dimension-number callable")
    try:
        optimizer = optax.contrib.muon(
            learning_rate=learning_rate, weight_decay=0, adam_weight_decay=0,
            consistent_rms=0.2, muon_weight_dimension_numbers=dimension_numbers)
    except Exception as exc:
        raise ValueError("Cannot construct strict Muon with the required settings; update Optax "
                         "or explicitly select a supported alternative optimizer") from exc

    def init(params):
        state = optimizer.init(params)
        muon_partition_diagnostics(state, params, dimension_numbers)
        return state

    def update(grads, state, params=None, **extra_args):
        # Shape/tree checks also validate restored states, once per JIT trace.
        muon_partition_diagnostics(state, params, dimension_numbers)
        return optimizer.update(grads, state, params, **extra_args)

    return optax.GradientTransformationExtraArgs(init, update)


class UpdateDiagnosticsState(NamedTuple):
    inner_state: object
    count: object
    clipped_count: object
    clip_factor_sum: object
    raw_gradient_norm: object
    normalized_gradient_norm: object
    update_parameter_ratio: object
    max_update_parameter_ratio: object
    prelude_gradient_norm: object
    recurrent_core_gradient_norm: object
    recurrent_core_scaled_gradient_norm: object
    coda_gradient_norm: object
    other_gradient_norm: object


def _layer_tuple(value):
    if value is None:
        return ()
    if isinstance(value, (int, float)):
        return (int(value),)
    return tuple(int(x) for x in value)


def _path_name(path):
    return "/".join(str(getattr(part, "key", getattr(part, "idx", part))) for part in path)


def normalized_optimizer(optimizer, objective_scale=1.0, clip_global_norm=None,
                         prelude_layers=None, recurrent_core_layers=None,
                         coda_layers=None, recurrent_gradient_scale=1.0):
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
    layer_groups = {
        "prelude": frozenset(_layer_tuple(prelude_layers)),
        "core": frozenset(_layer_tuple(recurrent_core_layers)),
        "coda": frozenset(_layer_tuple(coda_layers)),
    }
    if any(a & b for i, a in enumerate(layer_groups.values())
           for b in list(layer_groups.values())[i + 1:]):
        raise ValueError("Transformer gradient layer groups must be disjoint")
    recurrent_gradient_scale = float(recurrent_gradient_scale)
    if not 0 < recurrent_gradient_scale <= 1:
        raise ValueError("recurrent_gradient_scale must be in (0, 1]")
    ordered_layers = tuple(sorted(set().union(*layer_groups.values())))
    optimizer = optax.with_extra_args_support(optimizer)

    def group_for_name(name):
        if "loop_state_gain" in name or "loop_input_gain" in name:
            return "core"
        matches = re.findall(r"_l(\d+)(?:\D|$)", name)
        if matches:
            layer = int(matches[-1])
            for group, layers in layer_groups.items():
                if layer in layers:
                    return group
        return "other"

    def stacked_mask(name, gradient, layers):
        if not ordered_layers or gradient.ndim < 1:
            return None
        depth = max(ordered_layers)
        size = gradient.shape[0]
        if size == depth:
            layer_numbers = range(1, depth + 1)
        elif "W_moe_" in name and size < depth:
            # Routed MoE stacks omit the dense prefix. Their first axis still
            # follows transformer layer order, beginning after that prefix.
            layer_numbers = range(depth - size + 1, depth + 1)
        elif ("W_ff1_layers" in name or "W_ff2_layers" in name) and size < depth:
            # Dense FFN stacks precede the routed MoE suffix.
            layer_numbers = range(1, size + 1)
        else:
            return None
        mask = jnp.asarray([layer in layers for layer in layer_numbers], gradient.dtype)
        return mask.reshape((gradient.shape[0],) + (1,) * (gradient.ndim - 1))

    def select_group(path, gradient, group):
        name = _path_name(path)
        if "_layers" in name:
            mask = stacked_mask(
                name, gradient, layer_groups[group] if group != "other" else set())
            if mask is not None:
                return gradient * (1 - sum(
                    stacked_mask(name, gradient, layers) for layers in layer_groups.values()
                )) if group == "other" else gradient * mask
        return gradient if group_for_name(name) == group else jnp.zeros_like(gradient)

    def scale_core(path, gradient):
        name = _path_name(path)
        if "_layers" in name:
            mask = stacked_mask(name, gradient, layer_groups["core"])
            if mask is not None:
                return gradient * (1 + (recurrent_gradient_scale - 1) * mask)
        if group_for_name(name) == "core":
            return gradient * recurrent_gradient_scale
        return gradient

    def group_norm(tree, group):
        selected = jax.tree_util.tree_map_with_path(
            lambda path, gradient: select_group(path, gradient, group), tree)
        return _tree_norm(selected)

    def init(params):
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        return UpdateDiagnosticsState(optimizer.init(params), jnp.int32(0), jnp.int32(0),
                                      zero, zero, zero, zero, zero,
                                      zero, zero, zero, zero, zero)

    def update(grads, state, params=None, **extra_args):
        raw_norm = _tree_norm(grads)
        scaled = jax.tree.map(lambda g: g * objective_scale, grads)
        prelude_norm = group_norm(scaled, "prelude")
        core_norm = group_norm(scaled, "core")
        coda_norm = group_norm(scaled, "coda")
        other_norm = group_norm(scaled, "other")
        scaled = jax.tree_util.tree_map_with_path(scale_core, scaled)
        core_scaled_norm = group_norm(scaled, "core")
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
            prelude_norm, core_norm, core_scaled_norm, coda_norm, other_norm,
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
    result = {
        "update_diagnostics_status": "actual_optimizer_updates",
        "update_count": count,
        "clipped_update_count": clipped,
        "clipped_update_fraction": clipped / count if count else 0.0,
        "mean_clip_factor": float(s.clip_factor_sum) / count if count else 1.0,
        "last_raw_gradient_norm": float(s.raw_gradient_norm),
        "last_normalized_gradient_norm": float(s.normalized_gradient_norm),
        "last_update_parameter_ratio": float(s.update_parameter_ratio),
        "max_update_parameter_ratio": float(s.max_update_parameter_ratio),
        "last_prelude_gradient_norm": float(s.prelude_gradient_norm),
        "last_recurrent_core_gradient_norm": float(s.recurrent_core_gradient_norm),
        "last_recurrent_core_scaled_gradient_norm": float(s.recurrent_core_scaled_gradient_norm),
        "last_coda_gradient_norm": float(s.coda_gradient_norm),
        "last_other_gradient_norm": float(s.other_gradient_norm),
    }
    if _states(s.inner_state, "MuonState"):
        result.update(muon_partition_diagnostics(s.inner_state))
    return result
