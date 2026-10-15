"""Constant-size summaries of device-resident gradient trees."""

import math

import jax
import jax.numpy as jnp


def gradient_summary(grads):
    """Reduce on device, without requiring JAX's global x64 mode.

    A scaled sum of squares avoids overflow when finite FP32 gradients are
    squared. Final scalar arithmetic uses Python doubles, as the old host
    reduction did. Two uint32 words preserve exact nonfinite counts even for
    trees larger than 2**32 elements. Element counts come from static shapes.
    """
    scale, ssq = jnp.float32(0), jnp.float32(0)
    lo, hi = jnp.uint32(0), jnp.uint32(0)
    for leaf in jax.tree.leaves(grads):
        if not leaf.size:
            continue
        finite = jnp.isfinite(leaf)
        values = jnp.where(finite, jnp.abs(leaf), 0).astype(
            jnp.result_type(jnp.abs(leaf).dtype, jnp.float32))
        leaf_scale = jnp.max(values)
        leaf_ssq = jnp.sum(jnp.square(values / jnp.where(leaf_scale > 0, leaf_scale, 1)))
        new_scale = jnp.maximum(scale, leaf_scale)
        divisor = jnp.where(new_scale > 0, new_scale, 1)
        ssq = ssq * jnp.square(scale / divisor) + leaf_ssq * jnp.square(leaf_scale / divisor)
        scale = new_scale
        invalid = jnp.ravel(~finite)
        # Each partial sum fits in uint32, even for a single enormous leaf.
        for start in range(0, leaf.size, 1 << 30):
            count = jnp.sum(invalid[start:start + (1 << 30)], dtype=jnp.uint32)
            new_lo = lo + count
            hi = hi + (new_lo < lo).astype(jnp.uint32)
            lo = new_lo
    return {"scale": scale, "ssq": ssq, "nonfinite_lo": lo, "nonfinite_hi": hi}


def gradient_statistics(summary, n_elements):
    """Format an already-host-resident scalar summary; never accepts gradients."""
    scale, ssq = float(summary["scale"]), float(summary["ssq"])
    n = int(n_elements)
    return {
        "grad_l2": scale * math.sqrt(ssq),
        "grad_rms": scale * math.sqrt(ssq / max(n, 1)),
        "grad_max_abs": scale,
        "grad_n_nonfinite": (int(summary["nonfinite_hi"]) << 32) + int(summary["nonfinite_lo"]),
        "grad_n_elements": n,
    }
