"""Exact grouped expert matmuls for MoE training on GPU (Pallas Triton).

Assignments are laid out expert-major with every expert's rows starting on a
row-tile boundary, so each output tile belongs to exactly one expert (the
megablox layout with per-expert padding instead of tile splitting). Work is
proportional to the accepted assignments: no dense E-way expansion.

  gmm(x, w, layout):   y[r] = x[r] @ w[group(r)]          (forward, dX via w^T)
  tgmm(x, dy, layout): dw[e] = sum_{r in e} x[r]^T dy[r]   (weight gradient)
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

# f32 (IEEE) at d = h = 576, 5880 assignments, fwd+bwd: 3.0 ms on an RTX 4090
# Laptop and 3.7 ms on an RTX 5080 Laptop, vs 16-21 ms for the dense-expanded
# ragged_dot. num_stages=2 hits a pathological schedule on the 4090 (13x slower).
BM, BN, BK = 64, 64, 32
NUM_WARPS, NUM_STAGES = 2, 3


def supported(x, w1, w2):
    """Tile divisibility of the model width and both expert matmuls."""
    d, h2 = w1.shape[1], w1.shape[2]
    return (x.shape[-1] == d and d % BK == 0 and h2 % (2 * BN) == 0 and (h2 // 2) % BK == 0
            and w2.shape[2] % BN == 0)


def register_device():
    """Pallas Triton picks its target from a fixed device-name table. Register
    other CUDA GPUs (e.g. the GeForce RTX 5080 Laptop, sm_120) by capability.
    ROCm devices report a gfx name (e.g. 'gfx1100') and need no registration."""
    from jax._src.pallas.triton import gpu_info
    device = jax.devices()[0]
    kind = device.device_kind
    if device.platform != "gpu" or kind in gpu_info.registry or gpu_info.gpu_version_from_device_kind(kind):
        return
    version = str(device.compute_capability).split(".")
    if len(version) != 2 or not all(v.isdigit() for v in version):
        return
    major, minor = (int(v) for v in version)
    gpu_info.registry[kind] = lambda: gpu_info.GpuInfo(
        gpu_version=None, arch_name=f"{major}.{minor}", compute_capability=10 * major + minor)


def layout(expert_ids, experts, bm=None):
    """Padded expert-major positions for assignments; ids == experts are dropped.

    Returns (positions, tile_group, tile_start, tile_count, padded_rows). Dropped
    assignments get distinct positions past the end, so scatters stay unique.
    """
    bm = BM if bm is None else bm
    assignments = expert_ids.shape[0]
    tiles = -(-assignments // bm) + experts
    one_hot = jax.nn.one_hot(expert_ids, experts + 1, dtype=jnp.int32)
    rank = (jnp.cumsum(one_hot, axis=0) * one_hot).sum(-1) - 1
    counts = one_hot.sum(0)[:experts]
    tile_count = (counts + bm - 1) // bm
    tile_end = jnp.cumsum(tile_count)
    tile_start = tile_end - tile_count
    kept = expert_ids < experts
    safe = jnp.minimum(expert_ids, experts - 1)
    positions = jnp.where(kept, tile_start[safe] * bm + rank,
                          tiles * bm + jnp.arange(assignments))
    tile_group = jnp.sum(jnp.arange(tiles)[:, None] >= tile_end[None, :], axis=1).astype(jnp.int32)
    return positions, tile_group, tile_start.astype(jnp.int32), tile_count.astype(jnp.int32), tiles * bm


def _gmm_kernel(group_ref, x_ref, w_ref, o_ref, *, experts, k_dim, transpose_w):
    t, j = pl.program_id(0), pl.program_id(1)
    e = group_ref[t]
    rows, cols = pl.ds(t * BM, BM), pl.ds(j * BN, BN)

    @pl.when(e < experts)
    def _():
        def body(kk, acc):
            a = x_ref[rows, pl.ds(kk * BK, BK)]
            if transpose_w:
                return acc + plgpu.dot(a, w_ref[e, cols, pl.ds(kk * BK, BK)], trans_b=True)
            return acc + plgpu.dot(a, w_ref[e, pl.ds(kk * BK, BK), cols])
        acc = jax.lax.fori_loop(0, k_dim // BK, body, jnp.zeros((BM, BN), jnp.float32))
        o_ref[rows, cols] = acc.astype(o_ref.dtype)

    @pl.when(e >= experts)
    def _():
        o_ref[rows, cols] = jnp.zeros((BM, BN), o_ref.dtype)


def _gmm_call(x, w, tile_group, transpose_w=False):
    rows, k_dim = x.shape
    experts = w.shape[0]
    n_dim = w.shape[1] if transpose_w else w.shape[2]
    assert rows % BM == 0 and k_dim % BK == 0 and n_dim % BN == 0, (x.shape, w.shape)
    kernel = functools.partial(_gmm_kernel, experts=experts, k_dim=k_dim, transpose_w=transpose_w)
    return pl.pallas_call(
        kernel, out_shape=jax.ShapeDtypeStruct((rows, n_dim), x.dtype),
        grid=(rows // BM, n_dim // BN),
        compiler_params=plgpu.CompilerParams(num_warps=NUM_WARPS, num_stages=NUM_STAGES),
        name="moe_gmm_t" if transpose_w else "moe_gmm",
    )(tile_group, x, w)


def _tgmm_kernel(start_ref, count_ref, x_ref, dy_ref, o_ref):
    e, i, j = pl.program_id(0), pl.program_id(1), pl.program_id(2)
    start, count = start_ref[e], count_ref[e]

    def body(s, acc):
        rows = pl.ds((start + s) * BM, BM)
        return acc + plgpu.dot(x_ref[rows, pl.ds(i * BK, BK)], dy_ref[rows, pl.ds(j * BN, BN)], trans_a=True)
    acc = jax.lax.fori_loop(0, count, body, jnp.zeros((BK, BN), jnp.float32))
    o_ref[e, pl.ds(i * BK, BK), pl.ds(j * BN, BN)] = acc.astype(o_ref.dtype)


def _tgmm_call(x, dy, tile_start, tile_count, experts, dtype):
    k_dim, n_dim = x.shape[1], dy.shape[1]
    return pl.pallas_call(
        _tgmm_kernel, out_shape=jax.ShapeDtypeStruct((experts, k_dim, n_dim), dtype),
        grid=(experts, k_dim // BK, n_dim // BN),
        compiler_params=plgpu.CompilerParams(num_warps=NUM_WARPS, num_stages=NUM_STAGES),
        name="moe_tgmm",
    )(tile_start, tile_count, x, dy)


@jax.custom_vjp
def gmm(x, w, tile_group, tile_start, tile_count):
    return _gmm_call(x, w.astype(x.dtype), tile_group)


def _gmm_fwd(x, w, tile_group, tile_start, tile_count):
    return gmm(x, w, tile_group, tile_start, tile_count), (x, w, tile_group, tile_start, tile_count)


def _gmm_bwd(res, dy):
    x, w, tile_group, tile_start, tile_count = res
    dx = _gmm_call(dy, w.astype(dy.dtype), tile_group, transpose_w=True)
    dw = _tgmm_call(x, dy, tile_start, tile_count, w.shape[0], w.dtype)
    return dx, dw, None, None, None


gmm.defvjp(_gmm_fwd, _gmm_bwd)


def gmm_experts(x, w1, w2, indices, keep=None):
    """Drop-in for strategize_moe.grouped_experts: exact work, same values."""
    n, k = indices.shape
    if not n:
        return jnp.zeros((n, k, x.shape[-1]), x.dtype)
    register_device()
    experts = w1.shape[0]
    keep = jnp.ones_like(indices, bool) if keep is None else keep
    expert_ids = jnp.where(keep.reshape(-1), indices.reshape(-1), experts)
    positions, tile_group, tile_start, tile_count, rows = layout(expert_ids, experts)
    token = jnp.arange(n * k) // k
    padded = jnp.zeros((rows, x.shape[-1]), x.dtype).at[positions].set(
        x[token], mode="drop", unique_indices=True)
    hidden = gmm(padded, w1, tile_group, tile_start, tile_count)
    gate, value = jnp.split(hidden, 2, axis=-1)
    y = gmm(jax.nn.silu(gate) * value, w2, tile_group, tile_start, tile_count)
    out = y.at[positions].get(mode="fill", fill_value=0, unique_indices=True)
    return jnp.where(keep[..., None], out.reshape(n, k, x.shape[-1]), 0)
