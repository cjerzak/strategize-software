"""Exact grouped expert matmuls (Pallas Triton) against the block reference."""
import jax
import jax.numpy as jnp
import pytest

from test_transformer_moe import close
import strategize_moe as moe
import strategize_moe_gmm as gmm

pytestmark = pytest.mark.skipif(jax.default_backend() != "gpu", reason="Pallas Triton kernels need a CUDA GPU")


@pytest.mark.parametrize("n", [1, 38, 257, 1000])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_gmm_experts_match_blocks_in_value_and_gradient(n, dtype):
    keys = jax.random.split(jax.random.PRNGKey(7), 4)
    experts, d, h = 4, 64, 64
    x = jax.random.normal(keys[0], (n, d)).astype(dtype)
    w1 = jax.random.normal(keys[1], (experts, d, 2 * h)) * .1
    w2 = jax.random.normal(keys[2], (experts, h, d)) * .1
    # Skewed routing, one expert never chosen, and capacity drops.
    indices = jnp.stack([jnp.arange(n) % 3, (jnp.arange(n) * 7 + 1) % 3], axis=1)
    keep = jax.random.uniform(keys[3], (n, 2)) > .1
    assert gmm.supported(x, w1, w2)

    def loss(f):
        return lambda x, a, b: jnp.square(f(x, a, b, indices, keep).astype(jnp.float32)).sum()
    got = jax.jit(jax.grad(loss(gmm.gmm_experts), (0, 1, 2)))(x, w1, w2)
    want = jax.jit(jax.grad(loss(moe.selected_experts), (0, 1, 2)))(x, w1, w2)
    value = jax.jit(gmm.gmm_experts)(x, w1, w2, indices, keep)
    if dtype == jnp.bfloat16:
        # Rounding points differ in half precision; bound errors by each
        # gradient's scale (dW sums hundreds of bf16 products).
        close(value.astype(jnp.float32), moe.selected_experts(x, w1, w2, indices, keep).astype(jnp.float32),
              rtol=.05, atol=.02)
        for a, b in zip(got, want):
            scale = float(jnp.max(jnp.abs(b.astype(jnp.float32))))
            assert float(jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32)))) <= .02 * scale
        return
    # Summation order differs from the blocks (dW sums hundreds of rows).
    close(value, moe.selected_experts(x, w1, w2, indices, keep), rtol=5e-5, atol=3e-5)
    close(got, want, rtol=5e-5, atol=3e-5)


def test_gmm_layout_gives_each_tile_one_expert():
    ids = jnp.array([2, 0, 4, 2, 2, 1, 0, 4], jnp.int32)  # 4 == dropped
    positions, group, start, count, rows = gmm.layout(ids, 4, bm=2)
    kept = ids < 4
    assert len(set(positions.tolist())) == ids.size
    assert bool(jnp.all(jnp.where(kept, positions < rows, positions >= rows)))
    for p, e in zip(positions[kept].tolist(), ids[kept].tolist()):
        assert int(group[p // 2]) == e
    assert count.tolist() == [1, 1, 2, 0] and start.tolist() == [0, 1, 2, 4]


def test_pallas_kernel_choice(monkeypatch):
    monkeypatch.delenv("STRATEGIZE_MOE_EXPERT_KERNEL", raising=False)
    assert moe.expert_kernel(True) == "pallas"
    monkeypatch.setenv("STRATEGIZE_MOE_EXPERT_KERNEL", "pallas")
    assert moe.expert_kernel(True) == "pallas"
    assert moe.expert_kernel(False) == "blocks"
    token = moe._blocks_only.set(True)
    try:
        assert moe.expert_kernel(True) == "blocks"
    finally:
        moe._blocks_only.reset(token)
