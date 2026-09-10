"""Memory contracts and numerical regression checks for the FM training path."""
import contextlib
import io
import math
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.ad_checkpoint import print_saved_residuals

from test_transformer_moe import CFG, close, fixture
from strategize_moe import selected_experts, swiglu, transformer_scan
from strategize_distributed import Runtime


@pytest.mark.parametrize("n", [1, 38, 128, 257])
def test_compact_experts_preserve_assignments_and_derivatives(n):
    keys = jax.random.split(jax.random.PRNGKey(25), 3)
    args = (jax.random.normal(keys[0], (n, 8)),
            jax.random.normal(keys[1], (4, 8, 16)) * .1,
            jax.random.normal(keys[2], (4, 8, 8)) * .1)
    indices = jnp.arange(n * 2).reshape(n, 2) % 4
    keep = jnp.arange(n * 2).reshape(n, 2) % 3 != 0

    def reference(x, a, b):
        all_values = jnp.stack([swiglu(x, a[e], b[e]) for e in range(4)], axis=1)
        chosen = jnp.take_along_axis(all_values, indices[..., None], axis=1)
        return jnp.where(keep[..., None], chosen, 0)

    fn = lambda x, a, b: selected_experts(x, a, b, indices, keep)
    close(jax.jit(fn)(*args), reference(*args))
    close(jax.jit(jax.grad(lambda *a: jnp.square(fn(*a)).sum(), (0, 1, 2)))(*args),
          jax.jit(jax.grad(lambda *a: jnp.square(reference(*a)).sum(), (0, 1, 2)))(*args))
    close(selected_experts(*args, indices, jnp.zeros_like(keep)), jnp.zeros((n, 2, 8)))


@pytest.mark.parametrize("n", [38, 128, 2048])
def test_expert_backward_does_not_save_token_copies_of_weights(n):
    args = (jax.ShapeDtypeStruct((n, 576), jnp.float32),
            jax.ShapeDtypeStruct((8, 576, 1152), jnp.float32),
            jax.ShapeDtypeStruct((8, 576, 576), jnp.float32),
            jax.ShapeDtypeStruct((n, 2), jnp.int32))
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        print_saved_residuals(lambda x, a, b, i: selected_experts(x, a, b, i).sum(), *args)
    shapes = re.findall(r"(?:f32|i32|bool)\[([\d,]+)\]", stream.getvalue())
    elements = sum(math.prod(map(int, shape.split(","))) for shape in shapes)
    # Formerly 1,023,051,776 bytes at n=128, mostly selected weight copies.
    assert elements * 4 < 64 * 1024**2, stream.getvalue()


def transformer_fixture(dtype, checkpointing):
    d, depth, heads = 8, 3, 2
    key = jax.random.PRNGKey(37)
    def rand(shape):
        nonlocal key
        key, sub = jax.random.split(key)
        return .1 * jax.random.normal(sub, shape)
    params = {name + "_layers": rand((depth, d, d)) for name in ("W_q", "W_k", "W_v", "W_o")}
    params.update({name + "_layers": jnp.ones((depth, d)) for name in ("RMS_attn", "RMS_ff")})
    params.update({name + "_layers": jnp.ones((depth, d // heads)) for name in ("RMS_q", "RMS_k")})
    params.update({name + "_layers": jnp.full((depth,), .1) for name in ("alpha_attn", "alpha_ff")})
    params.update(W_ff1_layers=rand((1, d, 2*d)), W_ff2_layers=rand((1, d, d)),
                  W_moe_router_layers=rand((depth-1, d, 4)),
                  W_moe_expert1_layers=rand((depth-1, 4, d, 2*d)),
                  W_moe_expert2_layers=rand((depth-1, 4, d, d)),
                  W_moe_shared1_layers=rand((depth-1, d, 2*d)),
                  W_moe_shared2_layers=rand((depth-1, d, d)), RMS_final=jnp.ones(d))
    cfg = dict(CFG, first_k_dense=1, n_moe_layers=depth-1, moe_d_ff=d,
               compute_dtype=dtype, activation_checkpointing=checkpointing)
    bias = jnp.zeros((depth-1, 4))
    def attention(q, k, v, mask, *_):
        return jax.nn.dot_product_attention(q, k, v,
            mask=(mask > 0)[:, None, None, :], implementation="xla")
    def fn(x, p):
        return transformer_scan(x, jnp.ones(x.shape[:-1]), p, cfg, bias, heads, d//heads,
                                attention, "xla", "auto", 8)
    return fn, rand((3, 9, d)), params


def test_transformer_rematerialization_and_mixed_precision():
    reference, x, p = transformer_fixture("float32", False)
    remat, _, _ = transformer_fixture("float32", True)
    close(jax.jit(reference)(x, p), jax.jit(remat)(x, p))
    objective = lambda fn: lambda x, p: jnp.square(fn(x, p) - .5).sum()
    close(jax.jit(jax.grad(objective(reference), (0, 1)))(x, p),
          jax.jit(jax.grad(objective(remat), (0, 1)))(x, p))
    mixed, _, _ = transformer_fixture("bfloat16", True)
    output = jax.jit(mixed)(x, p)
    assert output.dtype == jnp.float32
    close(output, reference(x, p), rtol=.04, atol=.025)
    grads = jax.jit(jax.grad(objective(mixed), (0, 1)))(x, p)
    assert all(a.dtype == jnp.float32 and np.isfinite(a).all() for a in jax.tree.leaves(grads))
    hlo = jax.jit(mixed).lower(x, p).as_text()
    assert "bf16" in hlo


def test_donation_preserves_host_best_and_checkpoint_resume(tmp_path):
    rt = Runtime(dict(enabled=len(jax.devices()) > 1))
    svi, state, args = fixture(rt)
    if rt.enabled:
        rt.configure_svi_sharding()
    batch = rt.place_batch(args)
    best = rt.host_copy(state)
    donated = rt.owned_state(rt.replicate(state))
    expected = jax.jit(lambda s: svi.stable_update(s, **args))(state)
    result = rt.update(svi, donated, batch, donate=True)
    close(result, expected)
    close(best, state)
    assert all(isinstance(a, np.ndarray) for a in jax.tree.leaves(best))
    assert rt.profiles[-1]["memory"]["alias_size_in_bytes"] > 0
    generation = rt.save_checkpoint(tmp_path, "latest", result[0], bytearray([0, 1, 255]))
    assert bytes(rt.load_checkpoint_payload(tmp_path)["payload"]) == bytes([0, 1, 255])
    restored = rt.restore_state(tmp_path, result[0], generation)
    expected_next = rt.update(svi, result[0], batch)
    close(rt.update(svi, restored, batch, donate=True), expected_next)
    close(best, state)


def test_donation_preserves_finite_update_rejection():
    rt = Runtime(dict(enabled=True, local_device_ids=[0]))
    svi, state, args = fixture(rt)
    rt.configure_svi_sharding()
    original = rt.host_copy(state)
    bad = dict(args, Y_obs=np.full_like(args["Y_obs"], np.nan))
    update = jax.jit(lambda s: svi.stable_update(s, **bad), donate_argnums=(0,))
    rejected, loss = update(rt.owned_state(state))
    jax.block_until_ready(rejected)
    assert np.isnan(loss)
    close(rejected.optim_state, original.optim_state)
    close(rejected.mutable_state, original.mutable_state)
    close(state, original)
    # The runtime rejects a nonfinite loss after execution. That failure must
    # also prevent retrying the consumed input through the R scan fallback.
    with pytest.raises(RuntimeError, match="Donated SVI update failed.*nonfinite loss"):
        rt.update(svi, rt.owned_state(rt.replicate(state)), rt.place_batch(bad), donate=True)
    close(state, original)


def test_raw_checkpoint_transport_does_not_expand_bytes():
    rt = Runtime()
    payload = bytearray(range(256))
    assert rt.broadcast_bytes(payload) == payload
    raw = rt._raw_bytes(payload)
    assert raw.dtype == np.uint8 and raw.nbytes == 256


def test_replica_verification_bounds_transport_and_detects_differences(monkeypatch):
    from jax.experimental import multihost_utils as mh
    rt = Runtime()
    rt.count = 2
    monkeypatch.setattr(rt, "require_equal", lambda *args: None)
    statuses, sizes = [], []
    monkeypatch.setattr(rt, "agree_status", lambda error, label: statuses.append(error))
    def broadcast(value, primary):
        sizes.append(value.nbytes)
        return value.copy()
    monkeypatch.setattr(mh, "broadcast_one_to_all", broadcast)
    rt.check_replicas({"weight": jnp.ones(600000, jnp.float32)})
    assert statuses == [None] and len(sizes) == 3 and max(sizes) <= 1024**2
    monkeypatch.setattr(mh, "broadcast_one_to_all", lambda value, primary: value + 1)
    rt.check_replicas({"weight": jnp.ones(3, jnp.float32)})
    assert "differ" in statuses[-1]
    rt.check_replicas({"weight": jnp.array([jnp.nan])})
    assert statuses[-1] == "nonfinite state"
