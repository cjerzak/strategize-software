"""Qualify with JAX_PLATFORMS=cpu and 2 or 4 simulated host devices."""
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import TraceMeanField_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import optax_to_numpyro
import optax
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "inst/python"))
from strategize_moe import (STATE, MoESVI, dispatch, route, swiglu, ffn, set_branch, wrap_model,
                            router_bias, diagnostics, _context)
from strategize_distributed import Runtime

CFG = dict(n_routed_experts=4, n_experts_per_tok=2, n_shared_experts=1,
           moe_d_ff=4, first_k_dense=0, n_moe_layers=1,
           routed_scaling_factor=1., capacity_factor=.2, router_bias_rate=.001)


def weights(seed=1):
    keys = jax.random.split(jax.random.PRNGKey(seed), 5)
    shapes = [(4, 4), (4, 4, 8), (4, 4, 4), (4, 8), (4, 4)]
    return tuple(.15 * jax.random.normal(key, shape) for key, shape in zip(keys, shapes))


def close(a, b, rtol=5e-5, atol=3e-6):
    assert jax.tree.structure(a) == jax.tree.structure(b)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        assert x.shape == y.shape
        if jnp.issubdtype(x.dtype, jnp.integer):
            np.testing.assert_array_equal(x, y)
        else:
            np.testing.assert_allclose(x, y, rtol=rtol, atol=atol)


def reference(x, mask, w, bias, cfg, training):
    """Independent all-expert expression with a NumPy arrival-order mask."""
    flat = x.reshape(-1, x.shape[-1])
    indices, gates = route(flat, w[0], bias, cfg["n_experts_per_tok"], cfg["routed_scaling_factor"])
    valid = np.asarray(mask).reshape(-1) > 0
    keep = np.broadcast_to(valid[:, None], indices.shape).copy()
    counts = np.zeros(cfg["n_routed_experts"], int)
    n = valid.sum()
    cap = max(int(cfg["capacity_factor"] * n * cfg["n_experts_per_tok"] / len(counts)), min(n, 256))
    for i, selections in enumerate(np.asarray(indices)):
        for k, e in enumerate(selections):
            if valid[i]:
                keep[i, k] &= not training or counts[e] < cap
                counts[e] += 1
    output = swiglu(flat, w[3], w[4])
    for e in range(len(counts)):
        contribution = (gates * ((indices == e) & keep)).sum(-1)
        output = output + contribution[:, None] * swiglu(flat, w[1][e], w[2][e])
    return jnp.where(valid[:, None], output, 0).reshape(x.shape)


def test_router_bias_changes_selection_but_not_mixture_weights():
    x = jnp.array([[1., 2., 3., 4.]], dtype=jnp.bfloat16)
    router = jnp.eye(4, dtype=jnp.bfloat16)
    bias = jnp.array([10., 9., -10., -9.])
    indices, gates = route(x, router, bias, 2, 1.5)
    np.testing.assert_array_equal(indices, [[0, 1]])
    unbiased = jax.nn.sigmoid(jnp.array([[1., 2.]], jnp.float32))
    np.testing.assert_allclose(gates, 1.5 * unbiased / unbiased.sum(-1, keepdims=True))
    assert gates.dtype == jnp.float32
    np.testing.assert_array_equal(jax.grad(lambda b: route(x, router, b, 2, 1.5)[1].sum())(bias), jnp.zeros_like(bias))


@pytest.mark.parametrize("training,n", [(False, 9), (True, 2), (True, 9)])
def test_dispatch_reference_masks_overflow_and_gradients(training, n):
    w = weights()
    x = jax.random.normal(jax.random.PRNGKey(8), (n, 80, 4))
    mask = jnp.ones(x.shape[:-1]).at[-1].set(0)
    bias = jnp.array([2., 1., -1., -2.])
    y, stats, _ = dispatch(x, mask, *w, bias, CFG, training=training)
    close(y, reference(x, mask, w, bias, CFG, training))
    assert np.isfinite(stats).all()
    assert float(stats[12]) == float(mask.sum())
    assert float(stats[8:12].sum()) > 0 if training and n > 2 else float(stats[8:12].sum()) == 0
    # Compare derivatives to all-expert evaluation with the identical fixed
    # selected indices and cutoff mask, away from discrete routing boundaries.
    idx, gate = route(x.reshape(-1, 4), w[0], bias, 2, 1.)
    onehot = jax.nn.one_hot(idx, 4, dtype=jnp.int32) * mask.reshape(-1, 1, 1).astype(jnp.int32)
    rank = (jnp.cumsum(onehot.reshape(-1, 4), axis=0) * onehot.reshape(-1, 4)).sum(-1) - 1
    cap = max(int(.2 * float(mask.sum()) * 2 / 4), min(int(mask.sum()), 256))
    kept = (rank.reshape(-1, 2) < cap) if training else jnp.ones_like(idx, bool)
    def dense_grad(args):
        xx, ww = args
        xx2 = xx.reshape(-1, 4)
        _, gg = route(xx2, ww[0], bias, 2, 1.)
        out = swiglu(xx2, ww[3], ww[4])
        for e in range(4):
            out += (gg * (idx == e) * kept).sum(-1)[:, None] * swiglu(xx2, ww[1][e], ww[2][e])
        return (jnp.where(mask.reshape(-1, 1) > 0, out, 0) ** 2).sum()
    def actual_grad(args):
        xx, ww = args
        return jnp.square(dispatch(xx, mask, *ww, bias, CFG, training=training)[0]).sum()
    close(jax.jit(jax.grad(actual_grad))((x, w)), jax.jit(jax.grad(dense_grad))((x, w)))


def test_prediction_batching_and_nested_derivatives():
    w, bias = weights(), jnp.array([2., 1., -1., -2.])
    x = jax.random.normal(jax.random.PRNGKey(8), (3, 5, 4))
    def predict(x):
        return dispatch(x, None, *w, bias, CFG)[0]
    close(predict(x), jnp.concatenate([predict(row[None]) for row in x]))
    close(jax.jit(jax.vmap(lambda row: predict(row[None])[0]))(x), predict(x))
    fun = lambda x: predict(x).sum()
    hv = jax.jit(lambda x: jax.jvp(jax.grad(fun), (x,), (jnp.ones_like(x),))[1])(x)
    assert np.isfinite(hv).all()
    scanned = jax.jit(lambda x: jax.lax.scan(lambda c, _: (jnp.tanh(predict(c)), None), x, None, length=2)[0])(x)
    assert np.isfinite(scanned).all()


def fixture(runtime=None, particles=1, n=5):
    w = weights()
    def model(X_left, Y_obs, obs_scale, X_single, Y_single_obs, obs_scale_single):
        head = numpyro.sample("head", dist.Normal(jnp.zeros(4), jnp.ones(4)).to_event(1))
        params = {f"W_moe_{name}_l1": numpyro.param(name, value) for name, value in
                  zip(("router", "expert1", "expert2", "shared1", "shared2"), w)}
        for branch, x, y, scale in (("pair", X_left, Y_obs, obs_scale),
                                     ("single", X_single, Y_single_obs, obs_scale_single)):
            set_branch(branch, scale, x.shape[0])
            # Two repeated candidate blocks stress global repeat-major order.
            z = jnp.concatenate((x, x * .8), axis=0) if branch == "pair" else x
            z = ffn(z, jnp.ones(z.shape[:-1]), params, CFG, 1)
            z = z[:x.shape[0]].mean(1)
            numpyro.factor(branch, (scale * dist.Normal(z @ head, 1).log_prob(y)).sum())
    wrapped = wrap_model(model, CFG, initial_bias=jnp.array([[2., 1., -1., -2.]]), runtime=runtime)
    svi = MoESVI(wrapped, AutoNormal(wrapped), optax_to_numpyro(optax.adam(1e-3)), TraceMeanField_ELBO(num_particles=particles), CFG)
    args = dict(X_left=np.asarray(jax.random.normal(jax.random.PRNGKey(9), (n, 80, 4))),
                Y_obs=np.linspace(-1, 1, n).astype(np.float32), obs_scale=np.ones(n, np.float32),
                X_single=np.ones((3, 24, 4), np.float32), Y_single_obs=np.ones(3, np.float32),
                obs_scale_single=np.array([1., 0., 1.], np.float32))
    state = svi.init(jax.random.PRNGKey(3), **args)
    return svi, state, args


@pytest.mark.parametrize("particles", [1, 3])
def test_svi_state_particles_rejection_and_scan(particles):
    svi, state, args = fixture(particles=particles)
    assert _context.get() is None
    original_bias = np.asarray(router_bias(state)).copy()
    updated, loss = jax.jit(lambda s: svi.stable_update(s, **args))(state)
    assert np.isfinite(loss)
    np.testing.assert_array_equal(router_bias(state), original_bias)
    assert int(updated.mutable_state[STATE]["updates"]) == 1
    assert not np.array_equal(router_bias(updated), original_bias)
    assert float(diagnostics(updated)["tokens"].sum()) == particles * (2 * 5 * 80 + 2 * 24)
    bad = dict(args, Y_obs=np.full_like(args["Y_obs"], np.nan))
    rejected, loss = jax.jit(lambda s: svi.stable_update(s, **bad))(updated)
    assert np.isnan(loss)
    close(rejected.optim_state, updated.optim_state)
    close(rejected.mutable_state, updated.mutable_state)
    expected, _ = svi.stable_update(updated, **args)
    scanned, _ = jax.jit(lambda s: jax.lax.scan(lambda c, _: svi.stable_update(c, **args), s, None, length=2))(state)
    close(scanned, expected)
    assert "_transformer_moe_bias" not in svi.optim.get_params(updated.optim_state)
    warm = svi.init(jax.random.PRNGKey(7), init_params=svi.get_params(updated), **args)
    close(router_bias(warm), router_bias(updated))
    assert int(warm.mutable_state[STATE]["updates"]) == 0
    assert not np.asarray(warm.mutable_state[STATE]["stats"]).any()
    mutable = jax.tree.map(lambda x: x, updated.mutable_state)
    mutable[STATE]["bias"] = -router_bias(updated)
    reversed_state = updated._replace(mutable_state=mutable)
    _, expected_loss = svi.stable_update(reversed_state, **args)
    close(svi.evaluate(reversed_state, **args), expected_loss)
    close(router_bias(reversed_state), -router_bias(updated))


@pytest.mark.parametrize("particles", [1, 2])
def test_distributed_update_gradients_and_recovery(particles, tmp_path):
    if len(jax.devices()) < 2:
        pytest.skip("requires simulated devices")
    rt = Runtime(dict(enabled=True))
    svi, state, args = fixture(rt, particles=particles)
    expected = jax.jit(lambda s: svi.stable_update(s, **args))(state)
    rt.configure_svi_sharding()
    batch = rt.place_batch(args)
    actual = rt.update(svi, rt.replicate(state), batch)
    close(actual, expected)
    generation = rt.save_checkpoint(tmp_path, "latest", actual[0], [1, 2])
    restored = rt.restore_state(tmp_path, actual[0], generation)
    close(restored, actual[0])
    single = Runtime(dict(enabled=False))
    local = single.restore_state(tmp_path, state, generation)
    expected2 = jax.jit(lambda s: svi.stable_update(s, **args))(local)
    close(rt.update(svi, restored, batch), expected2)
    chunks = rt.stack_batches([batch, batch])
    close(rt.update(svi, rt.replicate(state), chunks, scan=True)[0], expected2[0])
