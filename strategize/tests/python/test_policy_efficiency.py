import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("strategize_policy",
    Path(__file__).parents[2] / "inst/python/strategize_policy.py")
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


@pytest.mark.parametrize("binomial", [True, False])
def test_factored_glm_matches_pair_features_and_derivatives(binomial):
    rng = np.random.default_rng(19)
    x, y = jnp.asarray(rng.normal(size=(7, 5))), jnp.asarray(rng.normal(size=(4, 5)))
    coefs = jnp.asarray(rng.normal(size=(2, 8)))
    left, right = jnp.array([0, 1, 2]), jnp.array([3, 4, 4])
    model = policy.GLMPairs(jnp.arange(5), jnp.arange(5, 8), left, right,
                           binomial=binomial, ast_prop=.3, dag_prop=.7, strength=1.7)

    def original(a, b, c):
        # Existing pair evaluator constructs differences before weighting.
        delta = a[:, None, :] - b[None, :, :]
        delta_inter = ((a[:, left] * a[:, right])[:, None, :]
                       - (b[:, left] * b[:, right])[None, :, :])
        eta = jnp.einsum("ijk,lk->ijl", delta, c[:, :5])
        eta += jnp.einsum("ijk,lk->ijl", delta_inter, c[:, 5:])
        eta += jnp.array([.12, -.23])
        q = jax.nn.sigmoid(eta) if binomial else eta
        return q @ jnp.array([.3, .7])

    def factored(a, b, c):
        sa, sb = model.scores(a, c), model.scores(b, c)
        return model.population(sa[:, None, :], sb[None, :, :], .12, -.23)

    np.testing.assert_allclose(factored(x, y, coefs), original(x, y, coefs), atol=2e-6)
    for actual, expected in zip(
        jax.grad(lambda *args: jnp.sum(factored(*args)**2), (0, 1, 2))(x, y, coefs),
        jax.grad(lambda *args: jnp.sum(original(*args)**2), (0, 1, 2))(x, y, coefs)):
        np.testing.assert_allclose(actual, expected, atol=5e-5, rtol=5e-5)


@pytest.mark.parametrize("chunk", [1, 3, 16])
def test_chunked_pullback_including_tail_equals_full_jacobian(chunk):
    args = (jnp.arange(5, dtype=jnp.float32) / 8, jnp.array([[.3], [.4]]))
    def f(args):
        a, b = args
        return jnp.concatenate((jnp.sin(a * b[0, 0]), jnp.cos(b[:, 0] * a.sum())))[:, None]
    full = jax.jacrev(f)(args)
    chunks = policy.chunked_jacrev(f, args, chunk_size=chunk)
    for c, j in zip(chunks, full):
        assert isinstance(c, np.ndarray)
        np.testing.assert_allclose(c, np.asarray(j).reshape(7, -1), atol=1e-6)


@pytest.mark.parametrize("optimism", ["none", "ogda", "extragrad", "smp", "rain"])
def test_scan_remat_preserves_full_trace_sensitivities(optimism):
    # Sign is a tracer in the supplied gradients, as it is in the R bridge.
    def objective(a, b, theta, sign, key):
        payoff = jnp.sum(jnp.sin(a * theta) * b) + .03 * jax.random.normal(key)
        return sign * payoff - .2 * jnp.sum(jnp.where(sign > 0, a, b)**2)

    a, b, theta = jnp.array([.2, .4]), jnp.array([.5, -.2]), jnp.array([.7, .9])
    n = 5
    schedule = (jnp.arange(n), jnp.array([True, False, False, True, False]),
                jnp.array([0, 0, 1, 0, 1]), jnp.array([.2, .2, .2, .4, .4]),
                jnp.array([0, 0, 0, 1, 1]), jnp.zeros(n, bool))
    loops = [policy.PolicyLoop(jax.jit(jax.value_and_grad(objective, 0)),
                              jax.jit(jax.value_and_grad(objective, 1)),
                              n, adversarial=True, optimism=optimism, remat=r)
             for r in (False, True)]
    def result(loop, theta, history):
        return loop.run(a, b, jax.random.PRNGKey(9), (theta,), schedule, history=history)
    np.testing.assert_allclose(result(loops[0], theta, True)["a"],
                               result(loops[1], theta, False)["a"], atol=1e-6)
    assert result(loops[1], theta, False)["history"] is None
    jacobians = [jax.jacrev(lambda t: result(loop, t, False)["a"])(theta) for loop in loops]
    np.testing.assert_allclose(*jacobians, atol=1e-6)
