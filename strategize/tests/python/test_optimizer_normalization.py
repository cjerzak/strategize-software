import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, str(Path(__file__).parents[2] / "inst" / "python"))
from strategize_optim import normalized_optimizer, update_diagnostics


def test_population_scaling_precedes_clipping_and_tracks_actual_updates():
    optim = normalized_optimizer(optax.sgd(0.1), 1e-6, 10.0)
    params = {"w": jnp.array([3., 4.])}
    state = optim.init(params)
    step = jax.jit(optim.update)
    updates, state = step({"w": jnp.array([3e6, 4e6])}, state, params)
    np.testing.assert_allclose(updates["w"], [-.3, -.4], rtol=1e-6)
    assert update_diagnostics(state)["clipped_update_count"] == 0
    updates, state = step({"w": jnp.array([12e6, 16e6])}, state, params)
    np.testing.assert_allclose(updates["w"], [-.6, -.8], rtol=1e-6)
    diag = update_diagnostics(state)
    assert diag["update_count"] == 2
    assert diag["clipped_update_fraction"] == .5
    assert np.isclose(diag["mean_clip_factor"], .75)
    assert np.isclose(diag["last_update_parameter_ratio"], .2)


def test_scaling_preserves_likelihood_to_kl_ratio():
    n = 1000
    def loss(w):
        likelihood = n * (w - 2.) ** 2
        kl = w ** 2 / 2
        return likelihood + kl
    params = jnp.array(.7)
    grad = jax.grad(loss)(params)
    optim = normalized_optimizer(optax.sgd(.01), 1/n, None)
    updates, _ = optim.update(grad, optim.init(params), params)
    expected = -.01 * jax.grad(lambda w: loss(w)/n)(params)
    np.testing.assert_allclose(updates, expected, rtol=1e-6)


def test_recurrent_group_norms_and_use_count_normalization():
    params = {name: jnp.array([1.]) for name in (
        "W_q_l1", "W_q_l2", "W_q_l3", "loop_input_gain", "head")}
    grads = {
        "W_q_l1": jnp.array([1.]),
        "W_q_l2": jnp.array([3.]),
        "W_q_l3": jnp.array([2.]),
        "loop_input_gain": jnp.array([4.]),
        "head": jnp.array([6.]),
    }
    optim = normalized_optimizer(
        optax.sgd(1.), prelude_layers=[1], recurrent_core_layers=[2],
        coda_layers=[3], recurrent_gradient_scale=.5)
    updates, state = jax.jit(optim.update)(grads, optim.init(params), params)
    np.testing.assert_allclose(updates["W_q_l1"], [-1.])
    np.testing.assert_allclose(updates["W_q_l2"], [-1.5])
    np.testing.assert_allclose(updates["loop_input_gain"], [-2.])
    np.testing.assert_allclose(updates["W_q_l3"], [-2.])
    np.testing.assert_allclose(updates["head"], [-6.])
    diag = update_diagnostics(state)
    assert np.isclose(diag["last_prelude_gradient_norm"], 1.)
    assert np.isclose(diag["last_recurrent_core_gradient_norm"], 5.)
    assert np.isclose(diag["last_recurrent_core_scaled_gradient_norm"], 2.5)
    assert np.isclose(diag["last_coda_gradient_norm"], 2.)
    assert np.isclose(diag["last_other_gradient_norm"], 6.)


def test_recurrent_group_mapping_covers_full_and_moe_suffix_stacks():
    params = {
        "W_q_layers": jnp.ones((8, 1)),
        "W_moe_expert1_layers": jnp.ones((7, 1)),
        "W_ff1_layers": jnp.ones((1, 1)),
    }
    grads = jax.tree.map(jnp.ones_like, params)
    optim = normalized_optimizer(
        optax.sgd(1.), prelude_layers=[1],
        recurrent_core_layers=[2, 3, 4, 5, 6, 7], coda_layers=[8],
        recurrent_gradient_scale=.5)
    updates, state = jax.jit(optim.update)(grads, optim.init(params), params)
    np.testing.assert_allclose(updates["W_q_layers"][:, 0],
                               [-1., -.5, -.5, -.5, -.5, -.5, -.5, -1.])
    np.testing.assert_allclose(updates["W_moe_expert1_layers"][:, 0],
                               [-.5, -.5, -.5, -.5, -.5, -.5, -1.])
    np.testing.assert_allclose(updates["W_ff1_layers"][:, 0], [-1.])
    diag = update_diagnostics(state)
    assert np.isclose(diag["last_prelude_gradient_norm"], np.sqrt(2.))
    assert np.isclose(diag["last_recurrent_core_gradient_norm"], np.sqrt(12.))
    assert np.isclose(diag["last_recurrent_core_scaled_gradient_norm"], np.sqrt(3.))
    assert np.isclose(diag["last_coda_gradient_norm"], np.sqrt(2.))


def test_numpyro_state_and_nonfinite_updates_preserve_telemetry():
    import numpyro
    from numpyro.infer import SVI, Trace_ELBO
    from numpyro.optim import optax_to_numpyro
    def model(target):
        w = numpyro.param("w", jnp.array(1.))
        numpyro.factor("obs", -1000 * (w - target) ** 2)
    optim = normalized_optimizer(optax.adam(.01), .001, 10.)
    svi = SVI(model, lambda target: None, optax_to_numpyro(optim), Trace_ELBO())
    state = svi.init(jax.random.key(1), 0.)
    state, _ = jax.jit(svi.stable_update)(state, 0.)
    assert update_diagnostics(state.optim_state)["update_count"] == 1
    after, _ = jax.jit(svi.stable_update)(state, jnp.nan)
    assert update_diagnostics(after.optim_state)["update_count"] == 1
    leaves, treedef = jax.tree.flatten(after)
    restored = jax.tree.unflatten(treedef, leaves)
    assert update_diagnostics(restored.optim_state) == update_diagnostics(after.optim_state)
