"""Strict Muon verifies real state, partitioning, and resumed updates."""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "inst/python"))
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import optax

from strategize_optim import strict_muon_optimizer, normalized_optimizer, update_diagnostics


def dims(params):
    return {k: optax.contrib.MuonDimensionNumbers(v.ndim - 2, v.ndim - 1)
            if k.startswith("hidden") else None for k, v in params.items()}


class StrictMuonTests(unittest.TestCase):
    def setUp(self):
        self.params = {"hidden": jnp.ones((3, 4)), "hidden_experts": jnp.ones((2, 3, 4)),
                       "router": jnp.ones((3, 2)), "embedding": jnp.ones((5, 4)),
                       "output": jnp.ones((4, 1)), "scale": jnp.ones((3, 4))}

    def test_state_counts_and_resumed_schedule(self):
        schedule = optax.warmup_cosine_decay_schedule(1e-5, 5e-4, 2, 8, 1e-5)
        optim = normalized_optimizer(strict_muon_optimizer(schedule, dims), .25, 1.)
        state = optim.init(self.params)
        info = update_diagnostics(state)
        self.assertEqual(info["muon_parameter_count"], 36)
        self.assertEqual(info["auxiliary_adam_parameter_count"], 42)
        params = self.params

        @jax.jit
        def step(p, s):
            grads = jax.tree.map(lambda x: x * .2, p)
            updates, s = optim.update(grads, s, p)
            return optax.apply_updates(p, updates), s

        for _ in range(3):
            params, state = step(params, state)
        restored = pickle.loads(pickle.dumps(jax.device_get((params, state))))
        for _ in range(3):
            params, state = step(params, state)
            restored = step(*restored)
        for expected, actual in zip(jax.tree.leaves((params, state)), jax.tree.leaves(restored)):
            np.testing.assert_array_equal(expected, actual)
        self.assertEqual(update_diagnostics(state)["update_count"], 6)

    def test_experts_are_orthogonalized_independently(self):
        p = {"hidden_experts": jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4) / 10}
        optim = strict_muon_optimizer(.001, dims)
        batched, _ = optim.update(p, optim.init(p), p)
        for i in range(2):
            single = {"hidden": p["hidden_experts"][i]}
            update, _ = optim.update(single, optim.init(single), single)
            np.testing.assert_allclose(batched["hidden_experts"][i], update["hidden"], rtol=1e-5, atol=1e-7)

    def test_incompatible_api_is_not_retried(self):
        with patch.object(optax.contrib, "muon", side_effect=TypeError("consistent_rms")) as factory:
            with self.assertRaisesRegex(ValueError, "required settings"):
                strict_muon_optimizer(.001, dims)
            self.assertEqual(factory.call_count, 1)
            self.assertEqual(factory.call_args.kwargs["consistent_rms"], .2)
            self.assertEqual(factory.call_args.kwargs["weight_decay"], 0)
            self.assertEqual(factory.call_args.kwargs["adam_weight_decay"], 0)
        with self.assertRaisesRegex(ValueError, "explicit"):
            strict_muon_optimizer(.001, None)

    def test_weight_decay_reaches_muon_and_is_validated(self):
        with patch.object(optax.contrib, "muon", side_effect=TypeError("stop")) as factory:
            with self.assertRaises(ValueError):
                strict_muon_optimizer(.001, dims, weight_decay=.1)
            self.assertEqual(factory.call_args.kwargs["weight_decay"], .1)
            self.assertEqual(factory.call_args.kwargs["adam_weight_decay"], 0)
        for bad in (-.1, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "finite nonnegative"):
                strict_muon_optimizer(.001, dims, weight_decay=bad)

    def test_weight_decay_changes_only_muon_matrices(self):
        lr, wd = .01, .3
        grads = jax.tree.map(lambda x: x * .2, self.params)
        plain = strict_muon_optimizer(lr, dims)
        decayed = strict_muon_optimizer(lr, dims, weight_decay=wd)
        base, base_state = plain.update(grads, plain.init(self.params), self.params)
        extra, extra_state = decayed.update(grads, decayed.init(self.params), self.params)
        for key in self.params:
            expected = base[key] - lr * wd * self.params[key] if key.startswith("hidden") else base[key]
            np.testing.assert_allclose(extra[key], expected, rtol=1e-6, atol=1e-7)
        for a, b in zip(jax.tree.leaves(base_state), jax.tree.leaves(extra_state)):
            np.testing.assert_array_equal(a, b)

    def test_parameter_norm_reads_params_through_a_barrier(self):
        # With weight decay each Muon update reads its parameter; the diagnostic
        # parameter norm must not share that read, or XLA fuses every leaf's update
        # into one single-block reduction that ROCm cannot launch.
        optim = normalized_optimizer(strict_muon_optimizer(1e-3, dims, weight_decay=.1), 1., 10.)
        state = optim.init(self.params)
        grads = jax.tree.map(lambda x: x * .2, self.params)
        jaxpr = jax.make_jaxpr(optim.update)(grads, state, self.params)
        barriers = [e for e in jaxpr.jaxpr.eqns if e.primitive.name == "optimization_barrier"]
        self.assertEqual([len(e.invars) for e in barriers], [len(self.params)])
        updates, state = optim.update(grads, state, self.params)
        expected = optax.tree.norm(updates) / optax.tree.norm(self.params)
        self.assertAlmostEqual(update_diagnostics(state)["last_update_parameter_ratio"], float(expected), places=6)

    def test_empty_or_substituted_muon_state_fails(self):
        optim = strict_muon_optimizer(.001, dims)
        with self.assertRaisesRegex(ValueError, "no eligible"):
            optim.init({"output": jnp.ones((2, 2))})
        with self.assertRaisesRegex(ValueError, "no eligible"):
            optim.init({"hidden": jnp.ones((0, 2)), "output": jnp.ones((2,))})
        factory = optax.contrib.muon
        with patch.object(optax.contrib, "muon", side_effect=lambda **kw: factory(kw["learning_rate"])):
            with self.assertRaisesRegex(ValueError, "weight partition"):
                strict_muon_optimizer(.001, dims).init(self.params)
        with patch.object(optax.contrib, "muon", return_value=optax.adam(.001)):
            with self.assertRaisesRegex(ValueError, "Muon momentum"):
                strict_muon_optimizer(.001, dims).init(self.params)
        adam_state = optax.adam(.001).init(self.params)
        with self.assertRaisesRegex(ValueError, "Muon momentum"):
            optim.update(self.params, adam_state, self.params)


if __name__ == "__main__":
    unittest.main()


def test_distributed_muon_update_and_full_state_resume(tmp_path):
    import pytest
    import numpyro
    from numpyro.infer import SVI, Trace_ELBO
    from numpyro.optim import optax_to_numpyro
    from strategize_distributed import Runtime
    if len(jax.devices()) < 2:
        pytest.skip("requires two simulated CPU devices")

    def model(X_left, Y_obs, obs_scale):
        w = numpyro.param("hidden", jnp.ones((3, 2)))
        bias = numpyro.param("output", jnp.ones((2,)))
        error = X_left @ w + bias - Y_obs
        numpyro.factor("loss", -jnp.sum(obs_scale[:, None] * error**2))

    schedule = optax.cosine_decay_schedule(.001, 8)
    optim = normalized_optimizer(strict_muon_optimizer(schedule, dims), 1/8, 1.)
    svi = SVI(model, lambda **kw: None, optax_to_numpyro(optim), Trace_ELBO())
    args = dict(X_left=np.arange(24, dtype=np.float32).reshape(8, 3)/10,
                Y_obs=np.ones((8, 2), np.float32), obs_scale=np.ones(8, np.float32))
    state = svi.init(jax.random.PRNGKey(7), **args)
    runtime = Runtime(dict(enabled=True))
    runtime.configure_svi_sharding()
    batch = runtime.place_batch(args)
    expected = jax.jit(lambda s: svi.stable_update(s, **args))(state)
    actual = runtime.update(svi, runtime.replicate(state), batch)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
    generation = runtime.save_checkpoint(tmp_path, "latest", actual[0], [1])
    single = Runtime(dict(enabled=False))
    restored = single.restore_state(tmp_path, state, generation)
    adam_optimizer = optax_to_numpyro(normalized_optimizer(optax.adam(.001)))
    adam_template = state._replace(optim_state=adam_optimizer.init(svi.optim.get_params(state.optim_state)))
    with pytest.raises(RuntimeError, match="optimizer/SVI structure differs"):
        single.restore_state(tmp_path, adam_template, generation)
    resumed = jax.jit(lambda s: svi.stable_update(s, **args))(restored)
    uninterrupted = runtime.update(svi, actual[0], batch)
    for a, b in zip(jax.tree.leaves(resumed), jax.tree.leaves(uninterrupted)):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
    assert update_diagnostics(restored.optim_state)["muon_parameter_count"] == 6
