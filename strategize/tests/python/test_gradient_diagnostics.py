"""Run on CPU (including virtual meshes) and ROCm; no model-sized host output."""

import sys
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "inst/python"))
from strategize_diagnostics import gradient_summary, gradient_statistics
from strategize_distributed import Runtime
from test_data_parallel import fixture, close


def test_diagnostics_survive_temporary_module_search_path(tmp_path):
    # R's import_from_path removes the runtime directory from sys.path after
    # import. A fresh interpreter prevents other tests from preloading helpers.
    code = """
import sys
sys.path.insert(0, sys.argv[1])
from strategize_distributed import Runtime
sys.path.pop(0)
import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import SVI, Trace_ELBO
from numpyro.optim import Adam
def model(x):
    w = numpyro.param('w', jnp.ones(3))
    numpyro.factor('loss', -(x * w).sum())
svi = SVI(model, lambda **kw: None, Adam(.001), Trace_ELBO())
args = dict(x=jnp.ones(3))
runtime = Runtime(dict(enabled=True))
state = runtime.replicate(svi.init(jax.random.PRNGKey(0), **args))
stats = runtime.gradients(svi, state, runtime.place_batch(args))
assert stats['grad_n_elements'] == 3 and stats['grad_n_nonfinite'] == 0
assert abs(stats['grad_l2'] - 3**.5) < 1e-6
"""
    result = subprocess.run([sys.executable, "-c", code,
        str(Path(__file__).resolve().parents[2] / "inst/python")],
        cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def reference(tree):
    leaves = [np.asarray(a) for a in jax.tree.leaves(tree)]
    n = sum(a.size for a in leaves)
    finite = [np.where(np.isfinite(a), a, 0).astype(np.float64) for a in leaves]
    ssq = sum(np.square(a).sum() for a in finite)
    return dict(grad_l2=float(np.sqrt(ssq)), grad_rms=float(np.sqrt(ssq / max(n, 1))),
                grad_max_abs=max((np.abs(a).max() for a in finite if a.size), default=0.),
                grad_n_nonfinite=sum(int((~np.isfinite(a)).sum()) for a in leaves),
                grad_n_elements=n)


@pytest.mark.parametrize("values", [[], [0., 0.], [3., -4.],
    [np.nan, np.inf, -np.inf, 3., -4.], [np.nan, np.inf], [1e30, -2e30], [1e-30, -2e-30]])
def test_scalar_summary_matches_host_reference(values):
    grads = {"matrix": jnp.array(values, dtype=jnp.float32), "empty": jnp.zeros((0, 3))}
    fn = jax.jit(gradient_summary)
    result = fn(grads)
    assert all(a.ndim == 0 for a in jax.tree.leaves(result))
    actual = gradient_statistics(jax.device_get(result), len(values))
    expected = reference(grads)
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], rtol=2e-6, atol=0)
    assert actual["grad_n_nonfinite"] == expected["grad_n_nonfinite"]


def test_empty_tree_and_large_exact_counts():
    assert gradient_statistics(jax.device_get(jax.jit(gradient_summary)({})), 0) == reference({})
    stats = gradient_statistics(dict(scale=0., ssq=0., nonfinite_lo=19, nonfinite_hi=3), 2**34)
    assert stats["grad_n_nonfinite"] == 3 * 2**32 + 19
    assert stats["grad_n_elements"] == 2**34


@pytest.mark.parametrize("manual", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_distributed_metrics_after_global_reduction(manual, normalize, monkeypatch):
    runtime = Runtime(dict(enabled=True)) # also exercises a one-device ROCm mesh
    svi, state, args = fixture(n=5, m=3, normalize=normalize)
    # Manual sharding uses the production branch/weight argument convention.
    if manual:
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import SVI, TraceMeanField_ELBO
        from numpyro.infer.autoguide import AutoNormal
        from numpyro.optim import Adam
        def model(X_left, Y_obs, obs_scale, X_single, Y_single_obs, obs_scale_single):
            w = numpyro.sample("w", dist.Normal(jnp.zeros(4), jnp.ones(4)).to_event(1))
            numpyro.factor("pair", jnp.sum(obs_scale * dist.Bernoulli(logits=X_left @ w).log_prob(Y_obs)))
            numpyro.factor("single", jnp.sum(obs_scale_single * dist.Normal(X_single @ w, 1).log_prob(Y_single_obs)))
        args = dict(X_left=args["x"], Y_obs=args["y"], obs_scale=args["weights"],
                    X_single=args["x_single"], Y_single_obs=args["y_single"], obs_scale_single=args["weights_single"])
        svi = SVI(model, AutoNormal(model), Adam(.001), TraceMeanField_ELBO())
        state = svi.init(jax.random.PRNGKey(7), **args)
        runtime.configure_svi_sharding()
    batch = runtime.place_batch(args)
    replicated = runtime.replicate(state)
    before = jax.tree.map(lambda a: np.array(a), state)
    raw = runtime._compile(svi, batch, replicated, gradients=True)(replicated, batch)
    expected = reference(raw)
    summary_fn = runtime._compile(svi, batch, replicated, gradients=True, gradient_stats=True)
    assert all(a.ndim == 0 for a in jax.tree.leaves(summary_fn(replicated, batch)))
    compile_count = runtime.compile_count
    device_get = jax.device_get
    def scalar_only(tree):
        assert all(a.ndim == 0 for a in jax.tree.leaves(tree))
        return device_get(tree)
    monkeypatch.setattr(jax, "device_get", scalar_only)
    with jax.transfer_guard("disallow"):
        actual = runtime.gradients(svi, replicated, batch)
    assert runtime.compile_count == compile_count
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], rtol=2e-6, atol=1e-7)
    close(replicated, before)
    # Diagnostics neither consume the next random key nor modify the optimizer.
    close(runtime.update(svi, replicated, batch), runtime.update(svi, runtime.replicate(state), batch))


def test_summary_output_size_is_independent_of_gradient_size():
    sizes = []
    for n in (16, 1 << 20):
        value = jnp.ones(n, jnp.float32)
        fn = jax.jit(gradient_summary).lower({"weight": value}).compile()
        result = fn({"weight": value})
        sizes.append(sum(a.size * a.dtype.itemsize for a in jax.tree.leaves(result)))
    assert sizes == [16, 16]
