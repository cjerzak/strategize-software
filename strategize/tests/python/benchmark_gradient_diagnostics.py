"""Warmed, synchronized before/after diagnostic benchmark (CPU or ROCm)."""

import json
import statistics
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro.infer import SVI, Trace_ELBO
from numpyro.optim import Adam

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "inst/python"))
from strategize_distributed import Runtime


def benchmark(n):
    def model(X_left):
        w = numpyro.param("weight", jnp.linspace(-1, 1, n))
        numpyro.factor("objective", -jnp.mean(X_left) * jnp.square(jnp.sin(w)).sum())
    svi = SVI(model, lambda **kw: None, Adam(.001), Trace_ELBO())
    runtime = Runtime(dict(enabled=True))
    args = dict(X_left=jnp.arange(8, dtype=jnp.float32))
    state = runtime.replicate(svi.init(jax.random.PRNGKey(3), **args))
    batch = runtime.place_batch(args)
    raw = runtime._compile(svi, batch, state, gradients=True)
    def before():
        grads = raw(state, batch)
        leaves = [np.asarray(a) for a in jax.tree.leaves(runtime.local_tree(grads))]
        finite = [np.where(np.isfinite(a), a, 0).astype(np.float64) for a in leaves]
        ssq = sum(float(np.square(a).sum()) for a in finite)
        return dict(grad_l2=ssq ** .5, grad_rms=(ssq / n) ** .5,
                    grad_max_abs=max(float(np.abs(a).max()) for a in finite),
                    grad_n_nonfinite=sum(int((~np.isfinite(a)).sum()) for a in leaves),
                    grad_n_elements=n)
    after = lambda: runtime.gradients(svi, state, batch)
    a, b = before(), after()
    for key in a:
        np.testing.assert_allclose(a[key], b[key], rtol=2e-6, atol=1e-7)
    elapsed = {"before": [], "after": []}
    for _ in range(2):
        before()
        after()
    for _ in range(7):
        for name, fn in (("before", before), ("after", after)):
            start = time.monotonic()
            fn() # both paths synchronize when returning host scalars
            elapsed[name].append(1000 * (time.monotonic() - start))
    return dict(parameters=n, host_bytes_before=4*n, host_bytes_after=16,
                median_ms={name: statistics.median(v) for name, v in elapsed.items()})


if __name__ == "__main__":
    print(json.dumps(dict(devices=[str(d) + ":" + d.device_kind for d in jax.devices()],
                          results=[benchmark(n) for n in (1 << 14, 1 << 20, 1 << 22)]), indent=2))
