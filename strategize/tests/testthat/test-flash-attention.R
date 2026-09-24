test_that("Pallas flash attention matches XLA values and gradients on GPU", {
  skip_on_cran()
  skip_if_no_jax()
  path <- system.file("python", package = "strategize")
  transformer <- reticulate::import_from_path("strategize_transformer", path = path)
  reticulate::import_from_path("strategize_attention", path = path)
  if (!reticulate::import("jax")$default_backend() %in% c("gpu", "cuda", "rocm")) skip("No GPU")
  status <- transformer$pallas_status(15L)
  if (nzchar(status)) skip(status)
  reticulate::py_run_string("
import jax, jax.numpy as jnp
from strategize_transformer import self_attention
from strategize_attention import mla_attention

def rel(a, b):
    return max(float(jnp.max(jnp.abs(x - y)) / (jnp.max(jnp.abs(y)) + 1e-12))
               for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)))

key = iter(jax.random.split(jax.random.PRNGKey(0), 32))
B, S, H, D = 6, 37, 2, 15  # S is not a block multiple; D is not a multiple of 8
mask = (jax.random.uniform(next(key), (B, S)) < .7).at[:, 0].set(True).at[1, 1:].set(False).astype(jnp.float32)
qkv = [jax.random.normal(next(key), (2, B, S, H, D)) for _ in range(3)]

def helper_loss(q, k, v, backend):
    run = jax.checkpoint(lambda q, k, v: self_attention(q, k, v, mask, H * D, H, D, backend, 'auto', 8))
    out = jax.vmap(run)(q, k, v)
    return jnp.sum(jnp.where(mask[None, :, :, None, None] > 0, out, 0) ** 2), out

helper = [jax.jit(jax.value_and_grad(helper_loss, argnums=(0, 1, 2), has_aux=True), static_argnums=3)(*qkv, b)
          for b in ('pallas', 'xla')]
helper_finite = bool(jnp.all(jnp.isfinite(helper[0][0][1])))

dims, rank = H * D, 16
shapes = dict(W_q=(dims, rank), RMS_q_latent=(rank,), W_k=(dims, rank), RMS_kv_latent=(rank,),
              W_q_up=(rank, H * D), W_v=(rank, 2 * H * D), W_o=(H * D, dims), RMS_q=(D,), RMS_k=(D,))
params = {n: (1. + .1 * jax.random.normal(next(key), s)) if n.startswith('RMS') else
          jax.random.normal(next(key), s) / s[0] ** .5 for n, s in shapes.items()}
x = jax.random.normal(next(key), (B, S, dims))

def mla_loss(x, params, backend):
    cfg = dict(architecture='mla', top_k=32, attention_backend=backend, attention_dtype='auto')
    out = mla_attention(x, mask, params, cfg, H, D)[0]
    return jnp.sum(out ** 2)

mla = [jax.jit(jax.value_and_grad(mla_loss, argnums=(0, 1)), static_argnums=2)(x, params, b)
       for b in ('pallas', 'xla')]
flash_attention_errors = dict(helper_value=rel(jnp.where(mask[None, :, :, None, None] > 0, helper[0][0][1], 0),
                               jnp.where(mask[None, :, :, None, None] > 0, helper[1][0][1], 0)),
              helper_grad=rel(helper[0][1], helper[1][1]),
              mla_value=rel(mla[0][0], mla[1][0]), mla_grad=rel(mla[0][1], mla[1][1]),
              finite=helper_finite, cuda='cuda' in jax.devices()[0].client.platform_version.lower())
")
  errors <- reticulate::py$flash_attention_errors
  # CUDA Pallas dots use TF32, like XLA's default; ROCm matches to FP32 rounding.
  tolerance <- if (isTRUE(errors$cuda)) 2e-2 else 1e-4
  expect_true(errors$finite)
  for (name in c("helper_value", "helper_grad", "mla_value", "mla_grad")) {
    expect_lt(errors[[name]], tolerance, label = name)
  }
})
