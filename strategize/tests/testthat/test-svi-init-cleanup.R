test_that("post-init cleanup is inert unless explicitly enabled", {
  withr::local_envvar(STRATEGIZE_SVI_POST_INIT_CLEANUP = "false")
  expect_null(strategize:::strategize_svi_post_init_cleanup(NULL, runtime = list()))
})

test_that("post-init cleanup preserves donated SVI updates and guide state", {
  skip_if_no_jax()
  if (!reticulate::py_module_available("numpyro")) skip("NumPyro unavailable")
  # Use an isolated Python dictionary so the regression leaves no SVI state
  # or compiled update in reticulate's shared main module.
  py <- reticulate::py_run_string(paste(c(
    "import pickle, random",
    "import jax, jax.numpy as jnp, numpy as np, numpyro",
    "import numpyro.distributions as dist",
    "from numpyro.infer import SVI, TraceMeanField_ELBO",
    "from numpyro.infer.autoguide import AutoNormal",
    "def model(x, y, numpyro=numpyro, jnp=jnp, dist=dist):",
    "    w = numpyro.param('weight', jnp.ones((4, 4)) * 0.1)",
    "    b = numpyro.sample('bias', dist.Normal(0., 1.))",
    "    numpyro.sample('y', dist.Normal((x @ w).sum(-1) + b, 1.), obs=y)",
    "x = jnp.arange(24, dtype=jnp.float32).reshape(6, 4) / 24",
    "y = jnp.linspace(-0.5, 0.5, 6)",
    "guide = AutoNormal(model)",
    "svi = SVI(model, guide, numpyro.optim.Adam(0.01), TraceMeanField_ELBO())",
    "initial = svi.init(jax.random.PRNGKey(42), x, y)",
    "initial_snapshot = [np.array(a) for a in jax.tree.leaves(initial)]",
    "control = jax.tree.map(lambda a: a.copy(), initial)",
    "treatment = jax.tree.map(lambda a: a.copy(), initial)",
    "update = jax.jit(svi.stable_update, donate_argnums=(0,))",
    "control_after, control_loss = update(control, x, y)",
    "jax.block_until_ready(control_after)",
    "rng_before = np.array(treatment.rng_key)",
    "random_before = pickle.dumps((random.getstate(), np.random.get_state()))"
  ), collapse = "\n"), local = TRUE, convert = FALSE)
  withr::local_seed(12L)
  r_rng_before <- .Random.seed
  runtime <- list(jax = reticulate::import("jax", convert = FALSE),
                  py_gc = reticulate::import("gc", convert = FALSE))
  expect_message(report <- strategize:::strategize_svi_post_init_cleanup(
    py$treatment, runtime = runtime, enabled = TRUE
  ), "SVI_POST_INIT_MEMORY")
  expect_named(report, c("before", "after"))
  expect_named(report$before[[1L]], c("device", "memory"))
  expect_named(report$after[[1L]], c("device", "memory"))
  expect_identical(.Random.seed, r_rng_before)
  # Execute comparisons in the same dictionary that owns these Python objects.
  builtins <- reticulate::import_builtins(convert = FALSE)
  expect_no_error(builtins$exec(paste(c(
    "assert pickle.dumps((random.getstate(), np.random.get_state())) == random_before",
    "np.testing.assert_array_equal(treatment.rng_key, rng_before)",
    "treatment_after, treatment_loss = update(treatment, x, y)",
    "jax.block_until_ready(treatment_after)",
    "for a, b in zip(jax.tree.leaves(control_after), jax.tree.leaves(treatment_after)):",
    "    np.testing.assert_array_equal(a, b)",
    "np.testing.assert_array_equal(control_loss, treatment_loss)",
    "assert np.isfinite(np.asarray(treatment_loss))",
    "for a, b in zip(initial_snapshot, jax.tree.leaves(initial)):",
    "    np.testing.assert_array_equal(a, b)",
    "control_second, control_loss_second = update(control_after, x, y)",
    "treatment_second, treatment_loss_second = update(treatment_after, x, y)",
    "jax.block_until_ready(treatment_second)",
    "for a, b in zip(jax.tree.leaves(control_second), jax.tree.leaves(treatment_second)):",
    "    np.testing.assert_array_equal(a, b)",
    "np.testing.assert_array_equal(control_loss_second, treatment_loss_second)",
    "params = svi.get_params(treatment_second)",
    "assert np.any(np.asarray(params['weight']) != 0.1)",
    "posterior = guide.sample_posterior(jax.random.PRNGKey(11), params, sample_shape=(2,))",
    "assert all(np.isfinite(np.asarray(a)).all() for a in jax.tree.leaves(posterior))"
  ), collapse = "\n"), py, py))
})
