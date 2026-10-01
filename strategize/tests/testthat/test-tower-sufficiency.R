test_that("tower sufficiency controls resolve and validate", {
  resolve <- strategize:::neural_resolve_tower_sufficiency
  expect_false(resolve(NULL)$enabled)
  expect_false(resolve(FALSE, 16L)$enabled)
  expect_identical(resolve(TRUE, 16L), list(enabled = TRUE, weight = 1, mode = "distill"))
  expect_identical(resolve(list(weight = 0.25), 16L)$weight, 0.25)
  expect_false(resolve(list(enabled = FALSE, weight = 3), 0L)$enabled)
  expect_error(resolve(TRUE, 0L), "low_rank_interaction_rank > 0")
  expect_error(resolve(list(weight = 0), 16L), "positive finite")
  expect_error(resolve(list(lambda = 1), 16L), "Unknown tower_sufficiency")
})

test_that("tower term leaves the logit but feeds the stop-gradient surrogate", {
  skip_on_cran()
  skip_if_no_jax()
  strategize:::initialize_jax()
  env <- strategize:::strenv
  jnp <- env$jnp
  np <- reticulate::import("numpy", convert = FALSE)
  as_r <- function(x) reticulate::py_to_r(np$asarray(x))
  set.seed(1)
  d <- 6L; k <- 3L
  params <- list(W_rc_r = jnp$array(matrix(rnorm(d * k), d, k), dtype = jnp$float32),
                 W_rc_c = jnp$array(matrix(rnorm(d * k), d, k), dtype = jnp$float32),
                 W_rc_out = jnp$array(matrix(rnorm(k), k, 1L), dtype = jnp$float32))
  r <- jnp$array(matrix(rnorm(4L * d), 4L, d), dtype = jnp$float32)
  c <- jnp$array(matrix(rnorm(4L * d), 4L, d), dtype = jnp$float32)
  logits <- strategize:::neural_low_rank_interaction_logits
  expect_gt(max(abs(as_r(logits(r, c, params)))), 0)
  params$W_ts_cand <- jnp$array(matrix(rnorm(d), d, 1L), dtype = jnp$float32)
  expect_true(strategize:::neural_has_tower_sufficiency(params))
  expect_equal(max(abs(as_r(logits(r, c, params)))), 0)
  expect_gt(max(abs(as_r(logits(r, c, params, surrogate = TRUE)))), 0)

  old <- list(active = env$tower_sufficiency_active, pending = env$tower_sufficiency_pending)
  on.exit({
    env$tower_sufficiency_active <- old$active
    env$tower_sufficiency_pending <- old$pending
  }, add = TRUE)
  env$tower_sufficiency_active <- TRUE
  target <- jnp$array(matrix(c(0.5, -1, 2, 0), 4L, 1L), dtype = jnp$float32)
  env$tower_sufficiency_pending <- list()
  strategize:::neural_collect_tower_sufficiency(target, target)
  expect_equal(as.numeric(as_r(env$tower_sufficiency_pending[[1L]])), 0)
  env$tower_sufficiency_pending <- list()
  strategize:::neural_collect_tower_sufficiency(target, target + 1, weights = c(1, 1, 0, 0))
  expect_equal(as.numeric(as_r(env$tower_sufficiency_pending[[1L]])), 1, tolerance = 1e-6)

  # Distillation must not pull the joint utility: zero gradient w.r.t. the target.
  loss_of <- reticulate::py_func(function(t, s) {
    env$tower_sufficiency_pending <- list()
    strategize:::neural_collect_tower_sufficiency(t, s)
    env$tower_sufficiency_pending[[1L]]
  })
  surrogate <- jnp$array(matrix(c(1, 1, 1, 1), 4L, 1L), dtype = jnp$float32)
  grads <- env$jax$grad(loss_of, argnums = reticulate::tuple(0L, 1L))(target, surrogate)
  expect_equal(max(abs(as_r(grads[[1L]]))), 0)
  expect_gt(max(abs(as_r(grads[[2L]]))), 0)
})

test_that("tower-only mode resolves and routes utility through the towers", {
  resolve <- strategize:::neural_resolve_tower_sufficiency
  expect_identical(resolve(list(mode = "tower_only"), 16L), list(enabled = TRUE, weight = 0, mode = "tower_only"))
  expect_error(resolve(list(mode = "tower_only"), 0L), "low_rank_interaction_rank > 0")
  expect_error(resolve(list(mode = "joint"), 16L), "'distill' or 'tower_only'")
  combine <- strategize:::neural_combine_tower_term
  expect_identical(combine(1, 2, list()), 3)
  expect_identical(combine(1, 2, list(W_ts_cand = 1)), 3)
  expect_identical(combine(1, 2, list(W_to_cand = 1)), 2)
  expect_true(strategize:::neural_has_tower_sufficiency(list(W_to_cand = 1)))
  expect_identical(strategize:::neural_tower_candidate_head(list(W_to_cand = "to", W_ts_cand = "ts")), "to")
})
