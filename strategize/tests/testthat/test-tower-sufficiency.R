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

test_that("tower-only soft policy paths reproduce the hard tower prediction", {
  skip_on_cran()
  skip_if_no_jax()
  withr::local_envvar(c(STRATEGIZE_NEURAL_SKIP_EVAL = "1"))
  set.seed(7)
  n_pairs <- 24L
  W <- data.frame(
    feature = sample(c("A", "B", "C"), 2L * n_pairs, replace = TRUE),
    other = sample(c("x", "y"), 2L * n_pairs, replace = TRUE)
  )
  left_wins <- as.integer(W$feature[c(TRUE, FALSE)] == "A")
  Y <- as.numeric(rbind(left_wins, 1L - left_wins))
  names_list <- strategize:::cs2step_build_names_list(W)
  W_idx <- strategize:::cs2step_encode_W_indices(W, names_list = names_list, unknown = "error")
  factor_levels <- vapply(names_list, function(x) length(x[[1L]]), integer(1))
  fit <- suppressWarnings(strategize:::cs2step_eval_outcome_model_neural(
    Y = Y, W_idx = W_idx, names_list = names_list, factor_levels = factor_levels,
    diff = TRUE, pair_id = rep(seq_len(n_pairs), each = 2L), profile_order = rep(1:2, n_pairs),
    neural_mcmc_control = list(
      ModelDims = 8L, ModelDepth = 1L, subsample_method = "batch_vi",
      uncertainty_scope = "none", transformer_ffn = "swiglu",
      low_rank_interaction_rank = 2L, tower_sufficiency = list(mode = "tower_only"),
      additive_utility = "off", svi_steps = 8L, batch_size = 8L, early_stopping = FALSE,
      eval_enabled = FALSE, gradient_diagnostics = FALSE
    )
  ))
  info <- fit$neural_model_info
  expect_true(strategize:::neural_tower_only(info$params))
  env <- strategize:::strenv
  jnp <- env$jnp
  as_r <- function(x) as.numeric(reticulate::py_to_r(env$np$asarray(x)))
  party <- strategize:::neural_model_party_missing_index(info)
  resp_party <- strategize:::neural_model_resp_party_missing_index(info)
  # Implicit layout: each factor's last level is the remainder, so (1, 0 | 1) is
  # profile (A, x) and (0, 0 | 0) is (C, y). At these one-hot policies the soft
  # relaxation must equal the hard prediction; before the fix it read the zero
  # W_out head and returned Q = 0.5 whatever the profiles.
  pi_left <- jnp$array(c(1, 0, 1), dtype = env$dtj)
  pi_right <- jnp$array(c(0, 0, 0), dtype = env$dtj)
  hard <- as_r(strategize:::neural_predict_pair_core_prepared(
    params = info$params, model_info = info,
    Xl = matrix(c(0L, 0L), 1L), Xr = matrix(c(2L, 1L), 1L),
    pl = party, pr = party, resp_p = resp_party
  ))
  soft <- as_r(strategize:::neural_predict_pair_soft(
    pi_left, pi_right, party, party, resp_party, info
  ))
  expect_gt(abs(hard - 0.5), 1e-6)
  expect_equal(soft, hard, tolerance = 1e-5)

  utility <- function(pi) strategize:::neural_candidate_utility_soft(
    pi, party, resp_party, stage_idx = NULL, model_info = info
  )
  expect_gt(abs(as_r(utility(pi_left)) - as_r(utility(pi_right))), 1e-6)
  grad <- env$jax$grad(reticulate::py_func(function(pi) jnp$sum(utility(pi))))(
    jnp$array(c(0.4, 0.3, 0.5), dtype = env$dtj)
  )
  expect_gt(max(abs(as_r(grad))), 0)
})
