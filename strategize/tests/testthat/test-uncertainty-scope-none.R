# uncertainty_scope = "none" (pure maximum likelihood) and the residual init
# policy (residual_gate_init, output_proj_init = "depth_scaled").

mle_param <- function(model_info, name, layer = 1L) {
  params <- model_info$params
  value <- params[[paste0(name, "_l", layer)]] %||% params[[name]]
  if (is.null(value) && !is.null(params[[paste0(name, "_layers")]])) {
    stacked <- strategize:::cs2step_neural_to_r_array(params[[paste0(name, "_layers")]])
    dims <- dim(stacked)
    if (is.null(dims) || length(dims) == 1L) {
      return(as.numeric(stacked)[layer])
    }
    return(asplit(stacked, 1L)[[layer]])
  }
  if (is.null(value)) {
    return(NULL)
  }
  strategize:::cs2step_neural_to_r_array(value)
}

mle_pairwise_data <- function(n_pairs = 16L, seed = 11L) {
  set.seed(seed)
  W <- data.frame(
    feature = sample(c("A", "B", "C"), 2L * n_pairs, replace = TRUE),
    other = sample(c("x", "y"), 2L * n_pairs, replace = TRUE)
  )
  left_wins <- stats::rbinom(n_pairs, 1L, 0.5)
  Y <- as.numeric(rbind(left_wins, 1L - left_wins))
  list(W = W, Y = Y, pair_id = rep(seq_len(n_pairs), each = 2L), profile_order = rep(1:2, n_pairs))
}

mle_fit <- function(data, control, diff = TRUE) {
  names_list <- strategize:::cs2step_build_names_list(data$W)
  W_idx <- strategize:::cs2step_encode_W_indices(data$W, names_list = names_list, unknown = "error")
  factor_levels <- vapply(names_list, function(x) length(x[[1L]]), integer(1))
  fit <- suppressWarnings(strategize:::cs2step_eval_outcome_model_neural(
    Y = data$Y, W_idx = W_idx, names_list = names_list, factor_levels = factor_levels,
    diff = diff,
    pair_id = if (isTRUE(diff)) data$pair_id else NULL,
    profile_order = if (isTRUE(diff)) data$profile_order else NULL,
    neural_mcmc_control = control
  ))
  list(fit = fit, names_list = names_list, factor_levels = factor_levels)
}

test_that("uncertainty_scope = 'none' fits point estimates with zero theta_var", {
  skip_on_cran()
  skip_if_no_jax()
  withr::local_envvar(c(STRATEGIZE_NEURAL_SKIP_EVAL = "1"))
  data <- mle_pairwise_data()
  out <- mle_fit(data, list(
    ModelDims = 8L, ModelDepth = 1L, subsample_method = "batch_vi",
    uncertainty_scope = "none", transformer_ffn = "swiglu",
    svi_steps = 4L, svi_num_draws = 2L, batch_size = 4L,
    early_stopping = FALSE, eval_enabled = FALSE, gradient_diagnostics = FALSE
  ))
  fit <- out$fit
  info <- fit$neural_model_info
  expect_identical(info$uncertainty_scope, "none")
  diagnostics <- info$optimizer_diagnostics
  expect_identical(diagnostics$optimizer, "muon")
  expect_identical(diagnostics$muon_partition_status, "verified")
  expect_true(length(fit$theta_mean) > 0L)
  expect_true(all(is.finite(fit$theta_mean)))
  expect_identical(length(fit$theta_var), length(fit$theta_mean))
  expect_true(all(fit$theta_var == 0))

  # Bundle round trip: predictions are identical whether parameters are
  # preloaded or rebuilt from theta_mean, and interval draws are refused.
  tmp <- tempfile(fileext = ".rds")
  on.exit(unlink(tmp), add = TRUE)
  save_neural_outcome_bundle(
    file = tmp, theta_mean = fit$theta_mean, theta_var = fit$theta_var,
    neural_model_info = info, names_list = out$names_list,
    factor_levels = out$factor_levels, mode = "pairwise", overwrite = TRUE
  )
  left <- data$W[data$profile_order == 1L, , drop = FALSE]
  right <- data$W[data$profile_order == 2L, , drop = FALSE]
  lazy <- load_neural_outcome_bundle(tmp, preload_params = FALSE)
  eager <- load_neural_outcome_bundle(tmp, preload_params = TRUE)
  p_lazy <- predict_pair(lazy, W_left = left, W_right = right)
  p_eager <- predict_pair(eager, W_left = left, W_right = right)
  expect_true(all(is.finite(p_lazy)) && all(p_lazy >= 0 & p_lazy <= 1))
  expect_equal(p_lazy, p_eager, tolerance = 1e-6)
  expect_error(
    predict_pair(lazy, W_left = left, W_right = right, interval = "ci", n_draws = 5L),
    "maximum likelihood"
  )
})

test_that("pure MLE starts normal-outcome heads at data-informed values", {
  skip_on_cran()
  skip_if_no_jax()
  withr::local_envvar(c(STRATEGIZE_NEURAL_SKIP_EVAL = "true", STRATEGIZE_NEURAL_FAST_MCMC = "true"))
  # Same design as the average-case neural tests (24 rows, 3 binary factors),
  # shifted and scaled so the base-rate and MAD inits are distinguishable.
  withr::local_seed(20260326)
  W <- matrix(sample(LETTERS[1:2], 24L * 3L, replace = TRUE), nrow = 24L,
              dimnames = list(NULL, paste0("V", 1:3)))
  signal <- rowSums((W == "B") * rep(seq(0.5, 0.2, length.out = 3L), each = 24L))
  data <- list(W = W, Y = 3 + 2 * as.numeric(signal + stats::rnorm(24L, sd = 0.1)))
  params <- default_strategize_params(fast = TRUE)
  params$diff <- FALSE
  params$force_gaussian <- TRUE
  params$compute_se <- FALSE
  params$outcome_model_type <- "neural"
  params$nMonte_Qglm <- 4L
  params$neural_mcmc_control <- modifyList(params$neural_mcmc_control %||% list(), list(
    ModelDims = 8L, ModelDepth = 1L, subsample_method = "batch_vi",
    uncertainty_scope = "none", optimizer = "adam", transformer_ffn = "swiglu",
    svi_steps = 1L, svi_lr = 1e-9, batch_size = 8L, eval_enabled = FALSE,
    early_stopping = FALSE, gradient_diagnostics = FALSE,
    warn_stage_imbalance_pct = 0, warn_min_cell_n = 0L
  ))
  res <- suppressWarnings(do.call(strategize, c(
    list(Y = data$Y, W = data$W, p_list = generate_test_p_list(data$W)),
    params
  )))
  info <- res$neural_model_info$ast %||% res$neural_model_info$dag %||%
    res$neural_model_info$ast0 %||% res$neural_model_info$dag0
  expect_identical(info$uncertainty_scope, "none")
  expect_equal(as.numeric(mle_param(info, "b_out")), mean(data$Y), tolerance = 1e-4)
  expect_equal(as.numeric(mle_param(info, "sigma")), stats::mad(data$Y), tolerance = 1e-3)
  expect_lt(max(abs(mle_param(info, "W_out"))), 1e-4)
})

test_that("residual init policy starts gates at 1 and depth-scales residual writes", {
  skip_on_cran()
  skip_if_no_jax()
  withr::local_envvar(c(STRATEGIZE_NEURAL_SKIP_EVAL = "1"))
  data <- mle_pairwise_data(n_pairs = 24L)
  control <- list(
    ModelDims = 32L, ModelDepth = 2L, subsample_method = "batch_vi",
    uncertainty_scope = "none", transformer_ffn = "swiglu", optimizer = "adam",
    svi_steps = 1L, svi_lr = 1e-9, batch_size = 8L,
    early_stopping = FALSE, eval_enabled = FALSE, gradient_diagnostics = FALSE
  )
  legacy <- mle_fit(data, control)$fit$neural_model_info
  scaled <- mle_fit(data, modifyList(control, list(
    residual_gate_init = 1, output_proj_init = "depth_scaled"
  )))$fit$neural_model_info

  expect_equal(scaled$init_policy$gate_init_value, 1)
  expect_equal(scaled$init_policy$output_proj_multiplier, 1 / sqrt(2 * 2))
  for (gate in c("alpha_attn", "alpha_ff")) {
    expect_equal(as.numeric(mle_param(scaled, gate)), 1, tolerance = 1e-4)
    expect_equal(as.numeric(mle_param(legacy, gate)), legacy$init_policy$gate_sd_scale,
                 tolerance = 1e-4)
  }
  # At depth 2 the regular sd is residual_weight_sd_scale = sqrt(2/32); the
  # depth-scaled projections start at half of it (W_ff2 also carries its fan scale).
  sd_ratio <- function(name) stats::sd(as.numeric(mle_param(scaled, name))) /
    stats::sd(as.numeric(mle_param(legacy, name)))
  expect_equal(sd_ratio("W_o"), 0.5, tolerance = 0.02)
  expect_equal(sd_ratio("W_ff2"), 0.5, tolerance = 0.02)
  expect_equal(sd_ratio("W_q"), 1, tolerance = 0.02)
})
