# =============================================================================
# Neural Outcome Model Tests
# =============================================================================

get_neural_fit <- local({
  cache <- NULL
  function() {
    if (!is.null(cache)) {
      return(cache)
    }

    skip_on_cran()
    skip_if_no_jax()

    withr::local_envvar(c(
      STRATEGIZE_NEURAL_FAST_MCMC = "true",
      STRATEGIZE_NEURAL_EVAL_FOLDS = "2",
      STRATEGIZE_NEURAL_EVAL_SEED = "123"
    ))

    data <- generate_test_data(n = 40, seed = 123)
    params <- default_strategize_params(fast = TRUE)
    params$outcome_model_type <- "neural"
    base_neural_control <- params$neural_mcmc_control
    if (is.null(base_neural_control)) {
      base_neural_control <- list()
    }
    params$neural_mcmc_control <- modifyList(
      base_neural_control,
      list(
        subsample_method = "batch_vi",
        batch_size = 16L
      )
    )

    p_list <- generate_test_p_list(data$W)

    res <- do.call(strategize, c(
      list(Y = data$Y, W = data$W, p_list = p_list),
      data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order")],
      params
    ))

    cache <<- list(res = res, data = data, p_list = p_list)
    cache
  }
})

get_neural_fit_perf <- local({
  cache <- NULL
  function() {
    if (!is.null(cache)) {
      return(cache)
    }

    skip_on_cran()
    skip_if_no_jax()

    withr::local_envvar(c(
      STRATEGIZE_NEURAL_FAST_MCMC = "true",
      STRATEGIZE_NEURAL_EVAL_FOLDS = "2",
      STRATEGIZE_NEURAL_EVAL_SEED = "123"
    ))

    data <- generate_pairwise_performance_test_data(
      n_pairs = 1000L,
      n_factors = 3,
      n_levels = 2,
      seed = 20260327
    )
    data <- add_adversarial_structure(data, seed = 20260328)
    params <- default_strategize_params(fast = TRUE)
    params$outcome_model_type <- "neural"
    base_neural_control <- params$neural_mcmc_control
    if (is.null(base_neural_control)) {
      base_neural_control <- list()
    }
    params$neural_mcmc_control <- modifyList(
      base_neural_control,
      list(
        subsample_method = "batch_vi",
        batch_size = 128L,
        ModelDims = 16L,
        ModelDepth = 1L
      )
    )

    p_list <- generate_test_p_list(data$W)

    res <- do.call(strategize, c(
      list(Y = data$Y, W = data$W, p_list = p_list),
      data[c(
        "pair_id",
        "respondent_id",
        "respondent_task_id",
        "profile_order",
        "competing_group_variable_respondent",
        "competing_group_variable_candidate",
        "competing_group_competition_variable_candidate"
      )],
      params
    ))

    cache <<- list(res = res, data = data, p_list = p_list)
    cache
  }
})

get_neural_fit_attn <- local({
  cache <- NULL
  function() {
    if (!is.null(cache)) {
      return(cache)
    }

    skip_on_cran()
    skip_if_no_jax()

    withr::local_envvar(c(
      STRATEGIZE_NEURAL_FAST_MCMC = "true",
      STRATEGIZE_NEURAL_EVAL_FOLDS = "2",
      STRATEGIZE_NEURAL_EVAL_SEED = "123"
    ))

    data <- generate_test_data(n = 30, seed = 321)
    params <- default_strategize_params(fast = TRUE)
    params$outcome_model_type <- "neural"
    base_neural_control <- params$neural_mcmc_control
    if (is.null(base_neural_control)) {
      base_neural_control <- list()
    }
    params$neural_mcmc_control <- modifyList(
      base_neural_control,
      list(
        subsample_method = "batch_vi",
        batch_size = 16L,
        cross_candidate_encoder = "attn",
        ModelDims = 16L,
        ModelDepth = 1L
      )
    )

    p_list <- generate_test_p_list(data$W)

    res <- do.call(strategize, c(
      list(Y = data$Y, W = data$W, p_list = p_list),
      data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order")],
      params
    ))

    cache <<- list(res = res, data = data, p_list = p_list)
    cache
  }
})

get_neural_fit_attn_output_vi <- local({
  cache <- NULL
  function() {
    if (!is.null(cache)) {
      return(cache)
    }

    skip_on_cran()
    skip_if_no_jax()

    withr::local_envvar(c(
      STRATEGIZE_NEURAL_FAST_MCMC = "true"
    ))

    data <- generate_test_data(n = 24, seed = 20260326)
    params <- default_strategize_params(fast = TRUE)
    params$outcome_model_type <- "neural"
    params$neural_mcmc_control <- modifyList(
      params$neural_mcmc_control,
      list(
        cross_candidate_encoder = "attn",
        ModelDims = 16L,
        ModelDepth = 1L,
        subsample_method = "batch_vi",
        uncertainty_scope = "output",
        svi_steps = "optimal",
        batch_size = 16L,
        eval_enabled = FALSE,
        warn_stage_imbalance_pct = 0,
        warn_min_cell_n = 0L
      )
    )

    p_list <- generate_test_p_list(data$W)

    res <- suppressWarnings(do.call(strategize, c(
      list(Y = data$Y, W = data$W, p_list = p_list),
      data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order")],
      params
    )))

    cache <<- list(res = res, data = data, p_list = p_list)
    cache
  }
})

get_neural_model_info <- function(res) {
  model_info <- res$neural_model_info$ast
  if (is.null(model_info)) model_info <- res$neural_model_info$dag
  if (is.null(model_info)) model_info <- res$neural_model_info$ast0
  if (is.null(model_info)) model_info <- res$neural_model_info$dag0
  model_info
}

compute_binary_null_metrics <- function(y) {
  y <- as.numeric(y)
  y <- y[is.finite(y)]
  p_null <- mean(y)
  p_null <- min(max(p_null, 1e-6), 1 - 1e-6)
  list(
    log_loss = -mean(y * log(p_null) + (1 - y) * log(1 - p_null)),
    accuracy = max(mean(y), 1 - mean(y)),
    brier = mean((p_null - y) ^ 2)
  )
}

local_strenv_bindings <- function(bindings) {
  binding_names <- as.character(bindings)
  had_binding <- setNames(logical(length(binding_names)), binding_names)
  old_values <- setNames(vector("list", length(binding_names)), binding_names)

  for (binding in binding_names) {
    had_binding[[binding]] <- exists(binding, envir = strategize:::strenv, inherits = FALSE)
    if (had_binding[[binding]]) {
      old_values[[binding]] <- get(binding, envir = strategize:::strenv, inherits = FALSE)
    }
  }

  withr::defer({
    for (binding in binding_names) {
      if (had_binding[[binding]]) {
        assign(binding, old_values[[binding]], envir = strategize:::strenv)
      } else if (exists(binding, envir = strategize:::strenv, inherits = FALSE)) {
        rm(list = binding, envir = strategize:::strenv)
      }
    }
  })
}

# pkgload::load_all() can error on strategize:::strenv$foo <- value and surface
# loadNamespace("*tmp*"), so use assign() for test-time strenv mutations.
set_strenv_bindings <- function(values) {
  for (binding in names(values)) {
    assign(binding, values[[binding]], envir = strategize:::strenv)
  }
}

generate_average_case_neural_data <- function(n = 24, n_factors = 3, seed = 20260326) {
  withr::local_seed(seed)

  levels <- LETTERS[1:2]
  W <- matrix(
    sample(levels, n * n_factors, replace = TRUE),
    nrow = n,
    ncol = n_factors
  )
  colnames(W) <- paste0("V", seq_len(n_factors))

  effect_sizes <- seq(0.5, 0.2, length.out = n_factors)
  signal <- rowSums((W == "B") * rep(effect_sizes, each = n))
  Y <- as.numeric(signal + rnorm(n, sd = 0.1))

  list(Y = Y, W = W)
}

run_average_case_neural_fit <- function(vi_guide = "auto_diagonal",
                                        compute_se = FALSE,
                                        nMonte_Qglm = 4L,
                                        svi_steps = 20L,
                                        seed = 20260326) {
  skip_on_cran()
  skip_if_no_jax()

  withr::local_envvar(c(
    STRATEGIZE_NEURAL_FAST_MCMC = "true",
    STRATEGIZE_NEURAL_SKIP_EVAL = "true"
  ))

  data <- generate_average_case_neural_data(seed = seed)
  params <- default_strategize_params(fast = TRUE)
  params$diff <- FALSE
  params$force_gaussian <- TRUE
  params$compute_se <- compute_se
  params$outcome_model_type <- "neural"
  params$nMonte_Qglm <- as.integer(nMonte_Qglm)
  base_neural_control <- params$neural_mcmc_control
  if (is.null(base_neural_control)) {
    base_neural_control <- list()
  }
  params$neural_mcmc_control <- modifyList(
    base_neural_control,
    list(
      subsample_method = "batch_vi",
      uncertainty_scope = "output",
      vi_guide = vi_guide,
      ModelDims = 8L,
      ModelDepth = 1L,
      batch_size = 16L,
      svi_steps = as.integer(svi_steps),
      svi_num_draws = 5L,
      eval_enabled = FALSE,
      warn_stage_imbalance_pct = 0,
      warn_min_cell_n = 0L
    )
  )

  p_list <- generate_test_p_list(data$W)
  res <- suppressWarnings(do.call(strategize, c(
    list(Y = data$Y, W = data$W, p_list = p_list),
    params
  )))

  list(res = res, data = data, p_list = p_list)
}

test_that("neural output site init values target scalar normal regression only", {
  init <- strategize:::neural_build_output_site_init_values(
    Y = c(1, 3, 5, NA_real_),
    likelihood = "normal",
    nOutcomes = 1L
  )
  expect_equal(init$b_out, mean(c(1, 3, 5)))
  expect_equal(init$tau_b, abs(mean(c(1, 3, 5))))
  expect_equal(init$sigma, stats::mad(c(1, 3, 5)))

  constant_init <- strategize:::neural_build_output_site_init_values(
    Y = rep(2, 4),
    likelihood = "normal",
    nOutcomes = 1L
  )
  expect_equal(constant_init$b_out, 2)
  expect_equal(constant_init$tau_b, 2)
  expect_true(is.finite(constant_init$sigma))
  expect_gt(constant_init$sigma, 0)

  latent_init <- strategize:::neural_build_output_site_init_values(
    Y = c(-2, -2, -2),
    likelihood = "normal",
    nOutcomes = 1L,
    b_out_site_name = "b_out_z"
  )
  expect_equal(latent_init$tau_b, 2)
  expect_equal(latent_init$b_out_z * latent_init$tau_b, -2)
  expect_true(is.finite(latent_init$sigma))
  expect_gt(latent_init$sigma, 0)

  expect_equal(
    strategize:::neural_build_output_site_init_values(
      Y = c(0, 1),
      likelihood = "bernoulli",
      nOutcomes = 1L
    ),
    list()
  )
  expect_equal(
    strategize:::neural_build_output_site_init_values(
      Y = c(0, 1, 2),
      likelihood = "categorical",
      nOutcomes = 3L
    ),
    list()
  )
  expect_equal(
    strategize:::neural_build_output_site_init_values(
      Y = c(0, 1, 2),
      likelihood = "normal",
      nOutcomes = 2L
    ),
    list()
  )
})

test_that("strategize runs neural outcome model (non-adversarial)", {
  fit <- get_neural_fit()
  res <- fit$res
  data <- fit$data
  p_list <- fit$p_list

  expect_valid_strategize_output(res, n_factors = ncol(data$W))

  model <- res$Y_models$my_model_ast_jnp
  if (is.null(model)) {
    model <- res$Y_models$my_model_dag_jnp
  }
  expect_true(is.function(model))

  W_numeric <- as.matrix(sapply(seq_len(ncol(data$W)), function(d_) {
    match(data$W[, d_], names(p_list[[d_]]))
  }))
  idx_left <- which(data$profile_order == 1L)
  idx_right <- which(data$profile_order == 2L)
  X_left <- W_numeric[idx_left, , drop = FALSE]
  X_right <- W_numeric[idx_right, , drop = FALSE]
  p_lr <- as.numeric(model(X_left_new = X_left, X_right_new = X_right))
  p_rl <- as.numeric(model(X_left_new = X_right, X_right_new = X_left))
  expect_equal(p_lr + p_rl, rep(1, length(p_lr)), tolerance = 1e-4)
})

test_that("neural attn predictor remains antisymmetric", {
  fit <- get_neural_fit_attn()
  res <- fit$res
  data <- fit$data
  p_list <- fit$p_list

  expect_valid_strategize_output(res, n_factors = ncol(data$W))

  model <- res$Y_models$my_model_ast_jnp
  if (is.null(model)) {
    model <- res$Y_models$my_model_dag_jnp
  }
  expect_true(is.function(model))

  W_numeric <- as.matrix(sapply(seq_len(ncol(data$W)), function(d_) {
    match(data$W[, d_], names(p_list[[d_]]))
  }))
  idx_left <- which(data$profile_order == 1L)
  idx_right <- which(data$profile_order == 2L)
  X_left <- W_numeric[idx_left, , drop = FALSE]
  X_right <- W_numeric[idx_right, , drop = FALSE]
  p_lr <- as.numeric(model(X_left_new = X_left, X_right_new = X_right))
  p_rl <- as.numeric(model(X_left_new = X_right, X_right_new = X_left))
  expect_equal(p_lr + p_rl, rep(1, length(p_lr)), tolerance = 1e-4)
})

test_that("neural attn metadata marks the cross-candidate encoder as enabled", {
  fit <- get_neural_fit_attn()
  model_info <- get_neural_model_info(fit$res)

  expect_false(is.null(model_info))
  expect_identical(model_info$cross_candidate_encoder_mode, "attn")
  expect_true(isTRUE(model_info$cross_candidate_encoder))
  expect_true(isTRUE(model_info$has_qk_norm))
})

test_that("neural outcome bundles save and reload cleanly", {
  fit <- get_neural_fit_attn()
  res <- fit$res
  data <- fit$data
  p_list <- fit$p_list

  model_info <- get_neural_model_info(res)
  expect_true(!is.null(model_info))
  expect_identical(model_info$cross_candidate_encoder_mode, "attn")
  expect_true(isTRUE(model_info$cross_candidate_encoder))

  theta_mean <- tryCatch(
    as.numeric(reticulate::py_to_r(res$est_coefficients_jnp)),
    error = function(e) {
      tryCatch(
        as.numeric(res$est_coefficients_jnp),
        error = function(e2) {
          as.numeric(reticulate::py_to_r(strategize:::strenv$np$array(res$est_coefficients_jnp)))
        }
      )
    }
  )
  expect_true(is.numeric(theta_mean))

  vcov_vec <- res$vcov_outcome_model
  theta_var <- if (!is.null(vcov_vec) && length(vcov_vec) > 1L) {
    as.numeric(vcov_vec[-1])
  } else {
    NULL
  }

  tmp <- tempfile(fileext = ".rds")
  save_neural_outcome_bundle(
    file = tmp,
    theta_mean = theta_mean,
    theta_var = theta_var,
    neural_model_info = model_info,
    p_list = p_list,
    mode = "pairwise",
    overwrite = TRUE
  )

  bundle <- readRDS(tmp)
  has_py_object <- function(x) {
    if (reticulate::is_py_object(x)) {
      return(TRUE)
    }
    if (is.list(x)) {
      return(any(vapply(x, has_py_object, logical(1))))
    }
    FALSE
  }
  expect_false(has_py_object(bundle))
  expect_false(has_py_object(bundle$fit$neural_model_info))
  expect_identical(bundle$fit$neural_model_info$cross_candidate_encoder_mode, "attn")
  expect_true(isTRUE(bundle$fit$neural_model_info$cross_candidate_encoder))
  expect_true(isTRUE(bundle$fit$neural_model_info$has_qk_norm))

  fit_loaded <- load_neural_outcome_bundle(tmp, preload_params = FALSE)
  expect_true(inherits(fit_loaded, "strategic_predictor"))
  expect_true(is.null(fit_loaded$fit$params))
  expect_identical(fit_loaded$fit$neural_model_info$cross_candidate_encoder_mode, "attn")
  expect_true(isTRUE(fit_loaded$fit$neural_model_info$cross_candidate_encoder))
  expect_true(isTRUE(fit_loaded$fit$neural_model_info$has_qk_norm))

  idx_left <- which(data$profile_order == 1L)
  idx_right <- which(data$profile_order == 2L)
  W_left <- data$W[idx_left, , drop = FALSE]
  W_right <- data$W[idx_right, , drop = FALSE]
  p <- predict_pair(fit_loaded, W_left = W_left, W_right = W_right)
  expect_true(is.numeric(p))
  expect_true(all(is.finite(p)))
  expect_true(all(p >= 0 & p <= 1))
})

test_that("output-only optimal SVI uses the pairwise batch_vi heuristic path", {
  fit <- get_neural_fit_attn_output_vi()
  model_info <- get_neural_model_info(fit$res)

  expect_false(is.null(model_info))
  expect_true(isTRUE(model_info$pairwise_mode))
  expect_identical(model_info$uncertainty_scope, "output")
  expect_identical(model_info$cross_candidate_encoder_mode, "attn")
  expect_true(isTRUE(model_info$cross_candidate_encoder))
  expect_true(isTRUE(model_info$has_qk_norm))
  expect_true(is.numeric(model_info$svi_loss_curve))
  expect_gt(length(model_info$svi_loss_curve), 0L)

  expected_steps <- strategize:::neural_optimal_svi_steps(
    n_obs = length(unique(fit$data$pair_id)),
    n_factors = as.integer(model_info$n_factors),
    factor_levels = as.integer(model_info$factor_levels),
    model_dims = as.integer(model_info$model_dims),
    model_depth = as.integer(model_info$model_depth),
    n_party_levels = as.integer(model_info$n_party_levels),
    n_resp_party_levels = length(model_info$resp_party_levels),
    n_resp_covariates = as.integer(model_info$n_resp_covariates),
    n_outcomes = 1L,
    pairwise_mode = isTRUE(model_info$pairwise_mode),
    use_matchup_token = isTRUE(model_info$has_matchup_token),
    use_cross_encoder = identical(model_info$cross_candidate_encoder_mode, "full"),
    use_cross_term = identical(model_info$cross_candidate_encoder_mode, "term"),
    use_cross_attn = identical(model_info$cross_candidate_encoder_mode, "attn"),
    use_qk_norm = isTRUE(model_info$has_qk_norm),
    batch_size = 16L,
    subsample_method = "batch_vi"
  )

  expect_length(model_info$svi_loss_curve, expected_steps)
  expect_identical(as.integer(model_info$svi_steps), as.integer(expected_steps))
})

test_that("SVI ELBO plot title reports the rounded last-20 finite mean", {
  expect_identical(
    strategize:::neural_format_svi_elbo_plot_title(seq_len(25)),
    "SVI ELBO Loss [15.5000]"
  )

  expect_identical(
    strategize:::neural_format_svi_elbo_plot_title(c(Inf, NA_real_, 2, 4, 6)),
    "SVI ELBO Loss [4.0000]"
  )

  expect_identical(
    strategize:::neural_format_svi_elbo_plot_title(c(NA_real_, Inf, -Inf)),
    "SVI ELBO Loss"
  )
})

test_that("output-only single-model normal batch_vi applies the production SVI floor by default", {
  budget <- strategize:::neural_resolve_svi_budget(
    svi_steps_input = "optimal",
    svi_num_draws_input = 100L,
    user_supplied_svi_steps = FALSE,
    user_supplied_svi_num_draws = FALSE,
    n_obs = 2000L,
    n_factors = 10L,
    factor_levels = rep(2L, 10L),
    model_dims = 64L,
    model_depth = 2L,
    n_party_levels = 1L,
    n_resp_party_levels = 1L,
    n_resp_covariates = 0L,
    n_outcomes = 1L,
    pairwise_mode = FALSE,
    use_matchup_token = FALSE,
    use_cross_encoder = FALSE,
    use_cross_term = FALSE,
    use_cross_attn = FALSE,
    use_qk_norm = FALSE,
    batch_size = 512L,
    subsample_method = "batch_vi",
    output_only_mode = TRUE,
    likelihood = "normal"
  )

  expect_true(isTRUE(budget$used_optimal))
  expect_true(isTRUE(budget$used_default_svi_steps))
  expect_true(isTRUE(budget$used_default_svi_num_draws))
  expect_true(isTRUE(budget$output_single_normal_batch_vi))
  expect_true(isTRUE(budget$applied_output_single_normal_batch_vi_floor))
  expect_true(isTRUE(budget$applied_output_single_normal_batch_vi_draw_floor))
  expect_identical(as.integer(budget$svi_steps), 2000L)
  expect_identical(as.integer(budget$svi_num_draws), 200L)
})

test_that("explicit SVI overrides are honored even when below the production floor", {
  budget <- strategize:::neural_resolve_svi_budget(
    svi_steps_input = 200L,
    svi_num_draws_input = 100L,
    user_supplied_svi_steps = TRUE,
    user_supplied_svi_num_draws = TRUE,
    n_obs = 2000L,
    n_factors = 10L,
    factor_levels = rep(2L, 10L),
    model_dims = 64L,
    model_depth = 2L,
    n_party_levels = 1L,
    n_resp_party_levels = 1L,
    n_resp_covariates = 0L,
    n_outcomes = 1L,
    pairwise_mode = FALSE,
    use_matchup_token = FALSE,
    use_cross_encoder = FALSE,
    use_cross_term = FALSE,
    use_cross_attn = FALSE,
    use_qk_norm = FALSE,
    batch_size = 512L,
    subsample_method = "batch_vi",
    output_only_mode = TRUE,
    likelihood = "normal"
  )

  expect_false(isTRUE(budget$applied_output_single_normal_batch_vi_floor))
  expect_false(isTRUE(budget$applied_output_single_normal_batch_vi_draw_floor))
  expect_identical(as.integer(budget$svi_steps), 200L)
  expect_identical(as.integer(budget$svi_num_draws), 100L)
})

test_that("neural prior predictive probabilities are not overly concentrated", {
  fit <- get_neural_fit()
  res <- fit$res
  data <- fit$data
  p_list <- fit$p_list

  model <- res$Y_models$my_model_ast_jnp
  if (is.null(model)) {
    model <- res$Y_models$my_model_dag_jnp
  }
  expect_true(is.function(model))

  model_env <- environment(model)
  strenv <- get("strenv", envir = model_env)
  likelihood <- get("likelihood", envir = model_env)
  if (!likelihood %in% c("bernoulli", "categorical")) {
    skip("Prior predictive check supports bernoulli/categorical only.")
  }
  pairwise_mode <- isTRUE(get("pairwise_mode", envir = model_env))
  model_fn <- if (pairwise_mode) {
    get("BayesianPairTransformerModel", envir = model_env)
  } else {
    get("BayesianSingleTransformerModel", envir = model_env)
  }

  W_numeric <- as.matrix(sapply(seq_len(ncol(data$W)), function(d_) {
    match(data$W[, d_], names(p_list[[d_]]))
  }))
  to_index_matrix <- if (exists("to_index_matrix", envir = model_env, inherits = TRUE)) {
    get("to_index_matrix", envir = model_env)
  } else {
    function(x_mat) {
      x_mat <- as.matrix(x_mat)
      if (anyNA(x_mat)) {
        x_mat[is.na(x_mat)] <- 1L
      }
      x_int <- matrix(as.integer(x_mat) - 1L, nrow = nrow(x_mat), ncol = ncol(x_mat))
      x_int[x_int < 0L] <- 0L
      x_int
    }
  }

  if (pairwise_mode) {
    idx_left <- which(data$profile_order == 1L)
    idx_right <- which(data$profile_order == 2L)
    X_left <- W_numeric[idx_left, , drop = FALSE]
    X_right <- W_numeric[idx_right, , drop = FALSE]
    n_obs <- nrow(X_left)
    X_left_jnp <- strenv$jnp$array(to_index_matrix(X_left))$astype(strenv$jnp$int32)
    X_right_jnp <- strenv$jnp$array(to_index_matrix(X_right))$astype(strenv$jnp$int32)
    party_left_jnp <- strenv$jnp$zeros(list(n_obs), dtype = strenv$jnp$int32)
    party_right_jnp <- strenv$jnp$zeros(list(n_obs), dtype = strenv$jnp$int32)
    resp_party_jnp <- strenv$jnp$zeros(list(n_obs), dtype = strenv$jnp$int32)
    resp_cov_jnp <- strenv$jnp$zeros(list(n_obs, 0L), dtype = strenv$jnp$float32)
  } else {
    n_obs <- nrow(W_numeric)
    X_single_jnp <- strenv$jnp$array(to_index_matrix(W_numeric))$astype(strenv$jnp$int32)
    party_single_jnp <- strenv$jnp$zeros(list(n_obs), dtype = strenv$jnp$int32)
    resp_party_jnp <- strenv$jnp$zeros(list(n_obs), dtype = strenv$jnp$int32)
    resp_cov_jnp <- strenv$jnp$zeros(list(n_obs, 0L), dtype = strenv$jnp$float32)
  }

  coerce_prob_numeric <- function(x) {
    if (is.null(x)) {
      return(numeric(0))
    }
    out <- tryCatch(
      reticulate::py_to_r(strenv$np$asarray(x)),
      error = function(e) NULL
    )
    if (is.null(out)) {
      out <- tryCatch(
        reticulate::py_to_r(strenv$np$array(x)),
        error = function(e) NULL
      )
    }
    if (is.null(out)) {
      out <- tryCatch(
        reticulate::py_to_r(strenv$jax$device_get(x)),
        error = function(e) NULL
      )
    }
    if (is.null(out)) {
      out <- tryCatch(reticulate::py_to_r(x), error = function(e) NULL)
    }
    if (is.null(out)) {
      return(numeric(0))
    }
    if (is.list(out)) {
      out <- unlist(out, use.names = FALSE)
    }
    if (!is.numeric(out)) {
      return(numeric(0))
    }
    as.numeric(out)
  }

  model_info <- NULL
  if (!is.null(res$neural_model_info)) {
    if (!is.null(res$Y_models$my_model_ast_jnp)) {
      model_info <- res$neural_model_info$ast
    } else if (!is.null(res$Y_models$my_model_dag_jnp)) {
      model_info <- res$neural_model_info$dag
    }
  }
  if (is.null(model_info)) {
    skip("Neural model info unavailable for prior predictive check.")
  }
  model_dims <- as.integer(model_info$model_dims)
  model_depth <- as.integer(model_info$model_depth)
  n_heads <- as.integer(model_info$n_heads)
  head_dim <- as.integer(model_info$head_dim)
  cross_candidate_encoder <- isTRUE(model_info$cross_candidate_encoder)
  n_factors <- ncol(W_numeric)
  n_resp_covariates <- if (!is.null(model_info$n_resp_covariates)) {
    as.integer(model_info$n_resp_covariates)
  } else {
    0L
  }

  get_trace_value <- function(trace, name) {
    site <- tryCatch(trace[[name]], error = function(e) NULL)
    if (is.null(site)) {
      return(NULL)
    }
    val <- tryCatch(site$value, error = function(e) NULL)
    if (is.null(val)) {
      val <- site
    }
    val
  }

  build_params_from_trace <- function(trace) {
    params <- list()
    for (d_ in seq_len(n_factors)) {
      name <- paste0("E_factor_", d_)
      val <- get_trace_value(trace, name)
      if (is.null(val)) {
        raw <- get_trace_value(trace, paste0(name, "_raw"))
        if (!is.null(raw)) {
          n_real <- if (!is.null(model_info$factor_levels)) {
            as.integer(model_info$factor_levels[[d_]])
          } else {
            NA_integer_
          }
          n_raw <- tryCatch(
            as.integer(reticulate::py_to_r(raw$shape[[1]])),
            error = function(e) NA_integer_
          )
          if (!is.na(n_real) && !is.na(n_raw) && n_raw > n_real) {
            real_idx <- strenv$jnp$arange(as.integer(n_real))
            real_rows <- strenv$jnp$take(raw, real_idx, axis = 0L)
            real_mean <- strenv$jnp$mean(real_rows, axis = 0L, keepdims = TRUE)
            real_centered <- real_rows - real_mean
            missing_row <- strenv$jnp$take(raw, as.integer(n_real), axis = 0L)
            missing_row <- strenv$jnp$reshape(missing_row, list(1L, model_dims))
            val <- strenv$jnp$concatenate(list(real_centered, missing_row), axis = 0L)
          } else {
            val <- raw - strenv$jnp$mean(raw, axis = 0L, keepdims = TRUE)
          }
        }
      }
      params[[name]] <- val
    }
    base_names <- c(
      "E_feature_id", "E_party", "E_rel", "E_resp_party", "E_stage", "E_matchup", "E_choice",
      "E_sep", "E_segment",
      "W_resp_x", "RMS_final", "W_out", "b_out", "M_cross", "W_cross_out",
      "RMS_q_cross", "RMS_k_cross"
    )
    for (nm in base_names) {
      params[[nm]] <- get_trace_value(trace, nm)
    }
    for (l_ in seq_len(model_depth)) {
      params[[paste0("RMS_attn_l", l_)]] <- get_trace_value(trace, paste0("RMS_attn_l", l_))
      params[[paste0("RMS_q_l", l_)]] <- get_trace_value(trace, paste0("RMS_q_l", l_))
      params[[paste0("RMS_k_l", l_)]] <- get_trace_value(trace, paste0("RMS_k_l", l_))
      params[[paste0("RMS_ff_l", l_)]] <- get_trace_value(trace, paste0("RMS_ff_l", l_))
      params[[paste0("W_q_l", l_)]] <- get_trace_value(trace, paste0("W_q_l", l_))
      params[[paste0("W_k_l", l_)]] <- get_trace_value(trace, paste0("W_k_l", l_))
      params[[paste0("W_v_l", l_)]] <- get_trace_value(trace, paste0("W_v_l", l_))
      params[[paste0("W_o_l", l_)]] <- get_trace_value(trace, paste0("W_o_l", l_))
      params[[paste0("W_ff1_l", l_)]] <- get_trace_value(trace, paste0("W_ff1_l", l_))
      params[[paste0("W_ff2_l", l_)]] <- get_trace_value(trace, paste0("W_ff2_l", l_))
    }
    params
  }

  rms_norm <- function(x, g, eps = 1e-6) {
    if (is.null(g)) {
      return(x)
    }
    mean_sq <- strenv$jnp$mean(x * x, axis = -1L, keepdims = TRUE)
    inv_rms <- strenv$jnp$reciprocal(strenv$jnp$sqrt(mean_sq + eps))
    x * inv_rms * g
  }

  run_transformer <- function(tokens, params) {
    for (l_ in seq_len(model_depth)) {
      Wq <- params[[paste0("W_q_l", l_)]]
      Wk <- params[[paste0("W_k_l", l_)]]
      Wv <- params[[paste0("W_v_l", l_)]]
      Wo <- params[[paste0("W_o_l", l_)]]
      Wff1 <- params[[paste0("W_ff1_l", l_)]]
      Wff2 <- params[[paste0("W_ff2_l", l_)]]
      RMS_attn <- params[[paste0("RMS_attn_l", l_)]]
      RMS_q <- params[[paste0("RMS_q_l", l_)]]
      RMS_k <- params[[paste0("RMS_k_l", l_)]]
      RMS_ff <- params[[paste0("RMS_ff_l", l_)]]

      tokens_norm <- rms_norm(tokens, RMS_attn)
      Q <- strenv$jnp$einsum("ntm,mk->ntk", tokens_norm, Wq)
      K <- strenv$jnp$einsum("ntm,mk->ntk", tokens_norm, Wk)
      V <- strenv$jnp$einsum("ntm,mk->ntk", tokens_norm, Wv)

      Qh <- strenv$jnp$reshape(Q, list(Q$shape[[1]], Q$shape[[2]], n_heads, head_dim))
      Kh <- strenv$jnp$reshape(K, list(K$shape[[1]], K$shape[[2]], n_heads, head_dim))
      Vh <- strenv$jnp$reshape(V, list(V$shape[[1]], V$shape[[2]], n_heads, head_dim))
      Qh <- rms_norm(Qh, RMS_q)
      Kh <- rms_norm(Kh, RMS_k)
      scale_ <- strenv$jnp$sqrt(strenv$jnp$array(as.numeric(head_dim)))
      scores <- strenv$jnp$einsum("nqhd,nkhd->nhqk", Qh, Kh) / scale_
      attn <- strenv$jax$nn$softmax(scores, axis = -1L)
      context_h <- strenv$jnp$einsum("nhqk,nkhd->nqhd", attn, Vh)
      context <- strenv$jnp$reshape(context_h, list(context_h$shape[[1]], context_h$shape[[2]], model_dims))
      attn_out <- strenv$jnp$einsum("ntm,mk->ntk", context, Wo)

      h1 <- tokens + attn_out
      h1_norm <- rms_norm(h1, RMS_ff)
      ff_pre <- strenv$jnp$einsum("ntm,mf->ntf", h1_norm, Wff1)
      ff_act <- strenv$jax$nn$swish(ff_pre)
      ff_out <- strenv$jnp$einsum("ntf,fm->ntm", ff_act, Wff2)
      tokens <- h1 + ff_out
    }
    rms_norm(tokens, params$RMS_final)
  }

  embed_candidate <- function(X_idx, party_idx, resp_p, params) {
    N_batch <- as.integer(X_idx$shape[[1]])
    D_local <- as.integer(X_idx$shape[[2]])
    token_list <- vector("list", D_local)
    for (d_ in seq_len(D_local)) {
      E_d <- params[[paste0("E_factor_", d_)]]
      idx_d <- strenv$jnp$take(X_idx, as.integer(d_ - 1L), axis = 1L)
      token_list[[d_]] <- strenv$jnp$take(E_d, idx_d, axis = 0L)
    }
    tokens <- strenv$jnp$stack(token_list, axis = 1L)
    if (!is.null(params$E_feature_id)) {
      feature_tok <- strenv$jnp$reshape(params$E_feature_id, list(1L, D_local, model_dims))
      tokens <- tokens + feature_tok
    }
    party_tok <- strenv$jnp$take(params$E_party, party_idx, axis = 0L)
    party_tok <- strenv$jnp$reshape(party_tok, list(N_batch, 1L, model_dims))
    tokens <- strenv$jnp$concatenate(list(tokens, party_tok), axis = 1L)
    if (!is.null(params$E_rel) && !is.null(model_info$cand_party_to_resp_idx)) {
      cand_map <- strenv$jnp$atleast_1d(model_info$cand_party_to_resp_idx)
      cand_resp_idx <- strenv$jnp$take(cand_map, party_idx, axis = 0L)
      is_known <- cand_resp_idx >= 0L
      is_match <- strenv$jnp$equal(cand_resp_idx, resp_p)
      rel_idx <- strenv$jnp$where(is_match, as.integer(0L),
                                  strenv$jnp$where(is_known, as.integer(1L), as.integer(2L)))
      rel_idx <- strenv$jnp$astype(rel_idx, strenv$jnp$int32)
      rel_tok <- strenv$jnp$take(params$E_rel, rel_idx, axis = 0L)
      rel_tok <- strenv$jnp$reshape(rel_tok, list(N_batch, 1L, model_dims))
      tokens <- strenv$jnp$concatenate(list(tokens, rel_tok), axis = 1L)
    }
    tokens
  }

  build_context_tokens <- function(stage_idx, resp_p, resp_c, matchup_idx, params) {
    N_batch <- as.integer(resp_p$shape[[1]])
    token_list <- list()
    if (!is.null(params$E_stage) && !is.null(stage_idx)) {
      stage_tok <- params$E_stage[resp_p, stage_idx]
      stage_tok <- strenv$jnp$reshape(stage_tok, list(N_batch, 1L, model_dims))
      token_list[[length(token_list) + 1L]] <- stage_tok
    }
    if (!is.null(params$E_resp_party)) {
      resp_tok <- strenv$jnp$take(params$E_resp_party, resp_p, axis = 0L)
      resp_tok <- strenv$jnp$reshape(resp_tok, list(N_batch, 1L, model_dims))
      token_list[[length(token_list) + 1L]] <- resp_tok
    }
    if (!is.null(params$E_matchup) && !is.null(matchup_idx)) {
      matchup_tok <- strenv$jnp$take(params$E_matchup, matchup_idx, axis = 0L)
      matchup_tok <- strenv$jnp$reshape(matchup_tok, list(N_batch, 1L, model_dims))
      token_list[[length(token_list) + 1L]] <- matchup_tok
    }
    if (!is.null(params$W_resp_x) && n_resp_covariates > 0L) {
      resp_cov_tok <- strenv$jnp$einsum("nc,cm->nm", resp_c, params$W_resp_x)
      resp_cov_tok <- strenv$jnp$reshape(resp_cov_tok, list(N_batch, 1L, model_dims))
      token_list[[length(token_list) + 1L]] <- resp_cov_tok
    }
    if (length(token_list) == 0L) {
      return(NULL)
    }
    strenv$jnp$concatenate(token_list, axis = 1L)
  }

  encode_candidate <- function(X_idx, party_idx, resp_p, resp_c, stage_idx, matchup_idx, params) {
    N_batch <- as.integer(X_idx$shape[[1]])
    choice_vec <- if (!is.null(params$E_choice)) {
      params$E_choice
    } else {
      strenv$jnp$zeros(list(model_dims), dtype = strenv$dtj)
    }
    choice_tok <- strenv$jnp$reshape(choice_vec, list(1L, 1L, model_dims))
    choice_tok <- choice_tok * strenv$jnp$ones(list(N_batch, 1L, 1L))
    ctx_tokens <- build_context_tokens(stage_idx, resp_p, resp_c, matchup_idx, params)
    cand_tokens <- embed_candidate(X_idx, party_idx, resp_p, params)
    if (!is.null(ctx_tokens)) {
      tokens <- strenv$jnp$concatenate(list(choice_tok, ctx_tokens, cand_tokens), axis = 1L)
    } else {
      tokens <- strenv$jnp$concatenate(list(choice_tok, cand_tokens), axis = 1L)
    }
    tokens <- run_transformer(tokens, params)
    choice_out <- strenv$jnp$take(tokens, strenv$jnp$arange(1L), axis = 1L)
    strenv$jnp$squeeze(choice_out, axis = 1L)
  }

  prior_predictive_probs <- function(trace) {
    params <- build_params_from_trace(trace)
    if (pairwise_mode) {
      stage_idx <- strenv$jnp$equal(party_left_jnp, party_right_jnp)
      stage_idx <- strenv$jnp$astype(stage_idx, strenv$jnp$int32)
      matchup_idx <- NULL
      if (!is.null(params$E_matchup)) {
        n_party_levels <- if (!is.null(model_info$n_party_levels)) {
          as.integer(model_info$n_party_levels)
        } else if (!is.null(model_info$party_levels)) {
          length(model_info$party_levels)
        } else {
          1L
        }
        p_min <- strenv$jnp$minimum(party_left_jnp, party_right_jnp)
        p_max <- strenv$jnp$maximum(party_left_jnp, party_right_jnp)
        half_term <- strenv$jnp$floor_divide(p_min * (p_min - 1L), as.integer(2L))
        matchup_idx <- strenv$jnp$astype(
          p_min * as.integer(n_party_levels) - half_term + (p_max - p_min),
          strenv$jnp$int32
        )
      }
      phi_left <- encode_candidate(X_left_jnp, party_left_jnp, resp_party_jnp, resp_cov_jnp,
                                   stage_idx, matchup_idx, params)
      phi_right <- encode_candidate(X_right_jnp, party_right_jnp, resp_party_jnp, resp_cov_jnp,
                                    stage_idx, matchup_idx, params)
      u_left <- strenv$jnp$einsum("nm,mo->no", phi_left, params$W_out) + params$b_out
      u_right <- strenv$jnp$einsum("nm,mo->no", phi_right, params$W_out) + params$b_out
      logits <- u_left - u_right
      if (isTRUE(cross_candidate_encoder) && !is.null(params$M_cross) && !is.null(params$W_cross_out)) {
        cross_term <- strenv$jnp$einsum("nm,mp,np->n", phi_left, params$M_cross, phi_right)
        cross_term <- strenv$jnp$reshape(cross_term, list(-1L, 1L))
        cross_out <- strenv$jnp$reshape(params$W_cross_out, list(1L, -1L))
        logits <- logits + cross_term * cross_out
      }
    } else {
      phi_single <- encode_candidate(X_single_jnp, party_single_jnp, resp_party_jnp, resp_cov_jnp,
                                     stage_idx = NULL, matchup_idx = NULL, params = params)
      logits <- strenv$jnp$einsum("nm,mo->no", phi_single, params$W_out) + params$b_out
    }
    if (likelihood == "bernoulli") {
      logits_vec <- strenv$jnp$squeeze(logits, axis = 1L)
      return(strenv$jax$nn$sigmoid(logits_vec))
    }
    strenv$jax$nn$softmax(logits, axis = -1L)
  }

  n_draws <- 25L
  prob_samples <- numeric(0)
  for (i in seq_len(n_draws)) {
    rng_key <- strenv$jax$random$PRNGKey(as.integer(1000L + i))
    tracer <- strenv$numpyro$handlers$trace(
      strenv$numpyro$handlers$seed(model_fn, rng_key)
    )
    trace <- if (pairwise_mode) {
      tracer$get_trace(
        X_left = X_left_jnp,
        X_right = X_right_jnp,
        party_left = party_left_jnp,
        party_right = party_right_jnp,
        resp_party = resp_party_jnp,
        resp_cov = resp_cov_jnp,
        Y_obs = NULL
      )
    } else {
      tracer$get_trace(
        X = X_single_jnp,
        party = party_single_jnp,
        resp_party = resp_party_jnp,
        resp_cov = resp_cov_jnp,
        Y_obs = NULL
      )
    }
    prob <- prior_predictive_probs(trace)
    prob_samples <- c(
      prob_samples,
      coerce_prob_numeric(prob)
    )
  }

  prob_samples <- prob_samples[is.finite(prob_samples)]
  expect_true(length(prob_samples) > 0L)
  sd_prob <- stats::sd(prob_samples)
  expect_true(is.finite(sd_prob))
  expect_true(
    sd_prob >= 0.10,
    info = sprintf("Prior predictive SD %.3f below 0.10", sd_prob)
  )
})

test_that("non-pairwise neural prior trace handles NULL obs under minibatching", {
  skip_on_cran()
  skip_if_no_jax()

  withr::local_envvar(c(
    STRATEGIZE_NEURAL_FAST_MCMC = "true",
    STRATEGIZE_NEURAL_SKIP_EVAL = "true"
  ))

  data <- generate_test_data(n = 24, seed = 20260329)
  params <- default_strategize_params(fast = TRUE)
  params$diff <- FALSE
  params$outcome_model_type <- "neural"
  base_neural_control <- params$neural_mcmc_control
  if (is.null(base_neural_control)) {
    base_neural_control <- list()
  }
  params$neural_mcmc_control <- modifyList(
    base_neural_control,
    list(
      subsample_method = "batch_vi",
      batch_size = 16L,
      ModelDims = 16L,
      ModelDepth = 1L,
      eval_enabled = FALSE,
      warn_stage_imbalance_pct = 0,
      warn_min_cell_n = 0L
    )
  )

  p_list <- generate_test_p_list(data$W)
  res <- suppressWarnings(do.call(strategize, c(
    list(Y = data$Y, W = data$W, p_list = p_list),
    data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order")],
    params
  )))

  model <- res$Y_models$my_model_ast_jnp
  if (is.null(model)) {
    model <- res$Y_models$my_model_dag_jnp
  }
  expect_true(is.function(model))

  model_env <- environment(model)
  strenv <- get("strenv", envir = model_env)
  likelihood <- get("likelihood", envir = model_env)
  pairwise_mode <- isTRUE(get("pairwise_mode", envir = model_env))
  expect_identical(likelihood, "bernoulli")
  expect_false(pairwise_mode)

  model_fn <- get("BayesianSingleTransformerModel", envir = model_env)
  expect_true(is.function(model_fn))

  W_numeric <- as.matrix(sapply(seq_len(ncol(data$W)), function(d_) {
    match(data$W[, d_], names(p_list[[d_]]))
  }))
  to_index_matrix <- if (exists("to_index_matrix", envir = model_env, inherits = TRUE)) {
    get("to_index_matrix", envir = model_env)
  } else {
    function(x_mat) {
      x_mat <- as.matrix(x_mat)
      if (anyNA(x_mat)) {
        x_mat[is.na(x_mat)] <- 1L
      }
      x_int <- matrix(as.integer(x_mat) - 1L, nrow = nrow(x_mat), ncol = ncol(x_mat))
      x_int[x_int < 0L] <- 0L
      x_int
    }
  }

  coerce_prob_numeric <- function(x) {
    if (is.null(x)) {
      return(numeric(0))
    }
    out <- tryCatch(
      reticulate::py_to_r(strenv$np$asarray(x)),
      error = function(e) NULL
    )
    if (is.null(out)) {
      out <- tryCatch(reticulate::py_to_r(x), error = function(e) NULL)
    }
    if (is.null(out)) {
      return(numeric(0))
    }
    if (is.list(out)) {
      out <- unlist(out, use.names = FALSE)
    }
    if (!is.numeric(out)) {
      return(numeric(0))
    }
    as.numeric(out)
  }

  n_obs <- nrow(W_numeric)
  X_single_jnp <- strenv$jnp$array(to_index_matrix(W_numeric))$astype(strenv$jnp$int32)
  party_single_jnp <- strenv$jnp$zeros(list(n_obs), dtype = strenv$jnp$int32)
  resp_party_jnp <- strenv$jnp$zeros(list(n_obs), dtype = strenv$jnp$int32)
  resp_cov_jnp <- strenv$jnp$zeros(list(n_obs, 0L), dtype = strenv$jnp$float32)

  rng_key <- strenv$jax$random$PRNGKey(20260329L)
  tracer <- strenv$numpyro$handlers$trace(
    strenv$numpyro$handlers$seed(model_fn, rng_key)
  )
  trace <- tracer$get_trace(
    X = X_single_jnp,
    party = party_single_jnp,
    resp_party = resp_party_jnp,
    resp_cov = resp_cov_jnp,
    Y_obs = NULL
  )

  prob_obj <- tryCatch(trace$obs$fn$probs, error = function(e) NULL)
  if (is.null(prob_obj)) {
    logits_obj <- tryCatch(trace$obs$fn$logits, error = function(e) NULL)
    if (!is.null(logits_obj)) {
      prob_obj <- strenv$jax$nn$sigmoid(logits_obj)
    }
  }
  probs <- coerce_prob_numeric(prob_obj)

  expect_true(length(probs) > 0L)
  expect_true(all(is.finite(probs)))
  expect_true(all(probs >= 0 & probs <= 1))
})

test_that("neural outcome model exports cross-fitted OOS fit metrics", {
  fit <- get_neural_fit_perf()
  res <- fit$res

  info <- NULL
  if (!is.null(res$neural_model_info)) {
    info <- res$neural_model_info$ast
    if (is.null(info)) {
      info <- res$neural_model_info$dag
    }
  }
  if (is.null(info)) {
    skip("Neural model info unavailable for fit-metrics check.")
  }

  metrics <- info$fit_metrics
  diag_info <- format_oos_failure_details(
    metrics,
    stage_diagnostics = info$stage_diagnostics
  )
  expect_type(metrics, "list")
  expect_true(is.character(metrics$eval_note))
  expect_match(metrics$eval_note, "^oos_\\d+fold$")
  expect_equal(metrics$n_folds, 2L)
  expect_equal(metrics$seed, 123L)
  expect_true(!is.null(metrics$n_eval), info = diag_info)
  expect_true(is.numeric(metrics$n_eval) && length(metrics$n_eval) == 1L && !is.na(metrics$n_eval), info = diag_info)
  expect_true(metrics$n_eval >= 1000L, info = diag_info)
  expect_true(is.list(metrics$by_fold))
  expect_true(length(metrics$by_fold) >= 2L)
  expect_true(is.list(metrics$in_sample_metrics), info = diag_info)
  expect_false(isTRUE(info$stage_diagnostics$single_stage_only), info = diag_info)
  expect_false(isTRUE(info$stage_diagnostics$sparse_cells), info = diag_info)

  if (identical(metrics$likelihood, "bernoulli")) {
    expect_true(is.numeric(metrics$pred_quantiles), info = diag_info)
    expect_setequal(names(metrics$pred_quantiles), c("p05", "p25", "p50", "p75", "p95"))
    expect_setequal(names(metrics$confusion_0_5), c("tn", "fp", "fn", "tp"))
    expect_true(is.finite(metrics$log_loss), info = diag_info)
    expect_true(metrics$log_loss >= 0, info = diag_info)
  } else if (identical(metrics$likelihood, "categorical")) {
    expect_true(is.finite(metrics$log_loss), info = diag_info)
    expect_true(metrics$log_loss >= 0, info = diag_info)
  } else if (identical(metrics$likelihood, "normal")) {
    expect_true(is.finite(metrics$rmse), info = diag_info)
    expect_true(metrics$rmse >= 0, info = diag_info)
  }
})

test_that("neural pairwise OOS fit beats an intercept-only observable baseline", {
  fit <- get_neural_fit_perf()
  res <- fit$res
  data <- fit$data

  info <- get_neural_model_info(res)
  if (is.null(info) || is.null(info$fit_metrics)) {
    skip("Neural fit metrics unavailable for observable-fit check.")
  }

  metrics <- info$fit_metrics
  if (!identical(metrics$likelihood, "bernoulli")) {
    skip("Observable-fit baseline check currently supports bernoulli pairwise outcomes only.")
  }

  y_eval <- data$Y[data$profile_order == 1L]
  null_metrics <- compute_binary_null_metrics(y_eval)
  diag_info <- format_oos_failure_details(
    metrics,
    null_metrics = null_metrics,
    stage_diagnostics = info$stage_diagnostics,
    label = "Neural pairwise observable-baseline comparison"
  )

  expect_gte(length(y_eval), 1000L)
  expect_false(any(data$identical_pair))
  expect_true(all(data$pair_margin > 0))
  expect_false(isTRUE(info$stage_diagnostics$single_stage_only), info = diag_info)
  expect_false(isTRUE(info$stage_diagnostics$sparse_cells), info = diag_info)
  expect_equal(metrics$n_eval, length(y_eval))
  expect_true(is.finite(metrics$auc), info = diag_info)
  expect_true(metrics$auc > 0.5, info = diag_info)
  expect_true(metrics$log_loss < null_metrics$log_loss, info = diag_info)
  expect_true(metrics$brier < null_metrics$brier, info = diag_info)
})

test_that("non-pairwise AutoDelta runs on the average-case normal neural path", {
  fit <- run_average_case_neural_fit(
    vi_guide = "auto_delta",
    compute_se = FALSE,
    svi_steps = 5L
  )
  res <- fit$res
  model_info <- get_neural_model_info(res)

  expect_valid_strategize_output(res, n_factors = ncol(fit$data$W))
  expect_false(is.null(model_info))
  expect_false(isTRUE(model_info$pairwise_mode))
  expect_identical(model_info$likelihood, "normal")
  expect_true(all(is.finite(as.numeric(res$Q_point))))
})

test_that("full-data normal neural path runs with direct MCMC warm-start init", {
  skip_on_cran()
  skip_if_no_jax()

  withr::local_envvar(c(
    STRATEGIZE_NEURAL_FAST_MCMC = "false",
    STRATEGIZE_NEURAL_SKIP_EVAL = "true"
  ))

  data <- generate_average_case_neural_data(seed = 20260327)
  params <- default_strategize_params(fast = TRUE)
  params$diff <- FALSE
  params$force_gaussian <- TRUE
  params$compute_se <- FALSE
  params$outcome_model_type <- "neural"
  params$nMonte_Qglm <- 4L
  params$nSGD <- 1L
  params$optim_type <- "gd"
  base_neural_control <- params$neural_mcmc_control
  if (is.null(base_neural_control)) {
    base_neural_control <- list()
  }
  params$neural_mcmc_control <- modifyList(
    base_neural_control,
    list(
      subsample_method = "full",
      uncertainty_scope = "all",
      ModelDims = 8L,
      ModelDepth = 1L,
      n_samples_warmup = 1L,
      n_samples_mcmc = 1L,
      n_chains = 1L,
      chain_method = "sequential",
      eval_enabled = FALSE,
      warn_stage_imbalance_pct = 0,
      warn_min_cell_n = 0L
    )
  )

  p_list <- generate_test_p_list(data$W)
  res <- suppressWarnings(do.call(strategize, c(
    list(Y = data$Y, W = data$W, p_list = p_list),
    params
  )))

  model_info <- get_neural_model_info(res)
  expect_valid_strategize_output(res, n_factors = ncol(data$W))
  expect_false(is.null(model_info))
  expect_identical(model_info$likelihood, "normal")
  expect_true(all(is.finite(as.numeric(res$Q_point))))
})

test_that("AutoDelta is rejected when neural SEs are requested", {
  expect_error(
    run_average_case_neural_fit(
      vi_guide = "auto_delta",
      compute_se = TRUE,
      svi_steps = 5L
    ),
    "compute_se = TRUE is not supported when neural_mcmc_control\\$vi_guide = 'auto_delta'"
  )
})

test_that("average-case neural gaussian Q helper can enumerate exact single-party support", {
  skip_on_cran()
  skip_if_no_jax()

  if (!"jnp" %in% ls(envir = strategize:::strenv)) {
    strategize:::initialize_jax(conda_env = "strategize_env", conda_env_required = TRUE)
  }

  pi_ast <- strategize:::strenv$jnp$array(c(0.60, 0.40, 0.30, 0.70), dtype = strategize:::strenv$dtj)
  pi_dag <- strategize:::strenv$jnp$array(c(0.55, 0.45, 0.20, 0.80), dtype = strategize:::strenv$dtj)
  seed <- strategize:::strenv$jax$random$PRNGKey(123L)
  d_locator <- strategize:::strenv$jnp$array(c(1L, 1L, 2L, 2L))

  neural_draws <- strategize:::draw_average_case_q_profiles(
    pi_star_ast = pi_ast,
    pi_star_dag = pi_dag,
    outcome_model_type = "neural",
    glm_family = "gaussian",
    nMonte_Qglm = 4L,
    seed_in = seed,
    temperature = 0.5,
    ParameterizationType = "Full",
    d_locator_use = d_locator,
    sampler = strategize:::strenv$getMultinomialSamp,
    use_exact_support = TRUE,
    exact_support_single_party = TRUE
  )

  ast_profiles <- reticulate::py_to_r(strategize:::strenv$np$array(neural_draws$pi_star_ast_f_all))
  dag_profiles <- reticulate::py_to_r(strategize:::strenv$np$array(neural_draws$pi_star_dag_f_all))
  profile_weights <- as.numeric(reticulate::py_to_r(strategize:::strenv$np$array(neural_draws$profile_weights)))

  expect_true(isTRUE(neural_draws$use_mc_q))
  expect_true(isTRUE(neural_draws$exact_support))
  expect_identical(as.integer(neural_draws$n_draws), 4L)
  expect_identical(dim(ast_profiles), c(4L, 4L))
  expect_identical(dim(dag_profiles), c(4L, 4L))
  expect_equal(rowSums(ast_profiles[, 1:2, drop = FALSE]), rep(1, 4))
  expect_equal(rowSums(ast_profiles[, 3:4, drop = FALSE]), rep(1, 4))
  expect_true(all(vapply(seq_len(nrow(dag_profiles)), function(i) {
    isTRUE(all.equal(dag_profiles[i, ], c(0.55, 0.45, 0.20, 0.80), tolerance = 1e-6))
  }, logical(1))))
  expect_equal(sum(profile_weights), 1, tolerance = 1e-6)
})

test_that("multinomial group spec prefers concrete locator over stale globals", {
  skip_if_no_jax()

  if (!"jnp" %in% ls(envir = strategize:::strenv)) {
    strategize:::initialize_jax(conda_env = "strategize_env", conda_env_required = TRUE)
  }

  local_strenv_bindings(c("nUniqueFactors", "nUniqueLevelsByFactors"))
  set_strenv_bindings(list(
    nUniqueFactors = 3L,
    nUniqueLevelsByFactors = c(2L, 2L, 2L)
  ))

  spec <- strategize:::resolve_multinomial_group_spec(
    d_locator_use = strategize:::strenv$jnp$array(c(1L, 1L)),
    ParameterizationType = "Full"
  )

  expect_identical(spec$n_unique_factors, 1L)
  expect_identical(spec$n_unique_levels_by_factors, 2L)
})

test_that("multinomial group spec falls back to globals when locator unavailable", {
  local_strenv_bindings(c("nUniqueFactors", "nUniqueLevelsByFactors"))
  set_strenv_bindings(list(
    nUniqueFactors = 2L,
    nUniqueLevelsByFactors = c(2L, 3L)
  ))

  spec <- strategize:::resolve_multinomial_group_spec(
    d_locator_use = NULL,
    ParameterizationType = "Full"
  )

  expect_identical(spec$n_unique_factors, 2L)
  expect_identical(spec$n_unique_levels_by_factors, c(2L, 3L))
})

test_that("multinomial group spec preserves implicit holdout levels", {
  skip_if_no_jax()

  if (!"jnp" %in% ls(envir = strategize:::strenv)) {
    strategize:::initialize_jax(conda_env = "strategize_env", conda_env_required = TRUE)
  }

  local_strenv_bindings(c("nUniqueFactors", "nUniqueLevelsByFactors"))
  set_strenv_bindings(list(
    nUniqueFactors = 2L,
    nUniqueLevelsByFactors = c(4L, 4L)
  ))

  spec <- strategize:::resolve_multinomial_group_spec(
    d_locator_use = strategize:::strenv$jnp$array(c(1L, 1L)),
    ParameterizationType = "Implicit"
  )

  expect_identical(spec$n_unique_factors, 1L)
  expect_identical(spec$n_unique_levels_by_factors, 3L)
})

test_that("Q evaluation resolver aligns non-adversarial neural objective/report to hard semantics", {
  avg_obj <- strategize:::resolve_q_eval_spec(
    phase = "objective",
    adversarial = FALSE,
    outcome_model_type = "neural",
    glm_family = "gaussian",
    nMonte_Qglm = 9L,
    ParameterizationType = "Full",
    d_locator_use = strategize:::strenv$jnp$array(c(1L, 1L, 2L, 2L)),
    single_party = TRUE
  )
  avg_report <- strategize:::resolve_q_eval_spec(
    phase = "report",
    adversarial = FALSE,
    outcome_model_type = "neural",
    glm_family = "gaussian",
    nMonte_Qglm = 9L,
    ParameterizationType = "Full",
    d_locator_use = strategize:::strenv$jnp$array(c(1L, 1L, 2L, 2L)),
    single_party = TRUE
  )
  adv_obj <- strategize:::resolve_q_eval_spec(
    phase = "objective",
    adversarial = TRUE,
    outcome_model_type = "neural",
    glm_family = "gaussian",
    nMonte_Qglm = 9L,
    nMonte_adversarial = 5L
  )
  adv_large_obj <- strategize:::resolve_q_eval_spec(
    phase = "objective",
    adversarial = TRUE,
    outcome_model_type = "neural",
    glm_family = "gaussian",
    nMonte_Qglm = 9L,
    nMonte_adversarial = 5L,
    ParameterizationType = "Full",
    d_locator_use = strategize:::strenv$jnp$array(rep(seq_len(12L), each = 2L))
  )
  avg_large_obj <- strategize:::resolve_q_eval_spec(
    phase = "objective",
    adversarial = FALSE,
    outcome_model_type = "neural",
    glm_family = "gaussian",
    nMonte_Qglm = 9L,
    ParameterizationType = "Full",
    d_locator_use = strategize:::strenv$jnp$array(rep(seq_len(12L), each = 2L)),
    single_party = TRUE
  )
  glm_exact <- strategize:::resolve_q_eval_spec(
    phase = "report",
    adversarial = FALSE,
    outcome_model_type = "glm",
    glm_family = "gaussian",
    nMonte_Qglm = 9L
  )

  expect_identical(avg_obj$profile_draw_mode, "hard")
  expect_identical(avg_report$profile_draw_mode, "hard")
  expect_true(isTRUE(avg_obj$use_exact_support))
  expect_true(isTRUE(avg_report$use_exact_support))
  expect_identical(avg_obj$objective_gradient_mode, "exact")
  expect_identical(avg_report$objective_gradient_mode, "exact")
  expect_identical(as.integer(avg_obj$n_draws), 4L)
  expect_identical(as.integer(avg_report$n_draws), 4L)
  expect_identical(adv_obj$profile_draw_mode, "relaxed")
  expect_identical(adv_obj$objective_gradient_mode, "pathwise")
  expect_identical(adv_obj$n_draws, 5L)
  expect_identical(avg_large_obj$profile_draw_mode, "hard")
  expect_false(isTRUE(avg_large_obj$use_exact_support))
  expect_true(isTRUE(avg_large_obj$is_large_support))
  expect_identical(avg_large_obj$objective_gradient_mode, "reinforce")
  expect_identical(adv_large_obj$profile_draw_mode, "hard")
  expect_true(isTRUE(adv_large_obj$is_large_support))
  expect_identical(adv_large_obj$objective_gradient_mode, "reinforce")
  expect_true(isTRUE(glm_exact$use_exact_q))
  expect_identical(glm_exact$profile_draw_mode, "exact")
})

test_that("hard profile draw mode emits one-hot average-case samples", {
  skip_on_cran()
  skip_if_no_jax()

  if (!"jnp" %in% ls(envir = strategize:::strenv)) {
    strategize:::initialize_jax(conda_env = "strategize_env", conda_env_required = TRUE)
  }

  pi_ast <- strategize:::strenv$jnp$array(c(0.25, 0.75), dtype = strategize:::strenv$dtj)
  pi_dag <- strategize:::strenv$jnp$array(c(0.60, 0.40), dtype = strategize:::strenv$dtj)
  seed <- strategize:::strenv$jax$random$PRNGKey(123L)
  local_strenv_bindings(c("nUniqueFactors", "nUniqueLevelsByFactors", "getMultinomialSampHard"))
  set_strenv_bindings(list(
    nUniqueFactors = 3L,
    nUniqueLevelsByFactors = c(2L, 2L, 2L),
    getMultinomialSampHard = strategize:::strenv$jax$jit(
      strategize:::getMultinomialSampHard_R,
      static_argnums = 3L,
      static_argnames = c("ParameterizationType")
    )
  ))

  hard_draws <- strategize:::draw_average_case_q_profiles(
    pi_star_ast = pi_ast,
    pi_star_dag = pi_dag,
    outcome_model_type = "neural",
    glm_family = "gaussian",
    nMonte_Qglm = 8L,
    seed_in = seed,
    temperature = 0.5,
    ParameterizationType = "Full",
    d_locator_use = strategize:::strenv$jnp$array(c(1L, 1L)),
    profile_draw_mode = "hard"
  )

  ast_mat <- unclass(strategize:::strenv$np$array(hard_draws$pi_star_ast_f_all))
  dag_mat <- unclass(strategize:::strenv$np$array(hard_draws$pi_star_dag_f_all))

  expect_true(all(ast_mat %in% c(0, 1)))
  expect_true(all(dag_mat %in% c(0, 1)))
  expect_identical(dim(ast_mat), c(8L, 2L))
  expect_identical(dim(dag_mat), c(8L, 2L))
  expect_true(all(rowSums(ast_mat) == 1))
  expect_true(all(rowSums(dag_mat) == 1))
})
