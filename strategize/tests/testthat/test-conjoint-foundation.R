foundation_test_control <- function() {
  list(
    neural_mcmc_control = list(
      ModelDims = 16L,
      ModelDepth = 1L,
      subsample_method = "batch_vi",
      uncertainty_scope = "output",
      optimizer = "adam",
      svi_steps = 8L,
      svi_num_draws = 2L,
      batch_size = 16L,
      early_stopping = FALSE
    )
  )
}

foundation_test_text_embedding_fn <- function(x) {
  x <- as.character(x)
  cbind(
    nchar = nchar(x),
    has_space = 1 * grepl("\\s", x),
    has_upper = 1 * grepl("[A-Z]", x)
  )
}

foundation_test_covariates <- function(W_df) {
  n <- nrow(W_df)
  idx <- seq_len(n)
  data.frame(
    income = idx / max(n, 1L),
    `household size` = 1 + (idx %% 4L) + 0.5 * (W_df[[1]] == "B"),
    GOPScore = as.numeric(W_df[[ncol(W_df)]] == "B") - 0.5,
    local_bonus = seq(-1, 1, length.out = n),
    check.names = FALSE
  )
}

foundation_test_experiment <- function(seed,
                                       experiment_id,
                                       factor_names,
                                       x_names = NULL,
                                       canonical_factor_id = NULL) {
  data <- generate_test_data(
    n = 40,
    n_factors = length(factor_names),
    n_levels = 2,
    seed = seed
  )
  W_df <- as.data.frame(data$W, stringsAsFactors = FALSE)
  colnames(W_df) <- factor_names
  if (is.null(canonical_factor_id)) {
    canonical_factor_id <- stats::setNames(factor_names, factor_names)
  }
  x_full <- foundation_test_covariates(W_df)
  X <- if (is.null(x_names) || length(x_names) < 1L) {
    NULL
  } else {
    x_full[, x_names, drop = FALSE]
  }

  list(
    experiment_id = experiment_id,
    experiment_description = paste(
      "Experiment", experiment_id, "with factors", paste(factor_names, collapse = ", ")
    ),
    Y = data$Y,
    W = W_df,
    X = X,
    pair_id = data$pair_id,
    profile_order = data$profile_order,
    respondent_id = data$respondent_id,
    respondent_task_id = data$respondent_task_id,
    canonical_factor_id = canonical_factor_id
  )
}

foundation_test_single_experiment <- function(seed,
                                              experiment_id,
                                              factor_names = c("price", "message"),
                                              x_names = NULL,
                                              likelihood = c("bernoulli", "normal")) {
  likelihood <- match.arg(likelihood)
  set.seed(seed)
  n <- 24L
  W_df <- data.frame(
    matrix(sample(c("A", "B"), n * length(factor_names), replace = TRUE), nrow = n),
    stringsAsFactors = FALSE
  )
  colnames(W_df) <- factor_names
  x_full <- foundation_test_covariates(W_df)
  X <- if (is.null(x_names) || length(x_names) < 1L) {
    NULL
  } else {
    x_full[, x_names, drop = FALSE]
  }
  signal <- 0.8 * (W_df[[1]] == "B") - 0.5 * (W_df[[2]] == "A")
  Y <- if (identical(likelihood, "bernoulli")) {
    stats::rbinom(n, size = 1L, prob = stats::plogis(-0.25 + signal))
  } else {
    stats::rnorm(n, mean = 0.4 + signal, sd = 0.2)
  }

  list(
    experiment_id = experiment_id,
    experiment_description = paste("Single experiment", experiment_id),
    Y = Y,
    W = W_df,
    X = X,
    mode = "single",
    likelihood = likelihood,
    canonical_factor_id = stats::setNames(factor_names, factor_names)
  )
}

foundation_test_control_with_embeddings <- function() {
  modifyList(
    strategize:::cs_foundation_default_control(),
    modifyList(
      foundation_test_control(),
      list(text_embedding_fn = foundation_test_text_embedding_fn)
    )
  )
}

foundation_pairwise_group_key <- function(context_mode = "stage_free",
                                          likelihood = "bernoulli",
                                          n_outcomes = 1L) {
  paste("pairwise", context_mode, likelihood, as.integer(n_outcomes), sep = "::")
}

foundation_universal_group_key <- function() {
  "universal::mixed::v1"
}

test_that("foundation defaults use language-span factor tokenization", {
  control <- strategize:::cs_foundation_default_control()
  expect_identical(control$factor_tokenization, "language_span")
  expect_identical(control$max_factor_tokens, 256L)
  expect_identical(control$shared_projection_value_encoder, "name_dist_moe")
})

test_that("foundation token info preserves raw covariate order and token budget", {
  study_a <- foundation_test_experiment(
    seed = 7001,
    experiment_id = "study_a",
    factor_names = c("price", "message"),
    x_names = c("income", "household size")
  )
  study_b <- foundation_test_experiment(
    seed = 7002,
    experiment_id = "study_b",
    factor_names = c("price", "message", "messenger"),
    x_names = c("GOPScore", "income")
  )
  experiments_norm <- list(
    strategize:::cs_foundation_normalize_experiment(study_a, index = 1L),
    strategize:::cs_foundation_normalize_experiment(study_b, index = 2L)
  )
  registry <- strategize:::cs_foundation_build_group_registry(experiments_norm)
  control <- foundation_test_control_with_embeddings()
  control$factor_tokenization <- "legacy_indexed"
  control$max_covariate_tokens <- 12L

  pooled <- strategize:::cs_foundation_build_group_training_data(experiments_norm, registry, control)
  expect_identical(pooled$token_control$max_covariate_tokens, 12L)
  expect_identical(pooled$token_info$max_covariate_tokens, 12L)
  expect_identical(pooled$token_control$shared_projection_value_encoder, "name_dist_moe")
  expect_identical(pooled$token_info$shared_projection_value_encoder, "name_dist_moe")
  expect_identical(pooled$x_schema$base_x_names, c("income", "household size", "GOPScore"))
  expect_identical(pooled$token_info$covariate_order_by_experiment[[1]], c(0L, 1L))
  expect_identical(pooled$token_info$covariate_order_by_experiment[[2]], c(2L, 0L))
})

test_that("foundation token info preserves raw factor order and factor token budget", {
  study_a <- foundation_test_experiment(
    seed = 7101,
    experiment_id = "study_a",
    factor_names = c("price", "message"),
    x_names = c("income")
  )
  study_b <- foundation_test_experiment(
    seed = 7102,
    experiment_id = "study_b",
    factor_names = c("messenger", "price", "message"),
    x_names = c("income")
  )
  experiments_norm <- list(
    strategize:::cs_foundation_normalize_experiment(study_a, index = 1L),
    strategize:::cs_foundation_normalize_experiment(study_b, index = 2L)
  )
  registry <- strategize:::cs_foundation_build_group_registry(experiments_norm)
  control <- foundation_test_control_with_embeddings()
  control$factor_tokenization <- "legacy_indexed"
  control$max_factor_tokens <- 20L

  pooled <- strategize:::cs_foundation_build_group_training_data(experiments_norm, registry, control)

  expect_identical(pooled$token_control$factor_tokenization, "legacy_indexed")
  expect_identical(pooled$token_control$max_factor_tokens, 20L)
  expect_identical(pooled$token_info$factor_tokenization, "legacy_indexed")
  expect_identical(pooled$token_info$max_factor_tokens, 20L)
  expect_identical(pooled$token_info$factor_order_by_experiment[[1]], c(0L, 1L))
  expect_identical(pooled$token_info$factor_order_by_experiment[[2]], c(2L, 0L, 1L))
})

test_that("fit_conjoint_foundation_model pools compatible pairwise studies with X-name semantics", {
  skip_on_cran()
  skip_if_no_jax()
  withr::local_envvar(c(STRATEGIZE_NEURAL_SKIP_EVAL = "1"))

  study_a <- foundation_test_experiment(
    seed = 7001,
    experiment_id = "study_a",
    factor_names = c("price", "message"),
    x_names = c("income", "household size")
  )
  study_b <- foundation_test_experiment(
    seed = 7002,
    experiment_id = "study_b",
    factor_names = c("price", "message", "messenger"),
    x_names = c("income", "GOPScore")
  )
  control <- foundation_test_control_with_embeddings()

  fit <- fit_conjoint_foundation_model(
    experiments = list(study_a, study_b),
    foundation_control = control
  )

  expect_s3_class(fit, "conjoint_foundation_model")
  expect_length(fit$groups, 1L)
  group <- fit$groups[[foundation_universal_group_key()]]
  expect_false(is.null(group))
  expect_identical(group$pairwise_context_mode, "stage_free")
  expect_identical(
    group$x_feature_names,
    c("income", "household size", "GOPScore")
  )
  expect_identical(group$token_control$experiment_token_mode, "description")
  expect_identical(group$token_control$covariate_value_encoding, "shared_projection")
  expect_identical(group$token_control$shared_projection_value_encoder, "name_dist_moe")
  expect_identical(group$token_control$max_covariate_tokens, 512L)
  expect_identical(group$x_schema$base_x_names, group$x_feature_names)
  expect_identical(group$x_schema$semantic_feature_names, character(0))
  expect_identical(group$x_schema$experiment_indicator_names, character(0))
  expect_identical(group$x_schema$experiment_token_levels, c("study_a", "study_b"))
  expect_identical(
    group$text_registry$x_feature_names,
    group$x_schema$base_x_names
  )
  expect_equal(
    nrow(group$text_registry$x_feature_embedding),
    length(group$x_schema$base_x_names)
  )
  expect_identical(group$fit$neural_model_info$covariate_names, group$x_schema$base_x_names)
  expect_identical(
    rownames(group$fit$neural_model_info$covariate_name_text),
    group$x_schema$base_x_names
  )
  expect_true(isTRUE(group$fit$neural_model_info$has_covariate_tokens))
  expect_true(isTRUE(group$fit$neural_model_info$has_covariate_span_tokens))
  expect_true(isTRUE(group$fit$neural_model_info$has_token_family_embedding))
  expect_true(isTRUE(group$fit$neural_model_info$has_experiment_token))
  expect_false(isTRUE(group$fit$neural_model_info$has_experiment_id_embedding))
  expect_true(isTRUE(group$fit$neural_model_info$has_experiment_text_projection))
  expect_true(isTRUE(group$fit$neural_model_info$has_factor_name_text))
  expect_true(isTRUE(group$fit$neural_model_info$has_level_name_text))
  expect_true(isTRUE(group$fit$neural_model_info$has_covariate_name_text))
  expect_true(isTRUE(group$fit$neural_model_info$has_shared_covariate_value_projection))
  expect_true(isTRUE(group$fit$neural_model_info$has_conditioned_covariate_value_encoder))
  expect_false(isTRUE(group$fit$neural_model_info$has_candidate_group_context))
  expect_false(isTRUE(group$fit$neural_model_info$has_respondent_group_context))
  expect_false(isTRUE(group$fit$neural_model_info$has_relation_token_context))
  expect_false(isTRUE(group$fit$neural_model_info$has_stage_context))
  expect_false(isTRUE(group$fit$neural_model_info$has_matchup_context))
  expect_false(isTRUE(group$fit$neural_model_info$has_stage_token))
  expect_false(isTRUE(group$fit$neural_model_info$has_matchup_token))
  expect_identical(
    group$fit$neural_model_info$shared_projection_value_encoder,
    "name_dist_moe"
  )
  expect_length(
    as.numeric(strategize:::cs2step_neural_to_r_array(group$fit$neural_model_info$resp_cov_scale)),
    length(group$x_schema$base_x_names)
  )
  expect_true(all(
    c("factor_candidate", "covariate", "experiment", "choice", "separator") %in%
      group$fit$neural_model_info$token_family_levels
  ))
  expect_false(any(
    c("party", "relation", "stage", "resp_party", "matchup") %in%
      group$fit$neural_model_info$token_family_levels
  ))

  experiments_norm <- list(
    cs_foundation_normalize_experiment(study_a, index = 1L),
    cs_foundation_normalize_experiment(study_b, index = 2L)
  )
  registry <- cs_foundation_build_group_registry(experiments_norm)
  pooled <- cs_foundation_build_group_training_data(experiments_norm, registry, control)
  expect_identical(colnames(pooled$X), group$x_schema$base_x_names)
  expect_identical(colnames(pooled$X_present), group$x_schema$base_x_names)
  expect_false(any(grepl("^semantic_", colnames(pooled$X))))
  expect_identical(pooled$token_info$covariate_names, group$x_schema$base_x_names)
  expect_identical(
    rownames(pooled$token_info$covariate_name_text),
    group$x_schema$base_x_names
  )
  expect_identical(
    rownames(pooled$token_info$experiment_description_text),
    c("study_a", "study_b")
  )
  expect_true(all(pooled$token_info$experiment_description_present))
  expect_identical(pooled$token_info$experiment_token_mode, "description")
  expect_identical(pooled$token_info$covariate_value_encoding, "shared_projection")
  expect_identical(pooled$token_info$max_covariate_tokens, 512L)
  expect_identical(pooled$token_info$covariate_order_by_experiment[[1]], c(0L, 1L))
  expect_identical(pooled$token_info$covariate_order_by_experiment[[2]], c(0L, 2L))
  expect_true(all(pooled$X_present[seq_len(nrow(study_a$W)), "GOPScore"] == 0))
  expect_true(all(
    pooled$X_present[-seq_len(nrow(study_a$W)), "household size"] == 0
  ))
  expect_true(any(pooled$X_present[, "income"] == 1))

  runtime_info <- group$fit$neural_model_info
  params <- group$fit$neural_model_info$params
  n_cov <- length(group$x_schema$base_x_names)
  resp_party_idx <- strategize:::neural_get_resp_party_index(runtime_info)

  tok_study_a <- strategize:::add_context_tokens(
    model_info = runtime_info,
    resp_party_idx = resp_party_idx,
    resp_cov = matrix(0, nrow = 1L, ncol = n_cov),
    resp_cov_present = matrix(0, nrow = 1L, ncol = n_cov),
    experiment_idx = 0L,
    params = params,
    batch = FALSE
  )
  tok_study_b <- strategize:::add_context_tokens(
    model_info = runtime_info,
    resp_party_idx = resp_party_idx,
    resp_cov = matrix(0, nrow = 1L, ncol = n_cov),
    resp_cov_present = matrix(0, nrow = 1L, ncol = n_cov),
    experiment_idx = 1L,
    params = params,
    batch = FALSE
  )
  tok_study_a_r <- reticulate::py_to_r(strategize:::strenv$np$array(tok_study_a))
  tok_study_b_r <- reticulate::py_to_r(strategize:::strenv$np$array(tok_study_b))
  expect_gt(max(abs(tok_study_a_r[1, 1, ] - tok_study_b_r[1, 1, ])), 1e-8)

  tok_cov0 <- strategize:::add_context_tokens(
    model_info = runtime_info,
    resp_party_idx = resp_party_idx,
    resp_cov = matrix(0, nrow = 1L, ncol = n_cov),
    resp_cov_present = matrix(1, nrow = 1L, ncol = n_cov),
    experiment_idx = 0L,
    params = params,
    batch = FALSE
  )
  tok_cov1 <- strategize:::add_context_tokens(
    model_info = runtime_info,
    resp_party_idx = resp_party_idx,
    resp_cov = matrix(1, nrow = 1L, ncol = n_cov),
    resp_cov_present = matrix(1, nrow = 1L, ncol = n_cov),
    experiment_idx = 0L,
    params = params,
    batch = FALSE
  )
  tok_cov0_r <- reticulate::py_to_r(strategize:::strenv$np$array(tok_cov0))
  tok_cov1_r <- reticulate::py_to_r(strategize:::strenv$np$array(tok_cov1))
  expect_gt(max(abs(tok_cov1_r - tok_cov0_r)), 1e-8)
})

test_that("fit_conjoint_foundation_model separates stage-aware pairwise studies", {
  skip_on_cran()
  skip_if_no_jax()
  withr::local_envvar(c(STRATEGIZE_NEURAL_SKIP_EVAL = "1"))

  study_a <- add_foundation_pairwise_context(
    foundation_test_experiment(
      seed = 7051,
      experiment_id = "stage_a",
      factor_names = c("price", "message"),
      x_names = c("income", "household size")
    ),
    seed = 11
  )
  study_b <- add_foundation_pairwise_context(
    foundation_test_experiment(
      seed = 7052,
      experiment_id = "stage_b",
      factor_names = c("price", "message", "messenger"),
      x_names = c("income", "GOPScore")
    ),
    seed = 12
  )

  fit <- fit_conjoint_foundation_model(
    experiments = list(study_a, study_b),
    foundation_control = foundation_test_control_with_embeddings()
  )

  group <- fit$groups[[foundation_universal_group_key()]]
  expect_false(is.null(group))
  expect_identical(group$pairwise_context_mode, "stage_aware")
  expect_true(isTRUE(group$fit$neural_model_info$has_candidate_group_context))
  expect_true(isTRUE(group$fit$neural_model_info$has_respondent_group_context))
  expect_true(isTRUE(group$fit$neural_model_info$has_relation_token_context))
  expect_true(isTRUE(group$fit$neural_model_info$has_stage_context))
  expect_true(isTRUE(group$fit$neural_model_info$has_matchup_context))
  expect_true(isTRUE(group$fit$neural_model_info$has_stage_token))
  expect_true(isTRUE(group$fit$neural_model_info$has_matchup_token))
  expect_true(all(
    c("party", "relation", "stage", "resp_party", "matchup") %in%
      group$fit$neural_model_info$token_family_levels
  ))
  expect_gt(group$fit$neural_model_info$stage_diagnostics$n_primary, 0L)
  expect_gt(group$fit$neural_model_info$stage_diagnostics$n_general, 0L)
  expect_false(isTRUE(group$fit$neural_model_info$stage_diagnostics$single_stage_only))
})

test_that("pairwise studies with group metadata but one observed stage stay stage-free", {
  skip_on_cran()
  skip_if_no_jax()
  withr::local_envvar(c(STRATEGIZE_NEURAL_SKIP_EVAL = "1"))

  study <- add_foundation_pairwise_context(
    foundation_test_experiment(
      seed = 7053,
      experiment_id = "single_stage",
      factor_names = c("price", "message"),
      x_names = c("income")
    ),
    seed = 13,
    single_stage = TRUE
  )

  fit <- fit_conjoint_foundation_model(
    experiments = list(study),
    foundation_control = foundation_test_control_with_embeddings()
  )

  group <- fit$groups[[foundation_universal_group_key()]]
  expect_false(is.null(group))
  expect_identical(group$pairwise_context_mode, "stage_free")
  expect_true(isTRUE(group$fit$neural_model_info$has_candidate_group_context))
  expect_true(isTRUE(group$fit$neural_model_info$has_respondent_group_context))
  expect_true(isTRUE(group$fit$neural_model_info$has_relation_token_context))
  expect_false(isTRUE(group$fit$neural_model_info$has_stage_context))
  expect_false(isTRUE(group$fit$neural_model_info$has_matchup_context))
  expect_false(isTRUE(group$fit$neural_model_info$has_stage_token))
  expect_false(isTRUE(group$fit$neural_model_info$has_matchup_token))
  expect_true(isTRUE(group$fit$neural_model_info$stage_diagnostics$single_stage_only))
})

test_that("pairwise studies stay stage-aware with partially missing respondent groups", {
  skip_on_cran()
  skip_if_no_jax()
  withr::local_envvar(c(STRATEGIZE_NEURAL_SKIP_EVAL = "1"))

  study <- add_foundation_pairwise_context(
    foundation_test_experiment(
      seed = 7054,
      experiment_id = "missing_resp_stage",
      factor_names = c("price", "message"),
      x_names = c("income")
    ),
    seed = 14
  )
  study$competing_group_variable_respondent[
    seq(1L, length(study$competing_group_variable_respondent), by = 7L)
  ] <- NA_character_

  fit <- fit_conjoint_foundation_model(
    experiments = list(study),
    foundation_control = foundation_test_control_with_embeddings()
  )

  group <- fit$groups[[foundation_universal_group_key()]]
  expect_false(is.null(group))
  expect_identical(group$pairwise_context_mode, "stage_aware")
  expect_true(isTRUE(group$fit$neural_model_info$has_stage_context))
  expect_true(isTRUE(group$fit$neural_model_info$has_matchup_context))
  expect_true(isTRUE(group$fit$neural_model_info$has_stage_token))
  expect_true(isTRUE(group$fit$neural_model_info$has_matchup_token))
})

test_that("fit_conjoint_foundation_model splits incompatible likelihood families", {
  skip_on_cran()
  skip_if_no_jax()
  withr::local_envvar(c(STRATEGIZE_NEURAL_SKIP_EVAL = "1"))

  pairwise_data <- generate_test_data(n = 40, n_factors = 2, n_levels = 2, seed = 7011)
  single_W <- data.frame(
    policy = sample(c("A", "B", "C"), 30, replace = TRUE),
    messenger = sample(c("Local", "National"), 30, replace = TRUE),
    stringsAsFactors = FALSE
  )
  single_Y <- rnorm(30, mean = 0.5 * (single_W$policy == "B"), sd = 0.1)

  fit <- fit_conjoint_foundation_model(
    experiments = list(
      list(
        experiment_id = "pairwise_binary",
        Y = pairwise_data$Y,
        W = pairwise_data$W,
        pair_id = pairwise_data$pair_id,
        profile_order = pairwise_data$profile_order
      ),
      list(
        experiment_id = "single_normal",
        Y = single_Y,
        W = single_W,
        mode = "single",
        likelihood = "normal"
      )
    ),
    foundation_control = foundation_test_control()
  )

  expect_s3_class(fit, "conjoint_foundation_model")
  expect_identical(names(fit$groups), foundation_universal_group_key())
  expect_identical(sort(fit$groups[[foundation_universal_group_key()]]$supported_modes), c("pairwise", "single"))
  expect_identical(sort(fit$groups[[foundation_universal_group_key()]]$supported_likelihoods), c("bernoulli", "normal"))
})

test_that("mixed-family universal fits report OOS metrics across Bernoulli and normal rows", {
  skip_on_cran()
  skip_if_no_jax()
  withr::local_envvar(c(
    STRATEGIZE_NEURAL_FAST_MCMC = "true",
    STRATEGIZE_NEURAL_EVAL_FOLDS = "2",
    STRATEGIZE_NEURAL_EVAL_SEED = "123"
  ))

  control <- foundation_test_control()
  control$neural_mcmc_control$eval_enabled <- TRUE

  fit <- fit_conjoint_foundation_model(
    experiments = list(
      foundation_test_single_experiment(8011, "single_binary", likelihood = "bernoulli"),
      foundation_test_single_experiment(8012, "single_normal", likelihood = "normal")
    ),
    foundation_control = control
  )

  group <- fit$groups[[foundation_universal_group_key()]]
  metrics <- group$fit$fit_metrics
  expect_identical(metrics$likelihood, "mixed")
  expect_true(is.finite(metrics$nll))
  expect_true(all(c("bernoulli", "normal") %in% names(metrics$by_family)))
  expect_true(is.finite(metrics$by_family$bernoulli$log_loss))
  expect_true(is.finite(metrics$by_family$normal$nll))
})

test_that("mixed-family universal fits use NLL early stopping", {
  skip_on_cran()
  skip_if_no_jax()
  withr::local_envvar(c(
    STRATEGIZE_NEURAL_FAST_MCMC = "true",
    STRATEGIZE_NEURAL_EVAL_SEED = "123"
  ))

  control <- foundation_test_control()
  control$neural_mcmc_control$eval_enabled <- FALSE
  control$neural_mcmc_control$early_stopping <- TRUE

  fit <- fit_conjoint_foundation_model(
    experiments = list(
      foundation_test_single_experiment(8021, "single_binary_es", likelihood = "bernoulli"),
      foundation_test_single_experiment(8022, "single_normal_es", likelihood = "normal")
    ),
    foundation_control = control
  )

  group <- fit$groups[[foundation_universal_group_key()]]
  es <- group$fit$neural_model_info$early_stopping
  expect_true(isTRUE(es$enabled))
  expect_true(isTRUE(es$active))
  expect_identical(es$metric, "nll")
  expect_true(length(es$validation_loss_history) >= 1L)
  expect_true(is.finite(es$best_metric))
})

test_that("foundation early stopping forwards validation size controls", {
  skip_on_cran()
  skip_if_no_jax()
  withr::local_envvar(c(
    STRATEGIZE_NEURAL_FAST_MCMC = "true",
    STRATEGIZE_NEURAL_EVAL_SEED = "123"
  ))

  control <- foundation_test_control()
  control$neural_mcmc_control$eval_enabled <- FALSE
  control$neural_mcmc_control$early_stopping <- TRUE
  control$neural_mcmc_control$early_stopping_validation_frac <- 1
  control$neural_mcmc_control$early_stopping_validation_max_n <- 4L

  fit <- fit_conjoint_foundation_model(
    experiments = list(
      foundation_test_single_experiment(8031, "single_binary_es_cap", likelihood = "bernoulli"),
      foundation_test_single_experiment(8032, "single_normal_es_cap", likelihood = "normal")
    ),
    foundation_control = control
  )

  group <- fit$groups[[foundation_universal_group_key()]]
  es <- group$fit$neural_model_info$early_stopping
  expect_identical(es$validation_frac, 1)
  expect_identical(as.integer(es$validation_max_n), 4L)
  expect_lte(as.integer(es$validation_target_n), 4L)
  expect_identical(as.integer(es$validation_target_n), as.integer(es$n_validation))
  expect_lte(as.integer(es$n_validation), 4L)
})

test_that("adapt_conjoint_foundation_model builds shared and local covariate tokens and predictions stay finite", {
  skip_on_cran()
  skip_if_no_jax()
  withr::local_envvar(c(STRATEGIZE_NEURAL_SKIP_EVAL = "1"))

  foundation_fit <- fit_conjoint_foundation_model(
    experiments = list(
      foundation_test_experiment(
        seed = 7021,
        experiment_id = "study_a",
        factor_names = c("price", "message"),
        x_names = c("income", "household size")
      ),
      foundation_test_experiment(
        seed = 7022,
        experiment_id = "study_b",
        factor_names = c("price", "message", "messenger"),
        x_names = c("income", "GOPScore")
      )
    ),
    foundation_control = foundation_test_control_with_embeddings()
  )

  adapt_data <- foundation_test_experiment(
    seed = 7023,
    experiment_id = "target_study",
    factor_names = c("price", "message"),
    x_names = c("income", "household size", "local_bonus")
  )
  group <- foundation_fit$groups[[foundation_universal_group_key()]]
  experiment_norm <- cs_foundation_normalize_experiment(adapt_data, index = 1L)
  exp_map <- cs_foundation_build_local_factor_map(experiment_norm)
  adaptation_control <- modifyList(
    cs_foundation_default_adaptation_control(),
    list(text_embedding_fn = foundation_test_text_embedding_fn)
  )
  X_aug <- cs_foundation_build_adaptation_x(
    group = group,
    experiment = experiment_norm,
    exp_map = exp_map,
    adaptation_control = adaptation_control
  )
  X_present_aug <- attr(X_aug, "resp_cov_present", exact = TRUE)
  token_info <- attr(X_aug, "token_info", exact = TRUE)

  expect_true("local_bonus" %in% colnames(X_aug))
  expect_identical(
    colnames(X_aug),
    c(group$x_schema$base_x_names, "local_bonus")
  )
  expect_identical(colnames(X_present_aug), colnames(X_aug))
  expect_true(all(X_present_aug[, "local_bonus"] == 1))
  expect_true(all(X_present_aug[, "GOPScore"] == 0))
  expect_equal(
    token_info$covariate_names,
    colnames(X_aug)
  )
  expect_false("local_bonus" %in% group$x_schema$base_x_names)
  expect_identical(
    rownames(token_info$covariate_name_text),
    colnames(X_aug)
  )
  expect_true(isTRUE(token_info$default_experiment_text_present))
  expect_equal(nrow(token_info$default_experiment_text), 1L)
  expect_identical(token_info$experiment_token_mode, "description")
  expect_identical(token_info$covariate_value_encoding, "shared_projection")

  predictor <- adapt_conjoint_foundation_model(
    foundation_model = foundation_fit,
    Y = adapt_data$Y,
    W = adapt_data$W,
    X = adapt_data$X,
    mode = "pairwise",
    pair_id = adapt_data$pair_id,
    profile_order = adapt_data$profile_order,
    experiment_id = adapt_data$experiment_id,
    experiment_description = adapt_data$experiment_description,
    canonical_factor_id = adapt_data$canonical_factor_id,
    neural_mcmc_control = foundation_test_control()$neural_mcmc_control,
    foundation_adaptation_control = list(
      text_embedding_fn = foundation_test_text_embedding_fn
    )
  )

  expect_s3_class(predictor, "strategic_predictor")
  expect_identical(
    predictor$fit$neural_model_info$covariate_names,
    colnames(X_aug)
  )
  expect_true(isTRUE(predictor$fit$neural_model_info$has_covariate_tokens))
  expect_true(isTRUE(predictor$fit$neural_model_info$default_experiment_text_present))
  expect_null(predictor$fit$neural_model_info$default_experiment_index)

  preds <- predict(
    predictor,
    newdata = list(
      W = adapt_data$W,
      pair_id = adapt_data$pair_id,
      profile_order = adapt_data$profile_order
    )
  )
  preds_with_x <- predict(
    predictor,
    newdata = list(
      W = adapt_data$W,
      X = adapt_data$X,
      pair_id = adapt_data$pair_id,
      profile_order = adapt_data$profile_order,
      experiment_id = adapt_data$experiment_id
    )
  )
  expect_true(all(is.finite(preds)))
  expect_true(all(preds >= 0 & preds <= 1))
  expect_true(all(is.finite(preds_with_x)))
  expect_true(all(preds_with_x >= 0 & preds_with_x <= 1))

  preds_with_desc <- predict(
    predictor,
    newdata = list(
      W = adapt_data$W,
      X = adapt_data$X,
      pair_id = adapt_data$pair_id,
      profile_order = adapt_data$profile_order,
      experiment_description = adapt_data$experiment_description
    )
  )
  expect_true(all(is.finite(preds_with_desc)))
  expect_true(all(preds_with_desc >= 0 & preds_with_desc <= 1))
})

test_that("foundation semantics stay backward compatible when text_embedding_fn is NULL", {
  skip_on_cran()
  skip_if_no_jax()
  withr::local_envvar(c(STRATEGIZE_NEURAL_SKIP_EVAL = "1"))

  fit <- fit_conjoint_foundation_model(
    experiments = list(
      foundation_test_experiment(
        seed = 7031,
        experiment_id = "study_a",
        factor_names = c("price", "message"),
        x_names = c("income", "household size")
      ),
      foundation_test_experiment(
        seed = 7032,
        experiment_id = "study_b",
        factor_names = c("price", "message", "messenger"),
        x_names = c("income", "GOPScore")
      )
    ),
    foundation_control = foundation_test_control()
  )

  group <- fit$groups[[foundation_universal_group_key()]]
  expect_null(group$text_registry)
  expect_identical(group$token_control$experiment_token_mode, "description")
  expect_identical(group$token_control$covariate_value_encoding, "shared_projection")
  expect_identical(group$x_schema$semantic_feature_names, character(0))
  expect_identical(group$x_schema$experiment_indicator_names, character(0))
  expect_false(any(grepl("^semantic_x_", group$x_feature_names)))
  expect_false(any(grepl("^semantic_factor_", group$x_feature_names)))
  expect_false(any(grepl("^semantic_level_", group$x_feature_names)))
  expect_identical(
    group$x_feature_names,
    group$x_schema$base_x_names
  )
  expect_null(group$fit$neural_model_info$factor_name_text)
  expect_null(group$fit$neural_model_info$level_name_text)
  expect_null(group$fit$neural_model_info$covariate_name_text)
  expect_true(isTRUE(group$fit$neural_model_info$has_covariate_tokens))
  expect_false(isTRUE(group$fit$neural_model_info$has_factor_name_text))
  expect_false(isTRUE(group$fit$neural_model_info$has_level_name_text))
  expect_false(isTRUE(group$fit$neural_model_info$has_covariate_name_text))
  expect_true(isTRUE(group$fit$neural_model_info$has_shared_covariate_value_projection))
  expect_true(isTRUE(group$fit$neural_model_info$has_conditioned_covariate_value_encoder))
})

test_that("legacy fine-tuning token controls remain available for ablation", {
  skip_on_cran()
  skip_if_no_jax()
  withr::local_envvar(c(STRATEGIZE_NEURAL_SKIP_EVAL = "1"))

  fit <- fit_conjoint_foundation_model(
    experiments = list(
      foundation_test_experiment(
        seed = 7037,
        experiment_id = "study_a",
        factor_names = c("price", "message"),
        x_names = c("income", "household size")
      ),
      foundation_test_experiment(
        seed = 7038,
        experiment_id = "study_b",
        factor_names = c("price", "message", "messenger"),
        x_names = c("income", "GOPScore")
      )
    ),
    foundation_control = modifyList(
      foundation_test_control_with_embeddings(),
      list(
        experiment_token_mode = "legacy_id",
        covariate_value_encoding = "legacy_linear"
      )
    )
  )

  group <- fit$groups[[foundation_universal_group_key()]]
  expect_identical(group$token_control$experiment_token_mode, "legacy_id")
  expect_identical(group$token_control$covariate_value_encoding, "legacy_linear")
  expect_true(isTRUE(group$fit$neural_model_info$has_experiment_id_embedding))
  expect_false(isTRUE(group$fit$neural_model_info$has_experiment_text_projection))
  expect_false(isTRUE(group$fit$neural_model_info$has_shared_covariate_value_projection))
})

test_that("foundation bundles preserve covariate token metadata across save/load", {
  skip_on_cran()
  skip_if_no_jax()
  withr::local_envvar(c(STRATEGIZE_NEURAL_SKIP_EVAL = "1"))

  foundation_fit <- fit_conjoint_foundation_model(
    experiments = list(
      foundation_test_experiment(
        seed = 7041,
        experiment_id = "study_a",
        factor_names = c("price", "message"),
        x_names = c("income", "household size")
      ),
      foundation_test_experiment(
        seed = 7042,
        experiment_id = "study_b",
        factor_names = c("price", "message", "messenger"),
        x_names = c("income", "GOPScore")
      )
    ),
    foundation_control = foundation_test_control_with_embeddings()
  )

  tmp <- tempfile(fileext = ".rds")
  save_conjoint_foundation_bundle(tmp, foundation_fit, overwrite = TRUE)
  loaded <- load_conjoint_foundation_bundle(tmp, preload_params = FALSE)

  expect_s3_class(loaded, "conjoint_foundation_model")
  orig_group <- foundation_fit$groups[[foundation_universal_group_key()]]
  loaded_group <- loaded$groups[[foundation_universal_group_key()]]
  expect_identical(loaded_group$pairwise_context_mode, orig_group$pairwise_context_mode)
  expect_identical(loaded_group$x_schema$base_x_names, orig_group$x_schema$base_x_names)
  expect_identical(loaded_group$x_schema$semantic_feature_names, orig_group$x_schema$semantic_feature_names)
  expect_identical(loaded_group$x_schema$experiment_token_levels, orig_group$x_schema$experiment_token_levels)
  expect_identical(loaded_group$text_registry$x_feature_names, orig_group$text_registry$x_feature_names)
  expect_equal(loaded_group$text_registry$x_feature_embedding, orig_group$text_registry$x_feature_embedding)
  expect_identical(
    loaded_group$fit$neural_model_info$covariate_names,
    orig_group$fit$neural_model_info$covariate_names
  )
  expect_identical(
    loaded_group$fit$neural_model_info$token_family_levels,
    orig_group$fit$neural_model_info$token_family_levels
  )
  expect_equal(
    loaded_group$fit$neural_model_info$covariate_name_text,
    orig_group$fit$neural_model_info$covariate_name_text
  )

  adapt_data <- foundation_test_experiment(
    seed = 7043,
    experiment_id = "loaded_target",
    factor_names = c("price", "message"),
    x_names = c("income", "household size", "local_bonus")
  )
  predictor <- adapt_conjoint_foundation_model(
    foundation_model = loaded,
    Y = adapt_data$Y,
    W = adapt_data$W,
    X = adapt_data$X,
    mode = "pairwise",
    pair_id = adapt_data$pair_id,
    profile_order = adapt_data$profile_order,
    experiment_id = adapt_data$experiment_id,
    experiment_description = adapt_data$experiment_description,
    canonical_factor_id = adapt_data$canonical_factor_id,
    neural_mcmc_control = foundation_test_control()$neural_mcmc_control
  )
  preds <- predict(
    predictor,
    newdata = list(
      W = adapt_data$W,
      pair_id = adapt_data$pair_id,
      profile_order = adapt_data$profile_order
    )
  )
  tmp_pred <- tempfile(fileext = ".rds")
  save_strategic_predictor(predictor, tmp_pred, overwrite = TRUE)
  predictor_loaded <- load_strategic_predictor(tmp_pred)
  preds_loaded <- predict(
    predictor_loaded,
    newdata = list(
      W = adapt_data$W,
      X = adapt_data$X,
      pair_id = adapt_data$pair_id,
      profile_order = adapt_data$profile_order,
      experiment_id = adapt_data$experiment_id
    )
  )

  expect_true(all(is.finite(preds)))
  expect_true(all(is.finite(preds_loaded)))
})
