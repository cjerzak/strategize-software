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

foundation_test_control_with_embeddings <- function() {
  modifyList(
    cs_foundation_default_control(),
    modifyList(
      foundation_test_control(),
      list(text_embedding_fn = foundation_test_text_embedding_fn)
    )
  )
}

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
  group <- fit$groups[["pairwise::bernoulli::1"]]
  expect_false(is.null(group))
  expect_identical(
    group$x_feature_names,
    c("income", "household size", "GOPScore")
  )
  expect_identical(group$token_control$experiment_token_mode, "description")
  expect_identical(group$token_control$covariate_value_encoding, "shared_projection")
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
  expect_true(isTRUE(group$fit$neural_model_info$has_token_family_embedding))
  expect_true(isTRUE(group$fit$neural_model_info$has_experiment_token))
  expect_false(isTRUE(group$fit$neural_model_info$has_experiment_id_embedding))
  expect_true(isTRUE(group$fit$neural_model_info$has_experiment_text_projection))
  expect_true(isTRUE(group$fit$neural_model_info$has_factor_name_text))
  expect_true(isTRUE(group$fit$neural_model_info$has_level_name_text))
  expect_true(isTRUE(group$fit$neural_model_info$has_covariate_name_text))
  expect_true(isTRUE(group$fit$neural_model_info$has_shared_covariate_value_projection))
  expect_length(
    as.numeric(strategize:::cs2step_neural_to_r_array(group$fit$neural_model_info$resp_cov_scale)),
    length(group$x_schema$base_x_names)
  )
  expect_true(all(
    c("factor_candidate", "covariate", "experiment", "stage",
      "resp_party", "matchup", "choice", "separator") %in%
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
  expect_true(all(pooled$X_present[seq_len(nrow(study_a$W)), "GOPScore"] == 0))
  expect_true(all(
    pooled$X_present[-seq_len(nrow(study_a$W)), "household size"] == 0
  ))
  expect_true(any(pooled$X_present[, "income"] == 1))
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
  expect_equal(sort(names(fit$groups)), sort(c("pairwise::bernoulli::1", "single::normal::1")))
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
  group <- foundation_fit$groups[["pairwise::bernoulli::1"]]
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

  group <- fit$groups[["pairwise::bernoulli::1"]]
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

  group <- fit$groups[["pairwise::bernoulli::1"]]
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
  orig_group <- foundation_fit$groups[["pairwise::bernoulli::1"]]
  loaded_group <- loaded$groups[["pairwise::bernoulli::1"]]
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
