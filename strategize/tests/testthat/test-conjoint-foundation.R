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
  expect_true(length(group$x_feature_names) > 0L)
  expect_identical(
    group$text_registry$x_feature_names,
    group$x_schema$base_x_names
  )
  expect_equal(
    nrow(group$text_registry$x_feature_embedding),
    length(group$x_schema$base_x_names)
  )
  expect_true(any(grepl("^semantic_factor_", group$x_feature_names)))
  expect_true(any(grepl("^semantic_level_", group$x_feature_names)))
  expect_true(any(grepl("^semantic_x_", group$x_feature_names)))
  expect_equal(
    sum(grepl("^semantic_x_", group$x_feature_names)),
    group$text_registry$dim
  )

  experiments_norm <- list(
    cs_foundation_normalize_experiment(study_a, index = 1L),
    cs_foundation_normalize_experiment(study_b, index = 2L)
  )
  registry <- cs_foundation_build_group_registry(experiments_norm)
  pooled <- cs_foundation_build_group_training_data(experiments_norm, registry, control)
  semantic_x_cols <- grep("^semantic_x_", colnames(pooled$X), value = TRUE)

  expect_true(length(semantic_x_cols) > 0L)
  expect_true(any(abs(as.matrix(pooled$X[, semantic_x_cols, drop = FALSE])) > 0))
  expect_true(any(grepl("^semantic_factor_", colnames(pooled$X))))
  expect_true(any(grepl("^semantic_level_", colnames(pooled$X))))
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

test_that("adapt_conjoint_foundation_model rebuilds shared X semantics and predictions stay finite", {
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

  expect_true("local_bonus" %in% colnames(X_aug))
  expect_equal(
    sum(grepl("^semantic_x_", colnames(X_aug))),
    group$text_registry$dim
  )
  expect_false("local_bonus" %in% group$x_schema$base_x_names)

  predictor <- adapt_conjoint_foundation_model(
    foundation_model = foundation_fit,
    Y = adapt_data$Y,
    W = adapt_data$W,
    X = adapt_data$X,
    mode = "pairwise",
    pair_id = adapt_data$pair_id,
    profile_order = adapt_data$profile_order,
    experiment_id = adapt_data$experiment_id,
    canonical_factor_id = adapt_data$canonical_factor_id,
    neural_mcmc_control = foundation_test_control()$neural_mcmc_control,
    foundation_adaptation_control = list(
      text_embedding_fn = foundation_test_text_embedding_fn
    )
  )

  expect_s3_class(predictor, "strategic_predictor")
  preds <- predict(
    predictor,
    newdata = list(
      W = adapt_data$W,
      pair_id = adapt_data$pair_id,
      profile_order = adapt_data$profile_order
    )
  )
  expect_true(all(is.finite(preds)))
  expect_true(all(preds >= 0 & preds <= 1))
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
  expect_identical(group$x_schema$semantic_feature_names, character(0))
  expect_false(any(grepl("^semantic_x_", group$x_feature_names)))
  expect_false(any(grepl("^semantic_factor_", group$x_feature_names)))
  expect_false(any(grepl("^semantic_level_", group$x_feature_names)))
  expect_identical(
    group$x_feature_names,
    c(group$x_schema$base_x_names, group$x_schema$experiment_indicator_names)
  )
})

test_that("foundation bundles preserve X semantic metadata across save/load", {
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
  expect_identical(loaded_group$text_registry$x_feature_names, orig_group$text_registry$x_feature_names)
  expect_equal(loaded_group$text_registry$x_feature_embedding, orig_group$text_registry$x_feature_embedding)

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

  expect_true(all(is.finite(preds)))
})
