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

test_that("fit_conjoint_foundation_model pools compatible pairwise studies", {
  skip_on_cran()
  skip_if_no_jax()
  withr::local_envvar(c(STRATEGIZE_NEURAL_SKIP_EVAL = "1"))

  study_a <- generate_test_data(n = 40, n_factors = 2, n_levels = 2, seed = 7001)
  study_b <- generate_test_data(n = 40, n_factors = 3, n_levels = 2, seed = 7002)
  colnames(study_a$W) <- c("price", "message")
  colnames(study_b$W) <- c("price", "message", "messenger")

  fit <- fit_conjoint_foundation_model(
    experiments = list(
      list(
        experiment_id = "study_a",
        Y = study_a$Y,
        W = study_a$W,
        pair_id = study_a$pair_id,
        profile_order = study_a$profile_order,
        canonical_factor_id = c(price = "price", message = "message")
      ),
      list(
        experiment_id = "study_b",
        Y = study_b$Y,
        W = study_b$W,
        pair_id = study_b$pair_id,
        profile_order = study_b$profile_order,
        canonical_factor_id = c(price = "price", message = "message", messenger = "messenger")
      )
    ),
    foundation_control = foundation_test_control()
  )

  expect_s3_class(fit, "conjoint_foundation_model")
  expect_length(fit$groups, 1L)
  expect_true("pairwise::bernoulli::1" %in% names(fit$groups))
  expect_true(length(fit$groups[[1]]$encoder$factor_names) >= 3L)
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

test_that("adapt_conjoint_foundation_model returns a predictor and foundation bundles round-trip", {
  skip_on_cran()
  skip_if_no_jax()
  withr::local_envvar(c(STRATEGIZE_NEURAL_SKIP_EVAL = "1"))

  foundation_data <- generate_test_data(n = 40, n_factors = 2, n_levels = 2, seed = 7021)
  colnames(foundation_data$W) <- c("price", "message")
  foundation_fit <- fit_conjoint_foundation_model(
    experiments = list(
      list(
        experiment_id = "study_a",
        Y = foundation_data$Y,
        W = foundation_data$W,
        pair_id = foundation_data$pair_id,
        profile_order = foundation_data$profile_order,
        canonical_factor_id = c(price = "price", message = "message")
      ),
      list(
        experiment_id = "study_b",
        Y = foundation_data$Y,
        W = foundation_data$W,
        pair_id = foundation_data$pair_id,
        profile_order = foundation_data$profile_order,
        canonical_factor_id = c(price = "price", message = "message")
      )
    ),
    foundation_control = foundation_test_control()
  )

  adapt_data <- generate_test_data(n = 40, n_factors = 2, n_levels = 2, seed = 7022)
  colnames(adapt_data$W) <- c("price", "message")
  predictor <- adapt_conjoint_foundation_model(
    foundation_model = foundation_fit,
    Y = adapt_data$Y,
    W = adapt_data$W,
    mode = "pairwise",
    pair_id = adapt_data$pair_id,
    profile_order = adapt_data$profile_order,
    experiment_id = "study_a",
    canonical_factor_id = c(price = "price", message = "message"),
    neural_mcmc_control = foundation_test_control()$neural_mcmc_control
  )

  expect_s3_class(predictor, "strategic_predictor")
  expect_identical(predictor$model_type, "neural")
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

  tmp <- tempfile(fileext = ".rds")
  save_conjoint_foundation_bundle(tmp, foundation_fit, overwrite = TRUE)
  loaded <- load_conjoint_foundation_bundle(tmp, preload_params = FALSE)
  expect_s3_class(loaded, "conjoint_foundation_model")
  expect_equal(names(loaded$groups), names(foundation_fit$groups))
})
