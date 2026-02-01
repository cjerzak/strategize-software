test_that("strategic predictors can be saved and loaded (GLM)", {
  dat <- generate_test_data(n = 300, n_factors = 3, n_levels = 2, seed = 20260201)

  fit <- strategic_prediction(
    Y = dat$Y,
    W = dat$W,
    model = "glm",
    mode = "single",
    use_regularization = FALSE
  )

  tmp <- tempfile(fileext = ".rds")
  save_strategic_predictor(fit, tmp, overwrite = TRUE)
  fit_loaded <- load_strategic_predictor(tmp)

  preds1 <- predict(fit, newdata = dat$W)
  preds2 <- predict(fit_loaded, newdata = dat$W)

  expect_equal(preds1, preds2, tolerance = 1e-8)
  expect_true(is.list(fit_loaded$fit$fit_metrics))
})

test_that("strategic_prediction() can reuse cached predictors", {
  dat <- generate_test_data(n = 200, n_factors = 2, n_levels = 2, seed = 20260202)

  tmp <- tempfile(fileext = ".rds")
  fit_cached <- strategic_prediction(
    Y = dat$Y,
    W = dat$W,
    model = "glm",
    mode = "single",
    use_regularization = FALSE,
    cache_path = tmp,
    cache_overwrite = TRUE
  )

  fit_reused <- strategic_prediction(
    Y = dat$Y,
    W = dat$W,
    model = "glm",
    mode = "single",
    use_regularization = FALSE,
    cache_path = tmp,
    cache_overwrite = FALSE
  )

  preds_cached <- predict(fit_cached, newdata = dat$W)
  preds_reused <- predict(fit_reused, newdata = dat$W)

  expect_equal(preds_cached, preds_reused, tolerance = 1e-8)
})
