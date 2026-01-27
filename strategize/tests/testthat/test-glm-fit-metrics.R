test_that("GLM outcome model exports overall fit metrics (pairwise)", {
  dat <- generate_test_data(n = 200, n_factors = 3, n_levels = 2, seed = 20260127)

  predictor <- strategic_prediction(
    Y = dat$Y,
    W = dat$W,
    model = "glm",
    mode = "pairwise",
    pair_id = dat$pair_id,
    profile_order = dat$profile_order
  )

  metrics <- predictor$fit$fit_metrics
  expect_type(metrics, "list")
  expect_identical(metrics$likelihood, "binomial")
  expect_equal(metrics$n_eval, length(dat$Y) / 2)

  expect_true(is.finite(metrics$auc))
  expect_gte(metrics$auc, 0)
  expect_lte(metrics$auc, 1)

  expect_true(is.finite(metrics$accuracy))
  expect_gte(metrics$accuracy, 0)
  expect_lte(metrics$accuracy, 1)

  expect_true(is.finite(metrics$log_loss))
  expect_gte(metrics$log_loss, 0)

  expect_true(is.finite(metrics$brier))
  expect_gte(metrics$brier, 0)
  expect_lte(metrics$brier, 1)
})

