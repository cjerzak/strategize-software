# =============================================================================
# Cross-Validation (cv_strategize) Tests
# =============================================================================
# Tests for the cv_strategize() function which performs cross-validation
# for lambda selection.
# =============================================================================

test_that("cv_strategize selects lambda with single value", {
  skip_on_cran()
  skip_if_no_jax()

  data <- generate_test_data(n = 500, seed = 42)
  params <- default_strategize_params(fast = TRUE)

  res <- do.call(cv_strategize, c(
    list(Y = data$Y, W = data$W),
    data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order")],
    params
  ))

  expect_type(res, "list")
  expect_true("lambda" %in% names(res))
  expect_true("CVInfo" %in% names(res))
  expect_true("pi_star_point" %in% names(res))
  expect_true("p_list" %in% names(res))
})

test_that("cv_strategize handles vector of lambda values", {
  skip_on_cran()
  skip_if_no_jax()

  data <- generate_test_data(n = 500, seed = 42)
  params <- default_strategize_params(fast = TRUE)
  params$lambda <- c(0.01, 0.1, 1.0)

  res <- do.call(cv_strategize, c(
    list(Y = data$Y, W = data$W),
    data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order")],
    params
  ))

  expect_type(res, "list")
  expect_true("lambda" %in% names(res))
  expect_true("CVInfo" %in% names(res))
})

test_that("cv_strategize handles K > 1 (multi-cluster)", {
  skip_on_cran()
  skip_if_no_jax()
  skip_if_no_tgp()

  data <- generate_test_data(n = 500, seed = 42)
  data <- add_respondent_covariates(data)
  params <- default_strategize_params(fast = TRUE)
  params$K <- 2

  res <- do.call(cv_strategize, c(
    list(Y = data$Y, W = data$W, X = data$X),
    data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order")],
    params
  ))

  expect_type(res, "list")
  expect_true("lambda" %in% names(res))
  expect_true("pi_star_point" %in% names(res))
  expect_equal(length(res$pi_star_point), 2)
})

test_that("cv_strategize returns expected output structure", {
  skip_on_cran()
  skip_if_no_jax()

  data <- generate_test_data(n = 500, seed = 42)
  params <- default_strategize_params(fast = TRUE)

  res <- do.call(cv_strategize, c(
    list(Y = data$Y, W = data$W),
    data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order")],
    params
  ))

  expected_fields <- c("lambda", "CVInfo", "pi_star_point", "p_list")
  for (field in expected_fields) {
    expect_true(field %in% names(res), info = paste("Missing field:", field))
  }
})
