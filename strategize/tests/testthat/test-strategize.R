# =============================================================================
# Core strategize() Function Tests
# =============================================================================
# Tests for the main strategize() function.
# These tests require the conda environment with JAX.
# =============================================================================

test_that("strategize returns valid result with GLM outcome model", {
  skip_on_cran()
  skip_if_no_jax()

  data <- generate_test_data(n = 500, seed = 42)
  params <- default_strategize_params(fast = TRUE)

  res <- do.call(strategize, c(
    list(Y = data$Y, W = data$W),
    data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order")],
    params
  ))

  expect_valid_strategize_output(res, n_factors = ncol(data$W))
})

test_that("strategize handles K > 1 (multi-cluster)", {
  skip_on_cran()
  skip_if_no_jax()
  skip_if_no_tgp()

  data <- generate_test_data(n = 500, seed = 42)
  data <- add_respondent_covariates(data)
  params <- default_strategize_params(fast = TRUE)
  params$K <- 2

  res <- do.call(strategize, c(
    list(Y = data$Y, W = data$W, X = data$X),
    data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order")],
    params
  ))

  expect_valid_strategize_output(res)
  expect_equal(length(res$pi_star_point), 2)
  expect_true(all(c("k1", "k2") %in% names(res$pi_star_point)))
})

test_that("strategize handles diff = FALSE (non-difference mode)", {
  skip_on_cran()
  skip_if_no_jax()

  data <- generate_test_data(n = 500, seed = 42)
  params <- default_strategize_params(fast = TRUE)
  params$diff <- FALSE

  res <- do.call(strategize, c(
    list(Y = data$Y, W = data$W),
    data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order")],
    params
  ))

  expect_valid_strategize_output(res)
})

test_that("strategize handles use_regularization = FALSE", {
  skip_on_cran()
  skip_if_no_jax()

  data <- generate_test_data(n = 500, seed = 42)
  params <- default_strategize_params(fast = TRUE)

  res <- do.call(strategize, c(
    list(Y = data$Y, W = data$W, use_regularization = FALSE),
    data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order")],
    params
  ))

  expect_valid_strategize_output(res)
})

test_that("strategize computes standard errors when requested", {
  skip_on_cran()
  skip_if_no_jax()

  data <- generate_test_data(n = 500, seed = 42)
  params <- default_strategize_params(fast = TRUE)
  params$compute_se <- TRUE

  res <- do.call(strategize, c(
    list(Y = data$Y, W = data$W),
    data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order")],
    params
  ))

  expect_valid_strategize_output(res)
  expect_true("pi_star_se" %in% names(res))
  expect_true("Q_se" %in% names(res))
})

test_that("strategize validates Y and W dimensions", {
  skip_on_cran()
  skip_if_no_jax()

  data <- generate_test_data(n = 500, seed = 42)
  params <- default_strategize_params(fast = TRUE)

  # Use mismatched W dimension
  W_wrong <- data$W[1:100, ]

  expect_error(
    do.call(strategize, c(
      list(Y = data$Y, W = W_wrong),
      data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order")],
      params
    ))
  )
})

test_that("strategize returns valid probability distributions", {
  skip_on_cran()
  skip_if_no_jax()

  data <- generate_test_data(n = 500, seed = 42)
  params <- default_strategize_params(fast = TRUE)

  res <- do.call(strategize, c(
    list(Y = data$Y, W = data$W),
    data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order")],
    params
  ))

  # Check pi_star_point distributions
 for (k in seq_along(res$pi_star_point)) {
    pi_k <- res$pi_star_point[[k]]
    for (d in seq_along(pi_k)) {
      expect_valid_probability(pi_k[[d]])
    }
  }

  # Check p_list distributions
  for (d in seq_along(res$p_list)) {
    expect_valid_probability(res$p_list[[d]])
  }
})

test_that("strategize returns all expected output fields", {
  skip_on_cran()
  skip_if_no_jax()

  data <- generate_test_data(n = 500, seed = 42)
  params <- default_strategize_params(fast = TRUE)

  res <- do.call(strategize, c(
    list(Y = data$Y, W = data$W),
    data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order")],
    params
  ))

  expected_fields <- c("pi_star_point", "Q_point", "p_list")
  for (field in expected_fields) {
    expect_true(field %in% names(res), info = paste("Missing field:", field))
  }

  expect_type(res$pi_star_point, "list")
  expect_type(res$p_list, "list")
  expect_equal(length(res$p_list), ncol(data$W))
})
