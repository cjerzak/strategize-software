test_that("optimistic updates require non-optax backend in strategize", {
  Y <- c(1, 0)
  W <- data.frame(a = c("x", "y"), b = c("u", "v"))

  expect_error(
    strategize(
      Y = Y,
      W = W,
      lambda = 0.1,
      nSGD = 1,
      compute_se = FALSE,
      use_optax = TRUE,
      optimism = "ogda"
    ),
    "only available",
    fixed = FALSE
  )
})

test_that("optimism argument is validated early", {
  Y <- c(1, 0)
  W <- data.frame(a = c("x", "y"), b = c("u", "v"))

  expect_error(
    strategize(
      Y = Y,
      W = W,
      lambda = 0.1,
      nSGD = 1,
      compute_se = FALSE,
      optimism = "not-a-valid-option"
    ),
    "should be one of",
    fixed = FALSE
  )
})

test_that("cv_strategize enforces optimism compatibility before JAX init", {
  Y <- c(1, 0)
  W <- data.frame(a = c("x", "y"), b = c("u", "v"))

  expect_error(
    cv_strategize(
      Y = Y,
      W = W,
      lambda_seq = 0.1,
      folds = 2,
      respondent_id = c(1, 1),
      optimism = "extragrad",
      use_optax = TRUE,
      nSGD = 1
    ),
    "only available",
    fixed = FALSE
  )
})
