test_that("validate_equilibrium validates nMonte and seed", {
  skip_on_cran()

  expect_error(validate_equilibrium(list(), nMonte = 0), "nMonte must be")
  expect_error(validate_equilibrium(list(), seed = -1), "seed must be")
})

test_that("validate_equilibrium errors on missing Q function", {
  skip_on_cran()

  mock_result <- list(
    convergence_history = list(adversarial = TRUE),
    FullGetQStar_ = NULL,
    strenv = list(jnp = 1)
  )

  expect_error(validate_equilibrium(mock_result), "Result does not contain Q function")
})

test_that("validate_equilibrium errors when JAX is unavailable", {
  skip_on_cran()

  mock_result <- list(
    convergence_history = list(adversarial = TRUE),
    FullGetQStar_ = function() NULL,
    strenv = list()
  )

  expect_error(validate_equilibrium(mock_result), "JAX environment not available")
})
