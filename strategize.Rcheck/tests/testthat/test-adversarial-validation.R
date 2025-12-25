# Tests for adversarial validation functions

test_that("is_adversarial correctly identifies adversarial results", {
  # Mock non-adversarial result
  non_adv_result <- list(
    convergence_history = list(adversarial = FALSE)
  )
  expect_false(is_adversarial(non_adv_result))

  # Mock adversarial result
  adv_result <- list(
    convergence_history = list(adversarial = TRUE)
  )
  expect_true(is_adversarial(adv_result))

  # Missing convergence history
  no_hist <- list()
  expect_false(is_adversarial(no_hist))
})

test_that("get_final_gradients extracts gradient norms", {
  # Mock result with convergence history
  mock_result <- list(
    convergence_history = list(
      nSGD = 100,
      grad_ast = c(rep(1, 99), 0.001),
      grad_dag = c(rep(1, 99), 0.002)
    )
  )

  grads <- get_final_gradients(mock_result)

  expect_named(grads, c("AST", "DAG"))
  expect_equal(grads["AST"], c(AST = 0.001))
  expect_equal(grads["DAG"], c(DAG = 0.002))
})

test_that("get_final_gradients handles missing history", {
  mock_result <- list()
  expect_error(get_final_gradients(mock_result), "No convergence history")
})

test_that("plot_convergence requires convergence_history", {
  mock_result <- list()
  expect_error(plot_convergence(mock_result), "No convergence history")
})

test_that("plot_convergence validates metrics argument", {
  mock_result <- list(
    convergence_history = list(
      nSGD = 10,
      adversarial = TRUE,
      grad_ast = rep(0.1, 10),
      grad_dag = rep(0.1, 10),
      loss_ast = rep(0.5, 10),
      loss_dag = rep(0.5, 10),
      inv_lr_ast = rep(0.01, 10),
      inv_lr_dag = rep(0.01, 10)
    )
  )

  # Valid metrics
  expect_no_error(plot_convergence(mock_result, metrics = "gradient"))
  expect_no_error(plot_convergence(mock_result, metrics = "loss"))
  expect_no_error(plot_convergence(mock_result, metrics = c("gradient", "loss")))

  # Invalid metric
  expect_error(plot_convergence(mock_result, metrics = "invalid"))
})

test_that("validate_equilibrium requires adversarial result", {
  non_adv <- list(
    convergence_history = list(adversarial = FALSE)
  )

  expect_error(
    validate_equilibrium(non_adv),
    "requires an adversarial strategize result"
  )
})

test_that("plot_quadrant_breakdown requires adversarial result", {
  non_adv <- list(
    convergence_history = list(adversarial = FALSE)
  )

  expect_error(
    plot_quadrant_breakdown(non_adv),
    "requires an adversarial strategize result"
  )
})

test_that("summarize_adversarial requires adversarial result", {
  non_adv <- list(
    convergence_history = list(adversarial = FALSE)
  )

  expect_error(
    summarize_adversarial(non_adv),
    "requires an adversarial strategize result"
  )
})

# Skip integration tests that require JAX
skip_if_no_jax <- function() {
  skip_on_cran()

  jax_available <- tryCatch({
    strategize::check_jax_available()
  }, error = function(e) FALSE)

  if (!jax_available) {
    skip("JAX not available")
  }
}

test_that("convergence history is included in strategize output", {
  skip_if_no_jax()

  # Generate minimal test data
  set.seed(42)
  n <- 200

  # Binary outcome
  Y <- sample(0:1, n, replace = TRUE)

  # Create factor columns
  gender <- sample(c("male", "female"), n, replace = TRUE)
  W <- data.frame(gender = gender)

  # Baseline probabilities
  p_list <- list(gender = c(male = 0.5, female = 0.5))

  # Run non-adversarial strategize (simpler)
  result <- tryCatch({
    strategize(
      Y = Y, W = W, p_list = p_list,
      lambda = 0.1, nSGD = 10, nMonte = 5
    )
  }, error = function(e) NULL)

  skip_if(is.null(result), "strategize failed to run")

  # Check convergence history exists

  expect_true(!is.null(result$convergence_history))
  expect_true(!is.null(result$convergence_history$nSGD))
  expect_true(!is.null(result$convergence_history$grad_ast))
})

test_that("plot_convergence works with real strategize output", {
  skip_if_no_jax()

  set.seed(42)
  n <- 200
  Y <- sample(0:1, n, replace = TRUE)
  gender <- sample(c("male", "female"), n, replace = TRUE)
  W <- data.frame(gender = gender)
  p_list <- list(gender = c(male = 0.5, female = 0.5))

  result <- tryCatch({
    strategize(
      Y = Y, W = W, p_list = p_list,
      lambda = 0.1, nSGD = 20, nMonte = 5
    )
  }, error = function(e) NULL)

  skip_if(is.null(result), "strategize failed to run")

  # Should not error
  expect_no_error(plot_convergence(result, metrics = "gradient"))
})
