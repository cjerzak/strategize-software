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

test_that("extragrad uses joint look-ahead for both players", {
  skip_if_no_jax()
  withr::local_seed(123)

  base <- generate_test_data(n = 50, n_factors = 2, n_levels = 2, seed = 321)
  adv_data <- add_adversarial_structure(base, seed = 222)
  p_list <- generate_test_p_list(adv_data$W)

  result <- strategize(
    Y = adv_data$Y,
    W = adv_data$W,
    p_list = p_list,
    lambda = 0.1,
    nSGD = 3,
    nMonte_Qglm = 5L,
    nMonte_adversarial = 3L,
    diff = TRUE,
    adversarial = TRUE,
    compute_se = FALSE,
    pair_id = adv_data$pair_id,
    respondent_id = adv_data$respondent_id,
    respondent_task_id = adv_data$respondent_task_id,
    profile_order = adv_data$profile_order,
    competing_group_variable_respondent = adv_data$competing_group_variable_respondent,
    competing_group_variable_candidate = adv_data$competing_group_variable_candidate,
    competing_group_competition_variable_candidate = adv_data$competing_group_competition_variable_candidate,
    optimism = "extragrad"
  )

  eval_points <- result$strenv$extragrad_eval_points
  non_null <- Filter(Negate(is.null), eval_points)
  expect_true(length(non_null) > 0)

  for (pt in non_null) {
    ast_pred <- result$strenv$np$array(pt$ast$a_pred_ast)
    dag_pred <- result$strenv$np$array(pt$dag$a_pred_ast)
    expect_equal(ast_pred, dag_pred, tolerance = 1e-6)
  }

  moved <- vapply(non_null, function(pt) {
    start_ast <- result$strenv$np$array(pt$start$a_ast)
    ast_pred <- result$strenv$np$array(pt$ast$a_pred_ast)
    any(abs(start_ast - ast_pred) > 1e-8)
  }, logical(1))
  expect_true(any(moved))
})
