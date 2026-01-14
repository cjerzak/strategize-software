# =============================================================================
# Primary push-forward tests
# =============================================================================

test_that("primary_pushforward toggles primary estimator in adversarial mode", {
  skip_on_cran()
  skip_if_no_jax()

  data <- generate_test_data(n = 200, seed = 123)
  data <- add_adversarial_structure(data, seed = 123)
  params <- default_strategize_params(fast = TRUE)

  res_mc <- strategize(
    Y = data$Y,
    W = data$W,
    lambda = params$lambda,
    pair_id = data$pair_id,
    respondent_id = data$respondent_id,
    respondent_task_id = data$respondent_task_id,
    profile_order = data$profile_order,
    competing_group_variable_respondent = data$competing_group_variable_respondent,
    competing_group_variable_candidate = data$competing_group_variable_candidate,
    competing_group_competition_variable_candidate = data$competing_group_competition_variable_candidate,
    adversarial = TRUE,
    diff = TRUE,
    nSGD = params$nSGD,
    outcome_model_type = params$outcome_model_type,
    force_gaussian = params$force_gaussian,
    nMonte_adversarial = params$nMonte_adversarial,
    nMonte_Qglm = params$nMonte_Qglm,
    primary_pushforward = "mc",
    compute_se = params$compute_se,
    conda_env = params$conda_env,
    conda_env_required = params$conda_env_required
  )

  res_linear <- strategize(
    Y = data$Y,
    W = data$W,
    lambda = params$lambda,
    pair_id = data$pair_id,
    respondent_id = data$respondent_id,
    respondent_task_id = data$respondent_task_id,
    profile_order = data$profile_order,
    competing_group_variable_respondent = data$competing_group_variable_respondent,
    competing_group_variable_candidate = data$competing_group_variable_candidate,
    competing_group_competition_variable_candidate = data$competing_group_competition_variable_candidate,
    adversarial = TRUE,
    diff = TRUE,
    nSGD = params$nSGD,
    outcome_model_type = params$outcome_model_type,
    force_gaussian = params$force_gaussian,
    nMonte_adversarial = params$nMonte_adversarial,
    nMonte_Qglm = params$nMonte_Qglm,
    primary_pushforward = "linearized",
    compute_se = params$compute_se,
    conda_env = params$conda_env,
    conda_env_required = params$conda_env_required
  )

  expect_equal(res_mc$convergence_history$primary_pushforward, "mc")
  expect_equal(res_linear$convergence_history$primary_pushforward, "linearized")
})
