# =============================================================================
# Neural Outcome Model Tests
# =============================================================================

test_that("strategize runs neural outcome model (non-adversarial)", {
  skip_on_cran()
  skip_if_no_jax()

  withr::local_envvar(c(STRATEGIZE_NEURAL_FAST_MCMC = "true"))

  data <- generate_test_data(n = 40, seed = 123)
  params <- default_strategize_params(fast = TRUE)
  params$outcome_model_type <- "neural"

  p_list <- generate_test_p_list(data$W)

  res <- do.call(strategize, c(
    list(Y = data$Y, W = data$W, p_list = p_list),
    data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order")],
    params
  ))

  expect_valid_strategize_output(res, n_factors = ncol(data$W))
})
