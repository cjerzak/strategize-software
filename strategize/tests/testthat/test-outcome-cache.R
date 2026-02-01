test_that("GLM outcome cache reloads for adversarial four strategy", {
  skip_on_cran()
  skip_if_no_jax()

  td <- withr::local_tempdir()
  withr::local_dir(td)

  data <- generate_test_data(n = 40, n_factors = 2, n_levels = 2, seed = 20260205)
  pair_ids <- data$pair_id
  resp_party <- ifelse(pair_ids %% 2L == 1L, "PartyA", "PartyB")
  comp_type <- ifelse(pair_ids %% 4L %in% c(1L, 2L), "Same", "Different")

  data$competing_group_variable_respondent <- resp_party
  data$competing_group_competition_variable_candidate <- comp_type
  data$competing_group_variable_candidate <- ifelse(
    comp_type == "Same",
    resp_party,
    ifelse(data$profile_order == 1L, "PartyA", "PartyB")
  )

  p_list <- generate_test_p_list(data$W)

  params <- default_strategize_params(fast = TRUE)
  params$nSGD <- 2L
  params$nMonte_adversarial <- 2L
  params$nMonte_Qglm <- 2L
  params$outcome_model_type <- "glm"
  params$use_regularization <- FALSE

  key <- "glm_cache_four"

  do.call(strategize, c(
    list(Y = data$Y, W = data$W, p_list = p_list),
    data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order",
           "competing_group_variable_respondent",
           "competing_group_variable_candidate",
           "competing_group_competition_variable_candidate")],
    params,
    list(
      adversarial = TRUE,
      adversarial_model_strategy = "four",
      save_outcome_model = TRUE,
      presaved_outcome_model = FALSE,
      outcome_model_key = key
    )
  ))

  groups <- sort(unique(data$competing_group_variable_candidate))
  coef_path <- file.path("StrategizeInternals", sprintf("coef_%s_0_%s.rds", groups[1], key))
  expect_true(file.exists(coef_path))

  coef <- readRDS(coef_path)
  sentinel <- 123.456
  coef$intercept_base <- sentinel
  saveRDS(coef, coef_path)

  res <- do.call(strategize, c(
    list(Y = data$Y, W = data$W, p_list = p_list),
    data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order",
           "competing_group_variable_respondent",
           "competing_group_variable_candidate",
           "competing_group_competition_variable_candidate")],
    params,
    list(
      adversarial = TRUE,
      adversarial_model_strategy = "four",
      save_outcome_model = FALSE,
      presaved_outcome_model = TRUE,
      outcome_model_key = key
    )
  ))

  expect_true(!is.null(res$REGRESSION_PARAMETERS_ast0))
  params_vec <- as.numeric(res$strenv$np$array(res$REGRESSION_PARAMETERS_ast0))
  expect_equal(params_vec[1], sentinel, tolerance = 1e-6)
})

test_that("neural outcome cache reloads from bundle", {
  skip_on_cran()
  skip_if_no_jax()

  td <- withr::local_tempdir()
  withr::local_dir(td)

  withr::local_envvar(c(
    STRATEGIZE_NEURAL_FAST_MCMC = "true",
    STRATEGIZE_NEURAL_EVAL_FOLDS = "2",
    STRATEGIZE_NEURAL_EVAL_SEED = "123"
  ))

  data <- generate_test_data(n = 24, n_factors = 2, n_levels = 2, seed = 20260206)
  p_list <- generate_test_p_list(data$W)

  params <- default_strategize_params(fast = TRUE)
  params$nSGD <- 2L
  params$nMonte_Qglm <- 2L
  params$outcome_model_type <- "neural"
  params$neural_mcmc_control <- modifyList(
    params$neural_mcmc_control,
    list(n_samples_warmup = 2L, n_samples_mcmc = 2L, ModelDims = 8L, ModelDepth = 1L)
  )

  key <- "neural_cache_test"

  do.call(strategize, c(
    list(Y = data$Y, W = data$W, p_list = p_list),
    data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order")],
    params,
    list(
      save_outcome_model = TRUE,
      presaved_outcome_model = FALSE,
      outcome_model_key = key
    )
  ))

  bundle_path <- file.path("StrategizeInternals", sprintf("neural_bundle_1_1_%s.rds", key))
  expect_true(file.exists(bundle_path))

  bundle <- readRDS(bundle_path)
  sentinel <- 123.456
  if (length(bundle$fit$theta_mean) > 0) {
    bundle$fit$theta_mean[1] <- sentinel
  }
  if (!is.null(bundle$theta_mean) && length(bundle$theta_mean) > 0) {
    bundle$theta_mean[1] <- sentinel
  }
  saveRDS(bundle, bundle_path)

  res <- do.call(strategize, c(
    list(Y = data$Y, W = data$W, p_list = p_list),
    data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order")],
    params,
    list(
      save_outcome_model = FALSE,
      presaved_outcome_model = TRUE,
      outcome_model_key = key
    )
  ))

  params_vec <- as.numeric(res$strenv$np$array(res$REGRESSION_PARAMETERS_ast))
  expect_true(length(params_vec) >= 2)
  expect_equal(params_vec[2], sentinel, tolerance = 1e-6)
})
