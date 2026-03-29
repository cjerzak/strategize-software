# =============================================================================
# Implicit standard error approximation tests
# =============================================================================

local_force_no_interactions <- function(value = TRUE) {
  old_exists <- exists(
    "force_no_interactions",
    envir = .GlobalEnv,
    inherits = FALSE
  )
  if (old_exists) {
    old_value <- get(
      "force_no_interactions",
      envir = .GlobalEnv,
      inherits = FALSE
    )
  }

  assign("force_no_interactions", value, envir = .GlobalEnv)

  withr::defer({
    if (old_exists) {
      assign("force_no_interactions", old_value, envir = .GlobalEnv)
    } else if (exists("force_no_interactions", envir = .GlobalEnv, inherits = FALSE)) {
      rm("force_no_interactions", envir = .GlobalEnv)
    }
  })
}

run_se_method_comparison <- function(adversarial = TRUE) {
  skip_on_cran()
  skip_if_no_jax()

  local_force_no_interactions(TRUE)

  data <- generate_test_data(n = 120, n_factors = 2, seed = 123)
  data <- add_adversarial_structure(data, seed = 456)

  params <- default_strategize_params(fast = TRUE)
  params$lambda <- 0.05
  params$nSGD <- 15L
  params$nMonte_adversarial <- 3L
  params$nMonte_Qglm <- 5L
  params$compute_se <- TRUE
  params$primary_pushforward <- "linearized"
  params$a_init_sd <- 0.0
  params$compute_hessian <- FALSE
  params$optimism <- "none"

  args <- c(
    list(
      Y = data$Y,
      W = data$W,
      pair_id = data$pair_id,
      respondent_id = data$respondent_id,
      respondent_task_id = data$respondent_task_id,
      profile_order = data$profile_order,
      adversarial = adversarial,
      competing_group_variable_respondent = data$competing_group_variable_respondent,
      competing_group_variable_candidate = data$competing_group_variable_candidate,
      competing_group_competition_variable_candidate =
        data$competing_group_competition_variable_candidate
    ),
    params
  )

  set.seed(999)
  res_full <- do.call(strategize, c(args, list(se_method = "full")))
  set.seed(999)
  res_impl <- do.call(strategize, c(args, list(se_method = "implicit")))

  list(full = res_full, implicit = res_impl)
}

expect_se_methods_close <- function(results,
                                    mean_tolerance,
                                    median_tolerance,
                                    max_tolerance,
                                    q_tolerance) {
  full_se <- results$full$pi_star_se_vec
  impl_se <- results$implicit$pi_star_se_vec
  keep <- is.finite(full_se) & is.finite(impl_se)
  diffs <- abs(full_se[keep] - impl_se[keep])

  expect_true(any(keep))
  expect_lt(mean(diffs), mean_tolerance)
  expect_lt(stats::median(diffs), median_tolerance)
  expect_lt(max(diffs), max_tolerance)
  expect_true(is.finite(results$full$Q_se))
  expect_true(is.finite(results$implicit$Q_se))
  expect_lt(abs(results$full$Q_se - results$implicit$Q_se), q_tolerance)
}

test_that("REINFORCE diagnostics are tracer-safe under jacrev", {
  skip_on_cran()
  skip_if_no_jax()

  if (!"jnp" %in% ls(envir = strategize:::strenv)) {
    strategize:::initialize_jax(conda_env = "strategize_env", conda_env_required = TRUE)
  }

  x0 <- strategize:::strenv$jnp$array(c(1, 2), dtype = strategize:::strenv$dtj)
  jacobian <- tryCatch(
    strategize:::strenv$jax$jacrev(function(x) {
      diag <- strategize:::build_reinforce_diag(
        reward_mean = x[[1L]],
        reward_var = x[[1L]] * x[[1L]],
        baseline_prev = x[[1L]],
        grad_total = x
      )
      strategize:::strenv$jnp$add(diag$reward_mean, diag$reward_var)
    })(x0),
    error = identity
  )

  expect_false(
    inherits(jacobian, "error"),
    info = if (inherits(jacobian, "error")) conditionMessage(jacobian) else NULL
  )
  if (!inherits(jacobian, "error")) {
    expect_true(all(is.finite(as.numeric(strategize:::strenv$np$array(jacobian)))))
  }
})

test_that("implicit SEs match full-trace SEs in adversarial mode", {
  results <- run_se_method_comparison(adversarial = TRUE)
  expect_se_methods_close(
    results,
    mean_tolerance = 0.1,
    median_tolerance = 0.05,
    max_tolerance = 0.25,
    q_tolerance = 0.05
  )
})

test_that("implicit SEs match full-trace SEs in non-adversarial mode", {
  results <- run_se_method_comparison(adversarial = FALSE)
  expect_se_methods_close(
    results,
    mean_tolerance = 0.01,
    median_tolerance = 0.005,
    max_tolerance = 0.01,
    q_tolerance = 0.01
  )
})

test_that("implicit SEs complete for large-support neural average-case fits", {
  skip_on_cran()
  skip_if_no_jax()

  withr::local_envvar(c(
    STRATEGIZE_NEURAL_FAST_MCMC = "true",
    STRATEGIZE_NEURAL_EVAL_FOLDS = "2",
    STRATEGIZE_NEURAL_EVAL_SEED = "123"
  ))

  fixture <- generate_linear_average_case_fixture(
    n_obs = 120L,
    k_factors = 12L,
    monte_i = 1L
  )
  params <- default_strategize_params(fast = TRUE)
  params$lambda <- fixture$lambda
  params$nSGD <- 15L
  params$nMonte_Qglm <- 5L
  params$outcome_model_type <- "neural"
  params$diff <- FALSE
  params$adversarial <- FALSE
  params$compute_se <- TRUE
  params$se_method <- "implicit"
  params$penalty_type <- "L2"
  params$use_regularization <- FALSE
  params$use_optax <- FALSE
  params$force_gaussian <- FALSE
  params$a_init_sd <- 0.0
  params$primary_pushforward <- "linearized"
  params$compute_hessian <- FALSE
  params$optimism <- "none"
  params$neural_mcmc_control <- modifyList(
    params$neural_mcmc_control,
    list(
      subsample_method = "batch_vi",
      batch_size = 16L,
      ModelDims = 8L,
      ModelDepth = 1L,
      vi_guide = "auto_diagonal",
      uncertainty_scope = "output",
      eval_enabled = FALSE,
      warn_stage_imbalance_pct = 0,
      warn_min_cell_n = 0L
    )
  )

  withr::local_seed(20260326L)
  res <- suppressWarnings(do.call(strategize, c(
    list(Y = fixture$Y, W = fixture$W),
    params
  )))

  info_msg <- sprintf(
    paste0(
      "objective_gradient_mode=%s; Q_se=%s; finite_pi_se=%d; ",
      "reinforce_nonfinite_ast_steps=%d"
    ),
    res$convergence_history$objective_gradient_mode,
    format(res$Q_se, digits = 6),
    sum(is.finite(res$pi_star_se_vec)),
    as.integer(res$convergence_history$reinforce_nonfinite_ast_steps)
  )

  expect_identical(res$convergence_history$objective_gradient_mode, "reinforce", info = info_msg)
  expect_true(is.finite(res$Q_se), info = info_msg)
  expect_true(any(is.finite(res$pi_star_se_vec)), info = info_msg)
  expect_identical(as.integer(res$convergence_history$reinforce_nonfinite_ast_steps), 0L, info = info_msg)
})
