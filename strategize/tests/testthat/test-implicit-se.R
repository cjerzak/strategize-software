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
