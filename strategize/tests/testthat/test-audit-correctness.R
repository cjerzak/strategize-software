test_that("the screen recovers an orientation-balanced profile interaction", {
  withr::local_seed(910)
  profiles <- expand.grid(A = 0:1, B = 0:1, C = 0:1)
  pairs <- expand.grid(a = seq_len(8), b = seq_len(8), replicate = seq_len(40))
  a <- profiles[pairs$a, ]; b <- profiles[pairs$b, ]
  utility <- function(w) 2 * (2 * w$A - 1) * (2 * w$B - 1)
  y <- rbinom(nrow(pairs), 1, plogis(utility(a) - utility(b)))
  main <- as.matrix(a - b)
  interactions <- cbind(a$A * a$B - b$A * b$B,
                         a$A * a$C - b$A * b$C,
                         a$B * a$C - b$B * b$C)
  selected <- cs_glm_screen_features(main, interactions, y, "binomial", intercept = FALSE)
  expect_true(1L %in% selected)
  reversed <- cs_glm_screen_features(-main, -interactions, 1 - y, "binomial", intercept = FALSE)
  expect_true(1L %in% reversed)
})

test_that("adaptive candidates use training data and the first feasible penalty", {
  p <- list(A = c(a = .5, b = .5))
  control <- cs_crossfit_q_default_control(list(adaptive_lambda = TRUE,
    lambda_path = c(.01, .1, 1), design_abs_ess_min = 10, design_max_weight = 1.5))
  calls <- list()
  fit <- function(Y, p_list, lambda) {
    calls[[length(calls) + 1L]] <<- list(Y = Y, lambda = lambda)
    q <- if (lambda < .1) c(a = .99, b = .01) else c(a = .6, b = .4)
    list(pi_star_point = list(k1 = list(A = q)))
  }
  selected <- cs_crossfit_select_policy(list(Y = c(0, 1), p_list = p, lambda = .01),
                                        control, fold = 1L, n_oriented = 100, fit = fit)
  expect_equal(vapply(calls, `[[`, numeric(1), "lambda"), c(.01, .1))
  expect_true(all(vapply(calls, function(x) identical(x$Y, c(0, 1)), logical(1))))
  expect_true(selected$info$lambda_selection_pass)
  expect_equal(selected$info$selected_lambda, .1)
  expect_equal(selected$candidates$design_constraints_pass, c(FALSE, TRUE))
  control$design_abs_ess_min <- 1000
  failed <- cs_crossfit_select_policy(list(Y = c(0, 1), p_list = p, lambda = .01),
                                      control, 1L, 100, fit = fit)
  expect_false(failed$info$lambda_selection_pass)
  expect_equal(failed$info$lambda_selection_reason, "no_lambda_passed")
})

test_that("ordinary fold records reconstruct their recorded Hajek estimate", {
  p <- list(A = c(a = .5, b = .5), B = c(x = .5, y = .5))
  W <- expand.grid(A = names(p$A), B = names(p$B), stringsAsFactors = FALSE)
  fit <- list(pi_star_point = list(k1 = list(A = c(a = .7, b = .3), B = p$B)),
              est_intercept_jnp = 0, est_coefficients_jnp = c(1, 0, 0))
  # Explicit feature map avoids reliance on a Python runtime.
  fit$glm_feature_info <- list(overall = cs_crossfit_q_reconstruct_feature_info(p))
  fit$est_coefficients_jnp <- rep(0, cs_crossfit_q_feature_info_ncols(fit$glm_feature_info$overall))
  control <- cs_crossfit_q_default_control(list(n_policy_draws = 50L))
  values <- cs_crossfit_q_fold_eval(fit, c(1, 0, 0, 1), W,
    matrix(1:4, ncol = 2, byrow = TRUE), 1:2, p, control, 1L,
    pair_id = c("p1", "p1", "p2", "p2"), respondent_id = rep("r1", 4))
  records <- attr(values, "contributions")
  expect_equal(nrow(records), 4L)
  expect_equal(unique(records$respondent_id), "r1")
  estimate <- cs_crossfit_q_dr_hajek(records$mu_policy, records$w_used, records$y, records$m_obs)
  expect_equal(estimate, values$Q_crossfit[values$estimator == "dr_hajek"])
})
