test_that("crossfit Q control validates defaults and overrides", {
  control <- cs_crossfit_q_default_control(list(
    folds = 2,
    estimators = c("ips", "dr", "ips"),
    headline = "ips",
    return_fold_results = FALSE
  ))

  expect_equal(control$folds, 2L)
  expect_equal(control$estimators, c("ips", "dr"))
  expect_equal(control$headline, "ips")
  expect_false(control$return_fold_results)

  expect_error(
    cs_crossfit_q_default_control(list(estimators = "unknown")),
    "unknown value"
  )
  expect_error(
    cs_crossfit_q_default_control(list(folds = 1)),
    "integer >= 2"
  )
  expect_error(
    cs_crossfit_q_default_control(list(headline = "snips", estimators = "dr")),
    "must be included"
  )
})

test_that("crossfit Q validation enforces pairwise v1 constraints", {
  W <- data.frame(
    A = c("a", "b", "a", "b"),
    B = c("x", "x", "y", "y"),
    stringsAsFactors = FALSE
  )
  Y <- c(1, 0, 0, 1)
  pair_id <- c(1, 1, 2, 2)
  profile_order <- c(1, 2, 1, 2)
  p_list <- list(
    A = c(a = 0.5, b = 0.5),
    B = c(x = 0.5, y = 0.5)
  )
  control <- cs_crossfit_q_default_control(list(folds = 2))

  validated <- cs_crossfit_q_validate(
    Y = Y,
    W = W,
    pair_id = pair_id,
    profile_order = profile_order,
    p_list = p_list,
    diff = TRUE,
    adversarial = FALSE,
    K = 1,
    outcome_model_type = "glm",
    control = control
  )
  expect_equal(nrow(validated$pair_mat), 2L)

  expect_error(
    cs_crossfit_q_validate(
      Y = Y, W = W, pair_id = pair_id, profile_order = profile_order,
      p_list = p_list, diff = FALSE, adversarial = FALSE, K = 1,
      outcome_model_type = "glm", control = control
    ),
    "diff = TRUE"
  )
  expect_error(
    cs_crossfit_q_validate(
      Y = c(1, 1, 0, 1), W = W, pair_id = pair_id,
      profile_order = profile_order, p_list = p_list, diff = TRUE,
      adversarial = FALSE, K = 1, outcome_model_type = "glm",
      control = control
    ),
    "exactly one selected"
  )
  expect_error(
    cs_crossfit_q_validate(
      Y = Y, W = W, pair_id = pair_id, profile_order = profile_order,
      p_list = p_list, diff = TRUE, adversarial = FALSE, K = 2,
      outcome_model_type = "glm", control = control
    ),
    "K = 1"
  )
  expect_error(
    cs_crossfit_q_validate(
      Y = Y, W = W, pair_id = pair_id, profile_order = profile_order,
      p_list = p_list, diff = TRUE, adversarial = FALSE, K = 1,
      outcome_model_type = "glm", force_gaussian = TRUE, control = control
    ),
    "force_gaussian = FALSE"
  )
})

test_that("crossfit Q probability weights and diagnostics are computed", {
  W <- data.frame(
    A = c("a", "b", "a", "b"),
    B = c("x", "x", "y", "y"),
    stringsAsFactors = FALSE
  )
  p_list <- list(
    A = c(a = 0.5, b = 0.5),
    B = c(x = 0.5, y = 0.5)
  )
  policy <- list(
    A = c(a = 0.75, b = 0.25),
    B = c(x = 0.4, y = 0.6)
  )

  expect_equal(
    cs_crossfit_q_policy_prob(W, policy, p_list),
    c(0.75 * 0.4, 0.25 * 0.4, 0.75 * 0.6, 0.25 * 0.6)
  )

  diagnostics <- cs_crossfit_q_weight_diagnostics(c(1, 2, 3), c(1, 2, 2))
  expect_equal(diagnostics$n, 3L)
  expect_true(is.finite(diagnostics$ess))
  expect_true(diagnostics$clipped)
})

test_that("strategize can return first-class crossfit Q fields", {
  skip_on_cran()
  skip_if_no_jax()

  set.seed(2026)
  n_pairs <- 60L
  sample_profiles <- function(n) {
    data.frame(
      A = sample(c("a", "b"), n, replace = TRUE),
      B = sample(c("x", "y"), n, replace = TRUE),
      stringsAsFactors = FALSE
    )
  }
  W_left <- sample_profiles(n_pairs)
  W_right <- sample_profiles(n_pairs)
  Y_left <- stats::rbinom(n_pairs, size = 1, prob = 0.5)

  W <- rbind(W_left, W_right)
  Y <- c(Y_left, 1 - Y_left)
  pair_id <- c(seq_len(n_pairs), seq_len(n_pairs))
  respondent_id <- pair_id
  respondent_task_id <- pair_id
  profile_order <- c(rep(1L, n_pairs), rep(2L, n_pairs))

  plot_sink <- tempfile(fileext = ".pdf")
  grDevices::pdf(plot_sink)
  on.exit({
    grDevices::dev.off()
    unlink(plot_sink)
  }, add = TRUE)

  res <- strategize(
    Y = Y,
    W = W,
    lambda = 0.1,
    pair_id = pair_id,
    respondent_id = respondent_id,
    respondent_task_id = respondent_task_id,
    profile_order = profile_order,
    p_list = create_p_list(W, uniform = TRUE),
    K = 1,
    nSGD = 2L,
    diff = TRUE,
    outcome_model_type = "glm",
    force_gaussian = FALSE,
    use_regularization = FALSE,
    compute_se = FALSE,
    compute_hessian = FALSE,
    conda_env = "strategize_env",
    conda_env_required = TRUE,
    crossfit_q = TRUE,
    crossfit_q_control = list(
      folds = 2L,
      n_policy_draws = 8L,
      chunk_size = 24L,
      seed = 2026L
    )
  )

  expect_true("Q_crossfit" %in% names(res))
  expect_true("Q_reference_crossfit" %in% names(res))
  expect_true("Q_gain_crossfit" %in% names(res))
  expect_true("Q_crossfit_info" %in% names(res))
  expect_true(is.finite(res$Q_crossfit))
  expect_true(is.finite(res$Q_reference_crossfit))
  expect_true(is.finite(res$Q_gain_crossfit))
  expect_s3_class(res$Q_crossfit_info$summary, "data.frame")
  expect_true(all(c("dr", "ips", "snips", "model") %in% res$Q_crossfit_info$summary$estimator))
})
