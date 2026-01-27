test_that("strategic_prediction() fits GLM predictor (pairwise) and predicts probabilities", {
  data <- generate_test_data(n = 400, n_factors = 3, n_levels = 2, seed = 101)

  fit <- strategic_prediction(
    Y = data$Y,
    W = data$W,
    model = "glm",
    mode = "pairwise",
    pair_id = data$pair_id,
    profile_order = data$profile_order,
    use_regularization = FALSE
  )

  preds <- predict(
    fit,
    newdata = list(W = data$W, pair_id = data$pair_id, profile_order = data$profile_order)
  )

  testthat::expect_type(preds, "double")
  testthat::expect_length(preds, length(unique(data$pair_id)))
  testthat::expect_true(all(is.finite(preds)))
  testthat::expect_true(all(preds >= 0 & preds <= 1))
})

test_that("predict_pair() is approximately swap-invariant for symmetric data", {
  data <- generate_test_data(n = 2000, n_factors = 3, n_levels = 2, seed = 202)

  fit <- strategic_prediction(
    Y = data$Y,
    W = data$W,
    model = "glm",
    mode = "pairwise",
    pair_id = data$pair_id,
    profile_order = data$profile_order,
    use_regularization = FALSE
  )

  W_left <- as.data.frame(data$W[data$profile_order == 1L, , drop = FALSE])
  W_right <- as.data.frame(data$W[data$profile_order == 2L, , drop = FALSE])
  W_left <- W_left[seq_len(25), , drop = FALSE]
  W_right <- W_right[seq_len(25), , drop = FALSE]

  p_lr <- predict_pair(fit, W_left = W_left, W_right = W_right)
  p_rl <- predict_pair(fit, W_left = W_right, W_right = W_left)

  testthat::expect_true(all(is.finite(p_lr)))
  testthat::expect_true(all(is.finite(p_rl)))
  testthat::expect_equal(p_lr + p_rl, rep(1, length(p_lr)), tolerance = 0.05)
})

test_that("unseen levels map to holdout by default (GLM, single)", {
  data <- generate_test_data(n = 400, n_factors = 2, n_levels = 2, seed = 303)

  fit <- strategic_prediction(
    Y = data$Y,
    W = data$W,
    model = "glm",
    mode = "single",
    use_regularization = FALSE
  )

  W_new <- as.data.frame(data$W[1:10, , drop = FALSE])
  base_level <- sort(unique(as.character(data$W[, 1])))[2]
  W_holdout <- W_new
  W_holdout[1, 1] <- base_level

  W_new[1, 1] <- "UNSEEN_LEVEL"

  p_unseen <- predict(fit, newdata = W_new)
  p_holdout <- predict(fit, newdata = W_holdout)

  testthat::expect_true(is.finite(p_unseen[1]))
  testthat::expect_equal(p_unseen[1], p_holdout[1], tolerance = 1e-10)
})

test_that("predict() intervals return expected structure (GLM)", {
  data <- generate_test_data(n = 400, n_factors = 3, n_levels = 2, seed = 404)

  fit <- strategic_prediction(
    Y = data$Y,
    W = data$W,
    model = "glm",
    mode = "pairwise",
    pair_id = data$pair_id,
    profile_order = data$profile_order,
    use_regularization = FALSE
  )

  ci <- predict(
    fit,
    newdata = list(W = data$W, pair_id = data$pair_id, profile_order = data$profile_order),
    interval = "ci",
    n_draws = 200,
    seed = 1
  )

  testthat::expect_s3_class(ci, "data.frame")
  testthat::expect_true(all(c("fit", "lo", "hi") %in% colnames(ci)))
  testthat::expect_true(all(is.finite(ci$fit)))
  testthat::expect_true(all(ci$lo <= ci$fit & ci$fit <= ci$hi, na.rm = TRUE))
})

test_that("as_function() returns a scoring closure", {
  data <- generate_test_data(n = 400, n_factors = 3, n_levels = 2, seed = 505)

  fit_single <- strategic_prediction(
    Y = data$Y,
    W = data$W,
    model = "glm",
    mode = "single",
    use_regularization = FALSE
  )
  f_single <- as_function(fit_single)
  p1 <- predict(fit_single, newdata = as.data.frame(data$W[1:15, , drop = FALSE]))
  p2 <- f_single(as.data.frame(data$W[1:15, , drop = FALSE]))
  testthat::expect_equal(p1, p2)

  fit_pair <- strategic_prediction(
    Y = data$Y,
    W = data$W,
    model = "glm",
    mode = "pairwise",
    pair_id = data$pair_id,
    profile_order = data$profile_order,
    use_regularization = FALSE
  )
  f_pair <- as_function(fit_pair)

  W_left <- as.data.frame(data$W[data$profile_order == 1L, , drop = FALSE])[1:10, , drop = FALSE]
  W_right <- as.data.frame(data$W[data$profile_order == 2L, , drop = FALSE])[1:10, , drop = FALSE]
  p3 <- predict_pair(fit_pair, W_left = W_left, W_right = W_right)
  p4 <- f_pair(W_left, W_right)
  testthat::expect_equal(p3, p4)
})

test_that("neural backend errors with build_backend() hint when unavailable", {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    testthat::skip("reticulate not available")
  }
  conda_list <- tryCatch(reticulate::conda_list(), error = function(e) NULL)
  if (!is.null(conda_list) && "strategize_env" %in% conda_list$name) {
    jax_available <- tryCatch({
      reticulate::use_condaenv("strategize_env", required = TRUE)
      reticulate::py_module_available("jax")
    }, error = function(e) FALSE)
    if (isTRUE(jax_available)) {
      testthat::skip("JAX available; neural fit is slow and not exercised here")
    }
  }

  data <- generate_test_data(n = 40, n_factors = 2, n_levels = 2, seed = 606)
  testthat::expect_error(
    strategic_prediction(
      Y = data$Y,
      W = data$W,
      model = "neural",
      mode = "pairwise",
      pair_id = data$pair_id,
      profile_order = data$profile_order,
      conda_env_required = TRUE
    ),
    "build_backend"
  )
})

