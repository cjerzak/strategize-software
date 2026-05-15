prediction_test_foundation_control <- function() {
  modifyList(
    strategize:::cs_foundation_default_control(),
    list(
      neural_mcmc_control = list(
        ModelDims = 16L,
        ModelDepth = 1L,
        subsample_method = "batch_vi",
        uncertainty_scope = "output",
        optimizer = "adam",
        svi_steps = 8L,
        svi_num_draws = 2L,
        batch_size = 16L,
        early_stopping = FALSE
      )
    )
  )
}

prediction_test_experiment <- function(seed, experiment_id) {
  data <- generate_test_data(n = 40, n_factors = 2, n_levels = 2, seed = seed)
  W_df <- as.data.frame(data$W, stringsAsFactors = FALSE)
  colnames(W_df) <- c("price", "message")
  list(
    experiment_id = experiment_id,
    experiment_description = paste("Experiment", experiment_id),
    Y = data$Y,
    W = W_df,
    X = data.frame(income = seq_len(nrow(W_df)) / nrow(W_df)),
    pair_id = data$pair_id,
    profile_order = data$profile_order,
    respondent_id = data$respondent_id,
    respondent_task_id = data$respondent_task_id
  )
}

prediction_test_cache_model_info <- function() {
  list(
    model_depth = 1L,
    model_dims = 8L,
    n_heads = 1L,
    head_dim = 8L,
    residual_mode = "standard",
    cross_candidate_encoder_mode = "none",
    likelihood = "bernoulli",
    experiment_token_mode = "description",
    factor_tokenization = "index",
    max_factor_tokens = 2L,
    covariate_value_encoding = "legacy_linear",
    shared_projection_value_encoder = "none",
    max_covariate_tokens = 0L,
    n_candidate_tokens = 2L,
    n_party_levels = 2L,
    cand_party_to_resp_idx = c(0L, 1L),
    text_semantic_dim = 2L
  )
}

prediction_test_text_embedding <- function(text) {
  text <- as.character(text)
  t(vapply(text, function(x) {
    bytes <- utf8ToInt(x)
    c(sum(bytes), length(bytes))
  }, numeric(2)))
}

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

test_that("GLM pairwise predict preserves first-seen pair order across locales", {
  data <- generate_test_data(n = 400, n_factors = 3, n_levels = 2, seed = 606)

  fit <- strategic_prediction(
    Y = data$Y,
    W = data$W,
    model = "glm",
    mode = "pairwise",
    pair_id = data$pair_id,
    profile_order = data$profile_order,
    use_regularization = FALSE
  )

  candidate_pairs <- unique(data$pair_id)[seq_len(40L)]
  candidate_pred <- vapply(candidate_pairs, function(pair) {
    idx <- which(data$pair_id == pair)
    predict(
      fit,
      newdata = list(
        W = data$W[idx, , drop = FALSE],
        pair_id = rep("candidate_pair", length(idx)),
        profile_order = data$profile_order[idx]
      )
    )[[1L]]
  }, numeric(1))
  ordered_candidates <- order(candidate_pred)
  selected_pairs <- candidate_pairs[c(
    ordered_candidates[[length(ordered_candidates)]],
    ordered_candidates[[1L]],
    ordered_candidates[[ceiling(length(ordered_candidates) / 2L)]]
  )]

  rows <- unlist(lapply(selected_pairs, function(pair) {
    which(data$pair_id == pair)
  }), use.names = FALSE)
  first_seen_pair_id <- c("é_pair", "a_pair", "Z_pair")
  pair_id <- rep(first_seen_pair_id, each = 2L)
  W_new <- data$W[rows, , drop = FALSE]
  profile_order <- data$profile_order[rows]

  expected <- vapply(seq_along(first_seen_pair_id), function(i) {
    idx <- which(pair_id == first_seen_pair_id[[i]])
    predict(
      fit,
      newdata = list(
        W = W_new[idx, , drop = FALSE],
        pair_id = pair_id[idx],
        profile_order = profile_order[idx]
      )
    )[[1L]]
  }, numeric(1))
  testthat::expect_gt(diff(range(expected)), 1e-6)

  predict_batch <- function() {
    predict(
      fit,
      newdata = list(W = W_new, pair_id = pair_id, profile_order = profile_order)
    )
  }

  batch <- predict_batch()
  testthat::expect_equal(unname(batch), unname(expected), tolerance = 1e-10)

  old_collate <- Sys.getlocale("LC_COLLATE")
  withr::defer(Sys.setlocale("LC_COLLATE", old_collate))

  predict_under_locale <- function(locale) {
    set_locale <- suppressWarnings(Sys.setlocale("LC_COLLATE", locale))
    if (is.na(set_locale) || !nzchar(set_locale)) {
      return(NULL)
    }
    predict_batch()
  }

  c_batch <- predict_under_locale("C")
  testthat::expect_false(is.null(c_batch))
  testthat::expect_equal(unname(c_batch), unname(expected), tolerance = 1e-10)

  utf8_batch <- NULL
  for (locale in c("C.UTF-8", "en_US.UTF-8", "en_US.utf8")) {
    utf8_batch <- predict_under_locale(locale)
    if (!is.null(utf8_batch)) {
      break
    }
  }
  testthat::skip_if(is.null(utf8_batch), "No UTF-8 collation locale available")
  testthat::expect_equal(unname(utf8_batch), unname(expected), tolerance = 1e-10)
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

test_that("cs2step_unpack_newdata keeps pairwise group metadata separate from X", {
  newdata <- data.frame(
    price = c("A", "B"),
    message = c("A", "B"),
    pair_id = c(1L, 1L),
    profile_order = c(1L, 2L),
    competing_group_variable_candidate = c("PartyA", "PartyB"),
    competing_group_variable_respondent = c("PartyA", "PartyA"),
    income = c(0.1, 0.2),
    stringsAsFactors = FALSE
  )

  unpacked <- strategize:::cs2step_unpack_newdata(
    newdata = newdata,
    factor_names = c("price", "message"),
    mode = "pairwise"
  )

  testthat::expect_identical(
    unpacked$competing_group_variable_candidate,
    c("PartyA", "PartyB")
  )
  testthat::expect_identical(
    unpacked$competing_group_variable_respondent,
    c("PartyA", "PartyA")
  )
  testthat::expect_identical(colnames(unpacked$X), "income")
})

test_that("cs2step_build_pair_mat preserves first-seen pair order", {
  W <- data.frame(
    price = c("A", "B", "A", "B", "A", "B"),
    message = c("X", "Y", "Y", "X", "X", "Y"),
    stringsAsFactors = FALSE
  )
  pair_id <- c("b_pair", "b_pair", "a_pair", "a_pair", "c_pair", "c_pair")
  profile_order <- c(2L, 1L, 1L, 2L, 2L, 1L)

  pair_info <- strategize:::cs2step_build_pair_mat(
    pair_id = pair_id,
    W = W,
    profile_order = profile_order
  )

  testthat::expect_identical(names(pair_info$pair_sizes), c("b_pair", "a_pair", "c_pair"))
  testthat::expect_identical(unname(pair_info$pair_sizes), c(2L, 2L, 2L))
  testthat::expect_identical(
    unname(pair_info$pair_mat),
    matrix(c(2L, 1L, 3L, 4L, 6L, 5L), ncol = 2L, byrow = TRUE)
  )
})

test_that("cs2step_build_pair_mat is stable across collation locales", {
  W <- data.frame(
    price = c("A", "B", "A", "B", "A", "B"),
    message = c("X", "Y", "Y", "X", "X", "Y"),
    stringsAsFactors = FALSE
  )
  pair_id <- c("é_pair", "é_pair", "a_pair", "a_pair", "Z_pair", "Z_pair")
  profile_order <- c(1L, 2L, 1L, 2L, 1L, 2L)
  old_collate <- Sys.getlocale("LC_COLLATE")
  withr::defer(Sys.setlocale("LC_COLLATE", old_collate))

  build_under_locale <- function(locale) {
    set_locale <- suppressWarnings(Sys.setlocale("LC_COLLATE", locale))
    if (is.na(set_locale) || !nzchar(set_locale)) {
      return(NULL)
    }
    strategize:::cs2step_build_pair_mat(
      pair_id = pair_id,
      W = W,
      profile_order = profile_order
    )
  }

  c_pair_info <- build_under_locale("C")
  testthat::expect_false(is.null(c_pair_info))
  utf8_pair_info <- NULL
  for (locale in c("C.UTF-8", "en_US.UTF-8", "en_US.utf8")) {
    utf8_pair_info <- build_under_locale(locale)
    if (!is.null(utf8_pair_info)) {
      break
    }
  }
  testthat::skip_if(is.null(utf8_pair_info), "No UTF-8 collation locale available")

  testthat::expect_identical(utf8_pair_info$pair_mat, c_pair_info$pair_mat)
  testthat::expect_identical(utf8_pair_info$pair_sizes, c_pair_info$pair_sizes)
  testthat::expect_identical(names(c_pair_info$pair_sizes), unique(pair_id))
})

test_that("neural_model_jit_cache_key ignores legacy jit_cache_key noise", {
  model_a <- prediction_test_cache_model_info()
  model_b <- prediction_test_cache_model_info()
  model_a$jit_cache_key <- "legacy_random_a"
  model_b$jit_cache_key <- "legacy_random_b"

  testthat::expect_identical(
    strategize:::neural_model_jit_cache_key(model_a),
    strategize:::neural_model_jit_cache_key(model_b)
  )
})

test_that("experiment description content changes deterministic neural jit keys", {
  base_model <- prediction_test_cache_model_info()
  same_a <- strategize:::cs2step_neural_apply_experiment_description(
    model_info = base_model,
    experiment_description = "alpha experiment",
    n_rows = 1L,
    text_embedding_fn = prediction_test_text_embedding
  )
  same_b <- strategize:::cs2step_neural_apply_experiment_description(
    model_info = base_model,
    experiment_description = "alpha experiment",
    n_rows = 1L,
    text_embedding_fn = prediction_test_text_embedding
  )
  other <- strategize:::cs2step_neural_apply_experiment_description(
    model_info = base_model,
    experiment_description = "beta experiment",
    n_rows = 1L,
    text_embedding_fn = prediction_test_text_embedding
  )

  testthat::expect_identical(
    strategize:::neural_model_jit_cache_key(same_a),
    strategize:::neural_model_jit_cache_key(same_b)
  )
  testthat::expect_false(identical(
    strategize:::neural_model_jit_cache_key(same_a),
    strategize:::neural_model_jit_cache_key(other)
  ))
})

test_that("upgrade path strips legacy jit cache keys", {
  upgraded <- strategize:::cs2step_neural_upgrade_model_info(list(
    jit_cache_key = "legacy_random_value"
  ))

  testthat::expect_null(upgraded$jit_cache_key)
})

test_that("neural jit wrapper cache reuses identical experiment descriptions", {
  skip_on_cran()
  skip_if_no_jax()

  if (!"jnp" %in% ls(envir = strategize:::strenv)) {
    strategize:::initialize_jax(conda_env = "strategize_env", conda_env_required = TRUE)
  }

  cache_env <- strategize:::neural_prediction_jit_cache
  base_model <- prediction_test_cache_model_info()
  model_same <- strategize:::cs2step_neural_apply_experiment_description(
    model_info = base_model,
    experiment_description = "alpha experiment",
    n_rows = 1L,
    text_embedding_fn = prediction_test_text_embedding
  )
  model_other <- strategize:::cs2step_neural_apply_experiment_description(
    model_info = base_model,
    experiment_description = "beta experiment",
    n_rows = 1L,
    text_embedding_fn = prediction_test_text_embedding
  )

  same_entry <- paste(
    "predict",
    strategize:::neural_model_jit_cache_key(model_same),
    "single",
    "response",
    sep = "::"
  )
  other_entry <- paste(
    "predict",
    strategize:::neural_model_jit_cache_key(model_other),
    "single",
    "response",
    sep = "::"
  )
  existed_same <- exists(same_entry, envir = cache_env, inherits = FALSE)
  existed_other <- exists(other_entry, envir = cache_env, inherits = FALSE)
  on.exit({
    if (!existed_same && exists(same_entry, envir = cache_env, inherits = FALSE)) {
      rm(list = same_entry, envir = cache_env)
    }
    if (!existed_other && exists(other_entry, envir = cache_env, inherits = FALSE)) {
      rm(list = other_entry, envir = cache_env)
    }
  }, add = TRUE)

  strategize:::neural_get_predict_jit(
    model_info = model_same,
    pairwise = FALSE,
    return_logits = FALSE
  )
  entries_after_first <- ls(envir = cache_env, all.names = TRUE)

  strategize:::neural_get_predict_jit(
    model_info = model_same,
    pairwise = FALSE,
    return_logits = FALSE
  )
  entries_after_second <- ls(envir = cache_env, all.names = TRUE)

  strategize:::neural_get_predict_jit(
    model_info = model_other,
    pairwise = FALSE,
    return_logits = FALSE
  )
  entries_after_third <- ls(envir = cache_env, all.names = TRUE)

  testthat::expect_true(same_entry %in% entries_after_first)
  testthat::expect_identical(entries_after_second, entries_after_first)
  testthat::expect_true(other_entry %in% entries_after_third)
  testthat::expect_false(identical(same_entry, other_entry))
})

test_that("stage-aware neural predictors require long-format predict()", {
  predictor <- structure(
    list(
      model_type = "neural",
      mode = "pairwise",
      encoder = list(
        factor_names = c("price", "message"),
        names_list = list(price = c("A", "B"), message = c("A", "B")),
        factor_levels = c(price = 2L, message = 2L)
      ),
      fit = list(
        neural_model_info = list(pairwise_context_mode = "stage_aware")
      ),
      metadata = list()
    ),
    class = "strategic_predictor"
  )
  W_left <- data.frame(price = "A", message = "A")
  W_right <- data.frame(price = "B", message = "B")

  testthat::expect_error(
    predict_pair(predictor, W_left = W_left, W_right = W_right),
    "long-format newdata"
  )

  predictor$fit$neural_model_info <- list(
    has_stage_context = TRUE,
    has_matchup_token = TRUE
  )
  testthat::expect_error(
    predict_pair(predictor, W_left = W_left, W_right = W_right),
    "competing_group_variable_candidate"
  )
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
