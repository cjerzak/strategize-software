embedding_test_neural_control <- function(cross_candidate_encoder = NULL) {
  control <- list(
    ModelDims = 12L,
    ModelDepth = 1L,
    subsample_method = "batch_vi",
    uncertainty_scope = "output",
    optimizer = "adam",
    svi_steps = 6L,
    svi_num_draws = 2L,
    batch_size = 16L,
    early_stopping = FALSE
  )
  if (!is.null(cross_candidate_encoder)) {
    control$cross_candidate_encoder <- cross_candidate_encoder
  }
  control
}

embedding_test_text_embedding_fn <- function(x) {
  x <- as.character(x)
  cbind(
    nchar = nchar(x),
    has_space = 1 * grepl("\\s", x),
    has_upper = 1 * grepl("[A-Z]", x)
  )
}

embedding_test_covariates <- function(W_df) {
  n <- nrow(W_df)
  idx <- seq_len(n)
  data.frame(
    income = idx / max(n, 1L),
    `household size` = 1 + (idx %% 4L) + 0.5 * (W_df[[1]] == "B"),
    GOPScore = as.numeric(W_df[[ncol(W_df)]] == "B") - 0.5,
    local_bonus = seq(-1, 1, length.out = n),
    check.names = FALSE
  )
}

embedding_test_foundation_control <- function(with_embeddings = TRUE) {
  control <- strategize:::cs_foundation_default_control()
  control$neural_mcmc_control <- embedding_test_neural_control()
  if (isTRUE(with_embeddings)) {
    control$text_embedding_fn <- embedding_test_text_embedding_fn
  } else {
    control$add_text_semantics <- FALSE
    control$text_embedding_fn <- NULL
  }
  control
}

embedding_test_pairwise_experiment <- function(seed,
                                               experiment_id,
                                               factor_names,
                                               x_names = NULL,
                                               canonical_factor_id = NULL) {
  data <- generate_test_data(
    n = 40,
    n_factors = length(factor_names),
    n_levels = 2,
    seed = seed
  )
  W_df <- as.data.frame(data$W, stringsAsFactors = FALSE)
  colnames(W_df) <- factor_names
  if (is.null(canonical_factor_id)) {
    canonical_factor_id <- stats::setNames(factor_names, factor_names)
  }
  x_full <- embedding_test_covariates(W_df)
  X <- if (is.null(x_names) || length(x_names) < 1L) {
    NULL
  } else {
    x_full[, x_names, drop = FALSE]
  }
  list(
    experiment_id = experiment_id,
    experiment_description = paste(
      "Experiment", experiment_id, "with factors", paste(factor_names, collapse = ", ")
    ),
    Y = data$Y,
    W = W_df,
    X = X,
    pair_id = data$pair_id,
    profile_order = data$profile_order,
    respondent_id = data$respondent_id,
    respondent_task_id = data$respondent_task_id,
    canonical_factor_id = canonical_factor_id
  )
}

embedding_test_single_experiment <- function(seed,
                                             experiment_id,
                                             likelihood = c("bernoulli", "normal")) {
  likelihood <- match.arg(likelihood)
  withr::local_seed(seed)
  W_df <- data.frame(
    price = sample(c("A", "B"), 24, replace = TRUE),
    message = sample(c("A", "B"), 24, replace = TRUE),
    stringsAsFactors = FALSE
  )
  Y <- if (identical(likelihood, "bernoulli")) {
    as.numeric(W_df$price == "B")
  } else {
    stats::rnorm(24, mean = 0.5 * (W_df$message == "B"), sd = 0.1)
  }
  list(
    experiment_id = experiment_id,
    Y = Y,
    W = W_df,
    mode = "single",
    likelihood = likelihood
  )
}

embedding_test_predictor_fit <- local({
  cache <- new.env(parent = emptyenv())

  function(mode = c("single", "pairwise"),
           cross_candidate_encoder = NULL,
           seed = 1L) {
    mode <- match.arg(mode)
    cross_key <- if (is.null(cross_candidate_encoder)) "default" else as.character(cross_candidate_encoder)
    cache_key <- paste(mode, cross_key, as.integer(seed), sep = "::")
    if (exists(cache_key, envir = cache, inherits = FALSE)) {
      return(get(cache_key, envir = cache, inherits = FALSE))
    }

    skip_on_cran()
    skip_if_no_jax()
    withr::local_envvar(c(STRATEGIZE_NEURAL_SKIP_EVAL = "1"))

    data <- generate_test_data(
      n = if (identical(mode, "pairwise")) 40 else 30,
      n_factors = 3,
      n_levels = 2,
      seed = seed
    )
    fit <- suppressWarnings(strategic_prediction(
      Y = data$Y,
      W = data$W,
      model = "neural",
      mode = mode,
      pair_id = if (identical(mode, "pairwise")) data$pair_id else NULL,
      profile_order = if (identical(mode, "pairwise")) data$profile_order else NULL,
      neural_mcmc_control = embedding_test_neural_control(cross_candidate_encoder),
      conda_env_required = TRUE
    ))

    out <- list(fit = fit, data = data)
    assign(cache_key, out, envir = cache)
    out
  }
})

embedding_test_pairwise_foundation_fit <- local({
  cache <- NULL

  function() {
    if (!is.null(cache)) {
      return(cache)
    }

    skip_on_cran()
    skip_if_no_jax()
    withr::local_envvar(c(STRATEGIZE_NEURAL_SKIP_EVAL = "1"))

    foundation_fit <- suppressWarnings(fit_conjoint_foundation_model(
      experiments = list(
        embedding_test_pairwise_experiment(
          seed = 9101,
          experiment_id = "study_a",
          factor_names = c("price", "message"),
          x_names = c("income", "household size")
        ),
        embedding_test_pairwise_experiment(
          seed = 9102,
          experiment_id = "study_b",
          factor_names = c("price", "message", "messenger"),
          x_names = c("income", "GOPScore")
        )
      ),
      foundation_control = embedding_test_foundation_control(with_embeddings = TRUE)
    ))

    cache <<- foundation_fit
    foundation_fit
  }
})

embedding_test_multigroup_single_foundation_fit <- local({
  cache <- NULL

  function() {
    if (!is.null(cache)) {
      return(cache)
    }

    skip_on_cran()
    skip_if_no_jax()
    withr::local_envvar(c(STRATEGIZE_NEURAL_SKIP_EVAL = "1"))

    foundation_fit <- suppressWarnings(fit_conjoint_foundation_model(
      experiments = list(
        embedding_test_single_experiment(
          seed = 9201,
          experiment_id = "single_binary",
          likelihood = "bernoulli"
        ),
        embedding_test_single_experiment(
          seed = 9202,
          experiment_id = "single_normal",
          likelihood = "normal"
        )
      ),
      foundation_control = embedding_test_foundation_control(with_embeddings = FALSE)
    ))

    cache <<- foundation_fit
    foundation_fit
  }
})

test_that("extract_embeddings returns single-mode neural embeddings", {
  fit_obj <- embedding_test_predictor_fit(mode = "single", seed = 9301)
  emb <- extract_embeddings(
    fit_obj$fit,
    newdata = as.data.frame(fit_obj$data$W[1:12, , drop = FALSE])
  )

  expect_s3_class(emb, "strategic_embeddings")
  expect_identical(emb$mode, "single")
  expect_true(is.matrix(emb$embeddings))
  expect_equal(nrow(emb$embeddings), 12L)
  expect_true(all(is.finite(emb$embeddings)))
  expect_identical(emb$metadata$source_class, "strategic_predictor")
  expect_identical(emb$metadata$cross_candidate_encoder, "none")
})

test_that("extract_embeddings returns left and right matrices for pairwise term mode", {
  fit_obj <- embedding_test_predictor_fit(
    mode = "pairwise",
    cross_candidate_encoder = "term",
    seed = 9302
  )
  emb <- extract_embeddings(
    fit_obj$fit,
    newdata = list(
      W = fit_obj$data$W,
      pair_id = fit_obj$data$pair_id,
      profile_order = fit_obj$data$profile_order
    )
  )

  expect_s3_class(emb, "strategic_embeddings")
  expect_identical(emb$mode, "pairwise")
  expect_true(is.matrix(emb$left))
  expect_true(is.matrix(emb$right))
  expect_equal(nrow(emb$left), length(unique(fit_obj$data$pair_id)))
  expect_equal(dim(emb$left), dim(emb$right))
  expect_true(all(is.finite(emb$left)))
  expect_true(all(is.finite(emb$right)))
  expect_null(emb$joint)
  expect_identical(emb$metadata$cross_candidate_encoder, "term")
})

test_that("extract_embeddings returns post-attention pairwise embeddings for attn mode", {
  fit_obj <- embedding_test_predictor_fit(
    mode = "pairwise",
    cross_candidate_encoder = "attn",
    seed = 9303
  )
  emb <- extract_embeddings(
    fit_obj$fit,
    newdata = list(
      W = fit_obj$data$W,
      pair_id = fit_obj$data$pair_id,
      profile_order = fit_obj$data$profile_order
    )
  )

  expect_s3_class(emb, "strategic_embeddings")
  expect_true(is.matrix(emb$left))
  expect_true(is.matrix(emb$right))
  expect_equal(nrow(emb$left), length(unique(fit_obj$data$pair_id)))
  expect_true(all(is.finite(emb$left)))
  expect_true(all(is.finite(emb$right)))
  expect_null(emb$joint)
  expect_identical(emb$metadata$cross_candidate_encoder, "attn")
})

test_that("extract_embeddings returns joint readout for pairwise full mode", {
  fit_obj <- embedding_test_predictor_fit(
    mode = "pairwise",
    cross_candidate_encoder = "full",
    seed = 9304
  )
  emb <- extract_embeddings(
    fit_obj$fit,
    newdata = list(
      W = fit_obj$data$W,
      pair_id = fit_obj$data$pair_id,
      profile_order = fit_obj$data$profile_order
    )
  )

  expect_s3_class(emb, "strategic_embeddings")
  expect_true(is.matrix(emb$joint))
  expect_equal(nrow(emb$joint), length(unique(fit_obj$data$pair_id)))
  expect_true(all(is.finite(emb$joint)))
  expect_null(emb$left)
  expect_null(emb$right)
  expect_identical(emb$metadata$cross_candidate_encoder, "full")
})

test_that("extract_embeddings errors on non-neural strategic predictors", {
  data <- generate_test_data(n = 200, n_factors = 2, n_levels = 2, seed = 9305)
  fit <- strategic_prediction(
    Y = data$Y,
    W = data$W,
    model = "glm",
    mode = "single",
    use_regularization = FALSE
  )

  expect_error(
    extract_embeddings(fit, newdata = as.data.frame(data$W[1:10, , drop = FALSE])),
    "only available for neural models"
  )
})

test_that("extract_embeddings works on raw foundation groups with factor metadata supplied separately", {
  foundation_fit <- embedding_test_pairwise_foundation_fit()
  study_a <- embedding_test_pairwise_experiment(
    seed = 9101,
    experiment_id = "study_a",
    factor_names = c("price", "message"),
    x_names = c("income", "household size")
  )
  newdata <- data.frame(
    study_a$W,
    study_a$X,
    pair_id = study_a$pair_id,
    profile_order = study_a$profile_order,
    experiment_id = study_a$experiment_id,
    check.names = FALSE
  )

  emb <- extract_embeddings(
    foundation_fit,
    newdata = newdata,
    group_key = "pairwise::bernoulli::1",
    p_list = suppressMessages(create_p_list(study_a$W, uniform = TRUE))
  )

  expect_s3_class(emb, "strategic_embeddings")
  expect_true(is.matrix(emb$left))
  expect_true(is.matrix(emb$right))
  expect_equal(nrow(emb$left), length(unique(study_a$pair_id)))
  expect_identical(emb$metadata$source_class, "conjoint_foundation_model")
  expect_identical(emb$metadata$foundation_group_key, "pairwise::bernoulli::1")
  expect_identical(emb$metadata$unmatched_factors, character(0))
  expect_identical(emb$metadata$unmatched_levels, character(0))
})

test_that("extract_embeddings errors when foundation group selection is ambiguous", {
  foundation_fit <- embedding_test_multigroup_single_foundation_fit()
  experiment <- embedding_test_single_experiment(
    seed = 9201,
    experiment_id = "single_binary",
    likelihood = "bernoulli"
  )

  expect_error(
    extract_embeddings(
      foundation_fit,
      newdata = experiment$W
    ),
    "ambiguous"
  )
})

test_that("saved and loaded foundation bundles preserve extracted embeddings", {
  foundation_fit <- embedding_test_pairwise_foundation_fit()
  study_a <- embedding_test_pairwise_experiment(
    seed = 9101,
    experiment_id = "study_a",
    factor_names = c("price", "message"),
    x_names = c("income", "household size")
  )
  newdata <- list(
    W = study_a$W,
    X = study_a$X,
    pair_id = study_a$pair_id,
    profile_order = study_a$profile_order,
    experiment_id = study_a$experiment_id
  )
  emb_before <- extract_embeddings(
    foundation_fit,
    newdata = newdata,
    group_key = "pairwise::bernoulli::1",
    names_list = strategize:::cs_build_names_list(study_a$W)
  )

  tmp <- tempfile(fileext = ".rds")
  save_conjoint_foundation_bundle(tmp, foundation_fit, overwrite = TRUE)
  loaded <- load_conjoint_foundation_bundle(tmp, preload_params = FALSE)
  emb_after <- extract_embeddings(
    loaded,
    newdata = newdata,
    group_key = "pairwise::bernoulli::1",
    names_list = strategize:::cs_build_names_list(study_a$W)
  )

  expect_equal(emb_before$left, emb_after$left, tolerance = 1e-6)
  expect_equal(emb_before$right, emb_after$right, tolerance = 1e-6)
})

test_that("saved and loaded adapted predictors preserve extracted embeddings", {
  foundation_fit <- embedding_test_pairwise_foundation_fit()
  target <- embedding_test_pairwise_experiment(
    seed = 9306,
    experiment_id = "target_study",
    factor_names = c("price", "message"),
    x_names = c("income", "household size", "local_bonus")
  )
  predictor <- suppressWarnings(adapt_conjoint_foundation_model(
    foundation_model = foundation_fit,
    Y = target$Y,
    W = target$W,
    X = target$X,
    mode = "pairwise",
    pair_id = target$pair_id,
    profile_order = target$profile_order,
    experiment_id = target$experiment_id,
    experiment_description = target$experiment_description,
    canonical_factor_id = target$canonical_factor_id,
    neural_mcmc_control = embedding_test_neural_control()
  ))
  emb_before <- extract_embeddings(
    predictor,
    newdata = list(
      W = target$W,
      X = target$X,
      pair_id = target$pair_id,
      profile_order = target$profile_order
    )
  )

  tmp <- tempfile(fileext = ".rds")
  save_strategic_predictor(predictor, tmp, overwrite = TRUE)
  predictor_loaded <- load_strategic_predictor(tmp)
  emb_after <- extract_embeddings(
    predictor_loaded,
    newdata = list(
      W = target$W,
      X = target$X,
      pair_id = target$pair_id,
      profile_order = target$profile_order
    )
  )

  expect_equal(emb_before$left, emb_after$left, tolerance = 1e-6)
  expect_equal(emb_before$right, emb_after$right, tolerance = 1e-6)
})
