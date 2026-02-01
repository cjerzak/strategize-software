#' Prediction-Only Outcome Model API
#'
#' @description
#' Fit the package's outcome model (GLM or neural) without running the stochastic
#' intervention optimization in \code{\link{strategize}}. Returns a fitted
#' \code{strategic_predictor} object with a \code{\link[stats]{predict}} method.
#'
#' @name prediction-api
NULL

cs2step_build_names_list <- function(W) {
  cs_build_names_list(W = W)
}

cs2step_align_W <- function(W, factor_names) {
  if (is.null(W)) {
    stop("'W' is required.", call. = FALSE)
  }
  if (!is.data.frame(W) && !is.matrix(W)) {
    stop("'W' must be a data.frame or matrix.", call. = FALSE)
  }
  W <- as.data.frame(W)
  if (is.null(colnames(W))) {
    stop("'W' must have column names to align with the fitted model.", call. = FALSE)
  }
  missing_cols <- setdiff(factor_names, colnames(W))
  extra_cols <- setdiff(colnames(W), factor_names)
  if (length(missing_cols) > 0) {
    stop(
      "Missing factor columns in newdata: ",
      paste(missing_cols, collapse = ", "),
      call. = FALSE
    )
  }
  # Ignore extra columns (e.g., IDs) by default.
  W[, factor_names, drop = FALSE]
}

cs2step_encode_W_indices <- function(W, names_list, unknown = c("holdout", "error"), pad_unknown = 0L) {
  enc <- cs_encode_W_indices(
    W = W,
    names_list = names_list,
    unknown = unknown,
    pad_unknown = pad_unknown,
    align = "by_name"
  )
  enc$W_idx
}

cs2step_unpack_newdata <- function(newdata, factor_names, mode) {
  if (is.null(newdata)) {
    stop("'newdata' is required for prediction.", call. = FALSE)
  }

  if (is.list(newdata) && !is.data.frame(newdata)) {
    if (!"W" %in% names(newdata)) {
      stop("When newdata is a list, it must contain element 'W'.", call. = FALSE)
    }
    out <- list(
      W = newdata$W,
      pair_id = newdata$pair_id %||% NULL,
      profile_order = newdata$profile_order %||% NULL
    )
    return(out)
  }

  newdata <- as.data.frame(newdata)
  pair_id <- NULL
  profile_order <- NULL
  if (identical(mode, "pairwise")) {
    if ("pair_id" %in% colnames(newdata)) {
      pair_id <- newdata[["pair_id"]]
    }
    if ("profile_order" %in% colnames(newdata)) {
      profile_order <- newdata[["profile_order"]]
    }
  }
  W <- newdata[, factor_names, drop = FALSE]
  list(W = W, pair_id = pair_id, profile_order = profile_order)
}

`%||%` <- function(x, y) if (is.null(x)) y else x

cs2step_neural_param_cache <- new.env(parent = emptyenv())

cs2step_neural_cache_get <- function(cache_id) {
  if (is.null(cache_id) || !nzchar(cache_id)) {
    return(NULL)
  }
  if (exists(cache_id, envir = cs2step_neural_param_cache, inherits = FALSE)) {
    return(get(cache_id, envir = cs2step_neural_param_cache, inherits = FALSE))
  }
  NULL
}

cs2step_neural_cache_set <- function(cache_id, params) {
  if (is.null(cache_id) || !nzchar(cache_id)) {
    return(invisible(NULL))
  }
  assign(cache_id, params, envir = cs2step_neural_param_cache)
  invisible(NULL)
}

cs2step_has_reticulate <- function() {
  requireNamespace("reticulate", quietly = TRUE)
}

cs2step_py_to_r <- function(x) {
  if (is.null(x)) {
    return(NULL)
  }
  if (!cs2step_has_reticulate()) {
    return(x)
  }
  if (reticulate::is_py_object(x)) {
    has_np <- exists("strenv") && exists("np", envir = strenv, inherits = FALSE)
    if (isTRUE(has_np)) {
      return(tryCatch(
        reticulate::py_to_r(strenv$np$array(x)),
        error = function(e) {
          tryCatch(reticulate::py_to_r(x), error = function(e2) x)
        }
      ))
    }
    return(tryCatch(reticulate::py_to_r(x), error = function(e) x))
  }
  x
}

cs2step_eval_outcome_model_glm <- function(Y,
                                          W_idx,
                                          factor_levels,
                                          diff,
                                          pair_id = NULL,
                                          profile_order = NULL,
                                          varcov_cluster_variable = NULL,
                                          use_regularization = TRUE,
                                          nFolds_glm = 3L) {
  eval_env <- new.env(parent = environment())

  # Minimal strenv stub: generate_ModelOutcome uses only jnp$array + dtj in GLM path.
  strenv_stub <- list(
    dtj = NULL,
    jnp = list(array = function(x, dtype = NULL) x),
    np = list(array = function(x) x)
  )

  eval_env$strenv <- strenv_stub

  eval_env$adversarial <- FALSE
  eval_env$adversarial_model_strategy <- "four"
  eval_env$GroupsPool <- 1
  eval_env$GroupCounter <- 1
  eval_env$Round_ <- 1
  eval_env$outcome_model_key <- NULL
  eval_env$save_outcome_model <- FALSE
  eval_env$presaved_outcome_model <- FALSE
  eval_env$use_regularization <- isTRUE(use_regularization)
  eval_env$nFolds_glm <- as.integer(nFolds_glm)
  eval_env$folds <- NULL
  eval_env$K <- 1L
  eval_env$holdout_indicator <- 1L
  eval_env$diff <- isTRUE(diff)
  eval_env$glm_family <- "binomial"

  eval_env$w_orig <- W_idx
  eval_env$W <- W_idx
  eval_env$W_ <- W_idx
  eval_env$Y <- Y
  eval_env$Y_ <- Y

  eval_env$factor_levels <- factor_levels

  eval_env$varcov_cluster_variable <- varcov_cluster_variable
  eval_env$varcov_cluster_variable_ <- varcov_cluster_variable
  eval_env$pair_id <- pair_id
  eval_env$pair_id_ <- pair_id
  eval_env$profile_order <- profile_order
  eval_env$profile_order_ <- profile_order
  eval_env$competing_group_variable_candidate_ <- NULL
  eval_env$competing_group_competition_variable_candidate_ <- NULL
  eval_env$competing_group_variable_respondent_ <- NULL
  eval_env$X_ <- NULL
  eval_env$respondent_id <- NULL
  eval_env$respondent_task_id <- NULL

  initialize_model <- paste(deparse(generate_ModelOutcome), collapse = "\n")
  initialize_model <- gsub(initialize_model, pattern = "function \\(\\)", replacement = "")
  eval(parse(text = initialize_model), envir = eval_env)

  list(
    intercept = as.numeric(eval_env$EST_INTERCEPT_tf),
    coefficients = as.numeric(eval_env$EST_COEFFICIENTS_tf),
    vcov = eval_env$vcov_OutcomeModel,
    main_info = eval_env$main_info,
    interaction_info = eval_env$interaction_info,
    family = eval_env$glm_family,
    fit_metrics = eval_env$fit_metrics
  )
}

cs2step_eval_outcome_model_neural <- function(Y,
                                             W_idx,
                                             factor_levels,
                                             diff,
                                             pair_id = NULL,
                                             profile_order = NULL,
                                             X = NULL,
                                             conda_env = "strategize_env",
                                             conda_env_required = TRUE,
                                             neural_mcmc_control = NULL,
                                             varcov_cluster_variable = NULL,
                                             nFolds_glm = 3L) {
  if (!"jnp" %in% ls(envir = strenv) || !"np" %in% ls(envir = strenv)) {
    ok <- tryCatch({
      initialize_jax(conda_env = conda_env, conda_env_required = conda_env_required)
      TRUE
    }, error = function(e) FALSE)
    if (!isTRUE(ok)) {
      stop(
        "Neural backend not available.\n",
        "  Run strategize::build_backend() to create the JAX environment, then retry.\n",
        "  (You can also set conda_env=... to choose a different environment.)",
        call. = FALSE
      )
    }
  }

  eval_env <- new.env(parent = environment())
  eval_env$adversarial <- FALSE
  eval_env$adversarial_model_strategy <- "neural"
  eval_env$GroupsPool <- 1
  eval_env$GroupCounter <- 1
  eval_env$Round_ <- 1
  eval_env$outcome_model_key <- NULL
  eval_env$save_outcome_model <- FALSE
  eval_env$presaved_outcome_model <- FALSE
  eval_env$use_regularization <- FALSE
  eval_env$K <- 1L
  eval_env$holdout_indicator <- 1L
  eval_env$diff <- isTRUE(diff)
  eval_env$glm_family <- "binomial"

  eval_env$w_orig <- W_idx
  eval_env$W <- W_idx
  eval_env$W_ <- W_idx
  eval_env$Y <- Y
  eval_env$Y_ <- Y
  eval_env$factor_levels <- factor_levels

  eval_env$pair_id <- pair_id
  eval_env$pair_id_ <- pair_id
  eval_env$profile_order <- profile_order
  eval_env$profile_order_ <- profile_order
  eval_env$varcov_cluster_variable <- varcov_cluster_variable
  eval_env$varcov_cluster_variable_ <- varcov_cluster_variable
  eval_env$competing_group_variable_candidate_ <- NULL
  eval_env$competing_group_competition_variable_candidate_ <- NULL
  eval_env$competing_group_variable_respondent_ <- NULL
  eval_env$neural_mcmc_control <- neural_mcmc_control
  eval_env$nFolds_glm <- if (is.null(nFolds_glm)) NULL else as.integer(nFolds_glm)
  eval_env$X_ <- if (is.null(X)) NULL else as.matrix(X)
  eval_env$respondent_id <- NULL
  eval_env$respondent_task_id <- NULL

  initialize_model <- paste(deparse(generate_ModelOutcome_neural), collapse = "\n")
  initialize_model <- gsub(initialize_model, pattern = "function \\(\\)", replacement = "")
  eval(parse(text = initialize_model), envir = eval_env)

  theta_mean <- tryCatch(as.numeric(reticulate::py_to_r(strenv$np$array(eval_env$EST_COEFFICIENTS_tf))),
                         error = function(e) as.numeric(eval_env$EST_COEFFICIENTS_tf))
  vcov_vec <- eval_env$vcov_OutcomeModel
  if (!is.null(vcov_vec) && length(vcov_vec) >= 2L) {
    theta_var <- as.numeric(vcov_vec[-1])
  } else {
    theta_var <- numeric(0)
  }

  predict_pair_fxn <- if (exists("TransformerPredict_pair", envir = eval_env, inherits = FALSE)) {
    eval_env$TransformerPredict_pair
  } else {
    NULL
  }
  predict_single_fxn <- if (exists("TransformerPredict_single", envir = eval_env, inherits = FALSE)) {
    eval_env$TransformerPredict_single
  } else {
    NULL
  }

  # Drop large training artifacts that aren't needed for prediction.
  drop_names <- c(
    "PosteriorDraws", "posterior_samples",
    "X_left_jnp", "X_right_jnp", "X_single_jnp", "Y_jnp",
    "sampler", "kernel", "svi", "svi_result", "svi_state", "SVIParams"
  )
  rm(list = intersect(drop_names, ls(envir = eval_env, all.names = TRUE)), envir = eval_env)

  list(
    my_model = eval_env$my_model,
    predict_pair = predict_pair_fxn,
    predict_single = predict_single_fxn,
    neural_model_info = eval_env$neural_model_info,
    theta_mean = theta_mean,
    theta_var = theta_var,
    fit_metrics = eval_env$fit_metrics %||% eval_env$neural_model_info$fit_metrics
  )
}

cs2step_validate_binary_outcome <- function(Y) {
  if (is.null(Y) || missing(Y)) {
    stop("'Y' is required.", call. = FALSE)
  }
  if (!is.numeric(Y) && !is.integer(Y)) {
    stop("'Y' must be numeric (0/1).", call. = FALSE)
  }
  vals <- unique(stats::na.omit(as.numeric(Y)))
  if (!all(vals %in% c(0, 1))) {
    stop("This prediction API currently supports binary outcomes only (Y in {0,1}).",
         call. = FALSE)
  }
  TRUE
}

cs2step_validate_pairwise_ids <- function(pair_id, n) {
  if (is.null(pair_id)) {
    stop("Pairwise mode requires 'pair_id'.", call. = FALSE)
  }
  if (length(pair_id) != n) {
    stop(sprintf("'pair_id' has %d elements but W has %d rows.", length(pair_id), n),
         call. = FALSE)
  }
  sizes <- table(pair_id)
  if (any(sizes != 2L)) {
    bad <- names(sizes)[sizes != 2L]
    stop(
      "Pairwise mode requires exactly 2 rows per pair_id.\n",
      "  Bad pair_id values: ", paste(head(bad, 10), collapse = ", "),
      if (length(bad) > 10) " ..." else "",
      call. = FALSE
    )
  }
  TRUE
}

#' Fit a prediction-only outcome model
#'
#' @param Y Binary outcome in \code{0/1}.
#' @param W Factor matrix/data.frame (one column per conjoint factor).
#' @param X Optional covariate matrix/data.frame (neural backend).
#' @param ... Reserved for future extensions.
#' @param model \code{"glm"} or \code{"neural"}.
#' @param mode \code{"auto"}, \code{"pairwise"}, or \code{"single"}.
#' @param pair_id Optional pair identifier (required for pairwise).
#' @param profile_order Optional within-pair ordering (1/2).
#' @param varcov_cluster_variable Optional cluster IDs for robust GLM vcov.
#' @param conda_env Conda env name for neural backend.
#' @param conda_env_required Require conda env to exist (neural backend).
#' @param neural_mcmc_control Optional list passed to neural backend.
#' @param use_regularization Logical; run glinternet screening for GLM.
#' @param nFolds_glm Number of folds for glinternet CV (GLM).
#' @param cache_path Optional path to a cached predictor (.rds). If it exists and
#'   \code{cache_overwrite} is \code{FALSE}, the cached model is loaded instead of refitting.
#'   When a new model is fit, it is saved to this path.
#' @param cache_overwrite Logical; refit and overwrite any existing cache at \code{cache_path}.
#' @param cache_compress Compression setting passed to \code{saveRDS()}.
#' @return An object of class \code{strategic_predictor}.
#' @export
strategic_prediction <- function(Y,
                                W,
                                X = NULL,
                                ...,
                                model = c("glm", "neural"),
                                mode = c("auto", "pairwise", "single"),
                                pair_id = NULL,
                                profile_order = NULL,
                                varcov_cluster_variable = NULL,
                                conda_env = "strategize_env",
                                conda_env_required = TRUE,
                                neural_mcmc_control = NULL,
                                use_regularization = TRUE,
                                nFolds_glm = 3L,
                                cache_path = NULL,
                                cache_overwrite = FALSE,
                                cache_compress = TRUE) {
  if (!is.null(cache_path)) {
    cache_path <- as.character(cache_path)
    if (length(cache_path) != 1L || !nzchar(cache_path)) {
      stop("'cache_path' must be a non-empty character path.", call. = FALSE)
    }
    if (!isTRUE(cache_overwrite) && file.exists(cache_path)) {
      return(load_strategic_predictor(
        cache_path,
        conda_env = conda_env,
        conda_env_required = conda_env_required
      ))
    }
  }
  cs2step_validate_binary_outcome(Y)
  if (missing(W) || is.null(W)) {
    stop("'W' is required.", call. = FALSE)
  }
  if (!is.data.frame(W) && !is.matrix(W)) {
    stop("'W' must be a data.frame or matrix.", call. = FALSE)
  }
  if (nrow(W) != length(Y)) {
    stop(sprintf("Dimension mismatch: Y has %d elements but W has %d rows.",
                 length(Y), nrow(W)),
         call. = FALSE)
  }

  model <- match.arg(model)
  mode <- match.arg(mode)
  mode_use <- mode
  if (identical(mode, "auto")) {
    mode_use <- if (!is.null(pair_id)) "pairwise" else "single"
  }
  if (identical(mode_use, "pairwise")) {
    cs2step_validate_pairwise_ids(pair_id, nrow(W))
  }

  W_df <- as.data.frame(W)
  if (is.null(colnames(W_df))) {
    colnames(W_df) <- paste0("V", seq_len(ncol(W_df)))
  }

  names_list <- cs2step_build_names_list(W_df)
  factor_levels <- vapply(names_list, function(x) length(x[[1]]), integer(1))

  # Training encoding (no unknown padding).
  W_idx_train <- cs2step_encode_W_indices(W_df, names_list, unknown = "error", pad_unknown = 0L)

  diff <- identical(mode_use, "pairwise")

	  fit <- if (identical(model, "glm")) {
	    cs2step_eval_outcome_model_glm(
	      Y = as.numeric(Y),
	      W_idx = W_idx_train,
	      factor_levels = factor_levels,
      diff = diff,
      pair_id = pair_id,
      profile_order = profile_order,
      varcov_cluster_variable = varcov_cluster_variable,
      use_regularization = use_regularization,
      nFolds_glm = nFolds_glm
    )
  } else {
    cs2step_eval_outcome_model_neural(
      Y = as.numeric(Y),
      W_idx = W_idx_train,
      factor_levels = factor_levels,
      diff = diff,
      pair_id = pair_id,
      profile_order = profile_order,
      X = X,
	      conda_env = conda_env,
	      conda_env_required = conda_env_required,
	      neural_mcmc_control = neural_mcmc_control,
	      varcov_cluster_variable = varcov_cluster_variable,
	      nFolds_glm = nFolds_glm
	    )
	  }

	  if (identical(model, "neural") && is.null(fit$fit_metrics)) {
	    fit$fit_metrics <- fit$neural_model_info$fit_metrics %||% NULL
	  }

	  out <- structure(
	    list(
	      model_type = model,
	      mode = mode_use,
	      encoder = list(
	        factor_names = names(names_list),
	        names_list = names_list,
	        factor_levels = factor_levels,
	        unknown_policy = "holdout"
	      ),
	      fit = fit,
	      metadata = list(
	        call = match.call(),
	        timestamp = Sys.time(),
	        conda_env = if (identical(model, "neural")) conda_env else NULL,
	        conda_env_required = if (identical(model, "neural")) conda_env_required else NULL
	      )
	    ),
	    class = "strategic_predictor"
	  )
	  if (!is.null(cache_path)) {
	    save_strategic_predictor(
	      out,
	      file = cache_path,
	      overwrite = TRUE,
	      compress = cache_compress
	    )
	  }
	  out
	}

cs2step_glm_build_design <- function(W_idx, main_info, interaction_info) {
  W_idx <- as.matrix(W_idx)
  main_info <- as.data.frame(main_info)
  interaction_info <- as.data.frame(interaction_info)

  if (nrow(main_info) > 0) {
    main_dat <- apply(main_info, 1, function(row_) {
      d_ <- as.integer(row_[["d"]])
      l_ <- as.integer(row_[["l"]])
      1L * (W_idx[, d_] == l_)
    })
    if (length(main_dat) == nrow(W_idx)) {
      main_dat <- matrix(main_dat, ncol = 1)
    }
  } else {
    main_dat <- matrix(numeric(0), nrow = nrow(W_idx), ncol = 0)
  }

  if (nrow(interaction_info) > 0) {
    inter_dat <- apply(interaction_info, 1, function(row_) {
      d_ <- as.integer(row_[["d"]])
      l_ <- as.integer(row_[["l"]])
      dp_ <- as.integer(row_[["dp"]])
      lp_ <- as.integer(row_[["lp"]])
      1L * (W_idx[, d_] == l_) * 1L * (W_idx[, dp_] == lp_)
    })
    if (length(inter_dat) == nrow(W_idx)) {
      inter_dat <- matrix(inter_dat, ncol = 1)
    }
  } else {
    inter_dat <- matrix(numeric(0), nrow = nrow(W_idx), ncol = 0)
  }

  cbind(main_dat, inter_dat)
}

cs2step_rmvnorm <- function(n, mu, Sigma) {
  mu <- as.numeric(mu)
  p <- length(mu)
  Sigma <- as.matrix(Sigma)
  if (n == 0L) {
    return(matrix(numeric(0), nrow = 0, ncol = p))
  }
  if (!all(dim(Sigma) == c(p, p))) {
    stop("Sigma has incompatible dimensions.", call. = FALSE)
  }
  Sigma[!is.finite(Sigma)] <- 0

  R <- tryCatch(chol(Sigma), error = function(e) NULL)
  if (is.null(R)) {
    eig <- eigen(Sigma, symmetric = TRUE)
    vals <- pmax(eig$values, 0)
    R <- t(eig$vectors %*% diag(sqrt(vals), nrow = p, ncol = p))
  }
  Z <- matrix(stats::rnorm(n * p), nrow = n, ncol = p)
  sweep(Z %*% R, 2, mu, `+`)
}

cs2step_glm_predict_internal <- function(object,
                                        W_new,
                                        pair_id = NULL,
                                        profile_order = NULL,
                                        type = c("response", "link"),
                                        interval = c("none", "ci", "draws"),
                                        level = 0.95,
                                        n_draws = 0L,
                                        seed = NULL) {
  type <- match.arg(type)
  interval <- match.arg(interval)
  if (interval != "none" && (is.null(n_draws) || n_draws < 1L)) {
    n_draws <- 500L
  }
  n_draws <- as.integer(n_draws)

  enc <- object$encoder
  W_new <- cs2step_align_W(W_new, enc$factor_names)
  W_idx <- cs2step_encode_W_indices(W_new, enc$names_list, unknown = "holdout", pad_unknown = 0L)

  X <- cs2step_glm_build_design(W_idx, object$fit$main_info, object$fit$interaction_info)

  if (identical(object$mode, "pairwise")) {
    cs2step_validate_pairwise_ids(pair_id, nrow(W_idx))
    pair_info <- cs2step_build_pair_mat(
      pair_id = pair_id,
      W = W_idx,
      profile_order = profile_order,
      competing_group_variable_candidate = NULL
    )
    pair_mat <- pair_info$pair_mat
    X <- X[pair_mat[, 1], , drop = FALSE] - X[pair_mat[, 2], , drop = FALSE]
  }

  beta <- c(object$fit$intercept, object$fit$coefficients)
  if (ncol(X) != length(object$fit$coefficients)) {
    stop("Prediction design matrix is not aligned with fitted coefficients.", call. = FALSE)
  }
  eta <- as.numeric(beta[1] + X %*% beta[-1])
  pred <- if (type == "link") eta else stats::plogis(eta)

  if (interval == "none") {
    return(pred)
  }

  if (!is.null(seed)) {
    set.seed(seed)
  }

  V <- object$fit$vcov
  if (is.null(V) || !is.matrix(V) || any(!is.finite(dim(V)))) {
    stop("Fitted GLM object does not contain a usable variance-covariance matrix.", call. = FALSE)
  }

  beta_draws <- cs2step_rmvnorm(n_draws, mu = beta, Sigma = V)
  lin_draws <- X %*% t(beta_draws[, -1, drop = FALSE])
  lin_draws <- sweep(lin_draws, 2, beta_draws[, 1], `+`)
  draw_mat <- if (type == "link") {
    lin_draws
  } else {
    stats::plogis(lin_draws)
  }

  alpha <- (1 - level) / 2
  qs <- c(alpha, 1 - alpha)
  q_mat <- matrixStats::rowQuantiles(draw_mat, probs = qs, drop = FALSE)
  out_df <- data.frame(
    fit = pred,
    lo = q_mat[, 1],
    hi = q_mat[, 2]
  )
  if (interval == "ci") {
    return(out_df)
  }
  list(
    fit = pred,
    interval = out_df,
    draws = draw_mat,
    level = level
  )
}

cs2step_neural_to_index_matrix <- function(x_mat, factor_levels) {
  x_mat <- as.matrix(x_mat)
  x_int <- matrix(as.integer(x_mat), nrow = nrow(x_mat), ncol = ncol(x_mat))
  n_cols <- ncol(x_int)
  n_levels <- as.integer(factor_levels)
  if (length(n_levels) != n_cols) {
    n_levels <- rep_len(n_levels, n_cols)
  }
  for (d_ in seq_len(n_cols)) {
    max_level <- n_levels[[d_]]
    missing_level <- max_level + 1L
    col_vals <- x_int[, d_]
    col_vals[is.na(col_vals) | col_vals < 1L | col_vals > max_level] <- missing_level
    x_int[, d_] <- col_vals - 1L
  }
  x_int
}

cs2step_neural_prepare_resp_cov <- function(resp_cov_new, model_info, n_rows) {
  if (!is.null(resp_cov_new)) {
    resp_cov_new <- as.matrix(resp_cov_new)
    if (nrow(resp_cov_new) == 1L && n_rows > 1L) {
      resp_cov_new <- matrix(rep(resp_cov_new, each = n_rows), nrow = n_rows)
    }
    return(resp_cov_new)
  }
  resp_cov_mean <- cs2step_neural_to_r_array(model_info$resp_cov_mean)
  if (!is.null(resp_cov_mean)) {
    resp_cov_mean <- as.numeric(resp_cov_mean)
    if (length(resp_cov_mean) == 0L) {
      return(matrix(0, nrow = n_rows, ncol = 0L))
    }
    return(matrix(rep(resp_cov_mean, each = n_rows), nrow = n_rows))
  }
  matrix(0, nrow = n_rows, ncol = 0L)
}

cs2step_neural_prepare_prediction_data <- function(W_idx,
                                                   model_info,
                                                   pair_id = NULL,
                                                   profile_order = NULL,
                                                   mode = c("pairwise", "single")) {
  mode <- match.arg(mode)
  W_idx <- as.matrix(W_idx)

  pairwise <- identical(mode, "pairwise")
  if (pairwise) {
    pair_info <- cs2step_build_pair_mat(
      pair_id = pair_id,
      W = W_idx,
      profile_order = profile_order,
      competing_group_variable_candidate = NULL
    )
    pair_mat <- pair_info$pair_mat
    X_left_raw <- W_idx[pair_mat[, 1], , drop = FALSE]
    X_right_raw <- W_idx[pair_mat[, 2], , drop = FALSE]
    n_rows <- nrow(X_left_raw)
    X_left <- strenv$jnp$array(
      cs2step_neural_to_index_matrix(X_left_raw, model_info$factor_levels)
    )$astype(strenv$jnp$int32)
    X_right <- strenv$jnp$array(
      cs2step_neural_to_index_matrix(X_right_raw, model_info$factor_levels)
    )$astype(strenv$jnp$int32)
    party_left <- strenv$jnp$array(rep(0L, n_rows))$astype(strenv$jnp$int32)
    party_right <- strenv$jnp$array(rep(0L, n_rows))$astype(strenv$jnp$int32)
    resp_party <- strenv$jnp$array(rep(0L, n_rows))$astype(strenv$jnp$int32)
    resp_cov <- cs2step_neural_prepare_resp_cov(NULL, model_info, n_rows)
    resp_cov_jnp <- strenv$jnp$array(as.matrix(resp_cov))$astype(strenv$dtj)

    return(list(
      pairwise = TRUE,
      X_left = X_left,
      X_right = X_right,
      party_left = party_left,
      party_right = party_right,
      resp_party = resp_party,
      resp_cov = resp_cov_jnp
    ))
  }

  n_rows <- nrow(W_idx)
  X_single <- strenv$jnp$array(
    cs2step_neural_to_index_matrix(W_idx, model_info$factor_levels)
  )$astype(strenv$jnp$int32)
  party_single <- strenv$jnp$array(rep(0L, n_rows))$astype(strenv$jnp$int32)
  resp_party <- strenv$jnp$array(rep(0L, n_rows))$astype(strenv$jnp$int32)
  resp_cov <- cs2step_neural_prepare_resp_cov(NULL, model_info, n_rows)
  resp_cov_jnp <- strenv$jnp$array(as.matrix(resp_cov))$astype(strenv$dtj)

  list(
    pairwise = FALSE,
    X_single = X_single,
    party_single = party_single,
    resp_party = resp_party,
    resp_cov = resp_cov_jnp
  )
}

cs2step_neural_prepare_params <- function(object,
                                          conda_env = NULL,
                                          conda_env_required = TRUE) {
  if (!"jnp" %in% ls(envir = strenv)) {
    env_use <- conda_env %||% object$metadata$conda_env %||% "strategize_env"
    req_use <- if (is.null(conda_env_required)) TRUE else conda_env_required
    initialize_jax(conda_env = env_use, conda_env_required = req_use)
  }

  model_info <- object$fit$neural_model_info
  if (is.null(model_info)) {
    stop("Neural predictor is missing model metadata.", call. = FALSE)
  }

  if (!is.null(object$fit$params)) {
    params <- object$fit$params
    if (is.list(params) && length(params) > 0L) {
      needs_cast <- !(cs2step_has_reticulate() && reticulate::is_py_object(params[[1]]))
      if (isTRUE(needs_cast)) {
        params <- lapply(params, function(x) strenv$jnp$array(x)$astype(strenv$dtj))
      }
    }
    return(list(params = params, model_info = model_info))
  }

  cache_id <- object$metadata$cache_id %||% NULL
  cached_params <- cs2step_neural_cache_get(cache_id)
  if (!is.null(cached_params)) {
    return(list(params = cached_params, model_info = model_info))
  }

  theta_mean <- object$fit$theta_mean
  if (is.null(theta_mean) || !length(theta_mean)) {
    if (!is.null(model_info$params)) {
      params <- model_info$params
      if (is.list(params) && length(params) > 0L) {
        needs_cast <- !(cs2step_has_reticulate() && reticulate::is_py_object(params[[1]]))
        if (isTRUE(needs_cast)) {
          params <- lapply(params, function(x) strenv$jnp$array(x)$astype(strenv$dtj))
        }
      }
      return(list(params = params, model_info = model_info))
    }
    stop("Neural predictor is missing fitted parameters.", call. = FALSE)
  }
  theta_jnp <- strenv$jnp$array(as.numeric(theta_mean))$astype(strenv$dtj)
  params <- neural_params_from_theta(theta_jnp, model_info)
  cs2step_neural_cache_set(cache_id, params)
  list(params = params, model_info = model_info)
}

cs2step_neural_to_r_array <- function(x) {
  cs2step_py_to_r(x)
}

cs2step_neural_coerce_prediction_output <- function(pred, likelihood) {
  if (likelihood == "bernoulli") {
    return(as.numeric(cs2step_neural_to_r_array(pred)))
  }
  if (likelihood == "categorical") {
    return(as.matrix(cs2step_neural_to_r_array(pred)))
  }
  if (likelihood == "normal") {
    return(list(
      mu = as.numeric(cs2step_neural_to_r_array(pred$mu)),
      sigma = as.numeric(cs2step_neural_to_r_array(pred$sigma))
    ))
  }
  pred
}

cs2step_neural_predict_pair_prepared <- function(params, model_info, prep, return_logits = FALSE) {
  Xl <- prep$X_left
  Xr <- prep$X_right
  pl <- prep$party_left
  pr <- prep$party_right
  resp_p <- prep$resp_party
  resp_c <- prep$resp_cov

  mode <- neural_cross_encoder_mode(model_info)
  use_cross_encoder <- identical(mode, "full")
  use_cross_term <- identical(mode, "term")
  use_cross_attn <- identical(mode, "attn")

  stage_idx <- neural_stage_index(pl, pr, model_info)
  matchup_idx <- NULL
  if (!is.null(params$E_matchup)) {
    matchup_idx <- neural_matchup_index(pl, pr, model_info)
  }

  if (isTRUE(use_cross_encoder)) {
    choice_tok <- neural_build_choice_token(model_info, params)
    ctx_tokens <- neural_build_context_tokens_batch(model_info,
                                                    resp_party_idx = resp_p,
                                                    stage_idx = stage_idx,
                                                    matchup_idx = matchup_idx,
                                                    resp_cov = resp_c,
                                                    params = params)
    left_tokens <- neural_add_segment_embedding(
      neural_build_candidate_tokens_hard(Xl, pl,
                                         model_info = model_info,
                                         resp_party_idx = resp_p,
                                         params = params),
      0L,
      model_info = model_info,
      params = params
    )
    right_tokens <- neural_add_segment_embedding(
      neural_build_candidate_tokens_hard(Xr, pr,
                                         model_info = model_info,
                                         resp_party_idx = resp_p,
                                         params = params),
      1L,
      model_info = model_info,
      params = params
    )
    n_batch <- ai(Xl$shape[[1]])
    sep_tok <- neural_build_sep_token(model_info, n_batch = n_batch, params = params)
    token_parts <- list(choice_tok)
    if (!is.null(ctx_tokens)) {
      token_parts <- c(token_parts, list(ctx_tokens))
    }
    token_parts <- c(token_parts, list(sep_tok, left_tokens, sep_tok, right_tokens))
    tokens <- strenv$jnp$concatenate(token_parts, axis = 1L)
    tokens <- neural_run_transformer(tokens, model_info, params)
    cls_out <- strenv$jnp$take(tokens, strenv$jnp$arange(1L), axis = 1L)
    cls_out <- strenv$jnp$squeeze(cls_out, axis = 1L)
    logits <- neural_linear_head(cls_out, params$W_out, params$b_out)
  } else {
    choice_tok <- neural_build_choice_token(model_info, params)
    ctx_tokens <- neural_build_context_tokens_batch(model_info,
                                                    resp_party_idx = resp_p,
                                                    stage_idx = stage_idx,
                                                    matchup_idx = matchup_idx,
                                                    resp_cov = resp_c,
                                                    params = params)

    encode_candidate <- function(X_idx, party_idx, return_tokens = FALSE) {
      cand_tokens <- neural_build_candidate_tokens_hard(X_idx, party_idx,
                                                        model_info = model_info,
                                                        resp_party_idx = resp_p,
                                                        params = params)
      if (is.null(ctx_tokens)) {
        tokens <- strenv$jnp$concatenate(list(choice_tok, cand_tokens), axis = 1L)
      } else {
        tokens <- strenv$jnp$concatenate(list(choice_tok, ctx_tokens, cand_tokens), axis = 1L)
      }
      tokens <- neural_run_transformer(tokens, model_info, params)
      choice_out <- strenv$jnp$take(tokens, strenv$jnp$arange(1L), axis = 1L)
      phi <- strenv$jnp$squeeze(choice_out, axis = 1L)
      if (!isTRUE(return_tokens)) {
        return(list(phi = phi, cand_tokens_out = NULL))
      }
      T_total <- ai(tokens$shape[[2]])
      T_cand <- ai(model_info$n_candidate_tokens)
      cand_idx <- strenv$jnp$arange(ai(T_total - T_cand), ai(T_total))
      cand_out <- strenv$jnp$take(tokens, cand_idx, axis = 1L)
      list(phi = phi, cand_tokens_out = cand_out)
    }

    left_out <- encode_candidate(Xl, pl, return_tokens = isTRUE(use_cross_attn))
    right_out <- encode_candidate(Xr, pr, return_tokens = isTRUE(use_cross_attn))
    phi_l <- left_out$phi
    phi_r <- right_out$phi
    if (isTRUE(use_cross_attn)) {
      ctx_left <- neural_cross_attend_cls_to_tokens(phi_l, right_out$cand_tokens_out,
                                                    model_info = model_info,
                                                    params = params)
      ctx_right <- neural_cross_attend_cls_to_tokens(phi_r, left_out$cand_tokens_out,
                                                     model_info = model_info,
                                                     params = params)
      alpha_cross <- neural_param_or_default(params, "alpha_cross", 1.0)
      phi_l <- phi_l + alpha_cross * ctx_left
      phi_r <- phi_r + alpha_cross * ctx_right
    }
    u_l <- neural_linear_head(phi_l, params$W_out, params$b_out)
    u_r <- neural_linear_head(phi_r, params$W_out, params$b_out)
    logits <- u_l - u_r
    if (isTRUE(use_cross_term)) {
      logits <- neural_apply_cross_term(logits, phi_l, phi_r,
                                        params$M_cross, params$W_cross_out,
                                        out_dim = ai(params$W_out$shape[[2]]))
    }
  }

  if (isTRUE(return_logits)) {
    return(logits)
  }
  if (model_info$likelihood == "bernoulli") {
    return(strenv$jax$nn$sigmoid(strenv$jnp$squeeze(logits, axis = 1L)))
  }
  if (model_info$likelihood == "categorical") {
    return(strenv$jax$nn$softmax(logits, axis = -1L))
  }
  list(mu = strenv$jnp$squeeze(logits, axis = 1L),
       sigma = if (!is.null(params$sigma)) params$sigma else strenv$jnp$array(1.))
}

cs2step_neural_predict_single_prepared <- function(params, model_info, prep, return_logits = FALSE) {
  Xb <- prep$X_single
  party_idx <- prep$party_single
  resp_p <- prep$resp_party
  resp_c <- prep$resp_cov

  choice_tok <- neural_build_choice_token(model_info, params)
  ctx_tokens <- neural_build_context_tokens_batch(model_info,
                                                  resp_party_idx = resp_p,
                                                  resp_cov = resp_c,
                                                  params = params)
  cand_tokens <- neural_build_candidate_tokens_hard(Xb, party_idx,
                                                    model_info = model_info,
                                                    resp_party_idx = resp_p,
                                                    params = params)
  token_parts <- list(choice_tok)
  if (!is.null(ctx_tokens)) {
    token_parts <- c(token_parts, list(ctx_tokens))
  }
  token_parts <- c(token_parts, list(cand_tokens))
  tokens <- strenv$jnp$concatenate(token_parts, axis = 1L)
  tokens <- neural_run_transformer(tokens, model_info, params)
  choice_out <- strenv$jnp$take(tokens, strenv$jnp$arange(1L), axis = 1L)
  choice_out <- strenv$jnp$squeeze(choice_out, axis = 1L)
  logits <- neural_linear_head(choice_out, params$W_out, params$b_out)

  if (isTRUE(return_logits)) {
    return(logits)
  }
  if (model_info$likelihood == "bernoulli") {
    return(strenv$jax$nn$sigmoid(strenv$jnp$squeeze(logits, axis = 1L)))
  }
  if (model_info$likelihood == "categorical") {
    return(strenv$jax$nn$softmax(logits, axis = -1L))
  }
  list(mu = strenv$jnp$squeeze(logits, axis = 1L),
       sigma = if (!is.null(params$sigma)) params$sigma else strenv$jnp$array(1.))
}

cs2step_neural_predict_prepared <- function(params, model_info, prep, return_logits = FALSE) {
  if (isTRUE(prep$pairwise)) {
    return(cs2step_neural_predict_pair_prepared(params, model_info, prep, return_logits = return_logits))
  }
  cs2step_neural_predict_single_prepared(params, model_info, prep, return_logits = return_logits)
}

cs2step_neural_predict_internal <- function(object,
                                           W_new,
                                           pair_id = NULL,
                                           profile_order = NULL,
                                           type = c("response", "link"),
                                           interval = c("none", "ci", "draws"),
                                           level = 0.95,
                                           n_draws = 0L,
                                           seed = NULL) {
  type <- match.arg(type)
  interval <- match.arg(interval)
  if (interval != "none" && (is.null(n_draws) || n_draws < 1L)) {
    n_draws <- 200L
  }
  n_draws <- as.integer(n_draws)

  enc <- object$encoder
  W_new <- cs2step_align_W(W_new, enc$factor_names)
  W_idx <- cs2step_encode_W_indices(W_new, enc$names_list, unknown = "holdout", pad_unknown = 1L)

  use_internal <- is.function(object$fit$my_model)
  model_info <- object$fit$neural_model_info
  prep <- NULL

  if (identical(object$mode, "pairwise")) {
    cs2step_validate_pairwise_ids(pair_id, nrow(W_idx))
    pair_info <- cs2step_build_pair_mat(
      pair_id = pair_id,
      W = W_idx,
      profile_order = profile_order,
      competing_group_variable_candidate = NULL
    )
    pair_mat <- pair_info$pair_mat
    X_left <- W_idx[pair_mat[, 1], , drop = FALSE]
    X_right <- W_idx[pair_mat[, 2], , drop = FALSE]
    if (isTRUE(use_internal)) {
      p <- object$fit$my_model(X_left_new = X_left, X_right_new = X_right)
    } else {
      prep_params <- cs2step_neural_prepare_params(object)
      model_info <- prep_params$model_info
      prep <- cs2step_neural_prepare_prediction_data(
        W_idx = W_idx,
        model_info = model_info,
        pair_id = pair_id,
        profile_order = profile_order,
        mode = "pairwise"
      )
      p <- cs2step_neural_predict_prepared(
        params = prep_params$params,
        model_info = model_info,
        prep = prep,
        return_logits = identical(type, "link")
      )
    }
  } else {
    if (isTRUE(use_internal)) {
      p <- object$fit$my_model(X_new = W_idx)
    } else {
      prep_params <- cs2step_neural_prepare_params(object)
      model_info <- prep_params$model_info
      prep <- cs2step_neural_prepare_prediction_data(
        W_idx = W_idx,
        model_info = model_info,
        mode = "single"
      )
      p <- cs2step_neural_predict_prepared(
        params = prep_params$params,
        model_info = model_info,
        prep = prep,
        return_logits = identical(type, "link")
      )
    }
  }

  if (isTRUE(use_internal)) {
    if (type == "link") {
      eps <- .Machine$double.eps
      p <- pmin(pmax(p, eps), 1 - eps)
      pred <- stats::qlogis(p)
    } else {
      pred <- as.numeric(p)
    }
  } else if (identical(type, "link")) {
    pred <- as.numeric(cs2step_neural_to_r_array(p))
  } else {
    pred <- cs2step_neural_coerce_prediction_output(p, model_info$likelihood)
  }

  if (interval == "none") {
    return(pred)
  }

  draw_internal <- is.function(object$fit$predict_pair) || is.function(object$fit$predict_single)
  if (!isTRUE(draw_internal) && isTRUE(use_internal)) {
    stop("Neural predictor does not expose draw-capable prediction functions.", call. = FALSE)
  }

  theta_mean <- object$fit$theta_mean
  theta_var <- object$fit$theta_var
  if (length(theta_mean) != length(theta_var) || length(theta_mean) == 0L) {
    stop("Neural predictor does not contain parameter uncertainty information for draws.",
         call. = FALSE)
  }

  if (!is.null(seed)) {
    set.seed(seed)
  }
  theta_sd <- sqrt(pmax(theta_var, 0))
  theta_draws <- matrix(stats::rnorm(n_draws * length(theta_mean)), nrow = n_draws)
  theta_draws <- sweep(theta_draws, 2, theta_sd, `*`)
  theta_draws <- sweep(theta_draws, 2, theta_mean, `+`)

  if (is.null(model_info)) {
    model_info <- object$fit$neural_model_info
  }
  draw_pred <- matrix(NA_real_, nrow = length(pred), ncol = n_draws)

  party0 <- rep(0L, length(pred))
  resp0 <- rep(0L, length(pred))
  if (!isTRUE(draw_internal)) {
    if (is.null(prep)) {
      prep <- cs2step_neural_prepare_prediction_data(
        W_idx = W_idx,
        model_info = model_info,
        pair_id = pair_id,
        profile_order = profile_order,
        mode = object$mode
      )
    }
  }
  for (i in seq_len(n_draws)) {
    theta_i <- strenv$jnp$array(theta_draws[i, ])$astype(strenv$dtj)
    params_i <- neural_params_from_theta(theta_i, model_info)
    if (isTRUE(draw_internal)) {
      if (identical(object$mode, "pairwise")) {
        logits_or_p <- object$fit$predict_pair(
          params_i,
          X_left, X_right,
          party0, party0,
          resp0, NULL,
          return_logits = identical(type, "link")
        )
      } else {
        logits_or_p <- object$fit$predict_single(
          params_i,
          W_idx,
          party0,
          resp0,
          NULL,
          return_logits = identical(type, "link")
        )
      }
    } else {
      logits_or_p <- cs2step_neural_predict_prepared(
        params = params_i,
        model_info = model_info,
        prep = prep,
        return_logits = identical(type, "link")
      )
    }
    draw_pred[, i] <- as.numeric(cs2step_neural_to_r_array(logits_or_p))
  }

  if (type == "response") {
    draw_pred <- pmin(pmax(draw_pred, 0), 1)
  }

  alpha <- (1 - level) / 2
  qs <- c(alpha, 1 - alpha)
  q_mat <- matrixStats::rowQuantiles(draw_pred, probs = qs, drop = FALSE)
  out_df <- data.frame(
    fit = pred,
    lo = q_mat[, 1],
    hi = q_mat[, 2]
  )
  if (interval == "ci") {
    return(out_df)
  }
  list(
    fit = pred,
    interval = out_df,
    draws = draw_pred,
    level = level
  )
}

#' Predict from a fitted strategic predictor
#'
#' @param object A fitted \code{strategic_predictor}.
#' @param newdata New data. For \code{mode="single"}, a data.frame/matrix of factor columns.
#'   For \code{mode="pairwise"}, either:
#'   \itemize{
#'     \item a data.frame containing factor columns plus \code{pair_id} (and optionally \code{profile_order}), or
#'     \item a list with elements \code{W}, \code{pair_id}, and optional \code{profile_order}.
#'   }
#' @param type \code{"response"} (probability) or \code{"link"} (logit / linear predictor).
#' @param interval \code{"none"} (default), \code{"ci"}, or \code{"draws"}.
#' @param level Credible interval level for draws.
#' @param n_draws Number of posterior draws when \code{interval!="none"}.
#' @param seed Optional seed for draws.
#' @param ... Unused.
#' @export
#' @method predict strategic_predictor
predict.strategic_predictor <- function(object,
                                        newdata,
                                        type = c("response", "link"),
                                        interval = c("none", "ci", "draws"),
                                        level = 0.95,
                                        n_draws = 0L,
                                        seed = NULL,
                                        ...) {
  type <- match.arg(type)
  interval <- match.arg(interval)
  if (!inherits(object, "strategic_predictor")) {
    stop("predict.strategic_predictor requires a strategic_predictor object.", call. = FALSE)
  }

  unpacked <- cs2step_unpack_newdata(newdata, object$encoder$factor_names, object$mode)
  W_new <- unpacked$W
  pair_id <- unpacked$pair_id
  profile_order <- unpacked$profile_order

  if (identical(object$model_type, "glm")) {
    return(cs2step_glm_predict_internal(
      object = object,
      W_new = W_new,
      pair_id = pair_id,
      profile_order = profile_order,
      type = type,
      interval = interval,
      level = level,
      n_draws = n_draws,
      seed = seed
    ))
  }
  cs2step_neural_predict_internal(
    object = object,
    W_new = W_new,
    pair_id = pair_id,
    profile_order = profile_order,
    type = type,
    interval = interval,
    level = level,
    n_draws = n_draws,
    seed = seed
  )
}

#' Predict on wide-format pairwise data
#'
#' @param fit A fitted \code{strategic_predictor} with \code{mode="pairwise"}.
#' @param W_left Data frame/matrix of left profiles.
#' @param W_right Data frame/matrix of right profiles.
#' @param type \code{"response"} or \code{"link"}.
#' @param interval \code{"none"}, \code{"ci"}, or \code{"draws"}.
#' @param level Credible interval level for draws.
#' @param n_draws Number of posterior draws when \code{interval!="none"}.
#' @param seed Optional seed for draws.
#' @param ... Unused.
#' @return Predictions for each row-pair.
#' @export
predict_pair <- function(fit,
                         W_left,
                         W_right,
                         type = c("response", "link"),
                         interval = c("none", "ci", "draws"),
                         level = 0.95,
                         n_draws = 0L,
                         seed = NULL,
                         ...) {
  if (!inherits(fit, "strategic_predictor")) {
    stop("'fit' must be a strategic_predictor.", call. = FALSE)
  }
  if (!identical(fit$mode, "pairwise")) {
    stop("predict_pair() requires a pairwise strategic_predictor (mode='pairwise').", call. = FALSE)
  }
  W_left <- as.data.frame(W_left)
  W_right <- as.data.frame(W_right)
  if (nrow(W_left) != nrow(W_right)) {
    stop("W_left and W_right must have the same number of rows.", call. = FALSE)
  }
  n_pairs <- nrow(W_left)
  W_long <- rbind(W_left, W_right)
  pair_id <- c(seq_len(n_pairs), seq_len(n_pairs))
  profile_order <- c(rep(1L, n_pairs), rep(2L, n_pairs))
  predict(
    fit,
    newdata = list(W = W_long, pair_id = pair_id, profile_order = profile_order),
    type = type,
    interval = interval,
    level = level,
    n_draws = n_draws,
    seed = seed,
    ...
  )
}

#' Convert a fitted predictor to a scoring function
#'
#' @param fit A fitted \code{strategic_predictor}.
#' @return A closure that calls \code{predict()} (or \code{predict_pair()} for pairwise).
#' @export
as_function <- function(fit) {
  if (!inherits(fit, "strategic_predictor")) {
    stop("'fit' must be a strategic_predictor.", call. = FALSE)
  }
  if (identical(fit$mode, "pairwise")) {
    function(W_left, W_right, ...) {
      predict_pair(fit, W_left = W_left, W_right = W_right, ...)
    }
  } else {
    function(W, ...) {
      predict(fit, newdata = W, ...)
    }
  }
}

cs2step_neural_pack_model_info <- function(model_info, drop_params = TRUE) {
  if (is.null(model_info)) {
    return(NULL)
  }
  out <- model_info

  if (!is.null(out$params)) {
    out$params <- if (isTRUE(drop_params)) {
      NULL
    } else {
      lapply(out$params, cs2step_neural_to_r_array)
    }
  }
  if (!is.null(out$factor_index_list)) {
    out$factor_index_list <- lapply(out$factor_index_list, function(x) as.integer(cs2step_neural_to_r_array(x)))
  }
  if (!is.null(out$cand_party_to_resp_idx)) {
    out$cand_party_to_resp_idx <- as.integer(cs2step_neural_to_r_array(out$cand_party_to_resp_idx))
  }
  if (!is.null(out$resp_cov_mean)) {
    out$resp_cov_mean <- as.numeric(cs2step_neural_to_r_array(out$resp_cov_mean))
  }
  if (!is.null(out$factor_levels)) {
    out$factor_levels <- as.integer(out$factor_levels)
  }
  if (!is.null(out$param_shapes)) {
    out$param_shapes <- lapply(out$param_shapes, function(x) as.integer(cs2step_neural_to_r_array(x)))
  }
  if (!is.null(out$param_sizes)) {
    out$param_sizes <- lapply(out$param_sizes, function(x) as.integer(cs2step_neural_to_r_array(x)))
  }
  if (!is.null(out$param_offsets)) {
    out$param_offsets <- lapply(out$param_offsets, function(x) as.integer(cs2step_neural_to_r_array(x)))
  }

  int_fields <- c("n_params", "n_factors", "n_candidate_tokens", "n_party_levels",
                  "n_matchup_levels", "n_resp_covariates", "model_dims", "model_depth",
                  "n_heads", "head_dim", "choice_token_index")
  for (field in int_fields) {
    if (!is.null(out[[field]])) {
      out[[field]] <- as.integer(out[[field]])
    }
  }

  out
}

cs2step_pack_predictor <- function(object, include_metrics = TRUE) {
  if (!inherits(object, "strategic_predictor")) {
    stop("Can only save objects of class 'strategic_predictor'.", call. = FALSE)
  }

  packed <- list(
    schema_version = 1L,
    model_type = object$model_type,
    mode = object$mode,
    encoder = object$encoder,
    metadata = object$metadata
  )

  if (identical(object$model_type, "glm")) {
    fit <- object$fit
    packed$fit <- list(
      intercept = fit$intercept,
      coefficients = fit$coefficients,
      vcov = fit$vcov,
      main_info = fit$main_info,
      interaction_info = fit$interaction_info,
      family = fit$family
    )
    if (isTRUE(include_metrics) && !is.null(fit$fit_metrics)) {
      packed$fit$fit_metrics <- fit$fit_metrics
    }
  } else {
    fit <- object$fit
    drop_params <- !is.null(fit$theta_mean) && length(fit$theta_mean) > 0L
    model_info <- cs2step_neural_pack_model_info(fit$neural_model_info, drop_params = drop_params)
    packed$fit <- list(
      theta_mean = if (!is.null(fit$theta_mean)) as.numeric(fit$theta_mean) else NULL,
      theta_var = if (!is.null(fit$theta_var)) as.numeric(fit$theta_var) else NULL,
      neural_model_info = model_info
    )
    if (isTRUE(include_metrics)) {
      metrics <- fit$fit_metrics %||% (if (!is.null(model_info)) model_info$fit_metrics else NULL)
      if (!is.null(metrics)) {
        packed$fit$fit_metrics <- metrics
      }
    }
  }

  class(packed) <- c("strategic_predictor_bundle", "list")
  packed
}

cs2step_unpack_predictor <- function(bundle,
                                     conda_env = "strategize_env",
                                     conda_env_required = TRUE,
                                     preload_params = TRUE) {
  if (inherits(bundle, "strategic_predictor")) {
    return(bundle)
  }
  if (!is.list(bundle) || is.null(bundle$model_type)) {
    stop("Unrecognized predictor cache format.", call. = FALSE)
  }

  if (identical(bundle$model_type, "glm")) {
    fit <- list(
      intercept = bundle$fit$intercept,
      coefficients = bundle$fit$coefficients,
      vcov = bundle$fit$vcov,
      main_info = bundle$fit$main_info,
      interaction_info = bundle$fit$interaction_info,
      family = bundle$fit$family,
      fit_metrics = bundle$fit$fit_metrics %||% NULL
    )
    return(structure(
      list(
        model_type = bundle$model_type,
        mode = bundle$mode,
        encoder = bundle$encoder,
        fit = fit,
        metadata = bundle$metadata
      ),
      class = "strategic_predictor"
    ))
  }

  if (!cs2step_has_reticulate()) {
    stop("Loading neural predictors requires the 'reticulate' package.", call. = FALSE)
  }

  fit <- list(
    my_model = NULL,
    predict_pair = NULL,
    predict_single = NULL,
    neural_model_info = bundle$fit$neural_model_info,
    theta_mean = bundle$fit$theta_mean,
    theta_var = bundle$fit$theta_var,
    fit_metrics = bundle$fit$fit_metrics %||% (if (!is.null(bundle$fit$neural_model_info)) {
      bundle$fit$neural_model_info$fit_metrics
    } else {
      NULL
    })
  )

  metadata <- bundle$metadata %||% list()
  env_use <- if (!is.null(conda_env) && nzchar(conda_env)) {
    conda_env
  } else {
    metadata$conda_env
  }
  metadata$conda_env <- env_use %||% "strategize_env"
  metadata$conda_env_required <- if (is.null(conda_env_required)) {
    metadata$conda_env_required %||% TRUE
  } else {
    conda_env_required
  }

  out <- structure(
    list(
      model_type = bundle$model_type,
      mode = bundle$mode,
      encoder = bundle$encoder,
      fit = fit,
      metadata = metadata
    ),
    class = "strategic_predictor"
  )

  if (isTRUE(preload_params)) {
    prep <- cs2step_neural_prepare_params(out,
                                          conda_env = metadata$conda_env,
                                          conda_env_required = metadata$conda_env_required)
    out$fit$params <- prep$params
  }
  out
}

#' Save a strategic predictor to disk
#'
#' @param fit A fitted \code{strategic_predictor}.
#' @param file Path to save the cache (typically ending in \code{.rds}).
#' @param overwrite Logical; overwrite an existing file.
#' @param compress Compression passed to \code{saveRDS()}.
#' @param include_metrics Logical; include out-of-sample fit metrics when present.
#' @return The cache path (invisibly).
#' @export
save_strategic_predictor <- function(fit,
                                     file,
                                     overwrite = FALSE,
                                     compress = TRUE,
                                     include_metrics = TRUE) {
  if (!inherits(fit, "strategic_predictor")) {
    stop("'fit' must be a strategic_predictor.", call. = FALSE)
  }
  file <- as.character(file)
  if (length(file) != 1L || !nzchar(file)) {
    stop("'file' must be a non-empty path.", call. = FALSE)
  }
  if (file.exists(file) && !isTRUE(overwrite)) {
    stop("Cache file already exists; set overwrite = TRUE to replace it.", call. = FALSE)
  }
  dir.create(dirname(file), recursive = TRUE, showWarnings = FALSE)
  bundle <- cs2step_pack_predictor(fit, include_metrics = include_metrics)
  saveRDS(bundle, file = file, compress = compress)
  invisible(file)
}

#' Load a strategic predictor from disk
#'
#' @param file Path to a cached predictor created by \code{save_strategic_predictor()}.
#' @param conda_env Conda env name for neural predictors. Use \code{NULL} to
#'   defer to the cached metadata.
#' @param conda_env_required Require conda env to exist for neural predictors.
#' @return A \code{strategic_predictor} ready for \code{predict()}.
#' @export
load_strategic_predictor <- function(file,
                                     conda_env = "strategize_env",
                                     conda_env_required = TRUE) {
  file <- as.character(file)
  if (length(file) != 1L || !nzchar(file)) {
    stop("'file' must be a non-empty path.", call. = FALSE)
  }
  if (!file.exists(file)) {
    stop("Cache file does not exist.", call. = FALSE)
  }
  bundle <- readRDS(file)
  cs2step_unpack_predictor(bundle,
                           conda_env = conda_env,
                           conda_env_required = conda_env_required)
}

cs2step_normalize_names_list <- function(names_list) {
  if (is.null(names_list) || length(names_list) == 0L) {
    return(NULL)
  }
  out <- lapply(names_list, function(x) {
    if (is.list(x) && length(x) == 1L && is.atomic(x[[1]])) {
      return(as.character(x[[1]]))
    }
    if (is.atomic(x)) {
      return(as.character(x))
    }
    as.character(unlist(x))
  })
  if (is.null(names(out))) {
    names(out) <- names(names_list)
  }
  out
}

cs2step_build_neural_outcome_bundle <- function(theta_mean,
                                                theta_var = NULL,
                                                neural_model_info,
                                                names_list,
                                                factor_levels = NULL,
                                                mode = c("auto", "pairwise", "single"),
                                                fit_metrics = NULL,
                                                conda_env = "strategize_env",
                                                conda_env_required = TRUE,
                                                metadata = NULL) {
  if (is.null(neural_model_info)) {
    stop("'neural_model_info' is required.", call. = FALSE)
  }
  if (is.null(theta_mean)) {
    stop("'theta_mean' is required.", call. = FALSE)
  }
  names_list_norm <- cs2step_normalize_names_list(names_list)
  if (is.null(names_list_norm) || length(names_list_norm) == 0L) {
    stop("'names_list' must be provided to build a portable bundle.", call. = FALSE)
  }
  if (is.null(names(names_list_norm))) {
    names(names_list_norm) <- paste0("Factor", seq_len(length(names_list_norm)))
  }

  if (is.null(factor_levels)) {
    factor_levels <- vapply(names_list_norm, length, integer(1))
  }

  theta_mean_num <- as.numeric(cs2step_neural_to_r_array(theta_mean))
  theta_var_num <- if (!is.null(theta_var)) {
    as.numeric(cs2step_neural_to_r_array(theta_var))
  } else {
    NULL
  }

  mode <- match.arg(mode)
  if (identical(mode, "auto")) {
    mode <- if (!is.null(neural_model_info$pairwise_mode) &&
                isTRUE(neural_model_info$pairwise_mode)) {
      "pairwise"
    } else {
      "single"
    }
  }

  packed_info <- cs2step_neural_pack_model_info(neural_model_info, drop_params = TRUE)
  fit_metrics <- fit_metrics %||% packed_info$fit_metrics %||% NULL

  encoder <- list(
    factor_names = names(names_list_norm),
    names_list = lapply(names_list_norm, function(x) list(x)),
    factor_levels = as.integer(factor_levels),
    unknown_policy = "holdout"
  )

  meta_default <- list(
    created_at = Sys.time(),
    conda_env = conda_env,
    conda_env_required = conda_env_required
  )
  meta <- modifyList(meta_default, metadata %||% list())

  bundle <- list(
    schema_version = 1L,
    model_type = "neural",
    mode = mode,
    encoder = encoder,
    fit = list(
      theta_mean = theta_mean_num,
      theta_var = theta_var_num,
      neural_model_info = packed_info,
      fit_metrics = fit_metrics
    ),
    metadata = meta
  )
  bundle$theta_mean <- theta_mean_num
  bundle$theta_var <- theta_var_num
  bundle$neural_model_info <- packed_info
  bundle$fit_metrics <- fit_metrics
  class(bundle) <- c("strategic_predictor_bundle", "list")
  bundle
}

#' Save a portable neural outcome bundle
#'
#' @param file Path to save the bundle (typically ending in \code{.rds}).
#' @param theta_mean Numeric vector of posterior means for neural parameters.
#' @param theta_var Optional numeric vector of posterior variances.
#' @param neural_model_info Neural model metadata (can include reticulate objects).
#' @param names_list Optional list of factor level names (see \code{cs_prepare_W_encoding}).
#' @param p_list Optional \code{p_list} to derive factor level names when \code{names_list} is missing.
#' @param factor_levels Optional integer vector of factor levels (derived from \code{names_list} by default).
#' @param mode \code{"auto"}, \code{"pairwise"}, or \code{"single"}.
#' @param fit_metrics Optional fit metrics to include.
#' @param conda_env Conda env name for neural backend (stored in metadata).
#' @param conda_env_required Require conda env to exist (stored in metadata).
#' @param overwrite Logical; overwrite existing file.
#' @param compress Compression passed to \code{saveRDS()}.
#' @param metadata Optional list of extra metadata to include.
#' @return The bundle path (invisibly).
#' @export
save_neural_outcome_bundle <- function(file,
                                       theta_mean,
                                       theta_var = NULL,
                                       neural_model_info,
                                       names_list = NULL,
                                       p_list = NULL,
                                       factor_levels = NULL,
                                       mode = c("auto", "pairwise", "single"),
                                       fit_metrics = NULL,
                                       conda_env = "strategize_env",
                                       conda_env_required = TRUE,
                                       overwrite = FALSE,
                                       compress = TRUE,
                                       metadata = NULL) {
  file <- as.character(file)
  if (length(file) != 1L || !nzchar(file)) {
    stop("'file' must be a non-empty path.", call. = FALSE)
  }
  if (file.exists(file) && !isTRUE(overwrite)) {
    stop("Bundle file already exists; set overwrite = TRUE to replace it.", call. = FALSE)
  }

  if (is.null(names_list)) {
    if (!is.null(p_list) && length(p_list) > 0L) {
      names_list <- lapply(p_list, function(zer) {
        levs <- names(zer)
        if (is.null(levs)) {
          levs <- as.character(seq_len(length(zer)))
        }
        list(levs)
      })
      if (!is.null(names(p_list))) {
        names(names_list) <- names(p_list)
      }
    }
  }

  bundle <- cs2step_build_neural_outcome_bundle(
    theta_mean = theta_mean,
    theta_var = theta_var,
    neural_model_info = neural_model_info,
    names_list = names_list,
    factor_levels = factor_levels,
    mode = mode,
    fit_metrics = fit_metrics,
    conda_env = conda_env,
    conda_env_required = conda_env_required,
    metadata = metadata
  )

  dir.create(dirname(file), recursive = TRUE, showWarnings = FALSE)
  saveRDS(bundle, file = file, compress = compress)
  invisible(file)
}

#' Load a portable neural outcome bundle
#'
#' @param file Path to a bundle created by \code{save_neural_outcome_bundle()}.
#' @param conda_env Conda env name for neural backend. Use \code{NULL} to defer to metadata.
#' @param conda_env_required Require conda env to exist for neural backend.
#' @param preload_params Logical; if TRUE, reconstruct neural params immediately.
#' @return A \code{strategic_predictor} ready for \code{predict()} / \code{predict_pair()}.
#' @export
load_neural_outcome_bundle <- function(file,
                                       conda_env = "strategize_env",
                                       conda_env_required = TRUE,
                                       preload_params = FALSE) {
  file <- as.character(file)
  if (length(file) != 1L || !nzchar(file)) {
    stop("'file' must be a non-empty path.", call. = FALSE)
  }
  if (!file.exists(file)) {
    stop("Bundle file does not exist.", call. = FALSE)
  }

  bundle <- readRDS(file)
  if (!is.list(bundle)) {
    stop("Unrecognized bundle format.", call. = FALSE)
  }
  if (is.null(bundle$fit)) {
    bundle$fit <- list(
      theta_mean = bundle$theta_mean %||% NULL,
      theta_var = bundle$theta_var %||% NULL,
      neural_model_info = bundle$neural_model_info %||% NULL,
      fit_metrics = bundle$fit_metrics %||% NULL
    )
  } else {
    if (is.null(bundle$fit$theta_mean) && !is.null(bundle$theta_mean)) {
      bundle$fit$theta_mean <- bundle$theta_mean
    }
    if (is.null(bundle$fit$theta_var) && !is.null(bundle$theta_var)) {
      bundle$fit$theta_var <- bundle$theta_var
    }
    if (is.null(bundle$fit$neural_model_info) && !is.null(bundle$neural_model_info)) {
      bundle$fit$neural_model_info <- bundle$neural_model_info
    }
    if (is.null(bundle$fit$fit_metrics) && !is.null(bundle$fit_metrics)) {
      bundle$fit$fit_metrics <- bundle$fit_metrics
    }
  }
  if (is.null(bundle$fit$neural_model_info)) {
    stop("Bundle is missing neural_model_info.", call. = FALSE)
  }
  if (is.null(bundle$model_type)) {
    bundle$model_type <- "neural"
  }
  if (is.null(bundle$mode) && !is.null(bundle$fit$neural_model_info$pairwise_mode)) {
    bundle$mode <- if (isTRUE(bundle$fit$neural_model_info$pairwise_mode)) {
      "pairwise"
    } else {
      "single"
    }
  }
  if (is.null(bundle$mode)) {
    bundle$mode <- "single"
  }

  if (is.null(bundle$encoder) || is.null(bundle$encoder$names_list)) {
    factor_levels <- bundle$fit$neural_model_info$factor_levels
    if (is.null(factor_levels)) {
      stop("Bundle is missing encoder metadata.", call. = FALSE)
    }
    n_factors <- length(factor_levels)
    names_list <- lapply(seq_len(n_factors), function(j) {
      list(as.character(seq_len(factor_levels[[j]])))
    })
    names(names_list) <- paste0("Factor", seq_len(n_factors))
    bundle$encoder <- list(
      factor_names = names(names_list),
      names_list = names_list,
      factor_levels = as.integer(factor_levels),
      unknown_policy = "holdout"
    )
  }

  out <- cs2step_unpack_predictor(
    bundle,
    conda_env = conda_env,
    conda_env_required = conda_env_required,
    preload_params = preload_params
  )
  if (inherits(out, "strategic_predictor")) {
    out$metadata$cache_id <- sprintf("neural_cache_%d", as.integer(stats::runif(1, 1, 1e9)))
  }
  out
}
