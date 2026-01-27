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
  W <- as.data.frame(W)
  out <- lapply(seq_len(ncol(W)), function(j) {
    levs <- sort(names(table(as.factor(W[[j]]))), decreasing = FALSE)
    list(levs)
  })
  names(out) <- if (!is.null(colnames(W))) colnames(W) else paste0("V", seq_len(ncol(W)))
  out
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
  unknown <- match.arg(unknown)
  W <- as.data.frame(W)
  factor_names <- names(names_list)
  if (is.null(colnames(W))) {
    stop("'W' must have column names.", call. = FALSE)
  }
  W <- cs2step_align_W(W, factor_names)

  W_idx <- sapply(seq_along(factor_names), function(j) {
    levs <- names_list[[j]][[1]]
    idx <- match(as.character(W[[j]]), levs)
    if (unknown == "error" && any(is.na(idx))) {
      bad <- unique(as.character(W[[j]])[is.na(idx)])
      stop(
        sprintf("Unseen levels in factor '%s': %s",
                factor_names[[j]],
                paste(bad, collapse = ", ")),
        call. = FALSE
      )
    }
    if (unknown == "holdout") {
      holdout <- length(levs) + as.integer(pad_unknown)
      idx[is.na(idx)] <- holdout
    }
    as.integer(idx)
  })
  W_idx <- as.matrix(W_idx)
  colnames(W_idx) <- factor_names
  W_idx
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
                                             neural_mcmc_control = NULL) {
  if (!"jnp" %in% ls(envir = strenv)) {
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
  eval_env$competing_group_variable_candidate_ <- NULL
  eval_env$competing_group_competition_variable_candidate_ <- NULL
  eval_env$competing_group_variable_respondent_ <- NULL
  eval_env$neural_mcmc_control <- neural_mcmc_control
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
    theta_var = theta_var
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
                                nFolds_glm = 3L) {
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
	      neural_mcmc_control = neural_mcmc_control
	    )
	  }

	  structure(
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
	        timestamp = Sys.time()
	      )
	    ),
	    class = "strategic_predictor"
	  )
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
    p <- object$fit$my_model(X_left_new = X_left, X_right_new = X_right)
  } else {
    p <- object$fit$my_model(X_new = W_idx)
  }

  if (type == "link") {
    eps <- .Machine$double.eps
    p <- pmin(pmax(p, eps), 1 - eps)
    pred <- stats::qlogis(p)
  } else {
    pred <- as.numeric(p)
  }

  if (interval == "none") {
    return(pred)
  }

  if (is.null(object$fit$predict_pair) && is.null(object$fit$predict_single)) {
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

  model_info <- object$fit$neural_model_info
  draw_pred <- matrix(NA_real_, nrow = length(pred), ncol = n_draws)

  party0 <- rep(0L, length(pred))
  resp0 <- rep(0L, length(pred))
  for (i in seq_len(n_draws)) {
    theta_i <- strenv$jnp$array(theta_draws[i, ])$astype(strenv$dtj)
    params_i <- neural_params_from_theta(theta_i, model_info)
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
    draw_pred[, i] <- as.numeric(tryCatch(reticulate::py_to_r(strenv$np$array(logits_or_p)),
                                          error = function(e) logits_or_p))
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
