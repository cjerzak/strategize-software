# Controls for execution, storage and derivatives; none changes the objective.
cs_policy_control <- function(control = NULL) {
  defaults <- list(loop = "auto", trace = FALSE, remat = TRUE,
                   se_chunk_size = 16L, reuse_outcomes = TRUE,
                   .fit_cache = NULL, .evaluation_only = FALSE)
  if (is.null(control) || (is.list(control) && !length(control))) return(defaults)
  if (!is.list(control) || is.null(names(control)) ||
      any(!names(control) %in% names(defaults))) {
    stop("Unknown or unnamed policy_control fields.", call. = FALSE)
  }
  out <- modifyList(defaults, control, keep.null = TRUE)
  out$loop <- match.arg(out$loop, c("auto", "scan", "r"))
  for (name in c("trace", "remat", "reuse_outcomes", ".evaluation_only")) {
    if (!is.logical(out[[name]]) || length(out[[name]]) != 1L || is.na(out[[name]])) {
      stop(sprintf("policy_control$%s must be TRUE or FALSE.", name), call. = FALSE)
    }
  }
  if (length(out$se_chunk_size) != 1L || !is.numeric(out$se_chunk_size) ||
      !is.finite(out$se_chunk_size) || out$se_chunk_size < 1 ||
      out$se_chunk_size > .Machine$integer.max ||
      out$se_chunk_size != as.integer(out$se_chunk_size)) {
    stop("policy_control$se_chunk_size must be a positive integer.", call. = FALSE)
  }
  out$se_chunk_size <- as.integer(out$se_chunk_size)
  if (!is.null(out$.fit_cache) && !is.environment(out$.fit_cache)) {
    stop("Internal policy outcome cache must be an environment.", call. = FALSE)
  }
  out
}

cs_policy_module <- function() {
  if (is.null(strenv$policy_module)) {
    strenv$policy_module <- reticulate::import_from_path("strategize_policy",
      path = system.file("python", package = "strategize"), convert = TRUE)
  }
  strenv$policy_module
}

# This schedule is built once, before either optimization or differentiation.
# Reservoir draws use R's RNG, exactly as the original RAIN loop did.
cs_policy_schedule <- function(n, optimism, rain_lambda, rain_gamma,
                               rain_L, rain_eta, rain_output) {
  n <- as.integer(n)
  start <- choose <- rep(FALSE, n)
  end <- stage <- integer(n)
  lambda <- numeric(n)
  if (identical(optimism, "rain")) {
    L <- if (is.null(rain_L)) 1 / (8 * rain_eta) else rain_L
    cap <- Inf
    if (!is.null(rain_L) && rain_gamma > 0 && rain_lambda > 0) {
      cap <- max(1, ceiling(log(rain_L / rain_lambda) / log(1 + rain_gamma)))
    }
    i <- 0L; s <- 0L; lam <- rain_gamma * rain_lambda
    while (i < n && s < cap) {
      len <- if (lam > 0) ceiling(16 * L / lam) else n - i
      if (!is.finite(len) || len < 1) len <- 1L
      len <- as.integer(min(len, n - i))
      idx <- i + seq_len(len)
      start[idx[1L]] <- TRUE
      end[idx[len]] <- if (rain_output == "uniform_half") 2L else 1L
      stage[idx] <- s
      lambda[idx] <- lam
      if (rain_output == "uniform_half") choose[idx] <- stats::runif(len) <= 1 / seq_len(len)
      i <- i + len; s <- s + 1L; lam <- lam * (1 + rain_gamma)
      if (!is.finite(lam) || lam < 0) lam <- 0
    }
    n <- i
  }
  idx <- seq_len(n)
  out <- list(strenv$jnp$array(as.integer(idx - 1L)), strenv$jnp$array(start[idx]),
       strenv$jnp$array(end[idx]), strenv$jnp$array(lambda[idx], strenv$dtj),
       strenv$jnp$array(stage[idx]), strenv$jnp$array(choose[idx]))
  # reticulate unboxes length-one R vectors; scan still needs a leading axis.
  out <- lapply(out, function(x) strenv$jnp$reshape(x, list(n)))
  attr(out, "choose") <- choose[idx]
  out
}

cs_policy_store_history <- function(result, control, adversarial, optimism, nSGD) {
  # A single tree transfer replaces thousands of scalar Python calls.
  history <- reticulate::py_to_r(strenv$jax$device_get(result$history))
  mapping <- c(loss_ast = "loss_ast_vec", loss_dag = "loss_dag_vec",
    grad_ast = "grad_mag_ast_vec", grad_dag = "grad_mag_dag_vec",
    inv_lr_ast = "inv_learning_rate_ast_vec", inv_lr_dag = "inv_learning_rate_dag_vec",
    gamma_ast = "smp_gamma_ast_vec", gamma_dag = "smp_gamma_dag_vec",
    rain_lambda = "rain_lambda_vec", rain_lambda_sum = "rain_lambda_sum_vec",
    rain_stage_idx = "rain_stage_idx", rain_anchor_bar_norm_ast = "rain_anchor_bar_norm_ast",
    rain_anchor_bar_norm_dag = "rain_anchor_bar_norm_dag")
  for (key in names(mapping)) {
    value <- history[[key]]
    if (is.null(value)) value <- rep(NA_real_, nSGD)
    if (length(value) < nSGD) value <- c(value, rep(NA_real_, nSGD - length(value)))
    assign(mapping[[key]], as.numeric(value), envir = strenv)
  }
  strenv$rain_lambda_s_vec <- strenv$rain_lambda_vec
  strenv$policy_loop <- "scan"
  strenv$extragrad_eval_points <- NULL
  if (isTRUE(control$trace) && !is.null(history$trace)) {
    tr <- history$trace
    take <- function(x, i) matrix(x[i, , ], ncol = 1L)
    strenv$extragrad_eval_points <- lapply(seq_len(dim(tr$start_ast)[1L]), function(i) {
      pred <- list(a_pred_ast = take(tr$pred_ast, i), a_pred_dag = take(tr$pred_dag, i))
      list(start = list(a_ast = take(tr$start_ast, i), a_dag = take(tr$start_dag, i)),
           ast = pred, dag = pred)
    })
  }
  if (optimism == "smp") {
    strenv$smp_avg_ast <- result$a
    strenv$smp_avg_dag <- if (adversarial) result$b else NULL
    strenv$smp_sum_gamma_ast <- result$weight_a
    strenv$smp_sum_gamma_dag <- if (adversarial) result$weight_b else NULL
  }
}

cs_policy_numeric_history <- function(x) {
  if (is.null(x)) return(numeric(0))
  if (is.atomic(x)) return(as.numeric(x))
  if (is.list(x)) {
    x <- lapply(x, function(value) {
      if (is.null(value)) return(strenv$jnp$array(NA_real_))
      while (is.list(value)) value <- value[[1L]]
      strenv$jnp$reshape(strenv$jnp$array(value), list())
    })
    x <- strenv$jnp$stack(x)
  }
  as.numeric(strenv$np$array(x))
}

cs_policy_covariance <- function(jacobian, blocks) {
  # Independent blocks stay independent, including diagonal variances.
  out <- matrix(0, nrow(jacobian), nrow(jacobian))
  offset <- 0L
  for (block in blocks) {
    n <- if (is.null(dim(block))) length(block) else nrow(block)
    J <- jacobian[, offset + seq_len(n), drop = FALSE]
    out <- out + if (is.null(dim(block))) {
      tcrossprod(sweep(J, 2L, as.numeric(block), `*`), J)
    } else J %*% block %*% t(J)
    offset <- offset + n
  }
  out
}

# Cache only fitted outputs, not a strategize() call environment or optimizer.
# Each cache belongs to exactly one CV partition and lives for that CV call.
cs_policy_fit_glm <- function(env, cache = NULL) {
  key <- paste(env$Round_, env$GroupCounter, sep = ":")
  signature <- if (!is.null(cache)) {
    inputs <- c("Y", "W", "X", "Y_", "W_", "p_list", "factor_levels", "names_list",
      "pair_id", "profile_order", "respondent_id", "respondent_task_id",
      "varcov_cluster_variable", "competing_group_variable_respondent",
      "competing_group_variable_candidate", "competing_group_competition_variable_candidate",
      "K", "diff", "adversarial", "adversarial_model_strategy", "glm_family",
      "include_stage_interactions", "nFolds_glm", "holdout_indicator",
      "use_regularization", "force_no_interactions", "presaved_outcome_model", "outcome_model_key")
    inputs <- inputs[vapply(inputs, exists, logical(1), envir = env, inherits = TRUE)]
    digest::digest(mget(inputs, envir = env, inherits = TRUE))
  } else NULL
  if (!is.null(cache) && exists(key, cache, inherits = FALSE)) {
    entry <- get(key, cache, inherits = FALSE)
    if (identical(entry$signature, signature)) {
      list2env(entry$fields, envir = env)
      # A cache hit consumes the same R random stream as the original fit, so
      # policy initialization and RAIN reservoir draws remain reproducible.
      if (!is.null(entry$rng_after)) assign(".Random.seed", entry$rng_after, .GlobalEnv)
      return(invisible(NULL))
    }
  }
  eval(body(generate_ModelOutcome), envir = env)
  if (!is.null(cache)) {
    fields <- c("vcov_OutcomeModel", "vcov_OutcomeModel_by_k", "vcov_OutcomeModel_general",
      "main_info", "interaction_info", "interaction_info_PreRegularization",
      "main_info_PreRegularization", "main_info_all", "main_info_inter",
      "a_structure", "a_structure_leftoutLdminus1", "heldout_levels_list",
      "regularization_adjust_hash", "regularization_adjust_hash_PreRegularization",
      "main_dat", "my_mean", "my_mean_full", "EST_INTERCEPT_tf", "EST_COEFFICIENTS_tf",
      "EST_INTERCEPT_tf_general", "EST_COEFFICIENTS_tf_general", "my_model",
      "neural_model_info", "fit_metrics", "UsedRegularization", "use_regularization",
      "point_est_predict_clust", "factorhet_moderator_columns", "factorhet_dropped_moderator_columns")
    fields <- fields[vapply(fields, exists, logical(1), envir = env, inherits = FALSE)]
    assign(key, list(signature = signature, fields = mget(fields, envir = env,
      inherits = FALSE), rng_after = if (exists(".Random.seed", .GlobalEnv, inherits = FALSE)) {
        get(".Random.seed", .GlobalEnv)
      } else NULL), envir = cache)
  }
  invisible(NULL)
}

cs_policy_with_seed <- function(seed, expr) {
  had_seed <- exists(".Random.seed", .GlobalEnv, inherits = FALSE)
  old_seed <- if (had_seed) get(".Random.seed", .GlobalEnv) else NULL
  on.exit({
    if (had_seed) assign(".Random.seed", old_seed, .GlobalEnv)
    else if (exists(".Random.seed", .GlobalEnv, inherits = FALSE)) rm(".Random.seed", envir = .GlobalEnv)
  }, add = TRUE)
  set.seed(seed)
  force(expr)
}

cs_policy_eval_state <- function() {
  fields <- c("nUniqueFactors", "nUniqueLevelsByFactors", "d_locator_use",
    "ParameterizationType", "main_comp_mat", "shadow_comp_mat", "AstProp", "DagProp",
    "Vectorized_QMonteIter_MaxMin", "getMultinomialSamp", "getMultinomialSampHard")
  fields <- fields[vapply(fields, exists, logical(1), envir = strenv, inherits = FALSE)]
  mget(fields, envir = strenv, inherits = FALSE)
}

cs_policy_evaluation_context <- function(env) {
  out <- list(gather_fxn = env$gather_fxn, QFXN = env$QFXN, glm_family = env$glm_family,
    ParameterizationType = strenv$ParameterizationType, d_locator_use = strenv$d_locator_use,
    lambda = env$lambda, .policy_eval_state = cs_policy_eval_state())
  mapping <- c(REGRESSION_PARAMETERS_ast = "REGRESSION_PARAMS_jax_ast_jnp",
    REGRESSION_PARAMETERS_dag = "REGRESSION_PARAMS_jax_dag_jnp",
    REGRESSION_PARAMETERS_ast0 = "REGRESSION_PARAMS_jax_ast0_jnp",
    REGRESSION_PARAMETERS_dag0 = "REGRESSION_PARAMS_jax_dag0_jnp",
    P_VEC_FULL_ast = "p_vec_full_ast_jnp", P_VEC_FULL_dag = "p_vec_full_dag_jnp",
    SLATE_VEC_ast = "SLATE_VEC_ast_jnp", SLATE_VEC_dag = "SLATE_VEC_dag_jnp")
  for (name in names(mapping)) out[[name]] <- get(mapping[[name]], envir = env)
  out
}
