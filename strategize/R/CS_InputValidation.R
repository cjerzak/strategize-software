#' Input Validation for strategize Functions
#'
#' @description
#' Internal validation functions that provide clear, actionable error messages
#' for common input mistakes. These functions are called at the entry points
#' of main package functions to catch errors early.
#'
#' @name input-validation
#' @keywords internal
NULL

#' Validate inputs for strategize()
#'
#' @param Y Outcome vector
#' @param W Factor matrix
#' @param X Optional covariate matrix
#' @param lambda Regularization parameter
#' @param p_list Baseline probability list
#' @param K Number of clusters
#' @param pair_id Pair identifier
#' @param adversarial Adversarial mode flag
#' @param competing_group_variable_respondent Respondent group variable
#' @param competing_group_variable_candidate Candidate group variable
#' @param outcome_model_type Outcome model type
#' @param penalty_type Penalty type
#' @param diff Difference mode flag
#' @param primary_pushforward Primary-stage push-forward estimator
#' @return TRUE invisibly if validation passes; stops with error otherwise
#' @keywords internal
validate_strategize_inputs <- function(Y, W, X = NULL, lambda,
                                       p_list = NULL, K = 1,
                                       pair_id = NULL,
                                       adversarial = FALSE,
                                       competing_group_variable_respondent = NULL,
                                       competing_group_variable_candidate = NULL,
                                       outcome_model_type = "glm",
                                       penalty_type = "KL",
                                       diff = FALSE,
                                       primary_pushforward = "mc") {

  # ---- Y validation ----
  if (missing(Y) || is.null(Y)) {
    stop(
      "'Y' is required: provide a numeric outcome vector.\n",
      "  Y should be the observed outcomes (e.g., 0/1 for binary choice).",
      call. = FALSE
    )
  }
  if (!is.numeric(Y) && !is.integer(Y)) {
    stop(
      "'Y' must be numeric. Got: ", class(Y)[1], "\n",
      "  Convert Y to numeric with as.numeric() if needed.",
      call. = FALSE
    )
  }
  if (length(Y) == 0) {
    stop("'Y' cannot be empty.", call. = FALSE)
  }
  n_na <- sum(is.na(Y))
  if (n_na > 0) {
    warning(
      sprintf("'Y' contains %d NA values (%.1f%%). ", n_na, 100 * n_na / length(Y)),
      "These may affect results.",
      call. = FALSE
    )
  }

  # ---- W validation ----
  if (missing(W) || is.null(W)) {
    stop(
      "'W' is required: provide a factor matrix (data.frame or matrix).\n",
      "  W should have one column per conjoint factor.",
      call. = FALSE
    )
  }
  if (!is.data.frame(W) && !is.matrix(W)) {
    stop(
      "'W' must be a data.frame or matrix. Got: ", class(W)[1], "\n",
      "  Convert W with as.data.frame() or as.matrix().",
      call. = FALSE
    )
  }
  if (nrow(W) != length(Y)) {
    stop(
      sprintf("Dimension mismatch: Y has %d elements but W has %d rows.\n", length(Y), nrow(W)),
      "  Ensure each row of W corresponds to one element of Y.\n",
      "  For paired forced-choice, each profile appears as a separate row.",
      call. = FALSE
    )
  }
  if (ncol(W) < 1) {
    stop("'W' must have at least 1 column (factor).", call. = FALSE)
  }

  # ---- lambda validation ----
  if (missing(lambda) || is.null(lambda)) {
    stop(
      "'lambda' is required: provide a non-negative regularization value.\n",
      "  Tip: use cv_strategize() to select lambda via cross-validation.\n",
      "  Common values range from 0.01 to 1.0.",
      call. = FALSE
    )
  }
  if (!is.numeric(lambda) || length(lambda) != 1) {
    stop(
      "'lambda' must be a single numeric value. Got: ",
      paste(head(lambda, 3), collapse = ", "),
      if (length(lambda) > 3) "..." else "",
      call. = FALSE
    )
  }
  if (lambda < 0) {
    stop(
      "'lambda' must be non-negative. Got: ", lambda, "\n",
      "  Use lambda = 0 for no regularization (may overfit).",
      call. = FALSE
    )
  }

  # ---- p_list validation ----
  if (!is.null(p_list)) {
    if (!is.list(p_list)) {
      stop(
        "'p_list' must be a list of named probability vectors.\n",
        "  Each element should correspond to a factor in W.\n",
        "  Example: list(Gender = c(Male = 0.5, Female = 0.5))",
        call. = FALSE
      )
    }
    if (length(p_list) != ncol(W)) {
      stop(
        sprintf("'p_list' has %d elements but W has %d columns.\n", length(p_list), ncol(W)),
        "  Each factor in W needs a corresponding probability vector in p_list.\n",
        "  Tip: use create_p_list(W) to auto-generate from data.",
        call. = FALSE
      )
    }

    # Check level alignment for each factor
    for (i in seq_along(p_list)) {
      factor_name <- if (!is.null(colnames(W))) colnames(W)[i] else paste0("Column ", i)
      factor_levels <- sort(unique(as.character(W[, i])))
      p_levels <- sort(names(p_list[[i]]))

      if (is.null(names(p_list[[i]]))) {
        stop(
          sprintf("p_list[[%d]] ('%s') must have named elements.\n", i, factor_name),
          "  Example: c(Male = 0.5, Female = 0.5)",
          call. = FALSE
        )
      }

      if (!setequal(factor_levels, p_levels)) {
        missing_in_p <- setdiff(factor_levels, p_levels)
        extra_in_p <- setdiff(p_levels, factor_levels)
        msg <- sprintf("Factor '%s' (column %d) level mismatch:", factor_name, i)
        if (length(missing_in_p) > 0) {
          msg <- paste0(msg, "\n  Missing in p_list: ", paste(missing_in_p, collapse = ", "))
        }
        if (length(extra_in_p) > 0) {
          msg <- paste0(msg, "\n  Extra in p_list (not in W): ", paste(extra_in_p, collapse = ", "))
        }
        msg <- paste0(msg, "\n  Tip: use create_p_list(W) to auto-generate aligned p_list.")
        stop(msg, call. = FALSE)
      }

      # Check probabilities sum to 1
      prob_sum <- sum(p_list[[i]])
      if (abs(prob_sum - 1) > 1e-4) {
        warning(
          sprintf("p_list[[%d]] ('%s') probabilities sum to %.4f, not 1.0.\n", i, factor_name, prob_sum),
          "  Probabilities should sum to 1 for valid distribution.",
          call. = FALSE
        )
      }

      # Check for negative probabilities
      if (any(p_list[[i]] < 0)) {
        stop(
          sprintf("p_list[[%d]] ('%s') contains negative probabilities.\n", i, factor_name),
          "  All probabilities must be >= 0.",
          call. = FALSE
        )
      }
    }
  }

  # ---- adversarial mode validation ----
  if (isTRUE(adversarial)) {
    if (is.null(competing_group_variable_respondent)) {
      stop(
        "adversarial=TRUE requires 'competing_group_variable_respondent'.\n",
        "  This should be a vector indicating each respondent's group (e.g., party).\n",
        "  Length must match number of rows in W.",
        call. = FALSE
      )
    }
    if (is.null(competing_group_variable_candidate)) {
      stop(
        "adversarial=TRUE requires 'competing_group_variable_candidate'.\n",
        "  This should be a vector indicating each candidate profile's group.\n",
        "  Length must match number of rows in W.",
        call. = FALSE
      )
    }

    n_resp_groups <- length(unique(competing_group_variable_respondent))
    if (n_resp_groups != 2) {
      stop(
        sprintf(
          "Adversarial mode requires exactly 2 groups in competing_group_variable_respondent.\n"
        ),
        sprintf("  Found %d groups: %s\n", n_resp_groups,
                paste(head(unique(competing_group_variable_respondent), 5), collapse = ", ")),
        "  Adversarial mode models a two-player zero-sum game.",
        call. = FALSE
      )
    }

    if (length(competing_group_variable_respondent) != nrow(W)) {
      stop(
        sprintf(
          "competing_group_variable_respondent has %d elements but W has %d rows.\n",
          length(competing_group_variable_respondent), nrow(W)
        ),
        "  Lengths must match.",
        call. = FALSE
      )
    }
  }

  # ---- diff mode validation ----
  if (isTRUE(diff) && is.null(pair_id)) {
    warning(
      "diff=TRUE typically requires 'pair_id' to identify forced-choice pairs.\n",
      "  Without pair_id, the function may not correctly identify paired profiles.",
      call. = FALSE
    )
  }

  # ---- K and X validation ----
  if (!is.numeric(K) || length(K) != 1 || K < 1 || K != round(K)) {
    stop(
      "'K' must be a positive integer. Got: ", K,
      call. = FALSE
    )
  }
  if (K > 1 && is.null(X)) {
    warning(
      "K > 1 (multi-cluster) typically requires 'X' (respondent covariates).\n",
      "  Without X, cluster identification may be unreliable.",
      call. = FALSE
    )
  }

  # ---- outcome_model_type validation ----
  valid_model_types <- c("glm", "neural")
  if (!outcome_model_type %in% valid_model_types) {
    stop(
      sprintf("'outcome_model_type' must be one of: %s.\n",
              paste(valid_model_types, collapse = ", ")),
      sprintf("  Got: '%s'", outcome_model_type),
      call. = FALSE
    )
  }

  # ---- penalty_type validation ----
  valid_penalty_types <- c("KL", "L2", "LogMaxProb")
  if (!penalty_type %in% valid_penalty_types) {
    stop(
      sprintf("'penalty_type' must be one of: %s.\n",
              paste(valid_penalty_types, collapse = ", ")),
      sprintf("  Got: '%s'\n", penalty_type),
      "  KL = Kullback-Leibler divergence (default)\n",
      "  L2 = Euclidean distance\n",
      "  LogMaxProb = Log maximum probability",
      call. = FALSE
    )
  }

  # ---- primary_pushforward validation ----
  valid_pushforward <- c("mc", "linearized")
  if (!is.character(primary_pushforward) || length(primary_pushforward) != 1) {
    stop(
      "'primary_pushforward' must be a single character string.\n",
      "  Valid options: ", paste(valid_pushforward, collapse = ", "),
      call. = FALSE
    )
  }
  if (!tolower(primary_pushforward) %in% valid_pushforward) {
    stop(
      sprintf("'primary_pushforward' must be one of: %s.\n",
              paste(valid_pushforward, collapse = ", ")),
      sprintf("  Got: '%s'", primary_pushforward),
      call. = FALSE
    )
  }

  invisible(TRUE)
}


#' Validate inputs for cv_strategize()
#'
#' @param Y Outcome vector
#' @param W Factor matrix
#' @param lambda_seq Sequence of lambda values to cross-validate
#' @param folds Number of CV folds
#' @return TRUE invisibly if validation passes
#' @keywords internal
validate_cv_strategize_inputs <- function(Y, W, lambda_seq = NULL, folds = 2L) {

  # Basic Y/W validation (reuse from main function)
  if (missing(Y) || is.null(Y)) {
    stop("'Y' is required.", call. = FALSE)
  }
  if (missing(W) || is.null(W)) {
    stop("'W' is required.", call. = FALSE)
  }

  # folds validation
  if (!is.numeric(folds) || length(folds) != 1 || folds < 2) {
    stop(
      "'folds' must be an integer >= 2. Got: ", folds, "\n",
      "  Use folds = 2 for quick validation, folds = 5 or 10 for thorough CV.",
      call. = FALSE
    )
  }

  n_obs <- length(Y)
  if (folds > n_obs) {
    stop(
      sprintf("'folds' (%d) cannot exceed number of observations (%d).", folds, n_obs),
      call. = FALSE
    )
  }

  # lambda_seq validation
  if (!is.null(lambda_seq)) {
    if (!is.numeric(lambda_seq)) {
      stop("'lambda_seq' must be a numeric vector.", call. = FALSE)
    }
    if (any(lambda_seq < 0)) {
      stop("'lambda_seq' values must be non-negative.", call. = FALSE)
    }
    if (length(lambda_seq) < 2) {
      warning(
        "lambda_seq has only 1 value. Cross-validation is most useful with multiple values.\n",
        "  Consider: lambda_seq = c(0.01, 0.1, 0.5, 1.0)",
        call. = FALSE
      )
    }
  }

  invisible(TRUE)
}


#' Check if JAX/conda environment is available
#'
#' @param conda_env Name of conda environment
#' @param required If TRUE, stops with error; if FALSE, returns FALSE
#' @return TRUE if available, FALSE otherwise (or stops if required)
#' @keywords internal
check_jax_available <- function(conda_env = "strategize_env", required = FALSE) {
  # Check if strenv exists and has jnp
  if (exists("strenv", envir = .GlobalEnv) || exists("strenv", envir = parent.frame())) {
    env <- if (exists("strenv", envir = .GlobalEnv)) get("strenv", envir = .GlobalEnv)
           else get("strenv", envir = parent.frame())
    if ("jnp" %in% ls(envir = env)) {
      return(TRUE)
    }
  }

  # Check if reticulate can find the environment
  available <- tryCatch({
    envs <- reticulate::conda_list()
    conda_env %in% envs$name
  }, error = function(e) FALSE)

  if (!available && required) {
    stop(
      sprintf("Conda environment '%s' not found or JAX not initialized.\n\n", conda_env),
      "To set up the environment, run:\n",
      sprintf("  strategize::build_backend(conda_env = '%s')\n\n", conda_env),
      "To check available environments:\n",
      "  reticulate::conda_list()\n\n",
      "If you've already set up the environment, ensure conda is in your PATH.",
      call. = FALSE
    )
  }

  available
}
