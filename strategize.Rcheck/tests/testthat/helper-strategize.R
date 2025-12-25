# =============================================================================
# Test Helpers for the strategize Package
# =============================================================================
# This file is automatically loaded by testthat before running tests.
# It contains common utilities, skip conditions, and data generators.
# =============================================================================

# =============================================================================
# Skip Conditions
# =============================================================================

#' Skip tests if conda environment is not available
skip_if_no_conda <- function(conda_env = "strategize_env") {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    skip("reticulate package not available")
  }

  conda_list <- tryCatch(
    reticulate::conda_list(),
    error = function(e) NULL
  )

  if (is.null(conda_list) || !conda_env %in% conda_list$name) {
    skip(paste0("Conda environment '", conda_env, "' not available"))
  }
}

#' Skip tests if JAX is not available
skip_if_no_jax <- function(conda_env = "strategize_env") {
  skip_if_no_conda(conda_env)

  jax_available <- tryCatch({
    reticulate::use_condaenv(conda_env, required = TRUE)
    reticulate::py_module_available("jax")
  }, error = function(e) FALSE)

  if (!isTRUE(jax_available)) {
    skip("JAX not available in conda environment")
  }
}

#' Skip slow tests unless explicitly enabled
skip_if_slow <- function() {
  if (!identical(Sys.getenv("STRATEGIZE_RUN_SLOW_TESTS"), "true")) {
    skip("Slow test (set STRATEGIZE_RUN_SLOW_TESTS=true to run)")
  }
}

#' Skip tests if tgp package is not available (required for K > 1 multi-cluster)
skip_if_no_tgp <- function() {
  if (!requireNamespace("tgp", quietly = TRUE)) {
    skip("tgp package not available (required for K > 1 multi-cluster tests)")
  }
}

#' Skip one-step estimator tests (require larger datasets and more tuning)
skip_onestep_tests <- function() {
  skip("One-step estimator tests require larger datasets (set STRATEGIZE_RUN_ONESTEP_TESTS=true to run)")
}

# =============================================================================
# Test Data Generators
# =============================================================================

#' Generate standard conjoint test data
#'
#' @param n Number of observations (profiles)
#' @param n_factors Number of treatment factors
#' @param n_levels Number of levels per factor
#' @param seed Random seed
#' @return List with Y, W, pair_id, respondent_id, respondent_task_id, profile_order
generate_test_data <- function(n = 1000, n_factors = 3, n_levels = 2, seed = 1234321) {
  withr::local_seed(seed)

  # Generate factor matrix
  levels <- LETTERS[seq_len(n_levels)]
  W <- matrix(
    sample(levels, n * n_factors, replace = TRUE),
    nrow = n,
    ncol = n_factors
  )
  colnames(W) <- paste0("V", seq_len(n_factors))

  # Generate IDs for forced-choice pairs
  n_pairs <- n / 2
  pair_id <- c(seq_len(n_pairs), seq_len(n_pairs))
  respondent_id <- pair_id
  respondent_task_id <- pair_id
  profile_order <- c(rep(1L, n_pairs), rep(2L, n_pairs))

  # Generate outcome based on factor effects
  effects <- seq(0.4, 0.2, length.out = n_factors)
  latent_utility <- drop((W == "B") %*% effects)

  Y <- as.numeric(ave(
    latent_utility,
    respondent_task_id,
    FUN = function(g) rank(g, ties.method = "random") == length(g)
  ))

  list(
    Y = Y,
    W = W,
    pair_id = pair_id,
    respondent_id = respondent_id,
    respondent_task_id = respondent_task_id,
    profile_order = profile_order
  )
}

#' Generate test data with respondent-level covariates for K > 1
#'
#' @param base_data Output from generate_test_data
#' @param n_covariates Number of respondent-level covariates
#' @param seed Random seed
#' @return Base data with X matrix added
add_respondent_covariates <- function(base_data, n_covariates = 3, seed = 42) {
  withr::local_seed(seed)

  n_respondents <- length(unique(base_data$respondent_id))
  X_base <- matrix(rnorm(n_respondents * n_covariates), ncol = n_covariates)
  colnames(X_base) <- paste0("X", seq_len(n_covariates))

  # Duplicate for both profiles per respondent
  base_data$X <- X_base[base_data$respondent_id, ]
  base_data
}

#' Generate adversarial mode test data
#'
#' @param base_data Output from generate_test_data
#' @param seed Random seed
#' @return Base data with adversarial variables added
add_adversarial_structure <- function(base_data, seed = 42) {
  withr::local_seed(seed)

  n <- length(base_data$Y)
  n_pairs <- n / 2
  n_unique_respondents <- n_pairs

  # Assign each unique respondent to a party (50/50 split)
  respondent_party_base <- rep(c("PartyA", "PartyB"), each = n_unique_respondents / 2)
  competing_group_variable_respondent <- respondent_party_base[base_data$respondent_id]

  # Determine competition type for each pair
  competition_type_base <- sample(rep(c("Same", "Different"), each = n_unique_respondents / 2))
  pair_competition_type <- competition_type_base[base_data$pair_id]

  # Assign candidate party based on competition type
  competing_group_variable_candidate <- ifelse(
    pair_competition_type == "Same",
    competing_group_variable_respondent,
    ifelse(base_data$profile_order == 1, "PartyA", "PartyB")
  )

  base_data$competing_group_variable_respondent <- competing_group_variable_respondent
  base_data$competing_group_variable_candidate <- competing_group_variable_candidate
  base_data$competing_group_competition_variable_candidate <- pair_competition_type
  base_data
}

#' Generate probability list from factor matrix
#'
#' Creates a list of probability distributions for each factor,
#' where each distribution is uniform over the factor's levels.
#' Returns a named vector format compatible with strategize() and strategize_onestep().
#'
#' @param W Factor matrix from generate_test_data
#' @return List of named probability vectors for each factor
generate_test_p_list <- function(W) {
  n_factors <- ncol(W)

  p_list <- lapply(seq_len(n_factors), function(d) {
    factor_col <- W[, d]
    levels_d <- sort(unique(factor_col))
    n_levels <- length(levels_d)

    # Create uniform probability vector (named)
    probs <- rep(1 / n_levels, n_levels)
    names(probs) <- levels_d
    probs
  })

  names(p_list) <- colnames(W)
  p_list
}

# =============================================================================
# Common Strategize Call Parameters
# =============================================================================

#' Get default strategize parameters for testing
#'
#' @param fast Logical; if TRUE, use minimal iterations for faster tests
#' @return Named list of default parameters
default_strategize_params <- function(fast = TRUE) {
  list(
    lambda = 0.1,
    K = 1,
    nSGD = if (fast) 5L else 100L,
    outcome_model_type = "glm",
    force_gaussian = TRUE,
    nMonte_adversarial = if (fast) 5L else 24L,
    nMonte_Qglm = if (fast) 5L else 100L,
    diff = TRUE,
    compute_se = FALSE,
    conda_env = "strategize_env",
    conda_env_required = TRUE
  )
}

# =============================================================================
# Assertion Helpers
# =============================================================================

#' Check that a probability distribution is valid
#'
#' @param probs Numeric vector of probabilities
#' @param tolerance Tolerance for sum-to-one check
expect_valid_probability <- function(probs, tolerance = 1e-6) {
  testthat::expect_true(all(probs >= 0), info = "All probabilities must be non-negative")
  testthat::expect_equal(sum(probs), 1, tolerance = tolerance,
                         info = "Probabilities must sum to 1")
}

#' Check that strategize output has expected structure
#'
#' @param res Result from strategize()
#' @param n_factors Expected number of factors
expect_valid_strategize_output <- function(res, n_factors = NULL) {
  testthat::expect_type(res, "list")
  testthat::expect_true("pi_star_point" %in% names(res))
  testthat::expect_true("Q_point" %in% names(res))
  testthat::expect_true("p_list" %in% names(res))
  testthat::expect_type(res$pi_star_point, "list")
  testthat::expect_type(res$p_list, "list")

  if (!is.null(n_factors)) {
    testthat::expect_equal(length(res$p_list), n_factors)
  }
}
