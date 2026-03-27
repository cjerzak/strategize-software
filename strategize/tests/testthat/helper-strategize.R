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

  # Assign each unique respondent to a party (as close to 50/50 as possible)
  n_party_a <- ceiling(n_unique_respondents / 2)
  n_party_b <- floor(n_unique_respondents / 2)
  respondent_party_base <- c(rep("PartyA", n_party_a), rep("PartyB", n_party_b))
  competing_group_variable_respondent <- respondent_party_base[base_data$respondent_id]

  # Determine competition type for each pair
  competition_type_base <- sample(c(rep("Same", n_party_a), rep("Different", n_party_b)))
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
  suppressMessages(create_p_list(W, uniform = TRUE))
}

#' Build pairwise interaction columns for a binary factor matrix
#'
#' @param X Numeric matrix
#' @param KChoose2_combs Two-row matrix of pair indices from combn()
#' @return Numeric interaction matrix
build_pairwise_interaction_matrix <- function(X, KChoose2_combs) {
  X <- as.matrix(X)
  X_inter <- apply(KChoose2_combs, 2, function(idx) {
    X[, idx[1]] * X[, idx[2]]
  })
  if (is.null(dim(X_inter))) {
    X_inter <- matrix(X_inter, ncol = 1L)
  }
  X_inter
}

#' Generate the deterministic linear average-case fixture used in OptimizingSI
#'
#' This reproduces the misspecification-0 average-case branch of the linear
#' simulation design inside package tests without sourcing external code.
#'
#' @param n_obs Number of observations
#' @param k_factors Number of binary factors
#' @param monte_i Monte Carlo index used in the simulation seed path
#' @return List with deterministic fixture data and oracle quantities
generate_linear_average_case_fixture <- function(n_obs = 2000L,
                                                 k_factors = 10L,
                                                 monte_i = 1L) {
  seed_exists <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  if (seed_exists) {
    seed_value <- get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  }
  on.exit({
    if (seed_exists) {
      assign(".Random.seed", seed_value, envir = .GlobalEnv)
    } else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
      rm(".Random.seed", envir = .GlobalEnv)
    }
  }, add = TRUE)

  SEED_SCALER <- 10L
  treatProb <- 0.5
  sigma2 <- 0.1^2
  LinearR2_TARGET <- 0.75
  TARGETQ <- 1
  penaltyType <- "L2"
  KChoose2_combs <- combn(seq_len(k_factors), 2)

  getInteractionWts <- function(pi) {
    c(pi, pi[KChoose2_combs[1, ]] * pi[KChoose2_combs[2, ]])
  }

  penalty <- function(pi, lambda) {
    lambda * sum(c((pi - 0.5)^2, ((1 - pi) - 0.5)^2))
  }

  outer_seed <- SEED_SCALER * k_factors^2
  set.seed(outer_seed)
  beta_master <- rnorm(20L + choose(20L, 2), sd = 1)
  my_beta <- my_beta_orig <- beta_master[seq_len(k_factors + choose(k_factors, 2))]

  X_calibration <- matrix(
    rbinom(k_factors * 10000L, size = 1, prob = treatProb),
    nrow = 10000L
  )
  X_inter_calibration <- build_pairwise_interaction_matrix(
    X_calibration,
    KChoose2_combs
  )
  NormalizedInteractionWts <- my_beta_orig[-seq_len(k_factors)] /
    sqrt(sum(my_beta_orig[-seq_len(k_factors)]^2))
  interactionWts_vec <- 10^seq(-5, 5, length.out = 100)
  proposalR2_vec <- vapply(interactionWts_vec, function(scale_value) {
    tmp_beta <- my_beta_orig
    tmp_beta[-seq_len(k_factors)] <- NormalizedInteractionWts * scale_value
    Y_signal <- cbind(X_calibration, X_inter_calibration) %*% tmp_beta
    summary(lm(Y_signal ~ X_calibration))$adj.r.squared
  }, numeric(1))
  my_beta_orig[-seq_len(k_factors)] <- NormalizedInteractionWts *
    interactionWts_vec[which.min(abs(proposalR2_vec - LinearR2_TARGET))[1]]
  my_beta <- my_beta_orig

  getQ <- function(pi) {
    sum(my_beta * getInteractionWts(pi))
  }

  LambdaProposal_seq <- 10^seq(-3, 2, length.out = 500)
  SolAtLambdaProposal <- matrix(
    NA_real_,
    nrow = length(LambdaProposal_seq),
    ncol = k_factors
  )

  for (lambda_index in seq_along(LambdaProposal_seq)) {
    lambda_value <- LambdaProposal_seq[lambda_index]
    my_beta_main <- my_beta[seq_len(k_factors)]
    my_beta_inter <- my_beta[-seq_len(k_factors)]
    COEF_MAT <- sapply(seq_len(k_factors), function(k_index) {
      zero_vec <- rep(0, k_factors)
      zero_vec[k_index] <- -4 * lambda_value
      interindices_ref <- which(
        KChoose2_combs[1, ] == k_index | KChoose2_combs[2, ] == k_index
      )
      interindices <- vapply(interindices_ref, function(interaction_index) {
        KChoose2_combs[, interaction_index][KChoose2_combs[, interaction_index] != k_index]
      }, integer(1))
      zero_vec[interindices] <- my_beta_inter[interindices_ref]
      zero_vec
    })
    B_VEC <- -4 * lambda_value * rep(0.5, k_factors) - my_beta_main
    SolAtLambdaProposal[lambda_index, ] <- as.numeric(
      solve(COEF_MAT, as.matrix(B_VEC))
    )
  }

  whichEligible <- which(apply(SolAtLambdaProposal, 1, function(pi_star) {
    all(pi_star > 0.15) && all(pi_star < 0.85) && sd(pi_star) > 0.1
  }))
  impliedQ_vec <- apply(SolAtLambdaProposal[whichEligible, , drop = FALSE], 1, getQ)
  whichSelected <- whichEligible[which.min(abs(impliedQ_vec - TARGETQ))]
  lambda <- LambdaProposal_seq[whichSelected]
  pi_star_true <- as.numeric(SolAtLambdaProposal[whichSelected, ])
  trueQ <- as.numeric(getQ(pi_star_true))

  inner_seed <- SEED_SCALER * (k_factors * n_obs * monte_i)
  set.seed(inner_seed)
  X_obs <- matrix(
    rbinom(k_factors * n_obs, size = 1, prob = treatProb),
    nrow = n_obs
  )
  X_inter_obs <- build_pairwise_interaction_matrix(X_obs, KChoose2_combs)
  mu_true <- as.numeric(cbind(X_obs, X_inter_obs) %*% my_beta)
  observation_seed <- inner_seed + 303L
  set.seed(observation_seed)
  Yobs <- mu_true + rnorm(n_obs, sd = sqrt(sigma2))

  W <- apply(X_obs, 2, as.character)
  colnames(W) <- as.character(seq_len(k_factors))

  list(
    Y = as.numeric(Yobs),
    W = W,
    lambda = as.numeric(lambda),
    pi_star_true = pi_star_true,
    trueQ = trueQ,
    mu_true = as.numeric(mu_true),
    my_beta = as.numeric(my_beta),
    KChoose2_combs = KChoose2_combs,
    outer_seed = as.integer(outer_seed),
    inner_seed = as.integer(inner_seed),
    getInteractionWts = getInteractionWts,
    penalty = penalty,
    penaltyType = penaltyType
  )
}

# =============================================================================
# Common Strategize Call Parameters
# =============================================================================

#' Get default strategize parameters for testing
#'
#' @param fast Logical; if TRUE, use minimal iterations for faster tests
#' @return Named list of default parameters
default_strategize_params <- function(fast = TRUE) {
  params <- list(
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

  if (fast) {
    params$neural_mcmc_control <- list(
      n_samples_warmup = 10L,
      n_samples_mcmc = 10L
    )
  }

  params
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
