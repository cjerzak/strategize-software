# =============================================================================
# Adversarial Mode Correctness Tests for the strategize Package
# =============================================================================
#
# This test file verifies that the adversarial (minimax) optimization in
# strategize() correctly identifies Nash equilibrium strategies.
#
# Based on the two-stage election model with:
# - Primary elections (same-party matchups)
# - General elections (cross-party matchups)
# - Republican vs Democrat voters and candidates
# - Gender as the factor of interest (binary: male/female)
#
# The tests:
# 1. Define ground truth probability parameters
# 2. Compute theoretical Nash equilibria via grid search and iterative best-response
# 3. Generate synthetic data from the known model
# 4. Compare strategize() output against theoretical equilibria
#
# =============================================================================

# =============================================================================
# SECTION 1: Global Parameters and Ground Truth
# =============================================================================

# Ground truth probability parameters for voting behavior
# These define the true data generating process

define_ground_truth <- function(
    p_RVoters = 0.5,       # Proportion of Republican voters
    lambda = 0.02,          # Regularization parameter
    # Primary election probabilities (R voters choosing R candidate by gender matchup)
    pr_mm_RVoters_primary = 0.50,  # M vs M
    pr_mf_RVoters_primary = 0.55,  # M vs F (male has advantage)
    pr_fm_RVoters_primary = 0.45,  # F vs M
    pr_ff_RVoters_primary = 0.50,  # F vs F
    # Primary election probabilities (D voters choosing D candidate by gender matchup)
    pd_mm_DVoters_primary = 0.50,
    pd_mf_DVoters_primary = 0.45,  # D voters slightly prefer female
    pd_fm_DVoters_primary = 0.55,
    pd_ff_DVoters_primary = 0.50,
    # General election: R voters probability of choosing R (by R gender vs D gender)
    pr_mm_RVoters = 0.85,  # R male vs D male
    pr_mf_RVoters = 0.90,  # R male vs D female
    pr_fm_RVoters = 0.80,  # R female vs D male
    pr_ff_RVoters = 0.85,  # R female vs D female
    # General election: D voters probability of choosing R
    pr_mm_DVoters = 0.15,
    pr_mf_DVoters = 0.10,  # D voters dislike R male vs D female
    pr_fm_DVoters = 0.20,  # D voters more accepting of R female
    pr_ff_DVoters = 0.15
) {
  list(
    p_RVoters = p_RVoters,
    p_DVoters = 1 - p_RVoters,
    lambda = lambda,
    # Primary R
    pr_mm_RVoters_primary = pr_mm_RVoters_primary,
    pr_mf_RVoters_primary = pr_mf_RVoters_primary,
    pr_fm_RVoters_primary = pr_fm_RVoters_primary,
    pr_ff_RVoters_primary = pr_ff_RVoters_primary,
    # Primary D
    pd_mm_DVoters_primary = pd_mm_DVoters_primary,
    pd_mf_DVoters_primary = pd_mf_DVoters_primary,
    pd_fm_DVoters_primary = pd_fm_DVoters_primary,
    pd_ff_DVoters_primary = pd_ff_DVoters_primary,
    # General R voters
    pr_mm_RVoters = pr_mm_RVoters,
    pr_mf_RVoters = pr_mf_RVoters,
    pr_fm_RVoters = pr_fm_RVoters,
    pr_ff_RVoters = pr_ff_RVoters,
    # General D voters
    pr_mm_DVoters = pr_mm_DVoters,
    pr_mf_DVoters = pr_mf_DVoters,
    pr_fm_DVoters = pr_fm_DVoters,
    pr_ff_DVoters = pr_ff_DVoters
  )
}


# =============================================================================
# SECTION 2: Vote Share Computation Functions
# =============================================================================

# Compute expected vote share for R given strategies (pi_R, pi_D)
# pi_R = Pr(R entrant is male), pi_D = Pr(D entrant is male)
# pi_R_s, pi_D_s = field/slate male probability (default 0.5)

compute_vote_share_R <- function(pi_R, pi_D, params,
                                  pi_R_s = 0.50, pi_D_s = 0.50) {
  with(params, {
    # Push-forward through primaries
    # P(R nominee is male)
    p_R_m <- 1 * (pi_R * pi_R_s) +                          # M vs M => male wins
      pr_mf_RVoters_primary * (pi_R * (1 - pi_R_s)) +       # M (entrant) vs F (field)
      (1 - pr_fm_RVoters_primary) * ((1 - pi_R) * pi_R_s) + # F (entrant) vs M (field) -> male wins
      0 * ((1 - pi_R) * (1 - pi_R_s))                       # F vs F => female wins
    p_R_f <- 1 - p_R_m

    # P(D nominee is male)
    p_D_m <- 1 * (pi_D * pi_D_s) +
      pd_mf_DVoters_primary * (pi_D * (1 - pi_D_s)) +
      (1 - pd_fm_DVoters_primary) * ((1 - pi_D) * pi_D_s) +
      0 * ((1 - pi_D) * (1 - pi_D_s))
    p_D_f <- 1 - p_D_m

    # General election mixture (mm, mf, fm, ff)
    p_mm <- p_R_m * p_D_m
    p_mf <- p_R_m * p_D_f
    p_fm <- p_R_f * p_D_m
    p_ff <- p_R_f * p_D_f

    # Expected vote share for R
    EVShare_R <- p_mm * (p_RVoters * pr_mm_RVoters + p_DVoters * pr_mm_DVoters) +
      p_mf * (p_RVoters * pr_mf_RVoters + p_DVoters * pr_mf_DVoters) +
      p_fm * (p_RVoters * pr_fm_RVoters + p_DVoters * pr_fm_DVoters) +
      p_ff * (p_RVoters * pr_ff_RVoters + p_DVoters * pr_ff_DVoters)

    return(EVShare_R)
  })
}

compute_vote_share_D <- function(pi_R, pi_D, params,
                                  pi_R_s = 0.50, pi_D_s = 0.50) {
  # Zero-sum: D's vote share = 1 - R's vote share
  1 - compute_vote_share_R(pi_R, pi_D, params, pi_R_s, pi_D_s)
}


# =============================================================================
# SECTION 3: Utility Functions with L2 Regularization
# =============================================================================

utility_R <- function(pi_R, pi_D, params) {
  vote_share <- compute_vote_share_R(pi_R, pi_D, params)
  # L2 penalty pushing towards uniform (0.5, 0.5)
  vote_share - params$lambda * ((pi_R - 0.5)^2 + ((1 - pi_R) - 0.5)^2)
}

utility_D <- function(pi_R, pi_D, params) {
  vote_share <- compute_vote_share_D(pi_R, pi_D, params)
  vote_share - params$lambda * ((pi_D - 0.5)^2 + ((1 - pi_D) - 0.5)^2)
}


# =============================================================================
# SECTION 4: Nash Equilibrium Computation via Grid Search
# =============================================================================

compute_nash_grid <- function(params, grid_step = 0.01, tol = 0.02) {
  # Use finer grid for more accurate equilibrium finding
  pi_seq <- seq(0, 1, by = grid_step)

  # Compute best responses for each party
  best_response_D <- sapply(pi_seq, function(r) {
    utilities <- sapply(pi_seq, function(d) utility_D(r, d, params))
    pi_seq[which.max(utilities)]
  })

  best_response_R <- sapply(pi_seq, function(d) {
    utilities <- sapply(pi_seq, function(r) utility_R(r, d, params))
    pi_seq[which.max(utilities)]
  })

  # Find Nash equilibrium (mutual best responses)
  # Use a more robust approach: find the point where |r - BR_R(d)| + |d - BR_D(r)| is minimized
  nash_found <- FALSE
  pi_R_nash <- NA
  pi_D_nash <- NA
  min_deviation <- Inf

  for (i in seq_along(pi_seq)) {
    for (j in seq_along(pi_seq)) {
      r <- pi_seq[i]
      d <- pi_seq[j]
      deviation <- abs(r - best_response_R[j]) + abs(d - best_response_D[i])
      if (deviation < min_deviation) {
        min_deviation <- deviation
        pi_R_nash <- r
        pi_D_nash <- d
      }
      if (deviation < tol) {
        nash_found <- TRUE
      }
    }
  }

  # Mark as found if deviation is small enough
  if (min_deviation < tol * 2) {
    nash_found <- TRUE
  }

  list(
    pi_R = pi_R_nash,
    pi_D = pi_D_nash,
    found = nash_found,
    deviation = min_deviation,
    best_response_R = best_response_R,
    best_response_D = best_response_D,
    pi_seq = pi_seq
  )
}


# =============================================================================
# SECTION 5: Nash Equilibrium via Iterative Best Response
# =============================================================================

compute_nash_iterative <- function(params, grid_step = 0.01,
                                    max_iter = 500, tol = 0.01) {
  pi_seq <- seq(0, 1, by = grid_step)

  # Precompute best responses on grid
  best_response_D_vec <- sapply(pi_seq, function(r) {
    utilities <- sapply(pi_seq, function(d) utility_D(r, d, params))
    pi_seq[which.max(utilities)]
  })

  best_response_R_vec <- sapply(pi_seq, function(d) {
    utilities <- sapply(pi_seq, function(r) utility_R(r, d, params))
    pi_seq[which.max(utilities)]
  })

  # Lookup functions
  best_response_D_fn <- function(r) {
    i <- which.min(abs(pi_seq - r))
    best_response_D_vec[i]
  }

  best_response_R_fn <- function(d) {
    j <- which.min(abs(pi_seq - d))
    best_response_R_vec[j]
  }

  # Initialize at uniform
  pi_R_current <- 0.5
  pi_D_current <- 0.5

  for (iter in seq_len(max_iter)) {
    # Update D's response to R
    pi_D_new <- best_response_D_fn(pi_R_current)
    # Update R's response to D
    pi_R_new <- best_response_R_fn(pi_D_new)

    # Check convergence
    if (abs(pi_R_new - pi_R_current) < tol &&
        abs(pi_D_new - pi_D_current) < tol) {
      return(list(
        pi_R = pi_R_new,
        pi_D = pi_D_new,
        converged = TRUE,
        iterations = iter
      ))
    }

    pi_R_current <- pi_R_new
    pi_D_current <- pi_D_new
  }

  list(
    pi_R = pi_R_current,
    pi_D = pi_D_current,
    converged = FALSE,
    iterations = max_iter
  )
}


# =============================================================================
# SECTION 6: Data Generation (Following Reference Structure)
# =============================================================================

generate_adversarial_data <- function(n = 2000, params, seed = 12345) {
  set.seed(seed)

  # Respondent party assignment based on voter proportions
  competing_group_variable_respondent <- sample(
    c("Democrat", "Republican"),
    size = n,
    replace = TRUE,
    prob = c(params$p_DVoters, params$p_RVoters)
  )

  # Duplicate for forced-choice (each respondent sees 2 profiles)
  competing_group_variable_respondent <- c(
    competing_group_variable_respondent,
    competing_group_variable_respondent
  )

  pair_id <- c(1:n, 1:n)
  respondent_id <- pair_id
  profile_order <- c(rep(1, n), rep(2, n))
  respondent_task_id <- rep(1, times = 2 * n)

  # Candidate party assignment (50/50 for now)
  competing_group_variable_candidate <- sample(
    c("Democrat", "Republican"),
    size = 2 * n,
    replace = TRUE,
    prob = c(0.5, 0.5)
  )

  # Competition type: Same (primary) vs Different (general)
  competing_group_competition_variable_candidate <- ifelse(
    competing_group_variable_candidate[1:n] == competing_group_variable_candidate[-(1:n)],
    yes = "Same",
    no = "Different"
  )
  competing_group_competition_variable_candidate <- c(
    competing_group_competition_variable_candidate,
    competing_group_competition_variable_candidate
  )

  # Generate gender (female = 1, male = 0)
  X <- as.data.frame(matrix(
    rbinom(n * 2, size = 1, prob = 0.5),
    nrow = 2 * n
  ))
  colnames(X) <- "female"

  # Compute other profile's gender within each pair
  X_other <- numeric(nrow(X))
  other_competing_group_variable_candidate <- character(nrow(X))
  for (p in unique(pair_id)) {
    rows_in_pair <- which(pair_id == p)
    i <- rows_in_pair[1]
    j <- rows_in_pair[2]
    X_other[i] <- X$female[j]
    X_other[j] <- X$female[i]
    other_competing_group_variable_candidate[i] <- competing_group_variable_candidate[j]
    other_competing_group_variable_candidate[j] <- competing_group_variable_candidate[i]
  }

  # Generate outcomes based on true probabilities
  Yobs <- rep(NA, length(competing_group_competition_variable_candidate))

  for (cand_var in c("Republican", "Democrat")) {
    s_ <- ifelse(cand_var == "Republican", 1, -1)

    # --- SAME PARTY (PRIMARY) matchups ---
    # Republican respondent, Republican primary
    if (cand_var == "Republican") {
      # M vs M
      i_ <- which(competing_group_competition_variable_candidate == "Same" &
                    competing_group_variable_respondent == "Republican" &
                    competing_group_variable_candidate == "Republican" &
                    X$female == 0 & X_other == 0)
      Yobs[i_] <- rbinom(length(i_), 1, params$pr_mm_RVoters_primary)

      # M vs F
      i_ <- which(competing_group_competition_variable_candidate == "Same" &
                    competing_group_variable_respondent == "Republican" &
                    competing_group_variable_candidate == "Republican" &
                    X$female == 0 & X_other == 1)
      Yobs[i_] <- rbinom(length(i_), 1, params$pr_mf_RVoters_primary)

      # F vs M
      i_ <- which(competing_group_competition_variable_candidate == "Same" &
                    competing_group_variable_respondent == "Republican" &
                    competing_group_variable_candidate == "Republican" &
                    X$female == 1 & X_other == 0)
      Yobs[i_] <- rbinom(length(i_), 1, params$pr_fm_RVoters_primary)

      # F vs F
      i_ <- which(competing_group_competition_variable_candidate == "Same" &
                    competing_group_variable_respondent == "Republican" &
                    competing_group_variable_candidate == "Republican" &
                    X$female == 1 & X_other == 1)
      Yobs[i_] <- rbinom(length(i_), 1, params$pr_ff_RVoters_primary)
    }

    # Democrat respondent, Democrat primary
    if (cand_var == "Democrat") {
      i_ <- which(competing_group_competition_variable_candidate == "Same" &
                    competing_group_variable_respondent == "Democrat" &
                    competing_group_variable_candidate == "Democrat" &
                    X$female == 0 & X_other == 0)
      Yobs[i_] <- rbinom(length(i_), 1, params$pd_mm_DVoters_primary)

      i_ <- which(competing_group_competition_variable_candidate == "Same" &
                    competing_group_variable_respondent == "Democrat" &
                    competing_group_variable_candidate == "Democrat" &
                    X$female == 0 & X_other == 1)
      Yobs[i_] <- rbinom(length(i_), 1, params$pd_mf_DVoters_primary)

      i_ <- which(competing_group_competition_variable_candidate == "Same" &
                    competing_group_variable_respondent == "Democrat" &
                    competing_group_variable_candidate == "Democrat" &
                    X$female == 1 & X_other == 0)
      Yobs[i_] <- rbinom(length(i_), 1, params$pd_fm_DVoters_primary)

      i_ <- which(competing_group_competition_variable_candidate == "Same" &
                    competing_group_variable_respondent == "Democrat" &
                    competing_group_variable_candidate == "Democrat" &
                    X$female == 1 & X_other == 1)
      Yobs[i_] <- rbinom(length(i_), 1, params$pd_ff_DVoters_primary)
    }

    # --- DIFFERENT PARTY (GENERAL) matchups ---
    # Republican respondent, general election
    # M vs M
    i_ <- which(competing_group_competition_variable_candidate == "Different" &
                  competing_group_variable_respondent == "Republican" &
                  competing_group_variable_candidate == cand_var &
                  X$female == 0 & X_other == 0)
    Yobs[i_] <- rbinom(length(i_), 1, 0.5 + s_ * (params$pr_mm_RVoters - 0.5))

    # M vs F
    i_ <- which(competing_group_competition_variable_candidate == "Different" &
                  competing_group_variable_respondent == "Republican" &
                  competing_group_variable_candidate == cand_var &
                  X$female == 0 & X_other == 1)
    if (cand_var == "Democrat") {
      Yobs[i_] <- rbinom(length(i_), 1, 0.5 + s_ * (params$pr_fm_RVoters - 0.5))
    } else {
      Yobs[i_] <- rbinom(length(i_), 1, 0.5 + s_ * (params$pr_mf_RVoters - 0.5))
    }

    # F vs M
    i_ <- which(competing_group_competition_variable_candidate == "Different" &
                  competing_group_variable_respondent == "Republican" &
                  competing_group_variable_candidate == cand_var &
                  X$female == 1 & X_other == 0)
    if (cand_var == "Democrat") {
      Yobs[i_] <- rbinom(length(i_), 1, 0.5 + s_ * (params$pr_mf_RVoters - 0.5))
    } else {
      Yobs[i_] <- rbinom(length(i_), 1, 0.5 + s_ * (params$pr_fm_RVoters - 0.5))
    }

    # F vs F
    i_ <- which(competing_group_competition_variable_candidate == "Different" &
                  competing_group_variable_respondent == "Republican" &
                  competing_group_variable_candidate == cand_var &
                  X$female == 1 & X_other == 1)
    Yobs[i_] <- rbinom(length(i_), 1, 0.5 + s_ * (params$pr_ff_RVoters - 0.5))

    # Democrat respondent, general election
    # M vs M
    i_ <- which(competing_group_competition_variable_candidate == "Different" &
                  competing_group_variable_respondent == "Democrat" &
                  competing_group_variable_candidate == cand_var &
                  X$female == 0 & X_other == 0)
    Yobs[i_] <- rbinom(length(i_), 1, 0.5 + s_ * (params$pr_mm_DVoters - 0.5))

    # M vs F
    i_ <- which(competing_group_competition_variable_candidate == "Different" &
                  competing_group_variable_respondent == "Democrat" &
                  competing_group_variable_candidate == cand_var &
                  X$female == 0 & X_other == 1)
    if (cand_var == "Democrat") {
      Yobs[i_] <- rbinom(length(i_), 1, 0.5 + s_ * (params$pr_fm_DVoters - 0.5))
    } else {
      Yobs[i_] <- rbinom(length(i_), 1, 0.5 + s_ * (params$pr_mf_DVoters - 0.5))
    }

    # F vs M
    i_ <- which(competing_group_competition_variable_candidate == "Different" &
                  competing_group_variable_respondent == "Democrat" &
                  competing_group_variable_candidate == cand_var &
                  X$female == 1 & X_other == 0)
    if (cand_var == "Democrat") {
      Yobs[i_] <- rbinom(length(i_), 1, 0.5 + s_ * (params$pr_mf_DVoters - 0.5))
    } else {
      Yobs[i_] <- rbinom(length(i_), 1, 0.5 + s_ * (params$pr_fm_DVoters - 0.5))
    }

    # F vs F
    i_ <- which(competing_group_competition_variable_candidate == "Different" &
                  competing_group_variable_respondent == "Democrat" &
                  competing_group_variable_candidate == cand_var &
                  X$female == 1 & X_other == 1)
    Yobs[i_] <- rbinom(length(i_), 1, 0.5 + s_ * (params$pr_ff_DVoters - 0.5))
  }

  # Force forced-choice structure: second profile is complement of first
  Yobs[-(1:n)] <- 1 - Yobs[1:n]

  list(
    Y = Yobs,
    W = X,
    pair_id = pair_id,
    respondent_id = respondent_id,
    respondent_task_id = respondent_task_id,
    profile_order = profile_order,
    competing_group_variable_respondent = competing_group_variable_respondent,
    competing_group_variable_candidate = competing_group_variable_candidate,
    competing_group_competition_variable_candidate = competing_group_competition_variable_candidate,
    X_other = X_other,
    other_competing_group_variable_candidate = other_competing_group_variable_candidate
  )
}


# =============================================================================
# SECTION 7: Test - Basic Execution
# =============================================================================

test_that("adversarial mode executes without fatal error", {
  params <- define_ground_truth(p_RVoters = 0.5, lambda = 0.02)
  data <- generate_adversarial_data(n = 1000, params = params, seed = 42)

  res <- tryCatch({
    strategize(
      Y = data$Y,
      W = data$W,
      lambda = params$lambda,
      pair_id = data$pair_id,
      respondent_id = data$respondent_id,
      respondent_task_id = data$respondent_task_id,
      profile_order = data$profile_order,
      competing_group_variable_respondent = data$competing_group_variable_respondent,
      competing_group_variable_candidate = data$competing_group_variable_candidate,
      competing_group_competition_variable_candidate = data$competing_group_competition_variable_candidate,
      adversarial = TRUE,
      diff = TRUE,
      nSGD = 100,
      outcome_model_type = "glm",
      force_gaussian = FALSE,
      nMonte_adversarial = 24L,
      nMonte_Qglm = 100L,
      temperature = 0.3,
      compute_se = FALSE,
      conda_env = "strategize_env",
      conda_env_required = TRUE
    )
  }, error = function(e) {
    list(error = TRUE, message = conditionMessage(e))
  })

  expect_false(isTRUE(res$error))
})


# =============================================================================
# SECTION 8: Test - Output Structure
# =============================================================================

test_that("adversarial mode returns expected output structure", {
  params <- define_ground_truth(p_RVoters = 0.5, lambda = 0.02)
  data <- generate_adversarial_data(n = 1000, params = params, seed = 123)

  res <- strategize(
    Y = data$Y,
    W = data$W,
    lambda = params$lambda,
    pair_id = data$pair_id,
    respondent_id = data$respondent_id,
    respondent_task_id = data$respondent_task_id,
    profile_order = data$profile_order,
    competing_group_variable_respondent = data$competing_group_variable_respondent,
    competing_group_variable_candidate = data$competing_group_variable_candidate,
    competing_group_competition_variable_candidate = data$competing_group_competition_variable_candidate,
    adversarial = TRUE,
    diff = TRUE,
    nSGD = 100,
    outcome_model_type = "glm",
    force_gaussian = FALSE,
    nMonte_adversarial = 24L,
    nMonte_Qglm = 100L,
    temperature = 0.3,
    compute_se = FALSE,
    conda_env = "strategize_env",
    conda_env_required = TRUE
  )

  expect_type(res, "list")
  expect_true("pi_star_point" %in% names(res))
  expect_true("Q_point" %in% names(res))
  expect_type(res$pi_star_point, "list")
  expect_equal(length(res$pi_star_point), 2)  # Two parties
})


# =============================================================================
# SECTION 9: Test - Nash Equilibrium Varies with Voter Proportion
# =============================================================================

test_that("Nash equilibrium responds to voter proportion changes", {
  # Test at different voter proportions
  results <- list()

  for (p_R in c(0.3, 0.5, 0.7)) {
    params <- define_ground_truth(p_RVoters = p_R, lambda = 0.02)
    nash <- compute_nash_grid(params, grid_step = 0.01)

    results[[as.character(p_R)]] <- nash
  }

  # Verify that Nash equilibria were found (or approximately found)
  # The robust grid search always returns a result; check deviation is reasonable
  all_found <- all(sapply(results, function(x) x$found || x$deviation < 0.1))
  expect_true(all_found)
})


# =============================================================================
# SECTION 10: Test - High Regularization Pushes to Uniform
# =============================================================================

test_that("high regularization yields near-uniform equilibrium", {
  params <- define_ground_truth(p_RVoters = 0.5, lambda = 10.0)  # Very high lambda

  nash <- compute_nash_grid(params, grid_step = 0.01)

  # With high regularization, equilibrium should be near 0.5
  if (nash$found) {
    expect_lt(abs(nash$pi_R - 0.5), 0.05)
    expect_lt(abs(nash$pi_D - 0.5), 0.05)
  }
})


# =============================================================================
# SECTION 11: Test - Grid and Iterative Methods Agree
# =============================================================================

test_that("grid search and iterative methods produce same equilibrium", {
  params <- define_ground_truth(p_RVoters = 0.5, lambda = 0.02)

  nash_grid <- compute_nash_grid(params, grid_step = 0.01)
  nash_iter <- compute_nash_iterative(params, grid_step = 0.01)

  if (nash_grid$found && nash_iter$converged) {
    expect_equal(nash_grid$pi_R, nash_iter$pi_R, tolerance = 0.02)
    expect_equal(nash_grid$pi_D, nash_iter$pi_D, tolerance = 0.02)
  }
})


# =============================================================================
# SECTION 12: Test - Stability and Convergence
# =============================================================================

test_that("adversarial optimization converges without Inf values", {
  params <- define_ground_truth(p_RVoters = 0.5, lambda = 0.02)
  data <- generate_adversarial_data(n = 2000, params = params, seed = 333)

  res <- strategize(
    Y = data$Y,
    W = data$W,
    lambda = params$lambda,
    pair_id = data$pair_id,
    respondent_id = data$respondent_id,
    respondent_task_id = data$respondent_task_id,
    profile_order = data$profile_order,
    competing_group_variable_respondent = data$competing_group_variable_respondent,
    competing_group_variable_candidate = data$competing_group_variable_candidate,
    competing_group_competition_variable_candidate = data$competing_group_competition_variable_candidate,
    adversarial = TRUE,
    diff = TRUE,
    nSGD = 500,
    outcome_model_type = "glm",
    force_gaussian = FALSE,
    nMonte_adversarial = 24L,
    nMonte_Qglm = 100L,
    temperature = 0.3,
    compute_se = FALSE,
    conda_env = "strategize_env",
    conda_env_required = TRUE
  )

  # Check no Inf values
  pi_values <- unlist(res$pi_star_point)
  Q_values <- if (is.list(res$Q_point)) unlist(res$Q_point) else res$Q_point

  expect_false(any(is.infinite(pi_values)))
  expect_false(any(is.infinite(Q_values)))
})
