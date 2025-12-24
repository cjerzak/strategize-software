# =============================================================================
# Q* Experimentation Script
# =============================================================================
# This script explores conditions under which Q* (equilibrium vote share)
# deviates from 0.5 in the adversarial two-stage election model.
#
# Q* = 0.5 occurs in symmetric games where neither party has an advantage.
# To find Q* ≠ 0.5, we test asymmetric configurations:
# - Asymmetric voter proportions
# - Asymmetric primary effects
# - Asymmetric general election effects
# - Different regularization levels
# =============================================================================

options(error = NULL)
library(strategize)

setwd("/Users/cjerzak/Documents/strategize-software")

# =============================================================================
# Ground Truth Definition (from testAdversarialCorrectness.R)
# =============================================================================

define_ground_truth <- function(
    p_RVoters = 0.5,
    lambda = 0.02,
    pr_mm_RVoters_primary = 0.50,
    pr_mf_RVoters_primary = 0.55,
    pr_fm_RVoters_primary = 0.45,
    pr_ff_RVoters_primary = 0.50,
    pd_mm_DVoters_primary = 0.50,
    pd_mf_DVoters_primary = 0.45,
    pd_fm_DVoters_primary = 0.55,
    pd_ff_DVoters_primary = 0.50,
    pr_mm_RVoters = 0.85,
    pr_mf_RVoters = 0.90,
    pr_fm_RVoters = 0.80,
    pr_ff_RVoters = 0.85,
    pr_mm_DVoters = 0.15,
    pr_mf_DVoters = 0.10,
    pr_fm_DVoters = 0.20,
    pr_ff_DVoters = 0.15
) {
  list(
    p_RVoters = p_RVoters,
    p_DVoters = 1 - p_RVoters,
    lambda = lambda,
    pr_mm_RVoters_primary = pr_mm_RVoters_primary,
    pr_mf_RVoters_primary = pr_mf_RVoters_primary,
    pr_fm_RVoters_primary = pr_fm_RVoters_primary,
    pr_ff_RVoters_primary = pr_ff_RVoters_primary,
    pd_mm_DVoters_primary = pd_mm_DVoters_primary,
    pd_mf_DVoters_primary = pd_mf_DVoters_primary,
    pd_fm_DVoters_primary = pd_fm_DVoters_primary,
    pd_ff_DVoters_primary = pd_ff_DVoters_primary,
    pr_mm_RVoters = pr_mm_RVoters,
    pr_mf_RVoters = pr_mf_RVoters,
    pr_fm_RVoters = pr_fm_RVoters,
    pr_ff_RVoters = pr_ff_RVoters,
    pr_mm_DVoters = pr_mm_DVoters,
    pr_mf_DVoters = pr_mf_DVoters,
    pr_fm_DVoters = pr_fm_DVoters,
    pr_ff_DVoters = pr_ff_DVoters
  )
}

# =============================================================================
# Vote Share Computation
# =============================================================================

compute_vote_share_R <- function(pi_R, pi_D, params, pi_R_s = 0.50, pi_D_s = 0.50) {
  with(params, {
    p_R_m <- 1 * (pi_R * pi_R_s) +
      pr_mf_RVoters_primary * (pi_R * (1 - pi_R_s)) +
      (1 - pr_fm_RVoters_primary) * ((1 - pi_R) * pi_R_s) +
      0 * ((1 - pi_R) * (1 - pi_R_s))
    p_R_f <- 1 - p_R_m

    p_D_m <- 1 * (pi_D * pi_D_s) +
      pd_mf_DVoters_primary * (pi_D * (1 - pi_D_s)) +
      (1 - pd_fm_DVoters_primary) * ((1 - pi_D) * pi_D_s) +
      0 * ((1 - pi_D) * (1 - pi_D_s))
    p_D_f <- 1 - p_D_m

    p_mm <- p_R_m * p_D_m
    p_mf <- p_R_m * p_D_f
    p_fm <- p_R_f * p_D_m
    p_ff <- p_R_f * p_D_f

    EVShare_R <- p_mm * (p_RVoters * pr_mm_RVoters + p_DVoters * pr_mm_DVoters) +
      p_mf * (p_RVoters * pr_mf_RVoters + p_DVoters * pr_mf_DVoters) +
      p_fm * (p_RVoters * pr_fm_RVoters + p_DVoters * pr_fm_DVoters) +
      p_ff * (p_RVoters * pr_ff_RVoters + p_DVoters * pr_ff_DVoters)

    return(EVShare_R)
  })
}

# =============================================================================
# Utility Functions
# =============================================================================

utility_R <- function(pi_R, pi_D, params) {
  vote_share <- compute_vote_share_R(pi_R, pi_D, params)
  vote_share - params$lambda * ((pi_R - 0.5)^2 + ((1 - pi_R) - 0.5)^2)
}

utility_D <- function(pi_R, pi_D, params) {
  vote_share <- 1 - compute_vote_share_R(pi_R, pi_D, params)
  vote_share - params$lambda * ((pi_D - 0.5)^2 + ((1 - pi_D) - 0.5)^2)
}

# =============================================================================
# Nash Equilibrium Computation
# =============================================================================

compute_nash_grid <- function(params, grid_step = 0.01) {
  pi_seq <- seq(0, 1, by = grid_step)

  best_response_D <- sapply(pi_seq, function(r) {
    utilities <- sapply(pi_seq, function(d) utility_D(r, d, params))
    pi_seq[which.max(utilities)]
  })

  best_response_R <- sapply(pi_seq, function(d) {
    utilities <- sapply(pi_seq, function(r) utility_R(r, d, params))
    pi_seq[which.max(utilities)]
  })

  min_deviation <- Inf
  pi_R_nash <- NA
  pi_D_nash <- NA

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
    }
  }

  Q_star <- compute_vote_share_R(pi_R_nash, pi_D_nash, params)

  list(
    pi_R = pi_R_nash,
    pi_D = pi_D_nash,
    Q_star = Q_star,
    deviation = min_deviation
  )
}

# =============================================================================
# Data Generation
# =============================================================================

generate_adversarial_data <- function(n = 2000, params, seed = 12345) {
  set.seed(seed)

  competing_group_variable_respondent <- sample(
    c("Democrat", "Republican"),
    size = n,
    replace = TRUE,
    prob = c(params$p_DVoters, params$p_RVoters)
  )
  competing_group_variable_respondent <- c(
    competing_group_variable_respondent,
    competing_group_variable_respondent
  )

  pair_id <- c(1:n, 1:n)
  respondent_id <- pair_id
  profile_order <- c(rep(1, n), rep(2, n))
  respondent_task_id <- rep(1, times = 2 * n)

  competing_group_variable_candidate <- sample(
    c("Democrat", "Republican"),
    size = 2 * n,
    replace = TRUE,
    prob = c(0.5, 0.5)
  )

  competing_group_competition_variable_candidate <- ifelse(
    competing_group_variable_candidate[1:n] == competing_group_variable_candidate[-(1:n)],
    yes = "Same",
    no = "Different"
  )
  competing_group_competition_variable_candidate <- c(
    competing_group_competition_variable_candidate,
    competing_group_competition_variable_candidate
  )

  X <- as.data.frame(matrix(
    rbinom(n * 2, size = 1, prob = 0.5),
    nrow = 2 * n
  ))
  colnames(X) <- "female"

  X_other <- numeric(nrow(X))
  for (p in unique(pair_id)) {
    rows_in_pair <- which(pair_id == p)
    i <- rows_in_pair[1]
    j <- rows_in_pair[2]
    X_other[i] <- X$female[j]
    X_other[j] <- X$female[i]
  }

  Yobs <- rep(NA, length(competing_group_competition_variable_candidate))

  for (cand_var in c("Republican", "Democrat")) {
    s_ <- ifelse(cand_var == "Republican", 1, -1)

    if (cand_var == "Republican") {
      i_ <- which(competing_group_competition_variable_candidate == "Same" &
                    competing_group_variable_respondent == "Republican" &
                    competing_group_variable_candidate == "Republican" &
                    X$female == 0 & X_other == 0)
      Yobs[i_] <- rbinom(length(i_), 1, params$pr_mm_RVoters_primary)

      i_ <- which(competing_group_competition_variable_candidate == "Same" &
                    competing_group_variable_respondent == "Republican" &
                    competing_group_variable_candidate == "Republican" &
                    X$female == 0 & X_other == 1)
      Yobs[i_] <- rbinom(length(i_), 1, params$pr_mf_RVoters_primary)

      i_ <- which(competing_group_competition_variable_candidate == "Same" &
                    competing_group_variable_respondent == "Republican" &
                    competing_group_variable_candidate == "Republican" &
                    X$female == 1 & X_other == 0)
      Yobs[i_] <- rbinom(length(i_), 1, params$pr_fm_RVoters_primary)

      i_ <- which(competing_group_competition_variable_candidate == "Same" &
                    competing_group_variable_respondent == "Republican" &
                    competing_group_variable_candidate == "Republican" &
                    X$female == 1 & X_other == 1)
      Yobs[i_] <- rbinom(length(i_), 1, params$pr_ff_RVoters_primary)
    }

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

    i_ <- which(competing_group_competition_variable_candidate == "Different" &
                  competing_group_variable_respondent == "Republican" &
                  competing_group_variable_candidate == cand_var &
                  X$female == 0 & X_other == 0)
    Yobs[i_] <- rbinom(length(i_), 1, 0.5 + s_ * (params$pr_mm_RVoters - 0.5))

    i_ <- which(competing_group_competition_variable_candidate == "Different" &
                  competing_group_variable_respondent == "Republican" &
                  competing_group_variable_candidate == cand_var &
                  X$female == 0 & X_other == 1)
    if (cand_var == "Democrat") {
      Yobs[i_] <- rbinom(length(i_), 1, 0.5 + s_ * (params$pr_fm_RVoters - 0.5))
    } else {
      Yobs[i_] <- rbinom(length(i_), 1, 0.5 + s_ * (params$pr_mf_RVoters - 0.5))
    }

    i_ <- which(competing_group_competition_variable_candidate == "Different" &
                  competing_group_variable_respondent == "Republican" &
                  competing_group_variable_candidate == cand_var &
                  X$female == 1 & X_other == 0)
    if (cand_var == "Democrat") {
      Yobs[i_] <- rbinom(length(i_), 1, 0.5 + s_ * (params$pr_mf_RVoters - 0.5))
    } else {
      Yobs[i_] <- rbinom(length(i_), 1, 0.5 + s_ * (params$pr_fm_RVoters - 0.5))
    }

    i_ <- which(competing_group_competition_variable_candidate == "Different" &
                  competing_group_variable_respondent == "Republican" &
                  competing_group_variable_candidate == cand_var &
                  X$female == 1 & X_other == 1)
    Yobs[i_] <- rbinom(length(i_), 1, 0.5 + s_ * (params$pr_ff_RVoters - 0.5))

    i_ <- which(competing_group_competition_variable_candidate == "Different" &
                  competing_group_variable_respondent == "Democrat" &
                  competing_group_variable_candidate == cand_var &
                  X$female == 0 & X_other == 0)
    Yobs[i_] <- rbinom(length(i_), 1, 0.5 + s_ * (params$pr_mm_DVoters - 0.5))

    i_ <- which(competing_group_competition_variable_candidate == "Different" &
                  competing_group_variable_respondent == "Democrat" &
                  competing_group_variable_candidate == cand_var &
                  X$female == 0 & X_other == 1)
    if (cand_var == "Democrat") {
      Yobs[i_] <- rbinom(length(i_), 1, 0.5 + s_ * (params$pr_fm_DVoters - 0.5))
    } else {
      Yobs[i_] <- rbinom(length(i_), 1, 0.5 + s_ * (params$pr_mf_DVoters - 0.5))
    }

    i_ <- which(competing_group_competition_variable_candidate == "Different" &
                  competing_group_variable_respondent == "Democrat" &
                  competing_group_variable_candidate == cand_var &
                  X$female == 1 & X_other == 0)
    if (cand_var == "Democrat") {
      Yobs[i_] <- rbinom(length(i_), 1, 0.5 + s_ * (params$pr_mf_DVoters - 0.5))
    } else {
      Yobs[i_] <- rbinom(length(i_), 1, 0.5 + s_ * (params$pr_fm_DVoters - 0.5))
    }

    i_ <- which(competing_group_competition_variable_candidate == "Different" &
                  competing_group_variable_respondent == "Democrat" &
                  competing_group_variable_candidate == cand_var &
                  X$female == 1 & X_other == 1)
    Yobs[i_] <- rbinom(length(i_), 1, 0.5 + s_ * (params$pr_ff_DVoters - 0.5))
  }

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
    competing_group_competition_variable_candidate = competing_group_competition_variable_candidate
  )
}

# =============================================================================
# Run Strategize
# =============================================================================

run_strategize <- function(data, params, nSGD = 1000) {
  tryCatch({
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
      nSGD = nSGD,
      outcome_model_type = "glm",
      force_gaussian = FALSE,
      nMonte_adversarial = 24L,
      nMonte_Qglm = 100L,
      temperature = 0.3,
      learning_rate_max = 0.005,
      compute_se = FALSE,
      conda_env = "strategize_env",
      conda_env_required = TRUE
    )

    pi_R_est <- 1 - res$pi_star_point$Republican$female["1"]
    pi_D_est <- 1 - res$pi_star_point$Democrat$female["1"]
    Q_est <- as.numeric(res$Q_point)

    list(pi_R = pi_R_est, pi_D = pi_D_est, Q = Q_est, error = NULL)
  }, error = function(e) {
    list(pi_R = NA, pi_D = NA, Q = NA, error = conditionMessage(e))
  })
}

# =============================================================================
# EXPERIMENT 1: Varying Voter Proportions
# =============================================================================

cat("\n")
cat("================================================================\n")
cat("  EXPERIMENT 1: Varying Voter Proportions\n")
cat("================================================================\n\n")

exp1_results <- data.frame()

for (p_R in c(0.3, 0.4, 0.5, 0.6, 0.7)) {
  params <- define_ground_truth(p_RVoters = p_R, lambda = 0.02)
  nash <- compute_nash_grid(params)

  cat(sprintf("p_RVoters = %.1f:\n", p_R))
  cat(sprintf("  Theoretical: pi_R = %.3f, pi_D = %.3f, Q* = %.4f\n",
              nash$pi_R, nash$pi_D, nash$Q_star))

  exp1_results <- rbind(exp1_results, data.frame(
    p_RVoters = p_R,
    pi_R_theory = nash$pi_R,
    pi_D_theory = nash$pi_D,
    Q_star_theory = nash$Q_star,
    Q_deviation = nash$Q_star - 0.5
  ))
}

cat("\n--- Summary ---\n")
print(exp1_results)

# =============================================================================
# EXPERIMENT 2: Asymmetric Primary Effects
# =============================================================================

cat("\n")
cat("================================================================\n")
cat("  EXPERIMENT 2: Asymmetric Primary Effects\n")
cat("================================================================\n\n")

exp2_results <- data.frame()

# R has strong male advantage in primary, D has no gender preference
params_R_male <- define_ground_truth(
  p_RVoters = 0.5,
  lambda = 0.02,
  pr_mf_RVoters_primary = 0.70,  # R voters strongly prefer male
  pr_fm_RVoters_primary = 0.30,
  pd_mf_DVoters_primary = 0.50,  # D voters neutral
  pd_fm_DVoters_primary = 0.50
)
nash_R_male <- compute_nash_grid(params_R_male)
cat("Case: R primary favors males strongly, D neutral\n")
cat(sprintf("  Theoretical: pi_R = %.3f, pi_D = %.3f, Q* = %.4f\n",
            nash_R_male$pi_R, nash_R_male$pi_D, nash_R_male$Q_star))

exp2_results <- rbind(exp2_results, data.frame(
  case = "R_male_strong_D_neutral",
  pi_R = nash_R_male$pi_R,
  pi_D = nash_R_male$pi_D,
  Q_star = nash_R_male$Q_star
))

# D has strong female advantage in primary, R neutral
params_D_female <- define_ground_truth(
  p_RVoters = 0.5,
  lambda = 0.02,
  pr_mf_RVoters_primary = 0.50,
  pr_fm_RVoters_primary = 0.50,
  pd_mf_DVoters_primary = 0.30,  # D voters strongly prefer female
  pd_fm_DVoters_primary = 0.70
)
nash_D_female <- compute_nash_grid(params_D_female)
cat("Case: D primary favors females strongly, R neutral\n")
cat(sprintf("  Theoretical: pi_R = %.3f, pi_D = %.3f, Q* = %.4f\n",
            nash_D_female$pi_R, nash_D_female$pi_D, nash_D_female$Q_star))

exp2_results <- rbind(exp2_results, data.frame(
  case = "D_female_strong_R_neutral",
  pi_R = nash_D_female$pi_R,
  pi_D = nash_D_female$pi_D,
  Q_star = nash_D_female$Q_star
))

# Both parties favor same gender (male)
params_both_male <- define_ground_truth(
  p_RVoters = 0.5,
  lambda = 0.02,
  pr_mf_RVoters_primary = 0.70,
  pr_fm_RVoters_primary = 0.30,
  pd_mf_DVoters_primary = 0.70,
  pd_fm_DVoters_primary = 0.30
)
nash_both_male <- compute_nash_grid(params_both_male)
cat("Case: Both parties favor males\n")
cat(sprintf("  Theoretical: pi_R = %.3f, pi_D = %.3f, Q* = %.4f\n",
            nash_both_male$pi_R, nash_both_male$pi_D, nash_both_male$Q_star))

exp2_results <- rbind(exp2_results, data.frame(
  case = "both_male",
  pi_R = nash_both_male$pi_R,
  pi_D = nash_both_male$pi_D,
  Q_star = nash_both_male$Q_star
))

cat("\n--- Summary ---\n")
print(exp2_results)

# =============================================================================
# EXPERIMENT 3: Asymmetric General Election Effects
# =============================================================================

cat("\n")
cat("================================================================\n")
cat("  EXPERIMENT 3: Asymmetric General Election Effects\n")
cat("================================================================\n\n")

exp3_results <- data.frame()

# R male dominates in general election
params_gen_R_male <- define_ground_truth(
  p_RVoters = 0.5,
  lambda = 0.02,
  # General: R male wins big
  pr_mm_RVoters = 0.90, pr_mf_RVoters = 0.95, pr_fm_RVoters = 0.75, pr_ff_RVoters = 0.85,
  pr_mm_DVoters = 0.15, pr_mf_DVoters = 0.05, pr_fm_DVoters = 0.25, pr_ff_DVoters = 0.15
)
nash_gen_R_male <- compute_nash_grid(params_gen_R_male)
cat("Case: R male dominates in general election\n")
cat(sprintf("  Theoretical: pi_R = %.3f, pi_D = %.3f, Q* = %.4f\n",
            nash_gen_R_male$pi_R, nash_gen_R_male$pi_D, nash_gen_R_male$Q_star))

exp3_results <- rbind(exp3_results, data.frame(
  case = "R_male_general_dominant",
  pi_R = nash_gen_R_male$pi_R,
  pi_D = nash_gen_R_male$pi_D,
  Q_star = nash_gen_R_male$Q_star
))

# D female appeals across parties
params_gen_D_female <- define_ground_truth(
  p_RVoters = 0.5,
  lambda = 0.02,
  # General: D female appeals to R voters too
  pr_mm_RVoters = 0.85, pr_mf_RVoters = 0.88, pr_fm_RVoters = 0.70, pr_ff_RVoters = 0.75,
  pr_mm_DVoters = 0.15, pr_mf_DVoters = 0.12, pr_fm_DVoters = 0.30, pr_ff_DVoters = 0.25
)
nash_gen_D_female <- compute_nash_grid(params_gen_D_female)
cat("Case: D female has cross-party appeal\n")
cat(sprintf("  Theoretical: pi_R = %.3f, pi_D = %.3f, Q* = %.4f\n",
            nash_gen_D_female$pi_R, nash_gen_D_female$pi_D, nash_gen_D_female$Q_star))

exp3_results <- rbind(exp3_results, data.frame(
  case = "D_female_crossparty",
  pi_R = nash_gen_D_female$pi_R,
  pi_D = nash_gen_D_female$pi_D,
  Q_star = nash_gen_D_female$Q_star
))

cat("\n--- Summary ---\n")
print(exp3_results)

# =============================================================================
# EXPERIMENT 4: Lambda (Regularization) Effects
# =============================================================================

cat("\n")
cat("================================================================\n")
cat("  EXPERIMENT 4: Lambda Effects\n")
cat("================================================================\n\n")

exp4_results <- data.frame()

# Use asymmetric base params
base_params <- define_ground_truth(
  p_RVoters = 0.6,
  pr_mf_RVoters_primary = 0.60,
  pr_fm_RVoters_primary = 0.40
)

for (lambda in c(0.001, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0)) {
  params <- base_params
  params$lambda <- lambda
  nash <- compute_nash_grid(params)

  cat(sprintf("lambda = %.3f:\n", lambda))
  cat(sprintf("  Theoretical: pi_R = %.3f, pi_D = %.3f, Q* = %.4f\n",
              nash$pi_R, nash$pi_D, nash$Q_star))

  exp4_results <- rbind(exp4_results, data.frame(
    lambda = lambda,
    pi_R = nash$pi_R,
    pi_D = nash$pi_D,
    Q_star = nash$Q_star,
    Q_deviation = nash$Q_star - 0.5
  ))
}

cat("\n--- Summary ---\n")
print(exp4_results)

# =============================================================================
# EXPERIMENT 5: Validate with Strategize (select cases)
# =============================================================================

cat("\n")
cat("================================================================\n")
cat("  EXPERIMENT 5: Validate Selected Cases with strategize()\n")
cat("================================================================\n\n")

exp5_results <- data.frame()

# Case 1: Asymmetric voter proportions
cat("Running strategize for p_RVoters = 0.7...\n")
params <- define_ground_truth(p_RVoters = 0.7, lambda = 0.02)
nash <- compute_nash_grid(params)
data <- generate_adversarial_data(n = 3000, params = params, seed = 42)
res <- run_strategize(data, params, nSGD = 1000)

cat(sprintf("  Theory:     pi_R = %.3f, pi_D = %.3f, Q* = %.4f\n",
            nash$pi_R, nash$pi_D, nash$Q_star))
cat(sprintf("  Estimated:  pi_R = %.3f, pi_D = %.3f, Q  = %.4f\n",
            res$pi_R, res$pi_D, res$Q))

exp5_results <- rbind(exp5_results, data.frame(
  case = "p_R=0.7",
  pi_R_theory = nash$pi_R,
  pi_D_theory = nash$pi_D,
  Q_theory = nash$Q_star,
  pi_R_est = res$pi_R,
  pi_D_est = res$pi_D,
  Q_est = res$Q
))

# Case 2: Strong R male primary advantage
cat("\nRunning strategize for R male primary advantage...\n")
params <- define_ground_truth(
  p_RVoters = 0.5,
  lambda = 0.01,  # Lower lambda to allow more deviation
  pr_mf_RVoters_primary = 0.70,
  pr_fm_RVoters_primary = 0.30
)
nash <- compute_nash_grid(params)
data <- generate_adversarial_data(n = 3000, params = params, seed = 43)
res <- run_strategize(data, params, nSGD = 1000)

cat(sprintf("  Theory:     pi_R = %.3f, pi_D = %.3f, Q* = %.4f\n",
            nash$pi_R, nash$pi_D, nash$Q_star))
cat(sprintf("  Estimated:  pi_R = %.3f, pi_D = %.3f, Q  = %.4f\n",
            res$pi_R, res$pi_D, res$Q))

exp5_results <- rbind(exp5_results, data.frame(
  case = "R_male_primary",
  pi_R_theory = nash$pi_R,
  pi_D_theory = nash$pi_D,
  Q_theory = nash$Q_star,
  pi_R_est = res$pi_R,
  pi_D_est = res$pi_D,
  Q_est = res$Q
))

# Case 3: Low lambda with voter asymmetry
cat("\nRunning strategize for low lambda + voter asymmetry...\n")
params <- define_ground_truth(
  p_RVoters = 0.6,
  lambda = 0.005  # Very low lambda
)
nash <- compute_nash_grid(params)
data <- generate_adversarial_data(n = 3000, params = params, seed = 44)
res <- run_strategize(data, params, nSGD = 1500)

cat(sprintf("  Theory:     pi_R = %.3f, pi_D = %.3f, Q* = %.4f\n",
            nash$pi_R, nash$pi_D, nash$Q_star))
cat(sprintf("  Estimated:  pi_R = %.3f, pi_D = %.3f, Q  = %.4f\n",
            res$pi_R, res$pi_D, res$Q))

exp5_results <- rbind(exp5_results, data.frame(
  case = "low_lambda_p_R=0.6",
  pi_R_theory = nash$pi_R,
  pi_D_theory = nash$pi_D,
  Q_theory = nash$Q_star,
  pi_R_est = res$pi_R,
  pi_D_est = res$pi_D,
  Q_est = res$Q
))

cat("\n--- Summary ---\n")
print(exp5_results)

# =============================================================================
# FINAL REPORT
# =============================================================================

cat("\n")
cat("================================================================\n")
cat("  FINAL SUMMARY: When is Q* ≠ 0.5?\n")
cat("================================================================\n\n")

cat("Key findings:\n\n")

cat("1. VOTER PROPORTIONS:\n")
cat("   Q* deviates from 0.5 when p_RVoters ≠ 0.5.\n")
cat("   More R voters → Q* > 0.5 (R advantage)\n")
cat("   More D voters → Q* < 0.5 (D advantage)\n")
cat(sprintf("   Example: p_RVoters=0.7 → Q* = %.4f\n", exp1_results$Q_star_theory[5]))
cat(sprintf("   Example: p_RVoters=0.3 → Q* = %.4f\n", exp1_results$Q_star_theory[1]))
cat("\n")

cat("2. PRIMARY EFFECTS:\n")
cat("   Asymmetric primary gender preferences can shift Q*,\n")
cat("   but the effect is muted when general election dominates.\n")
cat(sprintf("   Example: R male advantage in primary → Q* = %.4f\n", exp2_results$Q_star[1]))
cat("\n")

cat("3. GENERAL ELECTION EFFECTS:\n")
cat("   Asymmetric cross-party appeal can shift equilibrium.\n")
cat(sprintf("   Example: R male dominates general → Q* = %.4f\n", exp3_results$Q_star[1]))
cat(sprintf("   Example: D female crossparty appeal → Q* = %.4f\n", exp3_results$Q_star[2]))
cat("\n")

cat("4. REGULARIZATION (LAMBDA):\n")
cat("   Lower lambda allows more deviation from uniform,\n")
cat("   amplifying the effect of structural advantages.\n")
cat(sprintf("   Example: lambda=0.001 → Q* = %.4f\n", exp4_results$Q_star[1]))
cat(sprintf("   Example: lambda=1.0 → Q* = %.4f (pushed to 0.5)\n", exp4_results$Q_star[7]))
cat("\n")

cat("================================================================\n")
cat("  Experiments Complete\n")
cat("================================================================\n")
