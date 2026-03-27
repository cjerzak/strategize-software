# =============================================================================
# Comparison Test for Q Monte Carlo Function Implementations
# =============================================================================
#
# This script tests and compares the performance of four different Q Monte Carlo
# implementations in the strategize package for adversarial equilibrium recovery:
#
# 1. InitializeQMonteFxns (original/default)
# 2. InitializeQMonteFxns_linearized
# 3. InitializeQMonteFxns_new0
# 4. InitializeQMonteFxns_new1
#
# The test generates data with known Nash equilibrium and measures how well
# each implementation recovers it.
#
# =============================================================================

options(error = NULL)
library(strategize)

# =============================================================================
# SECTION 1: Ground Truth and Helper Functions (from adversarial test)
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

compute_vote_share_R <- function(pi_R, pi_D, params,
                                  pi_R_s = 0.50, pi_D_s = 0.50) {
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

utility_R <- function(pi_R, pi_D, params) {
  vote_share <- compute_vote_share_R(pi_R, pi_D, params)
  vote_share - params$lambda * ((pi_R - 0.5)^2 + ((1 - pi_R) - 0.5)^2)
}

utility_D <- function(pi_R, pi_D, params) {
  vote_share <- 1 - compute_vote_share_R(pi_R, pi_D, params)
  vote_share - params$lambda * ((pi_D - 0.5)^2 + ((1 - pi_D) - 0.5)^2)
}

compute_nash_grid <- function(params, grid_step = 0.01, tol = 0.02) {
  pi_seq <- seq(0, 1, by = grid_step)
  best_response_D <- sapply(pi_seq, function(r) {
    utilities <- sapply(pi_seq, function(d) utility_D(r, d, params))
    pi_seq[which.max(utilities)]
  })
  best_response_R <- sapply(pi_seq, function(d) {
    utilities <- sapply(pi_seq, function(r) utility_R(r, d, params))
    pi_seq[which.max(utilities)]
  })
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
      if (deviation < tol) nash_found <- TRUE
    }
  }
  if (min_deviation < tol * 2) nash_found <- TRUE
  list(pi_R = pi_R_nash, pi_D = pi_D_nash, found = nash_found, deviation = min_deviation)
}

generate_adversarial_data <- function(n = 2000, params, seed = 12345) {
  set.seed(seed)
  competing_group_variable_respondent <- sample(
    c("Democrat", "Republican"), size = n, replace = TRUE,
    prob = c(params$p_DVoters, params$p_RVoters)
  )
  competing_group_variable_respondent <- c(competing_group_variable_respondent,
                                            competing_group_variable_respondent)
  pair_id <- c(1:n, 1:n)
  respondent_id <- pair_id
  profile_order <- c(rep(1, n), rep(2, n))
  respondent_task_id <- rep(1, times = 2 * n)
  competing_group_variable_candidate <- sample(
    c("Democrat", "Republican"), size = 2 * n, replace = TRUE, prob = c(0.5, 0.5)
  )
  competing_group_competition_variable_candidate <- ifelse(
    competing_group_variable_candidate[1:n] == competing_group_variable_candidate[-(1:n)],
    yes = "Same", no = "Different"
  )
  competing_group_competition_variable_candidate <- c(
    competing_group_competition_variable_candidate,
    competing_group_competition_variable_candidate
  )
  X <- as.data.frame(matrix(rbinom(n * 2, size = 1, prob = 0.5), nrow = 2 * n))
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
    Y = Yobs, W = X, pair_id = pair_id, respondent_id = respondent_id,
    respondent_task_id = respondent_task_id, profile_order = profile_order,
    competing_group_variable_respondent = competing_group_variable_respondent,
    competing_group_variable_candidate = competing_group_variable_candidate,
    competing_group_competition_variable_candidate = competing_group_competition_variable_candidate
  )
}

# =============================================================================
# SECTION 2: Function to Modify Package and Run Test
# =============================================================================

# Path to the source file
SOURCE_FILE <- "strategize/R/two_step_optimize_vectorize_q.R"

# Function to modify the package to use a specific implementation
modify_implementation <- function(impl_name) {
  # Read the file
  lines <- readLines(SOURCE_FILE)

  # Find the line where InitializeQMonteFxns is called in two_step_master.R
  master_file <- "strategize/R/two_step_master.R"
  master_lines <- readLines(master_file)

  # Find and modify the line that calls InitializeQMonteFxns
  for (i in seq_along(master_lines)) {
    if (grepl("InitializeQMonteFxns_ <- paste\\(deparse\\(InitializeQMonteFxns\\)", master_lines[i])) {
      if (impl_name == "InitializeQMonteFxns") {
        master_lines[i] <- "    InitializeQMonteFxns_ <- paste(deparse(InitializeQMonteFxns),collapse=\"\\n\")"
      } else {
        master_lines[i] <- sprintf("    InitializeQMonteFxns_ <- paste(deparse(%s),collapse=\"\\n\")", impl_name)
      }
      break
    }
  }

  writeLines(master_lines, master_file)
}

# Function to run a single test with current implementation
run_single_test <- function(params, data, nSGD = 500) {
  start_time <- Sys.time()

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
  }, error = function(e) {
    list(error = TRUE, message = conditionMessage(e))
  })

  end_time <- Sys.time()
  elapsed <- as.numeric(difftime(end_time, start_time, units = "secs"))

  if (isTRUE(res$error)) {
    return(list(
      pi_R = NA,
      pi_D = NA,
      Q_point = NA,
      time_secs = elapsed,
      error = res$message
    ))
  }

  # Extract pi (note: package stores Pr(female), so pi_male = 1 - pi_female)
  pi_R <- tryCatch(1 - res$pi_star_point$Republican$female["1"], error = function(e) NA)
  pi_D <- tryCatch(1 - res$pi_star_point$Democrat$female["1"], error = function(e) NA)
  Q_point <- tryCatch(as.numeric(res$Q_point), error = function(e) NA)

  list(
    pi_R = pi_R,
    pi_D = pi_D,
    Q_point = Q_point,
    time_secs = elapsed,
    error = NULL
  )
}


# =============================================================================
# SECTION 3: Run Comparison Tests
# =============================================================================

run_comparison <- function(n_obs = 3000, nSGD = 1000, seed = 42) {
  # Define parameters
  params <- define_ground_truth(
    p_RVoters = 0.5,
    lambda = 0.02,
    pr_mf_RVoters_primary = 0.55,
    pr_fm_RVoters_primary = 0.45,
    pd_mf_DVoters_primary = 0.45,
    pd_fm_DVoters_primary = 0.55,
    pr_mf_RVoters = 0.90,
    pr_fm_RVoters = 0.80,
    pr_mf_DVoters = 0.10,
    pr_fm_DVoters = 0.20
  )

  # Compute theoretical Nash equilibrium
  nash <- compute_nash_grid(params, grid_step = 0.01)

  cat("========================================\n")
  cat("THEORETICAL NASH EQUILIBRIUM\n")
  cat("========================================\n")
  cat(sprintf("pi_R = %.4f, pi_D = %.4f\n\n", nash$pi_R, nash$pi_D))

  # Generate data
  cat("Generating test data...\n")
  data <- generate_adversarial_data(n = n_obs, params = params, seed = seed)

  # List of implementations to test
  implementations <- c(
    "InitializeQMonteFxns",        # Original/default
    "InitializeQMonteFxns_linearized",
    "InitializeQMonteFxns_new0",
    "InitializeQMonteFxns_new1"
  )

  results <- list()

  for (impl in implementations) {
    cat("\n========================================\n")
    cat(sprintf("TESTING: %s\n", impl))
    cat("========================================\n")

    # Modify the package to use this implementation
    modify_implementation(impl)

    # Reinstall package to pick up changes
    cat("Reinstalling package...\n")
    suppressMessages(install.packages("strategize", repos = NULL, type = "source", quiet = TRUE))

    # Reload the package
    detach("package:strategize", unload = TRUE)
    library(strategize)

    # Run test
    cat("Running strategize...\n")
    res <- run_single_test(params, data, nSGD = nSGD)

    # Compute deviations
    dev_R <- if (!is.na(res$pi_R)) abs(res$pi_R - nash$pi_R) else NA
    dev_D <- if (!is.na(res$pi_D)) abs(res$pi_D - nash$pi_D) else NA
    total_dev <- if (!is.na(dev_R) && !is.na(dev_D)) dev_R + dev_D else NA

    results[[impl]] <- list(
      implementation = impl,
      pi_R = res$pi_R,
      pi_D = res$pi_D,
      Q_point = res$Q_point,
      dev_R = dev_R,
      dev_D = dev_D,
      total_dev = total_dev,
      time_secs = res$time_secs,
      error = res$error
    )

    cat(sprintf("\nResults for %s:\n", impl))
    cat(sprintf("  pi_R = %.4f (deviation: %.4f)\n", res$pi_R, dev_R))
    cat(sprintf("  pi_D = %.4f (deviation: %.4f)\n", res$pi_D, dev_D))
    cat(sprintf("  Q_point = %.4f\n", res$Q_point))
    cat(sprintf("  Time: %.1f seconds\n", res$time_secs))
    if (!is.null(res$error)) {
      cat(sprintf("  ERROR: %s\n", res$error))
    }
  }

  # Restore original implementation
  cat("\n========================================\n")
  cat("RESTORING ORIGINAL IMPLEMENTATION\n")
  cat("========================================\n")
  modify_implementation("InitializeQMonteFxns")
  suppressMessages(install.packages("strategize", repos = NULL, type = "source", quiet = TRUE))

  list(
    params = params,
    nash = nash,
    results = results,
    n_obs = n_obs,
    nSGD = nSGD,
    seed = seed
  )
}


# =============================================================================
# SECTION 4: Generate Markdown Report
# =============================================================================

generate_report <- function(comparison_results, output_file = "reports/FxnComparison.md") {

  params <- comparison_results$params
  nash <- comparison_results$nash
  results <- comparison_results$results

  # Create report content
  report <- c(
    "# Q Monte Carlo Function Implementation Comparison",
    "",
    "## Overview",
    "",
    "This report compares the performance of four different Q Monte Carlo function",
    "implementations in the strategize package for adversarial equilibrium recovery.",
    "",
    "### Implementations Tested",
    "",
    "1. **InitializeQMonteFxns** (Original/Default) - Full push-forward with relaxed Bernoulli sampling",
    "2. **InitializeQMonteFxns_linearized** - Analytical four-quadrant decomposition with scalar mixture weights",
    "3. **InitializeQMonteFxns_new0** - Relaxed Bernoulli primary sampling with single MC step",
    "4. **InitializeQMonteFxns_new1** - Straight-through binary-concrete estimator for primary sampling",
    "",
    "## Test Configuration",
    "",
    sprintf("- **Sample size**: %d observations", comparison_results$n_obs),
    sprintf("- **SGD iterations**: %d", comparison_results$nSGD),
    sprintf("- **Random seed**: %d", comparison_results$seed),
    sprintf("- **Lambda (regularization)**: %.3f", params$lambda),
    sprintf("- **Voter proportion (R)**: %.1f%%", params$p_RVoters * 100),
    "",
    "## Theoretical Nash Equilibrium",
    "",
    sprintf("Using grid search with step size 0.01:"),
    "",
    sprintf("- **pi_R** (Pr male for Republican): %.4f", nash$pi_R),
    sprintf("- **pi_D** (Pr male for Democrat): %.4f", nash$pi_D),
    "",
    "## Results Summary",
    "",
    "| Implementation | pi_R | pi_D | Q_point | Dev_R | Dev_D | Total Dev | Time (s) |",
    "|----------------|------|------|---------|-------|-------|-----------|----------|"
  )

  for (impl in names(results)) {
    r <- results[[impl]]
    pi_R_str <- if (is.na(r$pi_R)) "NA" else sprintf("%.4f", r$pi_R)
    pi_D_str <- if (is.na(r$pi_D)) "NA" else sprintf("%.4f", r$pi_D)
    Q_str <- if (is.na(r$Q_point)) "NA" else sprintf("%.4f", r$Q_point)
    dev_R_str <- if (is.na(r$dev_R)) "NA" else sprintf("%.4f", r$dev_R)
    dev_D_str <- if (is.na(r$dev_D)) "NA" else sprintf("%.4f", r$dev_D)
    total_dev_str <- if (is.na(r$total_dev)) "NA" else sprintf("%.4f", r$total_dev)
    time_str <- sprintf("%.1f", r$time_secs)

    # Shorten implementation name for table
    impl_short <- gsub("InitializeQMonteFxns", "", impl)
    if (impl_short == "") impl_short <- "Default"
    impl_short <- gsub("_", "", impl_short)

    report <- c(report, sprintf("| %s | %s | %s | %s | %s | %s | %s | %s |",
                                 impl_short, pi_R_str, pi_D_str, Q_str,
                                 dev_R_str, dev_D_str, total_dev_str, time_str))
  }

  # Add detailed analysis
  report <- c(report,
    "",
    "## Detailed Analysis",
    ""
  )

  for (impl in names(results)) {
    r <- results[[impl]]
    report <- c(report,
      sprintf("### %s", impl),
      ""
    )

    if (!is.null(r$error)) {
      report <- c(report,
        sprintf("**Error**: %s", r$error),
        ""
      )
    } else {
      report <- c(report,
        sprintf("- **Estimated pi_R**: %.4f (theoretical: %.4f, deviation: %.4f)",
                r$pi_R, nash$pi_R, r$dev_R),
        sprintf("- **Estimated pi_D**: %.4f (theoretical: %.4f, deviation: %.4f)",
                r$pi_D, nash$pi_D, r$dev_D),
        sprintf("- **Q value**: %.4f", r$Q_point),
        sprintf("- **Execution time**: %.1f seconds", r$time_secs),
        ""
      )

      # Assess quality
      if (r$total_dev < 0.1) {
        report <- c(report, "**Assessment**: Excellent equilibrium recovery (< 0.1 total deviation)", "")
      } else if (r$total_dev < 0.2) {
        report <- c(report, "**Assessment**: Good equilibrium recovery (< 0.2 total deviation)", "")
      } else if (r$total_dev < 0.5) {
        report <- c(report, "**Assessment**: Moderate equilibrium recovery (< 0.5 total deviation)", "")
      } else {
        report <- c(report, "**Assessment**: Poor equilibrium recovery (>= 0.5 total deviation)", "")
      }
    }
  }

  # Add ranking
  report <- c(report,
    "## Ranking (by Total Deviation from Theoretical)",
    ""
  )

  # Sort by total deviation
  devs <- sapply(results, function(r) if (is.na(r$total_dev)) Inf else r$total_dev)
  sorted_impls <- names(sort(devs))

  for (i in seq_along(sorted_impls)) {
    impl <- sorted_impls[i]
    r <- results[[impl]]
    dev_str <- if (is.na(r$total_dev)) "NA" else sprintf("%.4f", r$total_dev)
    report <- c(report, sprintf("%d. **%s** - Total deviation: %s", i, impl, dev_str))
  }

  report <- c(report,
    "",
    "## Conclusion",
    "",
    "Based on the comparison, the implementation with the lowest total deviation from",
    "the theoretical Nash equilibrium is recommended for use in production.",
    "",
    sprintf("**Recommended implementation**: %s", sorted_impls[1]),
    "",
    "---",
    sprintf("*Report generated: %s*", Sys.time())
  )

  # Write report
  dir.create(dirname(output_file), showWarnings = FALSE, recursive = TRUE)
  writeLines(report, output_file)
  cat(sprintf("\nReport written to: %s\n", output_file))

  invisible(report)
}


# =============================================================================
# SECTION 5: Main Execution
# =============================================================================

if (interactive() || !exists("SKIP_MAIN")) {
  cat("\n")
  cat("================================================================\n")
  cat("  Q MONTE CARLO FUNCTION IMPLEMENTATION COMPARISON TEST\n")
  cat("================================================================\n")
  cat("\n")

  # Run comparison (reduced parameters for faster testing)
  comparison_results <- run_comparison(n_obs = 3000, nSGD = 1000, seed = 42)

  # Generate report
  generate_report(comparison_results, "reports/FxnComparison.md")

  cat("\n")
  cat("================================================================\n")
  cat("  TEST COMPLETE - Package restored to use InitializeQMonteFxns\n")
  cat("================================================================\n")
}
