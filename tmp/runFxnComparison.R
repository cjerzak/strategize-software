# =============================================================================
# Direct Function Comparison Test Script
# =============================================================================
# This script compares 4 Q Monte Carlo implementations by:
# 1. Modifying the source file to use each implementation
# 2. Rebuilding and reinstalling the package
# 3. Running a test and capturing results
# 4. Generating a comparison report
# =============================================================================

options(error = NULL)

# Working directory should be strategize-software
setwd("/Users/cjerzak/Documents/strategize-software")

# =============================================================================
# Helper Functions
# =============================================================================

# Function to modify the implementation in two_step_master.R
set_implementation <- function(impl_name) {
  master_file <- "strategize/R/two_step_master.R"
  lines <- readLines(master_file)

  for (i in seq_along(lines)) {
    if (grepl("InitializeQMonteFxns_ <- paste\\(deparse\\(", lines[i])) {
      lines[i] <- sprintf("    InitializeQMonteFxns_ <- paste(deparse(%s),collapse=\"\\n\")", impl_name)
      break
    }
  }

  writeLines(lines, master_file)
  cat(sprintf("Set implementation to: %s\n", impl_name))
}

# Function to rebuild and reload package
rebuild_package <- function() {
  cat("Rebuilding package...\n")

  # Unload if loaded
  try(detach("package:strategize", unload = TRUE), silent = TRUE)

  # Install from source
  install.packages("strategize", repos = NULL, type = "source", quiet = TRUE)

  # Load
  library(strategize)
  cat("Package rebuilt and loaded.\n")
}

# =============================================================================
# Test Data Generation (simplified from adversarial test)
# =============================================================================

generate_test_data <- function(n = 2000, seed = 42) {
  set.seed(seed)

  p_RVoters <- 0.5
  p_DVoters <- 0.5

  competing_group_variable_respondent <- sample(
    c("Democrat", "Republican"), size = n, replace = TRUE, prob = c(0.5, 0.5)
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
    c("Democrat", "Republican"), size = 2 * n, replace = TRUE, prob = c(0.5, 0.5)
  )

  competing_group_competition_variable_candidate <- ifelse(
    competing_group_variable_candidate[1:n] == competing_group_variable_candidate[-(1:n)],
    "Same", "Different"
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

  # Generate outcomes with known probabilities
  Yobs <- rep(NA, 2 * n)

  # Simplified probability model
  pr_primary <- 0.52  # slight male advantage
  pr_general_R <- 0.85  # R voters prefer R
  pr_general_D <- 0.15  # D voters prefer R (low)

  for (i in 1:n) {
    resp <- competing_group_variable_respondent[i]
    comp <- competing_group_competition_variable_candidate[i]
    cand <- competing_group_variable_candidate[i]

    if (comp == "Same") {
      # Primary - slight effect
      if ((X$female[i] == 0 && X_other[i] == 1)) {
        prob <- pr_primary  # male vs female
      } else if ((X$female[i] == 1 && X_other[i] == 0)) {
        prob <- 1 - pr_primary  # female vs male
      } else {
        prob <- 0.5  # same gender
      }
    } else {
      # General election
      if (resp == "Republican") {
        prob <- pr_general_R
      } else {
        prob <- pr_general_D
      }
      # Adjust for candidate party
      if (cand == "Democrat") {
        prob <- 1 - prob
      }
    }

    Yobs[i] <- rbinom(1, 1, prob)
  }

  # Force forced-choice
  Yobs[(n+1):(2*n)] <- 1 - Yobs[1:n]

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
# Run Test with Current Implementation
# =============================================================================

run_test <- function(data, lambda = 0.02, nSGD = 500) {
  start_time <- Sys.time()

  res <- tryCatch({
    strategize(
      Y = data$Y,
      W = data$W,
      lambda = lambda,
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
      pi_R = NA, pi_D = NA, Q_point = NA,
      time_secs = elapsed, error = res$message
    ))
  }

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
# Main Comparison
# =============================================================================

run_full_comparison <- function() {
  implementations <- c(
    "InitializeQMonteFxns",           # Default
    "InitializeQMonteFxns_linearized",
    "InitializeQMonteFxns_new0",
    "InitializeQMonteFxns_new1"
  )

  # Generate test data once
  cat("\n=== Generating test data ===\n")
  data <- generate_test_data(n = 2000, seed = 42)

  results <- list()

  for (impl in implementations) {
    cat("\n")
    cat("========================================\n")
    cat(sprintf("Testing: %s\n", impl))
    cat("========================================\n")

    # Set implementation
    set_implementation(impl)

    # Rebuild package
    rebuild_package()

    # Run test
    cat("Running strategize...\n")
    res <- run_test(data, lambda = 0.02, nSGD = 500)

    results[[impl]] <- res

    cat(sprintf("\nResults:\n"))
    cat(sprintf("  pi_R = %.4f\n", res$pi_R))
    cat(sprintf("  pi_D = %.4f\n", res$pi_D))
    cat(sprintf("  Q_point = %.4f\n", res$Q_point))
    cat(sprintf("  Time: %.1f seconds\n", res$time_secs))
    if (!is.null(res$error)) {
      cat(sprintf("  ERROR: %s\n", res$error))
    }
  }

  # Restore default implementation
  cat("\n=== Restoring default implementation ===\n")
  set_implementation("InitializeQMonteFxns")
  rebuild_package()

  results
}

# =============================================================================
# Generate Report
# =============================================================================

generate_comparison_report <- function(results, output_file = "reports/FxnComparison.md") {
  # Reference equilibrium (from theoretical calculation at p_RVoters=0.5, lambda=0.02)
  # With symmetric parameters, Nash equilibrium is at (0.5, 0.5)
  ref_pi_R <- 0.50
  ref_pi_D <- 0.50

  report <- c(
    "# Q Monte Carlo Function Implementation Comparison Report",
    "",
    sprintf("*Generated: %s*", Sys.time()),
    "",
    "## Overview",
    "",
    "This report compares four Q Monte Carlo function implementations in the",
    "strategize package for adversarial Nash equilibrium recovery.",
    "",
    "## Test Configuration",
    "",
    "- **Sample size**: 2000 observations",
    "- **SGD iterations**: 500",
    "- **Lambda (regularization)**: 0.02",
    "- **Reference equilibrium**: pi_R = 0.50, pi_D = 0.50 (uniform)",
    "",
    "## Implementations",
    "",
    "| # | Name | Description |",
    "|---|------|-------------|",
    "| 1 | InitializeQMonteFxns | Default: Full push-forward with relaxed Bernoulli sampling |",
    "| 2 | InitializeQMonteFxns_linearized | Analytical four-quadrant decomposition |",
    "| 3 | InitializeQMonteFxns_new0 | Relaxed Bernoulli with single MC step |",
    "| 4 | InitializeQMonteFxns_new1 | Straight-through binary-concrete estimator |",
    "",
    "## Results",
    "",
    "| Implementation | pi_R | pi_D | Q_point | |pi_R - ref| | |pi_D - ref| | Total Dev | Time (s) |",
    "|----------------|------|------|---------|-------------|-------------|-----------|----------|"
  )

  for (impl in names(results)) {
    r <- results[[impl]]

    pi_R_str <- if (is.na(r$pi_R)) "NA" else sprintf("%.4f", r$pi_R)
    pi_D_str <- if (is.na(r$pi_D)) "NA" else sprintf("%.4f", r$pi_D)
    Q_str <- if (is.na(r$Q_point)) "NA" else sprintf("%.4f", r$Q_point)

    dev_R <- if (!is.na(r$pi_R)) abs(r$pi_R - ref_pi_R) else NA
    dev_D <- if (!is.na(r$pi_D)) abs(r$pi_D - ref_pi_D) else NA
    total_dev <- if (!is.na(dev_R) && !is.na(dev_D)) dev_R + dev_D else NA

    dev_R_str <- if (is.na(dev_R)) "NA" else sprintf("%.4f", dev_R)
    dev_D_str <- if (is.na(dev_D)) "NA" else sprintf("%.4f", dev_D)
    total_dev_str <- if (is.na(total_dev)) "NA" else sprintf("%.4f", total_dev)
    time_str <- sprintf("%.1f", r$time_secs)

    impl_short <- gsub("InitializeQMonteFxns_?", "", impl)
    if (impl_short == "") impl_short <- "Default"

    report <- c(report, sprintf("| %s | %s | %s | %s | %s | %s | %s | %s |",
                                 impl_short, pi_R_str, pi_D_str, Q_str,
                                 dev_R_str, dev_D_str, total_dev_str, time_str))
  }

  # Add ranking
  report <- c(report,
    "",
    "## Ranking by Total Deviation",
    ""
  )

  devs <- sapply(results, function(r) {
    if (is.na(r$pi_R) || is.na(r$pi_D)) return(Inf)
    abs(r$pi_R - ref_pi_R) + abs(r$pi_D - ref_pi_D)
  })
  sorted <- names(sort(devs))

  for (i in seq_along(sorted)) {
    impl <- sorted[i]
    d <- devs[[impl]]
    d_str <- if (is.infinite(d)) "NA" else sprintf("%.4f", d)
    report <- c(report, sprintf("%d. **%s** (deviation: %s)", i, impl, d_str))
  }

  # Add detailed output
  report <- c(report,
    "",
    "## Detailed Results",
    ""
  )

  for (impl in names(results)) {
    r <- results[[impl]]
    report <- c(report,
      sprintf("### %s", impl),
      "",
      sprintf("- pi_R: %s", if (is.na(r$pi_R)) "NA" else sprintf("%.4f", r$pi_R)),
      sprintf("- pi_D: %s", if (is.na(r$pi_D)) "NA" else sprintf("%.4f", r$pi_D)),
      sprintf("- Q_point: %s", if (is.na(r$Q_point)) "NA" else sprintf("%.4f", r$Q_point)),
      sprintf("- Execution time: %.1f seconds", r$time_secs),
      ""
    )
    if (!is.null(r$error)) {
      report <- c(report, sprintf("**Error**: %s", r$error), "")
    }
  }

  report <- c(report,
    "## Recommendation",
    "",
    sprintf("Based on total deviation from reference equilibrium, **%s** performs best.", sorted[1]),
    "",
    "---",
    "*Note: The package has been restored to use InitializeQMonteFxns (default) after testing.*"
  )

  dir.create(dirname(output_file), showWarnings = FALSE, recursive = TRUE)
  writeLines(report, output_file)
  cat(sprintf("\nReport saved to: %s\n", output_file))

  invisible(report)
}

# =============================================================================
# Execute
# =============================================================================

cat("\n")
cat("================================================================\n")
cat("  Q MONTE CARLO FUNCTION COMPARISON TEST\n")
cat("================================================================\n")

results <- run_full_comparison()
generate_comparison_report(results)

cat("\n")
cat("================================================================\n")
cat("  COMPARISON COMPLETE\n")
cat("================================================================\n")
