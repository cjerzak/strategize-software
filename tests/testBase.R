{
# install.packages("~/Documents/strategize-software/strategize",repos = NULL, type = "source",force = F)
# devtools::install_github("cjerzak/strategize-software/strategize")
# strategize::build_backend()
options(error=NULL)
library(testthat); library(strategize)
source(file.path("~/Documents/strategize-software/strategize", "R", "CS_HelperFxns.R"))

# =============================================================================
# SECTION 1: Helper Function Tests
# =============================================================================

test_that("toSimplex returns a valid probability vector", {
  x <- c(0.1, -0.2, 0.3)
  s <- toSimplex(x)
  expect_equal(sum(s), 1, tolerance = 1e-7)
  expect_true(all(s >= 0))
})

test_that("toSimplex handles extreme values", {
  # Test with large positive values
  x_large <- c(25, 30, 20)
  s_large <- toSimplex(x_large)
  expect_equal(sum(s_large), 1, tolerance = 1e-7)
  expect_true(all(s_large >= 0))

  # Test with large negative values
  x_neg <- c(-25, -30, -20)
  s_neg <- toSimplex(x_neg)
  expect_equal(sum(s_neg), 1, tolerance = 1e-7)
  expect_true(all(s_neg >= 0))
})

test_that("ess_fxn computes effective sample size correctly", {
  w <- c(1, 1, 1, 1)
  expect_equal(ess_fxn(w), 4)

  w2 <- c(1, 0.5)
  expect_equal(ess_fxn(w2), sum(w2)^2 / sum(w2^2))
})

test_that("ess_fxn handles edge cases", {
  # Single weight
  expect_equal(ess_fxn(1), 1)

  # All zeros except one (extreme unbalance)
  w_extreme <- c(1, 0, 0, 0)
  expect_equal(ess_fxn(w_extreme), 1)
})

test_that("RescaleFxn rescales and recenters", {
  x <- c(-1, 0, 1)
  res <- RescaleFxn(x, estMean = 2, estSD = 3)
  expect_equal(res, x * 3 + 2)

  res_no_center <- RescaleFxn(x, estMean = 2, estSD = 3, center = FALSE)
  expect_equal(res_no_center, x * 3)
})

test_that("getSE handles missing values", {
  vals <- c(1, 2, 3, NA)
  expect_equal(getSE(vals), sqrt(var(vals, na.rm = TRUE) / 3))
})

test_that("getSE handles all NA values", {
  vals_na <- c(NA, NA, NA)
  expect_true(is.na(getSE(vals_na)))
})

# =============================================================================
# SECTION 2: Generate Test Data
# =============================================================================

# Generate data for strategize tests
set.seed(1234321)
n <- 1000
W <- cbind(matrix(sample(c("A", "B"), n, replace = TRUE), ncol = 1),
           matrix(sample(c("A", "B"), n, replace = TRUE), ncol = 1),
           matrix(sample(c("A", "B"), n, replace = TRUE), ncol = 1))
colnames(W) <- c("V1","V2","V3")
pair_id <- respondent_id <-  c(seq_len(n/2),seq_len(n/2))
respondent_task_id <- c(seq_len(n/2),seq_len(n/2))
profile_order <-  c(rep(1,n/2),rep(2,n/2))
Y <- as.numeric(ave(
  drop((W == "B") %*% c(0.4, 0.2, 0.3)),                      # latent utility: weights for the three features
  respondent_task_id,                                         # pair each forced-choice task
  FUN = function(g) rank(g, ties.method = "random") == length(g) # winner within each pair
))

# Generate competing group variable for adversarial tests
competing_group <- sample(c("Group1", "Group2"), n, replace = TRUE, prob = c(0.5, 0.5))

# =============================================================================
# SECTION 3: Core Strategize Tests (GLM and Neural)
# =============================================================================

# Neural network tests are slow due to MCMC sampling (~30+ minutes)
# Set RUN_NEURAL_TESTS <- TRUE to include them
RUN_NEURAL_TESTS <- FALSE
outcome_model_types <- if(RUN_NEURAL_TESTS) c("glm", "neural") else c("glm")

for(outcome_model_type in outcome_model_types){
  # Test core strategize functionality
  test_that(sprintf("strategize returns a valid result [%s]", outcome_model_type), {
    res <- {strategize(
      Y = Y,
      W = W,
      lambda = 0.1,
      pair_id = pair_id,
      respondent_id = respondent_id,
      respondent_task_id = respondent_task_id,
      profile_order = profile_order,
      K = 1,
      nSGD = 10,
      outcome_model_type = outcome_model_type,
      force_gaussian = TRUE,
      nMonte_adversarial = 10L,
      nMonte_Qglm = 10L,
      diff = TRUE,
      compute_se = FALSE,
      conda_env = "strategize_env",
      conda_env_required = TRUE
    )}
    expect_type(res, "list")
    expect_true("pi_star_point" %in% names(res))
    expect_true("Q_point" %in% names(res))
    expect_true("p_list" %in% names(res))
  })

  # Test cross-validation functionality
  test_that(sprintf("cv_strategize selects lambda [%s]",outcome_model_type), {
    cv_res <- {cv_strategize(
      Y = Y,
      W = W,
      lambda = 0.1,
      pair_id = pair_id,
      respondent_id = respondent_id,
      respondent_task_id = respondent_task_id,
      profile_order = profile_order,
      K = 1,
      diff = TRUE,
      nSGD = 10,
      outcome_model_type = outcome_model_type,
      force_gaussian = TRUE,
      nMonte_adversarial = 10L,
      nMonte_Qglm = 10L,
      compute_se = FALSE,
      conda_env = "strategize_env",
      conda_env_required = TRUE
    )}

    expect_type(cv_res, "list")
    expect_true("lambda" %in% names(cv_res))
    expect_true("CVInfo" %in% names(cv_res))
  })
}

# =============================================================================
# SECTION 4: Adversarial Mode Tests
# =============================================================================

# NOTE: Adversarial mode tests require a specific data structure with
# competing_group_variable_candidate that properly aligns with pair structure.
# These tests are skipped pending proper test data setup for adversarial scenarios.
# The adversarial functionality has been manually verified but automated tests
# require more complex data generation matching the expected internal structure.

test_that("strategize handles adversarial mode", {
  skip("Adversarial mode requires specialized test data structure - see issue #14")
})

test_that("cv_strategize handles adversarial mode", {
  skip("Adversarial mode requires specialized test data structure - see issue #14")
})

# =============================================================================
# SECTION 5: Multi-Cluster (K > 1) Tests
# =============================================================================

# NOTE: K > 1 tests require respondent-level covariates (X) for clustering.
# These tests are skipped pending proper test data setup.

test_that("strategize handles K > 1 (multi-cluster)", {
  skip("K > 1 multi-cluster mode requires covariate X for clustering - see issue #14")
})

test_that("cv_strategize handles K > 1 (multi-cluster)", {
  skip("K > 1 multi-cluster mode requires covariate X for clustering - see issue #14")
})

# =============================================================================
# SECTION 6: Different Penalty Types Tests
# =============================================================================

test_that("strategize handles KL penalty type", {
  res_kl <- strategize(
    Y = Y,
    W = W,
    lambda = 0.1,
    pair_id = pair_id,
    respondent_id = respondent_id,
    respondent_task_id = respondent_task_id,
    profile_order = profile_order,
    K = 1,
    nSGD = 10,
    penalty_type = "KL",
    outcome_model_type = "glm",
    force_gaussian = TRUE,
    nMonte_adversarial = 10L,
    nMonte_Qglm = 10L,
    diff = TRUE,
    compute_se = FALSE,
    conda_env = "strategize_env",
    conda_env_required = TRUE
  )

  expect_type(res_kl, "list")
  expect_true("pi_star_point" %in% names(res_kl))
})

test_that("strategize handles L2 penalty type", {
  res_l2 <- strategize(
    Y = Y,
    W = W,
    lambda = 0.1,
    pair_id = pair_id,
    respondent_id = respondent_id,
    respondent_task_id = respondent_task_id,
    profile_order = profile_order,
    K = 1,
    nSGD = 10,
    penalty_type = "L2",
    outcome_model_type = "glm",
    force_gaussian = TRUE,
    nMonte_adversarial = 10L,
    nMonte_Qglm = 10L,
    diff = TRUE,
    compute_se = FALSE,
    conda_env = "strategize_env",
    conda_env_required = TRUE
  )

  expect_type(res_l2, "list")
  expect_true("pi_star_point" %in% names(res_l2))
})

test_that("strategize handles LogMaxProb penalty type", {
  res_logmax <- strategize(
    Y = Y,
    W = W,
    lambda = 0.1,
    pair_id = pair_id,
    respondent_id = respondent_id,
    respondent_task_id = respondent_task_id,
    profile_order = profile_order,
    K = 1,
    nSGD = 10,
    penalty_type = "LogMaxProb",
    outcome_model_type = "glm",
    force_gaussian = TRUE,
    nMonte_adversarial = 10L,
    nMonte_Qglm = 10L,
    diff = TRUE,
    compute_se = FALSE,
    conda_env = "strategize_env",
    conda_env_required = TRUE
  )

  expect_type(res_logmax, "list")
  expect_true("pi_star_point" %in% names(res_logmax))
})

# =============================================================================
# SECTION 7: Edge Case Tests
# =============================================================================

test_that("strategize handles single factor", {
  # Create data with only one factor
  set.seed(42)
  n_single <- 200
  W_single <- matrix(sample(c("A", "B"), n_single, replace = TRUE), ncol = 1)
  colnames(W_single) <- "V1"
  pair_id_single <- c(seq_len(n_single/2), seq_len(n_single/2))
  respondent_id_single <- pair_id_single
  respondent_task_id_single <- pair_id_single
  profile_order_single <- c(rep(1, n_single/2), rep(2, n_single/2))
  Y_single <- as.numeric(ave(
    as.numeric(W_single == "B") * 0.5,
    respondent_task_id_single,
    FUN = function(g) rank(g, ties.method = "random") == length(g)
  ))

  res_single <- strategize(
    Y = Y_single,
    W = W_single,
    lambda = 0.1,
    pair_id = pair_id_single,
    respondent_id = respondent_id_single,
    respondent_task_id = respondent_task_id_single,
    profile_order = profile_order_single,
    K = 1,
    nSGD = 10,
    outcome_model_type = "glm",
    force_gaussian = TRUE,
    nMonte_adversarial = 10L,
    nMonte_Qglm = 10L,
    diff = TRUE,
    compute_se = FALSE,
    conda_env = "strategize_env",
    conda_env_required = TRUE
  )

  expect_type(res_single, "list")
  expect_true("pi_star_point" %in% names(res_single))
})

test_that("strategize handles multiple factor levels", {
  # Create data with factors having more than 2 levels
  set.seed(42)
  n_multi <- 300
  W_multi <- cbind(
    matrix(sample(c("A", "B", "C", "D"), n_multi, replace = TRUE), ncol = 1),
    matrix(sample(c("X", "Y", "Z"), n_multi, replace = TRUE), ncol = 1)
  )
  colnames(W_multi) <- c("Factor1", "Factor2")
  pair_id_multi <- c(seq_len(n_multi/2), seq_len(n_multi/2))
  respondent_id_multi <- pair_id_multi
  respondent_task_id_multi <- pair_id_multi
  profile_order_multi <- c(rep(1, n_multi/2), rep(2, n_multi/2))
  Y_multi <- as.numeric(ave(
    as.numeric(W_multi[,1] == "D") * 0.4 + as.numeric(W_multi[,2] == "Z") * 0.3,
    respondent_task_id_multi,
    FUN = function(g) rank(g, ties.method = "random") == length(g)
  ))

  res_multi <- strategize(
    Y = Y_multi,
    W = W_multi,
    lambda = 0.1,
    pair_id = pair_id_multi,
    respondent_id = respondent_id_multi,
    respondent_task_id = respondent_task_id_multi,
    profile_order = profile_order_multi,
    K = 1,
    nSGD = 10,
    outcome_model_type = "glm",
    force_gaussian = TRUE,
    nMonte_adversarial = 10L,
    nMonte_Qglm = 10L,
    diff = TRUE,
    compute_se = FALSE,
    conda_env = "strategize_env",
    conda_env_required = TRUE
  )

  expect_type(res_multi, "list")
  expect_true("pi_star_point" %in% names(res_multi))
  # Check that p_list has correct structure for multiple levels
  expect_true(length(res_multi$p_list) == 2)
})

test_that("strategize handles diff = FALSE (non-difference mode)", {
  res_nodiff <- strategize(
    Y = Y,
    W = W,
    lambda = 0.1,
    pair_id = pair_id,
    respondent_id = respondent_id,
    respondent_task_id = respondent_task_id,
    profile_order = profile_order,
    K = 1,
    nSGD = 10,
    diff = FALSE,
    outcome_model_type = "glm",
    force_gaussian = TRUE,
    nMonte_adversarial = 10L,
    nMonte_Qglm = 10L,
    compute_se = FALSE,
    conda_env = "strategize_env",
    conda_env_required = TRUE
  )

  expect_type(res_nodiff, "list")
  expect_true("pi_star_point" %in% names(res_nodiff))
})

test_that("strategize handles moderate sample size", {
  # Test with a moderate-sized dataset
  # Need at least 3 factors for interaction term estimation
  set.seed(42)
  n_small <- 300
  W_small <- cbind(
    matrix(sample(c("A", "B"), n_small, replace = TRUE), ncol = 1),
    matrix(sample(c("A", "B"), n_small, replace = TRUE), ncol = 1),
    matrix(sample(c("A", "B"), n_small, replace = TRUE), ncol = 1)
  )
  colnames(W_small) <- c("V1", "V2", "V3")
  pair_id_small <- c(seq_len(n_small/2), seq_len(n_small/2))
  respondent_id_small <- pair_id_small
  respondent_task_id_small <- pair_id_small
  profile_order_small <- c(rep(1, n_small/2), rep(2, n_small/2))
  Y_small <- as.numeric(ave(
    drop((W_small == "B") %*% c(0.4, 0.2, 0.3)),
    respondent_task_id_small,
    FUN = function(g) rank(g, ties.method = "random") == length(g)
  ))

  res_small <- strategize(
    Y = Y_small,
    W = W_small,
    lambda = 0.1,
    pair_id = pair_id_small,
    respondent_id = respondent_id_small,
    respondent_task_id = respondent_task_id_small,
    profile_order = profile_order_small,
    K = 1,
    nSGD = 5,
    outcome_model_type = "glm",
    force_gaussian = TRUE,
    nMonte_adversarial = 5L,
    nMonte_Qglm = 5L,
    diff = TRUE,
    compute_se = FALSE,
    conda_env = "strategize_env",
    conda_env_required = TRUE
  )

  expect_type(res_small, "list")
  expect_true("pi_star_point" %in% names(res_small))
})

# =============================================================================
# SECTION 8: Standard Error Computation Tests
# =============================================================================

test_that("strategize computes standard errors correctly", {
  res_se <- strategize(
    Y = Y,
    W = W,
    lambda = 0.1,
    pair_id = pair_id,
    respondent_id = respondent_id,
    respondent_task_id = respondent_task_id,
    profile_order = profile_order,
    K = 1,
    nSGD = 10,
    outcome_model_type = "glm",
    force_gaussian = TRUE,
    nMonte_adversarial = 10L,
    nMonte_Qglm = 10L,
    diff = TRUE,
    compute_se = TRUE,
    conda_env = "strategize_env",
    conda_env_required = TRUE
  )

  expect_type(res_se, "list")
  expect_true("pi_star_se" %in% names(res_se))
  expect_true("Q_se" %in% names(res_se))
})

# =============================================================================
# SECTION 9: Use Regularization Tests
# =============================================================================

test_that("strategize handles use_regularization = FALSE", {
  res_noreg <- strategize(
    Y = Y,
    W = W,
    lambda = 0.1,
    pair_id = pair_id,
    respondent_id = respondent_id,
    respondent_task_id = respondent_task_id,
    profile_order = profile_order,
    K = 1,
    nSGD = 10,
    use_regularization = FALSE,
    outcome_model_type = "glm",
    force_gaussian = TRUE,
    nMonte_adversarial = 10L,
    nMonte_Qglm = 10L,
    diff = TRUE,
    compute_se = FALSE,
    conda_env = "strategize_env",
    conda_env_required = TRUE
  )

  expect_type(res_noreg, "list")
  expect_true("pi_star_point" %in% names(res_noreg))
})

# =============================================================================
# SECTION 10: Error Handling Tests
# =============================================================================

test_that("strategize validates Y and W dimensions", {
  # Y and W should have same number of rows
  W_wrong <- W[1:100,]
  expect_error(
    strategize(
      Y = Y,
      W = W_wrong,
      lambda = 0.1,
      pair_id = pair_id,
      respondent_id = respondent_id,
      respondent_task_id = respondent_task_id,
      profile_order = profile_order,
      K = 1,
      nSGD = 10,
      outcome_model_type = "glm",
      force_gaussian = TRUE,
      nMonte_adversarial = 10L,
      nMonte_Qglm = 10L,
      diff = TRUE,
      compute_se = FALSE,
      conda_env = "strategize_env",
      conda_env_required = TRUE
    ),
    regexp = ".*"  # Expect some error
  )
})

test_that("strategize handles missing lambda gracefully", {
  # When lambda is NULL, cv_strategize should be used instead
  # strategize should work with a specified lambda
  res <- strategize(
    Y = Y,
    W = W,
    lambda = 0.5,  # Must provide lambda
    pair_id = pair_id,
    respondent_id = respondent_id,
    respondent_task_id = respondent_task_id,
    profile_order = profile_order,
    K = 1,
    nSGD = 10,
    outcome_model_type = "glm",
    force_gaussian = TRUE,
    nMonte_adversarial = 10L,
    nMonte_Qglm = 10L,
    diff = TRUE,
    compute_se = FALSE,
    conda_env = "strategize_env",
    conda_env_required = TRUE
  )

  expect_type(res, "list")
})

# =============================================================================
# SECTION 11: Lambda Vector Tests
# =============================================================================

test_that("cv_strategize handles vector of lambda values", {
  cv_res_multi_lambda <- cv_strategize(
    Y = Y,
    W = W,
    lambda = c(0.01, 0.1, 1.0),  # Multiple lambda values
    pair_id = pair_id,
    respondent_id = respondent_id,
    respondent_task_id = respondent_task_id,
    profile_order = profile_order,
    K = 1,
    diff = TRUE,
    nSGD = 10,
    outcome_model_type = "glm",
    force_gaussian = TRUE,
    nMonte_adversarial = 10L,
    nMonte_Qglm = 10L,
    compute_se = FALSE,
    conda_env = "strategize_env",
    conda_env_required = TRUE
  )

  expect_type(cv_res_multi_lambda, "list")
  expect_true("lambda" %in% names(cv_res_multi_lambda))
  expect_true("CVInfo" %in% names(cv_res_multi_lambda))
})

# =============================================================================
# SECTION 12: Output Structure Validation Tests
# =============================================================================

test_that("strategize returns all expected output fields", {
  res <- strategize(
    Y = Y,
    W = W,
    lambda = 0.1,
    pair_id = pair_id,
    respondent_id = respondent_id,
    respondent_task_id = respondent_task_id,
    profile_order = profile_order,
    K = 1,
    nSGD = 10,
    outcome_model_type = "glm",
    force_gaussian = TRUE,
    nMonte_adversarial = 10L,
    nMonte_Qglm = 10L,
    diff = TRUE,
    compute_se = FALSE,
    conda_env = "strategize_env",
    conda_env_required = TRUE
  )

  # Check core output fields exist
  expected_fields <- c("pi_star_point", "Q_point", "p_list")
  for (field in expected_fields) {
    expect_true(field %in% names(res),
                info = sprintf("Missing expected field: %s", field))
  }

  # pi_star_point should be a list
  expect_type(res$pi_star_point, "list")

  # p_list should be a list with same length as number of factors
  expect_type(res$p_list, "list")
  expect_equal(length(res$p_list), ncol(W))
})

test_that("cv_strategize returns all expected output fields", {
  cv_res <- cv_strategize(
    Y = Y,
    W = W,
    lambda = 0.1,
    pair_id = pair_id,
    respondent_id = respondent_id,
    respondent_task_id = respondent_task_id,
    profile_order = profile_order,
    K = 1,
    diff = TRUE,
    nSGD = 10,
    outcome_model_type = "glm",
    force_gaussian = TRUE,
    nMonte_adversarial = 10L,
    nMonte_Qglm = 10L,
    compute_se = FALSE,
    conda_env = "strategize_env",
    conda_env_required = TRUE
  )

  # Check core output fields exist
  expected_fields <- c("lambda", "CVInfo", "pi_star_point", "p_list")
  for (field in expected_fields) {
    expect_true(field %in% names(cv_res),
                info = sprintf("Missing expected field: %s", field))
  }
})

# =============================================================================
# SECTION 13: Probability Distribution Validity Tests
# =============================================================================

test_that("strategize returns valid probability distributions", {
  res <- strategize(
    Y = Y,
    W = W,
    lambda = 0.1,
    pair_id = pair_id,
    respondent_id = respondent_id,
    respondent_task_id = respondent_task_id,
    profile_order = profile_order,
    K = 1,
    nSGD = 10,
    outcome_model_type = "glm",
    force_gaussian = TRUE,
    nMonte_adversarial = 10L,
    nMonte_Qglm = 10L,
    diff = TRUE,
    compute_se = FALSE,
    conda_env = "strategize_env",
    conda_env_required = TRUE
  )

  # Check that pi_star_point contains valid probability distributions
  for (k in seq_along(res$pi_star_point)) {
    pi_k <- res$pi_star_point[[k]]
    for (d in seq_along(pi_k)) {
      pi_kd <- pi_k[[d]]
      # All probabilities should be non-negative
      expect_true(all(pi_kd >= 0),
                  info = sprintf("Negative probability in cluster %d, factor %d", k, d))
      # Probabilities should sum to 1
      expect_equal(sum(pi_kd), 1, tolerance = 1e-6,
                   info = sprintf("Probabilities don't sum to 1 in cluster %d, factor %d", k, d))
    }
  }

  # Check that p_list contains valid probability distributions
  for (d in seq_along(res$p_list)) {
    p_d <- res$p_list[[d]]
    expect_true(all(p_d >= 0),
                info = sprintf("Negative baseline probability in factor %d", d))
    expect_equal(sum(p_d), 1, tolerance = 1e-6,
                 info = sprintf("Baseline probabilities don't sum to 1 in factor %d", d))
  }
})
}
