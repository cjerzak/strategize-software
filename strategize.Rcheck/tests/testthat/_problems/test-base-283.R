# Extracted from test-base.R:283

# setup ------------------------------------------------------------------------
library(testthat)
test_env <- simulate_test_env(package = "strategize", path = "..")
attach(test_env, warn.conflicts = FALSE)

# prequel ----------------------------------------------------------------------
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
n_unique_respondents <- n/2
respondent_party_base <- rep(c("PartyA", "PartyB"), each = n_unique_respondents/2)
competing_group_variable_respondent <- respondent_party_base[respondent_id]
competition_type_base <- rep(c("Same", "Different"), each = n_unique_respondents/2)
set.seed(42)
competition_type_base <- sample(competition_type_base)
pair_competition_type <- competition_type_base[pair_id]
competing_group_competition_variable_candidate <- pair_competition_type
competing_group_variable_candidate <- ifelse(
  pair_competition_type == "Same",
  # Same competition (primary): candidate party = respondent party
  competing_group_variable_respondent,
  # Different competition (general): profile_order 1 = PartyA, profile_order 2 = PartyB
  ifelse(profile_order == 1, "PartyA", "PartyB")
)
n_respondents <- n/2
X_base <- matrix(rnorm(n_respondents * 3), ncol = 3)
colnames(X_base) <- c("X1", "X2", "X3")
X <- X_base[respondent_id, ]
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

# test -------------------------------------------------------------------------
res_k2 <- strategize(
    Y = Y,
    W = W,
    X = X,
    lambda = 0.1,
    pair_id = pair_id,
    respondent_id = respondent_id,
    respondent_task_id = respondent_task_id,
    profile_order = profile_order,
    K = 2,
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
