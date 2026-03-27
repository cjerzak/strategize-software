options(error = NULL)
library(testthat)
library(strategize)

test_that("implicit SEs match full-trace SEs (adversarial)", {
  old_force_no_interactions <- if (exists("force_no_interactions", inherits = TRUE)) {
    get("force_no_interactions", inherits = TRUE)
  } else {
    NULL
  }
  assign("force_no_interactions", TRUE, envir = .GlobalEnv)
  on.exit({
    if (is.null(old_force_no_interactions)) {
      if (exists("force_no_interactions", inherits = FALSE, envir = .GlobalEnv)) {
        rm(force_no_interactions, envir = .GlobalEnv)
      }
    } else {
      assign("force_no_interactions", old_force_no_interactions, envir = .GlobalEnv)
    }
  }, add = TRUE)

  set.seed(123)
  n_pairs <- 60
  n <- n_pairs * 2
  W <- cbind(
    matrix(sample(c("A", "B"), n, replace = TRUE), ncol = 1),
    matrix(sample(c("A", "B"), n, replace = TRUE), ncol = 1)
  )
  colnames(W) <- c("V1", "V2")
  pair_id <- respondent_id <- rep(seq_len(n_pairs), each = 2)
  respondent_task_id <- pair_id
  profile_order <- rep(1:2, times = n_pairs)
  utilities <- drop((W == "B") %*% c(0.4, 0.2))
  Y <- as.numeric(ave(
    utilities,
    respondent_task_id,
    FUN = function(g) rank(g, ties.method = "random") == length(g)
  ))

  respondent_party_base <- rep(c("PartyA", "PartyB"), length.out = n_pairs)
  competing_group_variable_respondent <- respondent_party_base[respondent_id]
  competition_type_base <- rep(c("Same", "Different"), length.out = n_pairs)
  set.seed(456)
  competition_type_base <- sample(competition_type_base)
  pair_competition_type <- competition_type_base[pair_id]
  competing_group_competition_variable_candidate <- pair_competition_type
  competing_group_variable_candidate <- ifelse(
    pair_competition_type == "Same",
    competing_group_variable_respondent,
    ifelse(profile_order == 1, "PartyA", "PartyB")
  )

  common_args <- list(
    Y = Y,
    W = W,
    lambda = 0.05,
    pair_id = pair_id,
    respondent_id = respondent_id,
    respondent_task_id = respondent_task_id,
    profile_order = profile_order,
    adversarial = TRUE,
    diff = TRUE,
    competing_group_variable_respondent = competing_group_variable_respondent,
    competing_group_variable_candidate = competing_group_variable_candidate,
    competing_group_competition_variable_candidate = competing_group_competition_variable_candidate,
    nSGD = 15,
    nMonte_adversarial = 3L,
    nMonte_Qglm = 5L,
    primary_pushforward = "linearized",
    force_gaussian = TRUE,
    compute_se = TRUE,
    a_init_sd = 0.0,
    compute_hessian = FALSE,
    optimism = "none",
    conda_env = "strategize_env",
    conda_env_required = TRUE
  )

  set.seed(999)
  res_full <- do.call(strategize, c(common_args, list(se_method = "full")))
  set.seed(999)
  res_impl <- do.call(strategize, c(common_args, list(se_method = "implicit")))

  full_se <- res_full$pi_star_se_vec
  impl_se <- res_impl$pi_star_se_vec
  keep <- is.finite(full_se) & is.finite(impl_se)
  expect_true(any(keep))
  eps <- 5e-2
  expect_true(max(abs(full_se[keep] - impl_se[keep])) < eps)

  expect_true(is.finite(res_full$Q_se))
  expect_true(is.finite(res_impl$Q_se))
  expect_true(abs(res_full$Q_se - res_impl$Q_se) < eps)
})

test_that("implicit SEs match full-trace SEs (non-adversarial)", {
  old_force_no_interactions <- if (exists("force_no_interactions", inherits = TRUE)) {
    get("force_no_interactions", inherits = TRUE)
  } else {
    NULL
  }
  assign("force_no_interactions", TRUE, envir = .GlobalEnv)
  on.exit({
    if (is.null(old_force_no_interactions)) {
      if (exists("force_no_interactions", inherits = FALSE, envir = .GlobalEnv)) {
        rm(force_no_interactions, envir = .GlobalEnv)
      }
    } else {
      assign("force_no_interactions", old_force_no_interactions, envir = .GlobalEnv)
    }
  }, add = TRUE)

  set.seed(123)
  n_pairs <- 60
  n <- n_pairs * 2
  W <- cbind(
    matrix(sample(c("A", "B"), n, replace = TRUE), ncol = 1),
    matrix(sample(c("A", "B"), n, replace = TRUE), ncol = 1)
  )
  colnames(W) <- c("V1", "V2")
  pair_id <- respondent_id <- rep(seq_len(n_pairs), each = 2)
  respondent_task_id <- pair_id
  profile_order <- rep(1:2, times = n_pairs)
  utilities <- drop((W == "B") %*% c(0.4, 0.2))
  Y <- as.numeric(ave(
    utilities,
    respondent_task_id,
    FUN = function(g) rank(g, ties.method = "random") == length(g)
  ))

  respondent_party_base <- rep(c("PartyA", "PartyB"), length.out = n_pairs)
  competing_group_variable_respondent <- respondent_party_base[respondent_id]
  competition_type_base <- rep(c("Same", "Different"), length.out = n_pairs)
  set.seed(456)
  competition_type_base <- sample(competition_type_base)
  pair_competition_type <- competition_type_base[pair_id]
  competing_group_competition_variable_candidate <- pair_competition_type
  competing_group_variable_candidate <- ifelse(
    pair_competition_type == "Same",
    competing_group_variable_respondent,
    ifelse(profile_order == 1, "PartyA", "PartyB")
  )

  common_args <- list(
    Y = Y,
    W = W,
    lambda = 0.05,
    pair_id = pair_id,
    respondent_id = respondent_id,
    respondent_task_id = respondent_task_id,
    profile_order = profile_order,
    adversarial = FALSE,
    diff = TRUE,
    competing_group_variable_respondent = competing_group_variable_respondent,
    competing_group_variable_candidate = competing_group_variable_candidate,
    competing_group_competition_variable_candidate = competing_group_competition_variable_candidate,
    nSGD = 15,
    nMonte_adversarial = 3L,
    nMonte_Qglm = 5L,
    primary_pushforward = "linearized",
    force_gaussian = TRUE,
    compute_se = TRUE,
    a_init_sd = 0.0,
    compute_hessian = FALSE,
    optimism = "none",
    conda_env = "strategize_env",
    conda_env_required = TRUE
  )

  set.seed(999)
  res_full <- do.call(strategize, c(common_args, list(se_method = "full")))
  set.seed(999)
  res_impl <- do.call(strategize, c(common_args, list(se_method = "implicit")))

  full_se <- res_full$pi_star_se_vec
  impl_se <- res_impl$pi_star_se_vec
  keep <- is.finite(full_se) & is.finite(impl_se)
  expect_true(any(keep))
  eps <- 5e-2
  expect_true(max(abs(full_se[keep] - impl_se[keep])) < eps)

  expect_true(is.finite(res_full$Q_se))
  expect_true(is.finite(res_impl$Q_se))
  expect_true(abs(res_full$Q_se - res_impl$Q_se) < eps)
})
