#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(pkgload)
})

pkgload::load_all("/Users/cjerzak/Documents/strategize-software/strategize", quiet = TRUE)
source("/Users/cjerzak/Documents/strategize-software/strategize/tests/testthat/helper-strategize.R")

args <- commandArgs(trailingOnly = TRUE)
k_factors <- if (length(args) >= 1L) as.integer(args[[1L]]) else 20L
n_eval <- if (length(args) >= 2L) as.integer(args[[2L]]) else 50000L
n_mc_reps <- if (length(args) >= 3L) as.integer(args[[3L]]) else 100L
seed <- if (length(args) >= 4L) as.integer(args[[4L]]) else 20260326L

dir.create("/Users/cjerzak/Documents/strategize-software/Tmp", showWarnings = FALSE)
cache_file <- sprintf(
  "/Users/cjerzak/Documents/strategize-software/Tmp/neural_q_bias_fit_k%d_seed%d.rds",
  k_factors,
  seed
)
use_cache <- !identical(toupper(Sys.getenv("NO_CACHE", unset = "0")), "1")

extract_average_case_pi_hat <- function(res) {
  pi_obj <- res$pi_star_point
  if (is.list(pi_obj) && length(pi_obj) == 1L && is.list(pi_obj[[1L]])) {
    pi_obj <- pi_obj[[1L]]
  }
  vapply(pi_obj, function(prob_vec) {
    if (!is.null(names(prob_vec)) && "1" %in% names(prob_vec)) {
      return(as.numeric(prob_vec[["1"]]))
    }
    as.numeric(prob_vec[[2L]])
  }, numeric(1))
}

extract_average_case_neural_mu_hat <- function(res, W) {
  model <- res$Y_models$my_model_ast_jnp
  if (is.null(model)) {
    model <- res$Y_models$my_model_dag_jnp
  }
  if (!is.function(model)) {
    stop("Neural average-case fit did not expose a prediction function.", call. = FALSE)
  }

  W_df <- as.data.frame(W, stringsAsFactors = FALSE)
  if (!is.null(names(res$p_list)) && !is.null(colnames(W_df))) {
    W_df <- W_df[, names(res$p_list), drop = FALSE]
  }

  W_num <- as.matrix(vapply(seq_along(res$p_list), function(d_) {
    level_names <- names(res$p_list[[d_]])
    match(as.character(W_df[[d_]]), level_names)
  }, numeric(nrow(W_df))))

  pred <- model(X_new = W_num)
  if (is.list(pred) && !is.null(pred$mu)) {
    return(as.numeric(pred$mu))
  }
  as.numeric(pred)
}

fixture_true_mu_from_W <- function(fx, W) {
  X_obs <- apply(W, 2L, as.integer)
  X_inter <- build_pairwise_interaction_matrix(X_obs, fx$KChoose2_combs)
  as.numeric(cbind(X_obs, X_inter) %*% fx$my_beta)
}

sample_W_from_pi <- function(pi_vec, n, seed_draw) {
  set.seed(as.integer(seed_draw))
  X <- vapply(seq_along(pi_vec), function(j) {
    rbinom(n, size = 1L, prob = pi_vec[[j]])
  }, integer(n))
  if (is.null(dim(X))) {
    X <- matrix(X, ncol = length(pi_vec))
  }
  storage.mode(X) <- "integer"
  W <- apply(X, 2L, as.character)
  colnames(W) <- as.character(seq_len(ncol(X)))
  W
}

evaluate_distribution <- function(label, res, fx, W) {
  mu_hat <- extract_average_case_neural_mu_hat(res, W)
  mu_true <- fixture_true_mu_from_W(fx, W)
  data.frame(
    distribution = label,
    n = nrow(W),
    mean_hat = mean(mu_hat),
    mean_true = mean(mu_true),
    mean_bias = mean(mu_hat - mu_true),
    rel_bias_vs_trueQ = mean(mu_hat - mu_true) / fx$trueQ,
    rmse = sqrt(mean((mu_hat - mu_true) ^ 2)),
    mae = mean(abs(mu_hat - mu_true)),
    cor = suppressWarnings(stats::cor(mu_hat, mu_true))
  )
}

if (isTRUE(use_cache) && file.exists(cache_file)) {
  fit_obj <- readRDS(cache_file)
  fx <- fit_obj$fixture
  res <- fit_obj$res
} else {
  fx <- generate_linear_average_case_fixture(k_factors = k_factors)
  withr::local_seed(seed)
  res <- strategize(
    Y = fx$Y,
    W = fx$W,
    lambda = fx$lambda,
    outcome_model_type = "neural",
    diff = FALSE,
    adversarial = FALSE,
    compute_se = FALSE,
    penalty_type = "L2",
    use_regularization = FALSE,
    use_optax = FALSE,
    force_gaussian = FALSE,
    nSGD = 1000L,
    nMonte_Qglm = 1000L,
    a_init_sd = 0.001,
    optim_type = "gd",
    neural_mcmc_control = list(
      subsample_method = "batch_vi",
      ModelDims = 64L,
      ModelDepth = 2L,
      qk_norm = FALSE,
      batch_size = 512L,
      optimizer = "adam",
      vi_guide = "auto_diagonal",
      uncertainty_scope = "output",
      eval_enabled = FALSE
    )
  )
  if (isTRUE(use_cache)) {
    saveRDS(list(fixture = fx, res = res), cache_file)
  }
}

pi_hat <- extract_average_case_pi_hat(res)
Q_point_mEst <- as.numeric(unlist(res$Q_point_mEst))[1L]
Q_point <- as.numeric(unlist(res$Q_point))[1L]
Q_hat <- if (is.finite(Q_point_mEst)) Q_point_mEst else Q_point

q_true_of_pi_hat <- sum(fx$my_beta * fx$getInteractionWts(pi_hat))
q_true_of_pi_true <- sum(fx$my_beta * fx$getInteractionWts(fx$pi_star_true))

W_pi_hat <- sample_W_from_pi(pi_hat, n_eval, seed + 101L)
W_pi_true <- sample_W_from_pi(fx$pi_star_true, n_eval, seed + 202L)
W_half <- sample_W_from_pi(rep(0.5, length(pi_hat)), n_eval, seed + 303L)

dist_results <- do.call(
  rbind,
  list(
    evaluate_distribution("observed_data", res, fx, fx$W),
    evaluate_distribution("pi_hat_draws", res, fx, W_pi_hat),
    evaluate_distribution("pi_true_draws", res, fx, W_pi_true),
    evaluate_distribution("half_draws", res, fx, W_half)
  )
)

q_model_pi_hat_mc <- mean(extract_average_case_neural_mu_hat(res, W_pi_hat))
q_model_pi_true_mc <- mean(extract_average_case_neural_mu_hat(res, W_pi_true))

q_mc_reps <- vapply(seq_len(n_mc_reps), function(rep_idx) {
  W_rep <- sample_W_from_pi(pi_hat, 1000L, seed + 1000L + rep_idx)
  mean(extract_average_case_neural_mu_hat(res, W_rep))
}, numeric(1))

cat(sprintf("k=%d seed=%d cache=%s\n", k_factors, seed, basename(cache_file)))
cat(sprintf("objective_gradient_mode=%s\n", res$convergence_history$objective_gradient_mode))
cat(sprintf("reinforce_nonfinite_ast_steps=%d\n", as.integer(res$convergence_history$reinforce_nonfinite_ast_steps)))
cat(sprintf("pi_rel_err=%.6f\n", mean(abs(pi_hat - fx$pi_star_true) / pmax(abs(fx$pi_star_true), 1e-8))))
cat(sprintf("Q_hat_report=%.6f\n", Q_hat))
cat(sprintf("trueQ=%.6f\n", fx$trueQ))
cat(sprintf("true_fixture_Q_at_pi_hat=%.6f\n", q_true_of_pi_hat))
cat(sprintf("true_fixture_Q_at_pi_true=%.6f\n", q_true_of_pi_true))
cat(sprintf("model_Q_mc_at_pi_hat=%.6f\n", q_model_pi_hat_mc))
cat(sprintf("model_Q_mc_at_pi_true=%.6f\n", q_model_pi_true_mc))
cat(sprintf("report_minus_model_Q_mc_at_pi_hat=%.6f\n", Q_hat - q_model_pi_hat_mc))
cat(sprintf("model_minus_true_fixture_Q_at_pi_hat=%.6f\n", q_model_pi_hat_mc - q_true_of_pi_hat))
cat(sprintf("model_minus_true_fixture_Q_at_pi_true=%.6f\n", q_model_pi_true_mc - fx$trueQ))
cat(sprintf("mc1000_mean=%.6f\n", mean(q_mc_reps)))
cat(sprintf("mc1000_sd=%.6f\n", stats::sd(q_mc_reps)))
cat(sprintf("mc1000_min=%.6f\n", min(q_mc_reps)))
cat(sprintf("mc1000_max=%.6f\n", max(q_mc_reps)))
cat("\nDistribution diagnostics\n")
print(dist_results, row.names = FALSE, digits = 6)
