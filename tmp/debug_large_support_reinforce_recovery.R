pkgload::load_all("/Users/cjerzak/Documents/strategize-software/strategize", quiet = TRUE)
source("/Users/cjerzak/Documents/strategize-software/strategize/tests/testthat/helper-strategize.R")

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

k_factors <- 20L
fixture <- generate_linear_average_case_fixture(k_factors = k_factors)

withr::local_seed(20260326L)
res <- strategize(
  Y = fixture$Y,
  W = fixture$W,
  lambda = fixture$lambda,
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

pi_hat <- extract_average_case_pi_hat(res)
Q_point_raw <- res$Q_point
Q_point_mEst_raw <- res$Q_point_mEst
Q_point <- if (!is.null(Q_point_raw)) as.numeric(unlist(Q_point_raw))[1L] else NA_real_
Q_point_mEst <- if (!is.null(Q_point_mEst_raw)) as.numeric(unlist(Q_point_mEst_raw))[1L] else NA_real_
Q_hat <- if (!is.na(Q_point_mEst) && is.finite(Q_point_mEst)) Q_point_mEst else Q_point

pi_rel_err <- mean(abs(pi_hat - fixture$pi_star_true) / pmax(abs(fixture$pi_star_true), 1e-8))
Q_rel_err <- abs(Q_hat - fixture$trueQ) / pmax(abs(fixture$trueQ), 1e-8)

q_true_of_pi_hat <- sum(fixture$my_beta * fixture$getInteractionWts(pi_hat))
q_true_of_pi_hat_rel_err <- abs(q_true_of_pi_hat - fixture$trueQ) / pmax(abs(fixture$trueQ), 1e-8)

mu_hat <- extract_average_case_neural_mu_hat(res, fixture$W)
rmse_y <- sqrt(mean((mu_hat - fixture$Y) ^ 2))
rmse_null <- sqrt(mean((mean(fixture$Y) - fixture$Y) ^ 2))
rmse_mu_true <- sqrt(mean((mu_hat - fixture$mu_true) ^ 2))
cor_mu <- suppressWarnings(stats::cor(mu_hat, fixture$mu_true))
bias_mu <- mean(mu_hat - fixture$mu_true)

cat(sprintf("k=%d\n", k_factors))
cat(sprintf("objective_gradient_mode=%s\n", res$convergence_history$objective_gradient_mode))
cat(sprintf("reinforce_nonfinite_ast_steps=%d\n", as.integer(res$convergence_history$reinforce_nonfinite_ast_steps)))
cat(sprintf("pi_rel_err=%.6f\n", pi_rel_err))
cat(sprintf("Q_rel_err=%.6f\n", Q_rel_err))
cat(sprintf("Q_point=%.6f\n", Q_point))
cat(sprintf("Q_point_mEst=%.6f\n", Q_point_mEst))
cat(sprintf("Q_hat_used=%.6f\n", Q_hat))
cat(sprintf("trueQ=%.6f\n", fixture$trueQ))
cat(sprintf("true_fixture_Q_at_pi_hat=%.6f\n", q_true_of_pi_hat))
cat(sprintf("true_fixture_Q_at_pi_hat_rel_err=%.6f\n", q_true_of_pi_hat_rel_err))
cat(sprintf("rmse_y=%.6f\n", rmse_y))
cat(sprintf("rmse_null=%.6f\n", rmse_null))
cat(sprintf("rmse_mu_true=%.6f\n", rmse_mu_true))
cat(sprintf("cor_mu=%.6f\n", cor_mu))
cat(sprintf("bias_mu=%.6f\n", bias_mu))
