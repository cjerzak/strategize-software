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

fx <- generate_linear_average_case_fixture()

for (seed in 20260326:20260329) {
  withr::with_seed(seed, {
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
        svi_steps = 200L,
        svi_num_draws = 100L,
        uncertainty_scope = "output",
        eval_enabled = FALSE
      )
    )

    pi_hat <- extract_average_case_pi_hat(res)
    Q_hat <- if (!is.null(res$Q_point_mEst) && all(is.finite(res$Q_point_mEst))) {
      as.numeric(res$Q_point_mEst)
    } else {
      as.numeric(res$Q_point)
    }
    mu_hat <- extract_average_case_neural_mu_hat(res, fx$W)

    cat(sprintf(
      paste0(
        "seed=%d pi_rel=%.4f Q_rel=%.4f Q_hat=%.4f ",
        "mean_mu=%.4f sd_mu=%.4f mean_true=%.4f sd_true=%.4f ",
        "rmse_mu_true=%.4f cor=%.4f\n"
      ),
      seed,
      mean(abs(pi_hat - fx$pi_star_true) / pmax(abs(fx$pi_star_true), 1e-8)),
      abs(Q_hat - fx$trueQ) / pmax(abs(fx$trueQ), 1e-8),
      Q_hat,
      mean(mu_hat),
      stats::sd(mu_hat),
      mean(fx$mu_true),
      stats::sd(fx$mu_true),
      sqrt(mean((mu_hat - fx$mu_true) ^ 2)),
      stats::cor(mu_hat, fx$mu_true)
    ))
  })
}
