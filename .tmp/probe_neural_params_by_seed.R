library(pkgload)
load_all('/Users/cjerzak/Documents/strategize-software/strategize', quiet=TRUE)
source('/Users/cjerzak/Documents/strategize-software/strategize/tests/testthat/helper-strategize.R')
extract_mu <- function(res, W) {
  model <- res$Y_models$my_model_ast_jnp
  if (is.null(model)) model <- res$Y_models$my_model_dag_jnp
  W_df <- as.data.frame(W, stringsAsFactors = FALSE)
  if (!is.null(names(res$p_list)) && !is.null(colnames(W_df))) W_df <- W_df[, names(res$p_list), drop = FALSE]
  W_num <- as.matrix(vapply(seq_along(res$p_list), function(d_) match(as.character(W_df[[d_]]), names(res$p_list[[d_]])), numeric(nrow(W_df))))
  pred <- model(X_new = W_num)
  as.numeric(pred$mu)
}
get_info <- function(res) {
  mi <- res$neural_model_info$ast
  if (is.null(mi)) mi <- res$neural_model_info$dag
  if (is.null(mi)) mi <- res$neural_model_info$ast0
  if (is.null(mi)) mi <- res$neural_model_info$dag0
  mi
}
fx <- generate_linear_average_case_fixture()
for (seed in 20260326:20260329) {
  withr::with_seed(seed, {
    res <- strategize(
      Y = fx$Y, W = fx$W, lambda = fx$lambda,
      outcome_model_type = 'neural', diff = FALSE, adversarial = FALSE,
      compute_se = FALSE, penalty_type = 'L2', use_regularization = FALSE,
      use_optax = FALSE, force_gaussian = FALSE, nSGD = 1000L, nMonte_Qglm = 1000L,
      a_init_sd = 0.001, optim_type = 'gd',
      neural_mcmc_control = list(
        subsample_method = 'batch_vi', ModelDims = 64L, ModelDepth = 2L,
        qk_norm = FALSE, batch_size = 512L, optimizer = 'adam',
        vi_guide = 'auto_diagonal', svi_steps = 200L, svi_num_draws = 100L,
        uncertainty_scope = 'output', eval_enabled = FALSE
      )
    )
    mi <- get_info(res)
    p <- mi$params
    alpha_attn <- sapply(1:mi$model_depth, function(l) as.numeric(reticulate::py_to_r(p[[paste0('alpha_attn_l', l)]])))
    alpha_ff <- sapply(1:mi$model_depth, function(l) as.numeric(reticulate::py_to_r(p[[paste0('alpha_ff_l', l)]])))
    sigma <- if (!is.null(p$sigma)) as.numeric(reticulate::py_to_r(p$sigma)) else NA_real_
    b_out <- if (!is.null(p$b_out)) as.numeric(reticulate::py_to_r(p$b_out)) else NA_real_
    W_out <- if (!is.null(p$W_out)) as.numeric(reticulate::py_to_r(strategize:::strenv$jnp$linalg$norm(p$W_out))) else NA_real_
    mu_hat <- extract_mu(res, fx$W)
    fit <- lm(mu_hat ~ fx$mu_true)
    Q_hat <- if (!is.null(res$Q_point_mEst) && all(is.finite(res$Q_point_mEst))) as.numeric(res$Q_point_mEst) else as.numeric(res$Q_point)
    cat(sprintf('seed=%d Qhat=%.4f sigma=%.4f b_out=%.4f ||W_out||=%.4f alpha_attn=%s alpha_ff=%s intercept=%.4f slope=%.4f rmse=%.4f cor=%.4f\n',
      seed, Q_hat, sigma, b_out, W_out,
      paste(sprintf('%.4f', alpha_attn), collapse=','),
      paste(sprintf('%.4f', alpha_ff), collapse=','),
      coef(fit)[1], coef(fit)[2], sqrt(mean((mu_hat - fx$mu_true)^2)), cor(mu_hat, fx$mu_true)
    ))
  })
}
