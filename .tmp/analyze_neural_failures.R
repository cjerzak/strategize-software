pkgload::load_all(
  "/Users/cjerzak/Documents/strategize-software/strategize",
  quiet = TRUE
)
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

theta_mean_from_res <- function(res) {
  tryCatch(
    as.numeric(reticulate::py_to_r(res$est_coefficients_jnp)),
    error = function(e) {
      as.numeric(
        reticulate::py_to_r(strategize:::strenv$np$array(res$est_coefficients_jnp))
      )
    }
  )
}

build_exact_grid <- function(fx) {
  k <- length(fx$pi_star_true)
  levs <- replicate(k, c("0", "1"), simplify = FALSE)
  grid <- expand.grid(levs, KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
  colnames(grid) <- colnames(fx$W)
  grid
}

profile_probabilities <- function(grid, pi_hat) {
  apply(grid, 1L, function(row_i) {
    prod(ifelse(row_i == "1", pi_hat, 1 - pi_hat))
  })
}

get_neural_model_info <- function(res) {
  model_info <- res$neural_model_info$ast
  if (is.null(model_info)) model_info <- res$neural_model_info$dag
  if (is.null(model_info)) model_info <- res$neural_model_info$ast0
  if (is.null(model_info)) model_info <- res$neural_model_info$dag0
  model_info
}

compute_binary_null_metrics <- function(y) {
  y <- as.numeric(y)
  y <- y[is.finite(y)]
  p_null <- mean(y)
  p_null <- min(max(p_null, 1e-6), 1 - 1e-6)
  list(
    log_loss = -mean(y * log(p_null) + (1 - y) * log(1 - p_null)),
    accuracy = max(mean(y), 1 - mean(y)),
    brier = mean((p_null - y) ^ 2)
  )
}

run_pairwise_fit <- function() {
  withr::local_envvar(c(
    STRATEGIZE_NEURAL_FAST_MCMC = "true",
    STRATEGIZE_NEURAL_EVAL_FOLDS = "2",
    STRATEGIZE_NEURAL_EVAL_SEED = "123"
  ))

  data <- generate_test_data(n = 40, seed = 123)
  params <- default_strategize_params(fast = TRUE)
  params$outcome_model_type <- "neural"
  p_list <- generate_test_p_list(data$W)

  res <- do.call(strategize, c(
    list(Y = data$Y, W = data$W, p_list = p_list),
    data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order")],
    params
  ))

  list(res = res, data = data, p_list = p_list)
}

compute_exact_qs <- function(res, fx, pi_hat) {
  model_info <- get_neural_model_info(res)
  theta_mean <- theta_mean_from_res(res)
  bundle <- strategize:::cs2step_build_neural_outcome_bundle(
    theta_mean = theta_mean,
    theta_var = NULL,
    neural_model_info = model_info,
    names_list = lapply(res$p_list, names),
    factor_levels = vapply(res$p_list, length, integer(1)),
    mode = "single",
    conda_env = "strategize_env",
    conda_env_required = TRUE
  )
  fit_loaded <- strategize:::cs2step_unpack_predictor(
    bundle,
    conda_env = "strategize_env",
    conda_env_required = TRUE,
    preload_params = TRUE
  )

  grid <- build_exact_grid(fx)
  grid_prob <- profile_probabilities(grid, pi_hat)
  hard_pred_raw <- stats::predict(fit_loaded, newdata = grid, type = "response")
  hard_pred <- if (is.list(hard_pred_raw) && !is.null(hard_pred_raw$mu)) {
    as.numeric(unlist(hard_pred_raw$mu))
  } else {
    as.numeric(unlist(hard_pred_raw))
  }
  q_exact_hard <- sum(grid_prob * hard_pred)

  full_env <- environment(res$FullGetQStar_)
  old_model_env <- strategize:::strenv$neural_model_env
  on.exit({
    strategize:::strenv$neural_model_env <- old_model_env
  }, add = TRUE)
  strategize:::strenv$neural_model_env <- full_env
  q_soft <- as.numeric(strategize:::strenv$np$array(
    strategize:::neural_getQStar_single(
      pi_star_ast = res$pi_star_ast_vec_jnp,
      EST_COEFFICIENTS_tf_ast = res$est_coefficients_jnp
    )
  ))[1L]

  list(q_exact_hard = q_exact_hard, q_soft = q_soft)
}

cat("=== average-case ===\n")
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
    mu_hat <- extract_average_case_neural_mu_hat(res, fx$W)
    q_hat <- if (!is.null(res$Q_point_mEst) && all(is.finite(res$Q_point_mEst))) {
      as.numeric(res$Q_point_mEst)
    } else {
      as.numeric(res$Q_point)
    }
    lm_fit <- lm(mu_hat ~ fx$mu_true)
    exact_q <- compute_exact_qs(res, fx, pi_hat)

    cat(sprintf(
      paste(
        "seed=%d",
        "pi_rel=%.4f",
        "Q_rel=%.4f",
        "Q_hat=%.4f",
        "q_exact_hard=%.4f",
        "q_soft=%.4f",
        "rmse_mu_true=%.4f",
        "cor=%.4f",
        "mean_hat=%.4f",
        "mean_true=%.4f",
        "sd_hat=%.4f",
        "sd_true=%.4f",
        "lm_int=%.4f",
        "lm_slope=%.4f",
        sep = " "
      ),
      seed,
      mean(abs(pi_hat - fx$pi_star_true) / pmax(abs(fx$pi_star_true), 1e-8)),
      abs(q_hat - fx$trueQ) / pmax(abs(fx$trueQ), 1e-8),
      q_hat,
      exact_q$q_exact_hard,
      exact_q$q_soft,
      sqrt(mean((mu_hat - fx$mu_true) ^ 2)),
      cor(mu_hat, fx$mu_true),
      mean(mu_hat),
      mean(fx$mu_true),
      sd(mu_hat),
      sd(fx$mu_true),
      coef(lm_fit)[1],
      coef(lm_fit)[2]
    ))
    cat("\n")
  })
}

cat("=== pairwise ===\n")
pair_fit <- run_pairwise_fit()
pair_info <- get_neural_model_info(pair_fit$res)
pair_metrics <- pair_info$fit_metrics
y_eval <- pair_fit$data$Y[pair_fit$data$profile_order == 1L]
null_metrics <- compute_binary_null_metrics(y_eval)
print(pair_metrics[c(
  "likelihood", "n_eval", "auc", "log_loss", "accuracy", "brier",
  "eval_note", "n_folds", "seed"
)])
cat("null\n")
print(null_metrics)
cat("by_fold\n")
print(pair_metrics$by_fold)
