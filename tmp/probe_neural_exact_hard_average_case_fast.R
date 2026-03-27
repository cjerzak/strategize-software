pkgload::load_all("/Users/cjerzak/Documents/strategize-software/strategize", quiet = TRUE)
source("/Users/cjerzak/Documents/strategize-software/strategize/tests/testthat/helper-strategize.R")

extract_average_case_pi_hat <- function(res) {
  pi_obj <- res[["pi_star_point"]]
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

get_neural_model_info <- function(res) {
  model_info <- res$neural_model_info$ast
  if (is.null(model_info)) model_info <- res$neural_model_info$dag
  if (is.null(model_info)) model_info <- res$neural_model_info$ast0
  if (is.null(model_info)) model_info <- res$neural_model_info$dag0
  model_info
}

theta_mean_from_res <- function(res) {
  tryCatch(
    as.numeric(reticulate::py_to_r(res$est_coefficients_jnp)),
    error = function(e) {
      as.numeric(reticulate::py_to_r(strategize:::strenv$np$array(res$est_coefficients_jnp)))
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
  apply(grid, 1L, function(row_i) prod(ifelse(row_i == "1", pi_hat, 1 - pi_hat)))
}

fx <- generate_linear_average_case_fixture()
seed <- 20260326L

set.seed(seed)
res <- strategize(
  Y = fx[["Y"]],
  W = fx[["W"]],
  lambda = fx[["lambda"]],
  outcome_model_type = "neural",
  diff = FALSE,
  adversarial = FALSE,
  compute_se = FALSE,
  penalty_type = "L2",
  use_regularization = FALSE,
  use_optax = FALSE,
  force_gaussian = FALSE,
  nSGD = 100L,
  nMonte_Qglm = 200L,
  a_init_sd = 0.001,
  optim_type = "gd",
  neural_mcmc_control = list(
    subsample_method = "batch_vi",
    ModelDims = 32L,
    ModelDepth = 2L,
    qk_norm = FALSE,
    batch_size = 512L,
    optimizer = "adam",
    vi_guide = "auto_diagonal",
    svi_steps = 50L,
    svi_num_draws = 50L,
    uncertainty_scope = "output",
    eval_enabled = FALSE
  )
)

pi_hat <- extract_average_case_pi_hat(res)
q_report <- as.numeric(unlist(res$Q_point_mEst))[1L]
model_info <- get_neural_model_info(res)
theta_mean <- theta_mean_from_res(res)

bundle <- strategize:::cs2step_build_neural_outcome_bundle(
  theta_mean = theta_mean,
  theta_var = NULL,
  neural_model_info = model_info,
  names_list = lapply(res$p_list, names),
  factor_levels = vapply(res$p_list, length, integer(1)),
  mode = "single"
)
grid <- build_exact_grid(fx)
grid_prob <- profile_probabilities(grid, pi_hat)
fit_loaded <- strategize:::cs2step_unpack_predictor(bundle, preload_params = TRUE)
prep_params <- strategize:::cs2step_neural_prepare_params(
  fit_loaded,
  conda_env = "strategize_env",
  conda_env_required = TRUE
)
grid_idx <- apply(as.matrix(grid), 2L, as.integer)
if (is.null(dim(grid_idx))) {
  grid_idx <- matrix(grid_idx, ncol = ncol(grid))
}
colnames(grid_idx) <- colnames(grid)
prep <- strategize:::cs2step_neural_prepare_prediction_data(
  W_idx = grid_idx,
  model_info = prep_params$model_info,
  mode = "single"
)
hard_pred_raw <- strategize:::cs2step_neural_predict_prepared(
  prep_params$params,
  prep_params$model_info,
  prep,
  return_logits = FALSE
)
hard_pred <- if (is.list(hard_pred_raw) && !is.null(hard_pred_raw$mu)) {
  as.numeric(strategize:::cs2step_neural_to_r_array(hard_pred_raw$mu))
} else {
  as.numeric(strategize:::cs2step_neural_to_r_array(hard_pred_raw))
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

cat(sprintf("seed=%d trueQ=%.6f\n", seed, fx$trueQ))
cat(sprintf("Q_report_hard_mc=%.6f\n", q_report))
cat(sprintf("Q_exact_hard_predict=%.6f\n", q_exact_hard))
cat(sprintf("Q_soft_direct=%.6f\n", q_soft))
cat(sprintf("pi_rel_err=%.6f\n",
            mean(abs(pi_hat - fx$pi_star_true) / pmax(abs(fx$pi_star_true), 1e-8))))
