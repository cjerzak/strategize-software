#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(pkgload)
})

pkgload::load_all("/Users/cjerzak/Documents/strategize-software/strategize", quiet = TRUE)

args <- commandArgs(trailingOnly = TRUE)
k_factors <- if (length(args) >= 1L) as.integer(args[[1L]]) else 20L
n_eval <- if (length(args) >= 2L) as.integer(args[[2L]]) else 5000L
seed <- if (length(args) >= 3L) as.integer(args[[3L]]) else 20260326L

cache_file <- sprintf(
  "/Users/cjerzak/Documents/strategize-software/Tmp/neural_q_bias_fit_k%d_seed%d.rds",
  k_factors,
  seed
)
if (!file.exists(cache_file)) {
  stop(sprintf("Cache file not found: %s", cache_file), call. = FALSE)
}

fit_obj <- readRDS(cache_file)
res <- fit_obj$res

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

sample_W_from_pi <- function(pi_vec, p_list, n, seed_draw) {
  set.seed(as.integer(seed_draw))
  W <- vapply(seq_along(pi_vec), function(j) {
    draws <- rbinom(n, size = 1L, prob = pi_vec[[j]])
    levels_j <- names(p_list[[j]])
    if (length(levels_j) != 2L) {
      stop("This probe assumes binary factors.", call. = FALSE)
    }
    ifelse(draws == 1L, levels_j[[2L]], levels_j[[1L]])
  }, character(n))
  if (is.null(dim(W))) {
    W <- matrix(W, ncol = length(pi_vec))
  }
  colnames(W) <- names(p_list)
  W
}

encode_reduced_policy_profiles <- function(W, p_list, parameterization_type) {
  W_df <- as.data.frame(W, stringsAsFactors = FALSE)
  out <- vector("list", length(p_list))
  for (d_ in seq_along(p_list)) {
    level_names <- names(p_list[[d_]])
    w_col <- as.character(W_df[[d_]])
    if (identical(parameterization_type, "Implicit")) {
      explicit_levels <- level_names[-length(level_names)]
      block <- vapply(explicit_levels, function(level_name) {
        as.numeric(w_col == level_name)
      }, numeric(nrow(W_df)))
      if (is.null(dim(block))) {
        block <- matrix(block, ncol = max(1L, length(explicit_levels)))
      }
    } else {
      block <- vapply(level_names, function(level_name) {
        as.numeric(w_col == level_name)
      }, numeric(nrow(W_df)))
      if (is.null(dim(block))) {
        block <- matrix(block, ncol = length(level_names))
      }
    }
    out[[d_]] <- block
  }
  encoded <- do.call(cbind, out)
  if (is.null(dim(encoded))) {
    encoded <- matrix(encoded, ncol = 1L)
  }
  storage.mode(encoded) <- "double"
  encoded
}

pi_hat <- extract_average_case_pi_hat(res)
W_pi_hat <- sample_W_from_pi(pi_hat, res$p_list, n_eval, seed + 501L)
public_pred <- extract_average_case_neural_mu_hat(res, W_pi_hat)
pi_reduced <- encode_reduced_policy_profiles(
  W_pi_hat,
  p_list = res$p_list,
  parameterization_type = res$ParameterizationType
)

full_env <- environment(res$FullGetQStar_)
old_model_env <- strategize:::strenv$neural_model_env
on.exit({
  strategize:::strenv$neural_model_env <- old_model_env
}, add = TRUE)
strategize:::strenv$neural_model_env <- full_env

pi_reduced_jnp <- strategize:::strenv$jnp$array(pi_reduced)$astype(strategize:::strenv$dtj)
internal_pred_raw <- strategize:::strenv$jax$vmap(
  function(pi_row) {
    strategize:::neural_getQStar_single(
      pi_star_ast = pi_row,
      EST_COEFFICIENTS_tf_ast = res$est_coefficients_jnp
    )
  },
  in_axes = 0L
)(pi_reduced_jnp)
internal_pred <- as.matrix(strategize:::strenv$np$array(internal_pred_raw))
internal_pred <- as.numeric(internal_pred[, 1L])

delta <- internal_pred - public_pred

cat(sprintf("k=%d n_eval=%d seed=%d\n", k_factors, n_eval, seed))
cat(sprintf("public_mean=%.6f\n", mean(public_pred)))
cat(sprintf("internal_mean=%.6f\n", mean(internal_pred)))
cat(sprintf("mean_delta_internal_minus_public=%.6f\n", mean(delta)))
cat(sprintf("rmse_internal_vs_public=%.6f\n", sqrt(mean(delta ^ 2))))
cat(sprintf("max_abs_delta=%.6f\n", max(abs(delta))))
cat(sprintf("cor_internal_public=%.6f\n", suppressWarnings(stats::cor(internal_pred, public_pred))))
