suppressPackageStartupMessages({
  library(pkgload)
  library(withr)
})

repo_root <- normalizePath("/Users/cjerzak/Documents/strategize-software", winslash = "/", mustWork = TRUE)
pkg_root <- file.path(repo_root, "strategize")

pkgload::load_all(pkg_root, quiet = TRUE, export_all = FALSE, helpers = FALSE)
sys.source(file.path(pkg_root, "tests", "testthat", "helper-strategize.R"), envir = globalenv())

get_neural_model_info_local <- function(res) {
  model_info <- res$neural_model_info$ast
  if (is.null(model_info)) model_info <- res$neural_model_info$dag
  if (is.null(model_info)) model_info <- res$neural_model_info$ast0
  if (is.null(model_info)) model_info <- res$neural_model_info$dag0
  model_info
}

extract_average_case_pi_hat_local <- function(res) {
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

extract_average_case_neural_mu_hat_local <- function(res, W) {
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

safe_num <- function(x) {
  if (is.null(x) || !length(x)) {
    return(NA_real_)
  }
  as.numeric(x[[1L]])
}

compute_binary_null_metrics_local <- function(y) {
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

param_l2 <- function(x) {
  np <- strategize:::strenv$np
  arr <- reticulate::py_to_r(np$array(x))
  sqrt(sum(as.numeric(arr) ^ 2))
}

patch_full_attn_block2 <- function() {
  ns <- asNamespace("strategize")
  fn <- function(tokens,
                 model_info,
                 params = NULL,
                 return_details = FALSE) {
    if (is.null(params)) {
      params <- model_info$params
    }
    residual_mode <- strategize:::neural_transformer_residual_mode(model_info)
    use_full_attn_residual <- identical(residual_mode, "full_attn")
    if (isTRUE(use_full_attn_residual)) {
      strategize:::neural_validate_full_attn_compatibility(
        model_info = model_info,
        params = params,
        context = "Neural transformer"
      )
    }
    residual_history <- if (isTRUE(use_full_attn_residual)) {
      strategize:::neural_init_residual_history(tokens)
    } else {
      NULL
    }
    for (l_ in 1L:strategize:::ai(model_info$model_depth)) {
      Wq <- params[[paste0("W_q_l", l_)]]
      Wk <- params[[paste0("W_k_l", l_)]]
      Wv <- params[[paste0("W_v_l", l_)]]
      Wo <- params[[paste0("W_o_l", l_)]]
      Wff1 <- params[[paste0("W_ff1_l", l_)]]
      Wff2 <- params[[paste0("W_ff2_l", l_)]]
      RMS_attn <- params[[paste0("RMS_attn_l", l_)]]
      RMS_ff <- params[[paste0("RMS_ff_l", l_)]]
      RMS_q <- params[[paste0("RMS_q_l", l_)]]
      RMS_k <- params[[paste0("RMS_k_l", l_)]]

      if (isTRUE(use_full_attn_residual)) {
        h_attn <- strategize:::neural_full_attn_residual_from_history(
          residual_history,
          pseudo_query = params[[paste0("pseudo_query_attn_l", l_)]],
          model_dims = model_info$model_dims
        )
        tokens_norm <- strategize:::neural_rms_norm(h_attn, RMS_attn, model_info$model_dims)
      } else {
        alpha_attn <- strategize:::neural_param_or_default(params, paste0("alpha_attn_l", l_), 1.0)
        alpha_ff <- strategize:::neural_param_or_default(params, paste0("alpha_ff_l", l_), 1.0)
        tokens_norm <- strategize:::neural_rms_norm(tokens, RMS_attn, model_info$model_dims)
      }

      Q <- strategize:::strenv$jnp$einsum("ntm,mk->ntk", tokens_norm, Wq)
      K <- strategize:::strenv$jnp$einsum("ntm,mk->ntk", tokens_norm, Wk)
      V <- strategize:::strenv$jnp$einsum("ntm,mk->ntk", tokens_norm, Wv)

      Qh <- strategize:::strenv$jnp$reshape(
        Q,
        list(Q$shape[[1]], Q$shape[[2]],
             strategize:::ai(model_info$n_heads), strategize:::ai(model_info$head_dim))
      )
      Kh <- strategize:::strenv$jnp$reshape(
        K,
        list(K$shape[[1]], K$shape[[2]],
             strategize:::ai(model_info$n_heads), strategize:::ai(model_info$head_dim))
      )
      Vh <- strategize:::strenv$jnp$reshape(
        V,
        list(V$shape[[1]], V$shape[[2]],
             strategize:::ai(model_info$n_heads), strategize:::ai(model_info$head_dim))
      )
      Qh <- strategize:::neural_rms_norm(Qh, RMS_q, model_info$head_dim)
      Kh <- strategize:::neural_rms_norm(Kh, RMS_k, model_info$head_dim)
      scale_ <- strategize:::strenv$jnp$sqrt(
        strategize:::strenv$jnp$array(as.numeric(strategize:::ai(model_info$head_dim)))
      )
      scores <- strategize:::strenv$jnp$einsum("nqhd,nkhd->nhqk", Qh, Kh) / scale_
      attn <- strategize:::strenv$jax$nn$softmax(scores, axis = -1L)
      context_h <- strategize:::strenv$jnp$einsum("nhqk,nkhd->nqhd", attn, Vh)
      context <- strategize:::strenv$jnp$reshape(
        context_h,
        list(context_h$shape[[1]], context_h$shape[[2]],
             strategize:::ai(model_info$model_dims))
      )
      attn_out <- strategize:::strenv$jnp$einsum("ntm,mk->ntk", context, Wo)

      if (isTRUE(use_full_attn_residual)) {
        partial_block <- h_attn + attn_out
        partial_history <- strategize:::neural_append_residual_history(residual_history, partial_block)
        h_ff <- strategize:::neural_full_attn_residual_from_history(
          partial_history,
          pseudo_query = params[[paste0("pseudo_query_ff_l", l_)]],
          model_dims = model_info$model_dims
        )
        h_ff_norm <- strategize:::neural_rms_norm(h_ff, RMS_ff, model_info$model_dims)
        ff_pre <- strategize:::strenv$jnp$einsum("ntm,mf->ntf", h_ff_norm, Wff1)
      } else {
        h1 <- tokens + alpha_attn * attn_out
        h1_norm <- strategize:::neural_rms_norm(h1, RMS_ff, model_info$model_dims)
        ff_pre <- strategize:::strenv$jnp$einsum("ntm,mf->ntf", h1_norm, Wff1)
      }

      ff_act <- strategize:::strenv$jax$nn$swish(ff_pre)
      ff_out <- strategize:::strenv$jnp$einsum("ntf,fm->ntm", ff_act, Wff2)

      if (isTRUE(use_full_attn_residual)) {
        block_out <- partial_block + ff_out
        residual_history <- strategize:::neural_append_residual_history(residual_history, block_out)
        tokens <- block_out
      } else {
        tokens <- h1 + alpha_ff * ff_out
      }
    }

    tokens_final <- strategize:::neural_rms_norm(tokens, params$RMS_final, model_info$model_dims)
    readout_tokens <- tokens_final
    if (isTRUE(use_full_attn_residual)) {
      h_out <- strategize:::neural_full_attn_residual_from_history(
        residual_history,
        pseudo_query = params[["pseudo_query_final"]],
        model_dims = model_info$model_dims
      )
      readout_tokens <- strategize:::neural_rms_norm(h_out, params$RMS_final, model_info$model_dims)
    }
    if (isTRUE(return_details)) {
      return(list(tokens = tokens_final, readout_tokens = readout_tokens))
    }
    tokens_final
  }

  unlockBinding("neural_run_transformer", ns)
  assign("neural_run_transformer", fn, envir = ns)
  lockBinding("neural_run_transformer", ns)
  invisible(NULL)
}

run_average_case <- function(use_patch = FALSE) {
  if (isTRUE(use_patch)) {
    patch_full_attn_block2()
  }
  fixture <- generate_linear_average_case_fixture()
  res <- withr::with_seed(20260326L, {
    strategize(
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
      nSGD = 200L,
      nMonte_Qglm = 1000L,
      a_init_sd = 0.001,
      optim_type = "gd",
      neural_mcmc_control = list(
        subsample_method = "batch_vi",
        ModelDims = 64L,
        ModelDepth = 4L,
        residual_mode = "full_attn",
        qk_norm = FALSE,
        batch_size = 512L,
        optimizer = "adam",
        vi_guide = "auto_diagonal",
        svi_steps = 200L,
        svi_num_draws = 25L,
        uncertainty_scope = "output",
        eval_enabled = FALSE
      )
    )
  })
  pi_hat <- extract_average_case_pi_hat_local(res)
  Q_hat <- if (!is.null(res$Q_point_mEst) && all(is.finite(res$Q_point_mEst))) {
    as.numeric(res$Q_point_mEst)
  } else {
    as.numeric(res$Q_point)
  }
  mu_hat <- extract_average_case_neural_mu_hat_local(res, fixture$W)
  model_info <- get_neural_model_info_local(res)
  params <- model_info$params
  query_names <- grep("^pseudo_query", names(params), value = TRUE)
  query_norms <- setNames(vapply(query_names, function(name) param_l2(params[[name]]), numeric(1)), query_names)

  data.frame(
    variant = if (isTRUE(use_patch)) "block2_patch" else "current_full_attn",
    pi_rel_err = mean(abs(pi_hat - fixture$pi_star_true) / pmax(abs(fixture$pi_star_true), 1e-8)),
    Q_rel_err = abs(Q_hat - fixture$trueQ) / pmax(abs(fixture$trueQ), 1e-8),
    Q_hat = Q_hat,
    rmse_mu_true = sqrt(mean((mu_hat - fixture$mu_true) ^ 2)),
    cor_mu = suppressWarnings(stats::cor(mu_hat, fixture$mu_true)),
    query_norm_mean = mean(query_norms),
    query_norm_max = max(query_norms),
    stringsAsFactors = FALSE
  )
}

run_pairwise <- function(use_patch = FALSE) {
  if (isTRUE(use_patch)) {
    patch_full_attn_block2()
  }
  withr::with_envvar(c(
    STRATEGIZE_NEURAL_FAST_MCMC = "true",
    STRATEGIZE_NEURAL_EVAL_FOLDS = "2",
    STRATEGIZE_NEURAL_EVAL_SEED = "123"
  ), {
    data <- generate_pairwise_performance_test_data(
      n_pairs = 400L,
      n_factors = 3,
      n_levels = 2,
      seed = 20260327
    )
    data <- add_adversarial_structure(data, seed = 20260328)

    params <- default_strategize_params(fast = TRUE)
    params$outcome_model_type <- "neural"
    base_neural_control <- params$neural_mcmc_control
    if (is.null(base_neural_control)) {
      base_neural_control <- list()
    }
    params$neural_mcmc_control <- modifyList(
      base_neural_control,
      list(
        subsample_method = "batch_vi",
        batch_size = 128L,
        ModelDims = 16L,
        ModelDepth = 4L,
        residual_mode = "full_attn"
      )
    )

    p_list <- generate_test_p_list(data$W)
    res <- do.call(strategize, c(
      list(Y = data$Y, W = data$W, p_list = p_list),
      data[c(
        "pair_id",
        "respondent_id",
        "respondent_task_id",
        "profile_order",
        "competing_group_variable_respondent",
        "competing_group_variable_candidate",
        "competing_group_competition_variable_candidate"
      )],
      params
    ))

    info <- get_neural_model_info_local(res)
    metrics <- info$fit_metrics
    y_eval <- data$Y[data$profile_order == 1L]
    null_metrics <- compute_binary_null_metrics_local(y_eval)

    data.frame(
      variant = if (isTRUE(use_patch)) "block2_patch" else "current_full_attn",
      n_eval = safe_num(metrics$n_eval),
      auc = safe_num(metrics$auc),
      log_loss = safe_num(metrics$log_loss),
      accuracy = safe_num(metrics$accuracy),
      brier = safe_num(metrics$brier),
      null_log_loss = safe_num(null_metrics$log_loss),
      stringsAsFactors = FALSE
    )
  })
}

mode <- Sys.getenv("FULL_ATTN_PROBE_MODE", "both")

if (mode %in% c("both", "average")) {
  avg_results <- do.call(rbind, list(
    run_average_case(use_patch = FALSE),
    run_average_case(use_patch = TRUE)
  ))
  cat("=== average_case ===\n")
  print(avg_results)
}

if (mode %in% c("both", "pairwise")) {
  pair_results <- do.call(rbind, list(
    run_pairwise(use_patch = FALSE),
    run_pairwise(use_patch = TRUE)
  ))
  cat("=== pairwise ===\n")
  print(pair_results)
}
