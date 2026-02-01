# =============================================================================
# Neural Outcome Model Tests
# =============================================================================

get_neural_fit <- local({
  cache <- NULL
  function() {
    if (!is.null(cache)) {
      return(cache)
    }

    skip_on_cran()
    skip_if_no_jax()

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

    cache <<- list(res = res, data = data, p_list = p_list)
    cache
  }
})

get_neural_fit_attn <- local({
  cache <- NULL
  function() {
    if (!is.null(cache)) {
      return(cache)
    }

    skip_on_cran()
    skip_if_no_jax()

    withr::local_envvar(c(
      STRATEGIZE_NEURAL_FAST_MCMC = "true",
      STRATEGIZE_NEURAL_EVAL_FOLDS = "2",
      STRATEGIZE_NEURAL_EVAL_SEED = "123"
    ))

    data <- generate_test_data(n = 30, seed = 321)
    params <- default_strategize_params(fast = TRUE)
    params$outcome_model_type <- "neural"
    params$neural_mcmc_control <- modifyList(
      params$neural_mcmc_control,
      list(cross_candidate_encoder = "attn", ModelDims = 16L, ModelDepth = 1L)
    )

    p_list <- generate_test_p_list(data$W)

    res <- do.call(strategize, c(
      list(Y = data$Y, W = data$W, p_list = p_list),
      data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order")],
      params
    ))

    cache <<- list(res = res, data = data, p_list = p_list)
    cache
  }
})

test_that("strategize runs neural outcome model (non-adversarial)", {
  fit <- get_neural_fit()
  res <- fit$res
  data <- fit$data
  p_list <- fit$p_list

  expect_valid_strategize_output(res, n_factors = ncol(data$W))

  model <- res$Y_models$my_model_ast_jnp
  if (is.null(model)) {
    model <- res$Y_models$my_model_dag_jnp
  }
  expect_true(is.function(model))

  W_numeric <- as.matrix(sapply(seq_len(ncol(data$W)), function(d_) {
    match(data$W[, d_], names(p_list[[d_]]))
  }))
  idx_left <- which(data$profile_order == 1L)
  idx_right <- which(data$profile_order == 2L)
  X_left <- W_numeric[idx_left, , drop = FALSE]
  X_right <- W_numeric[idx_right, , drop = FALSE]
  p_lr <- as.numeric(model(X_left_new = X_left, X_right_new = X_right))
  p_rl <- as.numeric(model(X_left_new = X_right, X_right_new = X_left))
  expect_equal(p_lr + p_rl, rep(1, length(p_lr)), tolerance = 1e-4)
})

test_that("neural attn predictor remains antisymmetric", {
  fit <- get_neural_fit_attn()
  res <- fit$res
  data <- fit$data
  p_list <- fit$p_list

  expect_valid_strategize_output(res, n_factors = ncol(data$W))

  model <- res$Y_models$my_model_ast_jnp
  if (is.null(model)) {
    model <- res$Y_models$my_model_dag_jnp
  }
  expect_true(is.function(model))

  W_numeric <- as.matrix(sapply(seq_len(ncol(data$W)), function(d_) {
    match(data$W[, d_], names(p_list[[d_]]))
  }))
  idx_left <- which(data$profile_order == 1L)
  idx_right <- which(data$profile_order == 2L)
  X_left <- W_numeric[idx_left, , drop = FALSE]
  X_right <- W_numeric[idx_right, , drop = FALSE]
  p_lr <- as.numeric(model(X_left_new = X_left, X_right_new = X_right))
  p_rl <- as.numeric(model(X_left_new = X_right, X_right_new = X_left))
  expect_equal(p_lr + p_rl, rep(1, length(p_lr)), tolerance = 1e-4)
})

test_that("neural outcome bundles save and reload cleanly", {
  fit <- get_neural_fit()
  res <- fit$res
  data <- fit$data
  p_list <- fit$p_list

  model_info <- res$neural_model_info$ast
  if (is.null(model_info)) model_info <- res$neural_model_info$dag
  if (is.null(model_info)) model_info <- res$neural_model_info$ast0
  if (is.null(model_info)) model_info <- res$neural_model_info$dag0
  expect_true(!is.null(model_info))

  theta_mean <- tryCatch(
    as.numeric(reticulate::py_to_r(res$est_coefficients_jnp)),
    error = function(e) {
      tryCatch(
        as.numeric(res$est_coefficients_jnp),
        error = function(e2) {
          as.numeric(reticulate::py_to_r(strategize:::strenv$np$array(res$est_coefficients_jnp)))
        }
      )
    }
  )
  expect_true(is.numeric(theta_mean))

  vcov_vec <- res$vcov_outcome_model
  theta_var <- if (!is.null(vcov_vec) && length(vcov_vec) > 1L) {
    as.numeric(vcov_vec[-1])
  } else {
    NULL
  }

  tmp <- tempfile(fileext = ".rds")
  save_neural_outcome_bundle(
    file = tmp,
    theta_mean = theta_mean,
    theta_var = theta_var,
    neural_model_info = model_info,
    p_list = p_list,
    mode = "pairwise",
    overwrite = TRUE
  )

  bundle <- readRDS(tmp)
  has_py_object <- function(x) {
    if (reticulate::is_py_object(x)) {
      return(TRUE)
    }
    if (is.list(x)) {
      return(any(vapply(x, has_py_object, logical(1))))
    }
    FALSE
  }
  expect_false(has_py_object(bundle))
  expect_false(has_py_object(bundle$fit$neural_model_info))

  fit_loaded <- load_neural_outcome_bundle(tmp, preload_params = FALSE)
  expect_true(inherits(fit_loaded, "strategic_predictor"))
  expect_true(is.null(fit_loaded$fit$params))

  idx_left <- which(data$profile_order == 1L)
  idx_right <- which(data$profile_order == 2L)
  W_left <- data$W[idx_left, , drop = FALSE]
  W_right <- data$W[idx_right, , drop = FALSE]
  p <- predict_pair(fit_loaded, W_left = W_left, W_right = W_right)
  expect_true(is.numeric(p))
  expect_true(all(is.finite(p)))
  expect_true(all(p >= 0 & p <= 1))
})

test_that("neural prior predictive probabilities are not overly concentrated", {
  fit <- get_neural_fit()
  res <- fit$res
  data <- fit$data
  p_list <- fit$p_list

  model <- res$Y_models$my_model_ast_jnp
  if (is.null(model)) {
    model <- res$Y_models$my_model_dag_jnp
  }
  expect_true(is.function(model))

  model_env <- environment(model)
  strenv <- get("strenv", envir = model_env)
  likelihood <- get("likelihood", envir = model_env)
  if (!likelihood %in% c("bernoulli", "categorical")) {
    skip("Prior predictive check supports bernoulli/categorical only.")
  }
  pairwise_mode <- isTRUE(get("pairwise_mode", envir = model_env))
  model_fn <- if (pairwise_mode) {
    get("BayesianPairTransformerModel", envir = model_env)
  } else {
    get("BayesianSingleTransformerModel", envir = model_env)
  }

  W_numeric <- as.matrix(sapply(seq_len(ncol(data$W)), function(d_) {
    match(data$W[, d_], names(p_list[[d_]]))
  }))
  to_index_matrix <- if (exists("to_index_matrix", envir = model_env, inherits = TRUE)) {
    get("to_index_matrix", envir = model_env)
  } else {
    function(x_mat) {
      x_mat <- as.matrix(x_mat)
      if (anyNA(x_mat)) {
        x_mat[is.na(x_mat)] <- 1L
      }
      x_int <- matrix(as.integer(x_mat) - 1L, nrow = nrow(x_mat), ncol = ncol(x_mat))
      x_int[x_int < 0L] <- 0L
      x_int
    }
  }

  if (pairwise_mode) {
    idx_left <- which(data$profile_order == 1L)
    idx_right <- which(data$profile_order == 2L)
    X_left <- W_numeric[idx_left, , drop = FALSE]
    X_right <- W_numeric[idx_right, , drop = FALSE]
    n_obs <- nrow(X_left)
    X_left_jnp <- strenv$jnp$array(to_index_matrix(X_left))$astype(strenv$jnp$int32)
    X_right_jnp <- strenv$jnp$array(to_index_matrix(X_right))$astype(strenv$jnp$int32)
    party_left_jnp <- strenv$jnp$zeros(list(n_obs), dtype = strenv$jnp$int32)
    party_right_jnp <- strenv$jnp$zeros(list(n_obs), dtype = strenv$jnp$int32)
    resp_party_jnp <- strenv$jnp$zeros(list(n_obs), dtype = strenv$jnp$int32)
    resp_cov_jnp <- strenv$jnp$zeros(list(n_obs, 0L), dtype = strenv$jnp$float32)
  } else {
    n_obs <- nrow(W_numeric)
    X_single_jnp <- strenv$jnp$array(to_index_matrix(W_numeric))$astype(strenv$jnp$int32)
    party_single_jnp <- strenv$jnp$zeros(list(n_obs), dtype = strenv$jnp$int32)
    resp_party_jnp <- strenv$jnp$zeros(list(n_obs), dtype = strenv$jnp$int32)
    resp_cov_jnp <- strenv$jnp$zeros(list(n_obs, 0L), dtype = strenv$jnp$float32)
  }

  coerce_prob_numeric <- function(x) {
    if (is.null(x)) {
      return(numeric(0))
    }
    out <- tryCatch(
      reticulate::py_to_r(strenv$np$asarray(x)),
      error = function(e) NULL
    )
    if (is.null(out)) {
      out <- tryCatch(
        reticulate::py_to_r(strenv$np$array(x)),
        error = function(e) NULL
      )
    }
    if (is.null(out)) {
      out <- tryCatch(
        reticulate::py_to_r(strenv$jax$device_get(x)),
        error = function(e) NULL
      )
    }
    if (is.null(out)) {
      out <- tryCatch(reticulate::py_to_r(x), error = function(e) NULL)
    }
    if (is.null(out)) {
      return(numeric(0))
    }
    if (is.list(out)) {
      out <- unlist(out, use.names = FALSE)
    }
    if (!is.numeric(out)) {
      return(numeric(0))
    }
    as.numeric(out)
  }

  model_info <- NULL
  if (!is.null(res$neural_model_info)) {
    if (!is.null(res$Y_models$my_model_ast_jnp)) {
      model_info <- res$neural_model_info$ast
    } else if (!is.null(res$Y_models$my_model_dag_jnp)) {
      model_info <- res$neural_model_info$dag
    }
  }
  if (is.null(model_info)) {
    skip("Neural model info unavailable for prior predictive check.")
  }
  model_dims <- as.integer(model_info$model_dims)
  model_depth <- as.integer(model_info$model_depth)
  n_heads <- as.integer(model_info$n_heads)
  head_dim <- as.integer(model_info$head_dim)
  cross_candidate_encoder <- isTRUE(model_info$cross_candidate_encoder)
  n_factors <- ncol(W_numeric)
  n_resp_covariates <- if (!is.null(model_info$n_resp_covariates)) {
    as.integer(model_info$n_resp_covariates)
  } else {
    0L
  }

  get_trace_value <- function(trace, name) {
    site <- tryCatch(trace[[name]], error = function(e) NULL)
    if (is.null(site)) {
      return(NULL)
    }
    val <- tryCatch(site$value, error = function(e) NULL)
    if (is.null(val)) {
      val <- site
    }
    val
  }

  build_params_from_trace <- function(trace) {
    params <- list()
    for (d_ in seq_len(n_factors)) {
      name <- paste0("E_factor_", d_)
      val <- get_trace_value(trace, name)
      if (is.null(val)) {
        raw <- get_trace_value(trace, paste0(name, "_raw"))
        if (!is.null(raw)) {
          n_real <- if (!is.null(model_info$factor_levels)) {
            as.integer(model_info$factor_levels[[d_]])
          } else {
            NA_integer_
          }
          n_raw <- tryCatch(
            as.integer(reticulate::py_to_r(raw$shape[[1]])),
            error = function(e) NA_integer_
          )
          if (!is.na(n_real) && !is.na(n_raw) && n_raw > n_real) {
            real_idx <- strenv$jnp$arange(as.integer(n_real))
            real_rows <- strenv$jnp$take(raw, real_idx, axis = 0L)
            real_mean <- strenv$jnp$mean(real_rows, axis = 0L, keepdims = TRUE)
            real_centered <- real_rows - real_mean
            missing_row <- strenv$jnp$take(raw, as.integer(n_real), axis = 0L)
            missing_row <- strenv$jnp$reshape(missing_row, list(1L, model_dims))
            val <- strenv$jnp$concatenate(list(real_centered, missing_row), axis = 0L)
          } else {
            val <- raw - strenv$jnp$mean(raw, axis = 0L, keepdims = TRUE)
          }
        }
      }
      params[[name]] <- val
    }
    base_names <- c(
      "E_feature_id", "E_party", "E_rel", "E_resp_party", "E_stage", "E_matchup", "E_choice",
      "E_sep", "E_segment",
      "W_resp_x", "RMS_final", "W_out", "b_out", "M_cross", "W_cross_out"
    )
    for (nm in base_names) {
      params[[nm]] <- get_trace_value(trace, nm)
    }
    for (l_ in seq_len(model_depth)) {
      params[[paste0("RMS_attn_l", l_)]] <- get_trace_value(trace, paste0("RMS_attn_l", l_))
      params[[paste0("RMS_ff_l", l_)]] <- get_trace_value(trace, paste0("RMS_ff_l", l_))
      params[[paste0("W_q_l", l_)]] <- get_trace_value(trace, paste0("W_q_l", l_))
      params[[paste0("W_k_l", l_)]] <- get_trace_value(trace, paste0("W_k_l", l_))
      params[[paste0("W_v_l", l_)]] <- get_trace_value(trace, paste0("W_v_l", l_))
      params[[paste0("W_o_l", l_)]] <- get_trace_value(trace, paste0("W_o_l", l_))
      params[[paste0("W_ff1_l", l_)]] <- get_trace_value(trace, paste0("W_ff1_l", l_))
      params[[paste0("W_ff2_l", l_)]] <- get_trace_value(trace, paste0("W_ff2_l", l_))
    }
    params
  }

  rms_norm <- function(x, g, eps = 1e-6) {
    mean_sq <- strenv$jnp$mean(x * x, axis = -1L, keepdims = TRUE)
    inv_rms <- strenv$jnp$reciprocal(strenv$jnp$sqrt(mean_sq + eps))
    g_use <- strenv$jnp$reshape(g, list(1L, 1L, model_dims))
    x * inv_rms * g_use
  }

  run_transformer <- function(tokens, params) {
    for (l_ in seq_len(model_depth)) {
      Wq <- params[[paste0("W_q_l", l_)]]
      Wk <- params[[paste0("W_k_l", l_)]]
      Wv <- params[[paste0("W_v_l", l_)]]
      Wo <- params[[paste0("W_o_l", l_)]]
      Wff1 <- params[[paste0("W_ff1_l", l_)]]
      Wff2 <- params[[paste0("W_ff2_l", l_)]]
      RMS_attn <- params[[paste0("RMS_attn_l", l_)]]
      RMS_ff <- params[[paste0("RMS_ff_l", l_)]]

      tokens_norm <- rms_norm(tokens, RMS_attn)
      Q <- strenv$jnp$einsum("ntm,mk->ntk", tokens_norm, Wq)
      K <- strenv$jnp$einsum("ntm,mk->ntk", tokens_norm, Wk)
      V <- strenv$jnp$einsum("ntm,mk->ntk", tokens_norm, Wv)

      Qh <- strenv$jnp$reshape(Q, list(Q$shape[[1]], Q$shape[[2]], n_heads, head_dim))
      Kh <- strenv$jnp$reshape(K, list(K$shape[[1]], K$shape[[2]], n_heads, head_dim))
      Vh <- strenv$jnp$reshape(V, list(V$shape[[1]], V$shape[[2]], n_heads, head_dim))
      scale_ <- strenv$jnp$sqrt(strenv$jnp$array(as.numeric(head_dim)))
      scores <- strenv$jnp$einsum("nqhd,nkhd->nhqk", Qh, Kh) / scale_
      attn <- strenv$jax$nn$softmax(scores, axis = -1L)
      context_h <- strenv$jnp$einsum("nhqk,nkhd->nqhd", attn, Vh)
      context <- strenv$jnp$reshape(context_h, list(context_h$shape[[1]], context_h$shape[[2]], model_dims))
      attn_out <- strenv$jnp$einsum("ntm,mk->ntk", context, Wo)

      h1 <- tokens + attn_out
      h1_norm <- rms_norm(h1, RMS_ff)
      ff_pre <- strenv$jnp$einsum("ntm,mf->ntf", h1_norm, Wff1)
      ff_act <- strenv$jax$nn$swish(ff_pre)
      ff_out <- strenv$jnp$einsum("ntf,fm->ntm", ff_act, Wff2)
      tokens <- h1 + ff_out
    }
    rms_norm(tokens, params$RMS_final)
  }

  embed_candidate <- function(X_idx, party_idx, resp_p, params) {
    N_batch <- as.integer(X_idx$shape[[1]])
    D_local <- as.integer(X_idx$shape[[2]])
    token_list <- vector("list", D_local)
    for (d_ in seq_len(D_local)) {
      E_d <- params[[paste0("E_factor_", d_)]]
      idx_d <- strenv$jnp$take(X_idx, as.integer(d_ - 1L), axis = 1L)
      token_list[[d_]] <- strenv$jnp$take(E_d, idx_d, axis = 0L)
    }
    tokens <- strenv$jnp$stack(token_list, axis = 1L)
    if (!is.null(params$E_feature_id)) {
      feature_tok <- strenv$jnp$reshape(params$E_feature_id, list(1L, D_local, model_dims))
      tokens <- tokens + feature_tok
    }
    party_tok <- strenv$jnp$take(params$E_party, party_idx, axis = 0L)
    party_tok <- strenv$jnp$reshape(party_tok, list(N_batch, 1L, model_dims))
    tokens <- strenv$jnp$concatenate(list(tokens, party_tok), axis = 1L)
    if (!is.null(params$E_rel) && !is.null(model_info$cand_party_to_resp_idx)) {
      cand_map <- strenv$jnp$atleast_1d(model_info$cand_party_to_resp_idx)
      cand_resp_idx <- strenv$jnp$take(cand_map, party_idx, axis = 0L)
      is_known <- cand_resp_idx >= 0L
      is_match <- strenv$jnp$equal(cand_resp_idx, resp_p)
      rel_idx <- strenv$jnp$where(is_match, as.integer(0L),
                                  strenv$jnp$where(is_known, as.integer(1L), as.integer(2L)))
      rel_idx <- strenv$jnp$astype(rel_idx, strenv$jnp$int32)
      rel_tok <- strenv$jnp$take(params$E_rel, rel_idx, axis = 0L)
      rel_tok <- strenv$jnp$reshape(rel_tok, list(N_batch, 1L, model_dims))
      tokens <- strenv$jnp$concatenate(list(tokens, rel_tok), axis = 1L)
    }
    tokens
  }

  build_context_tokens <- function(stage_idx, resp_p, resp_c, matchup_idx, params) {
    N_batch <- as.integer(resp_p$shape[[1]])
    token_list <- list()
    if (!is.null(params$E_stage) && !is.null(stage_idx)) {
      stage_tok <- params$E_stage[resp_p, stage_idx]
      stage_tok <- strenv$jnp$reshape(stage_tok, list(N_batch, 1L, model_dims))
      token_list[[length(token_list) + 1L]] <- stage_tok
    }
    if (!is.null(params$E_resp_party)) {
      resp_tok <- strenv$jnp$take(params$E_resp_party, resp_p, axis = 0L)
      resp_tok <- strenv$jnp$reshape(resp_tok, list(N_batch, 1L, model_dims))
      token_list[[length(token_list) + 1L]] <- resp_tok
    }
    if (!is.null(params$E_matchup) && !is.null(matchup_idx)) {
      matchup_tok <- strenv$jnp$take(params$E_matchup, matchup_idx, axis = 0L)
      matchup_tok <- strenv$jnp$reshape(matchup_tok, list(N_batch, 1L, model_dims))
      token_list[[length(token_list) + 1L]] <- matchup_tok
    }
    if (!is.null(params$W_resp_x) && n_resp_covariates > 0L) {
      resp_cov_tok <- strenv$jnp$einsum("nc,cm->nm", resp_c, params$W_resp_x)
      resp_cov_tok <- strenv$jnp$reshape(resp_cov_tok, list(N_batch, 1L, model_dims))
      token_list[[length(token_list) + 1L]] <- resp_cov_tok
    }
    if (length(token_list) == 0L) {
      return(NULL)
    }
    strenv$jnp$concatenate(token_list, axis = 1L)
  }

  encode_candidate <- function(X_idx, party_idx, resp_p, resp_c, stage_idx, matchup_idx, params) {
    N_batch <- as.integer(X_idx$shape[[1]])
    choice_vec <- if (!is.null(params$E_choice)) {
      params$E_choice
    } else {
      strenv$jnp$zeros(list(model_dims), dtype = strenv$dtj)
    }
    choice_tok <- strenv$jnp$reshape(choice_vec, list(1L, 1L, model_dims))
    choice_tok <- choice_tok * strenv$jnp$ones(list(N_batch, 1L, 1L))
    ctx_tokens <- build_context_tokens(stage_idx, resp_p, resp_c, matchup_idx, params)
    cand_tokens <- embed_candidate(X_idx, party_idx, resp_p, params)
    if (!is.null(ctx_tokens)) {
      tokens <- strenv$jnp$concatenate(list(choice_tok, ctx_tokens, cand_tokens), axis = 1L)
    } else {
      tokens <- strenv$jnp$concatenate(list(choice_tok, cand_tokens), axis = 1L)
    }
    tokens <- run_transformer(tokens, params)
    choice_out <- strenv$jnp$take(tokens, strenv$jnp$arange(1L), axis = 1L)
    strenv$jnp$squeeze(choice_out, axis = 1L)
  }

  prior_predictive_probs <- function(trace) {
    params <- build_params_from_trace(trace)
    if (pairwise_mode) {
      stage_idx <- strenv$jnp$equal(party_left_jnp, party_right_jnp)
      stage_idx <- strenv$jnp$astype(stage_idx, strenv$jnp$int32)
      matchup_idx <- NULL
      if (!is.null(params$E_matchup)) {
        n_party_levels <- if (!is.null(model_info$n_party_levels)) {
          as.integer(model_info$n_party_levels)
        } else if (!is.null(model_info$party_levels)) {
          length(model_info$party_levels)
        } else {
          1L
        }
        p_min <- strenv$jnp$minimum(party_left_jnp, party_right_jnp)
        p_max <- strenv$jnp$maximum(party_left_jnp, party_right_jnp)
        half_term <- strenv$jnp$floor_divide(p_min * (p_min - 1L), as.integer(2L))
        matchup_idx <- strenv$jnp$astype(
          p_min * as.integer(n_party_levels) - half_term + (p_max - p_min),
          strenv$jnp$int32
        )
      }
      phi_left <- encode_candidate(X_left_jnp, party_left_jnp, resp_party_jnp, resp_cov_jnp,
                                   stage_idx, matchup_idx, params)
      phi_right <- encode_candidate(X_right_jnp, party_right_jnp, resp_party_jnp, resp_cov_jnp,
                                    stage_idx, matchup_idx, params)
      u_left <- strenv$jnp$einsum("nm,mo->no", phi_left, params$W_out) + params$b_out
      u_right <- strenv$jnp$einsum("nm,mo->no", phi_right, params$W_out) + params$b_out
      logits <- u_left - u_right
      if (isTRUE(cross_candidate_encoder) && !is.null(params$M_cross) && !is.null(params$W_cross_out)) {
        cross_term <- strenv$jnp$einsum("nm,mp,np->n", phi_left, params$M_cross, phi_right)
        cross_term <- strenv$jnp$reshape(cross_term, list(-1L, 1L))
        cross_out <- strenv$jnp$reshape(params$W_cross_out, list(1L, -1L))
        logits <- logits + cross_term * cross_out
      }
    } else {
      phi_single <- encode_candidate(X_single_jnp, party_single_jnp, resp_party_jnp, resp_cov_jnp,
                                     stage_idx = NULL, matchup_idx = NULL, params = params)
      logits <- strenv$jnp$einsum("nm,mo->no", phi_single, params$W_out) + params$b_out
    }
    if (likelihood == "bernoulli") {
      logits_vec <- strenv$jnp$squeeze(logits, axis = 1L)
      return(strenv$jax$nn$sigmoid(logits_vec))
    }
    strenv$jax$nn$softmax(logits, axis = -1L)
  }

  n_draws <- 25L
  prob_samples <- numeric(0)
  for (i in seq_len(n_draws)) {
    rng_key <- strenv$jax$random$PRNGKey(as.integer(1000L + i))
    tracer <- strenv$numpyro$handlers$trace(
      strenv$numpyro$handlers$seed(model_fn, rng_key)
    )
    trace <- if (pairwise_mode) {
      tracer$get_trace(
        X_left = X_left_jnp,
        X_right = X_right_jnp,
        party_left = party_left_jnp,
        party_right = party_right_jnp,
        resp_party = resp_party_jnp,
        resp_cov = resp_cov_jnp,
        Y_obs = NULL
      )
    } else {
      tracer$get_trace(
        X = X_single_jnp,
        party = party_single_jnp,
        resp_party = resp_party_jnp,
        resp_cov = resp_cov_jnp,
        Y_obs = NULL
      )
    }
    prob <- prior_predictive_probs(trace)
    prob_samples <- c(
      prob_samples,
      coerce_prob_numeric(prob)
    )
  }

  prob_samples <- prob_samples[is.finite(prob_samples)]
  expect_true(length(prob_samples) > 0L)
  sd_prob <- stats::sd(prob_samples)
  expect_true(is.finite(sd_prob))
  expect_true(
    sd_prob >= 0.10,
    info = sprintf("Prior predictive SD %.3f below 0.10", sd_prob)
  )
})

test_that("neural outcome model exports cross-fitted OOS fit metrics", {
  fit <- get_neural_fit()
  res <- fit$res

  info <- NULL
  if (!is.null(res$neural_model_info)) {
    info <- res$neural_model_info$ast
    if (is.null(info)) {
      info <- res$neural_model_info$dag
    }
  }
  if (is.null(info)) {
    skip("Neural model info unavailable for fit-metrics check.")
  }

  metrics <- info$fit_metrics
  expect_type(metrics, "list")
  expect_true(is.character(metrics$eval_note))
  expect_match(metrics$eval_note, "^oos_\\d+fold$")
  expect_equal(metrics$n_folds, 2L)
  expect_equal(metrics$seed, 123L)
  expect_true(is.list(metrics$by_fold))
  expect_true(length(metrics$by_fold) >= 2L)

  if (identical(metrics$likelihood, "bernoulli")) {
    expect_true(is.finite(metrics$log_loss))
    expect_gte(metrics$log_loss, 0)
  } else if (identical(metrics$likelihood, "categorical")) {
    expect_true(is.finite(metrics$log_loss))
    expect_gte(metrics$log_loss, 0)
  } else if (identical(metrics$likelihood, "normal")) {
    expect_true(is.finite(metrics$rmse))
    expect_gte(metrics$rmse, 0)
  }
})
