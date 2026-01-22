suppressPackageStartupMessages({
  if (!requireNamespace("pkgload", quietly = TRUE)) {
    stop("pkgload is required to load the local strategize package")
  }
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("reticulate is required for JAX diagnostics")
  }
  library(pkgload)
  library(reticulate)
  library(withr)
})

pkgload::load_all("/Users/cjerzak/Documents/strategize-software/strategize", quiet = TRUE)
source("/Users/cjerzak/Documents/strategize-software/strategize/tests/testthat/helper-strategize.R")

# Ensure JAX is available in the configured conda env.
skip_if_no_jax()

withr::local_envvar(c(STRATEGIZE_NEURAL_FAST_MCMC = "true"))

# Set up the same data used in the failing test.
data <- generate_test_data(n = 40, seed = 123)
params <- default_strategize_params(fast = TRUE)
params$outcome_model_type <- "neural"
params$neural_mcmc_control <- list(
  subsample_method = "full",
  n_samples_warmup = 5L,
  n_samples_mcmc = 5L,
  n_chains = 1L,
  chain_method = "sequential",
  eval_enabled = FALSE
)

p_list <- generate_test_p_list(data$W)
res <- do.call(strategize, c(
  list(Y = data$Y, W = data$W, p_list = p_list),
  data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order")],
  params
))

model <- res$Y_models$my_model_ast_jnp
if (is.null(model)) {
  model <- res$Y_models$my_model_dag_jnp
}
if (!is.function(model)) {
  stop("Neural model function not available in strategize output")
}

model_env <- environment(model)
strenv <- get("strenv", envir = model_env)
likelihood <- get("likelihood", envir = model_env)
if (!likelihood %in% c("bernoulli", "categorical")) {
  stop("Prior predictive check supports bernoulli/categorical only")
}

pairwise_mode <- isTRUE(get("pairwise_mode", envir = model_env))
model_fn <- if (pairwise_mode) {
  get("BayesianPairTransformerModel", envir = model_env)
} else {
  get("BayesianSingleTransformerModel", envir = model_env)
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
  stop("Neural model info unavailable for prior predictive check")
}

model_dims <- as.integer(model_info$model_dims)
model_depth <- as.integer(model_info$model_depth)
n_heads <- as.integer(model_info$n_heads)
head_dim <- as.integer(model_info$head_dim)
cross_candidate_encoder <- isTRUE(model_info$cross_candidate_encoder)
n_resp_covariates <- if (!is.null(model_info$n_resp_covariates)) {
  as.integer(model_info$n_resp_covariates)
} else {
  0L
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

n_factors <- ncol(W_numeric)

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
  out <- tryCatch(reticulate::py_to_r(strenv$np$asarray(x)), error = function(e) NULL)
  if (is.null(out)) {
    out <- tryCatch(reticulate::py_to_r(strenv$np$array(x)), error = function(e) NULL)
  }
  if (is.null(out)) {
    out <- tryCatch(reticulate::py_to_r(strenv$jax$device_get(x)), error = function(e) NULL)
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

coerce_scalar <- function(x) {
  if (is.null(x)) {
    return(NA_real_)
  }
  out <- tryCatch(reticulate::py_to_r(strenv$np$asarray(x)), error = function(e) NULL)
  if (is.null(out)) {
    out <- tryCatch(reticulate::py_to_r(strenv$np$array(x)), error = function(e) NULL)
  }
  if (is.null(out)) {
    out <- tryCatch(reticulate::py_to_r(strenv$jax$device_get(x)), error = function(e) NULL)
  }
  if (is.null(out)) {
    out <- tryCatch(reticulate::py_to_r(x), error = function(e) NULL)
  }
  if (is.null(out)) {
    return(NA_real_)
  }
  if (is.list(out)) {
    out <- unlist(out, use.names = FALSE)
  }
  if (is.environment(out)) {
    return(NA_real_)
  }
  out <- as.numeric(out)
  if (length(out) == 0L || is.na(out[1])) {
    return(NA_real_)
  }
  out[1]
}

is_py_none <- function(x) {
  if (is.null(x)) {
    return(TRUE)
  }
  if (inherits(x, "python.builtin.NoneType")) {
    return(TRUE)
  }
  isTRUE(tryCatch(reticulate::py_is_null(x), error = function(e) FALSE))
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
  if (!is_py_none(params$E_feature_id)) {
    feature_tok <- strenv$jnp$reshape(params$E_feature_id, list(1L, D_local, model_dims))
    tokens <- tokens + feature_tok
  }
  party_tok <- strenv$jnp$take(params$E_party, party_idx, axis = 0L)
  party_tok <- strenv$jnp$reshape(party_tok, list(N_batch, 1L, model_dims))
  tokens <- strenv$jnp$concatenate(list(tokens, party_tok), axis = 1L)
  if (!is_py_none(params$E_rel) && !is.null(model_info$cand_party_to_resp_idx)) {
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
  if (!is_py_none(params$E_stage) && !is.null(stage_idx)) {
    stage_tok <- params$E_stage[resp_p, stage_idx]
    stage_tok <- strenv$jnp$reshape(stage_tok, list(N_batch, 1L, model_dims))
    token_list[[length(token_list) + 1L]] <- stage_tok
  }
  if (!is_py_none(params$E_resp_party)) {
    resp_tok <- strenv$jnp$take(params$E_resp_party, resp_p, axis = 0L)
    resp_tok <- strenv$jnp$reshape(resp_tok, list(N_batch, 1L, model_dims))
    token_list[[length(token_list) + 1L]] <- resp_tok
  }
  if (!is_py_none(params$E_matchup) && !is.null(matchup_idx)) {
    matchup_tok <- strenv$jnp$take(params$E_matchup, matchup_idx, axis = 0L)
    matchup_tok <- strenv$jnp$reshape(matchup_tok, list(N_batch, 1L, model_dims))
    token_list[[length(token_list) + 1L]] <- matchup_tok
  }
  if (!is_py_none(params$W_resp_x) && n_resp_covariates > 0L) {
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
  choice_vec <- if (!is_py_none(params$E_choice)) {
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

prior_predictive_probs_params <- function(params) {
  if (pairwise_mode) {
    stage_idx <- strenv$jnp$equal(party_left_jnp, party_right_jnp)
    stage_idx <- strenv$jnp$astype(stage_idx, strenv$jnp$int32)
    matchup_idx <- NULL
    if (!is_py_none(params$E_matchup)) {
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
    if (isTRUE(cross_candidate_encoder) && !is_py_none(params$M_cross) && !is_py_none(params$W_cross_out)) {
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

scale_params <- function(params, names, scale) {
  params_scaled <- params
  for (nm in names) {
    if (!is_py_none(params_scaled[[nm]])) {
      params_scaled[[nm]] <- strenv$jnp$multiply(params_scaled[[nm]], as.numeric(scale))
    }
  }
  params_scaled
}

# Parameter groups to probe.
embedding_names <- c(
  paste0("E_factor_", seq_len(n_factors)),
  "E_feature_id", "E_party", "E_rel", "E_resp_party", "E_stage",
  "E_matchup", "E_choice", "E_sep", "E_segment", "W_resp_x"
)
transformer_names <- unlist(lapply(seq_len(model_depth), function(l_) {
  c(
    paste0("W_q_l", l_), paste0("W_k_l", l_), paste0("W_v_l", l_),
    paste0("W_o_l", l_), paste0("W_ff1_l", l_), paste0("W_ff2_l", l_)
  )
}))
output_names <- c("W_out")
output_bias_names <- c("b_out")

# Draw priors and compute probability SDs.
set.seed(123)

n_draws <- 25L
prob_samples <- list(
  baseline = numeric(0),
  embed_x2 = numeric(0),
  transformer_x2 = numeric(0),
  output_x2 = numeric(0),
  bias_x2 = numeric(0)
)

scale_factor <- 2

prior_scales <- list(
  tau_factor = numeric(0),
  tau_context = numeric(0),
  tau_w_out = numeric(0),
  tau_w = numeric(0),
  tau_b = numeric(0)
)

for (i in seq_len(n_draws)) {
  tryCatch({
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

  # Capture prior scales.
  prior_scales$tau_factor <- c(prior_scales$tau_factor,
                               coerce_scalar(get_trace_value(trace, "tau_factor")))
  prior_scales$tau_context <- c(prior_scales$tau_context,
                                coerce_scalar(get_trace_value(trace, "tau_context")))
  prior_scales$tau_w_out <- c(prior_scales$tau_w_out, coerce_scalar(get_trace_value(trace, "tau_w_out")))
  prior_scales$tau_b <- c(prior_scales$tau_b, coerce_scalar(get_trace_value(trace, "tau_b")))
  tau_w_vals <- sapply(seq_len(model_depth), function(l_) {
    coerce_scalar(get_trace_value(trace, paste0("tau_w_", l_)))
  })
  prior_scales$tau_w <- c(prior_scales$tau_w, mean(tau_w_vals, na.rm = TRUE))

  params_draw <- build_params_from_trace(trace)

  required_names <- c(
    paste0("E_factor_", seq_len(n_factors)),
    "E_party", "RMS_final", "W_out", "b_out"
  )
  required_names <- c(
    required_names,
    unlist(lapply(seq_len(model_depth), function(l_) {
      c(
        paste0("RMS_attn_l", l_), paste0("RMS_ff_l", l_),
        paste0("W_q_l", l_), paste0("W_k_l", l_), paste0("W_v_l", l_),
        paste0("W_o_l", l_), paste0("W_ff1_l", l_), paste0("W_ff2_l", l_)
      )
    }))
  )
  missing <- required_names[vapply(required_names, function(nm) is_py_none(params_draw[[nm]]), logical(1))]
  if (length(missing) > 0L) {
    stop("Missing required params in prior draw: ", paste(missing, collapse = ", "))
  }

  prob_base <- tryCatch(
    prior_predictive_probs_params(params_draw),
    error = function(e) {
      cat("Error during baseline prior predictive draw:", conditionMessage(e), "\n")
      cat("Python error (if available):\n")
      print(reticulate::py_last_error())
      none_names <- names(params_draw)[vapply(params_draw, is_py_none, logical(1))]
      if (length(none_names) > 0L) {
        cat("Params with None values:\n")
        cat(paste(none_names, collapse = ", "), "\n")
      }
      stop(e)
    }
  )
  prob_samples$baseline <- c(prob_samples$baseline, coerce_prob_numeric(prob_base))

  params_embed <- scale_params(params_draw, embedding_names, scale_factor)
  prob_samples$embed_x2 <- c(prob_samples$embed_x2,
                             coerce_prob_numeric(prior_predictive_probs_params(params_embed)))

  params_trans <- scale_params(params_draw, transformer_names, scale_factor)
  prob_samples$transformer_x2 <- c(prob_samples$transformer_x2,
                                  coerce_prob_numeric(prior_predictive_probs_params(params_trans)))

  params_out <- scale_params(params_draw, output_names, scale_factor)
  prob_samples$output_x2 <- c(prob_samples$output_x2,
                              coerce_prob_numeric(prior_predictive_probs_params(params_out)))

  params_bias <- scale_params(params_draw, output_bias_names, scale_factor)
  prob_samples$bias_x2 <- c(prob_samples$bias_x2,
                            coerce_prob_numeric(prior_predictive_probs_params(params_bias)))
  }, error = function(e) {
    cat("Error in prior predictive draw", i, ":", conditionMessage(e), "\n")
    cat("Python error (if available):\n")
    print(reticulate::py_last_error())
    if (exists("params_draw", inherits = FALSE)) {
      none_names <- names(params_draw)[vapply(params_draw, is_py_none, logical(1))]
      if (length(none_names) > 0L) {
        cat("Params with None values:\n")
        cat(paste(none_names, collapse = ", "), "\n")
      }
    }
    stop(e)
  })
}

sd_table <- sapply(prob_samples, function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0L) {
    return(NA_real_)
  }
  stats::sd(x)
})

scale_summary <- sapply(prior_scales, function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0L) {
    return(c(mean = NA_real_, sd = NA_real_))
  }
  c(mean = mean(x), sd = stats::sd(x))
})

cat("Prior predictive SD diagnostics (n_draws =", n_draws, ")\n")
print(round(sd_table, 4))
cat("\nScale hyperparameter summaries (mean, sd across draws)\n")
print(round(scale_summary, 4))
cat("\nNotes:\n")
cat("- baseline: raw prior predictive SD\n")
cat("- *_x2: SD after scaling that parameter group by", scale_factor, "\n")
cat("- In pairwise mode, output bias cancels in u_left - u_right, so bias scaling may have no effect.\n")
