generate_ModelOutcome_neural <- function(){
  message("Defining MCMC parameters in generate_ModelOutcome_neural...")
  mcmc_control <- list(
    backend = "numpyro",
    n_samples_warmup = 500L,
    n_samples_mcmc   = 1000L,
    batch_size = 512L,
    chain_method = "parallel",
    subsample_method = "full",
    n_thin_by = 1L,
    n_chains = 2L,
    svi_steps = 1000L,
    svi_lr = 0.01,
    svi_num_particles = 1L,
    svi_num_draws = 200L,
    vi_guide = "auto_diagonal"
  )
  UsedRegularization <- FALSE
  uncertainty_scope <- "all"
  mcmc_overrides <- NULL
  eval_control <- list(enabled = TRUE, max_n = NULL, seed = 123L)
  if (exists("neural_mcmc_control", inherits = TRUE) &&
      !is.null(neural_mcmc_control)) {
    if (!is.list(neural_mcmc_control)) {
      stop("'neural_mcmc_control' must be a list.", call. = FALSE)
    }
    if (!is.null(neural_mcmc_control$uncertainty_scope)) {
      uncertainty_scope <- tolower(as.character(neural_mcmc_control$uncertainty_scope))
    }
    if (!is.null(neural_mcmc_control$eval_enabled)) {
      eval_control$enabled <- isTRUE(neural_mcmc_control$eval_enabled)
    }
    if (!is.null(neural_mcmc_control$eval_max_n)) {
      eval_control$max_n <- as.integer(neural_mcmc_control$eval_max_n)
    }
    if (!is.null(neural_mcmc_control$eval_seed)) {
      eval_control$seed <- as.integer(neural_mcmc_control$eval_seed)
    }
    mcmc_overrides <- neural_mcmc_control
    mcmc_overrides$uncertainty_scope <- NULL
    mcmc_overrides$n_bayesian_models <- NULL
  }
  fast_mcmc_flag <- tolower(Sys.getenv("STRATEGIZE_NEURAL_FAST_MCMC")) %in%
    c("1", "true", "yes")
  if (isTRUE(fast_mcmc_flag)) {
    mcmc_control$n_samples_warmup <- 50L
    mcmc_control$n_samples_mcmc <- 50L
    mcmc_control$batch_size <- 128L
    mcmc_control$n_chains <- 1L
    mcmc_control$chain_method <- "sequential"
    mcmc_control$svi_steps <- 200L
    mcmc_control$svi_num_draws <- 100L
  }
  if (!is.null(mcmc_overrides) && length(mcmc_overrides) > 0) {
    mcmc_control <- modifyList(mcmc_control, mcmc_overrides)
  }
  skip_eval_flag <- tolower(Sys.getenv("STRATEGIZE_NEURAL_SKIP_EVAL")) %in%
    c("1", "true", "yes")
  if (isTRUE(skip_eval_flag)) {
    eval_control$enabled <- FALSE
  }
  eval_max_env <- suppressWarnings(as.integer(Sys.getenv("STRATEGIZE_NEURAL_EVAL_MAX")))
  if (!is.na(eval_max_env) && eval_max_env > 0L) {
    eval_control$max_n <- eval_max_env
  }
  if (!uncertainty_scope %in% c("all", "output")) {
    stop("'neural_mcmc_control$uncertainty_scope' must be 'all' or 'output'.",
         call. = FALSE)
  }
  subsample_method <- if (!is.null(mcmc_control$subsample_method)) {
    tolower(as.character(mcmc_control$subsample_method))
  } else {
    "full"
  }
  if (length(subsample_method) != 1L || is.na(subsample_method) || !nzchar(subsample_method)) {
    subsample_method <- "full"
  }
  mcmc_control$subsample_method <- subsample_method

  # Hyperparameters
  ModelDims  <- ai(128L)
  ModelDepth <- ai(2L)
  WideMultiplicationFactor <- 3.75
  MD_int <- ai(ModelDims)
  cand_heads <- (1:MD_int)[(MD_int %% (1:MD_int)) == 0L]
  TransformerHeads <- ai(cand_heads[which.min(abs(cand_heads - 8L))])
  head_dim <- ai(ai(MD_int / TransformerHeads))
  FFDim <- ai(ai(round(MD_int * WideMultiplicationFactor)))

  # Pairwise mode for forced-choice
  pairwise_mode <- isTRUE(diff) && !is.null(pair_id_) && length(pair_id_) > 0

  # Main-info structure for downstream compatibility
  for(nrp in 1:2){
    main_info <- do.call(rbind, sapply(1:length(factor_levels), function(d_){
      list(data.frame(
        "d" = d_,
        "l" = 1:max(1, factor_levels[d_] - ifelse(nrp == 1, yes = 1, no = holdout_indicator))
      ))
    }))
    main_info <- cbind(main_info, "d_index" = 1:nrow(main_info))
    if(nrp == 1){ a_structure <- main_info }
  }
  if(holdout_indicator == 0){
    a_structure_leftoutLdminus1 <- main_info[which(c(diff(main_info$d),1)==0),]
    a_structure_leftoutLdminus1$d_index <- 1:nrow(a_structure_leftoutLdminus1)
  }

  interaction_info <- data.frame()
  interaction_info_PreRegularization <- interaction_info
  regularization_adjust_hash <- main_info$d
  names(regularization_adjust_hash) <- main_info$d

  factor_index_list <- vector("list", length(factor_levels))
  offset <- 0L
  for (d_ in seq_along(factor_levels)) {
    n_levels_use <- ai(factor_levels[d_] - holdout_indicator)
    idx <- if (n_levels_use > 0L) {
      as.integer(offset + seq_len(n_levels_use) - 1L)
    } else {
      integer(0)
    }
    factor_index_list[[d_]] <- strenv$jnp$array(idx)$astype(strenv$jnp$int32)
    offset <- offset + n_levels_use
  }
  n_candidate_tokens <- ai(length(factor_levels) + 1L)

  # Party token mapping (candidates)
  party_levels <- if (!is.null(competing_group_variable_candidate_)) {
    sort(unique(as.character(competing_group_variable_candidate_)))
  } else {
    "NA"
  }
  n_party_levels <- max(1L, length(party_levels))
  party_index <- if (!is.null(competing_group_variable_candidate_)) {
    match(as.character(competing_group_variable_candidate_), party_levels) - 1L
  } else {
    rep(0L, length(Y_))
  }
  party_index[is.na(party_index)] <- 0L

  # Respondent party mapping
  resp_party_levels <- if (!is.null(competing_group_variable_respondent_)) {
    sort(unique(as.character(competing_group_variable_respondent_)))
  } else {
    "NA"
  }
  n_resp_party_levels <- max(1L, length(resp_party_levels))
  resp_party_index <- if (!is.null(competing_group_variable_respondent_)) {
    match(as.character(competing_group_variable_respondent_), resp_party_levels) - 1L
  } else {
    rep(0L, length(Y_))
  }
  resp_party_index[is.na(resp_party_index)] <- 0L

  # Respondent covariates (optional)
  X_use <- NULL
  X_ <- NULL
  if (exists("X", inherits = TRUE) && !is.null(X)) {
    X_ <- as.matrix(X[indi_, , drop = FALSE])
  }

  # Helper to sanitize integer indices
  to_index_matrix <- function(x_mat){
    x_mat <- as.matrix(x_mat)
    if (anyNA(x_mat)) {
      x_mat[is.na(x_mat)] <- 1L
    }
    x_int <- matrix(as.integer(x_mat) - 1L, nrow = nrow(x_mat), ncol = ncol(x_mat))
    x_int[x_int < 0L] <- 0L
    x_int
  }

  # Build pairwise or single-candidate data
  pair_mat <- NULL
  if (pairwise_mode) {
    pair_indices_list <- tapply(seq_along(pair_id_), pair_id_, c)
    pair_sizes <- lengths(pair_indices_list)
    if (!all(pair_sizes == 2L)) {
      warning("pair_id does not define exactly 2 rows per pair; falling back to single-candidate model.")
      pairwise_mode <- FALSE
    } else {
      profile_order_present <- !is.null(profile_order_) && length(profile_order_) == length(Y_)
      row_key <- apply(W_, 1, function(row) {
        paste(ifelse(is.na(row), "NA", as.character(row)), collapse = "|")
      })
      row_hash <- vapply(row_key, function(key) {
        ints <- utf8ToInt(key)
        if (!length(ints)) {
          return(0)
        }
        sum(ints * seq_along(ints)) %% 2147483647
      }, numeric(1))

      pair_mat <- do.call(rbind, lapply(pair_indices_list, function(idx){
        order_by_profile <- profile_order_present &&
          length(idx) == 2L &&
          length(unique(profile_order_[idx])) == 2L &&
          !any(is.na(profile_order_[idx]))
        if (!is.null(competing_group_variable_candidate_)) {
          if (order_by_profile) {
            idx[order(competing_group_variable_candidate_[idx],
                      profile_order_[idx],
                      row_hash[idx],
                      idx)]
          } else {
            idx[order(competing_group_variable_candidate_[idx],
                      row_hash[idx],
                      idx)]
          }
        } else if (order_by_profile) {
          idx[order(profile_order_[idx],
                    row_hash[idx],
                    idx)]
        } else {
          idx[order(row_hash[idx], idx)]
        }
      }))
    }
  }

  if (pairwise_mode) {
    X_left <- W_[pair_mat[,1], , drop = FALSE]
    X_right <- W_[pair_mat[,2], , drop = FALSE]
    Y_use <- Y_[pair_mat[,1]]
    party_left <- party_index[pair_mat[,1]]
    party_right <- party_index[pair_mat[,2]]
    resp_party_use <- resp_party_index[pair_mat[,1]]
    if (!is.null(X_)) {
      X_use <- X_[pair_mat[,1], , drop = FALSE]
    }
  } else {
    X_single <- W_
    Y_use <- Y_
    party_single <- party_index
    resp_party_use <- resp_party_index
    X_use <- X_
  }

  n_resp_covariates <- if (!is.null(X_use)) ai(ncol(X_use)) else ai(0L)
  resp_cov_mean <- if (!is.null(X_use) && n_resp_covariates > 0L) {
    as.numeric(colMeans(X_use))
  } else {
    NULL
  }

  # Placeholder to keep model chunks consistent (no surrogate regression)
  main_dat <- matrix(0, nrow = 0L, ncol = 0L)

  # Likelihood selection
  is_binary <- all(unique(na.omit(as.numeric(Y_use))) %in% c(0, 1)) &&
    length(unique(na.omit(Y_use))) <= 2
  is_intvec <- all(!is.na(Y_use)) && all(abs(Y_use - round(Y_use)) < 1e-8)
  K_classes <- if (is_intvec) length(unique(ai(Y_use))) else NA_integer_

  if (is_binary) {
    likelihood <- "bernoulli"; nOutcomes <- ai(1L)
  } else if (!is.na(K_classes) && K_classes >= 2L && K_classes <= max(50L, ncol(W_) + 1L)) {
    likelihood <- "categorical"; nOutcomes <- ai(K_classes)
  } else {
    likelihood <- "normal"; nOutcomes <- ai(1L)
  }

  pdtype_ <- ddtype_ <- strenv$jnp$float32
  rms_norm <- function(x, g, eps = 1e-6) {
    mean_sq <- strenv$jnp$mean(x * x, axis = -1L, keepdims = TRUE)
    inv_rms <- strenv$jnp$reciprocal(strenv$jnp$sqrt(mean_sq + eps))
    g_use <- strenv$jnp$reshape(g, list(1L, 1L, ModelDims))
    x * inv_rms * g_use
  }

  BayesianPairTransformerModel <- function(X_left, X_right, party_left, party_right,
                                           resp_party, resp_cov, Y_obs) {
    N_local <- ai(X_left$shape[[1]])
    D_local <- ai(X_left$shape[[2]])

    for (d_ in 1L:D_local) {
      assign(paste0("E_factor_", d_),
             strenv$numpyro$sample(paste0("E_factor_", d_),
                                   strenv$numpyro$distributions$Normal(0., 1.),
                                   sample_shape = reticulate::tuple(ai(factor_levels[d_]), ModelDims)))
    }
    E_party <- strenv$numpyro$sample("E_party",
                                    strenv$numpyro$distributions$Normal(0., 1.),
                                    sample_shape = reticulate::tuple(ai(n_party_levels), ModelDims))
    E_role <- strenv$numpyro$sample("E_role",
                                   strenv$numpyro$distributions$Normal(0., 1.),
                                   sample_shape = reticulate::tuple(ai(2L), ModelDims))
    E_resp_party <- strenv$numpyro$sample("E_resp_party",
                                         strenv$numpyro$distributions$Normal(0., 1.),
                                         sample_shape = reticulate::tuple(ai(n_resp_party_levels), ModelDims))
    if (n_resp_covariates > 0L) {
      W_resp_x <- strenv$numpyro$sample("W_resp_x",
                                       strenv$numpyro$distributions$Normal(0., 1.),
                                       sample_shape = reticulate::tuple(ai(n_resp_covariates), ModelDims))
    }

    for (l_ in 1L:ModelDepth) {
      assign(paste0("RMS_attn_l", l_),
             strenv$numpyro$sample(paste0("RMS_attn_l", l_),
                                   strenv$numpyro$distributions$Normal(1., 0.1),
                                   sample_shape = reticulate::tuple(ModelDims)))
      assign(paste0("RMS_ff_l", l_),
             strenv$numpyro$sample(paste0("RMS_ff_l", l_),
                                   strenv$numpyro$distributions$Normal(1., 0.1),
                                   sample_shape = reticulate::tuple(ModelDims)))
      assign(paste0("W_q_l", l_),
             strenv$numpyro$sample(paste0("W_q_l", l_),
                                   strenv$numpyro$distributions$Normal(0., 1.),
                                   sample_shape = reticulate::tuple(ModelDims, ModelDims)))
      assign(paste0("W_k_l", l_),
             strenv$numpyro$sample(paste0("W_k_l", l_),
                                   strenv$numpyro$distributions$Normal(0., 1.),
                                   sample_shape = reticulate::tuple(ModelDims, ModelDims)))
      assign(paste0("W_v_l", l_),
             strenv$numpyro$sample(paste0("W_v_l", l_),
                                   strenv$numpyro$distributions$Normal(0., 1.),
                                   sample_shape = reticulate::tuple(ModelDims, ModelDims)))
      assign(paste0("W_o_l", l_),
             strenv$numpyro$sample(paste0("W_o_l", l_),
                                   strenv$numpyro$distributions$Normal(0., 1.),
                                   sample_shape = reticulate::tuple(ModelDims, ModelDims)))
      assign(paste0("W_ff1_l", l_),
             strenv$numpyro$sample(paste0("W_ff1_l", l_),
                                   strenv$numpyro$distributions$Normal(0., 1.),
                                   sample_shape = reticulate::tuple(ModelDims, FFDim)))
      assign(paste0("W_ff2_l", l_),
             strenv$numpyro$sample(paste0("W_ff2_l", l_),
                                   strenv$numpyro$distributions$Normal(0., 1.),
                                   sample_shape = reticulate::tuple(FFDim, ModelDims)))
    }

    W_out <- strenv$numpyro$sample("W_out",
                                  strenv$numpyro$distributions$Normal(0., 1.),
                                  sample_shape = reticulate::tuple(ModelDims, nOutcomes))
    b_out <- strenv$numpyro$sample("b_out",
                                  strenv$numpyro$distributions$Normal(0., 1.),
                                  sample_shape = reticulate::tuple(nOutcomes))
    if (likelihood == "normal") {
      sigma <- strenv$numpyro$sample("sigma", strenv$numpyro$distributions$HalfNormal(1.0))
    }

    embed_candidate <- function(X_idx, party_idx, role_id) {
      N_batch <- ai(X_idx$shape[[1]])
      token_list <- vector("list", D_local)
      for (d_ in 1L:D_local) {
        E_d <- get(paste0("E_factor_", d_))
        idx_d <- strenv$jnp$take(X_idx, ai(d_ - 1L), axis = 1L)
        token_list[[d_]] <- strenv$jnp$take(E_d, idx_d, axis = 0L)
      }
      tokens <- strenv$jnp$stack(token_list, axis = 1L)
      role_vec <- strenv$jnp$take(E_role, strenv$jnp$array(ai(role_id)), axis = 0L)
      role_vec <- strenv$jnp$reshape(role_vec, list(1L, 1L, ModelDims))
      tokens <- tokens + role_vec
      party_tok <- strenv$jnp$take(E_party, party_idx, axis = 0L)
      party_tok <- strenv$jnp$reshape(party_tok, list(N_batch, 1L, ModelDims))
      party_tok <- party_tok + role_vec
      strenv$jnp$concatenate(list(tokens, party_tok), axis = 1L)
    }

    run_transformer <- function(tokens) {
      for (l_ in 1L:ModelDepth) {
        Wq <- get(paste0("W_q_l", l_))
        Wk <- get(paste0("W_k_l", l_))
        Wv <- get(paste0("W_v_l", l_))
        Wo <- get(paste0("W_o_l", l_))
        Wff1 <- get(paste0("W_ff1_l", l_))
        Wff2 <- get(paste0("W_ff2_l", l_))
        RMS_attn <- get(paste0("RMS_attn_l", l_))
        RMS_ff <- get(paste0("RMS_ff_l", l_))

        tokens_norm <- rms_norm(tokens, RMS_attn)
        Q <- strenv$jnp$einsum("ntm,mm->ntm", tokens_norm, Wq)
        K <- strenv$jnp$einsum("ntm,mm->ntm", tokens_norm, Wk)
        V <- strenv$jnp$einsum("ntm,mm->ntm", tokens_norm, Wv)

        Qh <- strenv$jnp$reshape(Q, list(Q$shape[[1]], Q$shape[[2]], TransformerHeads, head_dim))
        Kh <- strenv$jnp$reshape(K, list(K$shape[[1]], K$shape[[2]], TransformerHeads, head_dim))
        Vh <- strenv$jnp$reshape(V, list(V$shape[[1]], V$shape[[2]], TransformerHeads, head_dim))

        scale_ <- strenv$jnp$sqrt(strenv$jnp$array(as.numeric(head_dim)))
        scores <- strenv$jnp$einsum("nqhd,nkhd->nhqk", Qh, Kh) / scale_
        attn <- strenv$jax$nn$softmax(scores, axis = -1L)
        context_h <- strenv$jnp$einsum("nhqk,nkhd->nqhd", attn, Vh)
        context <- strenv$jnp$reshape(context_h, list(context_h$shape[[1]], context_h$shape[[2]], ModelDims))
        attn_out <- strenv$jnp$einsum("ntm,mm->ntm", context, Wo)

        h1 <- tokens + attn_out
        h1_norm <- rms_norm(h1, RMS_ff)
        ff_pre <- strenv$jnp$einsum("ntm,mf->ntf", h1_norm, Wff1)
        ff_act <- strenv$jax$nn$swish(ff_pre)
        ff_out <- strenv$jnp$einsum("ntf,fm->ntm", ff_act, Wff2)
        tokens <- h1 + ff_out
      }
      tokens
    }

    do_forward_and_lik_ <- function(Xl, Xr, pl, pr, resp_p, resp_c, Yb) {
      N_batch <- ai(Xl$shape[[1]])
      tok_left <- embed_candidate(Xl, pl, 0L)
      tok_right <- embed_candidate(Xr, pr, 1L)
      resp_tokens <- list()
      resp_party_tok <- strenv$jnp$take(E_resp_party, resp_p, axis = 0L)
      resp_party_tok <- strenv$jnp$reshape(resp_party_tok, list(N_batch, 1L, ModelDims))
      resp_tokens[[length(resp_tokens) + 1L]] <- resp_party_tok
      if (n_resp_covariates > 0L) {
        resp_cov_tok <- strenv$jnp$einsum("nc,cm->nm", resp_c, W_resp_x)
        resp_cov_tok <- strenv$jnp$reshape(resp_cov_tok, list(N_batch, 1L, ModelDims))
        resp_tokens[[length(resp_tokens) + 1L]] <- resp_cov_tok
      }
      tokens <- strenv$jnp$concatenate(c(list(tok_left, tok_right), resp_tokens), axis = 1L)
      tokens <- run_transformer(tokens)

      n_tok <- ai(D_local + 1L)
      tok_left_out <- strenv$jnp$take(tokens, strenv$jnp$arange(n_tok), axis = 1L)
      tok_right_out <- strenv$jnp$take(tokens, strenv$jnp$arange(n_tok, 2L * n_tok), axis = 1L)
      h_left <- strenv$jnp$mean(tok_left_out, axis = 1L)
      h_right <- strenv$jnp$mean(tok_right_out, axis = 1L)
      h_diff <- h_left - h_right

      logits <- strenv$jnp$einsum("nm,mo->no", h_diff, W_out) + b_out

      if (likelihood == "bernoulli") {
        logits_vec <- strenv$jnp$squeeze(logits, axis = 1L)
        strenv$numpyro$sample("obs",
                              strenv$numpyro$distributions$Bernoulli(logits = logits_vec),
                              obs = Yb)
      }
      if (likelihood == "categorical") {
        strenv$numpyro$sample("obs",
                              strenv$numpyro$distributions$Categorical(logits = logits),
                              obs = Yb)
      }
      if (likelihood == "normal") {
        mu <- strenv$jnp$squeeze(logits, axis = 1L)
        strenv$numpyro$sample("obs",
                              strenv$numpyro$distributions$Normal(mu, sigma),
                              obs = Yb)
      }
    }

    local_lik <- function() {
      if (isTRUE(subsample_method %in% c("batch", "batch_vi"))) {
        with(strenv$numpyro$plate("data", size = N_local,
                                  subsample_size = ai(mcmc_control$batch_size),
                                  dim = -1L) %as% "idx", {
                                    Xl_b <- strenv$jnp$take(X_left, idx, axis = 0L)
                                    Xr_b <- strenv$jnp$take(X_right, idx, axis = 0L)
                                    pl_b <- strenv$jnp$take(party_left, idx, axis = 0L)
                                    pr_b <- strenv$jnp$take(party_right, idx, axis = 0L)
                                    resp_p_b <- strenv$jnp$take(resp_party, idx, axis = 0L)
                                    resp_c_b <- strenv$jnp$take(resp_cov, idx, axis = 0L)
                                    Yb <- strenv$jnp$take(Y_obs, idx, axis = 0L)
                                    do_forward_and_lik_(Xl_b, Xr_b, pl_b, pr_b, resp_p_b, resp_c_b, Yb)
                                  })
      } else {
        with(strenv$numpyro$plate("data", size = N_local, dim = -1L), {
          do_forward_and_lik_(X_left, X_right, party_left, party_right,
                              resp_party, resp_cov, Y_obs)
        })
      }
    }

    local_lik()
  }

  BayesianSingleTransformerModel <- function(X, party, resp_party, resp_cov, Y_obs) {
    N_local <- ai(X$shape[[1]])
    D_local <- ai(X$shape[[2]])

    for (d_ in 1L:D_local) {
      assign(paste0("E_factor_", d_),
             strenv$numpyro$sample(paste0("E_factor_", d_),
                                   strenv$numpyro$distributions$Normal(0., 1.),
                                   sample_shape = reticulate::tuple(ai(factor_levels[d_]), ModelDims)))
    }
    E_party <- strenv$numpyro$sample("E_party",
                                    strenv$numpyro$distributions$Normal(0., 1.),
                                    sample_shape = reticulate::tuple(ai(n_party_levels), ModelDims))
    E_role <- strenv$numpyro$sample("E_role",
                                   strenv$numpyro$distributions$Normal(0., 1.),
                                   sample_shape = reticulate::tuple(ai(1L), ModelDims))
    E_resp_party <- strenv$numpyro$sample("E_resp_party",
                                         strenv$numpyro$distributions$Normal(0., 1.),
                                         sample_shape = reticulate::tuple(ai(n_resp_party_levels), ModelDims))
    if (n_resp_covariates > 0L) {
      W_resp_x <- strenv$numpyro$sample("W_resp_x",
                                       strenv$numpyro$distributions$Normal(0., 1.),
                                       sample_shape = reticulate::tuple(ai(n_resp_covariates), ModelDims))
    }

    for (l_ in 1L:ModelDepth) {
      assign(paste0("RMS_attn_l", l_),
             strenv$numpyro$sample(paste0("RMS_attn_l", l_),
                                   strenv$numpyro$distributions$Normal(1., 0.1),
                                   sample_shape = reticulate::tuple(ModelDims)))
      assign(paste0("RMS_ff_l", l_),
             strenv$numpyro$sample(paste0("RMS_ff_l", l_),
                                   strenv$numpyro$distributions$Normal(1., 0.1),
                                   sample_shape = reticulate::tuple(ModelDims)))
      assign(paste0("W_q_l", l_),
             strenv$numpyro$sample(paste0("W_q_l", l_),
                                   strenv$numpyro$distributions$Normal(0., 1.),
                                   sample_shape = reticulate::tuple(ModelDims, ModelDims)))
      assign(paste0("W_k_l", l_),
             strenv$numpyro$sample(paste0("W_k_l", l_),
                                   strenv$numpyro$distributions$Normal(0., 1.),
                                   sample_shape = reticulate::tuple(ModelDims, ModelDims)))
      assign(paste0("W_v_l", l_),
             strenv$numpyro$sample(paste0("W_v_l", l_),
                                   strenv$numpyro$distributions$Normal(0., 1.),
                                   sample_shape = reticulate::tuple(ModelDims, ModelDims)))
      assign(paste0("W_o_l", l_),
             strenv$numpyro$sample(paste0("W_o_l", l_),
                                   strenv$numpyro$distributions$Normal(0., 1.),
                                   sample_shape = reticulate::tuple(ModelDims, ModelDims)))
      assign(paste0("W_ff1_l", l_),
             strenv$numpyro$sample(paste0("W_ff1_l", l_),
                                   strenv$numpyro$distributions$Normal(0., 1.),
                                   sample_shape = reticulate::tuple(ModelDims, FFDim)))
      assign(paste0("W_ff2_l", l_),
             strenv$numpyro$sample(paste0("W_ff2_l", l_),
                                   strenv$numpyro$distributions$Normal(0., 1.),
                                   sample_shape = reticulate::tuple(FFDim, ModelDims)))
    }

    W_out <- strenv$numpyro$sample("W_out",
                                  strenv$numpyro$distributions$Normal(0., 1.),
                                  sample_shape = reticulate::tuple(ModelDims, nOutcomes))
    b_out <- strenv$numpyro$sample("b_out",
                                  strenv$numpyro$distributions$Normal(0., 1.),
                                  sample_shape = reticulate::tuple(nOutcomes))
    if (likelihood == "normal") {
      sigma <- strenv$numpyro$sample("sigma", strenv$numpyro$distributions$HalfNormal(1.0))
    }

    embed_candidate <- function(X_idx, party_idx) {
      N_batch <- ai(X_idx$shape[[1]])
      token_list <- vector("list", D_local)
      for (d_ in 1L:D_local) {
        E_d <- get(paste0("E_factor_", d_))
        idx_d <- strenv$jnp$take(X_idx, ai(d_ - 1L), axis = 1L)
        token_list[[d_]] <- strenv$jnp$take(E_d, idx_d, axis = 0L)
      }
      tokens <- strenv$jnp$stack(token_list, axis = 1L)
      role_vec <- strenv$jnp$take(E_role, strenv$jnp$array(ai(0L)), axis = 0L)
      role_vec <- strenv$jnp$reshape(role_vec, list(1L, 1L, ModelDims))
      tokens <- tokens + role_vec
      party_tok <- strenv$jnp$take(E_party, party_idx, axis = 0L)
      party_tok <- strenv$jnp$reshape(party_tok, list(N_batch, 1L, ModelDims))
      party_tok <- party_tok + role_vec
      strenv$jnp$concatenate(list(tokens, party_tok), axis = 1L)
    }

    run_transformer <- function(tokens) {
      for (l_ in 1L:ModelDepth) {
        Wq <- get(paste0("W_q_l", l_))
        Wk <- get(paste0("W_k_l", l_))
        Wv <- get(paste0("W_v_l", l_))
        Wo <- get(paste0("W_o_l", l_))
        Wff1 <- get(paste0("W_ff1_l", l_))
        Wff2 <- get(paste0("W_ff2_l", l_))
        RMS_attn <- get(paste0("RMS_attn_l", l_))
        RMS_ff <- get(paste0("RMS_ff_l", l_))

        tokens_norm <- rms_norm(tokens, RMS_attn)
        Q <- strenv$jnp$einsum("ntm,mm->ntm", tokens_norm, Wq)
        K <- strenv$jnp$einsum("ntm,mm->ntm", tokens_norm, Wk)
        V <- strenv$jnp$einsum("ntm,mm->ntm", tokens_norm, Wv)

        Qh <- strenv$jnp$reshape(Q, list(Q$shape[[1]], Q$shape[[2]], TransformerHeads, head_dim))
        Kh <- strenv$jnp$reshape(K, list(K$shape[[1]], K$shape[[2]], TransformerHeads, head_dim))
        Vh <- strenv$jnp$reshape(V, list(V$shape[[1]], V$shape[[2]], TransformerHeads, head_dim))

        scale_ <- strenv$jnp$sqrt(strenv$jnp$array(as.numeric(head_dim)))
        scores <- strenv$jnp$einsum("nqhd,nkhd->nhqk", Qh, Kh) / scale_
        attn <- strenv$jax$nn$softmax(scores, axis = -1L)
        context_h <- strenv$jnp$einsum("nhqk,nkhd->nqhd", attn, Vh)
        context <- strenv$jnp$reshape(context_h, list(context_h$shape[[1]], context_h$shape[[2]], ModelDims))
        attn_out <- strenv$jnp$einsum("ntm,mm->ntm", context, Wo)

        h1 <- tokens + attn_out
        h1_norm <- rms_norm(h1, RMS_ff)
        ff_pre <- strenv$jnp$einsum("ntm,mf->ntf", h1_norm, Wff1)
        ff_act <- strenv$jax$nn$swish(ff_pre)
        ff_out <- strenv$jnp$einsum("ntf,fm->ntm", ff_act, Wff2)
        tokens <- h1 + ff_out
      }
      tokens
    }

    do_forward_and_lik_ <- function(Xb, pb, resp_p, resp_c, Yb) {
      N_batch <- ai(Xb$shape[[1]])
      tokens <- embed_candidate(Xb, pb)
      resp_tokens <- list()
      resp_party_tok <- strenv$jnp$take(E_resp_party, resp_p, axis = 0L)
      resp_party_tok <- strenv$jnp$reshape(resp_party_tok, list(N_batch, 1L, ModelDims))
      resp_tokens[[length(resp_tokens) + 1L]] <- resp_party_tok
      if (n_resp_covariates > 0L) {
        resp_cov_tok <- strenv$jnp$einsum("nc,cm->nm", resp_c, W_resp_x)
        resp_cov_tok <- strenv$jnp$reshape(resp_cov_tok, list(N_batch, 1L, ModelDims))
        resp_tokens[[length(resp_tokens) + 1L]] <- resp_cov_tok
      }
      tokens <- strenv$jnp$concatenate(c(list(tokens), resp_tokens), axis = 1L)
      tokens <- run_transformer(tokens)
      n_tok <- ai(D_local + 1L)
      tok_out <- strenv$jnp$take(tokens, strenv$jnp$arange(n_tok), axis = 1L)
      h <- strenv$jnp$mean(tok_out, axis = 1L)
      logits <- strenv$jnp$einsum("nm,mo->no", h, W_out) + b_out

      if (likelihood == "bernoulli") {
        logits_vec <- strenv$jnp$squeeze(logits, axis = 1L)
        strenv$numpyro$sample("obs",
                              strenv$numpyro$distributions$Bernoulli(logits = logits_vec),
                              obs = Yb)
      }
      if (likelihood == "categorical") {
        strenv$numpyro$sample("obs",
                              strenv$numpyro$distributions$Categorical(logits = logits),
                              obs = Yb)
      }
      if (likelihood == "normal") {
        mu <- strenv$jnp$squeeze(logits, axis = 1L)
        strenv$numpyro$sample("obs",
                              strenv$numpyro$distributions$Normal(mu, sigma),
                              obs = Yb)
      }
    }

    local_lik <- function() {
      if (isTRUE(subsample_method %in% c("batch", "batch_vi"))) {
        with(strenv$numpyro$plate("data", size = N_local,
                                  subsample_size = ai(mcmc_control$batch_size),
                                  dim = -1L) %as% "idx", {
                                    Xb <- strenv$jnp$take(X, idx, axis = 0L)
                                    pb <- strenv$jnp$take(party, idx, axis = 0L)
                                    resp_p_b <- strenv$jnp$take(resp_party, idx, axis = 0L)
                                    resp_c_b <- strenv$jnp$take(resp_cov, idx, axis = 0L)
                                    Yb <- strenv$jnp$take(Y_obs, idx, axis = 0L)
                                    do_forward_and_lik_(Xb, pb, resp_p_b, resp_c_b, Yb)
                                  })
      } else {
        with(strenv$numpyro$plate("data", size = N_local, dim = -1L), {
          do_forward_and_lik_(X, party, resp_party, resp_cov, Y_obs)
        })
      }
    }

    local_lik()
  }

  if (likelihood == "categorical") {
    y_fac <- ai(as.factor(Y_use)) - 1L
    Y_jnp <- strenv$jnp$array(y_fac)$astype(strenv$jnp$int32)
  } else {
    Y_jnp <- strenv$jnp$array(as.numeric(Y_use))$astype(ddtype_)
  }
  resp_party_jnp <- strenv$jnp$array(as.integer(resp_party_use))$astype(strenv$jnp$int32)
  if (n_resp_covariates > 0L) {
    resp_cov_jnp <- strenv$jnp$array(as.matrix(X_use))$astype(ddtype_)
  } else {
    resp_cov_jnp <- strenv$jnp$zeros(list(ai(length(Y_use)), ai(0L)), dtype = ddtype_)
  }

  if (pairwise_mode) {
    X_left_jnp <- strenv$jnp$array(to_index_matrix(X_left))$astype(strenv$jnp$int32)
    X_right_jnp <- strenv$jnp$array(to_index_matrix(X_right))$astype(strenv$jnp$int32)
    party_left_jnp <- strenv$jnp$array(as.integer(party_left))$astype(strenv$jnp$int32)
    party_right_jnp <- strenv$jnp$array(as.integer(party_right))$astype(strenv$jnp$int32)
  } else {
    X_single_jnp <- strenv$jnp$array(to_index_matrix(X_single))$astype(strenv$jnp$int32)
    party_single_jnp <- strenv$jnp$array(as.integer(party_single))$astype(strenv$jnp$int32)
  }

  t0_ <- Sys.time()
  if (identical(subsample_method, "batch_vi")) {
    message("Enlisting SVI with autoguide for minibatched likelihood...")
    model_fn <- if (pairwise_mode) BayesianPairTransformerModel else BayesianSingleTransformerModel
    guide_name <- if (!is.null(mcmc_control$vi_guide)) {
      tolower(as.character(mcmc_control$vi_guide))
    } else {
      "auto_diagonal"
    }
    if (length(guide_name) != 1L || is.na(guide_name) || !nzchar(guide_name)) {
      guide_name <- "auto_diagonal"
    }
    guide <- switch(guide_name,
                    auto_delta = strenv$numpyro$infer$autoguide$AutoDelta(model_fn),
                    auto_normal = strenv$numpyro$infer$autoguide$AutoNormal(model_fn),
                    auto_diagonal = strenv$numpyro$infer$autoguide$AutoDiagonalNormal(model_fn),
                    stop(sprintf("Unknown vi_guide '%s' for SVI.", guide_name), call. = FALSE))
    n_particles <- ai(mcmc_control$svi_num_particles)
    if (length(n_particles) != 1L || is.na(n_particles) || n_particles < 1L) {
      n_particles <- 1L
    }
    svi <- strenv$numpyro$infer$SVI(
      model = model_fn,
      guide = guide,
      optim = strenv$numpyro$optim$Adam(as.numeric(mcmc_control$svi_lr)),
      loss = strenv$numpyro$infer$Trace_ELBO(
        num_particles = n_particles
      )
    )
    rng_key <- strenv$jax$random$PRNGKey(ai(runif(1, 0, 10000)))
    if (pairwise_mode) {
      svi_result <- svi$run(rng_key,
                            ai(mcmc_control$svi_steps),
                            X_left = X_left_jnp,
                            X_right = X_right_jnp,
                            party_left = party_left_jnp,
                            party_right = party_right_jnp,
                            resp_party = resp_party_jnp,
                            resp_cov = resp_cov_jnp,
                            Y_obs = Y_jnp)
    } else {
      svi_result <- svi$run(rng_key,
                            ai(mcmc_control$svi_steps),
                            X = X_single_jnp,
                            party = party_single_jnp,
                            resp_party = resp_party_jnp,
                            resp_cov = resp_cov_jnp,
                            Y_obs = Y_jnp)
    }
    svi_state <- if (!is.null(svi_result$state)) {
      svi_result$state
    } else if (length(svi_result) > 0L) {
      svi_result[[1]]
    } else {
      svi_result
    }
    params <- svi$get_params(svi_state)
    n_draws <- ai(mcmc_control$svi_num_draws)
    if (length(n_draws) != 1L || is.na(n_draws) || !is.finite(n_draws) || n_draws < 1L) {
      n_draws <- 1L
    }
    sample_key <- strenv$jax$random$PRNGKey(ai(runif(1, 0, 10000)))
    posterior_samples <- guide$sample_posterior(
      sample_key,
      params,
      sample_shape = reticulate::tuple(ai(n_draws))
    )
    PosteriorDraws <- lapply(posterior_samples, function(x) {
      strenv$jnp$expand_dims(x, 0L)
    })
    names(PosteriorDraws) <- names(posterior_samples)
    message(sprintf("\n SVI Runtime: %.3f min",
                    as.numeric(difftime(Sys.time(), t0_, units = "secs"))/60))
  } else {
    strenv$numpyro$set_host_device_count(mcmc_control$n_chains)

    if (identical(subsample_method, "batch")) {
      message("Enlisting HMCECS kernels for subsampled likelihood...")
      kernel <- strenv$numpyro$infer$HMCECS(
        strenv$numpyro$infer$NUTS(if (pairwise_mode) BayesianPairTransformerModel else BayesianSingleTransformerModel),
        num_blocks = if (!is.null(mcmc_control$num_blocks)) ai(mcmc_control$num_blocks) else ai(4L)
      )
    } else {
      message("Enlisting NUTS kernels for full-data likelihood...")
      kernel <- strenv$numpyro$infer$NUTS(
        if (pairwise_mode) BayesianPairTransformerModel else BayesianSingleTransformerModel,
        max_tree_depth = ai(8L),
        target_accept_prob = 0.85
      )
    }

    sampler <- strenv$numpyro$infer$MCMC(
      sampler = kernel,
      num_warmup = mcmc_control$n_samples_warmup,
      num_samples = mcmc_control$n_samples_mcmc,
      thinning   = mcmc_control$n_thin_by,
      chain_method = ifelse(!is.null(mcmc_control$chain_method), yes = mcmc_control$chain_method, no = "parallel"),
      num_chains = mcmc_control$n_chains,
      jit_model_args = TRUE,
      progress_bar = TRUE
    )

    if (pairwise_mode) {
      sampler$run(strenv$jax$random$PRNGKey(ai(runif(1, 0, 10000))),
                  X_left = X_left_jnp,
                  X_right = X_right_jnp,
                  party_left = party_left_jnp,
                  party_right = party_right_jnp,
                  resp_party = resp_party_jnp,
                  resp_cov = resp_cov_jnp,
                  Y_obs = Y_jnp)
    } else {
      sampler$run(strenv$jax$random$PRNGKey(ai(runif(1, 0, 10000))),
                  X = X_single_jnp,
                  party = party_single_jnp,
                  resp_party = resp_party_jnp,
                  resp_cov = resp_cov_jnp,
                  Y_obs = Y_jnp)
    }
    PosteriorDraws <- sampler$get_samples(group_by_chain = TRUE)
    message(sprintf("\n MCMC Runtime: %.3f min",
                    as.numeric(difftime(Sys.time(), t0_, units = "secs"))/60))
  }

  mean_param <- function(x) { strenv$jnp$mean(x, 0L:1L) }

  ParamsMean <- list(
    W_out   = mean_param(PosteriorDraws$W_out),
    b_out   = mean_param(PosteriorDraws$b_out),
    E_party = mean_param(PosteriorDraws$E_party),
    E_role  = mean_param(PosteriorDraws$E_role),
    E_resp_party = mean_param(PosteriorDraws$E_resp_party)
  )
  if (likelihood == "normal") {
    ParamsMean$sigma <- mean_param(PosteriorDraws$sigma)
  }
  if (n_resp_covariates > 0L && !is.null(PosteriorDraws$W_resp_x)) {
    ParamsMean$W_resp_x <- mean_param(PosteriorDraws$W_resp_x)
  }
  for (d_ in 1:length(factor_levels)) {
    ParamsMean[[paste0("E_factor_", d_)]] <- mean_param(PosteriorDraws[[paste0("E_factor_", d_)]])
  }
  for (l_ in 1L:ModelDepth) {
    ParamsMean[[paste0("RMS_attn_l", l_)]] <- mean_param(PosteriorDraws[[paste0("RMS_attn_l", l_)]])
    ParamsMean[[paste0("RMS_ff_l", l_)]] <- mean_param(PosteriorDraws[[paste0("RMS_ff_l", l_)]])
    ParamsMean[[paste0("W_q_l", l_)]]  <- mean_param(PosteriorDraws[[paste0("W_q_l",  l_)]])
    ParamsMean[[paste0("W_k_l", l_)]]  <- mean_param(PosteriorDraws[[paste0("W_k_l",  l_)]])
    ParamsMean[[paste0("W_v_l", l_)]]  <- mean_param(PosteriorDraws[[paste0("W_v_l",  l_)]])
    ParamsMean[[paste0("W_o_l", l_)]]  <- mean_param(PosteriorDraws[[paste0("W_o_l",  l_)]])
    ParamsMean[[paste0("W_ff1_l", l_)]]<- mean_param(PosteriorDraws[[paste0("W_ff1_l",l_)]])
    ParamsMean[[paste0("W_ff2_l", l_)]]<- mean_param(PosteriorDraws[[paste0("W_ff2_l",l_)]])
  }

  TransformerPredict_pair <- function(params, Xl_new, Xr_new, pl_new, pr_new,
                                      resp_party_new = NULL, resp_cov_new = NULL,
                                      return_logits = FALSE) {
    Xl <- strenv$jnp$array(to_index_matrix(Xl_new))$astype(strenv$jnp$int32)
    Xr <- strenv$jnp$array(to_index_matrix(Xr_new))$astype(strenv$jnp$int32)
    pl <- strenv$jnp$array(as.integer(pl_new))$astype(strenv$jnp$int32)
    pr <- strenv$jnp$array(as.integer(pr_new))$astype(strenv$jnp$int32)
    if (is.null(resp_party_new)) {
      resp_party_new <- rep(0L, nrow(Xl_new))
    }
    resp_p <- strenv$jnp$array(as.integer(resp_party_new))$astype(strenv$jnp$int32)
    if (is.null(resp_cov_new)) {
      resp_cov_new <- if (!is.null(resp_cov_mean)) {
        matrix(rep(resp_cov_mean, each = nrow(Xl_new)), nrow = nrow(Xl_new))
      } else {
        matrix(0, nrow = nrow(Xl_new), ncol = 0L)
      }
    }
    resp_c <- strenv$jnp$array(as.matrix(resp_cov_new))$astype(ddtype_)

    embed_candidate <- function(X_idx, party_idx, role_id) {
      D_local <- ai(X_idx$shape[[2]])
      token_list <- vector("list", D_local)
      for (d_ in 1L:D_local) {
        E_d <- params[[paste0("E_factor_", d_)]]
        idx_d <- strenv$jnp$take(X_idx, ai(d_ - 1L), axis = 1L)
        token_list[[d_]] <- strenv$jnp$take(E_d, idx_d, axis = 0L)
      }
      tokens <- strenv$jnp$stack(token_list, axis = 1L)
      role_vec <- strenv$jnp$take(params$E_role, strenv$jnp$array(ai(role_id)), axis = 0L)
      role_vec <- strenv$jnp$reshape(role_vec, list(1L, 1L, ModelDims))
      tokens <- tokens + role_vec
      party_tok <- strenv$jnp$take(params$E_party, party_idx, axis = 0L)
      party_tok <- strenv$jnp$reshape(party_tok, list(tokens$shape[[1]], 1L, ModelDims))
      party_tok <- party_tok + role_vec
      strenv$jnp$concatenate(list(tokens, party_tok), axis = 1L)
    }

    run_transformer <- function(tokens) {
      for (l_ in 1L:ModelDepth) {
        Wq <- params[[paste0("W_q_l", l_)]]
        Wk <- params[[paste0("W_k_l", l_)]]
        Wv <- params[[paste0("W_v_l", l_)]]
        Wo <- params[[paste0("W_o_l", l_)]]
        Wff1 <- params[[paste0("W_ff1_l", l_)]]
        Wff2 <- params[[paste0("W_ff2_l", l_)]]
        RMS_attn <- params[[paste0("RMS_attn_l", l_)]]
        RMS_ff <- params[[paste0("RMS_ff_l", l_)]]

        tokens_norm <- rms_norm(tokens, RMS_attn)
        Q <- strenv$jnp$einsum("ntm,mm->ntm", tokens_norm, Wq)
        K <- strenv$jnp$einsum("ntm,mm->ntm", tokens_norm, Wk)
        V <- strenv$jnp$einsum("ntm,mm->ntm", tokens_norm, Wv)

        Qh <- strenv$jnp$reshape(Q, list(Q$shape[[1]], Q$shape[[2]], TransformerHeads, head_dim))
        Kh <- strenv$jnp$reshape(K, list(K$shape[[1]], K$shape[[2]], TransformerHeads, head_dim))
        Vh <- strenv$jnp$reshape(V, list(V$shape[[1]], V$shape[[2]], TransformerHeads, head_dim))

        scale_ <- strenv$jnp$sqrt(strenv$jnp$array(as.numeric(head_dim)))
        scores <- strenv$jnp$einsum("nqhd,nkhd->nhqk", Qh, Kh) / scale_
        attn <- strenv$jax$nn$softmax(scores, axis = -1L)
        context_h <- strenv$jnp$einsum("nhqk,nkhd->nqhd", attn, Vh)
        context <- strenv$jnp$reshape(context_h, list(context_h$shape[[1]], context_h$shape[[2]], ModelDims))
        attn_out <- strenv$jnp$einsum("ntm,mm->ntm", context, Wo)

        h1 <- tokens + attn_out
        h1_norm <- rms_norm(h1, RMS_ff)
        ff_pre <- strenv$jnp$einsum("ntm,mf->ntf", h1_norm, Wff1)
        ff_act <- strenv$jax$nn$swish(ff_pre)
        ff_out <- strenv$jnp$einsum("ntf,fm->ntm", ff_act, Wff2)
        tokens <- h1 + ff_out
      }
      tokens
    }

    tok_left <- embed_candidate(Xl, pl, 0L)
    tok_right <- embed_candidate(Xr, pr, 1L)
    resp_tokens <- list()
    resp_party_tok <- strenv$jnp$take(params$E_resp_party, resp_p, axis = 0L)
    resp_party_tok <- strenv$jnp$reshape(resp_party_tok, list(tok_left$shape[[1]], 1L, ModelDims))
    resp_tokens[[length(resp_tokens) + 1L]] <- resp_party_tok
    if (!is.null(params$W_resp_x) && ai(resp_c$shape[[2]]) > 0L) {
      resp_cov_tok <- strenv$jnp$einsum("nc,cm->nm", resp_c, params$W_resp_x)
      resp_cov_tok <- strenv$jnp$reshape(resp_cov_tok, list(tok_left$shape[[1]], 1L, ModelDims))
      resp_tokens[[length(resp_tokens) + 1L]] <- resp_cov_tok
    }
    tokens <- strenv$jnp$concatenate(c(list(tok_left, tok_right), resp_tokens), axis = 1L)
    tokens <- run_transformer(tokens)

    n_tok <- ai(Xl$shape[[2]] + 1L)
    tok_left_out <- strenv$jnp$take(tokens, strenv$jnp$arange(n_tok), axis = 1L)
    tok_right_out <- strenv$jnp$take(tokens, strenv$jnp$arange(n_tok, 2L * n_tok), axis = 1L)
    h_left <- strenv$jnp$mean(tok_left_out, axis = 1L)
    h_right <- strenv$jnp$mean(tok_right_out, axis = 1L)
    h_diff <- h_left - h_right

    logits <- strenv$jnp$einsum("nm,mo->no", h_diff, params$W_out) + params$b_out
    if (return_logits) {
      return(logits)
    }
    if (likelihood == "bernoulli") {
      return(strenv$jax$nn$sigmoid(strenv$jnp$squeeze(logits, axis = 1L)))
    }
    if (likelihood == "categorical") {
      return(strenv$jax$nn$softmax(logits, axis = -1L))
    }
    if (likelihood == "normal") {
      return(list(mu = strenv$jnp$squeeze(logits, axis = 1L),
                  sigma = if (!is.null(params$sigma)) params$sigma else strenv$jnp$array(1.)))
    }
  }

  TransformerPredict_single <- function(params, X_new, party_new,
                                        resp_party_new = NULL, resp_cov_new = NULL,
                                        return_logits = FALSE) {
    Xb <- strenv$jnp$array(to_index_matrix(X_new))$astype(strenv$jnp$int32)
    pb <- strenv$jnp$array(as.integer(party_new))$astype(strenv$jnp$int32)
    if (is.null(resp_party_new)) {
      resp_party_new <- rep(0L, nrow(X_new))
    }
    resp_p <- strenv$jnp$array(as.integer(resp_party_new))$astype(strenv$jnp$int32)
    if (is.null(resp_cov_new)) {
      resp_cov_new <- if (!is.null(resp_cov_mean)) {
        matrix(rep(resp_cov_mean, each = nrow(X_new)), nrow = nrow(X_new))
      } else {
        matrix(0, nrow = nrow(X_new), ncol = 0L)
      }
    }
    resp_c <- strenv$jnp$array(as.matrix(resp_cov_new))$astype(ddtype_)

    embed_candidate <- function(X_idx, party_idx) {
      D_local <- ai(X_idx$shape[[2]])
      token_list <- vector("list", D_local)
      for (d_ in 1L:D_local) {
        E_d <- params[[paste0("E_factor_", d_)]]
        idx_d <- strenv$jnp$take(X_idx, ai(d_ - 1L), axis = 1L)
        token_list[[d_]] <- strenv$jnp$take(E_d, idx_d, axis = 0L)
      }
      tokens <- strenv$jnp$stack(token_list, axis = 1L)
      role_vec <- strenv$jnp$take(params$E_role, strenv$jnp$array(ai(0L)), axis = 0L)
      role_vec <- strenv$jnp$reshape(role_vec, list(1L, 1L, ModelDims))
      tokens <- tokens + role_vec
      party_tok <- strenv$jnp$take(params$E_party, party_idx, axis = 0L)
      party_tok <- strenv$jnp$reshape(party_tok, list(tokens$shape[[1]], 1L, ModelDims))
      party_tok <- party_tok + role_vec
      strenv$jnp$concatenate(list(tokens, party_tok), axis = 1L)
    }

    run_transformer <- function(tokens) {
      for (l_ in 1L:ModelDepth) {
        Wq <- params[[paste0("W_q_l", l_)]]
        Wk <- params[[paste0("W_k_l", l_)]]
        Wv <- params[[paste0("W_v_l", l_)]]
        Wo <- params[[paste0("W_o_l", l_)]]
        Wff1 <- params[[paste0("W_ff1_l", l_)]]
        Wff2 <- params[[paste0("W_ff2_l", l_)]]
        RMS_attn <- params[[paste0("RMS_attn_l", l_)]]
        RMS_ff <- params[[paste0("RMS_ff_l", l_)]]

        tokens_norm <- rms_norm(tokens, RMS_attn)
        Q <- strenv$jnp$einsum("ntm,mm->ntm", tokens_norm, Wq)
        K <- strenv$jnp$einsum("ntm,mm->ntm", tokens_norm, Wk)
        V <- strenv$jnp$einsum("ntm,mm->ntm", tokens_norm, Wv)

        Qh <- strenv$jnp$reshape(Q, list(Q$shape[[1]], Q$shape[[2]], TransformerHeads, head_dim))
        Kh <- strenv$jnp$reshape(K, list(K$shape[[1]], K$shape[[2]], TransformerHeads, head_dim))
        Vh <- strenv$jnp$reshape(V, list(V$shape[[1]], V$shape[[2]], TransformerHeads, head_dim))

        scale_ <- strenv$jnp$sqrt(strenv$jnp$array(as.numeric(head_dim)))
        scores <- strenv$jnp$einsum("nqhd,nkhd->nhqk", Qh, Kh) / scale_
        attn <- strenv$jax$nn$softmax(scores, axis = -1L)
        context_h <- strenv$jnp$einsum("nhqk,nkhd->nqhd", attn, Vh)
        context <- strenv$jnp$reshape(context_h, list(context_h$shape[[1]], context_h$shape[[2]], ModelDims))
        attn_out <- strenv$jnp$einsum("ntm,mm->ntm", context, Wo)

        h1 <- tokens + attn_out
        h1_norm <- rms_norm(h1, RMS_ff)
        ff_pre <- strenv$jnp$einsum("ntm,mf->ntf", h1_norm, Wff1)
        ff_act <- strenv$jax$nn$swish(ff_pre)
        ff_out <- strenv$jnp$einsum("ntf,fm->ntm", ff_act, Wff2)
        tokens <- h1 + ff_out
      }
      tokens
    }

    tokens <- embed_candidate(Xb, pb)
    resp_tokens <- list()
    resp_party_tok <- strenv$jnp$take(params$E_resp_party, resp_p, axis = 0L)
    resp_party_tok <- strenv$jnp$reshape(resp_party_tok, list(tokens$shape[[1]], 1L, ModelDims))
    resp_tokens[[length(resp_tokens) + 1L]] <- resp_party_tok
    if (!is.null(params$W_resp_x) && ai(resp_c$shape[[2]]) > 0L) {
      resp_cov_tok <- strenv$jnp$einsum("nc,cm->nm", resp_c, params$W_resp_x)
      resp_cov_tok <- strenv$jnp$reshape(resp_cov_tok, list(tokens$shape[[1]], 1L, ModelDims))
      resp_tokens[[length(resp_tokens) + 1L]] <- resp_cov_tok
    }
    tokens <- strenv$jnp$concatenate(c(list(tokens), resp_tokens), axis = 1L)
    tokens <- run_transformer(tokens)
    n_tok <- ai(Xb$shape[[2]] + 1L)
    tok_out <- strenv$jnp$take(tokens, strenv$jnp$arange(n_tok), axis = 1L)
    h <- strenv$jnp$mean(tok_out, axis = 1L)
    logits <- strenv$jnp$einsum("nm,mo->no", h, params$W_out) + params$b_out

    if (return_logits) {
      return(logits)
    }
    if (likelihood == "bernoulli") {
      return(strenv$jax$nn$sigmoid(strenv$jnp$squeeze(logits, axis = 1L)))
    }
    if (likelihood == "categorical") {
      return(strenv$jax$nn$softmax(logits, axis = -1L))
    }
    if (likelihood == "normal") {
      return(list(mu = strenv$jnp$squeeze(logits, axis = 1L),
                  sigma = if (!is.null(params$sigma)) params$sigma else strenv$jnp$array(1.)))
    }
  }

  coerce_party_idx <- function(party_vec, n_rows) {
    if (is.null(party_vec)) {
      return(rep(0L, n_rows))
    }
    if (is.numeric(party_vec)) {
      idx <- as.integer(party_vec)
      if (any(idx >= n_party_levels)) {
        idx <- idx - 1L
      }
    } else {
      idx <- match(as.character(party_vec), party_levels) - 1L
    }
    idx[is.na(idx)] <- 0L
    idx
  }
  coerce_resp_party_idx <- function(party_vec, n_rows) {
    if (is.null(party_vec)) {
      return(rep(0L, n_rows))
    }
    if (is.numeric(party_vec)) {
      idx <- as.integer(party_vec)
      if (any(idx >= n_resp_party_levels)) {
        idx <- idx - 1L
      }
    } else {
      idx <- match(as.character(party_vec), resp_party_levels) - 1L
    }
    idx[is.na(idx)] <- 0L
    idx
  }

  my_model <- function(...) {
    args <- list(...)
    if (pairwise_mode) {
      X_left_new <- args$X_left_new
      X_right_new <- args$X_right_new
      party_left_new <- args$party_left_new
      party_right_new <- args$party_right_new
      resp_party_new <- args$resp_party_new
      resp_cov_new <- args$resp_cov_new

      if (is.null(X_left_new) || is.null(X_right_new)) {
        if (length(args) < 2L) {
          stop("pairwise my_model requires X_left_new and X_right_new.", call. = FALSE)
        }
        if (is.null(X_left_new)) X_left_new <- args[[1]]
        if (is.null(X_right_new)) X_right_new <- args[[2]]
        if (length(args) >= 3L && is.null(party_left_new)) party_left_new <- args[[3]]
        if (length(args) >= 4L && is.null(party_right_new)) party_right_new <- args[[4]]
        if (length(args) >= 5L && is.null(resp_party_new)) resp_party_new <- args[[5]]
        if (length(args) >= 6L && is.null(resp_cov_new)) resp_cov_new <- args[[6]]
      }

      party_left_new <- coerce_party_idx(party_left_new, nrow(X_left_new))
      party_right_new <- coerce_party_idx(party_right_new, nrow(X_right_new))
      resp_party_new <- coerce_resp_party_idx(resp_party_new, nrow(X_left_new))
      return(TransformerPredict_pair(ParamsMean, X_left_new, X_right_new,
                                     party_left_new, party_right_new,
                                     resp_party_new, resp_cov_new))
    }

    X_new <- args$X_new
    party_new <- args$party_new
    resp_party_new <- args$resp_party_new
    resp_cov_new <- args$resp_cov_new

    if (is.null(X_new)) {
      if (!is.null(args$X_left_new)) {
        X_new <- args$X_left_new
        if (is.null(party_new) && !is.null(args$party_left_new)) {
          party_new <- args$party_left_new
        }
      } else if (length(args) >= 1L) {
        X_new <- args[[1]]
        if (length(args) >= 2L && is.null(party_new)) party_new <- args[[2]]
        if (length(args) >= 3L && is.null(resp_party_new)) resp_party_new <- args[[3]]
        if (length(args) >= 4L && is.null(resp_cov_new)) resp_cov_new <- args[[4]]
      }
    }

    if (is.null(X_new)) {
      stop("my_model requires X_new for single-candidate predictions.", call. = FALSE)
    }
    party_new <- coerce_party_idx(party_new, nrow(X_new))
    resp_party_new <- coerce_resp_party_idx(resp_party_new, nrow(X_new))
    TransformerPredict_single(ParamsMean, X_new, party_new, resp_party_new, resp_cov_new)
  }

  # Neural parameter vector and diagonal posterior covariance
  param_names <- c(paste0("E_factor_", seq_len(length(factor_levels))),
                   "E_party", "E_role", "E_resp_party")
  if (n_resp_covariates > 0L && !is.null(ParamsMean$W_resp_x)) {
    param_names <- c(param_names, "W_resp_x")
  }
  for (l_ in 1L:ModelDepth) {
    param_names <- c(param_names,
                     paste0("RMS_attn_l", l_),
                     paste0("RMS_ff_l", l_),
                     paste0("W_q_l", l_),
                     paste0("W_k_l", l_),
                     paste0("W_v_l", l_),
                     paste0("W_o_l", l_),
                     paste0("W_ff1_l", l_),
                     paste0("W_ff2_l", l_))
  }
  param_names <- c(param_names, "W_out", "b_out")
  if (likelihood == "normal") {
    param_names <- c(param_names, "sigma")
  }
  param_names <- param_names[param_names %in% names(ParamsMean)]

  param_shapes <- lapply(param_names, function(name) {
    shape <- tryCatch(reticulate::py_to_r(ParamsMean[[name]]$shape), error = function(e) NULL)
    if (is.null(shape)) integer(0) else as.integer(shape)
  })
  param_sizes <- vapply(param_shapes, function(shape) {
    if (length(shape) == 0L) {
      1L
    } else {
      as.integer(prod(shape))
    }
  }, integer(1))
  param_offsets <- as.integer(cumsum(c(0L, param_sizes))[seq_len(length(param_sizes))])
  param_total <- sum(param_sizes)

  flatten_params <- function(params) {
    parts <- lapply(param_names, function(name) {
      strenv$jnp$ravel(params[[name]])
    })
    if (length(parts) == 0L) {
      return(strenv$jnp$array(numeric(0), dtype = ddtype_))
    }
    strenv$jnp$concatenate(parts, axis = 0L)
  }
  theta_mean <- flatten_params(ParamsMean)
  theta_mean_num <- as.numeric(strenv$np$array(theta_mean))

  var_parts <- lapply(seq_along(param_names), function(i_) {
    name <- param_names[[i_]]
    draws <- PosteriorDraws[[name]]
    if (is.null(draws)) {
      return(strenv$jnp$zeros(list(ai(param_sizes[[i_]])), dtype = ddtype_))
    }
    strenv$jnp$ravel(strenv$jnp$var(draws, 0L:1L))
  })
  param_var_vec <- if (length(var_parts) == 0L) {
    strenv$jnp$array(numeric(0), dtype = ddtype_)
  } else {
    strenv$jnp$concatenate(var_parts, axis = 0L)
  }
  param_var <- as.numeric(strenv$np$array(param_var_vec))
  if (length(param_var) < param_total) {
    param_var <- c(param_var, rep(0, param_total - length(param_var)))
  }
  if (length(param_var) > param_total) {
    param_var <- param_var[seq_len(param_total)]
  }
  if (uncertainty_scope == "output") {
    keep <- param_names %in% c("W_out", "b_out", "sigma")
    mask <- unlist(mapply(function(keep_i, size_i) rep(keep_i, size_i),
                          keep, param_sizes))
    if (length(mask) == length(param_var)) {
      param_var <- param_var * as.numeric(mask)
    }
  }
  vcov_OutcomeModel <- c(0, param_var)
  vcov_OutcomeModel_by_k <- NULL

  EST_INTERCEPT_tf <- strenv$jnp$array(matrix(0, nrow = 1L, ncol = 1L), dtype = strenv$dtj)
  EST_COEFFICIENTS_tf <- strenv$jnp$reshape(theta_mean, list(-1L, 1L))
  my_mean <- numeric(0)
  my_mean_full <- NULL
  if (exists("K", inherits = TRUE) && is.numeric(K) && K > 1) {
    base_vec <- c(0, theta_mean_num)
    my_mean_full <- matrix(rep(base_vec, K), ncol = K)
    vcov_OutcomeModel_by_k <- replicate(K, vcov_OutcomeModel, simplify = FALSE)
  }

  resp_cov_mean_jnp <- if (!is.null(resp_cov_mean)) {
    strenv$jnp$array(as.numeric(resp_cov_mean))$astype(ddtype_)
  } else {
    NULL
  }
  party_index_map <- NULL
  if (exists("GroupsPool", inherits = TRUE) && length(GroupsPool) > 0) {
    party_index_map <- setNames(vapply(GroupsPool, function(grp) {
      idx <- match(as.character(grp), party_levels) - 1L
      if (is.na(idx)) 0L else idx
    }, integer(1)), GroupsPool)
  }
  resp_party_index_map <- NULL
  if (exists("GroupsPool", inherits = TRUE) && length(GroupsPool) > 0) {
    resp_party_index_map <- setNames(vapply(GroupsPool, function(grp) {
      idx <- match(as.character(grp), resp_party_levels) - 1L
      if (is.na(idx)) 0L else idx
    }, integer(1)), GroupsPool)
  }

  fit_metrics <- NULL
  if (isTRUE(eval_control$enabled)) {
    n_total <- length(Y_use)
    if (n_total > 0L) {
      eval_idx <- seq_len(n_total)
      if (!is.null(eval_control$max_n) &&
          is.finite(eval_control$max_n) &&
          eval_control$max_n > 0L &&
          eval_control$max_n < n_total) {
        eval_seed <- eval_control$seed
        if (is.null(eval_seed) || is.na(eval_seed)) {
          eval_seed <- 123L
        }
        rng <- strenv$np$random$default_rng(as.integer(eval_seed))
        idx_py <- rng$choice(as.integer(n_total),
                             size = as.integer(eval_control$max_n),
                             replace = FALSE)
        eval_idx <- as.integer(reticulate::py_to_r(idx_py)) + 1L
      }

      to_numeric <- function(x) {
        as.numeric(strenv$np$array(x))
      }
      compute_auc <- function(y_true, y_score) {
        y_true <- as.numeric(y_true)
        y_score <- as.numeric(y_score)
        ok <- is.finite(y_true) & is.finite(y_score)
        y_true <- y_true[ok]
        y_score <- y_score[ok]
        if (!length(y_true)) return(NA_real_)
        pos <- y_true == 1
        neg <- y_true == 0
        n_pos <- sum(pos)
        n_neg <- sum(neg)
        if (n_pos == 0L || n_neg == 0L) return(NA_real_)
        ranks <- rank(y_score, ties.method = "average")
        (sum(ranks[pos]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
      }
      compute_log_loss <- function(y_true, y_score, eps = 1e-12) {
        y_true <- as.numeric(y_true)
        y_score <- as.numeric(y_score)
        ok <- is.finite(y_true) & is.finite(y_score)
        y_true <- y_true[ok]
        y_score <- y_score[ok]
        if (!length(y_true)) return(NA_real_)
        p <- pmin(pmax(y_score, eps), 1 - eps)
        -mean(y_true * log(p) + (1 - y_true) * log(1 - p))
      }
      compute_multiclass_log_loss <- function(y_true, prob_mat, eps = 1e-12) {
        if (is.null(dim(prob_mat))) {
          prob_mat <- matrix(prob_mat, nrow = length(y_true), byrow = TRUE)
        }
        n_eval <- nrow(prob_mat)
        if (length(y_true) != n_eval) return(NA_real_)
        ok <- !is.na(y_true)
        y_true <- y_true[ok]
        prob_mat <- prob_mat[ok, , drop = FALSE]
        if (!length(y_true)) return(NA_real_)
        idx <- cbind(seq_along(y_true), y_true + 1L)
        p <- prob_mat[idx]
        p <- pmin(pmax(p, eps), 1 - eps)
        -mean(log(p))
      }
      format_metric <- function(label, value, digits = 4) {
        if (is.null(value) || !is.finite(value)) return(NULL)
        fmt <- paste0("%s=%.", digits, "f")
        sprintf(fmt, label, value)
      }

      if (pairwise_mode) {
        X_left_eval <- X_left[eval_idx, , drop = FALSE]
        X_right_eval <- X_right[eval_idx, , drop = FALSE]
        party_left_eval <- party_left[eval_idx]
        party_right_eval <- party_right[eval_idx]
        resp_party_eval <- resp_party_use[eval_idx]
        resp_cov_eval <- if (!is.null(X_use) && n_resp_covariates > 0L) {
          X_use[eval_idx, , drop = FALSE]
        } else {
          NULL
        }
        pred <- TransformerPredict_pair(ParamsMean, X_left_eval, X_right_eval,
                                        party_left_eval, party_right_eval,
                                        resp_party_eval, resp_cov_eval)
      } else {
        X_eval <- X_single[eval_idx, , drop = FALSE]
        party_eval <- party_single[eval_idx]
        resp_party_eval <- resp_party_use[eval_idx]
        resp_cov_eval <- if (!is.null(X_use) && n_resp_covariates > 0L) {
          X_use[eval_idx, , drop = FALSE]
        } else {
          NULL
        }
        pred <- TransformerPredict_single(ParamsMean, X_eval, party_eval,
                                          resp_party_eval, resp_cov_eval)
      }

      y_eval <- Y_use[eval_idx]
      n_eval <- length(y_eval)
      subset_note <- if (n_eval < n_total) {
        sprintf("n=%d/%d", n_eval, n_total)
      } else {
        sprintf("n=%d", n_eval)
      }

      if (likelihood == "bernoulli") {
        y_eval <- as.numeric(y_eval)
        keep <- is.finite(y_eval) & (y_eval %in% c(0, 1))
        y_eval <- y_eval[keep]
        prob <- to_numeric(pred)
        prob <- prob[keep]
        auc <- compute_auc(y_eval, prob)
        log_loss <- compute_log_loss(y_eval, prob)
        accuracy <- if (length(y_eval)) mean((prob >= 0.5) == y_eval) else NA_real_
        brier <- if (length(y_eval)) mean((prob - y_eval) ^ 2) else NA_real_
        fit_metrics <- list(
          likelihood = likelihood,
          n_eval = length(y_eval),
          auc = auc,
          log_loss = log_loss,
          accuracy = accuracy,
          brier = brier,
          eval_note = subset_note
        )
        metric_items <- Filter(Negate(is.null), list(
          format_metric("AUC", auc, 4),
          format_metric("LogLoss", log_loss, 4),
          format_metric("Acc", accuracy, 3),
          format_metric("Brier", brier, 4)
        ))
      } else if (likelihood == "categorical") {
        y_eval <- as.integer(as.factor(y_eval)) - 1L
        prob_mat <- reticulate::py_to_r(strenv$np$array(pred))
        prob_mat <- as.matrix(prob_mat)
        keep <- !is.na(y_eval)
        y_eval <- y_eval[keep]
        prob_mat <- prob_mat[keep, , drop = FALSE]
        if (length(y_eval)) {
          log_loss <- compute_multiclass_log_loss(y_eval, prob_mat)
          pred_class <- max.col(prob_mat) - 1L
          accuracy <- mean(pred_class == y_eval, na.rm = TRUE)
        } else {
          log_loss <- NA_real_
          accuracy <- NA_real_
        }
        fit_metrics <- list(
          likelihood = likelihood,
          n_eval = length(y_eval),
          log_loss = log_loss,
          accuracy = accuracy,
          eval_note = subset_note
        )
        metric_items <- Filter(Negate(is.null), list(
          format_metric("LogLoss", log_loss, 4),
          format_metric("Acc", accuracy, 3)
        ))
      } else {
        y_eval <- as.numeric(y_eval)
        keep <- is.finite(y_eval)
        y_eval <- y_eval[keep]
        mu <- to_numeric(pred$mu)
        mu <- mu[keep]
        sigma <- to_numeric(pred$sigma)
        if (length(sigma) == 1L && length(y_eval) > 1L) {
          sigma <- rep(sigma, length(y_eval))
        } else {
          sigma <- sigma[keep]
        }
        rmse <- if (length(y_eval)) sqrt(mean((mu - y_eval) ^ 2)) else NA_real_
        mae <- if (length(y_eval)) mean(abs(mu - y_eval)) else NA_real_
        nll <- NA_real_
        if (length(y_eval) && length(sigma) == length(y_eval) && all(is.finite(sigma)) &&
            all(sigma > 0)) {
          nll <- mean(0.5 * log(2 * pi * sigma ^ 2) + (y_eval - mu) ^ 2 / (2 * sigma ^ 2))
        }
        fit_metrics <- list(
          likelihood = likelihood,
          n_eval = length(y_eval),
          rmse = rmse,
          mae = mae,
          nll = nll,
          eval_note = subset_note
        )
        metric_items <- Filter(Negate(is.null), list(
          format_metric("RMSE", rmse, 4),
          format_metric("MAE", mae, 4),
          format_metric("NLL", nll, 4)
        ))
      }

      if (!is.null(fit_metrics) && length(metric_items) > 0L) {
        message(sprintf("Neural fit metrics (%s, %s): %s",
                        ifelse(pairwise_mode, "pairwise", "single"),
                        subset_note,
                        paste(metric_items, collapse = ", ")))
      }
    }
  }

  neural_model_info <- list(
    params = ParamsMean,
    param_names = param_names,
    param_shapes = param_shapes,
    param_sizes = param_sizes,
    param_offsets = param_offsets,
    n_params = ai(param_total),
    uncertainty_scope = uncertainty_scope,
    factor_levels = factor_levels,
    factor_index_list = factor_index_list,
    implicit = isTRUE(holdout_indicator == 1L),
    n_factors = ai(length(factor_levels)),
    n_candidate_tokens = n_candidate_tokens,
    party_levels = party_levels,
    resp_party_levels = resp_party_levels,
    party_index_map = party_index_map,
    resp_party_index_map = resp_party_index_map,
    resp_cov_mean = resp_cov_mean_jnp,
    n_resp_covariates = n_resp_covariates,
    likelihood = likelihood,
    fit_metrics = fit_metrics,
    model_dims = ModelDims,
    model_depth = ModelDepth,
    n_heads = TransformerHeads,
    head_dim = head_dim
  )

  message(sprintf("Bayesian Transformer complete. Pairwise=%s, Heads=%d, Depth=%d, Hidden=%d; likelihood=%s.",
                  pairwise_mode, TransformerHeads, ModelDepth, MD_int, likelihood))
}
