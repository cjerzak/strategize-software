generate_ModelOutcome_neural <- function(){
  
  # define MCMC parameters 
  message("Defining MCMC parameters in generate_ModelOutcome_neural...")
  mcmc_control <- list(
                                backend = "numpyro",  # will override to use NumPyro-based MCMC
                                n_samples_warmup = 500L,
                                n_samples_mcmc   = 1000L,
                                batch_size = 512L, 
                                chain_method = "parallel", 
                                subsample_method = "full", 
                                n_thin_by = 1L, 
                                n_chains = 2L)
  
  # Hyperparameters
  ModelDims  <- ai(128L)      # hidden width
  ModelDepth <- ai(2L)                    # transformer blocks
  WideMultiplicationFactor <- 3.75                # SWiGLU expansion factor
  # choose heads that divide ModelDims evenly, near 8
  MD_int <- ai(ModelDims)
  cand_heads <- (1:MD_int)[(MD_int %% (1:MD_int)) == 0L]
  TransformerHeads <- ai(cand_heads[which.min(abs(cand_heads - 8L))])
  head_dim <- ai(ai(MD_int / TransformerHeads))
  FFDim <- ai(ai(round(MD_int * WideMultiplicationFactor)))
  
  # Data and likelihood selection
  N <- nrow(W_); D <- ncol(W_)
  is_binary <- all(unique(na.omit(as.numeric(Y_))) %in% c(0, 1)) && length(unique(Y_)) <= 2
  is_intvec <- all(!is.na(Y_)) && all(abs(Y_ - round(Y_)) < 1e-8)
  K_classes <- if (is_intvec) length(unique(ai(Y_))) else NA_integer_

  if (is_binary) {
    likelihood <- "bernoulli"; nOutcomes <- ai(1L)
  } else if (!is.na(K_classes) && K_classes >= 2L && K_classes <= max(50L, D + 1L)) {
    likelihood <- "categorical"; nOutcomes <- ai(K_classes)
  } else {
    likelihood <- "normal"; nOutcomes <- ai(1L)
  }
  
  
  # Dtypes & JAX numeric config
  pdtype_ <- ddtype_ <- strenv$jnp$float32
  
  # Model definition (Bayesian Transformer)
  BayesianTransformerModel <- function(X, Y_obs) {
    # Shapes
    N_local <- ai(X$shape[[1]])
    D_local <- ai(X$shape[[2]])
    
    # --- Parameter priors (global; outside plates) ---
    # Embedding table: (D, ModelDims). Each scalar feature becomes a token embedding via scalar * row-embedding.
    W_embed <- strenv$numpyro$sample("W_embed",
                              strenv$numpyro$distributions$Normal(0., 1.),
                              sample_shape = reticulate::tuple(D_local, ModelDims))
    
    # Transformer block params
    for (l_ in 1L:ModelDepth) {
      assign(paste0("W_q_l", l_),
            strenv$numpyro$sample(paste0("W_q_l", l_), strenv$numpyro$distributions$Normal(0., 1.),
                     sample_shape = reticulate::tuple(ModelDims, ModelDims)))
      assign(paste0("W_k_l", l_),
            strenv$numpyro$sample(paste0("W_k_l", l_), strenv$numpyro$distributions$Normal(0., 1.),
                         sample_shape = reticulate::tuple(ModelDims, ModelDims)))
      assign(paste0("W_v_l", l_),
            strenv$numpyro$sample(paste0("W_v_l", l_), strenv$numpyro$distributions$Normal(0., 1.),
                           sample_shape = reticulate::tuple(ModelDims, ModelDims)))
      assign(paste0("W_o_l", l_),
            strenv$numpyro$sample(paste0("W_o_l", l_), strenv$numpyro$distributions$Normal(0., 1.),
                           sample_shape = reticulate::tuple(ModelDims, ModelDims)))
      assign(paste0("W_ff1_l", l_),
            strenv$numpyro$sample(paste0("W_ff1_l", l_), strenv$numpyro$distributions$Normal(0., 1.),
                           sample_shape = reticulate::tuple(ModelDims, FFDim)))
      assign(paste0("W_ff2_l", l_),
            strenv$numpyro$sample(paste0("W_ff2_l", l_), strenv$numpyro$distributions$Normal(0., 1.),
                           sample_shape = reticulate::tuple(FFDim, ModelDims)))
    }
    
    # Output head
    W_out <- strenv$numpyro$sample("W_out", strenv$numpyro$distributions$Normal(0., 1.),
                          sample_shape = reticulate::tuple(ModelDims, nOutcomes))
    b_out <- strenv$numpyro$sample("b_out", strenv$numpyro$distributions$Normal(0., 1.), 
                          sample_shape = reticulate::tuple(nOutcomes))
    
    if (likelihood == "normal") {
      sigma <- strenv$numpyro$sample("sigma", strenv$numpyro$distributions$HalfNormal(1.0))
    }
    

    # --- Local likelihood (optionally subsampled) ---
    local_lik <- function() {
      # Optional subsampling index from plate
      if (isTRUE(mcmc_control$subsample_method == "batch")) {
        with(strenv$numpyro$plate("data", size = N_local,
                           subsample_size = ai(mcmc_control$batch_size),
                           dim = -1L) %as% "idx", {
                             X_b <- strenv$jnp$take(X, idx, axis = 0L)
                             Y_b <- strenv$jnp$take(Y_obs, idx, axis = 0L)
                             do_forward_and_lik_(X_b, Y_b)
                           })
      } else {
        with(strenv$numpyro$plate("data", size = N_local, dim = -1L), {
          do_forward_and_lik_(X, Y_obs)
        })
      }
    }
    
    # --- Forward pass + likelihood on a batch ---
    do_forward_and_lik_ <- function(Xb, Yb) {
      # Embed tokens: (N, D, ModelDims) via einsum: [n d] x [d m] -> [n d m]
      embedded <- strenv$jnp$einsum("nd,dm->ndm", Xb, W_embed)
      
      # Transformer blocks
      for (l_ in 1L:ModelDepth) {
        Wq <- get(paste0("W_q_l", l_)); Wk <- get(paste0("W_k_l", l_))
        Wv <- get(paste0("W_v_l", l_)); Wo <- get(paste0("W_o_l", l_))
        Wff1 <- get(paste0("W_ff1_l", l_)); Wff2 <- get(paste0("W_ff2_l", l_))
        
        # Projections: [n d m] x [m m] -> [n d m]
        Q <- strenv$jnp$einsum("ndm,mm->ndm", embedded, Wq)
        K <- strenv$jnp$einsum("ndm,mm->ndm", embedded, Wk)
        V <- strenv$jnp$einsum("ndm,mm->ndm", embedded, Wv)
        
        # Split into heads and compute attention
        # reshape to [n, d, h, hd]
        Qh <- strenv$jnp$reshape(Q, list(Q$shape[[1]], Q$shape[[2]], TransformerHeads, head_dim))
        Kh <- strenv$jnp$reshape(K, list(K$shape[[1]], K$shape[[2]], TransformerHeads, head_dim))
        Vh <- strenv$jnp$reshape(V, list(V$shape[[1]], V$shape[[2]], TransformerHeads, head_dim))
        
        # scores: [n, h, q, k]
        scale_ <- strenv$jnp$sqrt(strenv$jnp$array(as.numeric(head_dim)))
        scores <- strenv$jnp$einsum("nqhd,nkhd->nhqk", Qh, Kh) / scale_
        attn <- strenv$jax$nn$softmax(scores, axis = -1L)
        
        # context back to [n, d, m]
        context_h <- strenv$jnp$einsum("nhqk,nkhd->nqhd", attn, Vh)
        context <- strenv$jnp$reshape(context_h, list(context_h$shape[[1]], context_h$shape[[2]], ModelDims))
        attn_out <- strenv$jnp$einsum("ndm,mm->ndm", context, Wo)
        
        # Residual + FFN (SWiGLU-ish)
        h1 <- embedded + attn_out
        ff_pre <- strenv$jnp$einsum("ndm,mf->ndf", h1, Wff1)
        ff_act <- strenv$jax$nn$swish(ff_pre)
        ff_out <- strenv$jnp$einsum("ndf,fm->ndm", ff_act, Wff2)
        embedded <- h1 + ff_out
      }
      
      # Pool tokens (mean over D): [n, m]
      h <- strenv$jnp$mean(embedded, axis = 1L)
      logits <- strenv$jnp$einsum("nm,mo->no", h, W_out) + b_out
      
      if (likelihood == "bernoulli") {
        logits_vec <- strenv$jnp$squeeze(logits, axis = 1L)
        strenv$numpyro$sample("obs", strenv$numpyro$distributions$Bernoulli(logits = logits_vec), obs = Yb)
      }
      if (likelihood == "categorical") {
        # Yb must be int labels in [0, K-1]
        strenv$numpyro$sample("obs", strenv$numpyro$distributions$Categorical(logits = logits), obs = Yb)
      }
      if (likelihood == "normal") {
        mu <- strenv$jnp$squeeze(logits, axis = 1L)
        strenv$numpyro$sample("obs", strenv$numpyro$distributions$Normal(mu, sigma), obs = Yb)
      }
    }
    
    # run local likelihood
    local_lik()
  }
  
  # NumPyro / MCMC configuration 
  strenv$numpyro$set_host_device_count(mcmc_control$n_chains)  # use all chains’ CPUs if parallel. 
  
  # Kernel
  if ( mcmc_control$subsample_method == "batch") {
    message("Enlisting HMCECS kernels for subsampled likelihood...")
    kernel <- strenv$numpyro$infer$HMCECS(strenv$numpyro$infer$NUTS(BayesianTransformerModel),
                                   num_blocks = if (!is.null(mcmc_control$num_blocks)) ai(mcmc_control$num_blocks) else ai(4L))
  } else {
    message("Enlisting NUTS kernels for full-data likelihood...")
    kernel <- strenv$numpyro$infer$NUTS(BayesianTransformerModel,
                                 max_tree_depth = ai(8L),
                                 target_accept_prob = 0.85)
  }
  
  # MCMC driver
  sampler <- strenv$numpyro$infer$MCMC(
    sampler = kernel,
    num_warmup = mcmc_control$n_samples_warmup,
    num_samples = mcmc_control$n_samples_mcmc,
    thinning   = mcmc_control$n_thin_by,
    chain_method = ifelse(!is.null(mcmc_control$chain_method),yes = mcmc_control$chain_method, no = "parallel"),
    num_chains = mcmc_control$n_chains,
    jit_model_args = TRUE,
    progress_bar = TRUE
  )
  
  # Prepare data tensors
  if (likelihood == "categorical") {
    # map labels to 0..K-1 (NumPyro expects integer class ids)
    y_fac <- ai(as.factor(Y_)) - 1L
    Y_jnp <- strenv$jnp$array(y_fac)$astype(strenv$jnp$int32)
  } else {
    Y_jnp <- strenv$jnp$array(as.numeric(Y_))$astype(ddtype_)
  }
  
  # -------------------------------
  # Run sampling
  # -------------------------------
  browser()
  t0_ <- Sys.time()
  sampler$run(strenv$jax$random$PRNGKey(ai(runif(1, 0, 10000))),
              X = strenv$jnp$array(as.matrix(W_))$astype(ddtype_), 
              Y_obs = Y_jnp)
  PosteriorDraws <- sampler$get_samples(group_by_chain = TRUE)
  message(sprintf("\n MCMC Runtime: %.3f min",
                  as.numeric(difftime(Sys.time(), t0_, units = "secs"))/60))
  
  # -------------------------------
  # Posterior means (averaged over [chain, draw])
  # -------------------------------
  mean_param <- function(x) { strenv$jnp$mean(x, 0L:1L) }
  
  ParamsMean <- list(
    W_embed = mean_param(PosteriorDraws$W_embed),
    W_out   = mean_param(PosteriorDraws$W_out),
    b_out   = mean_param(PosteriorDraws$b_out)
  )
  if (likelihood == "normal") {
    ParamsMean$sigma <- mean_param(PosteriorDraws$sigma)
  }
  for (l_ in 1L:ModelDepth) {
    ParamsMean[[paste0("W_q_l", l_)]]  <- mean_param(PosteriorDraws[[paste0("W_q_l",  l_)]])
    ParamsMean[[paste0("W_k_l", l_)]]  <- mean_param(PosteriorDraws[[paste0("W_k_l",  l_)]])
    ParamsMean[[paste0("W_v_l", l_)]]  <- mean_param(PosteriorDraws[[paste0("W_v_l",  l_)]])
    ParamsMean[[paste0("W_o_l", l_)]]  <- mean_param(PosteriorDraws[[paste0("W_o_l",  l_)]])
    ParamsMean[[paste0("W_ff1_l", l_)]]<- mean_param(PosteriorDraws[[paste0("W_ff1_l",l_)]])
    ParamsMean[[paste0("W_ff2_l", l_)]]<- mean_param(PosteriorDraws[[paste0("W_ff2_l",l_)]])
  }
  
  # -------------------------------
  # Deterministic predictor (deployment)
  # -------------------------------
  TransformerPredict <- function(params, X_new) {
    # X_new: R numeric matrix [N, D]
    Xb <- strenv$jnp$array(as.matrix(X_new))$astype(ddtype_)
    embedded <- strenv$jnp$einsum("nd,dm->ndm", Xb, params$W_embed)
    for (l_ in 1L:ModelDepth) {
      Wq <- params[[paste0("W_q_l", l_)]]; Wk <- params[[paste0("W_k_l", l_)]]
      Wv <- params[[paste0("W_v_l", l_)]]; Wo <- params[[paste0("W_o_l", l_)]]
      Wff1 <- params[[paste0("W_ff1_l", l_)]]; Wff2 <- params[[paste0("W_ff2_l", l_)]]
      
      Q <- strenv$jnp$einsum("ndm,mm->ndm", embedded, Wq)
      K <- strenv$jnp$einsum("ndm,mm->ndm", embedded, Wk)
      V <- strenv$jnp$einsum("ndm,mm->ndm", embedded, Wv)
      
      Qh <- strenv$jnp$reshape(Q, list(Q$shape[[1]], Q$shape[[2]], TransformerHeads, head_dim))
      Kh <- strenv$jnp$reshape(K, list(K$shape[[1]], K$shape[[2]], TransformerHeads, head_dim))
      Vh <- strenv$jnp$reshape(V, list(V$shape[[1]], V$shape[[2]], TransformerHeads, head_dim))
      
      scale_ <- strenv$jnp$sqrt(strenv$jnp$array(as.numeric(head_dim)))
      scores <- strenv$jnp$einsum("nqhd,nkhd->nhqk", Qh, Kh) / scale_
      attn <- strenv$jax$nn$softmax(scores, axis = -1L)
      context_h <- strenv$jnp$einsum("nhqk,nkhd->nqhd", attn, Vh)
      context <- strenv$jnp$reshape(context_h, list(context_h$shape[[1]], context_h$shape[[2]], ModelDims))
      attn_out <- strenv$jnp$einsum("ndm,mm->ndm", context, Wo)
      
      h1 <- embedded + attn_out
      ff_pre <- strenv$jnp$einsum("ndm,mf->ndf", h1, Wff1)
      ff_act <- strenv$jax$nn$swish(ff_pre)
      ff_out <- strenv$jnp$einsum("ndf,fm->ndm", ff_act, Wff2)
      embedded <- h1 + ff_out
    }
    h <- strenv$jnp$mean(embedded, axis = 1L)
    logits <- strenv$jnp$einsum("nm,mo->no", h, params$W_out) + params$b_out
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
  
  # Export objects (match your pattern)
  EST_COEFFICIENTS_tf <- ParamsMean$W_out  # exposed for downstream use
  EST_INTERCEPT_tf <- strenv$jnp$expand_dims(ParamsMean$b_out, 0L)
  my_model <- function(X_new) TransformerPredict(ParamsMean, X_new)

  # placeholders for compatibility with downstream code
  main_info <- data.frame()
  interaction_info <- data.frame()
  interaction_info_PreRegularization <- data.frame()
  regularization_adjust_hash <- numeric(0)
  main_dat <- matrix(0, 0, 0)

  # Compute posterior covariance matrix for W_out (flattened)
  wout_shape <- PosteriorDraws$W_out$shape
  nchains <- wout_shape[[1L]]
  nsamps <- wout_shape[[2L]]
  p1 <- wout_shape[[3L]]
  p2 <- wout_shape[[4L]]
  total_params <- p1 * p2
  wout_flat <- strenv$jnp$reshape(PosteriorDraws$W_out, list(nchains, nsamps, total_params))
  wout_all <- strenv$jnp$reshape(wout_flat, list(nchains * nsamps, total_params))
  EST_VCOV_tf <- strenv$jnp$cov(wout_all, rowvar = FALSE)
  vcov_OutcomeModel <- as.matrix(reticulate::py_to_r(EST_VCOV_tf))
  my_mean <- as.numeric(reticulate::py_to_r(strenv$jnp$reshape(ParamsMean$W_out, -1L)))
  
  message(sprintf("Bayesian Transformer complete. Heads=%d, Depth=%d, Hidden=%d; likelihood=%s.",
                  TransformerHeads, ModelDepth, MD_int, likelihood))
}
