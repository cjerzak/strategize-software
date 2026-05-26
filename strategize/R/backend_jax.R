strategize_jax_cache_dir <- function() {
  cache_dir <- Sys.getenv("STRATEGIZE_JAX_CACHE_DIR", unset = "")
  if (!nzchar(cache_dir)) {
    cache_dir <- file.path("~", ".cache", "strategize", "jax_compilation_cache")
  }
  path.expand(cache_dir)
}

strategize_configure_jax_compilation_cache <- function(jax = NULL) {
  if (is.null(jax)) {
    jax <- strenv$jax
  }
  if (is.null(jax)) {
    return(invisible(FALSE))
  }

  cache_dir <- strategize_jax_cache_dir()
  dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
  if (!dir.exists(cache_dir)) {
    return(invisible(FALSE))
  }

  configured <- FALSE
  tryCatch({
    jax$config$update("jax_compilation_cache_dir", cache_dir)
    configured <- TRUE
  }, error = function(e) NULL)
  tryCatch({
    jax$config$update("jax_persistent_cache_min_compile_time_secs", 0)
  }, error = function(e) NULL)

  if (!isTRUE(configured)) {
    tryCatch({
      cache_mod <- reticulate::import(
        "jax.experimental.compilation_cache.compilation_cache",
        delay_load = TRUE
      )
      cache_mod$set_cache_dir(cache_dir)
      configured <- TRUE
    }, error = function(e) NULL)
  }

  strenv$jax_compilation_cache_dir <- cache_dir
  invisible(isTRUE(configured))
}

strategize_jax_block_until_ready <- function(x, max_depth = 20L) {
  walk <- function(value, depth) {
    if (is.null(value) || depth > max_depth) {
      return(invisible(value))
    }

    blocked <- tryCatch({
      block_fn <- value$block_until_ready
      if (!is.null(block_fn)) {
        block_fn()
        TRUE
      } else {
        FALSE
      }
    }, error = function(e) FALSE)
    if (isTRUE(blocked)) {
      return(invisible(value))
    }

    if (is.list(value) && !is.data.frame(value)) {
      for (item in value) {
        walk(item, depth + 1L)
      }
      return(invisible(value))
    }

    if (requireNamespace("reticulate", quietly = TRUE)) {
      is_py <- tryCatch(reticulate::is_py_object(value), error = function(e) FALSE)
      if (isTRUE(is_py)) {
        parts <- tryCatch(as.list(value), error = function(e) NULL)
        if (is.list(parts) && length(parts) > 0L) {
          for (item in parts) {
            walk(item, depth + 1L)
          }
        }
      }
    }

    invisible(value)
  }

  walk(x, 0L)
  invisible(x)
}

strategize_register_jax_transformer_helpers <- function() {
  if (!is.null(strenv$jax_transformer_scan_standard)) {
    return(invisible(TRUE))
  }
  helper_code <- paste(
    "import jax",
    "import jax.numpy as jnp",
    "",
    "def _strategize_rms_norm(x, g, eps=1e-6):",
    "    if g is None:",
    "        return x",
    "    g = jnp.asarray(g, dtype=x.dtype)",
    "    mean_sq = jnp.mean(x * x, axis=-1, keepdims=True)",
    "    inv_rms = jnp.reciprocal(jnp.sqrt(mean_sq + eps))",
    "    return x * inv_rms * g",
    "",
    "def strategize_transformer_scan_standard(",
    "    tokens, token_mask, W_q_layers, W_k_layers, W_v_layers, W_o_layers,",
    "    W_ff1_layers, W_ff2_layers, RMS_attn_layers, RMS_ff_layers,",
    "    RMS_q_layers, RMS_k_layers, alpha_attn_layers, alpha_ff_layers,",
    "    RMS_final, model_dims, n_heads, head_dim",
    "):",
    "    model_dims = int(model_dims)",
    "    n_heads = int(n_heads)",
    "    head_dim = int(head_dim)",
    "    use_qk_norm = RMS_q_layers is not None and RMS_k_layers is not None",
    "    if not use_qk_norm:",
    "        RMS_q_layers = jnp.zeros((W_q_layers.shape[0], head_dim), dtype=tokens.dtype)",
    "        RMS_k_layers = jnp.zeros((W_q_layers.shape[0], head_dim), dtype=tokens.dtype)",
    "",
    "    def body(tokens_carry, layer):",
    "        (Wq, Wk, Wv, Wo, Wff1, Wff2, RMS_attn, RMS_ff,",
    "         RMS_q, RMS_k, alpha_attn, alpha_ff) = layer",
    "        tokens_norm = _strategize_rms_norm(tokens_carry, RMS_attn)",
    "        Q = jnp.einsum('ntm,mk->ntk', tokens_norm, Wq)",
    "        K = jnp.einsum('ntm,mk->ntk', tokens_norm, Wk)",
    "        V = jnp.einsum('ntm,mk->ntk', tokens_norm, Wv)",
    "        Qh = jnp.reshape(Q, (Q.shape[0], Q.shape[1], n_heads, head_dim))",
    "        Kh = jnp.reshape(K, (K.shape[0], K.shape[1], n_heads, head_dim))",
    "        Vh = jnp.reshape(V, (V.shape[0], V.shape[1], n_heads, head_dim))",
    "        if use_qk_norm:",
    "            Qh = _strategize_rms_norm(Qh, RMS_q)",
    "            Kh = _strategize_rms_norm(Kh, RMS_k)",
    "        scale = jnp.sqrt(jnp.asarray(float(head_dim), dtype=tokens_carry.dtype))",
    "        scores = jnp.einsum('nqhd,nkhd->nhqk', Qh, Kh) / scale",
    "        if token_mask is not None:",
    "            mask_use = jnp.reshape((token_mask > 0).astype(scores.dtype),",
    "                                   (token_mask.shape[0], 1, 1, token_mask.shape[1]))",
    "            scores = jnp.where(mask_use > 0, scores, jnp.asarray(-1e9, dtype=scores.dtype))",
    "        attn = jax.nn.softmax(scores, axis=-1)",
    "        context_h = jnp.einsum('nhqk,nkhd->nqhd', attn, Vh)",
    "        context = jnp.reshape(context_h, (context_h.shape[0], context_h.shape[1], model_dims))",
    "        attn_out = jnp.einsum('ntm,mk->ntk', context, Wo)",
    "        h1 = tokens_carry + alpha_attn * attn_out",
    "        h1_norm = _strategize_rms_norm(h1, RMS_ff)",
    "        ff_pre = jnp.einsum('ntm,mf->ntf', h1_norm, Wff1)",
    "        ff_act = jax.nn.swish(ff_pre)",
    "        ff_out = jnp.einsum('ntf,fm->ntm', ff_act, Wff2)",
    "        return h1 + alpha_ff * ff_out, None",
    "",
    "    xs = (W_q_layers, W_k_layers, W_v_layers, W_o_layers,",
    "          W_ff1_layers, W_ff2_layers, RMS_attn_layers, RMS_ff_layers,",
    "          RMS_q_layers, RMS_k_layers, alpha_attn_layers, alpha_ff_layers)",
    "    tokens_out, _ = jax.lax.scan(body, tokens, xs)",
    "    return _strategize_rms_norm(tokens_out, RMS_final)",
    sep = "\n"
  )
  reticulate::py_run_string(helper_code)
  strenv$jax_transformer_scan_standard <- reticulate::py$strategize_transformer_scan_standard
  invisible(TRUE)
}

strategize_register_jax_svi_helpers <- function() {
  if (!is.null(strenv$jax_svi_update_scan)) {
    return(invisible(TRUE))
  }
  helper_code <- paste(
    "import jax",
    "",
    "def strategize_svi_update_scan(svi, svi_state, batch_args_chunks):",
    "    def body(state, step_args):",
    "        state, loss = svi.update(state, **step_args)",
    "        return state, loss",
    "    return jax.lax.scan(body, svi_state, batch_args_chunks)",
    sep = "\n"
  )
  reticulate::py_run_string(helper_code)
  strenv$jax_svi_update_scan <- reticulate::py$strategize_svi_update_scan
  invisible(TRUE)
}

initialize_jax <- function(conda_env = "strategize_env",
                           conda_env_required = TRUE) {
  # reticulate is declared in Imports - use :: syntax
  reticulate::use_condaenv(condaenv = conda_env, required = conda_env_required)
  
  # Import Python packages once, storing them in strenv
  strenv$jax <- reticulate::import("jax")
  strategize_configure_jax_compilation_cache(strenv$jax)
  strenv$jnp <- reticulate::import("jax.numpy")
  strenv$np  <- reticulate::import("numpy")
  strenv$eq  <- reticulate::import("equinox")
  strenv$py_gc  <- reticulate::import("gc")
  strenv$numpyro  <- reticulate::import("numpyro")
  strenv$optax  <- reticulate::import("optax")
  strenv$orbax_checkpoint <- tryCatch(
    reticulate::import("orbax.checkpoint"),
    error = function(e) NULL
  )
  strategize_register_jax_transformer_helpers()
  strategize_register_jax_svi_helpers()
  
  # setup numerical precisions
  strenv$jaxFloatType <- strenv$jnp$float32
  #strenv$dtj <- strenv$jnp$float64; strenv$jax$config$update("jax_enable_x64", TRUE) # use float64
  strenv$dtj <- strenv$jnp$float32; strenv$jax$config$update("jax_enable_x64", FALSE) # use float32
}
strenv <- new.env( parent = emptyenv() )
