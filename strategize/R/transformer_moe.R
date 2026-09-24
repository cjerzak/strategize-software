# Transformer MoE controls are distinct from the covariate value encoder.
neural_resolve_transformer_moe <- function(control = NULL, model_dims = 128L,
                                         model_depth = 2L, use_svi = TRUE) {
  control <- control %||% list()
  mode <- control$transformer_ffn %||% "auto"
  if (!is.character(mode) || length(mode) != 1L || is.na(mode) ||
      !mode %in% c("auto", "moe", "swiglu")) {
    stop("'transformer_ffn' must be 'auto', 'moe', or 'swiglu'.", call. = FALSE)
  }
  if (identical(mode, "auto")) mode <- if (isTRUE(use_svi)) "moe" else "swiglu"
  if (identical(mode, "moe") && !isTRUE(use_svi)) {
    stop("Transformer MoE requires SVI; use transformer_ffn='swiglu' for full MCMC.", call. = FALSE)
  }
  defaults <- list(n_routed_experts = 8L, n_experts_per_tok = 2L,
                   n_shared_experts = 1L, moe_d_ff = "auto", first_k_dense = "auto",
                   routed_scaling_factor = 1, capacity_factor = 1.5, router_bias_rate = 0.001,
                   compute_dtype = "bfloat16", activation_checkpointing = TRUE)
  overrides <- control$transformer_moe %||% list()
  if (!is.list(overrides) || (length(overrides) &&
      (is.null(names(overrides)) || any(!nzchar(names(overrides))) || anyDuplicated(names(overrides))))) {
    stop("'transformer_moe' must be a named list.", call. = FALSE)
  }
  unknown <- setdiff(names(overrides), names(defaults))
  if (length(unknown)) stop(sprintf("Unknown transformer_moe field(s): %s.", paste(unknown, collapse = ", ")), call. = FALSE)
  cfg <- modifyList(defaults, overrides)
  if (!is.character(cfg$compute_dtype) || length(cfg$compute_dtype) != 1L ||
      is.na(cfg$compute_dtype) || !cfg$compute_dtype %in% c("float32", "bfloat16")) {
    stop("'transformer_moe$compute_dtype' must be 'float32' or 'bfloat16'.", call. = FALSE)
  }
  if (!is.logical(cfg$activation_checkpointing) || length(cfg$activation_checkpointing) != 1L ||
      is.na(cfg$activation_checkpointing)) {
    stop("'transformer_moe$activation_checkpointing' must be TRUE or FALSE.", call. = FALSE)
  }
  if (identical(cfg$moe_d_ff, "auto")) cfg$moe_d_ff <- as.integer(model_dims)
  if (identical(cfg$first_k_dense, "auto")) cfg$first_k_dense <- min(1L, as.integer(model_depth) - 1L)
  for (name in c("n_routed_experts", "n_experts_per_tok", "n_shared_experts", "moe_d_ff", "first_k_dense")) {
    x <- cfg[[name]]
    lower <- if (name == "first_k_dense") 0 else 1
    if (!is.numeric(x) || length(x) != 1L || !is.finite(x) || x < lower || x != floor(x) || x > .Machine$integer.max) {
      stop(sprintf("'transformer_moe$%s' must be an integer >= %d.", name, lower), call. = FALSE)
    }
    cfg[[name]] <- as.integer(x)
  }
  if (cfg$n_experts_per_tok > cfg$n_routed_experts || cfg$first_k_dense >= model_depth) {
    stop("MoE requires top-k <= expert count and first_k_dense < ModelDepth.", call. = FALSE)
  }
  for (name in c("routed_scaling_factor", "capacity_factor", "router_bias_rate")) {
    x <- cfg[[name]]
    if (!is.numeric(x) || length(x) != 1L || !is.finite(x) ||
        (if (name == "router_bias_rate") x < 0 else x <= 0)) {
      stop(sprintf("'transformer_moe$%s' must be a finite %s number.", name,
                   if (name == "router_bias_rate") "nonnegative" else "positive"), call. = FALSE)
    }
  }
  cfg$n_moe_layers <- as.integer(model_depth) - cfg$first_k_dense
  list(transformer_ffn = mode, transformer_moe = if (mode == "moe") cfg else NULL)
}

neural_moe_weight_bases <- function() {
  paste0("W_moe_", c("router", "expert1", "expert2", "shared1", "shared2"), "_l")
}

neural_has_transformer_moe <- function(params) {
  any(grepl("^W_moe_router_(layers|l[0-9]+)$", names(params)))
}

strategize_register_transformer_module <- function() {
  if (is.null(strenv$jax_transformer)) {
    strenv$jax_transformer <- reticulate::import_from_path(
      "strategize_transformer", path = system.file("python", package = "strategize"), convert = TRUE
    )
  }
  invisible(strenv$jax_transformer)
}

strategize_register_moe_helpers <- function() {
  if (is.null(strenv$jax_moe)) {
    strenv$jax_moe <- reticulate::import_from_path(
      "strategize_moe", path = system.file("python", package = "strategize"), convert = TRUE
    )
  }
  invisible(TRUE)
}

neural_moe_config <- function(model_info, params = NULL) {
  cfg <- model_info[["transformer_moe"]] %||% params$transformer_moe_config
  if (is.null(cfg) && !is.null(strenv$jax_moe)) cfg <- strenv$jax_moe$current_config()
  if (is.null(cfg) && neural_has_transformer_moe(params)) {
    stop("MoE parameters require saved transformer_moe architecture metadata.", call. = FALSE)
  }
  cfg
}

neural_moe_set_branch <- function(branch, scale, rows) {
  if (!is.null(strenv$jax_moe)) strenv$jax_moe$set_branch(branch, scale, as.integer(rows))
  if (!is.null(strenv$jax_attention)) strenv$jax_attention$set_branch(branch, scale, as.integer(rows))
  invisible(NULL)
}

neural_moe_attach_params <- function(params, model_info) {
  if (neural_has_transformer_moe(params)) {
    cfg <- neural_moe_config(model_info, params)
    bias <- model_info$transformer_moe_router_bias %||% params$transformer_moe_router_bias
    if (is.null(bias)) stop("A saved MoE model is missing frozen router biases.", call. = FALSE)
    params$transformer_moe_router_bias <- strenv$jnp$asarray(bias, dtype = strenv$jnp$float32)
  }
  params
}

neural_moe_architecture_fields <- function(info, cfg, bias = NULL) {
  info$transformer_ffn <- if (is.null(cfg)) "swiglu" else "moe"
  info$transformer_moe <- cfg
  info$transformer_moe_router_bias <- bias
  info
}

neural_moe_control_from_info <- function(info) {
  cfg <- info[["transformer_moe"]]
  if (!is.null(cfg)) {
    cfg$n_moe_layers <- NULL
    # Older saved models predate explicit compute precision. Preserve their
    # FP32 behavior when resuming/adapting them under the new defaults.
    cfg$compute_dtype <- cfg$compute_dtype %||% "float32"
  }
  list(transformer_ffn = info$transformer_ffn %||% "swiglu", transformer_moe = cfg)
}

neural_validate_saved_transformer_moe <- function(info) {
  param_names <- c(info$param_names, names(info$params))
  has_moe <- any(grepl("^W_moe_", param_names))
  mode <- info$transformer_ffn %||% "swiglu"
  if (!identical(mode, "moe")) {
    if (has_moe || !is.null(info[["transformer_moe"]])) {
      stop("MoE weights require explicit transformer_ffn='moe' architecture metadata.", call. = FALSE)
    }
    return(invisible(TRUE))
  }
  if (is.null(info[["transformer_moe"]]) || is.null(info$transformer_moe_router_bias)) {
    stop("Saved MoE models require architecture metadata and frozen router biases.", call. = FALSE)
  }
  resolved <- neural_resolve_transformer_moe(neural_moe_control_from_info(info),
                                            info$model_dims, info$model_depth)$transformer_moe
  bias <- cs2step_neural_to_r_array(info$transformer_moe_router_bias)
  expected <- c(resolved$n_moe_layers, resolved$n_routed_experts)
  if (!is.numeric(bias) || !identical(as.integer(dim(bias)), as.integer(expected)) || any(!is.finite(bias))) {
    stop("Saved MoE router biases must be finite and match layer/expert counts.", call. = FALSE)
  }
  invisible(TRUE)
}
