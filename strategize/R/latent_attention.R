# Fresh-fit defaults are explicit; old saved bundles without this field stay MHA.
neural_resolve_latent_attention <- function(value = NULL, model_dims = 128L, fitting = FALSE) {
  if (is.null(value)) value <- list(architecture = if (isTRUE(fitting)) "mla_dsa" else "mha")
  if (is.character(value) && length(value) == 1L) value <- list(architecture = value)
  if (!is.list(value) || (length(value) && (is.null(names(value)) || any(!nzchar(names(value))) || anyDuplicated(names(value))))) stop("transformer_attention must be a named list or architecture name.", call. = FALSE)
  defaults <- list(architecture = if (isTRUE(fitting)) "mla_dsa" else "mha",
    q_rank = max(1L, min(128L, as.integer(model_dims) %/% 2L)),
    kv_rank = max(1L, min(128L, as.integer(model_dims) %/% 2L)),
    indexer_heads = 2L, indexer_dim = 16L, top_k = 32L, indexer_loss_weight = 0.01)
  unknown <- setdiff(names(value), names(defaults))
  if (length(unknown)) stop("Unknown transformer_attention controls: ", paste(unknown, collapse = ", "), call. = FALSE)
  cfg <- utils::modifyList(defaults, value)
  if (length(cfg$architecture) != 1L || is.na(cfg$architecture) ||
      !cfg$architecture %in% c("mha", "mla", "mla_dsa")) {
    stop("transformer_attention$architecture must be mha, mla, or mla_dsa.", call. = FALSE)
  }
  for (name in c("q_rank", "kv_rank", "indexer_heads", "indexer_dim", "top_k")) {
    v <- cfg[[name]]
    if (!is.numeric(v) || length(v) != 1L || !is.finite(v) || v < 1L || v != round(v))
      stop("transformer_attention$", name, " must be a positive integer.", call. = FALSE)
    cfg[[name]] <- as.integer(v)
  }
  if (cfg$q_rank > model_dims || cfg$kv_rank > model_dims)
    stop("MLA latent ranks cannot exceed model_dims.", call. = FALSE)
  v <- cfg$indexer_loss_weight
  if (!is.numeric(v) || length(v) != 1L || !is.finite(v) || v < 0 ||
      (identical(cfg$architecture, "mla_dsa") && v == 0))
    stop("DSA requires a positive finite indexer_loss_weight to train its discrete selector.", call. = FALSE)
  cfg
}

neural_latent_attention_param_bases <- function() {
  paste0(c("W_q_up", "RMS_q_latent", "RMS_kv_latent", "W_index_q", "W_index_k", "W_index_w", "LN_index_k", "b_index_k"), "_l")
}

neural_latent_attention_shapes <- function(cfg, dims) {
  shapes <- list(W_q_up = c(cfg$q_rank, dims), RMS_q_latent = cfg$q_rank, RMS_kv_latent = cfg$kv_rank)
  if (identical(cfg$architecture, "mla_dsa")) shapes <- c(shapes, list(
    W_index_q = c(cfg$q_rank, cfg$indexer_heads * cfg$indexer_dim),
    W_index_k = c(dims, cfg$indexer_dim), W_index_w = c(dims, cfg$indexer_heads),
    LN_index_k = cfg$indexer_dim, b_index_k = cfg$indexer_dim))
  shapes
}

strategize_register_attention_helpers <- function() {
  if (is.null(strenv$jax_attention)) {
    strategize_register_moe_helpers()
    strenv$jax_attention <- reticulate::import_from_path(
      "strategize_attention", path = system.file("python", package = "strategize"), convert = TRUE)
  }
  invisible(TRUE)
}

neural_latent_attention_config <- function(model_info, params = NULL) {
  cfg <- neural_resolve_latent_attention(model_info$transformer_attention, model_info$model_dims %||% 128L)
  has_latent <- !is.null(params$W_q_up_layers) || !is.null(params$W_q_up_l1)
  if (!is.null(params) && xor(has_latent, !identical(cfg$architecture, "mha")))
    stop("Saved attention architecture and latent parameter roster disagree.", call. = FALSE)
  cfg
}

# Validate before preparing or compiling a saved bundle, even if weights are lazy.
neural_validate_saved_latent_attention <- function(info) {
  roster <- unique(c(info$param_names, names(info$params)))
  has_latent <- any(grepl("^W_q_up_(layers|l[0-9]+)$", roster))
  cfg <- neural_resolve_latent_attention(info$transformer_attention, info$model_dims %||% 128L)
  if (length(roster) && xor(has_latent, cfg$architecture != "mha"))
    stop("Saved attention architecture and latent parameter roster disagree.", call. = FALSE)
  if (has_latent) {
    bases <- names(neural_latent_attention_shapes(cfg, info$model_dims))
    if (any(!vapply(bases, function(b) any(grepl(paste0("^", b, "_(layers|l[0-9]+)$"), roster)), logical(1))))
      stop("Saved MLA/DSA bundle is missing latent or indexer parameters.", call. = FALSE)
  }
  invisible(TRUE)
}
