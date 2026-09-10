# Backend startup must call this before any JAX array or device access.
strategize_initialize_data_parallel <- function(config = NULL) {
  if (is.null(strenv$data_parallel_module)) {
    module_dir <- system.file("python", package = "strategize")
    if (!nzchar(module_dir)) stop("Missing strategize Python runtime.", call. = FALSE)
    strenv$data_parallel_module <- reticulate::import_from_path(
      "strategize_distributed", path = module_dir, convert = FALSE
    )
  }
  if (!is.null(strenv$data_parallel) && is.null(config)) return(invisible(strenv$data_parallel))
  strenv$data_parallel <- strenv$data_parallel_module$initialize(config)
  strenv$dp_numpy <- reticulate::import("numpy", convert = FALSE)
  invisible(strenv$data_parallel)
}

strategize_dp_enabled <- function() {
  !is.null(strenv$data_parallel) && isTRUE(reticulate::py_to_r(strenv$data_parallel$enabled))
}

strategize_dp_validate_training_control <- function(control, enabled = strategize_dp_enabled()) {
  if (isTRUE(enabled) && isTRUE(control$universal_family_logprob_normalize)) {
    stop(
      "universal_family_logprob_normalize=TRUE is not supported with data parallelism: ",
      "family scales would depend on replica-local batches. Set it to FALSE.",
      call. = FALSE
    )
  }
  invisible(control)
}

strategize_dp_primary_rank <- function() {
  is.null(strenv$data_parallel) || isTRUE(reticulate::py_to_r(strenv$data_parallel$primary))
}

strategize_dp_primary <- function(fn, label = "primary operation") {
  if (!strategize_dp_enabled()) return(fn())
  value <- NULL
  error <- NULL
  if (strategize_dp_primary_rank()) {
    tryCatch(value <- fn(), error = function(e) error <<- conditionMessage(e))
  }
  strenv$data_parallel$agree_status(error, label)
  bytes <- if (strategize_dp_primary_rank()) as.integer(serialize(value, NULL, version = 3L)) else integer(0)
  bytes <- reticulate::py_to_r(strenv$data_parallel$broadcast_array(bytes))
  unserialize(as.raw(bytes))
}

strategize_dp_all_call <- function(fn, label) {
  if (!strategize_dp_enabled()) return(fn())
  value <- NULL
  error <- NULL
  tryCatch(value <- fn(), error = function(e) error <<- conditionMessage(e))
  strenv$data_parallel$agree_status(error, label)
  value
}

strategize_dp_batch <- function(args) {
  if (!strategize_dp_enabled()) return(args)
  reticulate::py_to_r(strenv$data_parallel$place_batch(args))
}

strategize_dp_local <- function(tree) {
  if (!strategize_dp_enabled() || is.null(tree)) return(tree)
  strenv$data_parallel$local_tree(tree)
}

strategize_dp_register_updates <- function() {
  if (!is.null(strenv$jax_svi_update_local)) return(invisible(NULL))
  strenv$jax_svi_update_local <- strenv$jax_svi_update
  strenv$jax_svi_update_scan_local <- strenv$jax_svi_update_scan
  strenv$jax_svi_gradient_diagnostics_local <- strenv$jax_svi_gradient_diagnostics
  strenv$jax_svi_update_jit_cache_info_local <- strenv$jax_svi_update_jit_cache_info
  strenv$jax_svi_update_jit_cache_clear_local <- strenv$jax_svi_update_jit_cache_clear
  strenv$jax_svi_update <- function(svi, state, args, ...) {
    if (strategize_dp_enabled()) return(reticulate::py_to_r(strenv$data_parallel$update(svi, state, args)))
    strenv$data_parallel$begin_update()
    start <- proc.time()[["elapsed"]]
    out <- strenv$jax_svi_update_local(svi, state, args, ...)
    strategize_jax_block_until_ready(out)
    strenv$data_parallel$record_local_timing(proc.time()[["elapsed"]] - start)
    out
  }
  strenv$jax_svi_update_scan <- function(svi, state, args, ...) {
    if (strategize_dp_enabled()) return(reticulate::py_to_r(strenv$data_parallel$update(svi, state, args, scan = TRUE)))
    strenv$data_parallel$begin_update()
    start <- proc.time()[["elapsed"]]
    out <- strenv$jax_svi_update_scan_local(svi, state, args, ...)
    strategize_jax_block_until_ready(out)
    strenv$data_parallel$record_local_timing(proc.time()[["elapsed"]] - start)
    out
  }
  strenv$jax_svi_gradient_diagnostics <- function(svi, state, args, ...) {
    if (strategize_dp_enabled()) return(reticulate::py_to_r(strenv$data_parallel$gradients(svi, state, args)))
    strenv$jax_svi_gradient_diagnostics_local(svi, state, args, ...)
  }
  strenv$jax_svi_update_jit_cache_info <- function() {
    if (strategize_dp_enabled()) return(reticulate::py_to_r(strenv$data_parallel$cache_info()))
    strenv$jax_svi_update_jit_cache_info_local()
  }
  strenv$jax_svi_update_jit_cache_clear <- function() {
    if (strategize_dp_enabled()) strenv$data_parallel$clear_cache()
    strenv$jax_svi_update_jit_cache_clear_local()
  }
  invisible(NULL)
}

strategize_dp_shutdown <- function() {
  if (!is.null(strenv$data_parallel)) strenv$data_parallel$shutdown()
  invisible(NULL)
}

strategize_dp_execution_identity <- function() {
  if (is.null(strenv$dp_code_identity)) {
    packages <- intersect(c("strategize", "preference.fm"), loadedNamespaces())
    source <- lapply(packages, function(package) {
      ns <- asNamespace(package)
      symbols <- sort(ls(ns, all.names = TRUE))
      symbols <- symbols[vapply(symbols, function(name) is.function(get(name, ns)), logical(1))]
      setNames(lapply(symbols, function(name) paste(deparse(get(name, ns), width.cutoff = 500L), collapse = "\n")), symbols)
    })
    strenv$dp_code_identity <- digest::digest(source, algo = "sha256")
  }
  meta <- reticulate::py_to_r(strenv$data_parallel$metadata())
  list(source_sha256 = strenv$dp_code_identity, runtime_sha256 = meta$runtime_sha256,
       versions = meta$versions, r_version = as.character(getRversion()), rng_kind = RNGkind(),
       precision = meta$precision, prng = meta$prng, x64 = meta$x64)
}
