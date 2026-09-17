# NULL means an ordinary stack, including every bundle predating recurrence.
neural_resolve_transformer_loop <- function(value = NULL, model_depth = 2L) {
  if (is.null(value)) value <- list(enabled = FALSE)
  if (is.logical(value) && length(value) == 1L) value <- list(enabled = value)
  if (!is.list(value) || (length(value) && (is.null(names(value)) ||
      any(!nzchar(names(value))) || anyDuplicated(names(value)))))
    stop("transformer_loop must be a logical flag or named list.", call. = FALSE)
  defaults <- list(enabled = TRUE, iterations = 2L, prelude_layers = "auto",
                   coda_layers = "auto", backprop_iterations = 0L)
  unknown <- setdiff(names(value), names(defaults))
  if (length(unknown)) stop("Unknown transformer_loop controls: ", paste(unknown, collapse = ", "), call. = FALSE)
  cfg <- utils::modifyList(defaults, value)
  if (!is.logical(cfg$enabled) || length(cfg$enabled) != 1L || is.na(cfg$enabled))
    stop("transformer_loop$enabled must be TRUE or FALSE.", call. = FALSE)
  integer_value <- function(x, name, minimum = 0L) {
    if (!is.numeric(x) || length(x) != 1L || !is.finite(x) || x < minimum ||
        x > .Machine$integer.max || x != round(x))
      stop("transformer_loop$", name, " must be an integer >= ", minimum, ".", call. = FALSE)
    as.integer(x)
  }
  depth <- integer_value(model_depth, "model_depth", 1L)
  cfg$iterations <- integer_value(cfg$iterations, "iterations", 1L)
  cfg$backprop_iterations <- integer_value(cfg$backprop_iterations, "backprop_iterations")
  if (identical(cfg$prelude_layers, "auto")) cfg$prelude_layers <- min(1L, depth - 1L)
  cfg$prelude_layers <- integer_value(cfg$prelude_layers, "prelude_layers")
  if (identical(cfg$coda_layers, "auto")) cfg$coda_layers <- min(1L, max(0L, depth - cfg$prelude_layers - 1L))
  cfg$coda_layers <- integer_value(cfg$coda_layers, "coda_layers")
  if (cfg$prelude_layers + as.double(cfg$coda_layers) >= depth)
    stop("transformer_loop must leave at least one layer in the recurrent core.", call. = FALSE)
  cfg
}

neural_transformer_loop_config <- function(model_info) {
  cfg <- neural_resolve_transformer_loop(model_info$transformer_loop, model_info$model_depth %||% 2L)
  if (cfg$enabled && !identical(model_info$residual_mode %||% "standard", "standard"))
    stop("Looped transformers require residual_mode='standard'.", call. = FALSE)
  cfg
}

neural_transformer_effective_depth <- function(cfg, model_depth) {
  if (!cfg$enabled) return(as.integer(model_depth))
  cfg$prelude_layers + cfg$coda_layers +
    (model_depth - cfg$prelude_layers - cfg$coda_layers) * as.double(cfg$iterations)
}
