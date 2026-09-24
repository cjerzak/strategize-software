# NULL means an ordinary stack, including every bundle predating recurrence.
neural_resolve_transformer_loop <- function(value = NULL, model_depth = 2L) {
  if (is.null(value)) value <- list(enabled = FALSE)
  if (is.logical(value) && length(value) == 1L) value <- list(enabled = value)
  if (!is.list(value) || (length(value) && (is.null(names(value)) ||
      any(!nzchar(names(value))) || anyDuplicated(names(value)))))
    stop("transformer_loop must be a logical flag or named list.", call. = FALSE)
  # Keep the backend defaults compatible with the first looped checkpoints.
  # New foundation fits opt in to sampled training depths and gated RMS
  # reinjection explicitly in preference.fm.
  defaults <- list(
    enabled = TRUE,
    iterations = 2L,
    prelude_layers = "auto",
    coda_layers = "auto",
    backprop_iterations = 0L,
    training_iterations = NULL,
    training_probabilities = NULL,
    reinjection = "scaled_add",
    normalize_core_gradients = FALSE
  )
  unknown <- setdiff(names(value), names(defaults))
  if (length(unknown)) stop("Unknown transformer_loop controls: ", paste(unknown, collapse = ", "), call. = FALSE)
  cfg <- utils::modifyList(defaults, value, keep.null = TRUE)
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
  if (is.null(cfg$training_iterations)) {
    if (!is.null(cfg$training_probabilities)) {
      stop("transformer_loop$training_probabilities requires training_iterations.", call. = FALSE)
    }
  } else {
    depths <- cfg$training_iterations
    if (!is.numeric(depths) || !length(depths) || any(!is.finite(depths)) ||
        any(depths < 1) || any(depths != round(depths)) || anyDuplicated(depths)) {
      stop("transformer_loop$training_iterations must contain unique positive integers.", call. = FALSE)
    }
    cfg$training_iterations <- as.integer(depths)
    probabilities <- cfg$training_probabilities
    if (is.null(probabilities)) {
      probabilities <- rep(1 / length(depths), length(depths))
    }
    if (!is.numeric(probabilities) || length(probabilities) != length(depths) ||
        any(!is.finite(probabilities)) || any(probabilities < 0) ||
        sum(probabilities) <= 0) {
      stop(paste0(
        "transformer_loop$training_probabilities must be nonnegative finite values ",
        "matching training_iterations, with positive total mass."
      ), call. = FALSE)
    }
    cfg$training_probabilities <- as.numeric(probabilities / sum(probabilities))
  }
  if (!is.character(cfg$reinjection) || length(cfg$reinjection) != 1L ||
      is.na(cfg$reinjection)) {
    stop("transformer_loop$reinjection must be 'scaled_add' or 'rms_gated'.", call. = FALSE)
  }
  cfg$reinjection <- tolower(cfg$reinjection)
  if (!cfg$reinjection %in% c("scaled_add", "rms_gated")) {
    stop("transformer_loop$reinjection must be 'scaled_add' or 'rms_gated'.", call. = FALSE)
  }
  if (!is.logical(cfg$normalize_core_gradients) ||
      length(cfg$normalize_core_gradients) != 1L ||
      is.na(cfg$normalize_core_gradients)) {
    stop("transformer_loop$normalize_core_gradients must be TRUE or FALSE.", call. = FALSE)
  }
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
  iterations <- neural_transformer_training_mean_iterations(cfg)
  cfg$prelude_layers + cfg$coda_layers +
    (model_depth - cfg$prelude_layers - cfg$coda_layers) * iterations
}

neural_transformer_training_mean_iterations <- function(cfg) {
  if (is.null(cfg$training_iterations)) return(as.double(cfg$iterations))
  sum(as.double(cfg$training_iterations) * as.double(cfg$training_probabilities))
}

neural_transformer_training_max_iterations <- function(cfg) {
  if (is.null(cfg$training_iterations)) return(as.integer(cfg$iterations))
  max(as.integer(cfg$training_iterations))
}

neural_sample_transformer_loop_iterations <- function(cfg) {
  if (!isTRUE(cfg$enabled)) return(NULL)
  if (is.null(cfg$training_iterations)) {
    return(strenv$jnp$array(as.integer(cfg$iterations), dtype = strenv$jnp$int32))
  }
  key <- neural_numpyro_prng_key()
  logits <- strenv$jnp$log(strenv$jnp$array(
    as.numeric(cfg$training_probabilities), dtype = strenv$jnp$float32
  ))
  index <- strenv$jax$random$categorical(key, logits)$astype(strenv$jnp$int32)
  if (strategize_dp_enabled() &&
      isTRUE(reticulate::py_to_r(strenv$data_parallel$in_svi_shard))) {
    # One rank samples for the whole data-parallel microbatch. This avoids
    # stragglers and preserves the requested depth distribution exactly.
    index <- strenv$jax$lax$psum(
      strenv$jnp$where(
        strenv$jax$lax$axis_index("data") == 0L,
        index,
        strenv$jnp$array(0L, dtype = strenv$jnp$int32)
      ),
      "data"
    )
  }
  strenv$jnp$take(
    strenv$jnp$array(as.integer(cfg$training_iterations), dtype = strenv$jnp$int32),
    index
  )
}

neural_transformer_loop_runtime_config <- function(model_info) {
  cfg <- neural_transformer_loop_config(model_info)
  training <- isTRUE(model_info$transformer_training)
  cfg$training <- training
  cfg$active_iterations <- model_info$transformer_loop_active_iterations %||%
    as.integer(cfg$iterations)
  cfg$max_iterations <- if (training) {
    neural_transformer_training_max_iterations(cfg)
  } else {
    as.integer(cfg$iterations)
  }
  cfg$jacobian_power_iterations <- as.integer(
    model_info$transformer_loop_jacobian_power_iterations %||% 0L
  )
  cfg
}

neural_recurrent_activation_diagnostics <- function(tokens,
                                                     model_info,
                                                     params = NULL,
                                                     token_mask = NULL,
                                                     jacobian_power_iterations = 4L) {
  cfg <- neural_transformer_loop_config(model_info)
  if (!isTRUE(cfg$enabled)) {
    stop("Recurrent activation diagnostics require an enabled transformer loop.", call. = FALSE)
  }
  power <- as.integer(jacobian_power_iterations)
  if (length(power) != 1L || is.na(power) || power < 0L ||
      power != jacobian_power_iterations) {
    stop("jacobian_power_iterations must be a nonnegative integer.", call. = FALSE)
  }
  diagnostic_info <- model_info
  diagnostic_info$transformer_training <- FALSE
  diagnostic_info$transformer_loop_collect_diagnostics <- TRUE
  diagnostic_info$transformer_loop_jacobian_power_iterations <- power
  result <- neural_run_transformer(
    tokens = tokens,
    model_info = diagnostic_info,
    params = params,
    token_mask = token_mask,
    return_details = TRUE
  )
  result$recurrent
}

neural_recurrent_prediction_diagnostics <- function(logits_by_iteration, outcomes) {
  logits <- as.matrix(logits_by_iteration)
  y <- as.numeric(outcomes)
  if (nrow(logits) != length(y) || !length(y) ||
      any(!is.finite(logits)) || any(!is.finite(y)) || any(!y %in% c(0, 1))) {
    stop("Require a finite examples-by-iteration logit matrix and binary outcomes.", call. = FALSE)
  }
  softplus <- function(x) pmax(x, 0) + log1p(exp(-abs(x)))
  nll <- vapply(seq_len(ncol(logits)), function(i) {
    mean(softplus(logits[, i]) - y * logits[, i])
  }, numeric(1))
  if (ncol(logits) == 1L) {
    return(data.frame(iteration = 1L, nll = nll, mean_abs_logit_change = NA_real_,
                      fraction_examples_improved = NA_real_))
  }
  row_nll <- softplus(logits) - y * logits
  data.frame(
    iteration = seq_len(ncol(logits)),
    nll = nll,
    mean_abs_logit_change = c(NA_real_, colMeans(abs(logits[, -1L, drop = FALSE] -
      logits[, -ncol(logits), drop = FALSE]))),
    fraction_examples_improved = c(NA_real_, colMeans(
      row_nll[, -1L, drop = FALSE] < row_nll[, -ncol(row_nll), drop = FALSE]
    ))
  )
}
