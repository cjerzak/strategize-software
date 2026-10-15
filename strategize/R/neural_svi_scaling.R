# Automatic step counts are a token-exposure heuristic, not an empirically
# established compute optimum. All accounting arithmetic is double precision;
# only the final, bounded count is converted to an R integer.
neural_optimal_svi_steps <- function(accounting,
                                     subsample_method = "full",
                                     tokens_per_param = 20,
                                     min_steps_full = 50L,
                                     max_steps_full = 1000L,
                                     min_steps_batch_vi = 200L,
                                     max_steps_batch_vi = 40000L,
                                     return_info = FALSE) {
  positive <- function(x, label) {
    x <- as.numeric(x)
    if (length(x) != 1L || !is.finite(x) || x <= 0) {
      stop(sprintf("Automatic SVI budget requires a positive finite %s.", label), call. = FALSE)
    }
    x
  }
  if (is.null(accounting)) {
    stop("Automatic SVI budget requires resolved model and token accounting.", call. = FALSE)
  }
  parameters <- positive(accounting$stored_model_parameters, "model parameter count")
  tokens <- positive(accounting$expected_global_tokens_per_update, "token count per update")
  ratio <- positive(tokens_per_param, "tokens_per_param")
  batched <- tolower(subsample_method) %in% c("batch", "batch_vi")
  lower <- positive(if (batched) min_steps_batch_vi else min_steps_full, "minimum steps")
  upper <- positive(if (batched) max_steps_batch_vi else max_steps_full, "maximum steps")
  if (lower != floor(lower) || upper != floor(upper) || upper < lower ||
      upper > .Machine$integer.max) {
    stop("Automatic SVI budget bounds must be ordered positive R integers.", call. = FALSE)
  }
  # Divide before multiplying to avoid unnecessary overflow. A positive Inf
  # quotient still saturates at the cap; it must never fall back to the floor.
  raw_steps <- ceiling((parameters / tokens) * ratio)
  steps <- as.integer(max(lower, min(raw_steps, upper)))
  if (!return_info) return(steps)
  c(accounting, list(
    accounting_version = 2L, budget_basis = "stored_model_parameters",
    token_basis = "nonpadding_encoder_positions_before_schema_dropout",
    tokens_per_param = ratio, unclamped_steps = raw_steps,
    min_steps = lower, max_steps = upper,
    applied_bound = if (raw_steps < lower) "minimum" else if (raw_steps > upper) "maximum" else "none",
    resolved_steps = steps
  ))
}

# Used by both the actual mixed-task sampler and its expected token accounting.
neural_svi_mixed_batch_counts <- function(n_pair, n_single, batch_size) {
  if (n_pair < 1 || n_single < 1 || batch_size < 2) {
    stop("Universal mixed-mode compact SVI batches must include pairwise and single rows.", call. = FALSE)
  }
  pair <- max(1, min(n_pair, round(as.double(batch_size) * n_pair / (n_pair + n_single))))
  single <- max(1, min(n_single, batch_size - pair))
  if (pair + single > batch_size) {
    if (pair > single) pair <- pair - 1 else single <- single - 1
  }
  c(pair = pair, single = single)
}

neural_svi_budget_accounting <- function(parameters, tokens_per_observation,
                                         train_idx = seq_along(tokens_per_observation),
                                         batch_size = 512L, subsample_method = "full",
                                         mixed_pair_count = NULL, balanced_state = NULL,
                                         model_depth = 1L, transformer_loop = list(enabled = FALSE),
                                         mixed_sampling = c("stratified", "independent")) {
  tokens <- as.double(tokens_per_observation)
  if (!length(train_idx) || anyNA(train_idx) || any(train_idx < 1 | train_idx > length(tokens)) ||
      any(!is.finite(tokens[train_idx]) | tokens[train_idx] <= 0)) {
    stop("Automatic SVI budget requires valid training rows and token counts.", call. = FALSE)
  }
  batch_size <- as.double(batch_size)
  if (length(batch_size) != 1L || !is.finite(batch_size) || batch_size < 1) {
    stop("Automatic SVI budget requires a positive batch size.", call. = FALSE)
  }
  batched <- tolower(subsample_method) %in% c("batch", "batch_vi")
  effective_batch <- if (batched) min(batch_size, length(train_idx)) else length(train_idx)
  sampling <- "uniform"
  if (!batched) {
    per_update <- sum(tokens[train_idx])
  } else if (!is.null(balanced_state)) {
    # The sampler draws a study, then a respondent, then one of their rows.
    unit_means <- vapply(balanced_state$obs_by_unit, function(idx) mean(tokens[idx]), numeric(1))
    study_means <- vapply(balanced_state$units_by_study, function(idx) mean(unit_means[idx]), numeric(1))
    effective_batch <- batch_size # sampling with replacement
    per_update <- effective_batch * mean(study_means)
    sampling <- "study_equal_respondent"
  } else if (!is.null(mixed_pair_count)) {
    pair <- train_idx[train_idx <= mixed_pair_count]
    single <- train_idx[train_idx > mixed_pair_count]
    mixed_sampling <- match.arg(mixed_sampling)
    counts <- if (mixed_sampling == "independent") {
      # Non-compact NumPyro uses a separate subsampling plate for each task.
      c(pair = min(batch_size, length(pair)), single = min(batch_size, length(single)))
    } else neural_svi_mixed_batch_counts(length(pair), length(single), batch_size)
    effective_batch <- sum(counts)
    per_update <- counts[["pair"]] * mean(tokens[pair]) + counts[["single"]] * mean(tokens[single])
    sampling <- paste0("mixed_task_", mixed_sampling)
  } else {
    per_update <- effective_batch * mean(tokens[train_idx])
  }
  loop <- transformer_loop
  effective_depth <- neural_transformer_effective_depth(loop, as.double(model_depth))
  max_depth <- if (isTRUE(loop$enabled)) {
    loop$prelude_layers + loop$coda_layers +
      (as.double(model_depth) - loop$prelude_layers - loop$coda_layers) *
      neural_transformer_training_max_iterations(loop)
  } else as.double(model_depth)
  c(parameters, list(n_training_observations = as.double(length(train_idx)),
    expected_global_batch_size = effective_batch, sampling = sampling,
    expected_global_tokens_per_update = per_update,
    mean_tokens_per_training_observation = mean(tokens[train_idx]),
    expected_layer_passes = effective_depth, max_layer_passes = max_depth))
}

# Count the valid positions assembled by the candidate/context packers, before
# schema dropout. Missing covariate values still have a learned missing-value
# token; absent schema slots (-1 in the order) are padding and do not count.
neural_svi_tokens_per_observation <- function(model_info, parameter_names, pairwise,
                                              experiment_index = NULL,
                                              party_left = NULL, party_right = NULL,
                                              resp_party = NULL) {
  n <- length(pairwise)
  has <- function(name) name %in% parameter_names
  order_count <- function(orders, default, index) {
    count <- function(order) sum(as.double(order) >= 0)
    if (is.null(index) || !length(orders)) return(rep(count(default), n))
    counts <- vapply(orders, count, numeric(1))
    index <- pmax(0, pmin(length(counts) - 1, as.integer(index)))
    rep_len(counts[index + 1L], n)
  }
  factors <- order_count(model_info$factor_order_by_experiment,
    model_info$default_factor_order %||% integer(0), experiment_index)
  ctx_index <- experiment_index %||% model_info$default_experiment_index
  covariates <- if (has("E_covariate_fused_base")) {
    order_count(model_info$covariate_order_by_experiment,
      model_info$default_covariate_order %||% (seq_along(model_info$covariate_names) - 1L), ctx_index)
  } else rep(0, n)
  present <- function(party = NULL, other = NULL, respondent = NULL) {
    out <- rep(TRUE, n)
    if (identical(model_info$context_present_masking, FALSE)) return(out)
    for (value in list(party, other)) {
      if (!is.null(value)) out <- out & (rep_len(value, n) != model_info$party_missing_index)
    }
    if (!is.null(respondent)) out <- out & (rep_len(respondent, n) != model_info$resp_party_missing_index)
    out
  }
  pair_present <- present(party_left, party_right, resp_party)
  context_present <- ifelse(pairwise, pair_present, present(respondent = resp_party))
  left_present <- ifelse(pairwise, pair_present, present(party_left, respondent = resp_party))
  right_present <- pair_present
  mode <- neural_experiment_token_mode(model_info)
  experiment <- (mode %in% c("legacy_id", "hybrid") && has("E_experiment") && !is.null(ctx_index)) ||
    (mode %in% c("description", "hybrid") && has("W_experiment_text") &&
      ((!is.null(ctx_index) && !is.null(model_info$experiment_description_text)) ||
        (!is.null(model_info$default_experiment_text) && isTRUE(model_info$default_experiment_text_present))))
  context <- covariates + as.double(experiment) +
    as.double(has("W_place_context") && neural_place_context_enabled(model_info)) +
    as.double(has("W_time_context") && neural_time_context_enabled(model_info)) +
    context_present * (as.double(has("E_resp_party")) +
      pairwise * (as.double(has("E_stage")) + as.double(has("E_matchup"))))
  auxiliary <- as.double(has("E_party")) + as.double(has("E_rel"))
  left <- factors + auxiliary * left_present
  right <- factors + auxiliary * right_present
  cross <- identical(model_info$cross_candidate_encoder_mode, "full")
  joint <- ifelse(pairwise,
    if (cross) 3 + context + left + right else 2 + 2 * context + left + right,
    1 + context + left)
  if (cross && isTRUE(neural_pairwise_antisymmetry_strict(model_info))) {
    joint[pairwise] <- 2 * joint[pairwise]
  }
  if (has("W_to_cand")) joint[] <- 0
  if (has("W_rc_r") && has("W_rc_c") && has("W_rc_out")) {
    # One respondent tower and one/two candidate towers, with shared weights.
    joint <- joint + 1 + context + 1 + left + ifelse(pairwise, 1 + right, 0)
  }
  as.double(joint)
}

# Dimensional arguments remain accepted for older internal callers, but automatic
# budgets require actual accounting; no dense-only fallback is inferred.
neural_resolve_svi_budget <- function(svi_steps_input,
                                      svi_num_draws_input,
                                      user_supplied_svi_steps = FALSE,
                                      user_supplied_svi_num_draws = FALSE,
                                      n_obs = NULL,
                                      n_factors = NULL,
                                      factor_levels = NULL,
                                      model_dims = NULL,
                                      model_depth = NULL,
                                      n_party_levels = 1L,
                                      n_resp_party_levels = 1L,
                                      n_resp_covariates = 0L,
                                      n_outcomes = 1L,
                                      pairwise_mode = FALSE,
                                      use_matchup_token = FALSE,
                                      use_cross_encoder = FALSE,
                                      use_cross_term = FALSE,
                                      use_cross_attn = FALSE,
                                      use_qk_norm = TRUE,
                                      batch_size = 512L,
                                      subsample_method = "full",
                                      output_only_mode = FALSE,
                                      likelihood = NULL,
                                      output_single_normal_batch_vi_floor = 2000L,
                                      output_single_normal_batch_vi_draw_floor = 200L,
                                      accounting = NULL,
                                      resume_budget = NULL) {
  steps_input <- svi_steps_input
  used_optimal <- FALSE
  budget_details <- NULL

  if (is.character(steps_input)) {
    steps_tag <- tolower(as.character(steps_input))
    if (length(steps_tag) == 1L && !is.na(steps_tag) && nzchar(steps_tag) &&
        identical(steps_tag, "optimal")) {
      used_optimal <- TRUE
      if (!is.null(resume_budget)) {
        if (!isTRUE(resume_budget$used_optimal) ||
            length(resume_budget$svi_steps) != 1L ||
            !is.finite(resume_budget$svi_steps) || resume_budget$svi_steps < 1 ||
            resume_budget$svi_steps != floor(resume_budget$svi_steps) ||
            resume_budget$svi_steps > .Machine$integer.max) {
          stop("Checkpoint does not contain a valid automatic SVI budget.", call. = FALSE)
        }
        budget_details <- resume_budget$accounting
        resolved_steps <- as.integer(resume_budget$svi_steps)
      } else {
        budget_details <- neural_optimal_svi_steps(
          accounting = accounting, subsample_method = subsample_method,
          return_info = TRUE
        )
        resolved_steps <- budget_details$resolved_steps
      }
    } else {
      resolved_steps <- suppressWarnings(as.integer(steps_input))
    }
  } else {
    resolved_steps <- suppressWarnings(as.integer(steps_input))
  }

  if (length(resolved_steps) != 1L || is.na(resolved_steps) || resolved_steps < 1L) {
    resolved_steps <- 1L
  }

  resolved_draws <- suppressWarnings(as.integer(svi_num_draws_input))
  if (length(resolved_draws) != 1L || is.na(resolved_draws) ||
      !is.finite(resolved_draws) || resolved_draws < 1L) {
    resolved_draws <- 1L
  }

  output_single_normal_batch_vi <- isTRUE(output_only_mode) &&
    !isTRUE(pairwise_mode) &&
    identical(tolower(as.character(subsample_method)), "batch_vi") &&
    identical(tolower(as.character(likelihood)), "normal")

  applied_step_floor <- FALSE
  if (!isTRUE(user_supplied_svi_steps) && isTRUE(output_single_normal_batch_vi)) {
    floor_steps <- suppressWarnings(as.integer(output_single_normal_batch_vi_floor))
    if (length(floor_steps) == 1L && !is.na(floor_steps) && floor_steps >= 1L) {
      applied_step_floor <- resolved_steps < floor_steps
      resolved_steps <- max(resolved_steps, floor_steps)
    }
  }

  # Retained for API compatibility, but draw inflation is intentionally disabled:
  # output-only normal batch_vi should not silently allocate larger posterior draws.
  applied_draw_floor <- FALSE

  result <- list(
    svi_steps = as.integer(resolved_steps),
    svi_num_draws = as.integer(resolved_draws),
    used_optimal = isTRUE(used_optimal),
    used_default_svi_steps = !isTRUE(user_supplied_svi_steps),
    used_default_svi_num_draws = !isTRUE(user_supplied_svi_num_draws),
    applied_output_single_normal_batch_vi_floor = isTRUE(applied_step_floor),
    applied_output_single_normal_batch_vi_draw_floor = isTRUE(applied_draw_floor),
    output_single_normal_batch_vi = isTRUE(output_single_normal_batch_vi)
  )
  if (!is.null(resume_budget) && used_optimal) {
    result$applied_output_single_normal_batch_vi_floor <-
      resume_budget$applied_output_single_normal_batch_vi_floor
  }
  if (!is.null(budget_details)) {
    budget_details$resolved_steps <- as.integer(resolved_steps)
    result$accounting <- budget_details
  }
  result
}
