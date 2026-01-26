neural_optimal_svi_steps <- function(n_obs,
                                     n_factors,
                                     factor_levels,
                                     model_dims,
                                     model_depth,
                                     n_party_levels = 1L,
                                     n_resp_party_levels = 1L,
                                     n_resp_covariates = 0L,
                                     n_outcomes = 1L,
                                     pairwise_mode = FALSE,
                                     use_matchup_token = FALSE,
                                     use_cross_encoder = FALSE,
                                     use_cross_term = FALSE,
                                     batch_size = 512L,
                                     subsample_method = "full",
                                     tokens_per_param = 20,
                                     min_steps_full = 50L,
                                     max_steps_full = 1000L,
                                     min_steps_batch_vi = 200L,
                                     max_steps_batch_vi = 20000L) {
  n_obs <- suppressWarnings(as.integer(n_obs))
  if (length(n_obs) != 1L || is.na(n_obs) || n_obs < 1L) {
    return(1L)
  }

  n_factors <- suppressWarnings(as.integer(n_factors))
  if (length(n_factors) != 1L || is.na(n_factors) || n_factors < 1L) {
    n_factors <- suppressWarnings(as.integer(length(factor_levels)))
    if (length(n_factors) != 1L || is.na(n_factors) || n_factors < 1L) {
      n_factors <- 1L
    }
  }

  factor_levels <- suppressWarnings(as.integer(factor_levels))
  if (length(factor_levels) < 1L) {
    factor_levels <- rep(1L, n_factors)
  } else if (length(factor_levels) != n_factors) {
    factor_levels <- rep_len(factor_levels, n_factors)
  }
  factor_levels[is.na(factor_levels) | factor_levels < 1L] <- 1L

  model_dims <- suppressWarnings(as.integer(model_dims))
  if (length(model_dims) != 1L || is.na(model_dims) || model_dims < 1L) {
    model_dims <- 1L
  }
  model_depth <- suppressWarnings(as.integer(model_depth))
  if (length(model_depth) != 1L || is.na(model_depth) || model_depth < 1L) {
    model_depth <- 1L
  }

  n_party_levels <- suppressWarnings(as.integer(n_party_levels))
  if (length(n_party_levels) != 1L || is.na(n_party_levels) || n_party_levels < 1L) {
    n_party_levels <- 1L
  }
  n_resp_party_levels <- suppressWarnings(as.integer(n_resp_party_levels))
  if (length(n_resp_party_levels) != 1L || is.na(n_resp_party_levels) || n_resp_party_levels < 1L) {
    n_resp_party_levels <- 1L
  }
  n_resp_covariates <- suppressWarnings(as.integer(n_resp_covariates))
  if (length(n_resp_covariates) != 1L || is.na(n_resp_covariates) || n_resp_covariates < 0L) {
    n_resp_covariates <- 0L
  }
  n_outcomes <- suppressWarnings(as.integer(n_outcomes))
  if (length(n_outcomes) != 1L || is.na(n_outcomes) || n_outcomes < 1L) {
    n_outcomes <- 1L
  }

  batch_size <- suppressWarnings(as.integer(batch_size))
  if (length(batch_size) != 1L || is.na(batch_size) || batch_size < 1L) {
    batch_size <- 1L
  }

  tokens_per_param <- as.numeric(tokens_per_param)
  if (length(tokens_per_param) != 1L || is.na(tokens_per_param) ||
      !is.finite(tokens_per_param) || tokens_per_param <= 0) {
    tokens_per_param <- 20
  }

  subsample_method <- tolower(as.character(subsample_method))
  if (length(subsample_method) != 1L || is.na(subsample_method) || !nzchar(subsample_method)) {
    subsample_method <- "full"
  }
  is_batch_vi <- identical(subsample_method, "batch_vi")

  ff_dim <- suppressWarnings(as.integer(round(as.numeric(model_dims) * 3.75)))
  if (is.na(ff_dim) || ff_dim < 1L) {
    ff_dim <- model_dims
  }

  # Approximate learned-parameter count (enough for scaling).
  n_params_factor_embed <- sum((factor_levels + 1L) * model_dims)
  n_params_feature_id <- n_factors * model_dims
  n_params_party <- n_party_levels * model_dims
  n_params_rel <- 3L * model_dims
  n_params_resp_party <- n_resp_party_levels * model_dims
  n_params_stage <- if (isTRUE(pairwise_mode)) n_resp_party_levels * 2L * model_dims else 0L
  n_params_matchup <- if (isTRUE(pairwise_mode) && isTRUE(use_matchup_token)) {
    as.integer(n_party_levels * (n_party_levels + 1L) / 2L) * model_dims
  } else {
    0L
  }
  n_params_choice <- model_dims
  n_params_sep <- if (isTRUE(pairwise_mode) && isTRUE(use_cross_encoder)) model_dims else 0L
  n_params_segment <- if (isTRUE(pairwise_mode) && isTRUE(use_cross_encoder)) 2L * model_dims else 0L
  n_params_resp_cov <- n_resp_covariates * model_dims
  n_params_transformer_layer <- 4L * model_dims * model_dims +
    2L * model_dims * ff_dim +
    4L * model_dims
  n_params_transformer <- model_depth * n_params_transformer_layer
  n_params_final_norm <- model_dims
  n_params_out <- model_dims * n_outcomes + n_outcomes
  n_params_cross <- if (isTRUE(pairwise_mode) && isTRUE(use_cross_term)) {
    model_dims * model_dims + n_outcomes
  } else {
    0L
  }
  n_params_total <- n_params_factor_embed +
    n_params_feature_id +
    n_params_party +
    n_params_rel +
    n_params_resp_party +
    n_params_stage +
    n_params_matchup +
    n_params_choice +
    n_params_sep +
    n_params_segment +
    n_params_resp_cov +
    n_params_transformer +
    n_params_final_norm +
    n_params_out +
    n_params_cross
  n_params_total <- max(1L, as.integer(n_params_total))

  # Approximate token length per observation passed through the transformer.
  # Here "token" refers to an input embedding position (choice token, context tokens,
  # factor tokens, plus party/rel tokens). Even though the factor levels are exogenous,
  # they still determine which embeddings/paths are exercised and they drive the
  # per-step compute.
  ctx_len <- 1L
  if (isTRUE(pairwise_mode)) {
    ctx_len <- ctx_len + 1L
    if (isTRUE(use_matchup_token)) {
      ctx_len <- ctx_len + 1L
    }
  }
  if (n_resp_covariates > 0L) {
    ctx_len <- ctx_len + 1L
  }
  cand_len <- n_factors + 2L
  tokens_single <- cand_len + ctx_len + 1L
  tokens_per_obs <- if (isTRUE(pairwise_mode) && isTRUE(use_cross_encoder)) {
    # choice + ctx + sep + left + sep + right
    1L + ctx_len + 1L + cand_len + 1L + cand_len
  } else if (isTRUE(pairwise_mode)) {
    # Two independent encoder passes.
    2L * tokens_single
  } else {
    tokens_single
  }
  tokens_per_obs <- max(1L, as.integer(tokens_per_obs))

  tokens_target <- tokens_per_param * as.numeric(n_params_total)
  if (!is.finite(tokens_target) || tokens_target <= 0) {
    tokens_target <- 1
  }

  eff_batch <- if (is_batch_vi) min(batch_size, n_obs) else n_obs
  eff_batch <- max(1L, eff_batch)
  tokens_per_step <- as.numeric(eff_batch) * as.numeric(tokens_per_obs)
  steps_raw <- as.integer(ceiling(tokens_target / tokens_per_step))
  if (length(steps_raw) != 1L || is.na(steps_raw) || steps_raw < 1L) {
    steps_raw <- 1L
  }

  if (is_batch_vi) {
    min_steps <- suppressWarnings(as.integer(min_steps_batch_vi))
    max_steps <- suppressWarnings(as.integer(max_steps_batch_vi))
  } else {
    min_steps <- suppressWarnings(as.integer(min_steps_full))
    max_steps <- suppressWarnings(as.integer(max_steps_full))
  }
  if (length(min_steps) != 1L || is.na(min_steps) || min_steps < 1L) {
    min_steps <- 1L
  }
  if (length(max_steps) != 1L || is.na(max_steps) || max_steps < min_steps) {
    max_steps <- max(min_steps, steps_raw)
  }

  max(min_steps, min(steps_raw, max_steps))
}
