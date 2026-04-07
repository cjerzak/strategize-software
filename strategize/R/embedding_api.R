#' Extract last-layer embeddings from fitted neural objects
#'
#' @param object A fitted neural object.
#' @param ... Additional method-specific arguments.
#' @return An object of class \code{strategic_embeddings}.
#' @export
extract_embeddings <- function(object, ...) {
  UseMethod("extract_embeddings")
}

cs2step_embeddings_as_matrix <- function(x) {
  out <- cs2step_neural_to_r_array(x)
  out <- as.matrix(out)
  storage.mode(out) <- "double"
  out
}

cs2step_build_embeddings_result <- function(mode,
                                            source_class,
                                            cross_candidate_encoder = "none",
                                            embeddings = NULL,
                                            left = NULL,
                                            right = NULL,
                                            joint = NULL,
                                            metadata = NULL) {
  out <- list(mode = mode)
  dims <- integer(0)
  if (!is.null(embeddings)) {
    out$embeddings <- embeddings
    dims <- as.integer(ncol(embeddings))
    names(dims) <- NULL
  }
  if (!is.null(left)) {
    out$left <- left
    dims <- c(dims, left = as.integer(ncol(left)))
  }
  if (!is.null(right)) {
    out$right <- right
    dims <- c(dims, right = as.integer(ncol(right)))
  }
  if (!is.null(joint)) {
    out$joint <- joint
    dims <- c(dims, joint = as.integer(ncol(joint)))
  }
  meta_default <- list(
    source_class = as.character(source_class),
    mode = as.character(mode),
    cross_candidate_encoder = as.character(cross_candidate_encoder),
    embedding_dim = dims
  )
  out$metadata <- modifyList(meta_default, metadata %||% list())
  structure(out, class = "strategic_embeddings")
}

cs2step_neural_extract_single_prepared <- function(params,
                                                   model_info,
                                                   prep) {
  neural_encode_candidate_core_prepared(
    params = params,
    model_info = model_info,
    X_idx = prep$X_single,
    party_idx = prep$party_single,
    resp_party_idx = prep$resp_party,
    resp_cov = prep$resp_cov,
    resp_cov_present = prep$resp_cov_present %||% NULL,
    resp_cov_order = prep$resp_cov_order %||% NULL,
    experiment_idx = prep$experiment_idx %||% NULL,
    factor_order = prep$factor_order %||% NULL,
    return_tokens = FALSE
  )
}

cs2step_neural_extract_pair_prepared <- function(params,
                                                 model_info,
                                                 prep) {
  mode <- neural_cross_encoder_mode(model_info)
  use_cross_encoder <- identical(mode, "full")
  use_cross_attn <- identical(mode, "attn")

  stage_idx <- neural_stage_index(prep$party_left, prep$party_right, model_info)
  matchup_idx <- NULL
  if (!is.null(params$E_matchup)) {
    matchup_idx <- neural_matchup_index(prep$party_left, prep$party_right, model_info)
  }

  if (isTRUE(use_cross_encoder)) {
    n_batch <- ai(prep$X_left$shape[[1]])
    choice_tok <- neural_prepare_choice_token_batch(model_info, params, n_batch)
    choice_mask <- strenv$jnp$ones(list(n_batch, 1L), dtype = strenv$dtj)
    ctx_info <- neural_build_context_tokens_batch(
      model_info = model_info,
      resp_party_idx = prep$resp_party,
      stage_idx = stage_idx,
      matchup_idx = matchup_idx,
      resp_cov = prep$resp_cov,
      resp_cov_present = prep$resp_cov_present %||% NULL,
      resp_cov_order = prep$resp_cov_order %||% NULL,
      experiment_idx = prep$experiment_idx %||% NULL,
      params = params,
      return_mask = TRUE
    )
    ctx_tokens <- ctx_info$tokens %||% NULL
    ctx_mask <- ctx_info$mask %||% NULL
    left_info <- neural_build_candidate_tokens_hard(
      prep$X_left,
      prep$party_left,
      model_info = model_info,
      resp_party_idx = prep$resp_party,
      experiment_idx = prep$experiment_idx %||% NULL,
      factor_order = prep$factor_order %||% NULL,
      params = params,
      return_mask = TRUE
    )
    left_tokens <- neural_add_segment_embedding(
      left_info$tokens,
      0L,
      model_info = model_info,
      params = params
    )
    right_info <- neural_build_candidate_tokens_hard(
      prep$X_right,
      prep$party_right,
      model_info = model_info,
      resp_party_idx = prep$resp_party,
      experiment_idx = prep$experiment_idx %||% NULL,
      factor_order = prep$factor_order %||% NULL,
      params = params,
      return_mask = TRUE
    )
    right_tokens <- neural_add_segment_embedding(
      right_info$tokens,
      1L,
      model_info = model_info,
      params = params
    )
    sep_tok <- neural_build_sep_token(model_info, n_batch = n_batch, params = params)
    token_parts <- list(choice_tok)
    mask_parts <- list(choice_mask)
    if (!is.null(ctx_tokens)) {
      token_parts <- c(token_parts, list(ctx_tokens))
      mask_parts <- c(mask_parts, list(ctx_mask))
    }
    token_parts <- c(token_parts, list(sep_tok, left_tokens, sep_tok, right_tokens))
    mask_parts <- c(
      mask_parts,
      list(
        strenv$jnp$ones(list(n_batch, 1L), dtype = strenv$dtj),
        left_info$mask,
        strenv$jnp$ones(list(n_batch, 1L), dtype = strenv$dtj),
        right_info$mask
      )
    )
    tokens <- strenv$jnp$concatenate(token_parts, axis = 1L)
    token_mask <- strenv$jnp$concatenate(mask_parts, axis = 1L)
    transformer_out <- neural_run_transformer(
      tokens,
      model_info,
      params,
      token_mask = token_mask,
      return_details = TRUE
    )
    return(list(joint = neural_extract_choice_representation(transformer_out)))
  }

  n_batch <- ai(prep$X_left$shape[[1]])
  X_all <- strenv$jnp$concatenate(list(prep$X_left, prep$X_right), axis = 0L)
  p_all <- strenv$jnp$concatenate(list(prep$party_left, prep$party_right), axis = 0L)
  resp_p_all <- strenv$jnp$concatenate(list(prep$resp_party, prep$resp_party), axis = 0L)
  resp_c_all <- if (is.null(prep$resp_cov)) NULL else {
    strenv$jnp$concatenate(list(prep$resp_cov, prep$resp_cov), axis = 0L)
  }
  resp_c_present_all <- if (is.null(prep$resp_cov_present)) NULL else {
    strenv$jnp$concatenate(list(prep$resp_cov_present, prep$resp_cov_present), axis = 0L)
  }
  resp_c_order_all <- if (is.null(prep$resp_cov_order)) NULL else {
    strenv$jnp$concatenate(list(prep$resp_cov_order, prep$resp_cov_order), axis = 0L)
  }
  experiment_idx_all <- if (is.null(prep$experiment_idx)) NULL else {
    strenv$jnp$concatenate(list(prep$experiment_idx, prep$experiment_idx), axis = 0L)
  }
  factor_order_all <- if (is.null(prep$factor_order)) NULL else {
    strenv$jnp$concatenate(list(prep$factor_order, prep$factor_order), axis = 0L)
  }
  stage_all <- if (is.null(stage_idx)) NULL else {
    strenv$jnp$concatenate(list(stage_idx, stage_idx), axis = 0L)
  }
  matchup_all <- if (is.null(matchup_idx)) NULL else {
    strenv$jnp$concatenate(list(matchup_idx, matchup_idx), axis = 0L)
  }

  if (isTRUE(use_cross_attn)) {
    enc_all <- neural_encode_candidate_core_prepared(
      params = params,
      model_info = model_info,
      X_idx = X_all,
      party_idx = p_all,
      resp_party_idx = resp_p_all,
      resp_cov = resp_c_all,
      resp_cov_present = resp_c_present_all,
      resp_cov_order = resp_c_order_all,
      experiment_idx = experiment_idx_all,
      factor_order = factor_order_all,
      stage_idx = stage_all,
      matchup_idx = matchup_all,
      return_tokens = TRUE
    )
    phi_all <- enc_all$phi
    cand_all <- enc_all$cand_tokens_out
    cand_mask_all <- enc_all$cand_token_mask
  } else {
    phi_all <- neural_encode_candidate_core_prepared(
      params = params,
      model_info = model_info,
      X_idx = X_all,
      party_idx = p_all,
      resp_party_idx = resp_p_all,
      resp_cov = resp_c_all,
      resp_cov_present = resp_c_present_all,
      resp_cov_order = resp_c_order_all,
      experiment_idx = experiment_idx_all,
      factor_order = factor_order_all,
      stage_idx = stage_all,
      matchup_idx = matchup_all,
      return_tokens = FALSE
    )
    cand_all <- NULL
    cand_mask_all <- NULL
  }

  idx_left <- strenv$jnp$arange(n_batch)
  idx_right <- strenv$jnp$arange(n_batch, 2L * n_batch)
  phi_left <- strenv$jnp$take(phi_all, idx_left, axis = 0L)
  phi_right <- strenv$jnp$take(phi_all, idx_right, axis = 0L)

  if (isTRUE(use_cross_attn)) {
    cand_left_out <- strenv$jnp$take(cand_all, idx_left, axis = 0L)
    cand_right_out <- strenv$jnp$take(cand_all, idx_right, axis = 0L)
    cand_left_mask <- strenv$jnp$take(cand_mask_all, idx_left, axis = 0L)
    cand_right_mask <- strenv$jnp$take(cand_mask_all, idx_right, axis = 0L)
    ctx_left <- neural_cross_attend_cls_to_tokens(
      phi_left,
      cand_right_out,
      model_info = model_info,
      params = params,
      kv_token_mask = cand_right_mask
    )
    ctx_right <- neural_cross_attend_cls_to_tokens(
      phi_right,
      cand_left_out,
      model_info = model_info,
      params = params,
      kv_token_mask = cand_left_mask
    )
    phi_left <- neural_merge_cross_attn_representation(
      phi_left,
      ctx_left,
      params,
      model_info$model_dims
    )
    phi_right <- neural_merge_cross_attn_representation(
      phi_right,
      ctx_right,
      params,
      model_info$model_dims
    )
  }

  list(left = phi_left, right = phi_right)
}

cs2step_neural_extract_prepared <- function(params,
                                            model_info,
                                            prep) {
  if (isTRUE(prep$pairwise)) {
    return(cs2step_neural_extract_pair_prepared(
      params = params,
      model_info = model_info,
      prep = prep
    ))
  }
  list(embeddings = cs2step_neural_extract_single_prepared(
    params = params,
    model_info = model_info,
    prep = prep
  ))
}

cs2step_neural_extract_internal <- function(object,
                                            W_new,
                                            X_new = NULL,
                                            factor_order_new = NULL,
                                            experiment_id = NULL,
                                            experiment_description = NULL,
                                            pair_id = NULL,
                                            profile_order = NULL,
                                            source_class = class(object)[[1]],
                                            extra_metadata = NULL) {
  if (!inherits(object, "strategic_predictor")) {
    stop("Internal neural extraction requires a strategic_predictor.", call. = FALSE)
  }
  if (!identical(object$model_type, "neural")) {
    stop("extract_embeddings() is only available for neural models.", call. = FALSE)
  }

  enc <- object$encoder
  W_raw <- as.data.frame(W_new, check.names = FALSE)
  W_new <- cs2step_align_W(W_new, enc$factor_names)
  W_idx <- cs2step_encode_W_indices(
    W = W_new,
    names_list = enc$names_list,
    unknown = "holdout",
    pad_unknown = 1L
  )
  prep_params <- cs2step_neural_prepare_params(
    object,
    conda_env = object$metadata$conda_env %||% NULL,
    conda_env_required = object$metadata$conda_env_required %||% TRUE
  )
  model_info <- prep_params$model_info
  if (identical(object$mode, "pairwise") &&
      !is.null(experiment_description) &&
      length(experiment_description) == nrow(W_idx)) {
    pair_info <- cs2step_build_pair_mat(
      pair_id = pair_id,
      W = W_idx,
      profile_order = profile_order,
      competing_group_variable_candidate = NULL
    )
    experiment_description <- experiment_description[pair_info$pair_mat[, 1], drop = TRUE]
  }
  model_info <- cs2step_neural_apply_experiment_description(
    model_info = model_info,
    experiment_description = experiment_description,
    n_rows = if (identical(object$mode, "pairwise")) {
      length(unique(pair_id))
    } else {
      nrow(W_idx)
    },
    text_embedding_fn = object$metadata$text_embedding_fn %||% NULL
  )
  prep <- cs2step_neural_prepare_prediction_data(
    W_idx = W_idx,
    model_info = model_info,
    resp_cov_new = X_new,
    factor_order_new = factor_order_new %||%
      if (identical(neural_factor_tokenization(model_info), "language_span")) {
        neural_factor_order_from_names(colnames(W_raw), enc$factor_names)
      } else {
        NULL
      },
    experiment_id = experiment_id,
    pair_id = pair_id,
    profile_order = profile_order,
    mode = object$mode
  )
  extracted <- cs2step_neural_extract_prepared(
    params = prep_params$params,
    model_info = model_info,
    prep = prep
  )
  cross_mode <- if (isTRUE(prep$pairwise)) {
    neural_cross_encoder_mode(model_info)
  } else {
    "none"
  }

  if (!isTRUE(prep$pairwise)) {
    return(cs2step_build_embeddings_result(
      mode = "single",
      source_class = source_class,
      cross_candidate_encoder = cross_mode,
      embeddings = cs2step_embeddings_as_matrix(extracted$embeddings),
      metadata = extra_metadata
    ))
  }
  if (!is.null(extracted$joint)) {
    return(cs2step_build_embeddings_result(
      mode = "pairwise",
      source_class = source_class,
      cross_candidate_encoder = cross_mode,
      joint = cs2step_embeddings_as_matrix(extracted$joint),
      metadata = extra_metadata
    ))
  }
  cs2step_build_embeddings_result(
    mode = "pairwise",
    source_class = source_class,
    cross_candidate_encoder = cross_mode,
    left = cs2step_embeddings_as_matrix(extracted$left),
    right = cs2step_embeddings_as_matrix(extracted$right),
    metadata = extra_metadata
  )
}

#' @rdname extract_embeddings
#' @param newdata New data in the same format accepted by \code{predict()}.
#' @export
extract_embeddings.strategic_predictor <- function(object,
                                                   newdata,
                                                   ...) {
  if (!inherits(object, "strategic_predictor")) {
    stop("extract_embeddings.strategic_predictor() requires a strategic_predictor object.",
         call. = FALSE)
  }
  unpacked <- cs2step_unpack_newdata(newdata, object$encoder$factor_names, object$mode)
  cs2step_neural_extract_internal(
    object = object,
    W_new = unpacked$W,
    X_new = unpacked$X,
    experiment_id = unpacked$experiment_id,
    experiment_description = unpacked$experiment_description,
    pair_id = unpacked$pair_id,
    profile_order = unpacked$profile_order,
    source_class = "strategic_predictor"
  )
}

cs_foundation_unpack_embedding_newdata <- function(newdata,
                                                   factor_names = NULL) {
  if (is.null(newdata)) {
    stop("'newdata' is required for embedding extraction.", call. = FALSE)
  }

  if (is.list(newdata) && !is.data.frame(newdata)) {
    if (!"W" %in% names(newdata)) {
      stop("When newdata is a list, it must contain element 'W'.", call. = FALSE)
    }
    return(list(
      W = newdata$W,
      X = newdata$X %||% NULL,
      pair_id = newdata$pair_id %||% NULL,
      profile_order = newdata$profile_order %||% NULL,
      experiment_id = newdata$experiment_id %||% NULL,
      experiment_description = newdata$experiment_description %||% NULL
    ))
  }

  newdata <- as.data.frame(newdata)
  pair_id <- if ("pair_id" %in% colnames(newdata)) newdata[["pair_id"]] else NULL
  profile_order <- if ("profile_order" %in% colnames(newdata)) newdata[["profile_order"]] else NULL
  experiment_id <- if ("experiment_id" %in% colnames(newdata)) newdata[["experiment_id"]] else NULL
  experiment_description <- if ("experiment_description" %in% colnames(newdata)) {
    newdata[["experiment_description"]]
  } else {
    NULL
  }
  special_cols <- c("pair_id", "profile_order", "experiment_id", "experiment_description")
  if (!is.null(factor_names)) {
    missing_cols <- setdiff(factor_names, colnames(newdata))
    if (length(missing_cols) > 0L) {
      stop(
        "Missing factor columns in newdata: ",
        paste(missing_cols, collapse = ", "),
        call. = FALSE
      )
    }
    W <- newdata[, factor_names, drop = FALSE]
    extra_cols <- setdiff(colnames(newdata), c(factor_names, special_cols))
  } else {
    extra_cols <- setdiff(colnames(newdata), special_cols)
    W <- newdata[, extra_cols, drop = FALSE]
    extra_cols <- character(0)
  }
  X <- if (length(extra_cols) > 0L) {
    newdata[, extra_cols, drop = FALSE]
  } else {
    NULL
  }
  list(
    W = W,
    X = X,
    pair_id = pair_id,
    profile_order = profile_order,
    experiment_id = experiment_id,
    experiment_description = experiment_description
  )
}

cs_foundation_normalize_request <- function(W,
                                            X = NULL,
                                            mode = c("auto", "pairwise", "single"),
                                            pair_id = NULL,
                                            profile_order = NULL,
                                            experiment_id = "embedding_request",
                                            names_list = NULL,
                                            p_list = NULL,
                                            canonical_factor_id = NULL,
                                            canonical_level_id = NULL) {
  if (is.null(W)) {
    stop("'W' is required.", call. = FALSE)
  }
  W_df <- as.data.frame(W, stringsAsFactors = FALSE)
  if (ncol(W_df) < 1L) {
    stop("'W' must contain at least one factor column.", call. = FALSE)
  }
  if (is.null(colnames(W_df))) {
    colnames(W_df) <- paste0("V", seq_len(ncol(W_df)))
  }
  mode_use <- cs_foundation_mode(match.arg(mode), pair_id)
  if (identical(mode_use, "pairwise")) {
    cs2step_validate_pairwise_ids(pair_id, nrow(W_df))
  }
  names_list_use <- cs_foundation_normalize_names_list_local(
    names_list = names_list,
    W = W_df,
    p_list = p_list
  )
  factor_names <- names(names_list_use)
  X_use <- cs_foundation_validate_numeric_matrix(
    X = X %||% NULL,
    n = nrow(W_df),
    arg = "newdata$X"
  )
  canonical_factor_id <- cs_foundation_normalize_factor_ids(
    values = canonical_factor_id %||% NULL,
    factor_names = factor_names
  )
  canonical_level_id <- cs_foundation_normalize_level_ids(
    values = canonical_level_id %||% NULL,
    factor_names = factor_names,
    names_list = names_list_use
  )
  list(
    experiment_id = as.character(experiment_id %||% "embedding_request"),
    W = W_df,
    X = X_use,
    names_list = names_list_use,
    factor_names = factor_names,
    factor_levels = vapply(names_list_use, function(x) length(x[[1]]), integer(1)),
    mode = mode_use,
    pair_id = pair_id,
    profile_order = profile_order,
    canonical_factor_id = canonical_factor_id,
    canonical_level_id = canonical_level_id
  )
}

cs_foundation_select_group <- function(foundation_model,
                                       group_key = NULL,
                                       mode = NULL,
                                       likelihood = NULL,
                                       n_outcomes = NULL) {
  group_keys <- names(foundation_model$groups %||% list())
  if (length(group_keys) < 1L) {
    stop("Foundation model does not contain any groups.", call. = FALSE)
  }
  if (!is.null(group_key)) {
    group_key <- as.character(group_key)
    group <- foundation_model$groups[[group_key]] %||% NULL
    if (is.null(group)) {
      stop(
        sprintf(
          "Unknown foundation group_key '%s'. Available group keys: %s",
          group_key,
          paste(group_keys, collapse = ", ")
        ),
        call. = FALSE
      )
    }
    return(group)
  }

  group_info <- data.frame(
    group_key = group_keys,
    mode = vapply(foundation_model$groups, `[[`, character(1), "mode"),
    likelihood = vapply(foundation_model$groups, `[[`, character(1), "likelihood"),
    n_outcomes = vapply(foundation_model$groups, function(x) as.integer(x$n_outcomes), integer(1)),
    stringsAsFactors = FALSE
  )
  if (!is.null(mode)) {
    group_info <- group_info[group_info$mode == as.character(mode), , drop = FALSE]
  }
  if (!is.null(likelihood)) {
    like_use <- tolower(as.character(likelihood))
    group_info <- group_info[tolower(group_info$likelihood) == like_use, , drop = FALSE]
  }
  if (!is.null(n_outcomes)) {
    group_info <- group_info[group_info$n_outcomes == as.integer(n_outcomes), , drop = FALSE]
  }
  if (nrow(group_info) == 1L) {
    return(foundation_model$groups[[group_info$group_key[[1L]]]])
  }
  if (nrow(group_info) < 1L) {
    stop(
      sprintf(
        paste(
          "No foundation group matched the supplied criteria.",
          "Available group keys: %s"
        ),
        paste(group_keys, collapse = ", ")
      ),
      call. = FALSE
    )
  }
  stop(
    sprintf(
      paste(
        "Foundation group selection is ambiguous.",
        "Supply group_key, or further narrow with mode/likelihood/n_outcomes.",
        "Matching group keys: %s"
      ),
      paste(group_info$group_key, collapse = ", ")
    ),
    call. = FALSE
  )
}

cs_foundation_map_request_to_group <- function(group,
                                               request,
                                               strict_schema_match = FALSE) {
  registry <- group$schema_registry %||% NULL
  if (is.null(registry) || is.null(registry$slot_table) || is.null(registry$slot_level_keys)) {
    stop("Selected foundation group is missing schema metadata for extraction.", call. = FALSE)
  }
  slot_table <- registry$slot_table
  slot_lookup <- stats::setNames(slot_table$slot_name, slot_table$slot_key)
  display_lookup <- stats::setNames(slot_table$slot_name, slot_table$display_label)
  request_experiment_id <- request$experiment_id
  if (length(request_experiment_id) > 1L) {
    request_experiment_id <- unique(stats::na.omit(as.character(request_experiment_id)))
    if (length(request_experiment_id) > 1L) {
      stop(
        paste(
          "Raw foundation extraction expects one local experiment schema per call.",
          "Supply a scalar experiment_id or one repeated value."
        ),
        call. = FALSE
      )
    }
  }
  request_experiment_id <- as.character(request_experiment_id)
  if (length(request_experiment_id) < 1L || all(is.na(request_experiment_id))) {
    request_experiment_id <- "embedding_request"
  } else {
    request_experiment_id <- request_experiment_id[[1L]]
  }
  request_map <- request
  request_map$experiment_id <- request_experiment_id
  local_map <- cs_foundation_build_local_factor_map(request_map)
  n_rows <- nrow(request$W)
  out <- as.data.frame(
    setNames(
      replicate(nrow(slot_table), rep(NA_character_, n_rows), simplify = FALSE),
      slot_table$slot_name
    ),
    stringsAsFactors = FALSE
  )
  unmatched_factors <- character(0)
  unmatched_levels <- character(0)
  used_slots <- character(0)
  factor_order_names <- character(0)

  for (factor_name in request$factor_names) {
    slot_name <- if (factor_name %in% slot_table$slot_name) {
      factor_name
    } else if (factor_name %in% slot_table$display_label) {
      matched_slot <- unname(display_lookup[factor_name])
      if (length(matched_slot) != 1L || is.na(matched_slot[[1L]]) || !nzchar(matched_slot[[1L]])) {
        NA_character_
      } else {
        matched_slot[[1L]]
      }
    } else {
      slot_key <- local_map$factor_map[[factor_name]]$slot_key
      matched_slot <- unname(slot_lookup[slot_key])
      if (length(matched_slot) < 1L || is.na(matched_slot[[1L]]) || !nzchar(matched_slot[[1L]])) {
        NA_character_
      } else {
        matched_slot[[1L]]
      }
    }
    if (is.na(slot_name) || !nzchar(slot_name)) {
      unmatched_factors <- c(unmatched_factors, factor_name)
      next
    }
    if (slot_name %in% used_slots) {
      stop(
        sprintf(
          "Multiple local factors map to pooled slot '%s'. Adjust canonical ids or extract after adaptation.",
          slot_name
        ),
        call. = FALSE
      )
    }
    used_slots <- c(used_slots, slot_name)
    factor_order_names <- c(factor_order_names, slot_name)

    level_key_map <- local_map$level_map[[factor_name]]
    raw_vals <- as.character(request$W[[factor_name]])
    allowed_keys <- registry$slot_level_keys[[slot_name]] %||% character(0)
    mapped_vals <- if (factor_name %in% slot_table$slot_name) {
      raw_vals
    } else {
      unname(level_key_map[raw_vals])
    }
    invalid <- !is.na(raw_vals) & (is.na(mapped_vals) | !(mapped_vals %in% allowed_keys))
    if (any(invalid)) {
      bad_vals <- unique(raw_vals[invalid])
      unmatched_levels <- c(unmatched_levels, paste0(factor_name, ":", bad_vals))
    }
    mapped_vals[is.na(mapped_vals) | !(mapped_vals %in% allowed_keys)] <- NA_character_
    out[[slot_name]] <- mapped_vals
  }

  if (length(used_slots) < 1L) {
    stop(
      "No local factors matched the selected foundation group. Supply canonical ids or use an adapted predictor.",
      call. = FALSE
    )
  }
  if (isTRUE(strict_schema_match) && length(unmatched_factors) > 0L) {
    stop(
      sprintf(
        "Unmatched local factors for selected foundation group: %s",
        paste(unique(unmatched_factors), collapse = ", ")
      ),
      call. = FALSE
    )
  }
  if (isTRUE(strict_schema_match) && length(unmatched_levels) > 0L) {
    stop(
      sprintf(
        "Unmatched local factor levels for selected foundation group: %s",
        paste(unique(unmatched_levels), collapse = ", ")
      ),
      call. = FALSE
    )
  }

  list(
    W_group = out,
    factor_order = neural_factor_order_from_names(factor_order_names, slot_table$slot_name),
    unmatched_factors = unique(unmatched_factors),
    unmatched_levels = unique(unmatched_levels)
  )
}

cs_foundation_extract_group_covariates <- function(group,
                                                   request,
                                                   allow_extra_covariates = TRUE) {
  base_names <- group$x_schema$base_x_names %||% character(0)
  if (is.null(request$X)) {
    if (length(base_names) < 1L) {
      return(NULL)
    }
    out <- matrix(0, nrow = nrow(request$W), ncol = length(base_names))
    colnames(out) <- base_names
    return(as.data.frame(out, check.names = FALSE))
  }
  request_x <- as.data.frame(request$X, check.names = FALSE)
  extra_names <- setdiff(colnames(request_x), base_names)
  if (length(extra_names) > 0L && !isTRUE(allow_extra_covariates)) {
    stop(
      sprintf(
        "Newdata includes covariates absent from the selected foundation group: %s",
        paste(extra_names, collapse = ", ")
      ),
      call. = FALSE
    )
  }
  aligned <- cs_foundation_align_covariate_block(
    schema_names = base_names,
    X_mat = as.matrix(request_x),
    n_rows = nrow(request$W)
  )
  as.data.frame(aligned$values, check.names = FALSE)
}

#' @rdname extract_embeddings
#' @param group_key Optional internal foundation group key to extract from.
#' @param mode Optional mode filter used for foundation-group selection.
#' @param likelihood Optional likelihood filter used for foundation-group selection.
#' @param n_outcomes Optional outcome-count filter used for foundation-group selection.
#' @param experiment_id Optional experiment identifier used for experiment-token lookup.
#' @param names_list Optional local factor-level metadata for raw foundation extraction.
#' @param p_list Optional \code{p_list} used to derive \code{names_list} when needed.
#' @param canonical_factor_id Optional canonical factor ids used to map local factors into a pooled foundation group.
#' @param canonical_level_id Optional canonical level ids used to map local levels into a pooled foundation group.
#' @param foundation_adaptation_control Optional list controlling schema matching when extracting from a raw foundation group.
#' @param conda_env Conda env name for neural extraction when params must be reconstructed.
#' @param conda_env_required Require the conda env to exist.
#' @export
extract_embeddings.conjoint_foundation_model <- function(object,
                                                         newdata,
                                                         group_key = NULL,
                                                         mode = c("auto", "pairwise", "single"),
                                                         likelihood = NULL,
                                                         n_outcomes = NULL,
                                                         experiment_id = NULL,
                                                         names_list = NULL,
                                                         p_list = NULL,
                                                         canonical_factor_id = NULL,
                                                         canonical_level_id = NULL,
                                                         foundation_adaptation_control = NULL,
                                                         conda_env = "strategize_env",
                                                         conda_env_required = TRUE,
                                                         ...) {
  if (!inherits(object, "conjoint_foundation_model")) {
    stop("extract_embeddings.conjoint_foundation_model() requires a conjoint_foundation_model.",
         call. = FALSE)
  }

  factor_names <- NULL
  if (!is.null(names_list) || !is.null(p_list)) {
    factor_names <- names(cs_foundation_normalize_names_list_local(
      names_list = names_list,
      W = if (is.list(newdata) && !is.data.frame(newdata)) newdata$W else newdata,
      p_list = p_list
    ))
  }
  unpacked <- cs_foundation_unpack_embedding_newdata(
    newdata = newdata,
    factor_names = factor_names
  )
  experiment_id_use <- experiment_id %||% unpacked$experiment_id %||% "embedding_request"
  mode_use <- cs_foundation_mode(match.arg(mode), unpacked$pair_id)
  group <- cs_foundation_select_group(
    foundation_model = object,
    group_key = group_key,
    mode = mode_use,
    likelihood = likelihood %||% NULL,
    n_outcomes = n_outcomes %||% NULL
  )
  group <- cs_foundation_prepare_group_fit(
    group = group,
    conda_env = conda_env,
    conda_env_required = conda_env_required
  )

  request <- cs_foundation_normalize_request(
    W = unpacked$W,
    X = unpacked$X,
    mode = group$mode,
    pair_id = unpacked$pair_id,
    profile_order = unpacked$profile_order,
    experiment_id = experiment_id_use,
    names_list = names_list,
    p_list = p_list,
    canonical_factor_id = canonical_factor_id,
    canonical_level_id = canonical_level_id
  )
  adaptation_control <- modifyList(
    cs_foundation_default_adaptation_control(),
    foundation_adaptation_control %||% list()
  )
  mapped <- cs_foundation_map_request_to_group(
    group = group,
    request = request,
    strict_schema_match = isTRUE(adaptation_control$strict_schema_match)
  )
  X_group <- cs_foundation_extract_group_covariates(
    group = group,
    request = request,
    allow_extra_covariates = isTRUE(adaptation_control$allow_extra_covariates)
  )

  tmp_predictor <- structure(
    list(
      model_type = "neural",
      mode = group$mode,
      encoder = group$encoder,
      fit = group$fit,
      metadata = list(
        conda_env = conda_env,
        conda_env_required = conda_env_required,
        text_embedding_fn = adaptation_control$text_embedding_fn %||%
          object$metadata$text_embedding_fn %||% NULL
      )
    ),
    class = "strategic_predictor"
  )

  cs2step_neural_extract_internal(
    object = tmp_predictor,
    W_new = mapped$W_group,
    X_new = X_group,
    factor_order_new = mapped$factor_order,
    experiment_id = request$experiment_id,
    experiment_description = unpacked$experiment_description %||% NULL,
    pair_id = request$pair_id,
    profile_order = request$profile_order,
    source_class = "conjoint_foundation_model",
    extra_metadata = list(
      foundation_group_key = group$group_key,
      foundation_experiment_ids = group$experiment_ids,
      unmatched_factors = mapped$unmatched_factors,
      unmatched_levels = mapped$unmatched_levels
    )
  )
}
