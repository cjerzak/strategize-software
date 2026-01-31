neural_get_index <- function(model_info,
                             party_label = NULL,
                             levels_field,
                             map_field) {
  if (is.null(model_info) || is.null(model_info[[levels_field]])) {
    return(ai(0L))
  }
  if (!is.null(model_info[[map_field]]) && !is.null(party_label)) {
    key <- as.character(party_label)
    if (key %in% names(model_info[[map_field]])) {
      return(ai(model_info[[map_field]][[key]]))
    }
  }
  if (is.null(party_label)) {
    return(ai(0L))
  }
  idx <- match(as.character(party_label), model_info[[levels_field]]) - 1L
  if (is.na(idx)) ai(0L) else ai(idx)
}

neural_get_party_index <- function(model_info, party_label = NULL){
  neural_get_index(model_info,
                   party_label = party_label,
                   levels_field = "party_levels",
                   map_field = "party_index_map")
}

neural_get_resp_party_index <- function(model_info, party_label = NULL){
  neural_get_index(model_info,
                   party_label = party_label,
                   levels_field = "resp_party_levels",
                   map_field = "resp_party_index_map")
}

neural_has_shape <- function(x) {
  tryCatch({
    x$shape
    TRUE
  }, error = function(e) FALSE)
}

neural_stage_index <- function(party_left_idx, party_right_idx, model_info = NULL) {
  mode <- NULL
  if (!is.null(model_info) && !is.null(model_info$stage_mode)) {
    mode <- tolower(as.character(model_info$stage_mode))
  }
  if (length(mode) != 1L || is.na(mode) || !nzchar(mode)) {
    mode <- "same"
  }

  if (neural_has_shape(party_left_idx) || neural_has_shape(party_right_idx)) {
    pl <- strenv$jnp$array(party_left_idx)
    pr <- strenv$jnp$array(party_right_idx)
    same <- strenv$jnp$equal(pl, pr)
    stage <- if (identical(mode, "different")) {
      strenv$jnp$logical_not(same)
    } else {
      same
    }
    return(strenv$jnp$astype(stage, strenv$jnp$int32))
  }

  same <- ai(party_left_idx) == ai(party_right_idx)
  if (identical(mode, "different")) {
    return(ifelse(same, 0L, 1L))
  }
  ifelse(same, 1L, 0L)
}

neural_matchup_index <- function(party_left_idx, party_right_idx, model_info){
  if (is.null(model_info)) {
    return(NULL)
  }
  n_party_levels <- if (!is.null(model_info$n_party_levels)) {
    as.integer(model_info$n_party_levels)
  } else if (!is.null(model_info$party_levels)) {
    length(model_info$party_levels)
  } else {
    NA_integer_
  }
  if (is.na(n_party_levels) || n_party_levels < 1L) {
    return(NULL)
  }
  pl <- strenv$jnp$array(party_left_idx)
  pr <- strenv$jnp$array(party_right_idx)
  p_min <- strenv$jnp$minimum(pl, pr)
  p_max <- strenv$jnp$maximum(pl, pr)
  half_term <- strenv$jnp$floor_divide(p_min * (p_min - 1L), ai(2L))
  idx <- p_min * ai(n_party_levels) - half_term + (p_max - p_min)
  strenv$jnp$astype(idx, strenv$jnp$int32)
}

neural_logits_to_q <- function(logits, likelihood){
  if (likelihood == "bernoulli") {
    prob <- strenv$jax$nn$sigmoid(strenv$jnp$squeeze(logits, axis = 1L))
    return(strenv$jnp$reshape(prob, list(-1L, 1L)))
  }
  if (likelihood == "categorical") {
    probs <- strenv$jax$nn$softmax(logits, axis = -1L)
    prob <- strenv$jnp$take(probs, 1L, axis = 1L)
    return(strenv$jnp$reshape(prob, list(-1L, 1L)))
  }
  mu <- strenv$jnp$squeeze(logits, axis = 1L)
  strenv$jnp$reshape(mu, list(-1L, 1L))
}

apply_implicit_parameterization_jnp <- function(p_sub,
                                                implicit = FALSE,
                                                axis = -1L,
                                                clip = TRUE) {
  p_sub <- strenv$jnp$array(p_sub)
  if (!isTRUE(implicit)) {
    return(p_sub)
  }
  axis_use <- as.integer(axis)
  ndim <- length(p_sub$shape)
  if (axis_use < 0L) {
    axis_use <- axis_use + ndim
  }
  holdout <- strenv$jnp$array(1., dtype = p_sub$dtype) -
    strenv$jnp$sum(p_sub, axis = axis_use)
  if (isTRUE(clip)) {
    holdout <- strenv$jnp$clip(holdout, 0., 1.)
  }
  holdout_exp <- strenv$jnp$expand_dims(holdout, axis = axis_use)
  strenv$jnp$concatenate(list(p_sub, holdout_exp), axis = axis_use)
}

neural_params_from_theta <- function(theta_vec, model_info){
  if (is.null(model_info)) {
    stop("neural_params_from_theta requires a non-null model_info.", call. = FALSE)
  }
  if (is.null(theta_vec)) {
    return(model_info$params)
  }
  schema_missing <- is.null(model_info$param_names) ||
    is.null(model_info$param_offsets) ||
    is.null(model_info$param_sizes) ||
    is.null(model_info$param_shapes)
  if (isTRUE(schema_missing)) {
    if (!is.null(model_info$params) &&
        !is.null(model_info$n_factors) &&
        !is.null(model_info$model_depth)) {
      schema <- neural_build_param_schema(
        params = model_info$params,
        n_factors = model_info$n_factors,
        model_depth = model_info$model_depth
      )
      model_info$param_names <- schema$param_names
      model_info$param_shapes <- schema$param_shapes
      model_info$param_sizes <- schema$param_sizes
      model_info$param_offsets <- schema$param_offsets
      if (is.null(model_info$n_params)) {
        model_info$n_params <- schema$n_params
      }
    } else {
      stop("model_info is missing parameter schema; cannot unpack theta_vec.", call. = FALSE)
    }
  }
  theta_vec <- strenv$jnp$reshape(theta_vec, list(-1L))
  offsets_num <- as.integer(unlist(model_info$param_offsets))
  sizes_num <- as.integer(unlist(model_info$param_sizes))
  if (length(offsets_num) != length(sizes_num)) {
    stop("param_offsets and param_sizes length mismatch in model_info.", call. = FALSE)
  }
  theta_len <- as.integer(theta_vec$shape[[1]])
  max_end <- if (length(sizes_num) > 0L) max(offsets_num + sizes_num) else 0L
  expected_total <- if (!is.null(model_info$n_params)) as.integer(model_info$n_params) else max_end
  if (is.finite(theta_len) && is.finite(expected_total) && theta_len != expected_total) {
    stop(sprintf("theta_vec length (%d) does not match expected parameter total (%d).",
                 theta_len, expected_total),
         call. = FALSE)
  }
  if (is.finite(theta_len) && is.finite(max_end) && theta_len < max_end) {
    stop(sprintf("theta_vec length (%d) is shorter than required (%d).",
                 theta_len, max_end),
         call. = FALSE)
  }
  params <- list()
  param_names <- model_info$param_names
  param_offsets <- model_info$param_offsets
  param_sizes <- model_info$param_sizes
  param_shapes <- model_info$param_shapes
  for (i_ in seq_along(param_names)) {
    start <- as.integer(param_offsets[[i_]])
    size <- as.integer(param_sizes[[i_]])
    if (is.finite(theta_len) && is.finite(start) && is.finite(size) &&
        (start < 0L || (start + size) > theta_len)) {
      stop(sprintf("theta_vec slice for '%s' (%d..%d) exceeds length (%d).",
                   param_names[[i_]], start, start + size - 1L, theta_len),
           call. = FALSE)
    }
    idx <- strenv$jnp$arange(ai(start), ai(start + size))
    slice <- strenv$jnp$take(theta_vec, idx, axis = 0L)
    shape_use <- param_shapes[[i_]]
    if (length(shape_use) == 0L) {
      shape_use <- c(1L)
    }
    shape_size <- as.integer(prod(shape_use))
    if (is.finite(size) && is.finite(shape_size) && size != shape_size) {
      stop(sprintf("Param '%s' size (%d) does not match shape product (%d).",
                   param_names[[i_]], size, shape_size),
           call. = FALSE)
    }
    params[[param_names[[i_]]]] <- strenv$jnp$reshape(slice, as.integer(shape_use))
  }
  params
}

neural_build_param_schema <- function(params,
                                      n_factors,
                                      model_depth) {
  if (is.null(params) || !is.list(params)) {
    stop("neural_build_param_schema requires a params list.", call. = FALSE)
  }
  n_factors <- as.integer(n_factors)
  model_depth <- as.integer(model_depth)
  if (is.na(n_factors) || n_factors < 1L) {
    stop("neural_build_param_schema requires n_factors >= 1.", call. = FALSE)
  }
  if (is.na(model_depth) || model_depth < 1L) {
    stop("neural_build_param_schema requires model_depth >= 1.", call. = FALSE)
  }

  param_names <- c(paste0("E_factor_", seq_len(n_factors)),
                   "E_feature_id",
                   "E_party", "E_resp_party", "E_choice",
                   "E_sep", "E_segment")
  if (!is.null(params$E_stage)) {
    param_names <- c(param_names, "E_stage")
  }
  if (!is.null(params$E_matchup)) {
    param_names <- c(param_names, "E_matchup")
  }
  if (!is.null(params$E_rel)) {
    param_names <- c(param_names, "E_rel")
  }
  if (!is.null(params$W_resp_x)) {
    param_names <- c(param_names, "W_resp_x")
  }
  if (!is.null(params$M_cross)) {
    param_names <- c(param_names, "M_cross")
  }
  if (!is.null(params$W_cross_out)) {
    param_names <- c(param_names, "W_cross_out")
  }
  for (l_ in 1L:model_depth) {
    param_names <- c(param_names,
                     paste0("alpha_attn_l", l_),
                     paste0("alpha_ff_l", l_),
                     paste0("RMS_attn_l", l_),
                     paste0("RMS_ff_l", l_),
                     paste0("W_q_l", l_),
                     paste0("W_k_l", l_),
                     paste0("W_v_l", l_),
                     paste0("W_o_l", l_),
                     paste0("W_ff1_l", l_),
                     paste0("W_ff2_l", l_))
  }
  param_names <- c(param_names, "RMS_final", "W_out", "b_out")
  if (!is.null(params$sigma)) {
    param_names <- c(param_names, "sigma")
  }
  param_names <- param_names[param_names %in% names(params)]

  param_shapes <- lapply(param_names, function(name) {
    shape <- tryCatch(reticulate::py_to_r(params[[name]]$shape), error = function(e) NULL)
    if (is.null(shape)) integer(0) else as.integer(shape)
  })
  param_sizes <- vapply(param_shapes, function(shape) {
    if (length(shape) == 0L) {
      1L
    } else {
      as.integer(prod(shape))
    }
  }, integer(1))
  param_offsets <- as.integer(cumsum(c(0L, param_sizes))[seq_len(length(param_sizes))])
  param_total <- sum(param_sizes)

  list(
    param_names = param_names,
    param_shapes = param_shapes,
    param_sizes = param_sizes,
    param_offsets = param_offsets,
    n_params = ai(param_total)
  )
}

neural_flatten_params <- function(params, schema, dtype = NULL) {
  if (is.null(dtype)) {
    dtype <- strenv$jnp$float32
  }
  parts <- lapply(schema$param_names, function(name) {
    strenv$jnp$ravel(params[[name]])
  })
  if (length(parts) == 0L) {
    return(strenv$jnp$array(numeric(0), dtype = dtype))
  }
  strenv$jnp$concatenate(parts, axis = 0L)
}

neural_normalize_cross_encoder_mode <- function(value) {
  if (is.null(value)) {
    return(NULL)
  }
  if (isTRUE(value)) {
    return("term")
  }
  if (identical(value, FALSE)) {
    return("none")
  }
  if (is.character(value)) {
    mode <- tolower(as.character(value))
    if (length(mode) == 1L && !is.na(mode) && nzchar(mode)) {
      if (mode %in% c("none", "term", "full")) {
        return(mode)
      }
      if (mode %in% c("true", "false")) {
        return(ifelse(mode == "true", "term", "none"))
      }
    }
  }
  NA_character_
}

neural_cross_encoder_mode <- function(model_info) {
  mode <- neural_normalize_cross_encoder_mode(model_info$cross_candidate_encoder_mode)
  if (is.character(mode) && length(mode) == 1L && !is.na(mode) && nzchar(mode)) {
    return(mode)
  }
  if (isTRUE(model_info$cross_candidate_encoder)) {
    return("term")
  }
  "none"
}

neural_rms_norm <- function(x, g, model_dims, eps = 1e-6) {
  if (is.null(g)) {
    return(x)
  }
  mean_sq <- strenv$jnp$mean(x * x, axis = -1L, keepdims = TRUE)
  inv_rms <- strenv$jnp$reciprocal(strenv$jnp$sqrt(mean_sq + eps))
  g_use <- strenv$jnp$reshape(g, list(1L, 1L, ai(model_dims)))
  x * inv_rms * g_use
}

neural_param_or_default <- function(params, name, default) {
  val <- params[[name]]
  if (is.null(val)) {
    return(default)
  }
  val
}

neural_linear_head <- function(phi, W_out, b_out = NULL, dtype = NULL) {
  logits <- strenv$jnp$einsum("nm,mo->no", phi, W_out)
  if (is.null(b_out)) {
    dtype_use <- dtype
    if (is.null(dtype_use)) {
      dtype_use <- tryCatch(W_out$dtype, error = function(e) NULL)
    }
    if (is.null(dtype_use)) {
      dtype_use <- strenv$dtj
    }
    b_out <- strenv$jnp$zeros(list(ai(W_out$shape[[2]])), dtype = dtype_use)
  }
  logits + b_out
}

neural_apply_cross_term <- function(logits, phi_left, phi_right,
                                    M_cross, W_cross_out = NULL,
                                    out_dim = NULL, dtype = NULL) {
  if (is.null(M_cross)) {
    return(logits)
  }
  cross_term <- strenv$jnp$einsum("nm,mp,np->n", phi_left, M_cross, phi_right)
  cross_term <- strenv$jnp$reshape(cross_term, list(-1L, 1L))
  if (is.null(W_cross_out)) {
    if (is.null(out_dim)) {
      out_dim <- tryCatch(ai(logits$shape[[2]]), error = function(e) NULL)
    }
    if (is.null(out_dim)) {
      out_dim <- ai(1L)
    }
    dtype_use <- dtype
    if (is.null(dtype_use)) {
      dtype_use <- tryCatch(logits$dtype, error = function(e) NULL)
    }
    if (is.null(dtype_use)) {
      dtype_use <- strenv$dtj
    }
    cross_out <- strenv$jnp$zeros(list(1L, ai(out_dim)), dtype = dtype_use)
  } else {
    cross_out <- strenv$jnp$reshape(W_cross_out, list(1L, -1L))
  }
  logits + cross_term * cross_out
}

neural_add_segment_embedding <- function(tokens, segment_idx, model_info, params = NULL){
  if (is.null(params)) {
    params <- model_info$params
  }
  if (is.null(params$E_segment)) {
    return(tokens)
  }
  seg_vec <- strenv$jnp$take(params$E_segment, ai(segment_idx), axis = 0L)
  seg_tok <- strenv$jnp$reshape(seg_vec, list(1L, 1L, ai(model_info$model_dims)))
  tokens + seg_tok
}

neural_build_sep_token <- function(model_info, n_batch = NULL, params = NULL){
  if (is.null(params)) {
    params <- model_info$params
  }
  sep_vec <- if (!is.null(params$E_sep)) {
    params$E_sep
  } else {
    strenv$jnp$zeros(list(ai(model_info$model_dims)), dtype = strenv$dtj)
  }
  sep_tok <- strenv$jnp$reshape(sep_vec, list(1L, 1L, ai(model_info$model_dims)))
  if (is.null(n_batch)) {
    return(sep_tok)
  }
  sep_tok * strenv$jnp$ones(list(ai(n_batch), 1L, 1L))
}

add_party_rel_tokens <- function(tokens,
                                       party_idx,
                                       model_info,
                                       resp_party_idx = NULL,
                                       params = NULL,
                                       use_role = FALSE,
                                       role_id = NULL,
                                       require_party = TRUE){
  if (is.null(params)) {
    params <- model_info$params
  }
  tok_ndim <- length(tokens$shape)

  if (!is.null(params$E_feature_id)) {
    dims <- ai(model_info$model_dims)
    feature_tok <- if (tok_ndim == 3L) {
      strenv$jnp$reshape(params$E_feature_id, list(1L, tokens$shape[[2]], dims))
    } else if (tok_ndim == 2L) {
      strenv$jnp$reshape(params$E_feature_id, list(tokens$shape[[1]], dims))
    } else {
      NULL
    }
    if (!is.null(feature_tok)) {
      tokens <- tokens + feature_tok
    }
  }

  if (isTRUE(require_party) && is.null(params$E_party)) {
    stop("E_party is required for party/rel tokens.")
  }

  party_idx_arr <- strenv$jnp$array(party_idx)
  party_idx_arr <- strenv$jnp$astype(party_idx_arr, strenv$jnp$int32)

  party_vec <- NULL
  if (!is.null(params$E_party)) {
    party_vec <- strenv$jnp$take(params$E_party, party_idx_arr, axis = 0L)
  }

  if (isTRUE(use_role) && !is.null(params$E_role)) {
    role_id_use <- if (is.null(role_id)) 0L else role_id
    n_roles <- ai(params$E_role$shape[[1]])
    role_use <- if (ai(role_id_use) >= n_roles) 0L else ai(role_id_use)
    role_vec <- strenv$jnp$take(params$E_role, strenv$jnp$array(ai(role_use)), axis = 0L)

    dims <- ai(model_info$model_dims)
    role_add <- if (tok_ndim == 3L) {
      strenv$jnp$reshape(role_vec, list(1L, 1L, dims))
    } else {
      strenv$jnp$reshape(role_vec, list(1L, dims))
    }
    tokens <- tokens + role_add

    if (!is.null(party_vec)) {
      party_vec <- party_vec + strenv$jnp$reshape(role_vec, list(dims))
    }
  }

  if (!is.null(party_vec)) {
    dims <- ai(model_info$model_dims)
    party_tok <- if (tok_ndim == 3L) {
      strenv$jnp$reshape(party_vec, list(tokens$shape[[1]], 1L, dims))
    } else {
      strenv$jnp$reshape(party_vec, list(1L, dims))
    }
    tokens <- strenv$jnp$concatenate(list(tokens, party_tok),
                                     axis = if (tok_ndim == 3L) 1L else 0L)
  }

  if (!is.null(params$E_rel)) {
    rel_idx <- if (is.null(model_info$cand_party_to_resp_idx) || is.null(resp_party_idx)) {
      if (tok_ndim == 3L) {
        strenv$jnp$full(list(tokens$shape[[1]]), ai(2L))
      } else {
        ai(2L)
      }
    } else {
      cand_map <- strenv$jnp$atleast_1d(model_info$cand_party_to_resp_idx)
      cand_resp_idx <- strenv$jnp$take(cand_map, party_idx_arr, axis = 0L)
      is_known <- cand_resp_idx >= 0L
      resp_arr <- strenv$jnp$array(resp_party_idx)
      is_match <- strenv$jnp$equal(cand_resp_idx, resp_arr)
      strenv$jnp$where(is_match, ai(0L),
                       strenv$jnp$where(is_known, ai(1L), ai(2L)))
    }
    rel_idx <- strenv$jnp$astype(strenv$jnp$array(rel_idx), strenv$jnp$int32)
    rel_vec <- strenv$jnp$take(params$E_rel, rel_idx, axis = 0L)
    dims <- ai(model_info$model_dims)
    rel_tok <- if (tok_ndim == 3L) {
      strenv$jnp$reshape(rel_vec, list(tokens$shape[[1]], 1L, dims))
    } else {
      strenv$jnp$reshape(rel_vec, list(1L, dims))
    }
    tokens <- strenv$jnp$concatenate(list(tokens, rel_tok),
                                     axis = if (tok_ndim == 3L) 1L else 0L)
  }

  tokens
}

add_context_tokens <- function(model_info,
                                      resp_party_idx,
                                      stage_idx = NULL,
                                      matchup_idx = NULL,
                                      resp_cov = NULL,
                                      params = NULL,
                                      batch = FALSE){
  if (is.null(params)) {
    params <- model_info$params
  }
  token_list <- list()
  dims <- ai(model_info$model_dims)
  is_batch <- isTRUE(batch)

  resp_party_idx_use <- if (is.null(resp_party_idx)) 0L else resp_party_idx
  if (!is_batch) {
    resp_party_idx_use <- ai(resp_party_idx_use)
    resp_party_idx_use <- strenv$jnp$atleast_1d(strenv$jnp$array(resp_party_idx_use))
  } else {
    resp_party_idx_use <- strenv$jnp$atleast_1d(resp_party_idx_use)
  }

  stage_idx_use <- NULL
  if (!is.null(stage_idx)) {
    if (!is_batch) {
      if (neural_has_shape(stage_idx)) {
        stage_use <- strenv$jnp$maximum(strenv$jnp$array(stage_idx), ai(0L))
        stage_idx_use <- strenv$jnp$atleast_1d(stage_use)
      } else {
        stage_use <- ai(stage_idx)
        if (ai(stage_use) < 0L) stage_use <- 0L
        stage_idx_use <- strenv$jnp$atleast_1d(strenv$jnp$array(stage_use))
      }
    } else {
      stage_idx_use <- strenv$jnp$atleast_1d(stage_idx)
    }
  }

  matchup_idx_use <- NULL
  if (!is.null(matchup_idx)) {
    matchup_idx_use <- strenv$jnp$atleast_1d(matchup_idx)
  }

  N_batch <- 1L
  if (is_batch) {
    N_batch <- tryCatch(ai(resp_party_idx_use$shape[[1L]]), error = function(e) 1L)
  }

  if (!is.null(params$E_stage) && !is.null(stage_idx_use)) {
    stage_vec <- params$E_stage[resp_party_idx_use, stage_idx_use]
    stage_tok <- strenv$jnp$reshape(stage_vec, list(-1L, 1L, dims))
    token_list[[length(token_list) + 1L]] <- stage_tok
  }
  if (!is.null(params$E_resp_party)) {
    resp_vec <- strenv$jnp$take(params$E_resp_party, resp_party_idx_use, axis = 0L)
    resp_tok <- strenv$jnp$reshape(resp_vec, list(-1L, 1L, dims))
    token_list[[length(token_list) + 1L]] <- resp_tok
  }
  if (!is.null(params$E_matchup) && !is.null(matchup_idx_use)) {
    matchup_vec <- strenv$jnp$take(params$E_matchup, matchup_idx_use, axis = 0L)
    matchup_tok <- strenv$jnp$reshape(matchup_vec, list(-1L, 1L, dims))
    token_list[[length(token_list) + 1L]] <- matchup_tok
  }
  if (!is.null(params$W_resp_x)) {
    use_resp_cov <- TRUE
    if (!is_batch) {
      use_resp_cov <- !is.null(model_info$resp_cov_mean) &&
        ai(model_info$n_resp_covariates) > 0L
    }
    if (use_resp_cov) {
      if (is.null(resp_cov) && !is.null(model_info$resp_cov_mean)) {
        resp_cov <- model_info$resp_cov_mean
      }
      if (!is.null(resp_cov)) {
        resp_cov_mat <- strenv$jnp$atleast_2d(resp_cov)
        if (ai(resp_cov_mat$shape[[2]]) > 0L) {
          if (ai(resp_cov_mat$shape[[1]]) == 1L && is_batch && N_batch > 1L) {
            resp_cov_mat <- resp_cov_mat * strenv$jnp$ones(list(N_batch, 1L))
          }
          cov_vec <- strenv$jnp$einsum("nc,cm->nm", resp_cov_mat, params$W_resp_x)
          cov_tok <- strenv$jnp$reshape(cov_vec, list(-1L, 1L, dims))
          token_list[[length(token_list) + 1L]] <- cov_tok
        }
      }
    }
  }

  if (length(token_list) == 0L) {
    return(NULL)
  }
  strenv$jnp$concatenate(token_list, axis = 1L)
}

neural_build_candidate_tokens_hard <- function(X_idx, party_idx, model_info,
                                               resp_party_idx = NULL,
                                               params = NULL){
  if (is.null(params)) {
    params <- model_info$params
  }
  D_local <- ai(X_idx$shape[[2]])
  token_list <- vector("list", D_local)
  for (d_ in 1L:D_local) {
    E_d <- params[[paste0("E_factor_", d_)]]
    idx_d <- strenv$jnp$take(X_idx, ai(d_ - 1L), axis = 1L)
    token_list[[d_]] <- strenv$jnp$take(E_d, idx_d, axis = 0L)
  }
  tokens <- strenv$jnp$stack(token_list, axis = 1L)
  add_party_rel_tokens(tokens,
                       party_idx = party_idx,
                       resp_party_idx = resp_party_idx,
                       model_info = model_info,
                       params = params,
                       require_party = FALSE)
}

neural_build_context_tokens_batch <- function(model_info,
                                              resp_party_idx,
                                              stage_idx = NULL,
                                              matchup_idx = NULL,
                                              resp_cov = NULL,
                                              params = NULL){
  add_context_tokens(model_info = model_info,
                     resp_party_idx = resp_party_idx,
                     stage_idx = stage_idx,
                     matchup_idx = matchup_idx,
                     resp_cov = resp_cov,
                     params = params,
                     batch = TRUE)
}

neural_build_candidate_tokens_soft <- function(pi_vec, party_idx, role_id, model_info, params = NULL,
                                               use_role = FALSE, resp_party_idx = NULL){
  if (is.null(params)) {
    params <- model_info$params
  }
  pi_vec <- strenv$jnp$reshape(pi_vec, list(-1L))
  token_list <- vector("list", ai(model_info$n_factors))
  for (d_ in seq_len(ai(model_info$n_factors))) {
    idx <- model_info$factor_index_list[[d_]]
    idx <- strenv$jnp$atleast_1d(idx)
    p_sub <- strenv$jnp$take(pi_vec, idx, axis = 0L)
    p_full <- apply_implicit_parameterization_jnp(p_sub,
                                                  implicit = isTRUE(model_info$implicit),
                                                  axis = 0L,
                                                  clip = TRUE)
    E_d <- params[[paste0("E_factor_", d_)]]
    n_p <- ai(p_full$shape[[1]])
    n_e <- ai(E_d$shape[[1]])
    if (!is.na(n_p) && !is.na(n_e) && n_p != n_e) {
      if (n_e > n_p) {
        pad_n <- ai(n_e - n_p)
        pad <- strenv$jnp$zeros(list(pad_n), dtype = strenv$dtj)
        p_full <- strenv$jnp$concatenate(list(p_full, pad), axis = 0L)
      } else {
        p_full <- strenv$jnp$take(p_full, strenv$jnp$arange(ai(n_e)), axis = 0L)
      }
    }
    token_list[[d_]] <- strenv$jnp$einsum("l,lm->m", p_full, E_d)
  }
  tokens <- strenv$jnp$stack(token_list, axis = 0L)
  tokens <- add_party_rel_tokens(tokens,
                                 party_idx = party_idx,
                                 role_id = role_id,
                                 use_role = use_role,
                                 resp_party_idx = resp_party_idx,
                                 model_info = model_info,
                                 params = params)
  strenv$jnp$reshape(tokens, list(1L, tokens$shape[[1]], model_info$model_dims))
}

neural_build_context_tokens <- function(model_info,
                                        resp_party_idx = NULL,
                                        stage_idx = NULL,
                                        matchup_idx = NULL,
                                        resp_cov_vec = NULL,
                                        params = NULL){
  add_context_tokens(model_info = model_info,
                     resp_party_idx = resp_party_idx,
                     stage_idx = stage_idx,
                     matchup_idx = matchup_idx,
                     resp_cov = resp_cov_vec,
                     params = params,
                     batch = FALSE)
}

neural_build_choice_token <- function(model_info, params = NULL){
  if (is.null(params)) {
    params <- model_info$params
  }
  if (!is.null(params$E_choice)) {
    choice_vec <- params$E_choice
  } else {
    choice_vec <- strenv$jnp$zeros(list(ai(model_info$model_dims)), dtype = strenv$dtj)
  }
  strenv$jnp$reshape(choice_vec, list(1L, 1L, ai(model_info$model_dims)))
}

neural_run_transformer <- function(tokens, model_info, params = NULL){
  if (is.null(params)) {
    params <- model_info$params
  }
  for (l_ in 1L:ai(model_info$model_depth)) {
    Wq <- params[[paste0("W_q_l", l_)]]
    Wk <- params[[paste0("W_k_l", l_)]]
    Wv <- params[[paste0("W_v_l", l_)]]
    Wo <- params[[paste0("W_o_l", l_)]]
    Wff1 <- params[[paste0("W_ff1_l", l_)]]
    Wff2 <- params[[paste0("W_ff2_l", l_)]]
    RMS_attn <- params[[paste0("RMS_attn_l", l_)]]
    RMS_ff <- params[[paste0("RMS_ff_l", l_)]]
    alpha_attn <- neural_param_or_default(params, paste0("alpha_attn_l", l_), 1.0)
    alpha_ff <- neural_param_or_default(params, paste0("alpha_ff_l", l_), 1.0)

    tokens_norm <- neural_rms_norm(tokens, RMS_attn, model_info$model_dims)
    Q <- strenv$jnp$einsum("ntm,mk->ntk", tokens_norm, Wq)
    K <- strenv$jnp$einsum("ntm,mk->ntk", tokens_norm, Wk)
    V <- strenv$jnp$einsum("ntm,mk->ntk", tokens_norm, Wv)

    Qh <- strenv$jnp$reshape(Q, list(Q$shape[[1]], Q$shape[[2]],
                                    ai(model_info$n_heads), ai(model_info$head_dim)))
    Kh <- strenv$jnp$reshape(K, list(K$shape[[1]], K$shape[[2]],
                                    ai(model_info$n_heads), ai(model_info$head_dim)))
    Vh <- strenv$jnp$reshape(V, list(V$shape[[1]], V$shape[[2]],
                                    ai(model_info$n_heads), ai(model_info$head_dim)))
    scale_ <- strenv$jnp$sqrt(strenv$jnp$array(as.numeric(ai(model_info$head_dim))))
    scores <- strenv$jnp$einsum("nqhd,nkhd->nhqk", Qh, Kh) / scale_
    attn <- strenv$jax$nn$softmax(scores, axis = -1L)
    context_h <- strenv$jnp$einsum("nhqk,nkhd->nqhd", attn, Vh)
    context <- strenv$jnp$reshape(context_h, list(context_h$shape[[1]],
                                                  context_h$shape[[2]],
                                                  ai(model_info$model_dims)))
    attn_out <- strenv$jnp$einsum("ntm,mk->ntk", context, Wo)

    h1 <- tokens + alpha_attn * attn_out
    h1_norm <- neural_rms_norm(h1, RMS_ff, model_info$model_dims)
    ff_pre <- strenv$jnp$einsum("ntm,mf->ntf", h1_norm, Wff1)
    ff_act <- strenv$jax$nn$swish(ff_pre)
    ff_out <- strenv$jnp$einsum("ntf,fm->ntm", ff_act, Wff2)
    tokens <- h1 + alpha_ff * ff_out
  }
  tokens <- neural_rms_norm(tokens, params$RMS_final, model_info$model_dims)
  tokens
}

neural_encode_candidate_soft <- function(pi_vec, party_idx, model_info,
                                         resp_party_idx = NULL,
                                         stage_idx = NULL,
                                         matchup_idx = NULL,
                                         resp_cov_vec = NULL,
                                         params = NULL, use_role = FALSE){
  if (is.null(params)) {
    params <- model_info$params
  }
  choice_tok <- neural_build_choice_token(model_info, params)
  resp_tokens <- neural_build_context_tokens(model_info,
                                             resp_party_idx = resp_party_idx,
                                             stage_idx = stage_idx,
                                             matchup_idx = matchup_idx,
                                             resp_cov_vec = resp_cov_vec,
                                             params = params)
  cand_tokens <- neural_build_candidate_tokens_soft(pi_vec, party_idx, 0L, model_info, params,
                                                    use_role = use_role,
                                                    resp_party_idx = resp_party_idx)
  if (!is.null(resp_tokens)) {
    tokens <- strenv$jnp$concatenate(list(choice_tok, resp_tokens, cand_tokens), axis = 1L)
  } else {
    tokens <- strenv$jnp$concatenate(list(choice_tok, cand_tokens), axis = 1L)
  }
  tokens <- neural_run_transformer(tokens, model_info, params)
  choice_out <- strenv$jnp$take(tokens, strenv$jnp$arange(1L), axis = 1L)
  strenv$jnp$squeeze(choice_out, axis = 1L)
}

neural_encode_pair_soft_batched <- function(pi_left, pi_right,
                                            party_left_idx, party_right_idx,
                                            model_info,
                                            resp_party_idx = NULL,
                                            stage_idx = NULL,
                                            matchup_idx = NULL,
                                            resp_cov_vec = NULL,
                                            params = NULL,
                                            use_role = FALSE){
  if (is.null(params)) {
    params <- model_info$params
  }
  choice_tok <- neural_build_choice_token(model_info, params)
  resp_tokens <- neural_build_context_tokens(model_info,
                                             resp_party_idx = resp_party_idx,
                                             stage_idx = stage_idx,
                                             matchup_idx = matchup_idx,
                                             resp_cov_vec = resp_cov_vec,
                                             params = params)
  left_tokens <- neural_build_candidate_tokens_soft(pi_left, party_left_idx, 0L, model_info, params,
                                                    use_role = use_role,
                                                    resp_party_idx = resp_party_idx)
  right_tokens <- neural_build_candidate_tokens_soft(pi_right, party_right_idx, 0L, model_info, params,
                                                     use_role = use_role,
                                                     resp_party_idx = resp_party_idx)
  if (!is.null(resp_tokens)) {
    tokens_left <- strenv$jnp$concatenate(list(choice_tok, resp_tokens, left_tokens), axis = 1L)
    tokens_right <- strenv$jnp$concatenate(list(choice_tok, resp_tokens, right_tokens), axis = 1L)
  } else {
    tokens_left <- strenv$jnp$concatenate(list(choice_tok, left_tokens), axis = 1L)
    tokens_right <- strenv$jnp$concatenate(list(choice_tok, right_tokens), axis = 1L)
  }
  tokens <- strenv$jnp$concatenate(list(tokens_left, tokens_right), axis = 0L)
  tokens <- neural_run_transformer(tokens, model_info, params)
  choice_out <- strenv$jnp$take(tokens, strenv$jnp$arange(1L), axis = 1L)
  phi_all <- strenv$jnp$squeeze(choice_out, axis = 1L)
  idx_left <- strenv$jnp$arange(1L)
  idx_right <- strenv$jnp$arange(1L, 2L)
  list(
    phi_left = strenv$jnp$take(phi_all, idx_left, axis = 0L),
    phi_right = strenv$jnp$take(phi_all, idx_right, axis = 0L)
  )
}

neural_candidate_utility_soft <- function(pi_vec, party_idx,
                                          resp_party_idx, stage_idx,
                                          model_info,
                                          resp_cov_vec = NULL,
                                          params = NULL,
                                          matchup_idx = NULL){
  if (is.null(params)) {
    params <- model_info$params
  }
  phi <- neural_encode_candidate_soft(pi_vec, party_idx, model_info,
                                      resp_party_idx = resp_party_idx,
                                      stage_idx = stage_idx,
                                      matchup_idx = matchup_idx,
                                      resp_cov_vec = resp_cov_vec,
                                      params = params)
  neural_linear_head(phi, params$W_out, params$b_out)
}

neural_predict_pair_soft <- function(pi_left, pi_right,
                                     party_left_idx, party_right_idx,
                                     resp_party_idx, model_info,
                                     resp_cov_vec = NULL,
                                     params = NULL,
                                     return_logits = FALSE){
  if (is.null(params)) {
    params <- model_info$params
  }
  mode <- neural_cross_encoder_mode(model_info)
  use_cross_encoder <- identical(mode, "full")
  use_cross_term <- identical(mode, "term")
  stage_idx <- neural_stage_index(party_left_idx, party_right_idx, model_info)
  matchup_idx <- NULL
  if (!is.null(params$E_matchup)) {
    matchup_idx <- neural_matchup_index(party_left_idx, party_right_idx, model_info)
  }
  if (isTRUE(use_cross_encoder)) {
    choice_tok <- neural_build_choice_token(model_info, params)
    ctx_tokens <- neural_build_context_tokens(model_info,
                                              resp_party_idx = resp_party_idx,
                                              stage_idx = stage_idx,
                                              matchup_idx = matchup_idx,
                                              resp_cov_vec = resp_cov_vec,
                                              params = params)
    left_tokens <- neural_add_segment_embedding(
      neural_build_candidate_tokens_soft(pi_left, party_left_idx, 0L, model_info, params,
                                         resp_party_idx = resp_party_idx),
      0L,
      model_info = model_info,
      params = params
    )
    right_tokens <- neural_add_segment_embedding(
      neural_build_candidate_tokens_soft(pi_right, party_right_idx, 1L, model_info, params,
                                         resp_party_idx = resp_party_idx),
      1L,
      model_info = model_info,
      params = params
    )
    sep_tok <- neural_build_sep_token(model_info, params = params)
    token_parts <- list(choice_tok)
    if (!is.null(ctx_tokens)) {
      token_parts <- c(token_parts, list(ctx_tokens))
    }
    token_parts <- c(token_parts, list(sep_tok, left_tokens, sep_tok, right_tokens))
    tokens <- strenv$jnp$concatenate(token_parts, axis = 1L)
    tokens <- neural_run_transformer(tokens, model_info, params)
    cls_out <- strenv$jnp$take(tokens, strenv$jnp$arange(1L), axis = 1L)
    cls_out <- strenv$jnp$squeeze(cls_out, axis = 1L)
    logits <- neural_linear_head(cls_out, params$W_out, params$b_out)
  } else {
    phi_pair <- neural_encode_pair_soft_batched(pi_left, pi_right,
                                                party_left_idx, party_right_idx,
                                                model_info,
                                                resp_party_idx = resp_party_idx,
                                                stage_idx = stage_idx,
                                                matchup_idx = matchup_idx,
                                                resp_cov_vec = resp_cov_vec,
                                                params = params)
    phi_left <- phi_pair$phi_left
    phi_right <- phi_pair$phi_right
    u_left <- neural_linear_head(phi_left, params$W_out, params$b_out)
    u_right <- neural_linear_head(phi_right, params$W_out, params$b_out)
    logits <- u_left - u_right
    if (isTRUE(use_cross_term)) {
      logits <- neural_apply_cross_term(logits, phi_left, phi_right,
                                        params$M_cross, params$W_cross_out,
                                        out_dim = ai(params$W_out$shape[[2]]))
    }
  }
  if (return_logits) {
    return(logits)
  }
  neural_logits_to_q(logits, model_info$likelihood)
}

neural_predict_single_soft <- function(pi_vec,
                                       party_idx,
                                       resp_party_idx,
                                       model_info,
                                       resp_cov_vec = NULL,
                                       params = NULL){
  logits <- neural_candidate_utility_soft(pi_vec, party_idx,
                                          resp_party_idx, stage_idx = NULL,
                                          model_info = model_info,
                                          resp_cov_vec = resp_cov_vec,
                                          params = params)
  neural_logits_to_q(logits, model_info$likelihood)
}

neural_resolve_model_info <- function(name) {
  if (exists(name, inherits = TRUE)) {
    return(get(name, inherits = TRUE))
  }
  if (exists("strenv", inherits = TRUE)) {
    model_env <- tryCatch(strenv$neural_model_env, error = function(e) NULL)
    if (is.environment(model_env) &&
        exists(name, envir = model_env, inherits = TRUE)) {
      return(get(name, envir = model_env, inherits = TRUE))
    }
  }
  NULL
}

neural_getQStar_single <- function(pi_star_ast,
                                   EST_COEFFICIENTS_tf_ast) {
  model_ast <- neural_resolve_model_info("neural_model_info_ast_jnp")
  if (is.null(model_ast)) {
    stop("neural_getQStar_single requires neural_model_info_ast_jnp.", call. = FALSE)
  }
  party_label <- if (exists("GroupsPool", inherits = TRUE) && length(GroupsPool) > 0) {
    GroupsPool[1]
  } else {
    NULL
  }
  party_idx <- neural_get_party_index(model_ast, party_label)
  resp_idx <- neural_get_resp_party_index(model_ast, party_label)
  params_ast <- neural_params_from_theta(EST_COEFFICIENTS_tf_ast, model_ast)
  Qhat <- neural_predict_single_soft(pi_star_ast, party_idx, resp_idx, model_ast,
                                     params = params_ast)
  strenv$jnp$concatenate(list(Qhat, Qhat, Qhat), 0L)
}

neural_getQStar_diff_BASE <- function(pi_star_ast, pi_star_dag,
                                      EST_COEFFICIENTS_tf_ast,
                                      EST_COEFFICIENTS_tf_dag) {
  model_ast <- neural_resolve_model_info("neural_model_info_ast_jnp")
  if (is.null(model_ast)) {
    stop("neural_getQStar_diff_BASE requires neural_model_info_ast_jnp.", call. = FALSE)
  }
  model_dag <- neural_resolve_model_info("neural_model_info_dag_jnp")
  if (is.null(model_dag)) {
    model_dag <- model_ast
  }

  party_label_ast <- if (exists("GroupsPool", inherits = TRUE) && length(GroupsPool) > 0) {
    GroupsPool[1]
  } else {
    NULL
  }
  party_label_dag <- if (exists("GroupsPool", inherits = TRUE) && length(GroupsPool) > 1) {
    GroupsPool[2]
  } else {
    party_label_ast
  }

  party_idx_ast <- neural_get_party_index(model_ast, party_label_ast)
  party_idx_dag <- neural_get_party_index(model_dag, party_label_dag)
  resp_idx_ast <- neural_get_resp_party_index(model_ast, party_label_ast)
  resp_idx_dag <- neural_get_resp_party_index(model_dag, party_label_dag)
  params_ast <- neural_params_from_theta(EST_COEFFICIENTS_tf_ast, model_ast)
  params_dag <- neural_params_from_theta(EST_COEFFICIENTS_tf_dag, model_dag)

  disaggregate <- if (exists("Q_DISAGGREGATE", inherits = TRUE)) {
    isTRUE(Q_DISAGGREGATE)
  } else if (exists("adversarial", inherits = TRUE)) {
    isTRUE(adversarial)
  } else {
    FALSE
  }
  if (!isTRUE(disaggregate)) {
    resp_idx_dag <- resp_idx_ast
    params_dag <- params_ast
  }

  Qhat_ast_among_ast <- neural_predict_pair_soft(
    pi_star_ast, pi_star_dag,
    party_idx_ast, party_idx_dag,
    resp_idx_ast, model_ast,
    params = params_ast
  )

  if (!isTRUE(disaggregate)) {
    Qhat_population <- Qhat_ast_among_dag <- Qhat_ast_among_ast
  } else {
    Qhat_ast_among_dag <- neural_predict_pair_soft(
      pi_star_ast, pi_star_dag,
      party_idx_ast, party_idx_dag,
      resp_idx_dag, model_dag,
      params = params_dag
    )
    Qhat_population <- Qhat_ast_among_ast * strenv$jnp$array(strenv$AstProp) +
      Qhat_ast_among_dag * strenv$jnp$array(strenv$DagProp)
  }

  strenv$jnp$concatenate(list(Qhat_population,
                              Qhat_ast_among_ast,
                              Qhat_ast_among_dag), 0L)
}

cs2step_build_pair_mat <- function(pair_id,
                                   W,
                                   profile_order = NULL,
                                   competing_group_variable_candidate = NULL) {
  if (is.null(pair_id) || !length(pair_id)) {
    return(NULL)
  }
  if (is.null(W) || !length(W)) {
    stop("cs2step_build_pair_mat requires a non-empty W.", call. = FALSE)
  }
  W <- as.matrix(W)
  pair_id <- as.vector(pair_id)
  if (length(pair_id) != nrow(W)) {
    stop(sprintf("pair_id has %d elements but W has %d rows.",
                 length(pair_id), nrow(W)),
         call. = FALSE)
  }

  pair_indices_list <- tapply(seq_along(pair_id), pair_id, c)
  profile_order_present <- !is.null(profile_order) &&
    length(profile_order) == length(pair_id)

  row_key <- apply(W, 1, function(row) {
    paste(ifelse(is.na(row), "NA", as.character(row)), collapse = "|")
  })
  row_hash <- vapply(row_key, function(key) {
    ints <- utf8ToInt(key)
    if (!length(ints)) {
      return(0)
    }
    sum(ints * seq_along(ints)) %% 2147483647
  }, numeric(1))

  pair_mat <- do.call(rbind, lapply(pair_indices_list, function(idx){
    order_by_profile <- profile_order_present &&
      length(idx) == 2L &&
      length(unique(profile_order[idx])) == 2L &&
      !any(is.na(profile_order[idx]))
    if (!is.null(competing_group_variable_candidate)) {
      if (order_by_profile) {
        idx[order(competing_group_variable_candidate[idx],
                  profile_order[idx],
                  row_hash[idx],
                  idx)]
      } else {
        idx[order(competing_group_variable_candidate[idx],
                  row_hash[idx],
                  idx)]
      }
    } else if (order_by_profile) {
      idx[order(profile_order[idx],
                row_hash[idx],
                idx)]
    } else {
      idx[order(row_hash[idx], idx)]
    }
  }))

  list(
    pair_mat = pair_mat,
    pair_sizes = lengths(pair_indices_list),
    profile_order_present = profile_order_present
  )
}

generate_ModelOutcome_neural <- function(){
  message("Defining MCMC parameters in generate_ModelOutcome_neural...")
  mcmc_control <- list(
    backend = "numpyro",
    n_samples_warmup = 500L,
    n_samples_mcmc   = 1000L,
    batch_size = 512L,
    chain_method = "parallel",
    subsample_method = "full",
    n_thin_by = 1L,
    n_chains = 2L,
    svi_steps = 1000L,
    svi_lr = 0.01,
    svi_num_particles = 1L,
    svi_num_draws = 200L,
    vi_guide = "auto_diagonal",
    optimizer = "adam",
    svi_lr_schedule = "warmup_cosine",
    svi_lr_warmup_frac = 0.1,
    svi_lr_end_factor = 0.01
  )
  RMS_scale = 0.5
  UsedRegularization <- FALSE
  uncertainty_scope <- "all"
  mcmc_overrides <- NULL
  eval_control <- list(enabled = TRUE, max_n = NULL, seed = 123L, n_folds = NULL)
  model_dims <- 128L
  model_depth <- 2L
  cross_candidate_encoder_mode <- "none"
  warn_stage_imbalance_pct <- 0.10
  warn_min_cell_n <- 50L
  normalize_cross_candidate_encoder <- function(value) {
    if (is.null(value)) {
      return("none")
    }
    mode <- neural_normalize_cross_encoder_mode(value)
    if (is.character(mode) && length(mode) == 1L && !is.na(mode) && nzchar(mode)) {
      return(mode)
    }
    stop(
      "'neural_mcmc_control$cross_candidate_encoder' must be TRUE/FALSE or one of ",
      "'none', 'term', or 'full'.",
      call. = FALSE
    )
  }
  if (exists("neural_mcmc_control", inherits = TRUE) &&
      !is.null(neural_mcmc_control)) {
    if (!is.list(neural_mcmc_control)) {
      stop("'neural_mcmc_control' must be a list.", call. = FALSE)
    }
    if (!is.null(neural_mcmc_control$uncertainty_scope)) {
      uncertainty_scope <- tolower(as.character(neural_mcmc_control$uncertainty_scope))
    }
    if (!is.null(neural_mcmc_control$eval_enabled)) {
      eval_control$enabled <- isTRUE(neural_mcmc_control$eval_enabled)
    }
    if (!is.null(neural_mcmc_control$eval_max_n)) {
      eval_control$max_n <- as.integer(neural_mcmc_control$eval_max_n)
    }
    if (!is.null(neural_mcmc_control$eval_seed)) {
      eval_control$seed <- as.integer(neural_mcmc_control$eval_seed)
    }
    if (!is.null(neural_mcmc_control$eval_n_folds)) {
      eval_control$n_folds <- as.integer(neural_mcmc_control$eval_n_folds)
    } else if (!is.null(neural_mcmc_control$eval_folds)) {
      eval_control$n_folds <- as.integer(neural_mcmc_control$eval_folds)
    }
    if (!is.null(neural_mcmc_control$ModelDims)) {
      model_dims <- neural_mcmc_control$ModelDims
    }
    if (!is.null(neural_mcmc_control$ModelDepth)) {
      model_depth <- neural_mcmc_control$ModelDepth
    }
    if (!is.null(neural_mcmc_control$cross_candidate_encoder)) {
      cross_candidate_encoder_mode <- normalize_cross_candidate_encoder(
        neural_mcmc_control$cross_candidate_encoder
      )
    }
    if (!is.null(neural_mcmc_control$warn_stage_imbalance_pct)) {
      warn_stage_imbalance_pct <- as.numeric(neural_mcmc_control$warn_stage_imbalance_pct)
    }
    if (!is.null(neural_mcmc_control$warn_min_cell_n)) {
      warn_min_cell_n <- as.integer(neural_mcmc_control$warn_min_cell_n)
    }
    mcmc_overrides <- neural_mcmc_control
    mcmc_overrides$uncertainty_scope <- NULL
    mcmc_overrides$n_bayesian_models <- NULL
    mcmc_overrides$ModelDims <- NULL
    mcmc_overrides$ModelDepth <- NULL
    mcmc_overrides$cross_candidate_encoder <- NULL
  }
  uncertainty_scope_env <- Sys.getenv("STRATEGIZE_NEURAL_UNCERTAINTY_SCOPE")
  if (nzchar(uncertainty_scope_env)) {
    uncertainty_scope <- tolower(as.character(uncertainty_scope_env))
  }
  fast_mcmc_flag <- tolower(Sys.getenv("STRATEGIZE_NEURAL_FAST_MCMC")) %in%
    c("1", "true", "yes")
  if (isTRUE(fast_mcmc_flag)) {
    mcmc_control$n_samples_warmup <- 50L
    mcmc_control$n_samples_mcmc <- 50L
    mcmc_control$batch_size <- 128L
    mcmc_control$n_chains <- 1L
    mcmc_control$chain_method <- "sequential"
    mcmc_control$svi_steps <- 200L
    mcmc_control$svi_num_draws <- 100L
  }
  if (!is.null(mcmc_overrides) && length(mcmc_overrides) > 0) {
    mcmc_control <- modifyList(mcmc_control, mcmc_overrides)
  }
  skip_eval_flag <- tolower(Sys.getenv("STRATEGIZE_NEURAL_SKIP_EVAL")) %in%
    c("1", "true", "yes")
  if (isTRUE(skip_eval_flag)) {
    eval_control$enabled <- FALSE
  }
  eval_max_env <- suppressWarnings(as.integer(Sys.getenv("STRATEGIZE_NEURAL_EVAL_MAX")))
  if (!is.na(eval_max_env) && eval_max_env > 0L) {
    eval_control$max_n <- eval_max_env
  }
  eval_seed_env <- suppressWarnings(as.integer(Sys.getenv("STRATEGIZE_NEURAL_EVAL_SEED")))
  if (!is.na(eval_seed_env) && eval_seed_env > 0L) {
    eval_control$seed <- eval_seed_env
  }
  eval_folds_env <- suppressWarnings(as.integer(Sys.getenv("STRATEGIZE_NEURAL_EVAL_FOLDS")))
  if (!is.na(eval_folds_env) && eval_folds_env > 0L) {
    eval_control$n_folds <- eval_folds_env
  }
  if (is.null(eval_control$n_folds) &&
      exists("nFolds_glm", inherits = TRUE) &&
      is.numeric(nFolds_glm) &&
      length(nFolds_glm) == 1L &&
      is.finite(nFolds_glm)) {
    eval_control$n_folds <- as.integer(nFolds_glm)
  }
  if (!uncertainty_scope %in% c("all", "output")) {
    stop("'neural_mcmc_control$uncertainty_scope' must be 'all' or 'output'.",
         call. = FALSE)
  }
  subsample_method <- if (!is.null(mcmc_control$subsample_method)) {
    tolower(as.character(mcmc_control$subsample_method))
  } else {
    "full"
  }
  if (length(subsample_method) != 1L || is.na(subsample_method) || !nzchar(subsample_method)) {
    subsample_method <- "full"
  }
  mcmc_control$subsample_method <- subsample_method

  if (!is.numeric(model_dims) || length(model_dims) != 1L || !is.finite(model_dims)) {
    stop("'neural_mcmc_control$ModelDims' must be a single finite numeric value.",
         call. = FALSE)
  }
  if (model_dims != round(model_dims) || model_dims < 1L) {
    stop("'neural_mcmc_control$ModelDims' must be an integer >= 1.",
         call. = FALSE)
  }
  if (!is.numeric(model_depth) || length(model_depth) != 1L || !is.finite(model_depth)) {
    stop("'neural_mcmc_control$ModelDepth' must be a single finite numeric value.",
         call. = FALSE)
  }
  if (model_depth != round(model_depth) || model_depth < 1L) {
    stop("'neural_mcmc_control$ModelDepth' must be an integer >= 1.",
         call. = FALSE)
  }
  # Hyperparameters
  ModelDims  <- ai(model_dims)
  ModelDepth <- ai(model_depth)
  WideMultiplicationFactor <- 3.75
  MD_int <- ai(ModelDims)
  cand_heads <- (1:MD_int)[(MD_int %% (1:MD_int)) == 0L]
  TransformerHeads <- ai(cand_heads[which.min(abs(cand_heads - 8L))])
  head_dim <- ai(ai(MD_int / TransformerHeads))
  FFDim <- ai(ai(round(MD_int * WideMultiplicationFactor)))
  weight_sd_scale <- sqrt(2) / sqrt(as.numeric(ModelDims))
  #weight_sd_scale <- sqrt(2 * log(1 + ModelDims/2))/sqrt(ModelDims)
  
  # Depth-aware scaling for priors and ReZero-style residual gates.
  depth_prior_scale <- sqrt(2) / sqrt(as.numeric(ModelDepth))
  gate_sd_scale <- 0.1 * depth_prior_scale
  embed_sd_scale <- 4 * weight_sd_scale
  factor_embed_sd_scale <- embed_sd_scale
  context_embed_sd_scale <- embed_sd_scale
  tau_b_scale <- 0.5
  
  # shrink M_cross more (initialize interactionto smaller than main temr)
  # cross_out does NOT need to shrink with ModelDims (doesn't scale with that)
  cross_weight_sd_scale <- weight_sd_scale / sqrt(as.numeric(ModelDims))
  cross_out_sd_scale    <- 0.5  # or 0.25 if you want it more conservative

  # Pairwise mode for forced-choice
  pairwise_mode <- isTRUE(diff) && !is.null(pair_id_) && length(pair_id_) > 0
  if (!isTRUE(pairwise_mode)) {
    cross_candidate_encoder_mode <- "none"
  }
  use_cross_term <- identical(cross_candidate_encoder_mode, "term")
  use_cross_encoder <- identical(cross_candidate_encoder_mode, "full")
  use_matchup_token <- isTRUE(pairwise_mode) &&
    !identical(cross_candidate_encoder_mode, "none")

  # Main-info structure for downstream compatibility
  for(nrp in 1:2){
    main_info <- do.call(rbind, sapply(1:length(factor_levels), function(d_){
      list(data.frame(
        "d" = d_,
        "l" = 1:max(1, factor_levels[d_] - ifelse(nrp == 1, yes = 1, no = holdout_indicator))
      ))
    }))
    main_info <- cbind(main_info, "d_index" = 1:nrow(main_info))
    if(nrp == 1){ a_structure <- main_info }
  }
  if(holdout_indicator == 0){
    a_structure_leftoutLdminus1 <- main_info[which(c(base::diff(main_info$d),1)==0),]
    a_structure_leftoutLdminus1$d_index <- 1:nrow(a_structure_leftoutLdminus1)
  }

  interaction_info <- data.frame()
  interaction_info_PreRegularization <- interaction_info
  regularization_adjust_hash <- main_info$d
  names(regularization_adjust_hash) <- main_info$d

  factor_levels_int <- as.integer(factor_levels)
  factor_levels_aug <- factor_levels_int + 1L
  factor_index_list <- vector("list", length(factor_levels))
  offset <- 0L
  for (d_ in seq_along(factor_levels)) {
    n_levels_use <- ai(factor_levels[d_] - holdout_indicator)
    idx <- if (n_levels_use > 0L) {
      as.integer(offset + seq_len(n_levels_use) - 1L)
    } else {
      integer(0)
    }
    factor_index_list[[d_]] <- strenv$jnp$array(idx)$astype(strenv$jnp$int32)
    offset <- offset + n_levels_use
  }
  n_rel_levels <- ai(3L)
  n_candidate_tokens <- ai(length(factor_levels) + 2L)

  # Party token mapping (candidates)
  party_levels_override <- NULL
  if (exists("party_levels_fixed", inherits = TRUE)) {
    party_levels_override <- get("party_levels_fixed", inherits = TRUE)
  }
  party_levels <- if (!is.null(party_levels_override)) {
    as.character(party_levels_override)
  } else if (!is.null(competing_group_variable_candidate_)) {
    sort(unique(as.character(competing_group_variable_candidate_)))
  } else {
    "NA"
  }
  n_party_levels <- max(1L, length(party_levels))
  n_matchup_levels <- if (isTRUE(use_matchup_token)) {
    as.integer(n_party_levels * (n_party_levels + 1L) / 2L)
  } else {
    0L
  }
  party_index <- if (!is.null(competing_group_variable_candidate_)) {
    match(as.character(competing_group_variable_candidate_), party_levels) - 1L
  } else {
    rep(0L, length(Y_))
  }
  party_index[is.na(party_index)] <- 0L

  # Respondent party mapping
  resp_party_levels_override <- NULL
  if (exists("resp_party_levels_fixed", inherits = TRUE)) {
    resp_party_levels_override <- get("resp_party_levels_fixed", inherits = TRUE)
  }
  resp_party_levels <- if (!is.null(resp_party_levels_override)) {
    as.character(resp_party_levels_override)
  } else if (!is.null(competing_group_variable_respondent_)) {
    sort(unique(as.character(competing_group_variable_respondent_)))
  } else {
    "NA"
  }
  n_resp_party_levels <- max(1L, length(resp_party_levels))
  resp_party_index <- if (!is.null(competing_group_variable_respondent_)) {
    match(as.character(competing_group_variable_respondent_), resp_party_levels) - 1L
  } else {
    rep(0L, length(Y_))
  }
  resp_party_index[is.na(resp_party_index)] <- 0L

  cand_party_to_resp_idx <- vapply(party_levels, function(party_label) {
    idx <- match(as.character(party_label), resp_party_levels)
    if (is.na(idx)) -1L else as.integer(idx - 1L)
  }, integer(1))
  cand_party_to_resp_idx_jnp <- strenv$jnp$array(as.integer(cand_party_to_resp_idx))
  cand_party_to_resp_idx_jnp <- strenv$jnp$atleast_1d(cand_party_to_resp_idx_jnp)$astype(strenv$jnp$int32)

  # Respondent covariates (optional)
  X_use <- NULL
  X_ <- NULL
  if (exists("X", inherits = TRUE) && !is.null(X)) {
    X_ <- as.matrix(X[indi_, , drop = FALSE])
  }

  # Helper to sanitize integer indices (adds explicit missing/OOV level per factor)
  to_index_matrix <- function(x_mat){
    x_mat <- as.matrix(x_mat)
    x_int <- matrix(as.integer(x_mat), nrow = nrow(x_mat), ncol = ncol(x_mat))
    n_cols <- ncol(x_int)
    n_levels <- factor_levels_int
    if (length(n_levels) != n_cols) {
      n_levels <- rep_len(n_levels, n_cols)
    }
    for (d_ in seq_len(n_cols)) {
      max_level <- n_levels[[d_]]
      missing_level <- max_level + 1L
      col_vals <- x_int[, d_]
      col_vals[is.na(col_vals) | col_vals < 1L | col_vals > max_level] <- missing_level
      x_int[, d_] <- col_vals - 1L
    }
    x_int
  }

  center_factor_embeddings <- function(E_factor_raw, n_real_levels) {
    n_real_levels <- ai(n_real_levels)
    real_idx <- strenv$jnp$arange(n_real_levels)
    real_rows <- strenv$jnp$take(E_factor_raw, real_idx, axis = 0L)
    real_mean <- strenv$jnp$mean(real_rows, axis = 0L, keepdims = TRUE)
    real_centered <- real_rows - real_mean
    missing_row <- strenv$jnp$take(E_factor_raw, ai(n_real_levels), axis = 0L)
    missing_row <- strenv$jnp$reshape(missing_row, list(1L, ModelDims))
    strenv$jnp$concatenate(list(real_centered, missing_row), axis = 0L)
  }

  # Build pairwise or single-candidate data
  pair_mat <- NULL
  if (pairwise_mode) {
    pair_info <- cs2step_build_pair_mat(
      pair_id = pair_id_,
      W = W_,
      profile_order = profile_order_,
      competing_group_variable_candidate = competing_group_variable_candidate_
    )
    if (is.null(pair_info) || is.null(pair_info$pair_sizes) ||
        !all(pair_info$pair_sizes == 2L)) {
      warning("pair_id does not define exactly 2 rows per pair; falling back to single-candidate model.")
      pairwise_mode <- FALSE
    } else {
      pair_mat <- pair_info$pair_mat
    }
  }

  if (pairwise_mode) {
    X_left <- W_[pair_mat[,1], , drop = FALSE]
    X_right <- W_[pair_mat[,2], , drop = FALSE]
    Y_use <- Y_[pair_mat[,1]]
    party_left <- party_index[pair_mat[,1]]
    party_right <- party_index[pair_mat[,2]]
    resp_party_use <- resp_party_index[pair_mat[,1]]
    if (!is.null(X_)) {
      X_use <- X_[pair_mat[,1], , drop = FALSE]
    }
  } else {
    X_single <- W_
    Y_use <- Y_
    party_single <- party_index
    resp_party_use <- resp_party_index
    X_use <- X_
  }

  stage_diagnostics <- NULL
  if (pairwise_mode) {
    stage_is_primary <- party_left == party_right
    n_total <- length(stage_is_primary)
    n_primary <- if (n_total > 0L) sum(stage_is_primary, na.rm = TRUE) else 0L
    n_general <- if (n_total > 0L) sum(!stage_is_primary, na.rm = TRUE) else 0L
    pct_primary <- if (n_total > 0L) n_primary / n_total else NA_real_
    stage_label <- ifelse(stage_is_primary, "primary", "general")
    resp_party_label <- if (!is.null(resp_party_levels)) {
      idx <- as.integer(resp_party_use) + 1L
      idx[idx < 1L | idx > length(resp_party_levels)] <- NA_integer_
      resp_party_levels[idx]
    } else {
      as.character(resp_party_use)
    }
    stage_table <- table(resp_party_label, stage_label)
    cell_counts <- as.integer(stage_table)
    min_cell_n <- if (length(cell_counts) > 0L) min(cell_counts) else NA_integer_
    single_stage_only <- isTRUE(pct_primary == 0 || pct_primary == 1)
    warn_stage_imbalance <- is.finite(pct_primary) &&
      (pct_primary < warn_stage_imbalance_pct || pct_primary > (1 - warn_stage_imbalance_pct))
    warn_sparse_cells <- !is.na(min_cell_n) && min_cell_n < warn_min_cell_n

    if (isTRUE(warn_stage_imbalance)) {
      warning(
        sprintf("Stage imbalance detected in neural training data (pct_primary=%.3f).", pct_primary),
        call. = FALSE
      )
    }
    if (isTRUE(warn_sparse_cells)) {
      warning(
        sprintf("Sparse stage/resp-party cells detected (min cell n=%d).", min_cell_n),
        call. = FALSE
      )
    }
    if (isTRUE(single_stage_only)) {
      warning(
        "Stage indicator has no variation; E_stage is not identified for the unused stage.",
        call. = FALSE
      )
    }

    stage_diagnostics <- list(
      n_primary = as.integer(n_primary),
      n_general = as.integer(n_general),
      pct_primary = pct_primary,
      resp_party_stage_table = stage_table,
      single_stage_only = single_stage_only,
      sparse_cells = warn_sparse_cells,
      min_cell_n = min_cell_n,
      warn_stage_imbalance_pct = warn_stage_imbalance_pct,
      warn_min_cell_n = warn_min_cell_n
    )
  }

  n_resp_covariates <- if (!is.null(X_use)) ai(ncol(X_use)) else ai(0L)
  resp_cov_sd <- if (n_resp_covariates > 0L) {
    0.5 / sqrt(as.numeric(n_resp_covariates))
  } else {
    NULL
  }
  resp_cov_mean <- if (!is.null(X_use) && n_resp_covariates > 0L) {
    as.numeric(colMeans(X_use))
  } else {
    NULL
  }

  # Placeholder to keep model chunks consistent (no surrogate regression)
  main_dat <- matrix(0, nrow = 0L, ncol = 0L)

  # Likelihood selection (allow overrides for cross-fit evaluation runs).
  likelihood_override <- NULL
  nOutcomes_override <- NULL
  if (exists("neural_likelihood_override", inherits = TRUE)) {
    likelihood_override <- get("neural_likelihood_override", inherits = TRUE)
  }
  if (exists("neural_nOutcomes_override", inherits = TRUE)) {
    nOutcomes_override <- get("neural_nOutcomes_override", inherits = TRUE)
  }

  is_binary <- all(unique(na.omit(as.numeric(Y_use))) %in% c(0, 1)) &&
    length(unique(na.omit(Y_use))) <= 2
  is_intvec <- all(!is.na(Y_use)) && all(abs(Y_use - round(Y_use)) < 1e-8)
  K_classes <- if (is_intvec) length(unique(ai(Y_use))) else NA_integer_

  if (!is.null(likelihood_override)) {
    likelihood <- tolower(as.character(likelihood_override))
    if (!likelihood %in% c("bernoulli", "categorical", "normal")) {
      stop("neural_likelihood_override must be one of 'bernoulli', 'categorical', or 'normal'.",
           call. = FALSE)
    }
    if (!is.null(nOutcomes_override)) {
      nOutcomes <- ai(nOutcomes_override)
    } else if (likelihood == "categorical") {
      stop("neural_nOutcomes_override must be provided when forcing categorical likelihood.",
           call. = FALSE)
    } else {
      nOutcomes <- ai(1L)
    }
  } else if (is_binary) {
    likelihood <- "bernoulli"; nOutcomes <- ai(1L)
  } else if (!is.na(K_classes) && K_classes >= 2L && K_classes <= max(50L, ncol(W_) + 1L)) {
    likelihood <- "categorical"; nOutcomes <- ai(K_classes)
  } else {
    likelihood <- "normal"; nOutcomes <- ai(1L)
  }
  sigma_prior_scale <- 1.0
  if (likelihood == "normal") {
    y_numeric <- as.numeric(Y_use)
    y_mad <- suppressWarnings(stats::mad(y_numeric, na.rm = TRUE))
    y_sd <- suppressWarnings(stats::sd(y_numeric, na.rm = TRUE))
    sigma_prior_scale <- if (is.finite(y_mad) && y_mad > 0) {
      y_mad
    } else if (is.finite(y_sd) && y_sd > 0) {
      y_sd
    } else {
      1.0
    }
  }

  pdtype_ <- ddtype_ <- strenv$jnp$float32
  manual_noncentered_loc_scale <- FALSE
  p2d_eps <- 1e-6
  p2d_hash31 <- function(x, mod = 2147483647) {
    bytes <- utf8ToInt(as.character(x))
    h <- 0
    for (b in bytes) {
      h <- (h * 31 + b) %% mod
    }
    if (!is.finite(h) || is.na(h) || h < 0) {
      h <- 0
    }
    as.integer(h)
  }
  p2d_draw_fixed_normal <- function(name, scale, shape_tuple, dtype = ddtype_) {
    seed <- p2d_hash31(paste0("p2d:", name))
    key <- strenv$jax$random$PRNGKey(ai(seed))
    draw <- strenv$jax$random$normal(key, shape = shape_tuple, dtype = dtype)
    as.numeric(scale) * draw
  }
  p2d_init_normal <- function(name, scale, shape_tuple, dtype = ddtype_) {
    p2d_draw_fixed_normal(paste0(name, ":init"), scale, shape_tuple, dtype = dtype)
  }
  p2d_init_halfnormal <- function(name, scale, shape_tuple, dtype = ddtype_) {
    strenv$jnp$abs(p2d_draw_fixed_normal(paste0(name, ":init"), scale, shape_tuple, dtype = dtype)) + p2d_eps
  }
  p2d_init_lognormal <- function(name, sd, shape_tuple, dtype = ddtype_) {
    strenv$jnp$exp(p2d_draw_fixed_normal(paste0(name, ":init_log"), sd, shape_tuple, dtype = dtype)) + p2d_eps
  }
  p2d_constraint_positive <- NULL
  if (!is.null(strenv$numpyro$distributions) &&
      reticulate::py_has_attr(strenv$numpyro$distributions, "constraints")) {
    constraints_mod <- strenv$numpyro$distributions$constraints
    if (!is.null(constraints_mod) && reticulate::py_has_attr(constraints_mod, "positive")) {
      p2d_constraint_positive <- constraints_mod$positive
    }
  }
  p2d <- function(name,
                  sample_fxn,
                  init_fxn,
                  constraint = NULL,
                  uncertainty_scope_arg = uncertainty_scope,
                  is_output_layer = FALSE) {
    scope <- tolower(as.character(uncertainty_scope_arg))
    if (identical(scope, "output") && !isTRUE(is_output_layer)) {
      init_val <- init_fxn()
      if (!is.null(constraint)) {
        return(strenv$numpyro$param(name, init_val, constraint = constraint))
      }
      return(strenv$numpyro$param(name, init_val))
    }
    sample_fxn()
  }
  sample_loc_scale <- function(name, scale, shape_tuple) {
    if (isTRUE(manual_noncentered_loc_scale)) {
      z <- strenv$numpyro$sample(
        paste0(name, "_z"),
        strenv$numpyro$distributions$Normal(0., 1.)$expand(shape_tuple)
      )
      strenv$numpyro$deterministic(name, scale * z)
    } else {
      strenv$numpyro$sample(
        name,
        strenv$numpyro$distributions$Normal(0., scale)$expand(shape_tuple)
      )
    }
  }

  sample_shared_transformer_params <- function(D_local, pairwise = FALSE) {
    output_only_mode <- identical(tolower(as.character(uncertainty_scope)), "output")

    tau_factor <- if (isTRUE(output_only_mode)) {
      as.numeric(factor_embed_sd_scale)
    } else {
      strenv$numpyro$sample(
        "tau_factor",
        strenv$numpyro$distributions$HalfNormal(as.numeric(factor_embed_sd_scale))
      )
    }
    tau_context <- if (isTRUE(output_only_mode)) {
      as.numeric(context_embed_sd_scale)
    } else {
      strenv$numpyro$sample(
        "tau_context",
        strenv$numpyro$distributions$HalfNormal(as.numeric(context_embed_sd_scale))
      )
    }

    E_factor_list <- vector("list", D_local)
    for (d_ in 1L:D_local) {
      raw_name <- paste0("E_factor_", d_, "_raw")
      raw_shape <- reticulate::tuple(ai(factor_levels_aug[d_]), ModelDims)
      E_factor_raw <- p2d(
        name = raw_name,
        sample_fxn = function() {
          strenv$numpyro$sample(
            raw_name,
            strenv$numpyro$distributions$Normal(0., tau_factor),
            sample_shape = raw_shape
          )
        },
        init_fxn = function() {
          p2d_init_normal(raw_name, tau_factor, raw_shape)
        }
      )
      E_factor_centered <- center_factor_embeddings(E_factor_raw, factor_levels_int[d_])
      E_factor_list[[d_]] <- strenv$numpyro$deterministic(
        paste0("E_factor_", d_),
        E_factor_centered
      )
    }

    E_feature_shape <- reticulate::tuple(ai(D_local), ModelDims)
    E_feature_id <- p2d(
      name = "E_feature_id",
      sample_fxn = function() {
        strenv$numpyro$sample(
          "E_feature_id",
          strenv$numpyro$distributions$Normal(0., tau_context),
          sample_shape = E_feature_shape
        )
      },
      init_fxn = function() {
        p2d_init_normal("E_feature_id", tau_context, E_feature_shape)
      }
    )

    E_party_shape <- reticulate::tuple(ai(n_party_levels), ModelDims)
    E_party <- p2d(
      name = "E_party",
      sample_fxn = function() {
        strenv$numpyro$sample(
          "E_party",
          strenv$numpyro$distributions$Normal(0., tau_context),
          sample_shape = E_party_shape
        )
      },
      init_fxn = function() {
        p2d_init_normal("E_party", tau_context, E_party_shape)
      }
    )

    E_rel_shape <- reticulate::tuple(ai(n_rel_levels), ModelDims)
    E_rel <- p2d(
      name = "E_rel",
      sample_fxn = function() {
        strenv$numpyro$sample(
          "E_rel",
          strenv$numpyro$distributions$Normal(0., tau_context),
          sample_shape = E_rel_shape
        )
      },
      init_fxn = function() {
        p2d_init_normal("E_rel", tau_context, E_rel_shape)
      }
    )

    E_resp_party_shape <- reticulate::tuple(ai(n_resp_party_levels), ModelDims)
    E_resp_party <- p2d(
      name = "E_resp_party",
      sample_fxn = function() {
        strenv$numpyro$sample(
          "E_resp_party",
          strenv$numpyro$distributions$Normal(0., tau_context),
          sample_shape = E_resp_party_shape
        )
      },
      init_fxn = function() {
        p2d_init_normal("E_resp_party", tau_context, E_resp_party_shape)
      }
    )

    E_stage <- NULL
    E_matchup <- NULL
    if (isTRUE(pairwise)) {
      E_stage_shape <- reticulate::tuple(ai(n_resp_party_levels), ai(2L), ModelDims)
      E_stage <- p2d(
        name = "E_stage",
        sample_fxn = function() {
          strenv$numpyro$sample(
            "E_stage",
            strenv$numpyro$distributions$Normal(0., tau_context),
            sample_shape = E_stage_shape
          )
        },
        init_fxn = function() {
          p2d_init_normal("E_stage", tau_context, E_stage_shape)
        }
      )
      if (isTRUE(use_matchup_token)) {
        E_matchup_shape <- reticulate::tuple(ai(n_matchup_levels), ModelDims)
        E_matchup <- p2d(
          name = "E_matchup",
          sample_fxn = function() {
            strenv$numpyro$sample(
              "E_matchup",
              strenv$numpyro$distributions$Normal(0., tau_context),
              sample_shape = E_matchup_shape
            )
          },
          init_fxn = function() {
            p2d_init_normal("E_matchup", tau_context, E_matchup_shape)
          }
        )
      }
    }

    E_choice_shape <- reticulate::tuple(ModelDims)
    E_choice <- p2d(
      name = "E_choice",
      sample_fxn = function() {
        strenv$numpyro$sample(
          "E_choice",
          strenv$numpyro$distributions$Normal(0., tau_context),
          sample_shape = E_choice_shape
        )
      },
      init_fxn = function() {
        p2d_init_normal("E_choice", tau_context, E_choice_shape)
      }
    )

    E_sep <- NULL
    E_segment <- NULL
    if (isTRUE(pairwise) && isTRUE(use_cross_encoder)) {
      E_sep_shape <- reticulate::tuple(ModelDims)
      E_sep <- p2d(
        name = "E_sep",
        sample_fxn = function() {
          strenv$numpyro$sample(
            "E_sep",
            strenv$numpyro$distributions$Normal(0., tau_context),
            sample_shape = E_sep_shape
          )
        },
        init_fxn = function() {
          p2d_init_normal("E_sep", tau_context, E_sep_shape)
        }
      )

      E_segment_shape <- reticulate::tuple(ai(2L), ModelDims)
      E_segment <- p2d(
        name = "E_segment",
        sample_fxn = function() {
          strenv$numpyro$sample(
            "E_segment",
            strenv$numpyro$distributions$Normal(0., tau_context),
            sample_shape = E_segment_shape
          )
        },
        init_fxn = function() {
          p2d_init_normal("E_segment", tau_context, E_segment_shape)
        }
      )
    }

    W_resp_x <- NULL
    if (n_resp_covariates > 0L) {
      W_resp_x_shape <- reticulate::tuple(ai(n_resp_covariates), ModelDims)
      W_resp_x <- p2d(
        name = "W_resp_x",
        sample_fxn = function() {
          strenv$numpyro$sample(
            "W_resp_x",
            strenv$numpyro$distributions$Normal(0., resp_cov_sd),
            sample_shape = W_resp_x_shape
          )
        },
        init_fxn = function() {
          p2d_init_normal("W_resp_x", resp_cov_sd, W_resp_x_shape)
        }
      )
    }

    layer_params <- list()
    for (l_ in 1L:ModelDepth) {
      tau_w_prior <- as.numeric(weight_sd_scale * depth_prior_scale)
      tau_w_l <- if (isTRUE(output_only_mode)) {
        tau_w_prior
      } else {
        strenv$numpyro$sample(
          paste0("tau_w_", l_),
          strenv$numpyro$distributions$HalfNormal(tau_w_prior)
        )
      }

      alpha_attn_name <- paste0("alpha_attn_l", l_)
      alpha_attn_l <- p2d(
        name = alpha_attn_name,
        sample_fxn = function() {
          strenv$numpyro$sample(
            alpha_attn_name,
            strenv$numpyro$distributions$HalfNormal(gate_sd_scale)
          )
        },
        init_fxn = function() {
          p2d_init_halfnormal(alpha_attn_name, gate_sd_scale, reticulate::tuple())
        },
        constraint = p2d_constraint_positive
      )

      alpha_ff_name <- paste0("alpha_ff_l", l_)
      alpha_ff_l <- p2d(
        name = alpha_ff_name,
        sample_fxn = function() {
          strenv$numpyro$sample(
            alpha_ff_name,
            strenv$numpyro$distributions$HalfNormal(gate_sd_scale)
          )
        },
        init_fxn = function() {
          p2d_init_halfnormal(alpha_ff_name, gate_sd_scale, reticulate::tuple())
        },
        constraint = p2d_constraint_positive
      )

      RMS_attn_name <- paste0("RMS_attn_l", l_)
      RMS_attn_shape <- reticulate::tuple(ModelDims)
      RMS_attn_l <- p2d(
        name = RMS_attn_name,
        sample_fxn = function() {
          strenv$numpyro$sample(
            RMS_attn_name,
            strenv$numpyro$distributions$LogNormal(0., RMS_scale),
            sample_shape = RMS_attn_shape
          )
        },
        init_fxn = function() {
          p2d_init_lognormal(RMS_attn_name, RMS_scale, RMS_attn_shape)
        },
        constraint = p2d_constraint_positive
      )

      RMS_ff_name <- paste0("RMS_ff_l", l_)
      RMS_ff_shape <- reticulate::tuple(ModelDims)
      RMS_ff_l <- p2d(
        name = RMS_ff_name,
        sample_fxn = function() {
          strenv$numpyro$sample(
            RMS_ff_name,
            strenv$numpyro$distributions$LogNormal(0., RMS_scale),
            sample_shape = RMS_ff_shape
          )
        },
        init_fxn = function() {
          p2d_init_lognormal(RMS_ff_name, RMS_scale, RMS_ff_shape)
        },
        constraint = p2d_constraint_positive
      )

      W_q_name <- paste0("W_q_l", l_)
      W_q_shape <- reticulate::tuple(ModelDims, ModelDims)
      W_q_l <- p2d(
        name = W_q_name,
        sample_fxn = function() {
          sample_loc_scale(W_q_name, tau_w_l, W_q_shape)
        },
        init_fxn = function() {
          p2d_init_normal(W_q_name, tau_w_l, W_q_shape)
        }
      )

      W_k_name <- paste0("W_k_l", l_)
      W_k_shape <- reticulate::tuple(ModelDims, ModelDims)
      W_k_l <- p2d(
        name = W_k_name,
        sample_fxn = function() {
          sample_loc_scale(W_k_name, tau_w_l, W_k_shape)
        },
        init_fxn = function() {
          p2d_init_normal(W_k_name, tau_w_l, W_k_shape)
        }
      )

      W_v_name <- paste0("W_v_l", l_)
      W_v_shape <- reticulate::tuple(ModelDims, ModelDims)
      W_v_l <- p2d(
        name = W_v_name,
        sample_fxn = function() {
          sample_loc_scale(W_v_name, tau_w_l, W_v_shape)
        },
        init_fxn = function() {
          p2d_init_normal(W_v_name, tau_w_l, W_v_shape)
        }
      )

      W_o_name <- paste0("W_o_l", l_)
      W_o_shape <- reticulate::tuple(ModelDims, ModelDims)
      W_o_l <- p2d(
        name = W_o_name,
        sample_fxn = function() {
          sample_loc_scale(W_o_name, tau_w_l, W_o_shape)
        },
        init_fxn = function() {
          p2d_init_normal(W_o_name, tau_w_l, W_o_shape)
        }
      )

      W_ff1_name <- paste0("W_ff1_l", l_)
      W_ff1_shape <- reticulate::tuple(ModelDims, FFDim)
      W_ff1_l <- p2d(
        name = W_ff1_name,
        sample_fxn = function() {
          sample_loc_scale(W_ff1_name, tau_w_l, W_ff1_shape)
        },
        init_fxn = function() {
          p2d_init_normal(W_ff1_name, tau_w_l, W_ff1_shape)
        }
      )

      W_ff2_name <- paste0("W_ff2_l", l_)
      W_ff2_shape <- reticulate::tuple(FFDim, ModelDims)
      W_ff2_l <- p2d(
        name = W_ff2_name,
        sample_fxn = function() {
          sample_loc_scale(W_ff2_name, tau_w_l, W_ff2_shape)
        },
        init_fxn = function() {
          p2d_init_normal(W_ff2_name, tau_w_l, W_ff2_shape)
        }
      )

      layer_params[[paste0("W_q_l", l_)]] <- W_q_l
      layer_params[[paste0("W_k_l", l_)]] <- W_k_l
      layer_params[[paste0("W_v_l", l_)]] <- W_v_l
      layer_params[[paste0("W_o_l", l_)]] <- W_o_l
      layer_params[[paste0("W_ff1_l", l_)]] <- W_ff1_l
      layer_params[[paste0("W_ff2_l", l_)]] <- W_ff2_l
      layer_params[[paste0("RMS_attn_l", l_)]] <- RMS_attn_l
      layer_params[[paste0("RMS_ff_l", l_)]] <- RMS_ff_l
      layer_params[[paste0("alpha_attn_l", l_)]] <- alpha_attn_l
      layer_params[[paste0("alpha_ff_l", l_)]] <- alpha_ff_l
    }

    RMS_final_shape <- reticulate::tuple(ModelDims)
    RMS_final <- p2d(
      name = "RMS_final",
      sample_fxn = function() {
        strenv$numpyro$sample(
          "RMS_final",
          strenv$numpyro$distributions$LogNormal(0., RMS_scale),
          sample_shape = RMS_final_shape
        )
      },
      init_fxn = function() {
        p2d_init_lognormal("RMS_final", RMS_scale, RMS_final_shape)
      },
      constraint = p2d_constraint_positive
    )
    tau_w_out <- strenv$numpyro$sample(
      "tau_w_out",
      strenv$numpyro$distributions$HalfNormal(as.numeric(weight_sd_scale))
    )
    W_out <- sample_loc_scale("W_out", tau_w_out,
                              reticulate::tuple(ModelDims, nOutcomes))
    tau_b <- strenv$numpyro$sample(
      "tau_b",
      strenv$numpyro$distributions$HalfNormal(as.numeric(tau_b_scale))
    )
    b_out <- sample_loc_scale("b_out", tau_b, reticulate::tuple(nOutcomes))

    M_cross <- NULL
    W_cross_out <- NULL
    if (isTRUE(pairwise) && isTRUE(use_cross_term)) {
      # Antisymmetric bilinear term enables opponent-dependent matchups.
      tau_cross <- if (isTRUE(output_only_mode)) {
        as.numeric(cross_weight_sd_scale)
      } else {
        strenv$numpyro$sample(
          "tau_cross",
          strenv$numpyro$distributions$HalfNormal(as.numeric(cross_weight_sd_scale))
        )
      }
      M_cross_shape <- reticulate::tuple(ModelDims, ModelDims)
      M_cross_raw <- p2d(
        name = "M_cross_raw",
        sample_fxn = function() {
          sample_loc_scale("M_cross_raw", tau_cross, M_cross_shape)
        },
        init_fxn = function() {
          p2d_init_normal("M_cross_raw", tau_cross, M_cross_shape)
        }
      )
      M_cross <- 0.5 * (M_cross_raw - strenv$jnp$transpose(M_cross_raw))
      M_cross <- strenv$numpyro$deterministic("M_cross", M_cross)
      W_cross_out <- strenv$numpyro$sample(
        "W_cross_out",
        strenv$numpyro$distributions$Normal(0., 0.25),
        sample_shape = reticulate::tuple(nOutcomes)
      )
    }

    sigma <- NULL
    if (likelihood == "normal") {
      sigma <- strenv$numpyro$sample(
        "sigma",
        strenv$numpyro$distributions$HalfNormal(as.numeric(sigma_prior_scale))
      )
    }

    params_view <- if (isTRUE(pairwise)) {
      list(
        E_feature_id = E_feature_id,
        E_party = E_party,
        E_rel = E_rel,
        E_resp_party = E_resp_party,
        E_stage = E_stage,
        E_choice = E_choice,
        E_sep = E_sep,
        E_segment = E_segment
      )
    } else {
      list(
        E_feature_id = E_feature_id,
        E_party = E_party,
        E_rel = E_rel,
        E_resp_party = E_resp_party,
        E_choice = E_choice
      )
    }
    if (isTRUE(pairwise) && isTRUE(use_matchup_token)) {
      params_view$E_matchup <- E_matchup
    }
    if (n_resp_covariates > 0L) {
      params_view$W_resp_x <- W_resp_x
    }
    for (d_ in 1L:D_local) {
      params_view[[paste0("E_factor_", d_)]] <- E_factor_list[[d_]]
    }
    params_view <- c(params_view, layer_params)
    params_view$RMS_final <- RMS_final
    params_view$W_out <- W_out
    params_view$b_out <- b_out
    if (isTRUE(pairwise) && isTRUE(use_cross_term)) {
      params_view$M_cross <- M_cross
      params_view$W_cross_out <- W_cross_out
    }

    list(
      params_view = params_view,
      E_choice = E_choice,
      W_out = W_out,
      b_out = b_out,
      sigma = sigma,
      M_cross = M_cross,
      W_cross_out = W_cross_out
    )
  }

  BayesianPairTransformerModel <- function(X_left, X_right, party_left, party_right,
                                           resp_party, resp_cov, Y_obs) {
    N_local <- ai(X_left$shape[[1]])
    D_local <- ai(X_left$shape[[2]])

    shared_params <- sample_shared_transformer_params(D_local = D_local, pairwise = TRUE)
    params_view <- shared_params$params_view
    E_choice <- shared_params$E_choice
    W_out <- shared_params$W_out
    b_out <- shared_params$b_out
    sigma <- shared_params$sigma
    M_cross <- shared_params$M_cross
    W_cross_out <- shared_params$W_cross_out

    transformer_model_info <- list(
      model_depth = ModelDepth,
      model_dims = ModelDims,
      n_heads = TransformerHeads,
      head_dim = head_dim
    )
    model_info_local <- list(
      model_dims = ModelDims,
      cand_party_to_resp_idx = cand_party_to_resp_idx_jnp,
      n_party_levels = ai(n_party_levels)
    )

    embed_candidate <- function(X_idx, party_idx, resp_p) {
      neural_build_candidate_tokens_hard(X_idx, party_idx,
                                         model_info = model_info_local,
                                         resp_party_idx = resp_p,
                                         params = params_view)
    }

    add_segment_embedding <- function(tokens, segment_idx) {
      neural_add_segment_embedding(tokens, segment_idx,
                                   model_info = model_info_local,
                                   params = params_view)
    }

    run_transformer <- function(tokens) {
      neural_run_transformer(tokens,
                             model_info = transformer_model_info,
                             params = params_view)
    }

    compute_matchup_idx <- function(pl, pr) {
      neural_matchup_index(pl, pr, model_info_local)
    }

    build_context_tokens <- function(stage_idx, resp_p, resp_c, matchup_idx = NULL) {
      neural_build_context_tokens_batch(model_info = model_info_local,
                                        resp_party_idx = resp_p,
                                        stage_idx = stage_idx,
                                        matchup_idx = matchup_idx,
                                        resp_cov = resp_c,
                                        params = params_view)
    }

    build_sep_token <- function(N_batch) {
      neural_build_sep_token(model_info_local,
                             n_batch = N_batch,
                             params = params_view)
    }

    encode_pair_cross <- function(Xl, Xr, pl, pr, resp_p, resp_c, stage_idx, matchup_idx = NULL) {
      N_batch <- ai(Xl$shape[[1]])
      choice_tok <- neural_build_choice_token(model_info_local, params_view)
      choice_tok <- choice_tok * strenv$jnp$ones(list(N_batch, 1L, 1L))
      ctx_tokens <- build_context_tokens(stage_idx, resp_p, resp_c, matchup_idx)
      left_tokens <- add_segment_embedding(embed_candidate(Xl, pl, resp_p), 0L)
      right_tokens <- add_segment_embedding(embed_candidate(Xr, pr, resp_p), 1L)
      sep_tok <- build_sep_token(N_batch)
      tokens <- strenv$jnp$concatenate(list(choice_tok, ctx_tokens, sep_tok,
                                            left_tokens, sep_tok, right_tokens),
                                       axis = 1L)
      tokens <- run_transformer(tokens)
      cls_out <- strenv$jnp$take(tokens, strenv$jnp$arange(1L), axis = 1L)
      cls_out <- strenv$jnp$squeeze(cls_out, axis = 1L)
      neural_linear_head(cls_out, W_out, b_out)
    }

    encode_candidate <- function(Xa, pa, resp_p, resp_c, stage_idx, matchup_idx = NULL) {
      N_batch <- ai(Xa$shape[[1]])
      choice_tok <- neural_build_choice_token(model_info_local, params_view)
      choice_tok <- choice_tok * strenv$jnp$ones(list(N_batch, 1L, 1L))
      ctx_tokens <- build_context_tokens(stage_idx, resp_p, resp_c, matchup_idx)
      cand_tokens <- embed_candidate(Xa, pa, resp_p)
      tokens <- strenv$jnp$concatenate(list(choice_tok, ctx_tokens, cand_tokens), axis = 1L)
      tokens <- run_transformer(tokens)
      choice_out <- strenv$jnp$take(tokens, strenv$jnp$arange(1L), axis = 1L)
      strenv$jnp$squeeze(choice_out, axis = 1L)
    }

    encode_candidate_pair <- function(Xl, Xr, pl, pr, resp_p, resp_c, stage_idx, matchup_idx = NULL) {
      N_batch <- ai(Xl$shape[[1]])
      X_all <- strenv$jnp$concatenate(list(Xl, Xr), axis = 0L)
      p_all <- strenv$jnp$concatenate(list(pl, pr), axis = 0L)
      resp_p_all <- strenv$jnp$concatenate(list(resp_p, resp_p), axis = 0L)
      resp_c_all <- if (is.null(resp_c)) NULL else strenv$jnp$concatenate(list(resp_c, resp_c), axis = 0L)
      stage_all <- if (is.null(stage_idx)) NULL else strenv$jnp$concatenate(list(stage_idx, stage_idx), axis = 0L)
      matchup_all <- if (is.null(matchup_idx)) NULL else strenv$jnp$concatenate(list(matchup_idx, matchup_idx), axis = 0L)
      phi_all <- encode_candidate(X_all, p_all, resp_p_all, resp_c_all, stage_all, matchup_all)
      idx_left <- strenv$jnp$arange(N_batch)
      idx_right <- strenv$jnp$arange(N_batch, 2L * N_batch)
      list(
        phi_left = strenv$jnp$take(phi_all, idx_left, axis = 0L),
        phi_right = strenv$jnp$take(phi_all, idx_right, axis = 0L)
      )
    }

    do_forward_and_lik_ <- function(Xl, Xr, pl, pr, resp_p, resp_c, Yb) {
      stage_idx <- neural_stage_index(pl, pr, model_info_local)
      matchup_idx <- NULL
      if (isTRUE(use_matchup_token)) {
        matchup_idx <- compute_matchup_idx(pl, pr)
      }
      if (isTRUE(use_cross_encoder)) {
        logits <- encode_pair_cross(Xl, Xr, pl, pr, resp_p, resp_c, stage_idx, matchup_idx)
      } else {
        phi_pair <- encode_candidate_pair(Xl, Xr, pl, pr, resp_p, resp_c, stage_idx, matchup_idx)
        phi_l <- phi_pair$phi_left
        phi_r <- phi_pair$phi_right
        u_l <- neural_linear_head(phi_l, W_out, b_out)
        u_r <- neural_linear_head(phi_r, W_out, b_out)
        logits <- u_l - u_r
        if (isTRUE(use_cross_term)) {
          logits <- neural_apply_cross_term(logits, phi_l, phi_r,
                                            M_cross, W_cross_out,
                                            out_dim = ai(W_out$shape[[2]]))
        }
      }

      if (likelihood == "bernoulli") {
        logits_vec <- strenv$jnp$squeeze(logits, axis = 1L)
        strenv$numpyro$sample("obs",
                              strenv$numpyro$distributions$Bernoulli(logits = logits_vec),
                              obs = Yb)
      }
      if (likelihood == "categorical") {
        strenv$numpyro$sample("obs",
                              strenv$numpyro$distributions$Categorical(logits = logits),
                              obs = Yb)
      }
      if (likelihood == "normal") {
        mu <- strenv$jnp$squeeze(logits, axis = 1L)
        strenv$numpyro$sample("obs",
                              strenv$numpyro$distributions$Normal(mu, sigma),
                              obs = Yb)
      }
    }

    local_lik <- function() {
      if (isTRUE(subsample_method %in% c("batch", "batch_vi"))) {
        with(strenv$numpyro$plate("data", size = N_local,
                                  # Clamp subsample_size to the available data to avoid
                                  # plate() errors when batch_size > N_local.
                                  subsample_size = ai(min(ai(mcmc_control$batch_size), N_local)),
                                  dim = -1L) %as% "idx", {
                                    Xl_b <- strenv$jnp$take(X_left, idx, axis = 0L)
                                    Xr_b <- strenv$jnp$take(X_right, idx, axis = 0L)
                                    pl_b <- strenv$jnp$take(party_left, idx, axis = 0L)
                                    pr_b <- strenv$jnp$take(party_right, idx, axis = 0L)
                                    resp_p_b <- strenv$jnp$take(resp_party, idx, axis = 0L)
                                    resp_c_b <- strenv$jnp$take(resp_cov, idx, axis = 0L)
                                    Yb <- strenv$jnp$take(Y_obs, idx, axis = 0L)
                                    do_forward_and_lik_(Xl_b, Xr_b, pl_b, pr_b, resp_p_b, resp_c_b, Yb)
                                  })
      } else {
        with(strenv$numpyro$plate("data", size = N_local, dim = -1L), {
          do_forward_and_lik_(X_left, X_right, party_left, party_right,
                              resp_party, resp_cov, Y_obs)
        })
      }
    }

    local_lik()
  }

  BayesianSingleTransformerModel <- function(X, party, resp_party, resp_cov, Y_obs) {
    N_local <- ai(X$shape[[1]])
    D_local <- ai(X$shape[[2]])

    shared_params <- sample_shared_transformer_params(D_local = D_local, pairwise = FALSE)
    params_view <- shared_params$params_view
    E_choice <- shared_params$E_choice
    W_out <- shared_params$W_out
    b_out <- shared_params$b_out
    sigma <- shared_params$sigma

    transformer_model_info <- list(
      model_depth = ModelDepth,
      model_dims = ModelDims,
      n_heads = TransformerHeads,
      head_dim = head_dim
    )
    model_info_local <- list(
      model_dims = ModelDims,
      cand_party_to_resp_idx = cand_party_to_resp_idx_jnp,
      n_party_levels = ai(n_party_levels)
    )

    embed_candidate <- function(X_idx, party_idx, resp_p) {
      neural_build_candidate_tokens_hard(X_idx, party_idx,
                                         model_info = model_info_local,
                                         resp_party_idx = resp_p,
                                         params = params_view)
    }

    run_transformer <- function(tokens) {
      neural_run_transformer(tokens,
                             model_info = transformer_model_info,
                             params = params_view)
    }

    do_forward_and_lik_ <- function(Xb, pb, resp_p, resp_c, Yb) {
      N_batch <- ai(Xb$shape[[1]])
      choice_tok <- neural_build_choice_token(model_info_local, params_view)
      choice_tok <- choice_tok * strenv$jnp$ones(list(N_batch, 1L, 1L))
      ctx_tokens <- neural_build_context_tokens_batch(model_info = model_info_local,
                                                      resp_party_idx = resp_p,
                                                      resp_cov = resp_c,
                                                      params = params_view)
      cand_tokens <- embed_candidate(Xb, pb, resp_p)
      token_parts <- list(choice_tok)
      if (!is.null(ctx_tokens)) {
        token_parts <- c(token_parts, list(ctx_tokens))
      }
      token_parts <- c(token_parts, list(cand_tokens))
      tokens <- strenv$jnp$concatenate(token_parts, axis = 1L)
      tokens <- run_transformer(tokens)
      choice_out <- strenv$jnp$take(tokens, strenv$jnp$arange(1L), axis = 1L)
      choice_out <- strenv$jnp$squeeze(choice_out, axis = 1L)
      logits <- neural_linear_head(choice_out, W_out, b_out)

      if (likelihood == "bernoulli") {
        logits_vec <- strenv$jnp$squeeze(logits, axis = 1L)
        strenv$numpyro$sample("obs",
                              strenv$numpyro$distributions$Bernoulli(logits = logits_vec),
                              obs = Yb)
      }
      if (likelihood == "categorical") {
        strenv$numpyro$sample("obs",
                              strenv$numpyro$distributions$Categorical(logits = logits),
                              obs = Yb)
      }
      if (likelihood == "normal") {
        mu <- strenv$jnp$squeeze(logits, axis = 1L)
        strenv$numpyro$sample("obs",
                              strenv$numpyro$distributions$Normal(mu, sigma),
                              obs = Yb)
      }
    }

    local_lik <- function() {
      if (isTRUE(subsample_method %in% c("batch", "batch_vi"))) {
        with(strenv$numpyro$plate("data", size = N_local,
                                  # Clamp subsample_size to the available data to avoid
                                  # plate() errors when batch_size > N_local.
                                  subsample_size = ai(min(ai(mcmc_control$batch_size), N_local)),
                                  dim = -1L) %as% "idx", {
                                    Xb <- strenv$jnp$take(X, idx, axis = 0L)
                                    pb <- strenv$jnp$take(party, idx, axis = 0L)
                                    resp_p_b <- strenv$jnp$take(resp_party, idx, axis = 0L)
                                    resp_c_b <- strenv$jnp$take(resp_cov, idx, axis = 0L)
                                    Yb <- strenv$jnp$take(Y_obs, idx, axis = 0L)
                                    do_forward_and_lik_(Xb, pb, resp_p_b, resp_c_b, Yb)
                                  })
      } else {
        with(strenv$numpyro$plate("data", size = N_local, dim = -1L), {
          do_forward_and_lik_(X, party, resp_party, resp_cov, Y_obs)
        })
      }
    }

    local_lik()
  }

  # Cross-fitted out-of-sample fit metrics (computed before final full-data fit).
  fit_metrics <- NULL
  neural_skip_oos_eval <- FALSE
  if (exists("neural_oos_eval_internal", inherits = TRUE)) {
    neural_skip_oos_eval <- isTRUE(get("neural_oos_eval_internal", inherits = TRUE))
  }
  if (!isTRUE(neural_skip_oos_eval) && isTRUE(eval_control$enabled)) {
    fit_metrics <- local({
      restore_rng_state <- function(old_seed) {
        if (is.null(old_seed)) {
          if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
            rm(".Random.seed", envir = .GlobalEnv)
          }
        } else {
          assign(".Random.seed", old_seed, envir = .GlobalEnv)
        }
      }
      old_seed_state <- if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
        get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
      } else {
        NULL
      }
      on.exit(restore_rng_state(old_seed_state), add = FALSE)

      fit_metrics <- tryCatch({
        fit_metrics <- NULL
        n_total <- length(Y_use)
        if (n_total > 0L) {
        eval_idx <- seq_len(n_total)
        if (!is.null(eval_control$max_n) &&
            is.finite(eval_control$max_n) &&
            eval_control$max_n > 0L &&
            eval_control$max_n < n_total) {
          eval_seed <- eval_control$seed
          if (is.null(eval_seed) || is.na(eval_seed)) {
            eval_seed <- 123L
          }
          rng <- strenv$np$random$default_rng(as.integer(eval_seed))
          idx_py <- rng$choice(as.integer(n_total),
                               size = as.integer(eval_control$max_n),
                               replace = FALSE)
          eval_idx <- as.integer(reticulate::py_to_r(idx_py)) + 1L
        }

        y_all <- NULL
        y_levels_full <- NULL
        ok_rows <- rep(TRUE, n_total)
        if (likelihood == "bernoulli") {
          y_all <- as.numeric(Y_use)
          ok_rows <- is.finite(y_all) & (y_all %in% c(0, 1))
        } else if (likelihood == "categorical") {
          y_levels_full <- levels(as.factor(Y_use))
          y_all <- match(as.character(Y_use), y_levels_full) - 1L
          ok_rows <- !is.na(y_all)
        } else {
          y_all <- as.numeric(Y_use)
          ok_rows <- is.finite(y_all)
        }
        if (!all(ok_rows)) {
          eval_idx <- eval_idx[ok_rows[eval_idx]]
        }

        n_eval_total <- length(eval_idx)
        if (n_eval_total >= 2L) {
          subset_note <- if (n_eval_total < n_total) {
            sprintf("n=%d/%d", n_eval_total, n_total)
          } else {
            sprintf("n=%d", n_eval_total)
          }

	        cluster_obs <- NULL
	        if (exists("varcov_cluster_variable_", inherits = TRUE)) {
	          cluster_raw <- get("varcov_cluster_variable_", inherits = TRUE)
	          if (!is.null(cluster_raw) && length(cluster_raw) > 0L) {
	            if (pairwise_mode && !is.null(pair_mat) && nrow(pair_mat) > 0) {
	              need <- suppressWarnings(max(pair_mat[, 1], na.rm = TRUE))
	              if (is.finite(need) && length(cluster_raw) >= need) {
	                cluster_obs <- cluster_raw[pair_mat[, 1]]
	              }
	            } else if (!pairwise_mode && length(cluster_raw) == n_total) {
	              cluster_obs <- cluster_raw
	            }
	          }
	        }
	        if (is.null(cluster_obs) || length(cluster_obs) != n_total) {
	          cluster_raw <- NULL

	          subset_by_indi <- function(x) {
	            if (is.null(x) || length(x) == 0L) {
	              return(NULL)
	            }
	            x <- as.vector(x)
	            if (!exists("indi_", inherits = TRUE)) {
	              return(x)
	            }
	            idx <- get("indi_", inherits = TRUE)
	            if (is.null(idx) || length(idx) == 0L) {
	              return(x)
	            }
	            if (length(x) == length(Y_)) {
	              return(x)
	            }
	            max_idx <- suppressWarnings(max(as.integer(idx), na.rm = TRUE))
	            if (is.finite(max_idx) && length(x) >= max_idx) {
	              return(x[idx])
	            }
	            x
	          }

	          resp_id <- if (exists("respondent_id", inherits = TRUE)) {
	            subset_by_indi(get("respondent_id", inherits = TRUE))
	          } else {
	            NULL
	          }
	          task_id <- if (exists("respondent_task_id", inherits = TRUE)) {
	            subset_by_indi(get("respondent_task_id", inherits = TRUE))
	          } else {
	            NULL
	          }

	          if (!is.null(resp_id) && length(resp_id) > 0L) {
	            cluster_raw <- resp_id
	          } else if (!is.null(task_id) && length(task_id) > 0L) {
	            cluster_raw <- task_id
	          }

	          if (!is.null(cluster_raw) && length(cluster_raw) > 0L) {
	            if (pairwise_mode && !is.null(pair_mat) && nrow(pair_mat) > 0) {
	              cluster_obs <- cluster_raw[pair_mat[, 1]]
	            } else if (!pairwise_mode && length(cluster_raw) == n_total) {
	              cluster_obs <- cluster_raw
	            }
	          }
	        }
	        cluster_eval <- if (!is.null(cluster_obs) && length(cluster_obs) == n_total) {
	          cluster_obs[eval_idx]
	        } else {
	          NULL
	        }

        make_folds <- function(n, n_folds, cluster = NULL, seed = 123L) {
          n <- as.integer(n)
          n_folds <- as.integer(n_folds)
          if (n <= 1L || n_folds < 2L) {
            return(NULL)
          }

          restore_rng <- function(old_seed) {
            if (is.null(old_seed)) {
              if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
                rm(".Random.seed", envir = .GlobalEnv)
              }
            } else {
              assign(".Random.seed", old_seed, envir = .GlobalEnv)
            }
          }

          old_seed <- if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
            get(".Random.seed", envir = .GlobalEnv)
          } else {
            NULL
          }
          set.seed(as.integer(seed))
          on.exit(restore_rng(old_seed), add = TRUE)

          if (!is.null(cluster) && length(cluster) == n) {
            cluster <- as.character(cluster)
            if (any(is.na(cluster))) {
              na_idx <- which(is.na(cluster))
              cluster[na_idx] <- paste0("__missing__", na_idx)
            }
            uniq <- unique(cluster)
            k <- min(n_folds, length(uniq))
            if (k >= 2L) {
              uniq <- sample(uniq, length(uniq))
              fold_map <- rep(seq_len(k), length.out = length(uniq))
              names(fold_map) <- uniq
              return(list(fold_id = as.integer(fold_map[cluster]), n_folds = k))
            }
          }

          k <- min(n_folds, n)
          if (k < 2L) {
            return(NULL)
          }
          fold_id <- sample(rep(seq_len(k), length.out = n))
          list(fold_id = as.integer(fold_id), n_folds = k)
        }

        n_folds <- eval_control$n_folds
        if (is.null(n_folds) || !is.finite(n_folds)) {
          n_folds <- 3L
        }
        n_folds <- as.integer(n_folds)
        if (n_folds < 2L) {
          n_folds <- 2L
        }

        folds_out <- make_folds(n_eval_total, n_folds, cluster = cluster_eval, seed = eval_control$seed)
        if (!is.null(folds_out) && !is.null(folds_out$fold_id)) {
          fold_id <- folds_out$fold_id
          n_folds_use <- as.integer(folds_out$n_folds)

          init_model <- body(generate_ModelOutcome_neural)

          compute_auc <- function(y_true, y_score) {
            y_true <- as.numeric(y_true)
            y_score <- as.numeric(y_score)
            ok <- is.finite(y_true) & is.finite(y_score)
            y_true <- y_true[ok]
            y_score <- y_score[ok]
            if (!length(y_true)) return(NA_real_)
            pos <- y_true == 1
            neg <- y_true == 0
            n_pos <- sum(pos)
            n_neg <- sum(neg)
            if (n_pos == 0L || n_neg == 0L) return(NA_real_)
            ranks <- rank(y_score, ties.method = "average")
            (sum(ranks[pos]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
          }
          compute_log_loss <- function(y_true, y_score, eps = 1e-12) {
            y_true <- as.numeric(y_true)
            y_score <- as.numeric(y_score)
            ok <- is.finite(y_true) & is.finite(y_score)
            y_true <- y_true[ok]
            y_score <- y_score[ok]
            if (!length(y_true)) return(NA_real_)
            p <- pmin(pmax(y_score, eps), 1 - eps)
            -mean(y_true * log(p) + (1 - y_true) * log(1 - p))
          }
          compute_multiclass_log_loss <- function(y_true, prob_mat, eps = 1e-12) {
            if (is.null(dim(prob_mat))) {
              prob_mat <- matrix(prob_mat, nrow = length(y_true), byrow = TRUE)
            }
            n_eval <- nrow(prob_mat)
            if (length(y_true) != n_eval) return(NA_real_)
            ok <- !is.na(y_true)
            y_true <- y_true[ok]
            prob_mat <- prob_mat[ok, , drop = FALSE]
            if (!length(y_true)) return(NA_real_)
            idx <- cbind(seq_along(y_true), y_true + 1L)
            p <- prob_mat[idx]
            p <- pmin(pmax(p, eps), 1 - eps)
            -mean(log(p))
          }
          format_metric <- function(label, value, digits = 4) {
            if (is.null(value) || !is.finite(value)) return(NULL)
            fmt <- paste0("%s=%.", digits, "f")
            sprintf(fmt, label, value)
          }

          compute_metrics <- function(y_eval, pred_eval) {
            if (likelihood == "bernoulli") {
              y_eval <- as.numeric(y_eval)
              keep <- is.finite(y_eval) & (y_eval %in% c(0, 1))
              y_eval <- y_eval[keep]
              p <- as.numeric(pred_eval)[keep]
              auc <- compute_auc(y_eval, p)
              log_loss <- compute_log_loss(y_eval, p)
              accuracy <- if (length(y_eval)) mean((p >= 0.5) == y_eval) else NA_real_
              brier <- if (length(y_eval)) mean((p - y_eval) ^ 2) else NA_real_
              return(list(
                likelihood = likelihood,
                n_eval = length(y_eval),
                auc = auc,
                log_loss = log_loss,
                accuracy = accuracy,
                brier = brier
              ))
            }
            if (likelihood == "categorical") {
              y_eval <- as.integer(y_eval)
              prob_mat <- as.matrix(pred_eval)
              keep <- !is.na(y_eval)
              y_eval <- y_eval[keep]
              prob_mat <- prob_mat[keep, , drop = FALSE]
              if (length(y_eval)) {
                log_loss <- compute_multiclass_log_loss(y_eval, prob_mat)
                pred_class <- max.col(prob_mat) - 1L
                accuracy <- mean(pred_class == y_eval, na.rm = TRUE)
              } else {
                log_loss <- NA_real_
                accuracy <- NA_real_
              }
              return(list(
                likelihood = likelihood,
                n_eval = length(y_eval),
                log_loss = log_loss,
                accuracy = accuracy
              ))
            }
            y_eval <- as.numeric(y_eval)
            pred_mu <- as.numeric(pred_eval$mu)
            pred_sigma <- as.numeric(pred_eval$sigma)
            keep <- is.finite(y_eval) & is.finite(pred_mu)
            y_eval <- y_eval[keep]
            pred_mu <- pred_mu[keep]
            if (length(pred_sigma) == 1L && length(y_eval) > 1L) {
              pred_sigma <- rep(pred_sigma, length(y_eval))
            } else if (length(pred_sigma) == length(keep)) {
              pred_sigma <- pred_sigma[keep]
            } else if (length(pred_sigma) == length(y_eval)) {
              pred_sigma <- pred_sigma
            } else {
              pred_sigma <- rep(NA_real_, length(y_eval))
            }
            rmse <- if (length(y_eval)) sqrt(mean((pred_mu - y_eval) ^ 2)) else NA_real_
            mae <- if (length(y_eval)) mean(abs(pred_mu - y_eval)) else NA_real_
            nll <- NA_real_
            if (length(y_eval) &&
                length(pred_sigma) == length(y_eval) &&
                all(is.finite(pred_sigma)) &&
                all(pred_sigma > 0)) {
              nll <- mean(0.5 * log(2 * pi * pred_sigma ^ 2) + (y_eval - pred_mu) ^ 2 / (2 * pred_sigma ^ 2))
            }
            list(
              likelihood = likelihood,
              n_eval = length(y_eval),
              rmse = rmse,
              mae = mae,
              nll = nll
            )
          }

          pred_oos <- NULL
          if (likelihood == "bernoulli") {
            pred_oos <- rep(NA_real_, n_total)
          } else if (likelihood == "categorical") {
            pred_oos <- matrix(NA_real_, nrow = n_total, ncol = as.integer(nOutcomes))
          } else {
            pred_oos <- list(mu = rep(NA_real_, n_total),
                             sigma = rep(NA_real_, n_total))
          }

          by_fold <- vector("list", n_folds_use)
          for (fold in seq_len(n_folds_use)) {
            test_pos <- which(fold_id == fold)
            train_pos <- which(fold_id != fold)
            if (!length(test_pos) || !length(train_pos)) {
              next
            }

            test_idx <- eval_idx[test_pos]
            train_idx <- eval_idx[train_pos]

            fallback <- NULL
            if (likelihood == "bernoulli") {
              y_train <- y_all[train_idx]
              fallback_val <- mean(y_train, na.rm = TRUE)
              if (!is.finite(fallback_val)) fallback_val <- 0.5
              fallback_val <- min(max(fallback_val, 0), 1)
              fallback <- rep(fallback_val, length(test_idx))
            } else if (likelihood == "categorical") {
              y_train <- y_all[train_idx]
              Kc <- as.integer(nOutcomes)
              probs <- vapply(0:(Kc - 1L), function(k_) mean(y_train == k_, na.rm = TRUE), numeric(1))
              if (!any(is.finite(probs)) || sum(probs, na.rm = TRUE) <= 0) {
                probs <- rep(1 / Kc, Kc)
              } else {
                probs[!is.finite(probs)] <- 0
                s <- sum(probs)
                if (s <= 0) probs <- rep(1 / Kc, Kc) else probs <- probs / s
              }
              fallback <- matrix(rep(probs, each = length(test_idx)),
                                 nrow = length(test_idx),
                                 ncol = Kc)
            } else {
              y_train <- y_all[train_idx]
              mu0 <- mean(y_train, na.rm = TRUE)
              if (!is.finite(mu0)) mu0 <- 0
              sigma0 <- suppressWarnings(stats::sd(y_train, na.rm = TRUE))
              if (!is.finite(sigma0) || sigma0 <= 0) sigma0 <- 1
              fallback <- list(mu = rep(mu0, length(test_idx)),
                               sigma = rep(sigma0, length(test_idx)))
            }

            raw_train <- if (pairwise_mode && !is.null(pair_mat) && nrow(pair_mat) > 0) {
              as.integer(c(pair_mat[train_idx, 1], pair_mat[train_idx, 2]))
            } else {
              as.integer(train_idx)
            }
            raw_train <- sort(unique(raw_train))
            if (!length(raw_train)) {
              next
            }

            fold_env <- new.env(parent = environment())
            fold_env$neural_oos_eval_internal <- TRUE
            fold_env$neural_likelihood_override <- likelihood
            fold_env$neural_nOutcomes_override <- nOutcomes
            fold_env$party_levels_fixed <- party_levels
            fold_env$resp_party_levels_fixed <- resp_party_levels
            if (!is.null(y_levels_full)) {
              fold_env$neural_y_levels_override <- y_levels_full
            }

            fold_env$W_ <- W_[raw_train, , drop = FALSE]
            fold_env$Y_ <- Y_[raw_train]
            fold_env$pair_id_ <- if (!is.null(pair_id_)) pair_id_[raw_train] else NULL
            fold_env$profile_order_ <- if (!is.null(profile_order_)) profile_order_[raw_train] else NULL
            fold_env$competing_group_competition_variable_candidate_ <- if (!is.null(competing_group_competition_variable_candidate_)) {
              competing_group_competition_variable_candidate_[raw_train]
            } else {
              NULL
            }
            fold_env$competing_group_variable_respondent_ <- if (!is.null(competing_group_variable_respondent_)) {
              competing_group_variable_respondent_[raw_train]
            } else {
              NULL
            }
            fold_env$competing_group_variable_candidate_ <- if (!is.null(competing_group_variable_candidate_)) {
              competing_group_variable_candidate_[raw_train]
            } else {
              NULL
            }
            if (exists("varcov_cluster_variable_", inherits = TRUE)) {
              cluster_full <- get("varcov_cluster_variable_", inherits = TRUE)
              fold_env$varcov_cluster_variable_ <- if (!is.null(cluster_full)) cluster_full[raw_train] else NULL
            } else {
              fold_env$varcov_cluster_variable_ <- NULL
            }
            if (exists("indi_", inherits = TRUE)) {
              indi_full <- get("indi_", inherits = TRUE)
              fold_env$indi_ <- if (!is.null(indi_full)) indi_full[raw_train] else raw_train
            } else {
            fold_env$indi_ <- raw_train
            }

            fold_seed <- eval_control$seed
            if (is.null(fold_seed) || is.na(fold_seed) || !is.finite(fold_seed)) {
              fold_seed <- 123L
            }
            set.seed(as.integer(fold_seed) + as.integer(fold))

            fit_ok <- tryCatch({
              eval(init_model, envir = fold_env)
              is.function(fold_env$my_model)
            }, error = function(e) FALSE)

            pred_fold <- NULL
            if (isTRUE(fit_ok)) {
              pred_fold <- tryCatch({
                if (pairwise_mode) {
                  X_left_test <- X_left[test_idx, , drop = FALSE]
                  X_right_test <- X_right[test_idx, , drop = FALSE]
                  party_left_test <- party_left[test_idx]
                  party_right_test <- party_right[test_idx]
                  resp_party_test <- resp_party_use[test_idx]
                  resp_cov_test <- if (!is.null(X_use) && n_resp_covariates > 0L) {
                    X_use[test_idx, , drop = FALSE]
                  } else {
                    NULL
                  }
                  fold_env$my_model(
                    X_left_new = X_left_test,
                    X_right_new = X_right_test,
                    party_left_new = party_left_test,
                    party_right_new = party_right_test,
                    resp_party_new = resp_party_test,
                    resp_cov_new = resp_cov_test
                  )
                } else {
                  X_test <- X_single[test_idx, , drop = FALSE]
                  party_test <- party_single[test_idx]
                  resp_party_test <- resp_party_use[test_idx]
                  resp_cov_test <- if (!is.null(X_use) && n_resp_covariates > 0L) {
                    X_use[test_idx, , drop = FALSE]
                  } else {
                    NULL
                  }
                  fold_env$my_model(
                    X_new = X_test,
                    party_new = party_test,
                    resp_party_new = resp_party_test,
                    resp_cov_new = resp_cov_test
                  )
                }
              }, error = function(e) NULL)
            }

            if (is.null(pred_fold)) {
              pred_fold <- fallback
            }

            if (likelihood == "bernoulli") {
              pred_oos[test_idx] <- as.numeric(pred_fold)
              fold_metrics <- compute_metrics(y_all[test_idx], pred_fold)
            } else if (likelihood == "categorical") {
              pred_oos[test_idx, ] <- as.matrix(pred_fold)
              fold_metrics <- compute_metrics(y_all[test_idx], pred_fold)
            } else {
              pred_oos$mu[test_idx] <- as.numeric(pred_fold$mu)
              pred_oos$sigma[test_idx] <- as.numeric(pred_fold$sigma)
              fold_metrics <- compute_metrics(y_all[test_idx], pred_fold)
            }
            fold_metrics$fold <- fold
            fold_metrics$eval_note <- "oos"
            fold_metrics$n_test <- length(test_idx)
            fold_metrics$n_train <- length(train_idx)
            fold_metric_items <- if (likelihood == "bernoulli") {
              Filter(Negate(is.null), list(
                format_metric("AUC", fold_metrics$auc, 4),
                format_metric("LogLoss", fold_metrics$log_loss, 4),
                format_metric("Acc", fold_metrics$accuracy, 3),
                format_metric("Brier", fold_metrics$brier, 4)
              ))
            } else if (likelihood == "categorical") {
              Filter(Negate(is.null), list(
                format_metric("LogLoss", fold_metrics$log_loss, 4),
                format_metric("Acc", fold_metrics$accuracy, 3)
              ))
            } else {
              Filter(Negate(is.null), list(
                format_metric("RMSE", fold_metrics$rmse, 4),
                format_metric("MAE", fold_metrics$mae, 4),
                format_metric("NLL", fold_metrics$nll, 4)
              ))
            }
            if (length(fold_metric_items) > 0L) {
              message(sprintf("Neural OOS fold %d/%d (%s): %s",
                              fold,
                              n_folds_use,
                              ifelse(pairwise_mode, "pairwise", "single"),
                              paste(fold_metric_items, collapse = ", ")))
            }
            by_fold[[fold]] <- fold_metrics
          }

          pred_eval <- NULL
          if (likelihood == "bernoulli") {
            pred_eval <- pred_oos[eval_idx]
          } else if (likelihood == "categorical") {
            pred_eval <- pred_oos[eval_idx, , drop = FALSE]
          } else {
            pred_eval <- list(mu = pred_oos$mu[eval_idx],
                              sigma = pred_oos$sigma[eval_idx])
          }

          overall_metrics <- compute_metrics(y_all[eval_idx], pred_eval)
          overall_metrics$eval_note <- sprintf("oos_%dfold", n_folds_use)
          overall_metrics$eval_subset <- subset_note
          overall_metrics$n_folds <- n_folds_use
          overall_metrics$seed <- eval_control$seed
          overall_metrics$by_fold <- by_fold

          if (pairwise_mode) {
            stage_primary <- party_left == party_right
              if (length(stage_primary) == n_total) {
                stage_primary <- stage_primary[eval_idx]
                by_stage <- list()
                if (any(stage_primary, na.rm = TRUE)) {
                  idx0 <- which(stage_primary)
                  pred_stage <- if (likelihood == "bernoulli") {
                    pred_eval[idx0]
                  } else if (likelihood == "categorical") {
                    pred_eval[idx0, , drop = FALSE]
                  } else {
                    list(mu = pred_eval$mu[idx0],
                         sigma = pred_eval$sigma[idx0])
                  }
                  by_stage$primary <- compute_metrics(y_all[eval_idx][idx0], pred_stage)
                }
                if (any(!stage_primary, na.rm = TRUE)) {
                  idx1 <- which(!stage_primary)
                  pred_stage <- if (likelihood == "bernoulli") {
                    pred_eval[idx1]
                  } else if (likelihood == "categorical") {
                    pred_eval[idx1, , drop = FALSE]
                  } else {
                    list(mu = pred_eval$mu[idx1],
                         sigma = pred_eval$sigma[idx1])
                  }
                  by_stage$general <- compute_metrics(y_all[eval_idx][idx1], pred_stage)
                }
                if (length(by_stage) > 0L) {
                  overall_metrics$by_stage <- by_stage
                }
              }
          }

          fit_metrics <- overall_metrics

          metric_items <- if (likelihood == "bernoulli") {
            Filter(Negate(is.null), list(
              format_metric("AUC", fit_metrics$auc, 4),
              format_metric("LogLoss", fit_metrics$log_loss, 4),
              format_metric("Acc", fit_metrics$accuracy, 3),
              format_metric("Brier", fit_metrics$brier, 4)
            ))
          } else if (likelihood == "categorical") {
            Filter(Negate(is.null), list(
              format_metric("LogLoss", fit_metrics$log_loss, 4),
              format_metric("Acc", fit_metrics$accuracy, 3)
            ))
          } else {
            Filter(Negate(is.null), list(
              format_metric("RMSE", fit_metrics$rmse, 4),
              format_metric("MAE", fit_metrics$mae, 4),
              format_metric("NLL", fit_metrics$nll, 4)
            ))
          }
          if (!is.null(fit_metrics) && length(metric_items) > 0L) {
            message(sprintf("Neural OOS fit metrics (%s, %s, %s): %s",
                            ifelse(pairwise_mode, "pairwise", "single"),
                            fit_metrics$eval_note,
                            subset_note,
                            paste(metric_items, collapse = ", ")))
          }
          }
        }
      }
      fit_metrics
      }, error = function(e) {
        message(sprintf("Neural OOS evaluation failed: %s", conditionMessage(e)))
        NULL
      })

      fit_metrics
  })
  }

  if (likelihood == "categorical") {
    y_levels_override <- NULL
    if (exists("neural_y_levels_override", inherits = TRUE)) {
      y_levels_override <- get("neural_y_levels_override", inherits = TRUE)
    }
    if (!is.null(y_levels_override)) {
      y_fac <- match(as.character(Y_use), as.character(y_levels_override)) - 1L
    } else {
      y_fac <- ai(as.factor(Y_use)) - 1L
    }
    if (any(is.na(y_fac))) {
      stop("Categorical outcome contains unseen/NA levels after applying neural_y_levels_override.",
           call. = FALSE)
    }
    Y_jnp <- strenv$jnp$array(ai(y_fac))$astype(strenv$jnp$int32)
  } else {
    Y_jnp <- strenv$jnp$array(as.numeric(Y_use))$astype(ddtype_)
  }
  resp_party_jnp <- strenv$jnp$array(as.integer(resp_party_use))$astype(strenv$jnp$int32)
  if (n_resp_covariates > 0L) {
    resp_cov_jnp <- strenv$jnp$array(as.matrix(X_use))$astype(ddtype_)
  } else {
    resp_cov_jnp <- strenv$jnp$zeros(list(ai(length(Y_use)), ai(0L)), dtype = ddtype_)
  }

  if (pairwise_mode) {
    X_left_jnp <- strenv$jnp$array(to_index_matrix(X_left))$astype(strenv$jnp$int32)
    X_right_jnp <- strenv$jnp$array(to_index_matrix(X_right))$astype(strenv$jnp$int32)
    party_left_jnp <- strenv$jnp$array(as.integer(party_left))$astype(strenv$jnp$int32)
    party_right_jnp <- strenv$jnp$array(as.integer(party_right))$astype(strenv$jnp$int32)
  } else {
    X_single_jnp <- strenv$jnp$array(to_index_matrix(X_single))$astype(strenv$jnp$int32)
    party_single_jnp <- strenv$jnp$array(as.integer(party_single))$astype(strenv$jnp$int32)
  }

  model_fn_base <- if (pairwise_mode) BayesianPairTransformerModel else BayesianSingleTransformerModel
  model_fn <- model_fn_base
  locscale_reparam <- NULL
  if (!is.null(strenv$numpyro$infer) &&
      reticulate::py_has_attr(strenv$numpyro$infer, "reparam")) {
    reparam_mod <- strenv$numpyro$infer$reparam
    if (reticulate::py_has_attr(reparam_mod, "LocScaleReparam")) {
      locscale_reparam <- reparam_mod$LocScaleReparam
    }
  }
  if (is.null(locscale_reparam)) {
    locscale_reparam <- tryCatch(
      reticulate::import("numpyro.infer.reparam", delay_load = TRUE)$LocScaleReparam,
      error = function(e) NULL
    )
  }
  if (!is.null(locscale_reparam) &&
      reticulate::py_has_attr(strenv$numpyro, "handlers")) {
    reparam_config <- list()
    for (l_ in 1L:ModelDepth) {
      for (base in c("W_q_l", "W_k_l", "W_v_l", "W_o_l", "W_ff1_l", "W_ff2_l")) {
        site <- paste0(base, l_)
        reparam_config[[site]] <- locscale_reparam(centered = 0)
      }
    }
    reparam_config[["W_out"]] <- locscale_reparam(centered = 0)
    reparam_config[["b_out"]] <- locscale_reparam(centered = 0)
    if (isTRUE(use_cross_term)) {
      reparam_config[["M_cross_raw"]] <- locscale_reparam(centered = 0)
    }
    model_fn <- tryCatch(
      strenv$numpyro$handlers$reparam(fn = model_fn_base, config = reparam_config),
      error = function(e) NULL
    )
    if (is.null(model_fn)) {
      model_fn <- model_fn_base
      manual_noncentered_loc_scale <- TRUE
    } else {
      manual_noncentered_loc_scale <- FALSE
    }
  } else {
    manual_noncentered_loc_scale <- TRUE
  }

  t0_ <- Sys.time()
  output_only_mode <- identical(tolower(as.character(uncertainty_scope)), "output")
  use_svi <- isTRUE(output_only_mode) || identical(subsample_method, "batch_vi")
  run_mcmc_after_svi <- isTRUE(output_only_mode) && isTRUE(subsample_method %in% c("batch", "full"))
  SVIParams <- NULL
  SVIInitValues <- NULL
  svi_loss_curve <- NULL
  if (isTRUE(use_svi)) {
    if (isTRUE(output_only_mode)) {
      message("Enlisting SVI with autoguide for output-only uncertainty...")
    } else {
      message("Enlisting SVI with autoguide for minibatched likelihood...")
    }
    if (!is.null(strenv$numpyro) && reticulate::py_has_attr(strenv$numpyro, "clear_param_store")) {
      tryCatch(strenv$numpyro$clear_param_store(), error = function(e) NULL)
    }
    guide_name <- if (!is.null(mcmc_control$vi_guide)) {
      tolower(as.character(mcmc_control$vi_guide))
    } else {
      "auto_diagonal"
    }
    if (length(guide_name) != 1L || is.na(guide_name) || !nzchar(guide_name)) {
      guide_name <- "auto_diagonal"
    }
    guide <- switch(guide_name,
                    auto_delta = strenv$numpyro$infer$autoguide$AutoDelta(model_fn),
                    auto_normal = strenv$numpyro$infer$autoguide$AutoNormal(model_fn),
                    auto_diagonal = strenv$numpyro$infer$autoguide$AutoDiagonalNormal(model_fn),
                    stop(sprintf("Unknown vi_guide '%s' for SVI.", guide_name), call. = FALSE))
    n_particles <- ai(mcmc_control$svi_num_particles)
    if (length(n_particles) != 1L || is.na(n_particles) || n_particles < 1L) {
      n_particles <- 1L
    }
    optimizer_tag <- if (!is.null(mcmc_control$optimizer)) {
      tolower(as.character(mcmc_control$optimizer))
    } else {
      "adam"
    }
    if (length(optimizer_tag) != 1L || is.na(optimizer_tag) || !nzchar(optimizer_tag)) {
      optimizer_tag <- "adam"
    }
    if (!optimizer_tag %in% c("adam", "adamw", "adabelief", "muon")) {
      stop(
        sprintf("Unknown optimizer '%s' for SVI.", optimizer_tag),
        call. = FALSE
      )
    }
    svi_lr <- as.numeric(mcmc_control$svi_lr)
    if (length(svi_lr) != 1L || is.na(svi_lr) || !is.finite(svi_lr) || svi_lr <= 0) {
      svi_lr <- 0.01
    }
    schedule_tag <- if (!is.null(mcmc_control$svi_lr_schedule)) {
      tolower(as.character(mcmc_control$svi_lr_schedule))
    } else {
      "warmup_cosine"
    }
    if (length(schedule_tag) != 1L || is.na(schedule_tag) || !nzchar(schedule_tag)) {
      schedule_tag <- "warmup_cosine"
    }
    if (!schedule_tag %in% c("none", "constant", "cosine", "warmup_cosine")) {
      stop(
        sprintf("Unknown svi_lr_schedule '%s'.", schedule_tag),
        call. = FALSE
      )
    }
    svi_steps_input <- mcmc_control$svi_steps
    svi_steps <- NULL
    if (is.character(svi_steps_input)) {
      steps_tag <- tolower(as.character(svi_steps_input))
      if (length(steps_tag) == 1L && !is.na(steps_tag) && nzchar(steps_tag) &&
          identical(steps_tag, "optimal")) {
        n_obs_svi <- length(Y_use)
        pairwise_scaling <- pairwise_mode
        if (isTRUE(pairwise_mode) && !is.null(pair_mat) && nrow(pair_mat) > 0L) {
          n_obs_svi <- nrow(pair_mat)
          pairwise_scaling <- TRUE
        }
        if ((is.null(pair_mat) || nrow(pair_mat) < 1L) &&
            isTRUE(diff) && !is.null(pair_id_) && length(pair_id_) > 0L) {
          pair_id_use <- pair_id_
          pair_id_use <- pair_id_use[!is.na(pair_id_use)]
          if (length(pair_id_use) > 0L) {
            n_obs_svi <- length(unique(pair_id_use))
            pairwise_scaling <- TRUE
          }
        }
        svi_subsample_method <- if (isTRUE(subsample_method %in% c("batch", "batch_vi"))) {
          "batch_vi"
        } else {
          subsample_method
        }
        svi_steps <- neural_optimal_svi_steps(
          n_obs = n_obs_svi,
          n_factors = length(factor_levels_int),
          factor_levels = factor_levels_int,
          model_dims = ModelDims,
          model_depth = ModelDepth,
          n_party_levels = n_party_levels,
          n_resp_party_levels = n_resp_party_levels,
          n_resp_covariates = n_resp_covariates,
          n_outcomes = nOutcomes,
          pairwise_mode = pairwise_scaling,
          use_matchup_token = use_matchup_token,
          use_cross_encoder = use_cross_encoder,
          use_cross_term = use_cross_term,
          batch_size = mcmc_control$batch_size,
          subsample_method = svi_subsample_method
        )
        mcmc_control$svi_steps <- svi_steps
        message(sprintf("Using svi_steps='optimal' => %d steps.", svi_steps))
      }
    }
    if (is.null(svi_steps)) {
      svi_steps <- as.integer(svi_steps_input)
      if (length(svi_steps) != 1L || is.na(svi_steps) || svi_steps < 1L) {
        svi_steps <- 1L
      }
    }
    warmup_frac <- if (!is.null(mcmc_control$svi_lr_warmup_frac)) {
      as.numeric(mcmc_control$svi_lr_warmup_frac)
    } else {
      0.1
    }
    if (length(warmup_frac) != 1L || is.na(warmup_frac) || !is.finite(warmup_frac)) {
      warmup_frac <- 0.1
    }
    warmup_frac <- max(0, min(warmup_frac, 0.9))
    decay_steps <- max(2L, svi_steps)
    warmup_steps <- if (schedule_tag == "warmup_cosine") {
      max(1L, min(as.integer(round(svi_steps * warmup_frac)), decay_steps - 1L))
    } else {
      0L
    }
    end_factor <- if (!is.null(mcmc_control$svi_lr_end_factor)) {
      as.numeric(mcmc_control$svi_lr_end_factor)
    } else {
      0.01
    }
    if (length(end_factor) != 1L || is.na(end_factor) || !is.finite(end_factor)) {
      end_factor <- 0.01
    }
    end_factor <- max(0, min(end_factor, 1))
    lr_schedule <- if (schedule_tag == "warmup_cosine") {
      strenv$optax$warmup_cosine_decay_schedule(
        init_value = svi_lr * end_factor,
        peak_value = svi_lr,
        warmup_steps = warmup_steps,
        decay_steps = decay_steps,
        end_value = svi_lr * end_factor
      )
    } else if (schedule_tag == "cosine") {
      strenv$optax$cosine_decay_schedule(
        init_value = svi_lr,
        decay_steps = decay_steps,
        alpha = end_factor
      )
    } else {
      svi_lr
    }
    svi_optim <- if (optimizer_tag == "adam") {
      strenv$numpyro$optim$Adam(lr_schedule)
    } else if (optimizer_tag == "adamw") {
      if (reticulate::py_has_attr(strenv$numpyro$optim, "AdamW")) {
        strenv$numpyro$optim$AdamW(lr_schedule)
      } else if (reticulate::py_has_attr(strenv$optax, "adamw")) {
        optax_optim <- strenv$optax$adamw(learning_rate = lr_schedule)
        if (reticulate::py_has_attr(strenv$numpyro$optim, "optax_to_numpyro")) {
          strenv$numpyro$optim$optax_to_numpyro(optax_optim)
        } else {
          optax_optim
        }
      } else {
        stop(
          "optimizer='adamw' requested, but neither numpyro.optim.AdamW nor optax.adamw is available.",
          call. = FALSE
        )
      }
    } else if (optimizer_tag == "muon") {
      if (reticulate::py_has_attr(strenv$optax, "contrib") &&
          reticulate::py_has_attr(strenv$optax$contrib, "muon")) {
        muon_dimnums <- NULL
        if (reticulate::py_has_attr(strenv$optax$contrib, "MuonDimensionNumbers")) {
          muon_dimnums <- tryCatch({
            reticulate::py_run_string(
              "import re\nimport jax\nimport optax\n\n_STRATEGIZE_MUON_KEY_RE = re.compile(r'^(W_(q|k|v|o)_l\\d+|W_ff(1|2)_l\\d+|W_out|M_cross_raw)$')\n\ndef _strategize_muon_dimnums(params):\n    tree_util = jax.tree_util\n\n    if hasattr(tree_util, 'tree_flatten_with_path'):\n        path_leaves, treedef = tree_util.tree_flatten_with_path(params)\n        out_leaves = []\n        for path, value in path_leaves:\n            name = ''\n            for entry in reversed(path):\n                if hasattr(entry, 'key'):\n                    name = str(entry.key)\n                    break\n\n            use_muon = False\n            try:\n                ndim = getattr(value, 'ndim', None)\n                if ndim == 2 and _STRATEGIZE_MUON_KEY_RE.match(name):\n                    use_muon = True\n            except Exception:\n                use_muon = False\n\n            out_leaves.append(optax.contrib.MuonDimensionNumbers() if use_muon else None)\n        return tree_util.tree_unflatten(treedef, out_leaves)\n\n    # Fallback: assume dict-like params and use last-level keys.\n    if hasattr(params, 'items'):\n        out = {}\n        for k, v in params.items():\n            name = str(k)\n            use_muon = getattr(v, 'ndim', None) == 2 and _STRATEGIZE_MUON_KEY_RE.match(name)\n            out[k] = optax.contrib.MuonDimensionNumbers() if use_muon else None\n        try:\n            return params.__class__(out)\n        except Exception:\n            return out\n\n    # Last resort: keep Muon off and fall back to AdamW.\n    return tree_util.tree_map(lambda _: None, params)\n"
            )
            reticulate::py_eval("_strategize_muon_dimnums")
          }, error = function(e) NULL)
        }

        muon_kwargs <- list(
          learning_rate = lr_schedule,
          adam_weight_decay = 1e-4,
          consistent_rms = 0.2
        )
        if (!is.null(muon_dimnums)) {
          muon_kwargs$muon_weight_dimension_numbers <- muon_dimnums
        }

        optax_optim <- tryCatch(
          do.call(strenv$optax$contrib$muon, muon_kwargs),
          error = function(e) {
            muon_kwargs_fallback <- list(learning_rate = lr_schedule)
            if (!is.null(muon_dimnums)) {
              muon_kwargs_fallback$muon_weight_dimension_numbers <- muon_dimnums
            }
            tryCatch(
              do.call(strenv$optax$contrib$muon, muon_kwargs_fallback),
              error = function(e2) strenv$optax$contrib$muon(learning_rate = lr_schedule)
            )
          }
        )
        if (reticulate::py_has_attr(strenv$numpyro$optim, "optax_to_numpyro")) {
          strenv$numpyro$optim$optax_to_numpyro(optax_optim)
        } else {
          optax_optim
        }
      } else {
        stop(
          "optimizer='muon' requested, but optax.contrib.muon is unavailable.",
          call. = FALSE
        )
      }
    } else {
      optax_optim <- strenv$optax$adabelief(learning_rate = lr_schedule)
      if (reticulate::py_has_attr(strenv$numpyro$optim, "optax_to_numpyro")) {
        strenv$numpyro$optim$optax_to_numpyro(optax_optim)
      } else {
        optax_optim
      }
    }
    svi <- strenv$numpyro$infer$SVI(
      model = model_fn,
      guide = guide,
      optim = svi_optim,
      loss = strenv$numpyro$infer$Trace_ELBO(
        num_particles = n_particles
      )
    )
    rng_key <- strenv$jax$random$PRNGKey(ai(runif(1, 0, 10000)))
    if (pairwise_mode) {
      svi_result <- svi$run(rng_key,
                            ai(svi_steps),
                            X_left = X_left_jnp,
                            X_right = X_right_jnp,
                            party_left = party_left_jnp,
                            party_right = party_right_jnp,
                            resp_party = resp_party_jnp,
                            resp_cov = resp_cov_jnp,
                            Y_obs = Y_jnp)
    } else {
      svi_result <- svi$run(rng_key,
                            ai(svi_steps),
                            X = X_single_jnp,
                            party = party_single_jnp,
                            resp_party = resp_party_jnp,
                            resp_cov = resp_cov_jnp,
                            Y_obs = Y_jnp)
    }
    svi_loss_curve <- tryCatch({
      if (reticulate::py_has_attr(svi_result, "losses")) {
        as.numeric(reticulate::py_to_r(svi_result$losses))
      } else if (!is.null(svi_result$losses)) {
        as.numeric(reticulate::py_to_r(svi_result$losses))
      } else {
        NULL
      }
    }, error = function(e) NULL)
    if (!is.null(svi_loss_curve) && length(svi_loss_curve) > 0L &&
        identical(subsample_method, "batch_vi")) {
      svi_loss_curve <- as.numeric(svi_loss_curve)
      svi_loss_curve[!is.finite(svi_loss_curve)] <- NA_real_
      try(suppressWarnings(plot(svi_loss_curve,
                                type = "l",
                                main = "SVI ELBO loss",
                                xlab = "Iteration",
                                ylab = "ELBO loss")), TRUE)
      finite_idx <- is.finite(svi_loss_curve)
      if (sum(finite_idx, na.rm = TRUE) >= 2L) {
        try(points(lowess(svi_loss_curve[finite_idx]),
                   type = "l",
                   lwd = 2,
                   col = "red"), TRUE)
      }
    }
    svi_state <- if (!is.null(svi_result$state)) {
      svi_result$state
    } else if (length(svi_result) > 0L) {
      svi_result[[1]]
    } else {
      svi_result
    }
    params <- svi$get_params(svi_state)
    SVIParams <- params
    n_draws <- ai(mcmc_control$svi_num_draws)
    if (length(n_draws) != 1L || is.na(n_draws) || !is.finite(n_draws) || n_draws < 1L) {
      n_draws <- 1L
    }
    sample_key <- strenv$jax$random$PRNGKey(ai(runif(1, 0, 10000)))
    posterior_samples <- guide$sample_posterior(
      sample_key,
      params,
      sample_shape = reticulate::tuple(ai(n_draws))
    )
    if (isTRUE(run_mcmc_after_svi)) {
      SVIInitValues <- lapply(posterior_samples, function(x) {
        strenv$jnp$mean(x, axis = 0L)
      })
      names(SVIInitValues) <- names(posterior_samples)
    } else {
      PosteriorDraws <- lapply(posterior_samples, function(x) {
        strenv$jnp$expand_dims(x, 0L)
      })
      names(PosteriorDraws) <- names(posterior_samples)
    }
    message(sprintf("\n SVI Runtime: %.3f min",
                    as.numeric(difftime(Sys.time(), t0_, units = "secs"))/60))
  }

  if (!isTRUE(use_svi) || isTRUE(run_mcmc_after_svi)) {
    strenv$numpyro$set_host_device_count(mcmc_control$n_chains)

    init_strategy <- NULL
    if (!is.null(SVIInitValues) && length(SVIInitValues) > 0L) {
      init_to_value <- NULL
      if (!is.null(strenv$numpyro$infer) &&
          reticulate::py_has_attr(strenv$numpyro$infer, "initialization") &&
          reticulate::py_has_attr(strenv$numpyro$infer$initialization, "init_to_value")) {
        init_to_value <- strenv$numpyro$infer$initialization$init_to_value
      } else if (!is.null(strenv$numpyro$infer) &&
                 reticulate::py_has_attr(strenv$numpyro$infer, "init_to_value")) {
        init_to_value <- strenv$numpyro$infer$init_to_value
      }
      if (!is.null(init_to_value)) {
        init_strategy <- init_to_value(values = SVIInitValues)
        message("Initializing MCMC with SVI posterior means...")
      }
    }

    if (identical(subsample_method, "batch")) {
      message("Enlisting HMCECS kernels for subsampled likelihood...")
      nuts_kernel <- if (is.null(init_strategy)) {
        strenv$numpyro$infer$NUTS(model_fn)
      } else {
        strenv$numpyro$infer$NUTS(model_fn, init_strategy = init_strategy)
      }
      kernel <- strenv$numpyro$infer$HMCECS(
        nuts_kernel,
        num_blocks = if (!is.null(mcmc_control$num_blocks)) ai(mcmc_control$num_blocks) else ai(4L)
      )
    } else {
      message("Enlisting NUTS kernels for full-data likelihood...")
      kernel <- if (is.null(init_strategy)) {
        strenv$numpyro$infer$NUTS(
          model_fn,
          max_tree_depth = ai(8L),
          target_accept_prob = 0.85
        )
      } else {
        strenv$numpyro$infer$NUTS(
          model_fn,
          max_tree_depth = ai(8L),
          target_accept_prob = 0.85,
          init_strategy = init_strategy
        )
      }
    }

    sampler <- strenv$numpyro$infer$MCMC(
      sampler = kernel,
      num_warmup = mcmc_control$n_samples_warmup,
      num_samples = mcmc_control$n_samples_mcmc,
      thinning   = mcmc_control$n_thin_by,
      chain_method = ifelse(!is.null(mcmc_control$chain_method), yes = mcmc_control$chain_method, no = "parallel"),
      num_chains = mcmc_control$n_chains,
      jit_model_args = TRUE,
      progress_bar = TRUE
    )

    if (pairwise_mode) {
      sampler$run(strenv$jax$random$PRNGKey(ai(runif(1, 0, 10000))),
                  X_left = X_left_jnp,
                  X_right = X_right_jnp,
                  party_left = party_left_jnp,
                  party_right = party_right_jnp,
                  resp_party = resp_party_jnp,
                  resp_cov = resp_cov_jnp,
                  Y_obs = Y_jnp)
    } else {
      sampler$run(strenv$jax$random$PRNGKey(ai(runif(1, 0, 10000))),
                  X = X_single_jnp,
                  party = party_single_jnp,
                  resp_party = resp_party_jnp,
                  resp_cov = resp_cov_jnp,
                  Y_obs = Y_jnp)
    }
    PosteriorDraws <- sampler$get_samples(group_by_chain = TRUE)
    message(sprintf("\n MCMC Runtime: %.3f min",
                    as.numeric(difftime(Sys.time(), t0_, units = "secs"))/60))
  }

  mean_param <- function(x) { strenv$jnp$mean(x, 0L:1L) }
  get_centered_factor_draws <- function(name) {
    draws <- PosteriorDraws[[name]]
    if (!is.null(draws)) {
      return(draws)
    }
    raw_name <- paste0(name, "_raw")
    raw_draws <- PosteriorDraws[[raw_name]]
    if (is.null(raw_draws)) {
      return(NULL)
    }
    d_idx <- suppressWarnings(as.integer(sub("E_factor_", "", name)))
    if (is.na(d_idx) || d_idx < 1L || d_idx > length(factor_levels_int)) {
      return(raw_draws - strenv$jnp$mean(raw_draws, axis = 2L, keepdims = TRUE))
    }
    n_real <- ai(factor_levels_int[d_idx])
    n_levels_raw <- tryCatch(
      as.integer(reticulate::py_to_r(raw_draws$shape[[3]])),
      error = function(e) NA_integer_
    )
    if (is.na(n_levels_raw) || n_levels_raw <= n_real) {
      return(raw_draws - strenv$jnp$mean(raw_draws, axis = 2L, keepdims = TRUE))
    }
    real_idx <- strenv$jnp$arange(n_real)
    raw_real <- strenv$jnp$take(raw_draws, real_idx, axis = 2L)
    mean_real <- strenv$jnp$mean(raw_real, axis = 2L, keepdims = TRUE)
    centered_real <- raw_real - mean_real
    missing_draw <- strenv$jnp$take(raw_draws, ai(n_real), axis = 2L)
    missing_draw <- strenv$jnp$expand_dims(missing_draw, axis = 2L)
    strenv$jnp$concatenate(list(centered_real, missing_draw), axis = 2L)
  }
  get_loc_scale_draws <- function(name, scale_name) {
    draws <- PosteriorDraws[[name]]
    if (!is.null(draws)) {
      return(draws)
    }
    base_names <- c(paste0(name, "_decentered"),
                    paste0(name, "_base"),
                    paste0(name, "_z"))
    base_draws <- NULL
    for (base_name in base_names) {
      if (!is.null(PosteriorDraws[[base_name]])) {
        base_draws <- PosteriorDraws[[base_name]]
        break
      }
    }
    if (is.null(base_draws)) {
      return(NULL)
    }
    scale_draws <- PosteriorDraws[[scale_name]]
    if (is.null(scale_draws)) {
      return(NULL)
    }
    scale_shape <- tryCatch(
      as.integer(reticulate::py_to_r(scale_draws$shape)),
      error = function(e) NULL
    )
    base_shape <- tryCatch(
      as.integer(reticulate::py_to_r(base_draws$shape)),
      error = function(e) NULL
    )
    if (is.null(scale_shape) || is.null(base_shape)) {
      return(scale_draws * base_draws)
    }
    extra_dims <- length(base_shape) - length(scale_shape)
    if (extra_dims > 0L) {
      reshape_dims <- c(scale_shape, rep(1L, extra_dims))
      scale_draws <- strenv$jnp$reshape(scale_draws, as.list(reshape_dims))
    }
    scale_draws * base_draws
  }
  get_cross_draws <- function() {
    draws <- PosteriorDraws$M_cross
    if (!is.null(draws)) {
      return(draws)
    }
    raw_draws <- PosteriorDraws$M_cross_raw
    if (is.null(raw_draws)) {
      raw_draws <- get_loc_scale_draws("M_cross_raw", "tau_cross")
    }
    if (is.null(raw_draws)) {
      return(NULL)
    }
    trans_axes <- list(0L, 1L, 3L, 2L)
    0.5 * (raw_draws - strenv$jnp$transpose(raw_draws, trans_axes))
  }
  get_param_draws <- function(name) {
    if (startsWith(name, "E_factor_")) {
      return(get_centered_factor_draws(name))
    }
    if (identical(name, "M_cross")) {
      return(get_cross_draws())
    }
    if (identical(name, "W_out")) {
      return(get_loc_scale_draws("W_out", "tau_w_out"))
    }
    if (identical(name, "b_out")) {
      return(get_loc_scale_draws("b_out", "tau_b"))
    }
    if (grepl("^W_(q|k|v|o)_l\\d+$", name) ||
        grepl("^W_ff(1|2)_l\\d+$", name)) {
      layer_id <- sub(".*_l", "", name)
      tau_name <- paste0("tau_w_", layer_id)
      return(get_loc_scale_draws(name, tau_name))
    }
    PosteriorDraws[[name]]
  }

  p2d_output_only <- isTRUE(output_only_mode)
  get_svi_param <- function(name) {
    if (is.null(SVIParams)) {
      return(NULL)
    }
    tryCatch(SVIParams[[name]], error = function(e) NULL)
  }
  get_site_mean_or_param <- function(name) {
    draws <- PosteriorDraws[[name]]
    if (!is.null(draws)) {
      return(mean_param(draws))
    }
    get_svi_param(name)
  }

  ParamsMean <- list()
  ParamsMean$E_party <- get_site_mean_or_param("E_party")
  ParamsMean$E_resp_party <- get_site_mean_or_param("E_resp_party")
  ParamsMean$E_choice <- get_site_mean_or_param("E_choice")
  if (is.null(ParamsMean$E_party) || is.null(ParamsMean$E_resp_party) || is.null(ParamsMean$E_choice)) {
    stop("Neural model is missing required embedding estimates.", call. = FALSE)
  }

  maybe_site <- function(name, assign_as = name) {
    value <- get_site_mean_or_param(name)
    if (!is.null(value)) {
      ParamsMean[[assign_as]] <<- value
    }
    invisible(value)
  }

  maybe_site("E_sep")
  maybe_site("E_segment")
  maybe_site("E_feature_id")
  maybe_site("E_rel")
  maybe_site("E_stage")
  maybe_site("E_matchup")

  W_out_draws <- get_loc_scale_draws("W_out", "tau_w_out")
  if (!is.null(W_out_draws)) {
    ParamsMean$W_out <- mean_param(W_out_draws)
  }
  b_out_draws <- get_loc_scale_draws("b_out", "tau_b")
  if (!is.null(b_out_draws)) {
    ParamsMean$b_out <- mean_param(b_out_draws)
  }

  cross_draws <- get_cross_draws()
  if (!is.null(cross_draws)) {
    ParamsMean$M_cross <- mean_param(cross_draws)
  } else if (isTRUE(p2d_output_only) && isTRUE(pairwise_mode) && isTRUE(use_cross_term)) {
    M_cross_raw <- get_svi_param("M_cross_raw")
    if (!is.null(M_cross_raw)) {
      ParamsMean$M_cross <- 0.5 * (M_cross_raw - strenv$jnp$transpose(M_cross_raw))
    }
  }

  if (!is.null(PosteriorDraws$W_cross_out)) {
    ParamsMean$W_cross_out <- mean_param(PosteriorDraws$W_cross_out)
  }
  if (likelihood == "normal") {
    ParamsMean$sigma <- mean_param(PosteriorDraws$sigma)
  }

  if (n_resp_covariates > 0L) {
    maybe_site("W_resp_x")
  }

  for (d_ in seq_along(factor_levels_int)) {
    name <- paste0("E_factor_", d_)
    draws <- get_centered_factor_draws(name)
    if (!is.null(draws)) {
      ParamsMean[[name]] <- mean_param(draws)
    } else if (isTRUE(p2d_output_only)) {
      raw_name <- paste0(name, "_raw")
      raw_val <- get_svi_param(raw_name)
      if (!is.null(raw_val)) {
        ParamsMean[[name]] <- center_factor_embeddings(raw_val, factor_levels_int[d_])
      }
    }
  }

  for (l_ in 1L:ModelDepth) {
    alpha_attn_name <- paste0("alpha_attn_l", l_)
    alpha_ff_name <- paste0("alpha_ff_l", l_)
    maybe_site(alpha_attn_name)
    maybe_site(alpha_ff_name)

    maybe_site(paste0("RMS_attn_l", l_))
    maybe_site(paste0("RMS_ff_l", l_))

    for (base in c("W_q_l", "W_k_l", "W_v_l", "W_o_l", "W_ff1_l", "W_ff2_l")) {
      name <- paste0(base, l_)
      tau_name <- paste0("tau_w_", l_)
      draws <- get_loc_scale_draws(name, tau_name)
      if (!is.null(draws)) {
        ParamsMean[[name]] <- mean_param(draws)
      } else if (isTRUE(p2d_output_only)) {
        value <- get_svi_param(name)
        if (!is.null(value)) {
          ParamsMean[[name]] <- value
        }
      }
    }
  }
  maybe_site("RMS_final")

  TransformerPredict_pair <- function(params, Xl_new, Xr_new, pl_new, pr_new,
                                      resp_party_new = NULL, resp_cov_new = NULL,
                                      return_logits = FALSE) {
    Xl <- strenv$jnp$array(to_index_matrix(Xl_new))$astype(strenv$jnp$int32)
    Xr <- strenv$jnp$array(to_index_matrix(Xr_new))$astype(strenv$jnp$int32)
    pl <- strenv$jnp$array(as.integer(pl_new))$astype(strenv$jnp$int32)
    pr <- strenv$jnp$array(as.integer(pr_new))$astype(strenv$jnp$int32)
    if (is.null(resp_party_new)) {
      resp_party_new <- rep(0L, nrow(Xl_new))
    }
    resp_p <- strenv$jnp$array(as.integer(resp_party_new))$astype(strenv$jnp$int32)
    if (is.null(resp_cov_new)) {
      resp_cov_new <- if (!is.null(resp_cov_mean)) {
        matrix(rep(resp_cov_mean, each = nrow(Xl_new)), nrow = nrow(Xl_new))
      } else {
        matrix(0, nrow = nrow(Xl_new), ncol = 0L)
      }
    }
    resp_c <- strenv$jnp$array(as.matrix(resp_cov_new))$astype(ddtype_)

    model_info_local <- list(
      model_dims = ModelDims,
      cand_party_to_resp_idx = cand_party_to_resp_idx_jnp,
      n_party_levels = ai(n_party_levels)
    )

    embed_candidate <- function(X_idx, party_idx, resp_p) {
      neural_build_candidate_tokens_hard(X_idx, party_idx,
                                         model_info = model_info_local,
                                         resp_party_idx = resp_p,
                                         params = params)
    }

    add_segment_embedding <- function(tokens, segment_idx) {
      neural_add_segment_embedding(tokens, segment_idx,
                                   model_info = model_info_local,
                                   params = params)
    }

    run_transformer <- function(tokens) {
      neural_run_transformer(tokens,
                             model_info = list(model_depth = ModelDepth,
                                               model_dims = ModelDims,
                                               n_heads = TransformerHeads,
                                               head_dim = head_dim),
                             params = params)
    }

    build_context_tokens <- function(stage_idx, resp_p, resp_c, matchup_idx = NULL) {
      neural_build_context_tokens_batch(model_info = model_info_local,
                                        resp_party_idx = resp_p,
                                        stage_idx = stage_idx,
                                        matchup_idx = matchup_idx,
                                        resp_cov = resp_c,
                                        params = params)
    }

    build_sep_token <- function(N_batch) {
      neural_build_sep_token(model_info_local,
                             n_batch = N_batch,
                             params = params)
    }

    encode_pair_cross <- function(Xl, Xr, pl, pr, resp_p, resp_c, stage_idx, matchup_idx = NULL) {
      N_batch <- ai(Xl$shape[[1]])
      choice_tok <- neural_build_choice_token(model_info_local, params)
      choice_tok <- choice_tok * strenv$jnp$ones(list(N_batch, 1L, 1L))
      ctx_tokens <- build_context_tokens(stage_idx, resp_p, resp_c, matchup_idx)
      left_tokens <- add_segment_embedding(embed_candidate(Xl, pl, resp_p), 0L)
      right_tokens <- add_segment_embedding(embed_candidate(Xr, pr, resp_p), 1L)
      sep_tok <- build_sep_token(N_batch)
      token_parts <- list(choice_tok)
      if (!is.null(ctx_tokens)) {
        token_parts <- c(token_parts, list(ctx_tokens))
      }
      token_parts <- c(token_parts, list(sep_tok, left_tokens, sep_tok, right_tokens))
      tokens <- strenv$jnp$concatenate(token_parts, axis = 1L)
      tokens <- run_transformer(tokens)
      cls_out <- strenv$jnp$take(tokens, strenv$jnp$arange(1L), axis = 1L)
      cls_out <- strenv$jnp$squeeze(cls_out, axis = 1L)
      neural_linear_head(cls_out, params$W_out, params$b_out)
    }

    encode_candidate <- function(Xa, pa, resp_p, resp_c, stage_idx, matchup_idx = NULL) {
      N_batch <- ai(Xa$shape[[1]])
      choice_tok <- neural_build_choice_token(model_info_local, params)
      choice_tok <- choice_tok * strenv$jnp$ones(list(N_batch, 1L, 1L))
      ctx_tokens <- build_context_tokens(stage_idx, resp_p, resp_c, matchup_idx)
      cand_tokens <- embed_candidate(Xa, pa, resp_p)
      if (is.null(ctx_tokens)) {
        tokens <- strenv$jnp$concatenate(list(choice_tok, cand_tokens), axis = 1L)
      } else {
        tokens <- strenv$jnp$concatenate(list(choice_tok, ctx_tokens, cand_tokens), axis = 1L)
      }
      tokens <- run_transformer(tokens)
      choice_out <- strenv$jnp$take(tokens, strenv$jnp$arange(1L), axis = 1L)
      strenv$jnp$squeeze(choice_out, axis = 1L)
    }

    encode_candidate_pair <- function(Xl, Xr, pl, pr, resp_p, resp_c, stage_idx, matchup_idx = NULL) {
      N_batch <- ai(Xl$shape[[1]])
      X_all <- strenv$jnp$concatenate(list(Xl, Xr), axis = 0L)
      p_all <- strenv$jnp$concatenate(list(pl, pr), axis = 0L)
      resp_p_all <- strenv$jnp$concatenate(list(resp_p, resp_p), axis = 0L)
      resp_c_all <- if (is.null(resp_c)) NULL else strenv$jnp$concatenate(list(resp_c, resp_c), axis = 0L)
      stage_all <- if (is.null(stage_idx)) NULL else strenv$jnp$concatenate(list(stage_idx, stage_idx), axis = 0L)
      matchup_all <- if (is.null(matchup_idx)) NULL else strenv$jnp$concatenate(list(matchup_idx, matchup_idx), axis = 0L)
      phi_all <- encode_candidate(X_all, p_all, resp_p_all, resp_c_all, stage_all, matchup_all)
      idx_left <- strenv$jnp$arange(N_batch)
      idx_right <- strenv$jnp$arange(N_batch, 2L * N_batch)
      list(
        phi_left = strenv$jnp$take(phi_all, idx_left, axis = 0L),
        phi_right = strenv$jnp$take(phi_all, idx_right, axis = 0L)
      )
    }

    stage_idx <- neural_stage_index(pl, pr, model_info_local)
    matchup_idx <- NULL
    if (!is.null(params$E_matchup)) {
      matchup_idx <- neural_matchup_index(pl, pr, model_info_local)
    }
    if (isTRUE(use_cross_encoder)) {
      logits <- encode_pair_cross(Xl, Xr, pl, pr, resp_p, resp_c, stage_idx, matchup_idx)
    } else {
      phi_pair <- encode_candidate_pair(Xl, Xr, pl, pr, resp_p, resp_c, stage_idx, matchup_idx)
      phi_l <- phi_pair$phi_left
      phi_r <- phi_pair$phi_right
      u_l <- neural_linear_head(phi_l, params$W_out, params$b_out)
      u_r <- neural_linear_head(phi_r, params$W_out, params$b_out)
      logits <- u_l - u_r
      if (isTRUE(use_cross_term)) {
        logits <- neural_apply_cross_term(logits, phi_l, phi_r,
                                          params$M_cross, params$W_cross_out,
                                          out_dim = ai(params$W_out$shape[[2]]))
      }
    }
    if (return_logits) {
      return(logits)
    }
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

  TransformerPredict_single <- function(params, X_new, party_new,
                                        resp_party_new = NULL, resp_cov_new = NULL,
                                        return_logits = FALSE) {
    Xb <- strenv$jnp$array(to_index_matrix(X_new))$astype(strenv$jnp$int32)
    pb <- strenv$jnp$array(as.integer(party_new))$astype(strenv$jnp$int32)
    if (is.null(resp_party_new)) {
      resp_party_new <- rep(0L, nrow(X_new))
    }
    resp_p <- strenv$jnp$array(as.integer(resp_party_new))$astype(strenv$jnp$int32)
    if (is.null(resp_cov_new)) {
      resp_cov_new <- if (!is.null(resp_cov_mean)) {
        matrix(rep(resp_cov_mean, each = nrow(X_new)), nrow = nrow(X_new))
      } else {
        matrix(0, nrow = nrow(X_new), ncol = 0L)
      }
    }
    resp_c <- strenv$jnp$array(as.matrix(resp_cov_new))$astype(ddtype_)

    model_info_local <- list(
      model_dims = ModelDims,
      cand_party_to_resp_idx = cand_party_to_resp_idx_jnp
    )

    embed_candidate <- function(X_idx, party_idx, resp_p) {
      neural_build_candidate_tokens_hard(X_idx, party_idx,
                                         model_info = model_info_local,
                                         resp_party_idx = resp_p,
                                         params = params)
    }

    run_transformer <- function(tokens) {
      neural_run_transformer(tokens,
                             model_info = list(model_depth = ModelDepth,
                                               model_dims = ModelDims,
                                               n_heads = TransformerHeads,
                                               head_dim = head_dim),
                             params = params)
    }

    ctx_tokens <- neural_build_context_tokens_batch(model_info = model_info_local,
                                                    resp_party_idx = resp_p,
                                                    resp_cov = resp_c,
                                                    params = params)
    choice_tok <- neural_build_choice_token(model_info_local, params)
    choice_tok <- choice_tok * strenv$jnp$ones(list(Xb$shape[[1]], 1L, 1L))
    cand_tokens <- embed_candidate(Xb, pb, resp_p)
    token_parts <- list(choice_tok)
    if (!is.null(ctx_tokens)) {
      token_parts <- c(token_parts, list(ctx_tokens))
    }
    token_parts <- c(token_parts, list(cand_tokens))
    tokens <- strenv$jnp$concatenate(token_parts, axis = 1L)
    tokens <- run_transformer(tokens)
    choice_out <- strenv$jnp$take(tokens, strenv$jnp$arange(1L), axis = 1L)
    choice_out <- strenv$jnp$squeeze(choice_out, axis = 1L)
    logits <- neural_linear_head(choice_out, params$W_out, params$b_out)

    if (return_logits) {
      return(logits)
    }
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

  coerce_party_idx_base <- function(party_vec, n_rows, levels, n_levels) {
    if (is.null(party_vec)) {
      return(rep(0L, n_rows))
    }
    if (is.numeric(party_vec)) {
      idx <- as.integer(party_vec)
      if (any(idx >= n_levels)) {
        idx <- idx - 1L
      }
    } else {
      idx <- match(as.character(party_vec), levels) - 1L
    }
    idx[is.na(idx)] <- 0L
    idx
  }
  coerce_party_idx <- function(party_vec, n_rows) {
    coerce_party_idx_base(party_vec, n_rows, party_levels, n_party_levels)
  }
  coerce_resp_party_idx <- function(party_vec, n_rows) {
    coerce_party_idx_base(party_vec, n_rows, resp_party_levels, n_resp_party_levels)
  }

  to_r_array <- function(x) {
    if (is.null(x) || is.numeric(x)) {
      return(x)
    }
    tryCatch(reticulate::py_to_r(strenv$np$array(x)),
             error = function(e) {
               tryCatch(reticulate::py_to_r(x), error = function(e2) x)
             })
  }

  coerce_prediction_output <- function(pred) {
    if (likelihood == "bernoulli") {
      return(as.numeric(to_r_array(pred)))
    }
    if (likelihood == "categorical") {
      return(as.matrix(to_r_array(pred)))
    }
    if (likelihood == "normal") {
      return(list(
        mu = as.numeric(to_r_array(pred$mu)),
        sigma = as.numeric(to_r_array(pred$sigma))
      ))
    }
    pred
  }

  my_model <- function(...) {
    args <- list(...)
    if (pairwise_mode) {
      X_left_new <- args$X_left_new
      X_right_new <- args$X_right_new
      party_left_new <- args$party_left_new
      party_right_new <- args$party_right_new
      resp_party_new <- args$resp_party_new
      resp_cov_new <- args$resp_cov_new

      if (is.null(X_left_new) || is.null(X_right_new)) {
        if (length(args) < 2L) {
          stop("pairwise my_model requires X_left_new and X_right_new.", call. = FALSE)
        }
        if (is.null(X_left_new)) X_left_new <- args[[1]]
        if (is.null(X_right_new)) X_right_new <- args[[2]]
        if (length(args) >= 3L && is.null(party_left_new)) party_left_new <- args[[3]]
        if (length(args) >= 4L && is.null(party_right_new)) party_right_new <- args[[4]]
        if (length(args) >= 5L && is.null(resp_party_new)) resp_party_new <- args[[5]]
        if (length(args) >= 6L && is.null(resp_cov_new)) resp_cov_new <- args[[6]]
      }

      party_left_new <- coerce_party_idx(party_left_new, nrow(X_left_new))
      party_right_new <- coerce_party_idx(party_right_new, nrow(X_right_new))
      resp_party_new <- coerce_resp_party_idx(resp_party_new, nrow(X_left_new))
      pred <- TransformerPredict_pair(ParamsMean, X_left_new, X_right_new,
                                      party_left_new, party_right_new,
                                      resp_party_new, resp_cov_new)
      return(coerce_prediction_output(pred))
    }

    X_new <- args$X_new
    party_new <- args$party_new
    resp_party_new <- args$resp_party_new
    resp_cov_new <- args$resp_cov_new

    if (is.null(X_new)) {
      if (!is.null(args$X_left_new)) {
        X_new <- args$X_left_new
        if (is.null(party_new) && !is.null(args$party_left_new)) {
          party_new <- args$party_left_new
        }
      } else if (length(args) >= 1L) {
        X_new <- args[[1]]
        if (length(args) >= 2L && is.null(party_new)) party_new <- args[[2]]
        if (length(args) >= 3L && is.null(resp_party_new)) resp_party_new <- args[[3]]
        if (length(args) >= 4L && is.null(resp_cov_new)) resp_cov_new <- args[[4]]
      }
    }

    if (is.null(X_new)) {
      stop("my_model requires X_new for single-candidate predictions.", call. = FALSE)
    }
    party_new <- coerce_party_idx(party_new, nrow(X_new))
    resp_party_new <- coerce_resp_party_idx(resp_party_new, nrow(X_new))
    pred <- TransformerPredict_single(ParamsMean, X_new, party_new,
                                      resp_party_new, resp_cov_new)
    coerce_prediction_output(pred)
  }

  # Neural parameter vector and diagonal posterior covariance
  param_schema <- neural_build_param_schema(
    params = ParamsMean,
    n_factors = length(factor_levels),
    model_depth = ModelDepth
  )
  param_names <- param_schema$param_names
  param_shapes <- param_schema$param_shapes
  param_sizes <- param_schema$param_sizes
  param_offsets <- param_schema$param_offsets
  param_total <- as.integer(param_schema$n_params)

  theta_mean <- neural_flatten_params(ParamsMean, param_schema, dtype = ddtype_)
  theta_mean_num <- as.numeric(strenv$np$array(theta_mean))

  var_parts <- lapply(seq_along(param_names), function(i_) {
    name <- param_names[[i_]]
    draws <- get_param_draws(name)
    if (is.null(draws)) {
      return(strenv$jnp$zeros(list(ai(param_sizes[[i_]])), dtype = ddtype_))
    }
    strenv$jnp$ravel(strenv$jnp$var(draws, 0L:1L))
  })
  param_var_vec <- if (length(var_parts) == 0L) {
    strenv$jnp$array(numeric(0), dtype = ddtype_)
  } else {
    strenv$jnp$concatenate(var_parts, axis = 0L)
  }
  param_var <- as.numeric(strenv$np$array(param_var_vec))
  if (length(param_var) < param_total) {
    param_var <- c(param_var, rep(0, param_total - length(param_var)))
  }
  if (length(param_var) > param_total) {
    param_var <- param_var[seq_len(param_total)]
  }
  if (uncertainty_scope == "output") {
    keep <- param_names %in% c("W_out", "b_out", "sigma", "W_cross_out")
    mask <- unlist(mapply(function(keep_i, size_i) rep(keep_i, size_i),
                          keep, param_sizes))
    if (length(mask) == length(param_var)) {
      param_var <- param_var * as.numeric(mask)
    }
  }
  vcov_OutcomeModel <- c(0, param_var)
  vcov_OutcomeModel_by_k <- NULL

  EST_INTERCEPT_tf <- strenv$jnp$array(matrix(0, nrow = 1L, ncol = 1L), dtype = strenv$dtj)
  EST_COEFFICIENTS_tf <- strenv$jnp$reshape(theta_mean, list(-1L, 1L))
  my_mean <- numeric(0)
  my_mean_full <- NULL
  if (exists("K", inherits = TRUE) && is.numeric(K) && K > 1) {
    base_vec <- c(0, theta_mean_num)
    my_mean_full <- matrix(rep(base_vec, K), ncol = K)
    vcov_OutcomeModel_by_k <- replicate(K, vcov_OutcomeModel, simplify = FALSE)
  }

  resp_cov_mean_jnp <- if (!is.null(resp_cov_mean)) {
    strenv$jnp$array(as.numeric(resp_cov_mean))$astype(ddtype_)
  } else {
    NULL
  }
  party_index_map <- NULL
  if (exists("GroupsPool", inherits = TRUE) && length(GroupsPool) > 0) {
    party_index_map <- setNames(vapply(GroupsPool, function(grp) {
      idx <- match(as.character(grp), party_levels) - 1L
      if (is.na(idx)) 0L else idx
    }, integer(1)), GroupsPool)
  }
  resp_party_index_map <- NULL
  if (exists("GroupsPool", inherits = TRUE) && length(GroupsPool) > 0) {
    resp_party_index_map <- setNames(vapply(GroupsPool, function(grp) {
      idx <- match(as.character(grp), resp_party_levels) - 1L
      if (is.na(idx)) 0L else idx
    }, integer(1)), GroupsPool)
  }

  # fit_metrics is computed above via cross-fitting (unless eval is disabled).

  if (!is.null(svi_loss_curve) && length(svi_loss_curve) > 0L) {
    finite_losses <- svi_loss_curve[is.finite(svi_loss_curve)]
    svi_summary <- list(
      svi_elbo = svi_loss_curve,
      svi_elbo_final = if (length(finite_losses)) tail(finite_losses, 1) else NA_real_
    )
    if (is.null(fit_metrics)) {
      fit_metrics <- svi_summary
    } else {
      fit_metrics <- c(fit_metrics, svi_summary)
    }
  }

  neural_model_info <- list(
    params = ParamsMean,
    param_names = param_names,
    param_shapes = param_shapes,
    param_sizes = param_sizes,
    param_offsets = param_offsets,
    n_params = ai(param_total),
    uncertainty_scope = uncertainty_scope,
    factor_levels = factor_levels,
    factor_index_list = factor_index_list,
    implicit = isTRUE(holdout_indicator == 1L),
    pairwise_mode = pairwise_mode,
    n_factors = ai(length(factor_levels)),
    n_candidate_tokens = n_candidate_tokens,
    party_levels = party_levels,
    n_party_levels = ai(n_party_levels),
    n_matchup_levels = ai(n_matchup_levels),
    resp_party_levels = resp_party_levels,
    party_index_map = party_index_map,
    resp_party_index_map = resp_party_index_map,
    cand_party_to_resp_idx = cand_party_to_resp_idx_jnp,
    resp_cov_mean = resp_cov_mean_jnp,
    n_resp_covariates = n_resp_covariates,
    has_stage_token = !is.null(ParamsMean$E_stage),
    has_matchup_token = !is.null(ParamsMean$E_matchup),
    has_resp_party_token = !is.null(ParamsMean$E_resp_party),
    has_rel_token = !is.null(ParamsMean$E_rel),
    has_feature_id_embedding = !is.null(ParamsMean$E_feature_id),
    has_segment_embedding = !is.null(ParamsMean$E_segment),
    has_sep_token = !is.null(ParamsMean$E_sep),
    has_stage_head = !is.null(ParamsMean$W_stage),
    has_ctx_head = !is.null(ParamsMean$W_ctx),
    has_choice_token = !is.null(ParamsMean$E_choice),
    cross_candidate_encoder = isTRUE(use_cross_term),
    cross_candidate_encoder_mode = cross_candidate_encoder_mode,
    has_cross_encoder = isTRUE(use_cross_encoder),
    has_cross_term = !is.null(ParamsMean$M_cross),
    choice_token_index = 0L,
    likelihood = likelihood,
    fit_metrics = fit_metrics,
    svi_loss_curve = svi_loss_curve,
    stage_diagnostics = stage_diagnostics,
    model_dims = ModelDims,
    model_depth = ModelDepth,
    n_heads = TransformerHeads,
    head_dim = head_dim
  )

  message(sprintf("Bayesian Transformer complete. Pairwise=%s, Heads=%d, Depth=%d, Hidden=%d; likelihood=%s.",
                  pairwise_mode, TransformerHeads, ModelDepth, MD_int, likelihood))
}
