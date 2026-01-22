neural_get_party_index <- function(model_info, party_label = NULL){
  if (is.null(model_info) || is.null(model_info$party_levels)) {
    return(ai(0L))
  }
  if (!is.null(model_info$party_index_map) && !is.null(party_label)) {
    key <- as.character(party_label)
    if (key %in% names(model_info$party_index_map)) {
      return(ai(model_info$party_index_map[[key]]))
    }
  }
  if (is.null(party_label)) {
    return(ai(0L))
  }
  idx <- match(as.character(party_label), model_info$party_levels) - 1L
  if (is.na(idx)) ai(0L) else ai(idx)
}

neural_get_resp_party_index <- function(model_info, party_label = NULL){
  if (is.null(model_info) || is.null(model_info$resp_party_levels)) {
    return(ai(0L))
  }
  if (!is.null(model_info$resp_party_index_map) && !is.null(party_label)) {
    key <- as.character(party_label)
    if (key %in% names(model_info$resp_party_index_map)) {
      return(ai(model_info$resp_party_index_map[[key]]))
    }
  }
  if (is.null(party_label)) {
    return(ai(0L))
  }
  idx <- match(as.character(party_label), model_info$resp_party_levels) - 1L
  if (is.na(idx)) ai(0L) else ai(idx)
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
    return(strenv$jnp$reshape(prob, list(1L, 1L)))
  }
  if (likelihood == "categorical") {
    probs <- strenv$jax$nn$softmax(logits, axis = -1L)
    prob <- strenv$jnp$take(probs, 1L, axis = 1L)
    return(strenv$jnp$reshape(prob, list(1L, 1L)))
  }
  mu <- strenv$jnp$squeeze(logits, axis = 1L)
  strenv$jnp$reshape(mu, list(1L, 1L))
}

neural_params_from_theta <- function(theta_vec, model_info){
  if (is.null(model_info$param_names) ||
      is.null(model_info$param_offsets) ||
      is.null(model_info$param_sizes)) {
    return(model_info$params)
  }
  theta_vec <- strenv$jnp$reshape(theta_vec, list(-1L))
  params <- list()
  param_names <- model_info$param_names
  param_offsets <- model_info$param_offsets
  param_sizes <- model_info$param_sizes
  param_shapes <- model_info$param_shapes
  for (i_ in seq_along(param_names)) {
    start <- as.integer(param_offsets[[i_]])
    size <- as.integer(param_sizes[[i_]])
    idx <- strenv$jnp$arange(ai(start), ai(start + size))
    slice <- strenv$jnp$take(theta_vec, idx, axis = 0L)
    shape_use <- param_shapes[[i_]]
    if (length(shape_use) == 0L) {
      shape_use <- c(1L)
    }
    params[[param_names[[i_]]]] <- strenv$jnp$reshape(slice, as.integer(shape_use))
  }
  params
}

neural_cross_encoder_mode <- function(model_info) {
  mode <- NULL
  if (!is.null(model_info$cross_candidate_encoder_mode)) {
    mode <- tolower(as.character(model_info$cross_candidate_encoder_mode))
  }
  if (length(mode) != 1L || is.na(mode) || !nzchar(mode)) {
    if (isTRUE(model_info$cross_candidate_encoder)) {
      return("term")
    }
    return("none")
  }
  if (mode %in% c("none", "term", "full")) {
    return(mode)
  }
  if (mode %in% c("true", "false")) {
    return(ifelse(mode == "true", "term", "none"))
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
    if (isTRUE(model_info$implicit)) {
      p_holdout <- strenv$jnp$clip(strenv$jnp$array(1., strenv$dtj) - strenv$jnp$sum(p_sub),
                                   0., 1.)
      p_full <- strenv$jnp$concatenate(
        list(p_sub, strenv$jnp$reshape(p_holdout, list(1L))),
        axis = 0L
      )
    } else {
      p_full <- p_sub
    }
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
  if (!is.null(params$E_feature_id)) {
    feature_tok <- strenv$jnp$reshape(params$E_feature_id,
                                      list(ai(model_info$n_factors), model_info$model_dims))
    tokens <- tokens + feature_tok
  }
  party_vec <- strenv$jnp$take(params$E_party, strenv$jnp$array(ai(party_idx)), axis = 0L)
  if (isTRUE(use_role) && !is.null(params$E_role)) {
    n_roles <- ai(params$E_role$shape[[1]])
    role_use <- if (ai(role_id) >= n_roles) 0L else ai(role_id)
    role_vec <- strenv$jnp$take(params$E_role, strenv$jnp$array(ai(role_use)), axis = 0L)
    role_vec <- strenv$jnp$reshape(role_vec, list(1L, model_info$model_dims))
    tokens <- tokens + role_vec
    party_vec <- party_vec + strenv$jnp$reshape(role_vec, list(model_info$model_dims))
  }
  party_tok <- strenv$jnp$reshape(party_vec, list(1L, model_info$model_dims))
  tokens <- strenv$jnp$concatenate(list(tokens, party_tok), axis = 0L)
  if (!is.null(params$E_rel)) {
    if (is.null(model_info$cand_party_to_resp_idx) || is.null(resp_party_idx)) {
      rel_idx <- ai(2L)
    } else {
      cand_map <- strenv$jnp$atleast_1d(model_info$cand_party_to_resp_idx)
      cand_resp_idx <- strenv$jnp$take(cand_map,
                                       strenv$jnp$array(ai(party_idx)), axis = 0L)
      is_known <- cand_resp_idx >= 0L
      is_match <- strenv$jnp$equal(cand_resp_idx, ai(resp_party_idx))
      rel_idx <- strenv$jnp$where(is_match, ai(0L),
                                  strenv$jnp$where(is_known, ai(1L), ai(2L)))
    }
    rel_idx <- strenv$jnp$array(rel_idx)
    rel_idx <- strenv$jnp$astype(rel_idx, strenv$jnp$int32)
    rel_vec <- strenv$jnp$take(params$E_rel, rel_idx, axis = 0L)
    rel_tok <- strenv$jnp$reshape(rel_vec, list(1L, model_info$model_dims))
    tokens <- strenv$jnp$concatenate(list(tokens, rel_tok), axis = 0L)
  }
  strenv$jnp$reshape(tokens, list(1L, tokens$shape[[1]], model_info$model_dims))
}

neural_build_context_tokens <- function(model_info,
                                        resp_party_idx = NULL,
                                        stage_idx = NULL,
                                        matchup_idx = NULL,
                                        resp_cov_vec = NULL,
                                        params = NULL){
  if (is.null(params)) {
    params <- model_info$params
  }
  token_list <- list()

  if (!is.null(params$E_stage) && !is.null(stage_idx)) {
    stage_use <- ai(stage_idx)
    resp_use <- if (is.null(resp_party_idx)) 0L else ai(resp_party_idx)
    stage_vec <- params$E_stage[resp_use, stage_use]
    stage_tok <- strenv$jnp$reshape(stage_vec, list(1L, 1L, model_info$model_dims))
    token_list[[length(token_list) + 1L]] <- stage_tok
  }
  if (!is.null(params$E_resp_party)) {
    resp_use <- if (is.null(resp_party_idx)) 0L else ai(resp_party_idx)
    resp_vec <- strenv$jnp$take(params$E_resp_party, strenv$jnp$array(resp_use), axis = 0L)
    resp_tok <- strenv$jnp$reshape(resp_vec, list(1L, 1L, model_info$model_dims))
    token_list[[length(token_list) + 1L]] <- resp_tok
  }
  if (!is.null(params$E_matchup) && !is.null(matchup_idx)) {
    matchup_vec <- strenv$jnp$take(params$E_matchup, strenv$jnp$array(matchup_idx), axis = 0L)
    matchup_tok <- strenv$jnp$reshape(matchup_vec, list(1L, 1L, model_info$model_dims))
    token_list[[length(token_list) + 1L]] <- matchup_tok
  }
  if (!is.null(params$W_resp_x) &&
      !is.null(model_info$resp_cov_mean) &&
      ai(model_info$n_resp_covariates) > 0L) {
    if (is.null(resp_cov_vec)) {
      resp_cov_vec <- model_info$resp_cov_mean
    }
    resp_cov_vec <- strenv$jnp$reshape(resp_cov_vec, list(-1L))
    cov_vec <- strenv$jnp$einsum("c,cm->m", resp_cov_vec, params$W_resp_x)
    cov_tok <- strenv$jnp$reshape(cov_vec, list(1L, 1L, model_info$model_dims))
    token_list[[length(token_list) + 1L]] <- cov_tok
  }
  if (length(token_list) == 0L) {
    return(NULL)
  }
  strenv$jnp$concatenate(token_list, axis = 1L)
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

    h1 <- tokens + attn_out
    h1_norm <- neural_rms_norm(h1, RMS_ff, model_info$model_dims)
    ff_pre <- strenv$jnp$einsum("ntm,mf->ntf", h1_norm, Wff1)
    ff_act <- strenv$jax$nn$swish(ff_pre)
    ff_out <- strenv$jnp$einsum("ntf,fm->ntm", ff_act, Wff2)
    tokens <- h1 + ff_out
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
  logits <- strenv$jnp$einsum("nm,mo->no", phi, params$W_out)
  if (!is.null(params$b_out)) {
    logits <- logits + params$b_out
  }
  logits
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
  stage_idx <- ifelse(ai(party_left_idx) == ai(party_right_idx), 1L, 0L)
  matchup_idx <- NULL
  if (!is.null(params$E_matchup)) {
    matchup_idx <- neural_matchup_index(party_left_idx, party_right_idx, model_info)
  }
  if (isTRUE(use_cross_encoder)) {
    add_segment_embedding <- function(tokens, segment_idx) {
      if (is.null(params$E_segment)) {
        return(tokens)
      }
      seg_vec <- strenv$jnp$take(params$E_segment, ai(segment_idx), axis = 0L)
      seg_tok <- strenv$jnp$reshape(seg_vec, list(1L, 1L, ai(model_info$model_dims)))
      tokens + seg_tok
    }
    build_sep_token <- function() {
      sep_vec <- if (!is.null(params$E_sep)) {
        params$E_sep
      } else {
        strenv$jnp$zeros(list(ai(model_info$model_dims)), dtype = strenv$dtj)
      }
      strenv$jnp$reshape(sep_vec, list(1L, 1L, ai(model_info$model_dims)))
    }
    choice_tok <- neural_build_choice_token(model_info, params)
    ctx_tokens <- neural_build_context_tokens(model_info,
                                              resp_party_idx = resp_party_idx,
                                              stage_idx = stage_idx,
                                              matchup_idx = matchup_idx,
                                              resp_cov_vec = resp_cov_vec,
                                              params = params)
    left_tokens <- add_segment_embedding(
      neural_build_candidate_tokens_soft(pi_left, party_left_idx, 0L, model_info, params,
                                         resp_party_idx = resp_party_idx),
      0L
    )
    right_tokens <- add_segment_embedding(
      neural_build_candidate_tokens_soft(pi_right, party_right_idx, 1L, model_info, params,
                                         resp_party_idx = resp_party_idx),
      1L
    )
    sep_tok <- build_sep_token()
    token_parts <- list(choice_tok)
    if (!is.null(ctx_tokens)) {
      token_parts <- c(token_parts, list(ctx_tokens))
    }
    token_parts <- c(token_parts, list(sep_tok, left_tokens, sep_tok, right_tokens))
    tokens <- strenv$jnp$concatenate(token_parts, axis = 1L)
    tokens <- neural_run_transformer(tokens, model_info, params)
    cls_out <- strenv$jnp$take(tokens, strenv$jnp$arange(1L), axis = 1L)
    cls_out <- strenv$jnp$squeeze(cls_out, axis = 1L)
    logits <- strenv$jnp$einsum("nm,mo->no", cls_out, params$W_out)
    if (!is.null(params$b_out)) {
      logits <- logits + params$b_out
    }
  } else {
    phi_left <- neural_encode_candidate_soft(pi_left, party_left_idx, model_info,
                                             resp_party_idx = resp_party_idx,
                                             stage_idx = stage_idx,
                                             matchup_idx = matchup_idx,
                                             resp_cov_vec = resp_cov_vec,
                                             params = params)
    phi_right <- neural_encode_candidate_soft(pi_right, party_right_idx, model_info,
                                              resp_party_idx = resp_party_idx,
                                              stage_idx = stage_idx,
                                              matchup_idx = matchup_idx,
                                              resp_cov_vec = resp_cov_vec,
                                              params = params)
    u_left <- strenv$jnp$einsum("nm,mo->no", phi_left, params$W_out)
    if (!is.null(params$b_out)) {
      u_left <- u_left + params$b_out
    }
    u_right <- strenv$jnp$einsum("nm,mo->no", phi_right, params$W_out)
    if (!is.null(params$b_out)) {
      u_right <- u_right + params$b_out
    }
    logits <- u_left - u_right
    if (isTRUE(use_cross_term) && !is.null(params$M_cross)) {
      cross_term <- strenv$jnp$einsum("nm,mp,np->n", phi_left, params$M_cross, phi_right)
      cross_term <- strenv$jnp$reshape(cross_term, list(-1L, 1L))
      cross_out <- if (!is.null(params$W_cross_out)) {
        strenv$jnp$reshape(params$W_cross_out, list(1L, -1L))
      } else {
        strenv$jnp$zeros(list(1L, ai(params$W_out$shape[[2]])), dtype = strenv$dtj)
      }
      logits <- logits + cross_term * cross_out
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

getQStar_single <- function(pi_star_ast, pi_star_dag,
                            EST_INTERCEPT_tf_ast, EST_COEFFICIENTS_tf_ast,
                            EST_INTERCEPT_tf_dag, EST_COEFFICIENTS_tf_dag){
  if (outcome_model_type == "neural" &&
      exists("neural_model_info_ast_jnp", inherits = TRUE)) {
    model_ast <- get("neural_model_info_ast_jnp", inherits = TRUE)
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
    return(strenv$jnp$concatenate(list(Qhat, Qhat, Qhat), 0L))
  }
  # note: here, dag ignored 
  pi_star_ast <- strenv$jnp$reshape(pi_star_ast, list(-1L, 1L))
  pi_star_dag <- strenv$jnp$reshape(pi_star_dag, list(-1L, 1L))

  # coef info
  main_coef <- strenv$jnp$expand_dims( strenv$jnp$take(EST_COEFFICIENTS_tf_ast, 
                                                       indices = main_indices_i0, 
                                                       axis = 0L),1L)
  main_term <- strenv$jnp$reshape(
    strenv$jnp$sum(strenv$jnp$reshape(main_coef, list(-1L)) *
                     strenv$jnp$reshape(pi_star_ast, list(-1L))),
    list(1L, 1L)
  )
  if(!is.null(inter_indices_i0)){ 
    inter_coef <- strenv$jnp$expand_dims( strenv$jnp$take(EST_COEFFICIENTS_tf_ast,
                                                          indices = inter_indices_i0, 
                                                          axis = 0L), 1L)
  
    # get interaction info
    pi_dp <- strenv$jnp$take(pi_star_ast, 
                             n2int( ai(interaction_info$dl_index_adj)-1L), axis=0L)
    pi_dpp <- strenv$jnp$take(pi_star_ast, 
                              n2int( ai(interaction_info$dplp_index_adj)-1L), axis=0L)
    inter_term <- strenv$jnp$reshape(
      strenv$jnp$sum(strenv$jnp$reshape(inter_coef, list(-1L)) *
                       strenv$jnp$reshape(pi_dp * pi_dpp, list(-1L))),
      list(1L, 1L)
    )
    Qhat <-  glm_outcome_transform( EST_INTERCEPT_tf_ast +
                                      main_term +
                                      inter_term )
  }

  if(is.null(inter_indices_i0)){ 
    Qhat <-  glm_outcome_transform( EST_INTERCEPT_tf_ast +
                          main_term )
  }
  
  if( length(Qhat$shape) == 3L ) {
    Qhat <- Qhat$squeeze(2L)
  }
  return( strenv$jnp$concatenate( list(Qhat, 
                                       Qhat, 
                                       Qhat), 0L)  ) # to keep sizes consistent with diff case 
}

getQStar_diff_BASE <- function(pi_star_ast, pi_star_dag,
                               EST_INTERCEPT_tf_ast, EST_COEFFICIENTS_tf_ast,
                               EST_INTERCEPT_tf_dag, EST_COEFFICIENTS_tf_dag){

  pi_star_ast <- strenv$jnp$reshape(pi_star_ast, list(-1L, 1L))
  pi_star_dag <- strenv$jnp$reshape(pi_star_dag, list(-1L, 1L))

  if (outcome_model_type == "neural" &&
      exists("neural_model_info_ast_jnp", inherits = TRUE)) {
    model_ast <- get("neural_model_info_ast_jnp", inherits = TRUE)
    model_dag <- if (exists("neural_model_info_dag_jnp", inherits = TRUE)) {
      get("neural_model_info_dag_jnp", inherits = TRUE)
    } else {
      model_ast
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

    if (!Q_DISAGGREGATE) {
      resp_idx_dag <- resp_idx_ast
      params_dag <- params_ast
    }

    Qhat_ast_among_ast <- neural_predict_pair_soft(
      pi_star_ast, pi_star_dag,
      party_idx_ast, party_idx_dag,
      resp_idx_ast, model_ast,
      params = params_ast
    )

    if (!Q_DISAGGREGATE) {
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
    return(strenv$jnp$concatenate(list(Qhat_population,
                                       Qhat_ast_among_ast,
                                       Qhat_ast_among_dag), 0L))
  }

  # coef
  main_coef_ast <- strenv$jnp$expand_dims(strenv$jnp$take(EST_COEFFICIENTS_tf_ast, 
                                   indices = main_indices_i0, axis = 0L),1L)
  DELTA_pi_star <- strenv$jnp$reshape((pi_star_ast - pi_star_dag), list(-1L))
  main_term_ast <- strenv$jnp$reshape(
    strenv$jnp$sum(strenv$jnp$reshape(main_coef_ast, list(-1L)) * DELTA_pi_star),
    list(1L, 1L)
  )
  
  if(!is.null(inter_indices_i0)){ 
    inter_coef_ast <- strenv$jnp$expand_dims(strenv$jnp$take(EST_COEFFICIENTS_tf_ast, 
                                                             indices = inter_indices_i0, 
                                                             axis = 0L),1L)
  
    # get interaction info
    pi_ast_dp <- strenv$jnp$take(pi_star_ast, n2int( ai(interaction_info$dl_index_adj)-1L), axis=0L)
    pi_ast_dpp <- strenv$jnp$take(pi_star_ast, n2int( ai(interaction_info$dplp_index_adj)-1L), axis=0L)
  
    pi_dag_dp <- strenv$jnp$take(pi_star_dag, n2int( ai(interaction_info$dl_index_adj)-1L), axis=0L)
    pi_dag_dpp <- strenv$jnp$take(pi_star_dag, n2int( ai(interaction_info$dplp_index_adj)-1L), axis=0L)
    DELTA_pi_star_prod <- strenv$jnp$reshape(pi_ast_dp * pi_ast_dpp - pi_dag_dp * pi_dag_dpp,
                                             list(-1L))
    inter_term_ast <- strenv$jnp$reshape(
      strenv$jnp$sum(strenv$jnp$reshape(inter_coef_ast, list(-1L)) * DELTA_pi_star_prod),
      list(1L, 1L)
    )
    
    Qhat_ast_among_ast <- glm_outcome_transform( 
              EST_INTERCEPT_tf_ast + 
              main_term_ast +
              inter_term_ast
            )
  }

  if(is.null(inter_indices_i0)){ 
    Qhat_ast_among_ast <- glm_outcome_transform( 
      EST_INTERCEPT_tf_ast +  main_term_ast )
  }

  if( !Q_DISAGGREGATE ){ Qhat_population <- Qhat_ast_among_dag <- Qhat_ast_among_ast }
  if( Q_DISAGGREGATE ){ # run if DisaggreateQ
    main_coef_dag <- strenv$jnp$expand_dims( strenv$jnp$take(EST_COEFFICIENTS_tf_dag, 
                                 indices = main_indices_i0, axis=0L), 1L)
    main_term_dag <- strenv$jnp$reshape(
      strenv$jnp$sum(strenv$jnp$reshape(main_coef_dag, list(-1L)) * DELTA_pi_star),
      list(1L, 1L)
    )
    if(!is.null(inter_indices_i0)){ 
      inter_coef_dag <- strenv$jnp$expand_dims( strenv$jnp$take(EST_COEFFICIENTS_tf_dag, 
                                                                indices = inter_indices_i0, axis=0L), 1L)
      inter_term_dag <- strenv$jnp$reshape(
        strenv$jnp$sum(strenv$jnp$reshape(inter_coef_dag, list(-1L)) * DELTA_pi_star_prod),
        list(1L, 1L)
      )
      Qhat_ast_among_dag <- glm_outcome_transform( 
                EST_INTERCEPT_tf_dag + 
                main_term_dag +
                inter_term_dag )
    }
    if(is.null(inter_indices_i0)){ 
      Qhat_ast_among_dag <- glm_outcome_transform( 
        EST_INTERCEPT_tf_dag +  main_term_dag )
    }
  
    # Pr( Ast | Ast Voter) * Pr(Ast Voters) +  Pr( Ast | Dag Voter) * Pr(Dag Voters)
    Qhat_population <- Qhat_ast_among_ast * strenv$jnp$array(strenv$AstProp) +  
                                Qhat_ast_among_dag * strenv$jnp$array(strenv$DagProp)
  }
  return( strenv$jnp$concatenate( list(Qhat_population, 
                                       Qhat_ast_among_ast, 
                                       Qhat_ast_among_dag), 0L)  )
}

FullGetQStar_ <- function(a_i_ast,                                #1 
                          a_i_dag,                                #2 
                          INTERCEPT_ast_, COEFFICIENTS_ast_,      #3,4       
                          INTERCEPT_dag_, COEFFICIENTS_dag_,      #5,6 
                          INTERCEPT_ast0_, COEFFICIENTS_ast0_,    #7,8
                          INTERCEPT_dag0_, COEFFICIENTS_dag0_,    #9,10
                          P_VEC_FULL_ast_, P_VEC_FULL_dag_,       #11,12
                          SLATE_VEC_ast_, SLATE_VEC_dag_,         #13,14
                          LAMBDA_,                                #15
                          Q_SIGN,                                 #16 
                          SEED_IN_LOOP                            #17
){
  
  # Map logits -> simplex (respecting ParameterizationType)
  pi_star_full_i_ast <- strenv$getPrettyPi_diff( pi_star_i_ast<-strenv$a2Simplex_diff_use(a_i_ast), 
                                                 strenv$ParameterizationType,
                                                 strenv$d_locator_use,       
                                                 strenv$main_comp_mat,   
                                                 strenv$shadow_comp_mat  )
  pi_star_full_i_dag <- strenv$getPrettyPi_diff( pi_star_i_dag<-strenv$a2Simplex_diff_use(a_i_dag),
                                                 strenv$ParameterizationType,
                                                 strenv$d_locator_use,       
                                                 strenv$main_comp_mat,   
                                                 strenv$shadow_comp_mat  )
  
  # Average-case path
  if(!adversarial){
    use_mc_q <- (outcome_model_type == "neural") ||
      ((glm_family != "gaussian") && (nMonte_Qglm > 1L))
    if(use_mc_q){
      n_draws <- if (outcome_model_type == "neural") {
        max(1L, nMonte_Qglm)
      } else {
        nMonte_Qglm
      }
      TSAMP_ast_all <- strenv$jax$vmap(function(s_){
        strenv$getMultinomialSamp(pi_star_i_ast, MNtemp,
                                  s_, strenv$ParameterizationType, strenv$jnp$array(strenv$d_locator_use))
      }, in_axes = list(0L))(strenv$jax$random$split(SEED_IN_LOOP, n_draws))
      SEED_IN_LOOP   <- strenv$jax$random$split(SEED_IN_LOOP)[[1L]]

      TSAMP_dag_all <- strenv$jax$vmap(function(s_){
        strenv$getMultinomialSamp(pi_star_i_dag, MNtemp,
                                  s_, strenv$ParameterizationType, strenv$jnp$array(strenv$d_locator_use))
      }, in_axes = list(0L))(strenv$jax$random$split(SEED_IN_LOOP, n_draws))
      SEED_IN_LOOP   <- strenv$jax$random$split(SEED_IN_LOOP)[[1L]]

      q_vec <- strenv$Vectorized_QMonteIter(
        TSAMP_ast_all,  TSAMP_dag_all,
        INTERCEPT_ast_, COEFFICIENTS_ast_,
        INTERCEPT_dag_, COEFFICIENTS_dag_
      )$mean(0L)
    } else {
      q_vec <- QFXN(pi_star_ast =  pi_star_i_ast,
                    pi_star_dag =  pi_star_i_dag,
                    EST_INTERCEPT_tf_ast = INTERCEPT_ast_,
                    EST_COEFFICIENTS_tf_ast = COEFFICIENTS_ast_,
                    EST_INTERCEPT_tf_dag = INTERCEPT_dag_,
                    EST_COEFFICIENTS_tf_dag = COEFFICIENTS_dag_)
    }
    q_max <- strenv$jnp$take(q_vec, 0L)
    # In non-adversarial mode, we always optimize the "ast" player
    indicator_UseAst <- 1.0
  }
  
  # Adversarial path: institution-aware push-forward (four-quadrant mixture)
  if(adversarial){
    
    if (primary_pushforward == "multi") {
      sample_pool <- function(pi_vec, n_draws, n_pool, seed_in) {
        n_total <- as.integer(n_draws * n_pool)
        # Split into n_total + 1 keys: n_total for sampling, 1 for advancing seed
        all_keys <- strenv$jax$random$split(seed_in, as.integer(n_total + 1L))
        # Last key is seed_next (independent of keys used for sampling)
        seed_next <- strenv$jnp$take(all_keys, -1L, axis = 0L)
        # First n_total keys for sampling
        seeds <- strenv$jnp$take(all_keys, strenv$jnp$arange(n_total), axis = 0L)
        seeds <- strenv$jnp$reshape(seeds, list(n_draws, n_pool, 2L))
        samples <- strenv$jax$vmap(function(seed_row){
          strenv$jax$vmap(function(seed_cell){
            strenv$getMultinomialSamp(pi_vec, MNtemp,
                                      seed_cell, strenv$ParameterizationType, strenv$jnp$array(strenv$d_locator_use))
          }, in_axes = list(0L))(seed_row)
        }, in_axes = list(0L))(seeds)
        list(samples = samples, seed_next = seed_next)
      }

      # Draw policy samples (entrant pools)
      samp_ast <- sample_pool(pi_star_i_ast, nMonte_adversarial, primary_n_entrants, SEED_IN_LOOP)
      TSAMP_ast_all <- samp_ast$samples
      SEED_IN_LOOP <- samp_ast$seed_next

      samp_dag <- sample_pool(pi_star_i_dag, nMonte_adversarial, primary_n_entrants, SEED_IN_LOOP)
      TSAMP_dag_all <- samp_dag$samples
      SEED_IN_LOOP <- samp_dag$seed_next

      # Draw field (slate) samples
      samp_ast_field <- sample_pool(SLATE_VEC_ast_, nMonte_adversarial, primary_n_field, SEED_IN_LOOP)
      TSAMP_ast_PrimaryComp_all <- samp_ast_field$samples
      SEED_IN_LOOP <- samp_ast_field$seed_next

      samp_dag_field <- sample_pool(SLATE_VEC_dag_, nMonte_adversarial, primary_n_field, SEED_IN_LOOP)
      TSAMP_dag_PrimaryComp_all <- samp_dag_field$samples
      SEED_IN_LOOP <- samp_dag_field$seed_next
    } else {
      # Draw policy samples
      TSAMP_ast_all <- strenv$jax$vmap(function(s_){ 
        strenv$getMultinomialSamp(pi_star_i_ast, MNtemp, 
                                  s_, strenv$ParameterizationType, strenv$jnp$array(strenv$d_locator_use))
      }, in_axes = list(0L))(strenv$jax$random$split(SEED_IN_LOOP, nMonte_adversarial))
      SEED_IN_LOOP   <- strenv$jax$random$split(SEED_IN_LOOP)[[1L]]
      
      TSAMP_dag_all <- strenv$jax$vmap(function(s_){ 
        strenv$getMultinomialSamp(pi_star_i_dag, MNtemp, 
                                  s_, strenv$ParameterizationType, strenv$jnp$array(strenv$d_locator_use))
      }, in_axes = list(0L))(strenv$jax$random$split(SEED_IN_LOOP, nMonte_adversarial))
      SEED_IN_LOOP   <- strenv$jax$random$split(SEED_IN_LOOP)[[1L]]
      
      # Draw field (slate) samples
      TSAMP_ast_PrimaryComp_all <- strenv$jax$vmap(function(s_){ 
        strenv$getMultinomialSamp(
                                  #strenv$jax$lax$stop_gradient(pi_star_i_ast), # if using optimized dist
                                  SLATE_VEC_ast_, # if using slate 
                                  MNtemp, 
                                  s_, strenv$ParameterizationType, strenv$jnp$array(strenv$d_locator_use))
      }, in_axes = list(0L))(strenv$jax$random$split(SEED_IN_LOOP, nMonte_adversarial))
      SEED_IN_LOOP   <- strenv$jax$random$split(SEED_IN_LOOP)[[1L]]
      
      TSAMP_dag_PrimaryComp_all <- strenv$jax$vmap(function(s_){ 
        strenv$getMultinomialSamp(
                                  #strenv$jax$lax$stop_gradient(pi_star_i_dag),  # if using optimized dist
                                  SLATE_VEC_dag_,  # if using slate
                                  MNtemp,
                                  s_, strenv$ParameterizationType, strenv$jnp$array(strenv$d_locator_use))
      }, in_axes = list(0L))(strenv$jax$random$split(SEED_IN_LOOP, nMonte_adversarial))
      SEED_IN_LOOP   <- strenv$jax$random$split(SEED_IN_LOOP)[[1L]]
    }
    
    # Evaluate institutional objective (push-forward over nominees)
    QMonteRes <- strenv$Vectorized_QMonteIter_MaxMin(
      TSAMP_ast_all, TSAMP_dag_all,
      TSAMP_ast_PrimaryComp_all, TSAMP_dag_PrimaryComp_all,
      a_i_ast, a_i_dag,
      INTERCEPT_ast_,  COEFFICIENTS_ast_,
      INTERCEPT_dag_,  COEFFICIENTS_dag_,
      INTERCEPT_ast0_, COEFFICIENTS_ast0_,
      INTERCEPT_dag0_, COEFFICIENTS_dag0_,
      P_VEC_FULL_ast_, P_VEC_FULL_dag_,
      LAMBDA_, Q_SIGN, 
      strenv$jax$random$split(SEED_IN_LOOP, TSAMP_ast_PrimaryComp_all$shape[[1]])
    )
    q_max_ast <- QMonteRes$q_ast$mean()
    q_max_dag <- QMonteRes$q_dag$mean()
    
    # Choose which side we’re optimizing in this call
    indicator_UseAst <- (0.5 * ( 1. + Q_SIGN ))
    q_max <- (indicator_UseAst * q_max_ast) + ( (1. - indicator_UseAst)* q_max_dag)
  }
  
  # ---- Regularization (unchanged), applied to the player being updated ----
  if(penalty_type %in% c("L1","L2")){
    PenFxn <- ifelse(penalty_type == "L1", 
                     yes = list(strenv$jnp$abs),
                     no = list(strenv$jnp$square))[[1]]
    var_pen_ast__ <- LAMBDA_ * strenv$jnp$negative(strenv$jnp$sum(PenFxn( pi_star_full_i_ast - P_VEC_FULL_ast_ )  ))
    var_pen_dag__ <- LAMBDA_ * strenv$jnp$negative(strenv$jnp$sum(PenFxn( pi_star_full_i_dag - P_VEC_FULL_dag_ )  ))
  } else if(penalty_type == "LInfinity"){
    var_pen_ast__ <- tapply(1:length(split_vec_full), split_vec_full, function(zer){
      list( strenv$jnp$max(strenv$jnp$take(pi_star_full_i_ast, indices = strenv$jnp$array( ai(zer-1L)),axis = 0L)) )})
    names(var_pen_ast__)<-NULL ; var_pen_ast__ <- strenv$jnp$negative(LAMBDA_*strenv$jnp$sum(strenv$jnp$stack(var_pen_ast__)))
    var_pen_dag__ <- tapply(1:length(split_vec_full), split_vec_full, function(zer){
      list( strenv$jnp$max(strenv$jnp$take(pi_star_full_i_dag, indices = strenv$jnp$array( ai(zer-1L)),axis = 0L)) )})
    names(var_pen_dag__)<-NULL ; var_pen_dag__ <- strenv$jnp$negative(LAMBDA_*strenv$jnp$sum(strenv$jnp$stack(var_pen_dag__)))
  } else {
    # "KL" default (with epsilon clipping to prevent log(0) = -Inf)
    eps <- 1e-8
    var_pen_ast__ <- LAMBDA_*strenv$jnp$negative(strenv$jnp$sum(P_VEC_FULL_ast_ * (strenv$jnp$log(strenv$jnp$clip(P_VEC_FULL_ast_, eps, 1.0)) - strenv$jnp$log(strenv$jnp$clip(pi_star_full_i_ast, eps, 1.0)))))
    var_pen_dag__ <- LAMBDA_*strenv$jnp$negative(strenv$jnp$sum(P_VEC_FULL_dag_ * (strenv$jnp$log(strenv$jnp$clip(P_VEC_FULL_dag_, eps, 1.0)) - strenv$jnp$log(strenv$jnp$clip(pi_star_full_i_dag, eps, 1.0)))))
  }
  
  myMaximize <- 
    q_max + ( (indicator_UseAst * var_pen_ast__) 
       + (1.- indicator_UseAst) * var_pen_dag__ )
  
  return( myMaximize )
}
