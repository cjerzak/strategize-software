cs_crossfit_q_default_control <- function(control = NULL) {
  defaults <- list(
    folds = 3L,
    seed = 123L,
    split_by = "pair_id",
    estimators = c("dr", "ips", "snips", "model"),
    headline = "dr",
    weight_clip = Inf,
    n_policy_draws = 1000L,
    chunk_size = 2000L,
    return_fold_results = TRUE,
    perspective_group = NULL
  )
  if (is.null(control)) {
    control <- list()
  }
  if (!is.list(control)) {
    stop("'crossfit_q_control' must be a list.", call. = FALSE)
  }
  out <- utils::modifyList(defaults, control)
  out$folds <- as.integer(out$folds)
  out$seed <- as.integer(out$seed)
  out$split_by <- match.arg(tolower(as.character(out$split_by)[[1]]),
                            c("pair_id", "respondent_id"))
  out$estimators <- unique(tolower(as.character(out$estimators)))
  out$headline <- tolower(as.character(out$headline)[[1]])
  out$weight_clip <- as.numeric(out$weight_clip)
  out$n_policy_draws <- as.integer(out$n_policy_draws)
  out$chunk_size <- as.integer(out$chunk_size)
  out$return_fold_results <- isTRUE(out$return_fold_results)
  out$perspective_group <- if (is.null(out$perspective_group)) {
    NULL
  } else {
    as.character(out$perspective_group)[[1L]]
  }
  valid_estimators <- c("dr", "ips", "snips", "model")
  bad_estimators <- setdiff(out$estimators, valid_estimators)
  if (length(bad_estimators)) {
    stop(sprintf(
      "'crossfit_q_control$estimators' contains unknown value(s): %s.",
      paste(bad_estimators, collapse = ", ")
    ), call. = FALSE)
  }
  if (!out$headline %in% out$estimators) {
    stop("'crossfit_q_control$headline' must be included in 'estimators'.", call. = FALSE)
  }
  if (is.na(out$folds) || out$folds < 2L) {
    stop("'crossfit_q_control$folds' must be an integer >= 2.", call. = FALSE)
  }
  if (is.na(out$seed)) {
    stop("'crossfit_q_control$seed' must be an integer.", call. = FALSE)
  }
  if (!is.finite(out$weight_clip) && !is.infinite(out$weight_clip)) {
    stop("'crossfit_q_control$weight_clip' must be finite or Inf.", call. = FALSE)
  }
  if (out$weight_clip <= 0) {
    stop("'crossfit_q_control$weight_clip' must be positive.", call. = FALSE)
  }
  if (is.na(out$n_policy_draws) || out$n_policy_draws < 1L) {
    stop("'crossfit_q_control$n_policy_draws' must be an integer >= 1.", call. = FALSE)
  }
  if (is.na(out$chunk_size) || out$chunk_size < 1L) {
    stop("'crossfit_q_control$chunk_size' must be an integer >= 1.", call. = FALSE)
  }
  out
}

cs_crossfit_q_validate <- function(Y, W, pair_id, profile_order, p_list,
                                   diff, adversarial, K, outcome_model_type,
                                   force_gaussian = FALSE,
                                   adversarial_model_strategy = "four",
                                   competing_group_variable_respondent = NULL,
                                   competing_group_variable_candidate = NULL,
                                   competing_group_competition_variable_candidate = NULL,
                                   competing_group_variable_respondent_proportions = NULL,
                                   respondent_id = NULL, control) {
  if (!isTRUE(diff)) {
    stop("crossfit_q currently requires diff = TRUE.", call. = FALSE)
  }
  if (as.integer(K) != 1L) {
    stop("crossfit_q currently supports K = 1 only.", call. = FALSE)
  }
  if (!identical(tolower(as.character(outcome_model_type)), "glm")) {
    stop("crossfit_q currently supports outcome_model_type = 'glm' only.", call. = FALSE)
  }
  if (isTRUE(force_gaussian)) {
    stop("crossfit_q currently requires force_gaussian = FALSE.", call. = FALSE)
  }
  if (is.null(pair_id) || is.null(profile_order)) {
    stop("crossfit_q requires non-null pair_id and profile_order.", call. = FALSE)
  }
  if (control$split_by == "respondent_id" && is.null(respondent_id)) {
    stop("crossfit_q_control$split_by = 'respondent_id' requires respondent_id.", call. = FALSE)
  }
  if (is.null(p_list) || !length(p_list)) {
    stop("crossfit_q requires a non-empty p_list.", call. = FALSE)
  }
  if (!all(names(p_list) %in% colnames(as.data.frame(W)))) {
    stop("crossfit_q requires p_list names to match W columns.", call. = FALSE)
  }
  y_num <- suppressWarnings(as.numeric(Y))
  if (!all(is.finite(y_num)) || !all(y_num %in% c(0, 1))) {
    stop("crossfit_q currently requires binary 0/1 forced-choice outcomes.", call. = FALSE)
  }
  pair_info <- cs2step_build_pair_mat(
    pair_id = pair_id,
    W = W,
    profile_order = profile_order,
    competing_group_variable_candidate = NULL
  )
  if (is.null(pair_info) || any(pair_info$pair_sizes != 2L)) {
    stop("crossfit_q requires every pair_id to identify exactly two rows.", call. = FALSE)
  }
  pair_mat <- pair_info$pair_mat
  pair_sums <- y_num[pair_mat[, 1]] + y_num[pair_mat[, 2]]
  if (any(pair_sums != 1, na.rm = TRUE)) {
    stop("crossfit_q requires exactly one selected profile per pair.", call. = FALSE)
  }

  if (!isTRUE(adversarial)) {
    return(list(
      mode = "average_pairwise_glm",
      pair_mat = pair_mat,
      pair_info = pair_info
    ))
  }

  cs_crossfit_q_validate_adversarial(
    Y = y_num,
    W = W,
    pair_id = pair_id,
    pair_mat = pair_mat,
    competing_group_variable_respondent = competing_group_variable_respondent,
    competing_group_variable_candidate = competing_group_variable_candidate,
    competing_group_competition_variable_candidate = competing_group_competition_variable_candidate,
    competing_group_variable_respondent_proportions = competing_group_variable_respondent_proportions,
    adversarial_model_strategy = adversarial_model_strategy,
    p_list = p_list,
    control = control
  )
}

cs_crossfit_q_normalize_group_vec <- function(x, n, name) {
  if (is.null(x)) {
    stop(sprintf("crossfit_q with adversarial = TRUE requires '%s'.", name), call. = FALSE)
  }
  if (length(x) != n) {
    stop(sprintf("'%s' has %d elements but W has %d rows.", name, length(x), n),
         call. = FALSE)
  }
  x <- as.character(x)
  if (any(is.na(x) | !nzchar(x))) {
    stop(sprintf("'%s' cannot contain missing or empty group labels.", name), call. = FALSE)
  }
  x
}

cs_crossfit_q_validate_adversarial <- function(Y, W, pair_id, pair_mat,
                                               competing_group_variable_respondent,
                                               competing_group_variable_candidate,
                                               competing_group_competition_variable_candidate,
                                               competing_group_variable_respondent_proportions,
                                               adversarial_model_strategy,
                                               p_list, control) {
  strategy <- tolower(as.character(adversarial_model_strategy)[[1L]])
  if (!identical(strategy, "four")) {
    stop(
      "crossfit_q with adversarial = TRUE currently requires adversarial_model_strategy = 'four'.",
      call. = FALSE
    )
  }

  n <- nrow(as.data.frame(W))
  respondent_group <- cs_crossfit_q_normalize_group_vec(
    competing_group_variable_respondent,
    n,
    "competing_group_variable_respondent"
  )
  candidate_group <- cs_crossfit_q_normalize_group_vec(
    competing_group_variable_candidate,
    n,
    "competing_group_variable_candidate"
  )
  competition_group <- cs_crossfit_q_normalize_group_vec(
    competing_group_competition_variable_candidate,
    n,
    "competing_group_competition_variable_candidate"
  )

  groups <- sort(unique(candidate_group))
  respondent_groups <- sort(unique(respondent_group))
  if (length(groups) != 2L || length(respondent_groups) != 2L ||
      !setequal(groups, respondent_groups)) {
    stop(
      "crossfit_q adversarial support requires exactly the same two candidate and respondent groups.",
      call. = FALSE
    )
  }

  perspective_group <- control$perspective_group %||% groups[[1L]]
  perspective_group <- as.character(perspective_group)
  if (!perspective_group %in% groups) {
    stop("'crossfit_q_control$perspective_group' must match one of the two candidate groups.",
         call. = FALSE)
  }
  opponent_group <- setdiff(groups, perspective_group)[[1L]]

  contests <- cs_crossfit_q_build_adversarial_contests(
    Y = Y,
    pair_id = pair_id,
    pair_mat = pair_mat,
    respondent_group = respondent_group,
    candidate_group = candidate_group,
    competition_group = competition_group,
    groups = groups,
    perspective_group = perspective_group,
    opponent_group = opponent_group
  )
  if (!nrow(contests)) {
    stop("crossfit_q adversarial support requires at least one cross-party heldout contest.",
         call. = FALSE)
  }

  counts_by_group <- table(factor(contests$respondent_group, levels = groups))
  if (any(counts_by_group == 0L)) {
    stop("crossfit_q adversarial support requires cross-party contests for both respondent groups.",
         call. = FALSE)
  }

  same_counts <- cs_crossfit_q_pair_stage_counts(pair_mat, respondent_group, candidate_group, groups)$same
  if (any(same_counts == 0L)) {
    stop(
      "crossfit_q with adversarial_model_strategy = 'four' requires same-party primary pairs for both groups.",
      call. = FALSE
    )
  }

  p_base <- cs_crossfit_q_policy_prob(
    as.data.frame(W, stringsAsFactors = FALSE, check.names = FALSE)[contests$base_idx, , drop = FALSE],
    p_list,
    p_list
  )
  p_other <- cs_crossfit_q_policy_prob(
    as.data.frame(W, stringsAsFactors = FALSE, check.names = FALSE)[contests$other_idx, , drop = FALSE],
    p_list,
    p_list
  )
  if (any(!is.finite(p_base) | !is.finite(p_other) | p_base <= 0 | p_other <= 0)) {
    stop(
      "crossfit_q adversarial support requires positive p_list probability for every heldout contest profile.",
      call. = FALSE
    )
  }

  rho <- cs_crossfit_q_resolve_group_proportions(
    respondent_group = respondent_group,
    groups = groups,
    proportions = competing_group_variable_respondent_proportions
  )

  list(
    mode = "adversarial_pairwise_glm",
    pair_mat = pair_mat,
    pair_info = list(pair_mat = pair_mat),
    contests = contests,
    groups = groups,
    base_group = groups[[1L]],
    other_group = groups[[2L]],
    perspective_group = perspective_group,
    opponent_group = opponent_group,
    respondent_group = respondent_group,
    candidate_group = candidate_group,
    competition_group = competition_group,
    rho = rho
  )
}

cs_crossfit_q_build_adversarial_contests <- function(Y, pair_id, pair_mat,
                                                     respondent_group,
                                                     candidate_group,
                                                     competition_group,
                                                     groups,
                                                     perspective_group,
                                                     opponent_group) {
  rows <- vector("list", nrow(pair_mat))
  for (i in seq_len(nrow(pair_mat))) {
    idx <- pair_mat[i, ]
    resp <- respondent_group[idx]
    cand <- candidate_group[idx]
    comp <- competition_group[idx]
    if (length(unique(resp)) != 1L) {
      stop("crossfit_q adversarial support requires both rows in each pair to share a respondent group.",
           call. = FALSE)
    }
    if (length(unique(comp)) != 1L) {
      stop("crossfit_q adversarial support requires both rows in each pair to share a competition type.",
           call. = FALSE)
    }

    cand_tab <- table(factor(cand, levels = groups))
    cross_party <- all(cand_tab == 1L)
    same_party <- any(cand_tab == 2L)
    if (!cross_party && !same_party) {
      stop("crossfit_q adversarial pairs must be either one-from-each-group or same-party pairs.",
           call. = FALSE)
    }
    if (cross_party && !identical(comp[[1L]], "Different")) {
      stop("crossfit_q adversarial cross-party pairs must have competition type 'Different'.",
           call. = FALSE)
    }
    if (same_party) {
      same_group <- names(cand_tab)[which(cand_tab == 2L)][[1L]]
      if (!identical(comp[[1L]], "Same") || !identical(same_group, resp[[1L]])) {
        stop("crossfit_q adversarial same-party pairs must have competition type 'Same' and match the respondent group.",
             call. = FALSE)
      }
    }
    if (!cross_party) {
      next
    }

    base_idx <- idx[cand == groups[[1L]]][[1L]]
    other_idx <- idx[cand == groups[[2L]]][[1L]]
    perspective_idx <- idx[cand == perspective_group][[1L]]
    opponent_idx <- idx[cand == opponent_group][[1L]]
    if ((Y[base_idx] + Y[other_idx]) != 1) {
      stop("crossfit_q adversarial support requires Y_perspective + Y_opponent = 1.",
           call. = FALSE)
    }

    rows[[i]] <- data.frame(
      pair_row = i,
      pair_id = pair_id[idx[[1L]]],
      respondent_group = resp[[1L]],
      competition_type = comp[[1L]],
      base_idx = base_idx,
      other_idx = other_idx,
      perspective_idx = perspective_idx,
      opponent_idx = opponent_idx,
      Y_base = as.numeric(Y[base_idx]),
      Y_perspective = as.numeric(Y[perspective_idx]),
      stringsAsFactors = FALSE
    )
  }
  rows <- Filter(Negate(is.null), rows)
  if (!length(rows)) {
    return(data.frame())
  }
  do.call(rbind, rows)
}

cs_crossfit_q_pair_stage_counts <- function(pair_mat, respondent_group, candidate_group, groups) {
  same <- setNames(integer(length(groups)), groups)
  cross <- setNames(integer(length(groups)), groups)
  for (i in seq_len(nrow(pair_mat))) {
    idx <- pair_mat[i, ]
    resp <- unique(respondent_group[idx])
    if (length(resp) != 1L || !resp %in% groups) {
      next
    }
    cand <- candidate_group[idx]
    cand_tab <- table(factor(cand, levels = groups))
    if (all(cand_tab == 1L)) {
      cross[[resp]] <- cross[[resp]] + 1L
    }
    if (any(cand_tab == 2L)) {
      same[[resp]] <- same[[resp]] + 1L
    }
  }
  list(same = same, cross = cross)
}

cs_crossfit_q_resolve_group_proportions <- function(respondent_group, groups, proportions = NULL) {
  if (is.null(proportions)) {
    tab <- table(factor(respondent_group, levels = groups))
    rho <- as.numeric(tab) / sum(tab)
    names(rho) <- groups
    return(rho)
  }
  if (is.null(names(proportions))) {
    if (length(proportions) != length(groups)) {
      stop("'competing_group_variable_respondent_proportions' must have one entry per group.",
           call. = FALSE)
    }
    rho <- as.numeric(proportions)
    names(rho) <- groups
  } else {
    rho <- as.numeric(proportions[groups])
    names(rho) <- groups
  }
  if (any(!is.finite(rho)) || any(rho < 0) || sum(rho) <= 0) {
    stop("'competing_group_variable_respondent_proportions' must be non-negative and finite.",
         call. = FALSE)
  }
  rho / sum(rho)
}

cs_crossfit_q_adversarial_fold_strata <- function(pair_mat, respondent_group, candidate_group, groups) {
  vapply(seq_len(nrow(pair_mat)), function(i) {
    idx <- pair_mat[i, ]
    resp <- respondent_group[idx][[1L]]
    cand <- candidate_group[idx]
    cand_tab <- table(factor(cand, levels = groups))
    stage <- if (all(cand_tab == 1L)) {
      "cross"
    } else if (any(cand_tab == 2L)) {
      "same"
    } else {
      "other"
    }
    paste(resp, stage, sep = "::")
  }, character(1))
}

cs_crossfit_q_validate_adversarial_folds <- function(fold_id, pair_mat,
                                                     respondent_group,
                                                     candidate_group,
                                                     groups) {
  for (fold in sort(unique(fold_id))) {
    train_pair_rows <- which(fold_id != fold)
    counts <- cs_crossfit_q_pair_stage_counts(
      pair_mat = pair_mat[train_pair_rows, , drop = FALSE],
      respondent_group = respondent_group,
      candidate_group = candidate_group,
      groups = groups
    )
    if (any(counts$same == 0L) || any(counts$cross == 0L)) {
      stop(
        "crossfit_q adversarial folds must leave same-party and cross-party training pairs for both respondent groups. Reduce crossfit_q_control$folds or use more data.",
        call. = FALSE
      )
    }
  }
  invisible(TRUE)
}

cs_crossfit_q_reconstruct_feature_info <- function(p_list) {
  factor_levels <- vapply(p_list, length, integer(1))
  main_rows <- list()
  for (d in seq_along(factor_levels)) {
    max_l <- max(1L, factor_levels[[d]] - 1L)
    for (l in seq_len(max_l)) {
      main_rows[[length(main_rows) + 1L]] <- data.frame(
        d = d,
        l = l,
        stringsAsFactors = FALSE
      )
    }
  }
  main_info <- do.call(rbind, main_rows)
  main_info$d_full_index <- seq_len(nrow(main_info))
  main_info$d_index <- seq_len(nrow(main_info))

  interaction_info <- data.frame()
  if (nrow(main_info) > 1L) {
    pairs <- utils::combn(seq_len(nrow(main_info)), 2L)
    pairs <- t(pairs)
    pairs <- pairs[main_info$d[pairs[, 1]] != main_info$d[pairs[, 2]], , drop = FALSE]
    if (nrow(pairs)) {
      interaction_info <- do.call(rbind, lapply(seq_len(nrow(pairs)), function(i) {
        comp1 <- main_info[pairs[i, 1], , drop = FALSE]
        comp2 <- main_info[pairs[i, 2], , drop = FALSE]
        data.frame(
          d = comp1$d,
          l = comp1$l,
          dl_index = pairs[i, 1],
          dp = comp2$d,
          lp = comp2$l,
          dplp_index = pairs[i, 2],
          inter_index = i,
          stringsAsFactors = FALSE
        )
      }))
    }
  }
  list(main_info = main_info, interaction_info = interaction_info)
}

cs_crossfit_q_to_numeric <- function(x) {
  out <- tryCatch(as.numeric(x), error = function(e) NULL)
  if (!is.null(out) && length(out)) {
    return(out)
  }
  if (exists("strenv", inherits = TRUE) && "np" %in% ls(envir = strenv)) {
    out <- tryCatch(as.numeric(strenv$np$array(x)), error = function(e) NULL)
    if (!is.null(out) && length(out)) {
      return(out)
    }
  }
  if (requireNamespace("reticulate", quietly = TRUE)) {
    out <- tryCatch(as.numeric(reticulate::py_to_r(x)), error = function(e) NULL)
    if (!is.null(out) && length(out)) {
      return(out)
    }
  }
  numeric(0)
}

cs_crossfit_q_extract_policy <- function(result) {
  pi_point <- result$pi_star_point
  if (is.null(pi_point)) {
    stop("Fold result did not contain pi_star_point.", call. = FALSE)
  }
  if (!is.null(pi_point$k1)) {
    return(pi_point$k1)
  }
  if (is.list(pi_point) && length(pi_point) && is.list(pi_point[[1L]])) {
    return(pi_point[[1L]])
  }
  stop("Could not extract a single average-case policy from pi_star_point.", call. = FALSE)
}

cs_crossfit_q_policy_prob <- function(W, policy, p_list) {
  W <- as.data.frame(W, stringsAsFactors = FALSE)
  probs <- rep(1, nrow(W))
  for (factor_name in names(p_list)) {
    vals <- as.character(W[[factor_name]])
    p <- policy[[factor_name]]
    if (is.null(p)) {
      stop(sprintf("Policy is missing factor '%s'.", factor_name), call. = FALSE)
    }
    probs <- probs * as.numeric(p[vals])
  }
  probs[is.na(probs)] <- 0
  probs
}

cs_crossfit_q_sample_policy <- function(policy, n, seed = NULL) {
  if (!is.null(seed)) {
    set.seed(as.integer(seed))
  }
  out <- lapply(policy, function(p) {
    sample(names(p), size = n, replace = TRUE, prob = as.numeric(p))
  })
  as.data.frame(out, stringsAsFactors = FALSE, check.names = FALSE)
}

cs_crossfit_q_profile_score <- function(W, result, p_list, feature_info = NULL) {
  W <- as.data.frame(W, stringsAsFactors = FALSE, check.names = FALSE)
  W <- W[, names(p_list), drop = FALSE]
  enc <- cs_prepare_W_encoding(
    W = W,
    p_list = p_list,
    unknown = "error",
    align = "by_name"
  )
  if (is.null(feature_info)) {
    feature_info <- cs_crossfit_q_reconstruct_feature_info(p_list)
  }
  X_design <- cs2step_glm_build_design(
    W_idx = enc$W_idx,
    main_info = feature_info$main_info,
    interaction_info = feature_info$interaction_info
  )
  beta <- cs_crossfit_q_to_numeric(result$est_coefficients_jnp)
  if (ncol(X_design) != length(beta)) {
    stop(sprintf(
      "Cross-fit design matrix has %d columns but fold model has %d coefficient(s).",
      ncol(X_design), length(beta)
    ), call. = FALSE)
  }
  as.numeric(X_design %*% beta)
}

cs_crossfit_q_profile_score_beta <- function(W, beta, p_list, feature_info = NULL) {
  W <- as.data.frame(W, stringsAsFactors = FALSE, check.names = FALSE)
  W <- W[, names(p_list), drop = FALSE]
  enc <- cs_prepare_W_encoding(
    W = W,
    p_list = p_list,
    unknown = "error",
    align = "by_name"
  )
  if (is.null(feature_info)) {
    feature_info <- cs_crossfit_q_reconstruct_feature_info(p_list)
  }
  X_design <- cs2step_glm_build_design(
    W_idx = enc$W_idx,
    main_info = feature_info$main_info,
    interaction_info = feature_info$interaction_info
  )
  if (ncol(X_design) != length(beta)) {
    stop(sprintf(
      "Cross-fit design matrix has %d columns but fold model has %d coefficient(s).",
      ncol(X_design), length(beta)
    ), call. = FALSE)
  }
  as.numeric(X_design %*% beta)
}

cs_crossfit_q_model_params <- function(result, suffix) {
  name <- paste0("REGRESSION_PARAMETERS_", suffix)
  params <- cs_crossfit_q_to_numeric(result[[name]])
  if (length(params) >= 2L) {
    return(list(intercept = params[[1L]], beta = params[-1L]))
  }
  intercept_name <- paste0("est_intercept_", suffix, "_jnp")
  coefficient_name <- paste0("est_coefficients_", suffix, "_jnp")
  intercept <- cs_crossfit_q_to_numeric(result[[intercept_name]])
  beta <- cs_crossfit_q_to_numeric(result[[coefficient_name]])
  if (!length(intercept) || !length(beta)) {
    stop(sprintf("Could not extract adversarial GLM parameters for '%s'.", suffix),
         call. = FALSE)
  }
  list(intercept = intercept[[1L]], beta = beta)
}

cs_crossfit_q_pair_predict_params <- function(focal_W, opponent_W, params, p_list,
                                              feature_info = NULL) {
  if (is.null(feature_info)) {
    feature_info <- cs_crossfit_q_reconstruct_feature_info(p_list)
  }
  score_focal <- cs_crossfit_q_profile_score_beta(focal_W, params$beta, p_list, feature_info)
  score_opponent <- cs_crossfit_q_profile_score_beta(opponent_W, params$beta, p_list, feature_info)
  stats::plogis(params$intercept + score_focal - score_opponent)
}

cs_crossfit_q_pair_predict <- function(focal_W, opponent_W, result, p_list,
                                       feature_info = NULL) {
  if (is.null(feature_info)) {
    feature_info <- cs_crossfit_q_reconstruct_feature_info(p_list)
  }
  intercept <- cs_crossfit_q_to_numeric(result$est_intercept_jnp)
  if (!length(intercept)) {
    intercept <- 0
  }
  score_focal <- cs_crossfit_q_profile_score(focal_W, result, p_list, feature_info)
  score_opponent <- cs_crossfit_q_profile_score(opponent_W, result, p_list, feature_info)
  eta <- intercept[[1L]] + score_focal - score_opponent
  family <- tryCatch(
    tolower(as.character(result$outcome_model_view$models$overall$glm_family)),
    error = function(e) "binomial"
  )
  if (identical(family, "binomial")) {
    stats::plogis(eta)
  } else {
    eta
  }
}

cs_crossfit_q_policy_model_mu <- function(policy, opponent_W, result, p_list,
                                          n_draws, seed, chunk_size) {
  feature_info <- cs_crossfit_q_reconstruct_feature_info(p_list)
  draws <- cs_crossfit_q_sample_policy(policy, n = n_draws, seed = seed)
  intercept <- cs_crossfit_q_to_numeric(result$est_intercept_jnp)
  if (!length(intercept)) {
    intercept <- 0
  }
  score_draws <- cs_crossfit_q_profile_score(draws, result, p_list, feature_info)
  score_opp <- cs_crossfit_q_profile_score(opponent_W, result, p_list, feature_info)
  family <- tryCatch(
    tolower(as.character(result$outcome_model_view$models$overall$glm_family)),
    error = function(e) "binomial"
  )
  out <- numeric(length(score_opp))
  starts <- seq.int(1L, length(score_opp), by = as.integer(chunk_size))
  for (start in starts) {
    end <- min(length(score_opp), start + as.integer(chunk_size) - 1L)
    idx <- start:end
    eta_mat <- intercept[[1L]] +
      matrix(score_draws, nrow = length(idx), ncol = length(score_draws), byrow = TRUE) -
      matrix(score_opp[idx], nrow = length(idx), ncol = length(score_draws))
    if (identical(family, "binomial")) {
      out[idx] <- rowMeans(stats::plogis(eta_mat))
    } else {
      out[idx] <- rowMeans(eta_mat)
    }
  }
  out
}

cs_crossfit_q_adversarial_extract_policies <- function(result, groups) {
  pi_point <- result$pi_star_point
  if (is.null(pi_point) || !is.list(pi_point)) {
    stop("Fold result did not contain adversarial pi_star_point.", call. = FALSE)
  }
  if (!all(groups %in% names(pi_point)) && all(c("k1", "k2") %in% names(pi_point))) {
    pi_point <- stats::setNames(pi_point[c("k1", "k2")], groups)
  }
  policies <- lapply(groups, function(g) {
    pol <- pi_point[[g]]
    if (is.null(pol)) {
      stop(sprintf("Fold result is missing policy for group '%s'.", g), call. = FALSE)
    }
    pol
  })
  names(policies) <- groups
  policies
}

cs_crossfit_q_validate_policy_support <- function(policy, p_list, label) {
  for (factor_name in names(p_list)) {
    p <- p_list[[factor_name]]
    pi <- policy[[factor_name]]
    if (is.null(pi)) {
      stop(sprintf("Policy '%s' is missing factor '%s'.", label, factor_name),
           call. = FALSE)
    }
    missing_levels <- setdiff(names(pi)[as.numeric(pi) > 0], names(p))
    if (length(missing_levels)) {
      stop(sprintf(
        "Policy '%s' uses level(s) absent from p_list for factor '%s': %s.",
        label,
        factor_name,
        paste(missing_levels, collapse = ", ")
      ), call. = FALSE)
    }
    zero_support <- names(pi)[as.numeric(pi) > 0 & as.numeric(p[names(pi)]) <= 0]
    if (length(zero_support)) {
      stop(sprintf(
        "Policy '%s' puts mass on zero-probability p_list level(s) for factor '%s': %s.",
        label,
        factor_name,
        paste(zero_support, collapse = ", ")
      ), call. = FALSE)
    }
  }
  invisible(TRUE)
}

cs_crossfit_q_adversarial_model_mu <- function(result, p_list, groups, policies,
                                               respondent_group, perspective_group,
                                               n_draws, seed) {
  feature_info <- cs_crossfit_q_reconstruct_feature_info(p_list)
  base_group <- groups[[1L]]
  other_group <- groups[[2L]]
  params <- cs_crossfit_q_model_params(
    result,
    if (identical(respondent_group, base_group)) "ast" else "dag"
  )
  base_draws <- cs_crossfit_q_sample_policy(policies[[base_group]], n = n_draws, seed = seed)
  other_draws <- cs_crossfit_q_sample_policy(policies[[other_group]], n = n_draws, seed = seed + 7919L)
  p_base_win <- mean(cs_crossfit_q_pair_predict_params(
    focal_W = base_draws,
    opponent_W = other_draws,
    params = params,
    p_list = p_list,
    feature_info = feature_info
  ))
  if (identical(perspective_group, base_group)) {
    p_base_win
  } else {
    1 - p_base_win
  }
}

cs_crossfit_q_adversarial_fold_records <- function(train_result, Y, W,
                                                   contests, test_pair_rows,
                                                   p_list, control, fold,
                                                   groups, perspective_group) {
  fold_contests <- contests[contests$pair_row %in% test_pair_rows, , drop = FALSE]
  if (!nrow(fold_contests)) {
    return(data.frame())
  }

  W <- as.data.frame(W, stringsAsFactors = FALSE, check.names = FALSE)
  feature_info <- cs_crossfit_q_reconstruct_feature_info(p_list)
  policies <- cs_crossfit_q_adversarial_extract_policies(train_result, groups)
  for (g in groups) {
    cs_crossfit_q_validate_policy_support(policies[[g]], p_list, g)
  }

  p_base <- cs_crossfit_q_policy_prob(W[fold_contests$base_idx, , drop = FALSE], p_list, p_list)
  p_other <- cs_crossfit_q_policy_prob(W[fold_contests$other_idx, , drop = FALSE], p_list, p_list)
  pi_base <- cs_crossfit_q_policy_prob(
    W[fold_contests$base_idx, , drop = FALSE],
    policies[[groups[[1L]]]],
    p_list
  )
  pi_other <- cs_crossfit_q_policy_prob(
    W[fold_contests$other_idx, , drop = FALSE],
    policies[[groups[[2L]]]],
    p_list
  )
  p_obs <- p_base * p_other
  if (any(!is.finite(p_obs) | p_obs <= 0)) {
    stop("crossfit_q adversarial encountered non-positive observed assignment probability.",
         call. = FALSE)
  }
  w <- (pi_base * pi_other) / p_obs
  if (any(!is.finite(w))) {
    stop("crossfit_q adversarial produced non-finite target importance weights.",
         call. = FALSE)
  }
  w_used <- pmin(w, control$weight_clip)

  m_base <- numeric(nrow(fold_contests))
  for (g in groups) {
    idx <- which(fold_contests$respondent_group == g)
    if (!length(idx)) {
      next
    }
    params <- cs_crossfit_q_model_params(
      train_result,
      if (identical(g, groups[[1L]])) "ast" else "dag"
    )
    m_base[idx] <- cs_crossfit_q_pair_predict_params(
      focal_W = W[fold_contests$base_idx[idx], , drop = FALSE],
      opponent_W = W[fold_contests$other_idx[idx], , drop = FALSE],
      params = params,
      p_list = p_list,
      feature_info = feature_info
    )
  }
  m_obs <- if (identical(perspective_group, groups[[1L]])) {
    m_base
  } else {
    1 - m_base
  }

  mu_target_by_group <- setNames(numeric(length(groups)), groups)
  mu_reference_by_group <- setNames(numeric(length(groups)), groups)
  reference_policies <- setNames(rep(list(p_list), length(groups)), groups)
  for (g in groups) {
    mu_target_by_group[[g]] <- cs_crossfit_q_adversarial_model_mu(
      result = train_result,
      p_list = p_list,
      groups = groups,
      policies = policies,
      respondent_group = g,
      perspective_group = perspective_group,
      n_draws = control$n_policy_draws,
      seed = control$seed + 1009L * fold + match(g, groups)
    )
    mu_reference_by_group[[g]] <- cs_crossfit_q_adversarial_model_mu(
      result = train_result,
      p_list = p_list,
      groups = groups,
      policies = reference_policies,
      respondent_group = g,
      perspective_group = perspective_group,
      n_draws = control$n_policy_draws,
      seed = control$seed + 2003L * fold + match(g, groups)
    )
  }
  mu_target <- as.numeric(mu_target_by_group[fold_contests$respondent_group])
  mu_reference <- as.numeric(mu_reference_by_group[fold_contests$respondent_group])

  data.frame(
    fold = fold,
    pair_row = fold_contests$pair_row,
    pair_id = fold_contests$pair_id,
    respondent_group = fold_contests$respondent_group,
    Y_perspective = fold_contests$Y_perspective,
    m_obs = m_obs,
    mu_target = mu_target,
    mu_reference = mu_reference,
    weight = w,
    weight_used = w_used,
    weight_clipped = w_used != w,
    p_obs = p_obs,
    pi_obs = pi_base * pi_other,
    stringsAsFactors = FALSE
  )
}

cs_crossfit_q_adversarial_aggregate <- function(records, control, rho, groups) {
  if (!nrow(records)) {
    stop("No adversarial cross-fit heldout records were available.", call. = FALSE)
  }
  n_by_group <- table(factor(records$respondent_group, levels = groups))
  if (any(n_by_group == 0L)) {
    stop("Adversarial cross-fit aggregation requires heldout records for both respondent groups.",
         call. = FALSE)
  }
  records$a <- as.numeric(rho[records$respondent_group]) /
    as.numeric(n_by_group[records$respondent_group])
  records$a <- records$a / sum(records$a)

  summarize_one <- function(df, strict_snips = TRUE) {
    w <- df$weight_used
    a <- df$a
    y <- df$Y_perspective
    denom <- sum(a * w, na.rm = TRUE)
    if (strict_snips && "snips" %in% control$estimators &&
        (!is.finite(denom) || denom <= 0)) {
      stop(
        "Adversarial crossfit_q SNIPS denominator is zero. Use a target policy with heldout support or remove 'snips' from crossfit_q_control$estimators.",
        call. = FALSE
      )
    }
    target <- c(
      ips = sum(a * w * y, na.rm = TRUE),
      snips = if (denom > 0) sum(a * w * y, na.rm = TRUE) / denom else NA_real_,
      model = sum(a * df$mu_target, na.rm = TRUE),
      dr = sum(a * (df$mu_target + w * (y - df$m_obs)), na.rm = TRUE)
    )
    reference <- c(
      ips = sum(a * y, na.rm = TRUE),
      snips = sum(a * y, na.rm = TRUE),
      model = sum(a * df$mu_reference, na.rm = TRUE),
      dr = sum(a * (df$mu_reference + (y - df$m_obs)), na.rm = TRUE)
    )
    estimators <- intersect(control$estimators, names(target))
    data.frame(
      estimator = estimators,
      Q_crossfit = as.numeric(target[estimators]),
      Q_reference_crossfit = as.numeric(reference[estimators]),
      Q_gain_crossfit = as.numeric(target[estimators] - reference[estimators]),
      n_records = nrow(df),
      ess = ess_fxn(a * w),
      ess_fraction = ess_fxn(a * w) / nrow(df),
      max_weight = max(df$weight, na.rm = TRUE),
      mean_weight = mean(df$weight, na.rm = TRUE),
      snips_denominator = denom,
      any_weight_clipped = any(df$weight_clipped),
      stringsAsFactors = FALSE
    )
  }

  summary_df <- summarize_one(records, strict_snips = TRUE)
  fold_rows <- lapply(split(records, records$fold), function(df) {
    out <- summarize_one(df, strict_snips = FALSE)
    out$fold <- df$fold[[1L]]
    out
  })
  fold_df <- do.call(rbind, fold_rows)
  fold_df <- fold_df[, c("fold", setdiff(names(fold_df), "fold")), drop = FALSE]

  list(
    records = records,
    summary = summary_df,
    target_summary = summary_df[, c("estimator", "Q_crossfit"), drop = FALSE],
    reference_summary = summary_df[, c("estimator", "Q_reference_crossfit"), drop = FALSE],
    folds = fold_df
  )
}

cs_crossfit_q_weight_diagnostics <- function(w, w_used) {
  quantile_probs <- c(0, 0.5, 0.9, 0.95, 0.99, 1)
  q <- stats::quantile(w, probs = quantile_probs, na.rm = TRUE, names = FALSE)
  names(q) <- paste0("p", c("00", "50", "90", "95", "99", "100"))
  list(
    n = length(w),
    ess = ess_fxn(w_used),
    ess_fraction = ess_fxn(w_used) / length(w_used),
    mean = mean(w, na.rm = TRUE),
    max = max(w, na.rm = TRUE),
    quantiles = q,
    clipped = any(w_used != w, na.rm = TRUE)
  )
}

cs_crossfit_q_fold_eval <- function(train_result, Y, W, pair_mat, test_pair_rows,
                                    p_list, control, fold) {
  W <- as.data.frame(W, stringsAsFactors = FALSE, check.names = FALSE)
  pair_mat_test <- pair_mat[test_pair_rows, , drop = FALSE]
  focal_idx <- c(pair_mat_test[, 1], pair_mat_test[, 2])
  opponent_idx <- c(pair_mat_test[, 2], pair_mat_test[, 1])
  y_oriented <- as.numeric(Y[focal_idx])
  focal_W <- W[focal_idx, , drop = FALSE]
  opponent_W <- W[opponent_idx, , drop = FALSE]

  policy <- cs_crossfit_q_extract_policy(train_result)
  m_obs <- cs_crossfit_q_pair_predict(focal_W, opponent_W, train_result, p_list)
  mu_policy <- cs_crossfit_q_policy_model_mu(
    policy = policy,
    opponent_W = opponent_W,
    result = train_result,
    p_list = p_list,
    n_draws = control$n_policy_draws,
    seed = control$seed + 1009L * fold,
    chunk_size = control$chunk_size
  )
  mu_reference <- cs_crossfit_q_policy_model_mu(
    policy = p_list,
    opponent_W = opponent_W,
    result = train_result,
    p_list = p_list,
    n_draws = control$n_policy_draws,
    seed = control$seed + 2003L * fold,
    chunk_size = control$chunk_size
  )

  p_focal <- cs_crossfit_q_policy_prob(focal_W, p_list, p_list)
  pi_focal <- cs_crossfit_q_policy_prob(focal_W, policy, p_list)
  w <- ifelse(p_focal > 0, pi_focal / p_focal, 0)
  w_ref <- ifelse(p_focal > 0, p_focal / p_focal, 0)
  w_used <- pmin(w, control$weight_clip)
  w_ref_used <- pmin(w_ref, control$weight_clip)

  estimates <- list(
    ips = mean(w_used * y_oriented, na.rm = TRUE),
    snips = if (sum(w_used, na.rm = TRUE) > 0) {
      sum(w_used * y_oriented, na.rm = TRUE) / sum(w_used, na.rm = TRUE)
    } else {
      NA_real_
    },
    model = mean(mu_policy, na.rm = TRUE),
    dr = mean(mu_policy + w_used * (y_oriented - m_obs), na.rm = TRUE)
  )
  reference <- list(
    ips = mean(w_ref_used * y_oriented, na.rm = TRUE),
    snips = if (sum(w_ref_used, na.rm = TRUE) > 0) {
      sum(w_ref_used * y_oriented, na.rm = TRUE) / sum(w_ref_used, na.rm = TRUE)
    } else {
      NA_real_
    },
    model = mean(mu_reference, na.rm = TRUE),
    dr = mean(mu_reference + w_ref_used * (y_oriented - m_obs), na.rm = TRUE)
  )
  estimates <- estimates[control$estimators]
  reference <- reference[control$estimators]
  data.frame(
    fold = fold,
    n_pairs = nrow(pair_mat_test),
    n_oriented = length(y_oriented),
    estimator = names(estimates),
    Q_crossfit = as.numeric(estimates),
    Q_reference_crossfit = as.numeric(reference),
    Q_gain_crossfit = as.numeric(estimates) - as.numeric(reference),
    ess = cs_crossfit_q_weight_diagnostics(w, w_used)$ess,
    ess_fraction = cs_crossfit_q_weight_diagnostics(w, w_used)$ess_fraction,
    max_weight = max(w, na.rm = TRUE),
    mean_weight = mean(w, na.rm = TRUE),
    weight_clipped = any(w_used != w, na.rm = TRUE),
    stringsAsFactors = FALSE
  )
}

cs_crossfit_q_strategize <- function(Y, W, X = NULL, lambda = NULL,
                                     varcov_cluster_variable = NULL,
                                     competing_group_variable_respondent = NULL,
                                     competing_group_variable_candidate = NULL,
                                     competing_group_competition_variable_candidate = NULL,
                                     competing_group_variable_respondent_proportions = NULL,
                                     pair_id = NULL,
                                     respondent_id = NULL,
                                     respondent_task_id = NULL,
                                     profile_order = NULL,
                                     p_list = NULL,
                                     slate_list = NULL,
                                     K = 1,
                                     nSGD = 100,
                                     diff = FALSE,
                                     adversarial = FALSE,
                                     adversarial_model_strategy = "four",
                                     include_stage_interactions = NULL,
                                     partial_pooling = NULL,
                                     partial_pooling_strength = 50,
                                     use_regularization = TRUE,
                                     force_gaussian = FALSE,
                                     force_reinforce = FALSE,
                                     a_init_sd = 0.001,
                                     outcome_model_type = "glm",
                                     neural_mcmc_control = NULL,
                                     penalty_type = "KL",
                                     se_method = c("full", "implicit"),
                                     conda_env = "strategize_env",
                                     conda_env_required = FALSE,
                                     conf_level = 0.90,
                                     nFolds_glm = 3L,
                                     folds = NULL,
                                     nMonte_adversarial = 5L,
                                     primary_pushforward = "mc",
                                     primary_strength = 1.0,
                                     primary_n_entrants = 1L,
                                     primary_n_field = 1L,
                                     nMonte_Qglm = 100L,
                                     learning_rate_max = 0.001,
                                     temperature = NULL,
                                     save_outcome_model = FALSE,
                                     presaved_outcome_model = FALSE,
                                     outcome_model_key = NULL,
                                     use_optax = FALSE,
                                     optim_type = "gd",
                                     optimism = "extragrad",
                                     optimism_coef = 1,
                                     rain_lambda = 1,
                                     rain_gamma = 0.01,
                                     rain_L = NULL,
                                     rain_eta = 0.001,
                                     rain_variant = "alg10_staged",
                                     rain_output = "last",
                                     control = NULL) {
  control <- cs_crossfit_q_default_control(control)
  validation <- cs_crossfit_q_validate(
    Y = Y,
    W = W,
    pair_id = pair_id,
    profile_order = profile_order,
    p_list = p_list,
    diff = diff,
    adversarial = adversarial,
    K = K,
    outcome_model_type = outcome_model_type,
    force_gaussian = force_gaussian,
    adversarial_model_strategy = adversarial_model_strategy,
    competing_group_variable_respondent = competing_group_variable_respondent,
    competing_group_variable_candidate = competing_group_variable_candidate,
    competing_group_competition_variable_candidate = competing_group_competition_variable_candidate,
    competing_group_variable_respondent_proportions = competing_group_variable_respondent_proportions,
    respondent_id = respondent_id,
    control = control
  )
  pair_mat <- validation$pair_mat

  split_cluster <- if (control$split_by == "respondent_id") {
    respondent_id[pair_mat[, 1]]
  } else {
    pair_id[pair_mat[, 1]]
  }
  fold_y <- if (isTRUE(adversarial)) {
    cs_crossfit_q_adversarial_fold_strata(
      pair_mat = pair_mat,
      respondent_group = validation$respondent_group,
      candidate_group = validation$candidate_group,
      groups = validation$groups
    )
  } else {
    as.numeric(Y[pair_mat[, 1]])
  }
  fold_obj <- cs_make_stratified_folds(
    n = nrow(pair_mat),
    n_folds = control$folds,
    y = fold_y,
    cluster = split_cluster,
    seed = control$seed
  )
  if (is.null(fold_obj)) {
    stop("Could not construct crossfit_q folds.", call. = FALSE)
  }
  if (isTRUE(adversarial)) {
    cs_crossfit_q_validate_adversarial_folds(
      fold_id = fold_obj$fold_id,
      pair_mat = pair_mat,
      respondent_group = validation$respondent_group,
      candidate_group = validation$candidate_group,
      groups = validation$groups
    )
  }

  fold_results <- list()
  for (fold in seq_len(fold_obj$n_folds)) {
    test_pair_rows <- which(fold_obj$fold_id == fold)
    train_pair_rows <- which(fold_obj$fold_id != fold)
    train_rows <- as.vector(t(pair_mat[train_pair_rows, , drop = FALSE]))

    train_args <- list(
      Y = Y[train_rows],
      W = as.data.frame(W, stringsAsFactors = FALSE, check.names = FALSE)[train_rows, , drop = FALSE],
      X = if (!is.null(X)) as.matrix(X)[train_rows, , drop = FALSE] else NULL,
      lambda = lambda,
      varcov_cluster_variable = varcov_cluster_variable[train_rows],
      competing_group_variable_respondent = if (!is.null(competing_group_variable_respondent)) competing_group_variable_respondent[train_rows] else NULL,
      competing_group_variable_candidate = if (!is.null(competing_group_variable_candidate)) competing_group_variable_candidate[train_rows] else NULL,
      competing_group_competition_variable_candidate = if (!is.null(competing_group_competition_variable_candidate)) competing_group_competition_variable_candidate[train_rows] else NULL,
      competing_group_variable_respondent_proportions = if (isTRUE(adversarial)) validation$rho else competing_group_variable_respondent_proportions,
      pair_id = pair_id[train_rows],
      respondent_id = if (!is.null(respondent_id)) respondent_id[train_rows] else NULL,
      respondent_task_id = if (!is.null(respondent_task_id)) respondent_task_id[train_rows] else NULL,
      profile_order = profile_order[train_rows],
      p_list = p_list,
      slate_list = slate_list,
      K = K,
      nSGD = nSGD,
      diff = diff,
      adversarial = adversarial,
      adversarial_model_strategy = adversarial_model_strategy,
      include_stage_interactions = include_stage_interactions,
      partial_pooling = partial_pooling,
      partial_pooling_strength = partial_pooling_strength,
      use_regularization = use_regularization,
      force_gaussian = force_gaussian,
      force_reinforce = force_reinforce,
      a_init_sd = a_init_sd,
      outcome_model_type = outcome_model_type,
      neural_mcmc_control = neural_mcmc_control,
      penalty_type = penalty_type,
      compute_se = FALSE,
      se_method = match.arg(se_method),
      conda_env = conda_env,
      conda_env_required = conda_env_required,
      conf_level = conf_level,
      nFolds_glm = nFolds_glm,
      folds = folds,
      nMonte_adversarial = nMonte_adversarial,
      primary_pushforward = primary_pushforward,
      primary_strength = primary_strength,
      primary_n_entrants = primary_n_entrants,
      primary_n_field = primary_n_field,
      nMonte_Qglm = nMonte_Qglm,
      learning_rate_max = learning_rate_max,
      temperature = temperature,
      save_outcome_model = FALSE,
      presaved_outcome_model = FALSE,
      outcome_model_key = NULL,
      use_optax = use_optax,
      optim_type = optim_type,
      optimism = optimism,
      optimism_coef = optimism_coef,
      rain_lambda = rain_lambda,
      rain_gamma = rain_gamma,
      rain_L = rain_L,
      rain_eta = rain_eta,
      rain_variant = rain_variant,
      rain_output = rain_output,
      compute_hessian = FALSE,
      crossfit_q = FALSE
    )
    train_result <- do.call(strategize, train_args)
    if (isTRUE(adversarial)) {
      fold_results[[fold]] <- cs_crossfit_q_adversarial_fold_records(
        train_result = train_result,
        Y = Y,
        W = W,
        contests = validation$contests,
        test_pair_rows = test_pair_rows,
        p_list = p_list,
        control = control,
        fold = fold,
        groups = validation$groups,
        perspective_group = validation$perspective_group
      )
    } else {
      fold_results[[fold]] <- cs_crossfit_q_fold_eval(
        train_result = train_result,
        Y = Y,
        W = W,
        pair_mat = pair_mat,
        test_pair_rows = test_pair_rows,
        p_list = p_list,
        control = control,
        fold = fold
      )
    }
  }

  if (isTRUE(adversarial)) {
    records <- do.call(rbind, fold_results)
    agg <- cs_crossfit_q_adversarial_aggregate(
      records = records,
      control = control,
      rho = validation$rho,
      groups = validation$groups
    )
    headline_row <- agg$summary[agg$summary$estimator == control$headline, , drop = FALSE]
    return(list(
      Q_crossfit = headline_row$Q_crossfit[[1L]],
      Q_reference_crossfit = headline_row$Q_reference_crossfit[[1L]],
      Q_gain_crossfit = headline_row$Q_gain_crossfit[[1L]],
      estimator = control$headline,
      mode = validation$mode,
      perspective_group = validation$perspective_group,
      opponent_group = validation$opponent_group,
      assignment_assumption = "independent_product_p_list",
      summary = agg$summary,
      target_summary = agg$target_summary,
      reference_summary = agg$reference_summary,
      folds = if (isTRUE(control$return_fold_results)) agg$folds else NULL,
      records = if (isTRUE(control$return_fold_results)) agg$records else NULL,
      control = control,
      split = list(
        split_by = control$split_by,
        n_folds = fold_obj$n_folds,
        fold_id_by_pair = fold_obj$fold_id,
        pair_id = pair_id[pair_mat[, 1]]
      )
    ))
  }

  fold_df <- do.call(rbind, fold_results)
  aggregate_cols <- c("Q_crossfit", "Q_reference_crossfit", "Q_gain_crossfit")
  summary_rows <- lapply(split(fold_df, fold_df$estimator), function(df) {
    weights <- df$n_oriented / sum(df$n_oriented)
    vals <- vapply(aggregate_cols, function(nm) {
      sum(df[[nm]] * weights, na.rm = TRUE)
    }, numeric(1))
    data.frame(
      estimator = df$estimator[[1L]],
      Q_crossfit = vals[["Q_crossfit"]],
      Q_reference_crossfit = vals[["Q_reference_crossfit"]],
      Q_gain_crossfit = vals[["Q_gain_crossfit"]],
      n_folds = length(unique(df$fold)),
      n_oriented = sum(df$n_oriented),
      mean_ess_fraction = mean(df$ess_fraction, na.rm = TRUE),
      max_weight = max(df$max_weight, na.rm = TRUE),
      any_weight_clipped = any(df$weight_clipped),
      stringsAsFactors = FALSE
    )
  })
  summary_df <- do.call(rbind, summary_rows)
  headline_row <- summary_df[summary_df$estimator == control$headline, , drop = FALSE]
  list(
    Q_crossfit = headline_row$Q_crossfit[[1L]],
    Q_reference_crossfit = headline_row$Q_reference_crossfit[[1L]],
    Q_gain_crossfit = headline_row$Q_gain_crossfit[[1L]],
    estimator = control$headline,
    summary = summary_df,
    folds = if (isTRUE(control$return_fold_results)) fold_df else NULL,
    control = control,
    split = list(
      split_by = control$split_by,
      n_folds = fold_obj$n_folds,
      fold_id_by_pair = fold_obj$fold_id,
      pair_id = pair_id[pair_mat[, 1]]
    )
  )
}
