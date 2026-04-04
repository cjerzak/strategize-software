#' Conjoint Foundation Models
#'
#' @description
#' Train a pooled neural representation across multiple conjoint experiments,
#' then adapt that representation to a single study through the existing
#' Bayesian neural outcome model.
#'
#' @details
#' Foundation-model training in \pkg{strategize} follows a two-stage workflow:
#'
#' \enumerate{
#'   \item Run \code{\link{build_backend}()} once so the JAX-backed neural
#'   backend is available.
#'   \item Prepare a list of experiment specifications and pass them to
#'   \code{\link{fit_conjoint_foundation_model}()}.
#'   \item Adapt the resulting shared representation to a target study with
#'   \code{\link{adapt_conjoint_foundation_model}()}.
#'   \item Optionally persist the fitted foundation object with
#'   \code{\link{save_conjoint_foundation_bundle}()} and reload it later with
#'   \code{\link{load_conjoint_foundation_bundle}()}.
#' }
#'
#' Pooled training does not force every experiment into one universal output
#' head. Experiments are grouped into compatible neural families defined by
#' \code{(mode, likelihood, n_outcomes)}. Each compatible family shares one
#' pooled schema-aware encoder and one neural fit, while incompatible families
#' are stored as separate internal groups in the returned
#' \code{conjoint_foundation_model}.
#'
#' Cross-study sharing is driven by explicit canonical ids. Raw label equality
#' alone does not force two studies to share factor or level identities.
#' Optional text embeddings act as side information for pooled features and
#' adaptation, not as the main identity mechanism.
#'
#' See \code{\link{fit_conjoint_foundation_model}()} for the full experiment
#' specification and \code{\link{adapt_conjoint_foundation_model}()} for the
#' target-study adaptation contract.
#'
#' @name conjoint-foundation
NULL

cs_foundation_default_control <- function() {
  list(
    add_experiment_indicators = TRUE,
    add_text_semantics = TRUE,
    text_embedding_fn = NULL,
    neural_mcmc_control = list(
      subsample_method = "batch_vi",
      uncertainty_scope = "output",
      optimizer = "muon",
      svi_lr_schedule = "warmup_cosine"
    )
  )
}

cs_foundation_default_adaptation_control <- function() {
  list(
    strict_schema_match = FALSE,
    allow_extra_covariates = TRUE,
    use_text_semantics = TRUE,
    text_embedding_fn = NULL
  )
}

cs_foundation_mode <- function(mode, pair_id) {
  if (!is.null(mode)) {
    mode_use <- tolower(as.character(mode))
    if (!mode_use %in% c("single", "pairwise", "auto")) {
      stop(
        "'mode' must be one of 'auto', 'single', or 'pairwise'.",
        call. = FALSE
      )
    }
    if (!identical(mode_use, "auto")) {
      return(mode_use)
    }
  }
  if (!is.null(pair_id)) "pairwise" else "single"
}

cs_foundation_infer_likelihood <- function(Y, W, likelihood = NULL, n_outcomes = NULL) {
  y_num <- suppressWarnings(as.numeric(Y))
  if (!is.null(likelihood)) {
    like <- tolower(as.character(likelihood))
    if (!like %in% c("bernoulli", "categorical", "normal", "auto")) {
      stop(
        "'likelihood' must be one of 'auto', 'bernoulli', 'categorical', or 'normal'.",
        call. = FALSE
      )
    }
    if (!identical(like, "auto")) {
      return(list(likelihood = like, n_outcomes = n_outcomes))
    }
  }

  vals <- unique(stats::na.omit(y_num))
  is_binary <- length(vals) <= 2L && all(vals %in% c(0, 1))
  is_intvec <- length(y_num) > 0L &&
    all(!is.na(y_num)) &&
    all(abs(y_num - round(y_num)) < 1e-8)
  k_classes <- if (is_intvec) length(unique(as.integer(y_num))) else NA_integer_

  if (is_binary) {
    return(list(likelihood = "bernoulli", n_outcomes = 1L))
  }
  if (!is.na(k_classes) && k_classes >= 2L && k_classes <= max(50L, ncol(W) + 1L)) {
    return(list(likelihood = "categorical", n_outcomes = k_classes))
  }
  list(likelihood = "normal", n_outcomes = 1L)
}

cs_foundation_normalize_categorical_y <- function(Y, n_outcomes = NULL) {
  y_num <- suppressWarnings(as.numeric(Y))
  if (anyNA(y_num)) {
    stop("Categorical outcomes cannot contain NA values.", call. = FALSE)
  }
  levels_obs <- sort(unique(as.integer(y_num)))
  y_map <- match(as.integer(y_num), levels_obs) - 1L
  n_obs <- length(levels_obs)
  n_outcomes_use <- as.integer(n_outcomes %||% n_obs)
  if (n_outcomes_use != n_obs) {
    stop(
      "For categorical outcomes, 'n_outcomes' must match the number of observed classes in v1.",
      call. = FALSE
    )
  }
  list(
    Y = as.integer(y_map),
    outcome_levels = levels_obs,
    n_outcomes = n_outcomes_use
  )
}

cs_foundation_normalize_names_list_local <- function(names_list, W, p_list = NULL) {
  if (is.null(names_list)) {
    return(cs_build_names_list(W = W, p_list = p_list))
  }
  out <- lapply(names_list, function(x) {
    if (is.list(x) && length(x) == 1L && is.atomic(x[[1]])) {
      return(as.character(x[[1]]))
    }
    if (is.atomic(x)) {
      return(as.character(x))
    }
    as.character(unlist(x))
  })
  if (is.null(names(out))) {
    names(out) <- colnames(as.data.frame(W))
  }
  lapply(out, function(x) list(as.character(x)))
}

cs_foundation_validate_numeric_matrix <- function(X, n, arg = "X") {
  if (is.null(X)) {
    return(NULL)
  }
  X_df <- as.data.frame(X)
  if (nrow(X_df) != n) {
    stop(
      sprintf("'%s' has %d rows but expected %d.", arg, nrow(X_df), n),
      call. = FALSE
    )
  }
  is_num <- vapply(X_df, function(col) is.numeric(col) || is.integer(col), logical(1))
  if (!all(is_num)) {
    bad <- names(X_df)[!is_num]
    stop(
      sprintf("'%s' must contain only numeric columns. Bad columns: %s",
              arg, paste(bad, collapse = ", ")),
      call. = FALSE
    )
  }
  X_mat <- as.matrix(X_df)
  storage.mode(X_mat) <- "double"
  if (is.null(colnames(X_mat))) {
    colnames(X_mat) <- paste0("X", seq_len(ncol(X_mat)))
  }
  X_mat
}

cs_foundation_normalize_factor_ids <- function(values, factor_names, arg = "canonical_factor_id") {
  out <- rep(NA_character_, length(factor_names))
  names(out) <- factor_names
  if (is.null(values)) {
    return(out)
  }
  if (is.list(values) && !is.atomic(values)) {
    if (!is.null(names(values))) {
      values <- values[factor_names]
    }
    values <- unlist(values, recursive = FALSE, use.names = TRUE)
  }
  values <- as.character(values)
  if (!is.null(names(values))) {
    idx <- match(names(values), factor_names)
    ok <- which(!is.na(idx))
    out[idx[ok]] <- values[ok]
  } else {
    if (length(values) != length(factor_names)) {
      stop(
        sprintf("'%s' must have length %d when unnamed.", arg, length(factor_names)),
        call. = FALSE
      )
    }
    out[] <- values
  }
  out
}

cs_foundation_normalize_level_ids <- function(values, factor_names, names_list) {
  out <- setNames(vector("list", length(factor_names)), factor_names)
  for (factor_name in factor_names) {
    levels_here <- names_list[[factor_name]][[1]]
    empty <- rep(NA_character_, length(levels_here))
    names(empty) <- levels_here
    out[[factor_name]] <- empty
  }
  if (is.null(values)) {
    return(out)
  }
  if (!is.list(values)) {
    stop("'canonical_level_id' must be a named list when provided.", call. = FALSE)
  }
  for (factor_name in intersect(names(values), factor_names)) {
    levs <- names_list[[factor_name]][[1]]
    raw <- values[[factor_name]]
    if (is.list(raw) && length(raw) == 1L && is.atomic(raw[[1]])) {
      raw <- raw[[1]]
    }
    raw <- as.character(raw)
    mapped <- rep(NA_character_, length(levs))
    names(mapped) <- levs
    if (!is.null(names(raw))) {
      idx <- match(names(raw), levs)
      ok <- which(!is.na(idx))
      mapped[idx[ok]] <- raw[ok]
    } else {
      if (length(raw) != length(levs)) {
        stop(
          sprintf(
            "canonical_level_id[['%s']] must have length %d when unnamed.",
            factor_name,
            length(levs)
          ),
          call. = FALSE
        )
      }
      mapped[] <- raw
    }
    out[[factor_name]] <- mapped
  }
  out
}

cs_foundation_make_factor_key <- function(experiment_id, factor_name, canonical_factor_id = NULL) {
  canon <- canonical_factor_id %||% NA_character_
  if (!is.na(canon) && nzchar(canon)) {
    paste0("canon::", canon)
  } else {
    paste0("local::", experiment_id, "::", factor_name)
  }
}

cs_foundation_make_level_key <- function(experiment_id,
                                         factor_key,
                                         factor_name,
                                         level_name,
                                         canonical_level_id = NULL) {
  canon <- canonical_level_id %||% NA_character_
  if (!is.na(canon) && nzchar(canon)) {
    paste0("canon::", canon)
  } else {
    paste0("local::", experiment_id, "::", factor_name, "::", level_name)
  }
}

cs_foundation_group_key <- function(mode, likelihood, n_outcomes) {
  paste(mode, likelihood, as.integer(n_outcomes %||% 1L), sep = "::")
}

cs_foundation_normalize_experiment <- function(experiment, index) {
  if (!is.list(experiment)) {
    stop("Each experiment must be supplied as a list.", call. = FALSE)
  }
  experiment_id <- as.character(experiment$experiment_id %||% sprintf("experiment_%03d", index))
  if (length(experiment_id) != 1L || !nzchar(experiment_id)) {
    stop("Each experiment must have a non-empty 'experiment_id'.", call. = FALSE)
  }
  if (is.null(experiment$Y)) {
    stop(sprintf("Experiment '%s' is missing 'Y'.", experiment_id), call. = FALSE)
  }
  if (is.null(experiment$W)) {
    stop(sprintf("Experiment '%s' is missing 'W'.", experiment_id), call. = FALSE)
  }

  W_df <- as.data.frame(experiment$W)
  if (ncol(W_df) < 1L) {
    stop(sprintf("Experiment '%s' must have at least one factor column.", experiment_id), call. = FALSE)
  }
  if (is.null(colnames(W_df))) {
    colnames(W_df) <- paste0("V", seq_len(ncol(W_df)))
  }
  Y_raw <- experiment$Y
  if (length(Y_raw) != nrow(W_df)) {
    stop(
      sprintf("Experiment '%s' has %d outcomes but %d profile rows.",
              experiment_id, length(Y_raw), nrow(W_df)),
      call. = FALSE
    )
  }

  mode <- cs_foundation_mode(experiment$mode %||% NULL, experiment$pair_id %||% NULL)
  if (identical(mode, "pairwise")) {
    cs2step_validate_pairwise_ids(experiment$pair_id, nrow(W_df))
  }

  names_list <- cs_foundation_normalize_names_list_local(
    names_list = experiment$names_list %||% NULL,
    W = W_df,
    p_list = experiment$p_list %||% NULL
  )
  factor_names <- names(names_list)
  factor_levels <- vapply(names_list, function(x) length(x[[1]]), integer(1))

  like_info <- cs_foundation_infer_likelihood(
    Y = Y_raw,
    W = W_df,
    likelihood = experiment$likelihood %||% NULL,
    n_outcomes = experiment$n_outcomes %||% NULL
  )
  likelihood <- like_info$likelihood
  n_outcomes <- as.integer(like_info$n_outcomes %||% 1L)
  outcome_levels <- NULL
  if (identical(likelihood, "bernoulli")) {
    vals <- unique(stats::na.omit(as.numeric(Y_raw)))
    if (!all(vals %in% c(0, 1))) {
      stop(
        sprintf("Experiment '%s' declared bernoulli but Y is not binary 0/1.", experiment_id),
        call. = FALSE
      )
    }
    Y_use <- as.numeric(Y_raw)
  } else if (identical(likelihood, "categorical")) {
    cat_info <- cs_foundation_normalize_categorical_y(Y_raw, n_outcomes = n_outcomes)
    Y_use <- cat_info$Y
    outcome_levels <- cat_info$outcome_levels
    n_outcomes <- cat_info$n_outcomes
  } else {
    Y_use <- as.numeric(Y_raw)
  }

  X_use <- cs_foundation_validate_numeric_matrix(
    X = experiment$X %||% NULL,
    n = nrow(W_df),
    arg = sprintf("experiments[['%s']]$X", experiment_id)
  )

  respondent_id <- experiment$respondent_id %||% NULL
  if (!is.null(respondent_id) && length(respondent_id) != nrow(W_df)) {
    stop(
      sprintf("Experiment '%s' has respondent_id length %d but %d rows.",
              experiment_id, length(respondent_id), nrow(W_df)),
      call. = FALSE
    )
  }
  respondent_task_id <- experiment$respondent_task_id %||% NULL
  if (!is.null(respondent_task_id) && length(respondent_task_id) != nrow(W_df)) {
    stop(
      sprintf("Experiment '%s' has respondent_task_id length %d but %d rows.",
              experiment_id, length(respondent_task_id), nrow(W_df)),
      call. = FALSE
    )
  }

  canonical_factor_id <- cs_foundation_normalize_factor_ids(
    values = experiment$canonical_factor_id %||% NULL,
    factor_names = factor_names
  )
  canonical_level_id <- cs_foundation_normalize_level_ids(
    values = experiment$canonical_level_id %||% NULL,
    factor_names = factor_names,
    names_list = names_list
  )

  list(
    experiment_id = experiment_id,
    experiment_label = as.character(experiment$experiment_label %||% experiment_id),
    Y = Y_use,
    Y_raw = Y_raw,
    W = W_df,
    names_list = names_list,
    factor_names = factor_names,
    factor_levels = factor_levels,
    mode = mode,
    likelihood = likelihood,
    n_outcomes = as.integer(n_outcomes),
    outcome_levels = outcome_levels,
    pair_id = experiment$pair_id %||% NULL,
    profile_order = experiment$profile_order %||% NULL,
    respondent_id = respondent_id,
    respondent_task_id = respondent_task_id,
    X = X_use,
    canonical_factor_id = canonical_factor_id,
    canonical_level_id = canonical_level_id
  )
}

cs_foundation_text_embed <- function(text_embedding_fn, texts) {
  texts <- as.character(texts)
  if (!length(texts)) {
    return(matrix(numeric(0), nrow = 0L, ncol = 0L))
  }
  out <- tryCatch(text_embedding_fn(texts), error = function(e) NULL)
  if (!is.null(out)) {
    out <- as.matrix(out)
    storage.mode(out) <- "double"
    if (nrow(out) == length(texts)) {
      return(out)
    }
  }
  out_rows <- lapply(texts, function(txt) {
    emb <- text_embedding_fn(txt)
    emb <- as.numeric(emb)
    if (!length(emb)) {
      stop("text_embedding_fn must return a numeric vector or matrix.", call. = FALSE)
    }
    emb
  })
  dims <- unique(vapply(out_rows, length, integer(1)))
  if (length(dims) != 1L) {
    stop("text_embedding_fn returned inconsistent embedding lengths.", call. = FALSE)
  }
  mat <- do.call(rbind, out_rows)
  storage.mode(mat) <- "double"
  mat
}

cs_foundation_build_group_registry <- function(experiments) {
  slot_key_to_name <- character(0)
  slot_names <- character(0)
  slot_key_to_index <- integer(0)
  slot_level_keys <- list()
  slot_level_labels <- list()
  slot_factor_labels <- character(0)
  experiment_maps <- list()

  for (exp in experiments) {
    factor_map <- vector("list", length(exp$factor_names))
    names(factor_map) <- exp$factor_names
    level_map <- setNames(vector("list", length(exp$factor_names)), exp$factor_names)

    for (j in seq_along(exp$factor_names)) {
      factor_name <- exp$factor_names[[j]]
      factor_key <- cs_foundation_make_factor_key(
        experiment_id = exp$experiment_id,
        factor_name = factor_name,
        canonical_factor_id = exp$canonical_factor_id[[factor_name]]
      )
      if (!factor_key %in% names(slot_key_to_name)) {
        slot_name <- sprintf("slot_%03d", length(slot_names) + 1L)
        slot_key_to_name[[factor_key]] <- slot_name
        slot_names <- c(slot_names, slot_name)
        slot_key_to_index[[factor_key]] <- length(slot_names)
        slot_level_keys[[slot_name]] <- character(0)
        slot_level_labels[[slot_name]] <- character(0)
        slot_factor_labels[[slot_name]] <- factor_name
      }
      slot_name <- slot_key_to_name[[factor_key]]
      factor_map[[factor_name]] <- list(
        slot_key = factor_key,
        slot_name = slot_name,
        slot_index = as.integer(slot_key_to_index[[factor_key]])
      )

      levels_here <- exp$names_list[[factor_name]][[1]]
      level_key_map <- character(length(levels_here))
      names(level_key_map) <- levels_here
      for (lvl in levels_here) {
        level_key <- cs_foundation_make_level_key(
          experiment_id = exp$experiment_id,
          factor_key = factor_key,
          factor_name = factor_name,
          level_name = lvl,
          canonical_level_id = exp$canonical_level_id[[factor_name]][[lvl]]
        )
        level_key_map[[lvl]] <- level_key
        if (!level_key %in% slot_level_keys[[slot_name]]) {
          slot_level_keys[[slot_name]] <- c(slot_level_keys[[slot_name]], level_key)
          slot_level_labels[[slot_name]] <- c(slot_level_labels[[slot_name]], lvl)
        }
      }
      level_map[[factor_name]] <- level_key_map
    }

    experiment_maps[[exp$experiment_id]] <- list(
      factor_map = factor_map,
      level_map = level_map
    )
  }

  pooled_names_list <- lapply(slot_names, function(slot_name) {
    list(slot_level_keys[[slot_name]])
  })
  names(pooled_names_list) <- slot_names
  slot_keys_ordered <- vapply(slot_names, function(slot_name) {
    names(slot_key_to_name)[match(slot_name, unname(slot_key_to_name))]
  }, character(1))

  slot_table <- data.frame(
    slot_name = slot_names,
    slot_key = slot_keys_ordered,
    display_label = unname(slot_factor_labels[slot_names]),
    stringsAsFactors = FALSE
  )

  list(
    slot_table = slot_table,
    pooled_names_list = pooled_names_list,
    slot_level_keys = slot_level_keys,
    slot_level_labels = slot_level_labels,
    experiment_maps = experiment_maps
  )
}

cs_foundation_build_text_registry <- function(experiments, registry, text_embedding_fn) {
  if (is.null(text_embedding_fn)) {
    return(NULL)
  }
  slot_keys <- registry$slot_table$slot_key
  slot_texts <- registry$slot_table$display_label
  slot_emb <- cs_foundation_text_embed(text_embedding_fn, slot_texts)
  rownames(slot_emb) <- slot_keys

  level_keys <- unlist(registry$slot_level_keys, use.names = FALSE)
  level_labels <- unlist(registry$slot_level_labels, use.names = FALSE)
  level_emb <- cs_foundation_text_embed(text_embedding_fn, level_labels)
  rownames(level_emb) <- level_keys

  list(
    factor_embedding = slot_emb,
    level_embedding = level_emb,
    dim = ncol(slot_emb)
  )
}

cs_foundation_get_embedding_rows <- function(emb_matrix, keys) {
  keys <- as.character(keys)
  if (is.null(emb_matrix) || !length(keys)) {
    return(matrix(numeric(0), nrow = length(keys), ncol = 0L))
  }
  out <- matrix(0, nrow = length(keys), ncol = ncol(emb_matrix))
  rownames(out) <- keys
  colnames(out) <- colnames(emb_matrix)
  matched <- match(keys, rownames(emb_matrix))
  ok <- which(!is.na(matched))
  if (length(ok) > 0L) {
    out[ok, ] <- emb_matrix[matched[ok], , drop = FALSE]
  }
  out
}

cs_foundation_row_semantics <- function(W_df, exp_map, text_registry) {
  if (is.null(text_registry)) {
    return(NULL)
  }
  dim_use <- as.integer(text_registry$dim)
  n <- nrow(W_df)
  factor_sum <- matrix(0, nrow = n, ncol = dim_use)
  level_sum <- matrix(0, nrow = n, ncol = dim_use)
  counts <- integer(n)

  for (factor_name in names(exp_map$factor_map)) {
    factor_meta <- exp_map$factor_map[[factor_name]]
    level_map <- exp_map$level_map[[factor_name]]
    factor_vec <- cs_foundation_get_embedding_rows(
      emb_matrix = text_registry$factor_embedding,
      keys = factor_meta$slot_key
    )
    vals <- as.character(W_df[[factor_name]])
    lvl_keys <- unname(level_map[vals])
    good <- !is.na(lvl_keys)
    if (!any(good)) {
      next
    }
    factor_sum[good, ] <- factor_sum[good, , drop = FALSE] +
      matrix(rep(factor_vec[1, ], each = sum(good)), nrow = sum(good))
    level_sum[good, ] <- level_sum[good, , drop = FALSE] +
      cs_foundation_get_embedding_rows(
        emb_matrix = text_registry$level_embedding,
        keys = lvl_keys[good]
      )
    counts[good] <- counts[good] + 1L
  }

  counts[counts < 1L] <- 1L
  factor_mean <- factor_sum / counts
  level_mean <- level_sum / counts
  out <- cbind(factor_mean, level_mean)
  colnames(out) <- c(
    paste0("semantic_factor_", seq_len(dim_use)),
    paste0("semantic_level_", seq_len(dim_use))
  )
  out
}

cs_foundation_stack_base_x <- function(experiments) {
  base_x_names <- unique(unlist(lapply(experiments, function(exp) {
    if (is.null(exp$X)) {
      character(0)
    } else {
      colnames(exp$X)
    }
  }), use.names = FALSE))
  list(
    base_x_names = base_x_names
  )
}

cs_foundation_build_group_training_data <- function(experiments, registry, control) {
  slot_names <- registry$slot_table$slot_name
  text_registry <- if (isTRUE(control$add_text_semantics)) {
    cs_foundation_build_text_registry(
      experiments = experiments,
      registry = registry,
      text_embedding_fn = control$text_embedding_fn %||% NULL
    )
  } else {
    NULL
  }

  x_schema <- cs_foundation_stack_base_x(experiments)
  experiment_indicator_names <- if (isTRUE(control$add_experiment_indicators) && length(experiments) > 1L) {
    paste0("experiment__", vapply(experiments, `[[`, character(1), "experiment_id"))
  } else {
    character(0)
  }

  W_all <- vector("list", length(experiments))
  Y_all <- vector("list", length(experiments))
  pair_all <- vector("list", length(experiments))
  profile_all <- vector("list", length(experiments))
  respondent_all <- vector("list", length(experiments))
  task_all <- vector("list", length(experiments))
  X_all <- vector("list", length(experiments))

  for (i in seq_along(experiments)) {
    exp <- experiments[[i]]
    exp_map <- registry$experiment_maps[[exp$experiment_id]]

    pooled_W <- matrix(NA_character_, nrow = nrow(exp$W), ncol = length(slot_names))
    colnames(pooled_W) <- slot_names
    pooled_W <- as.data.frame(pooled_W, stringsAsFactors = FALSE)

    for (factor_name in exp$factor_names) {
      factor_meta <- exp_map$factor_map[[factor_name]]
      level_map <- exp_map$level_map[[factor_name]]
      pooled_W[[factor_meta$slot_name]] <- unname(level_map[as.character(exp$W[[factor_name]])])
    }

    base_x <- if (length(x_schema$base_x_names) > 0L) {
      mat <- matrix(0, nrow = nrow(exp$W), ncol = length(x_schema$base_x_names))
      colnames(mat) <- x_schema$base_x_names
      if (!is.null(exp$X) && ncol(exp$X) > 0L) {
        idx <- match(colnames(exp$X), x_schema$base_x_names)
        ok <- which(!is.na(idx))
        if (length(ok) > 0L) {
          mat[, idx[ok]] <- exp$X[, ok, drop = FALSE]
        }
      }
      mat
    } else {
      matrix(0, nrow = nrow(exp$W), ncol = 0L)
    }

    indicator_x <- if (length(experiment_indicator_names) > 0L) {
      mat <- matrix(0, nrow = nrow(exp$W), ncol = length(experiment_indicator_names))
      colnames(mat) <- experiment_indicator_names
      col_idx <- match(paste0("experiment__", exp$experiment_id), experiment_indicator_names)
      if (!is.na(col_idx)) {
        mat[, col_idx] <- 1
      }
      mat
    } else {
      matrix(0, nrow = nrow(exp$W), ncol = 0L)
    }

    semantic_x <- cs_foundation_row_semantics(
      W_df = exp$W,
      exp_map = exp_map,
      text_registry = text_registry
    )
    if (is.null(semantic_x)) {
      semantic_x <- matrix(0, nrow = nrow(exp$W), ncol = 0L)
    }

    X_all[[i]] <- cbind(base_x, indicator_x, semantic_x)
    W_all[[i]] <- pooled_W
    Y_all[[i]] <- exp$Y
    pair_all[[i]] <- if (!is.null(exp$pair_id)) paste(exp$experiment_id, exp$pair_id, sep = "::") else NULL
    profile_all[[i]] <- exp$profile_order %||% NULL
    respondent_all[[i]] <- if (!is.null(exp$respondent_id)) paste(exp$experiment_id, exp$respondent_id, sep = "::") else NULL
    task_all[[i]] <- if (!is.null(exp$respondent_task_id)) paste(exp$experiment_id, exp$respondent_task_id, sep = "::") else NULL
  }

  X_feature_names <- if (length(X_all) > 0L && ncol(X_all[[1]]) > 0L) {
    colnames(X_all[[1]])
  } else {
    character(0)
  }

  list(
    Y = unlist(Y_all, use.names = FALSE),
    W = do.call(rbind, W_all),
    X = if (length(X_feature_names) > 0L) do.call(rbind, X_all) else NULL,
    pair_id = if (all(vapply(pair_all, is.null, logical(1)))) NULL else unlist(pair_all, use.names = FALSE),
    profile_order = if (all(vapply(profile_all, is.null, logical(1)))) NULL else unlist(profile_all, use.names = FALSE),
    respondent_id = if (all(vapply(respondent_all, is.null, logical(1)))) NULL else unlist(respondent_all, use.names = FALSE),
    respondent_task_id = if (all(vapply(task_all, is.null, logical(1)))) NULL else unlist(task_all, use.names = FALSE),
    names_list = registry$pooled_names_list,
    factor_levels = vapply(registry$pooled_names_list, function(x) length(x[[1]]), integer(1)),
    x_feature_names = X_feature_names,
    x_schema = list(
      base_x_names = x_schema$base_x_names,
      experiment_indicator_names = experiment_indicator_names,
      semantic_feature_names = if (!is.null(text_registry)) {
        c(
          paste0("semantic_factor_", seq_len(text_registry$dim)),
          paste0("semantic_level_", seq_len(text_registry$dim))
        )
      } else {
        character(0)
      }
    ),
    text_registry = text_registry
  )
}

cs_foundation_prepare_group_fit <- function(group,
                                            conda_env = "strategize_env",
                                            conda_env_required = TRUE) {
  if (!is.null(group$fit$neural_model_info$params)) {
    return(group)
  }
  if (!"jnp" %in% ls(envir = strenv) || !"np" %in% ls(envir = strenv)) {
    initialize_jax(conda_env = conda_env, conda_env_required = conda_env_required)
  }
  theta_jnp <- strenv$jnp$array(as.numeric(group$fit$theta_mean))$astype(strenv$dtj)
  group$fit$neural_model_info$params <- neural_params_from_theta(theta_jnp, group$fit$neural_model_info)
  group
}

cs_foundation_locscale_tau_name <- function(name) {
  if (grepl("^W_(q|k|v|o)_l[0-9]+$", name) || grepl("^W_ff(1|2)_l[0-9]+$", name)) {
    layer_id <- sub("^.*_l([0-9]+)$", "\\1", name)
    return(paste0("tau_w_", layer_id))
  }
  if (name %in% c("W_q_cross", "W_k_cross", "W_v_cross", "W_o_cross")) {
    return("tau_cross_attn")
  }
  if (identical(name, "W_out")) {
    return("tau_w_out")
  }
  if (identical(name, "b_out")) {
    return("tau_b")
  }
  if (identical(name, "M_cross_raw")) {
    return("tau_cross")
  }
  NULL
}

cs_foundation_add_init_value <- function(init_values, name, value) {
  if (is.null(value)) {
    return(init_values)
  }
  tau_name <- cs_foundation_locscale_tau_name(name)
  init_values[[name]] <- value
  if (!is.null(tau_name)) {
    init_values[[tau_name]] <- 1
    init_values[[paste0(name, "_z")]] <- value
  }
  init_values
}

cs_foundation_build_local_factor_map <- function(experiment) {
  factor_map <- vector("list", length(experiment$factor_names))
  names(factor_map) <- experiment$factor_names
  level_map <- setNames(vector("list", length(experiment$factor_names)), experiment$factor_names)
  for (j in seq_along(experiment$factor_names)) {
    factor_name <- experiment$factor_names[[j]]
    factor_key <- cs_foundation_make_factor_key(
      experiment_id = experiment$experiment_id,
      factor_name = factor_name,
      canonical_factor_id = experiment$canonical_factor_id[[factor_name]]
    )
    factor_map[[factor_name]] <- list(slot_key = factor_key)
    levels_here <- experiment$names_list[[factor_name]][[1]]
    level_key_map <- character(length(levels_here))
    names(level_key_map) <- levels_here
    for (lvl in levels_here) {
      level_key_map[[lvl]] <- cs_foundation_make_level_key(
        experiment_id = experiment$experiment_id,
        factor_key = factor_key,
        factor_name = factor_name,
        level_name = lvl,
        canonical_level_id = experiment$canonical_level_id[[factor_name]][[lvl]]
      )
    }
    level_map[[factor_name]] <- level_key_map
  }
  list(factor_map = factor_map, level_map = level_map)
}

cs_foundation_build_adaptation_x <- function(group,
                                             experiment,
                                             exp_map,
                                             adaptation_control) {
  x_schema <- group$x_schema %||% list(
    base_x_names = character(0),
    experiment_indicator_names = character(0),
    semantic_feature_names = character(0)
  )
  base_names <- x_schema$base_x_names %||% character(0)
  base_x <- if (length(base_names) > 0L) {
    mat <- matrix(0, nrow = nrow(experiment$W), ncol = length(base_names))
    colnames(mat) <- base_names
    if (!is.null(experiment$X) && ncol(experiment$X) > 0L) {
      idx <- match(colnames(experiment$X), base_names)
      ok <- which(!is.na(idx))
      if (length(ok) > 0L) {
        mat[, idx[ok]] <- experiment$X[, ok, drop = FALSE]
      }
    }
    mat
  } else {
    matrix(0, nrow = nrow(experiment$W), ncol = 0L)
  }

  indicator_names <- x_schema$experiment_indicator_names %||% character(0)
  indicator_x <- if (length(indicator_names) > 0L) {
    mat <- matrix(0, nrow = nrow(experiment$W), ncol = length(indicator_names))
    colnames(mat) <- indicator_names
    col_idx <- match(paste0("experiment__", experiment$experiment_id), indicator_names)
    if (!is.na(col_idx)) {
      mat[, col_idx] <- 1
    }
    mat
  } else {
    matrix(0, nrow = nrow(experiment$W), ncol = 0L)
  }

  text_registry <- NULL
  use_text <- isTRUE(adaptation_control$use_text_semantics) &&
    !is.null(group$text_registry)
  if (use_text) {
    if (!is.null(adaptation_control$text_embedding_fn)) {
      text_registry <- cs_foundation_build_text_registry(
        experiments = list(experiment),
        registry = list(
          slot_table = data.frame(
            slot_name = names(exp_map$factor_map),
            slot_key = vapply(exp_map$factor_map, function(x) x$slot_key, character(1)),
            display_label = names(exp_map$factor_map),
            stringsAsFactors = FALSE
          ),
          slot_level_keys = lapply(exp_map$level_map, unname),
          slot_level_labels = lapply(exp_map$level_map, names)
        ),
        text_embedding_fn = adaptation_control$text_embedding_fn
      )
    } else {
      text_registry <- group$text_registry
    }
  }
  semantic_x <- cs_foundation_row_semantics(
    W_df = experiment$W,
    exp_map = exp_map,
    text_registry = text_registry
  )
  if (is.null(semantic_x)) {
    semantic_names <- x_schema$semantic_feature_names %||% character(0)
    semantic_x <- matrix(0, nrow = nrow(experiment$W), ncol = length(semantic_names))
    colnames(semantic_x) <- semantic_names
  }

  X_core <- cbind(base_x, indicator_x, semantic_x)
  extra_x <- matrix(0, nrow = nrow(experiment$W), ncol = 0L)
  if (isTRUE(adaptation_control$allow_extra_covariates) && !is.null(experiment$X)) {
    extras <- setdiff(colnames(experiment$X), colnames(X_core))
    if (length(extras) > 0L) {
      extra_x <- experiment$X[, extras, drop = FALSE]
    }
  }
  out <- cbind(X_core, extra_x)
  if (!ncol(out)) {
    return(NULL)
  }
  out
}

cs_foundation_build_init_site_values <- function(group,
                                                 experiment,
                                                 local_x_feature_names,
                                                 strict_schema_match = FALSE,
                                                 conda_env = "strategize_env",
                                                 conda_env_required = TRUE) {
  group_prepped <- cs_foundation_prepare_group_fit(
    group = group,
    conda_env = conda_env,
    conda_env_required = conda_env_required
  )
  params <- group_prepped$fit$neural_model_info$params
  registry <- group$schema_registry
  exp_map <- cs_foundation_build_local_factor_map(experiment)
  slot_lookup <- setNames(registry$slot_table$slot_name, registry$slot_table$slot_key)
  init_values <- list()

  direct_names <- setdiff(names(params), c(
    grep("^E_factor_[0-9]+$", names(params), value = TRUE),
    "E_feature_id", "E_segment", "W_resp_x", "M_cross"
  ))
  for (name in direct_names) {
    init_values <- cs_foundation_add_init_value(
      init_values = init_values,
      name = name,
      value = cs2step_neural_to_r_array(params[[name]])
    )
  }

  feature_id_src <- if (!is.null(params$E_feature_id)) {
    cs2step_neural_to_r_array(params$E_feature_id)
  } else {
    NULL
  }
  if (!is.null(feature_id_src)) {
    feature_id_tgt <- matrix(0, nrow = length(experiment$factor_names), ncol = ncol(feature_id_src))
    for (j in seq_along(experiment$factor_names)) {
      factor_name <- experiment$factor_names[[j]]
      slot_name <- slot_lookup[[exp_map$factor_map[[factor_name]]$slot_key]] %||% NULL
      if (is.null(slot_name)) {
        if (isTRUE(strict_schema_match)) {
          stop(
            sprintf("No shared slot found for factor '%s' during adaptation.", factor_name),
            call. = FALSE
          )
        }
        next
      }
      src_idx <- match(slot_name, registry$slot_table$slot_name)
      if (!is.na(src_idx) && src_idx <= nrow(feature_id_src)) {
        feature_id_tgt[j, ] <- feature_id_src[src_idx, ]
      }
    }
    init_values[["E_feature_id_raw"]] <- feature_id_tgt
  }

  for (j in seq_along(experiment$factor_names)) {
    factor_name <- experiment$factor_names[[j]]
    slot_name <- slot_lookup[[exp_map$factor_map[[factor_name]]$slot_key]] %||% NULL
    local_levels <- experiment$names_list[[factor_name]][[1]]
    model_dims <- as.integer(group$fit$neural_model_info$model_dims)
    tgt <- matrix(0, nrow = length(local_levels) + 1L, ncol = model_dims)
    if (is.null(slot_name)) {
      if (isTRUE(strict_schema_match)) {
        stop(
          sprintf("No shared slot found for factor '%s' during adaptation.", factor_name),
          call. = FALSE
        )
      }
    } else {
      src_idx <- match(slot_name, registry$slot_table$slot_name)
      src_name <- paste0("E_factor_", src_idx)
      src_val <- params[[src_name]]
      if (!is.null(src_val)) {
        src_mat <- cs2step_neural_to_r_array(src_val)
        src_level_keys <- registry$slot_level_keys[[slot_name]]
        local_level_keys <- exp_map$level_map[[factor_name]]
        for (lvl in local_levels) {
          src_row <- match(local_level_keys[[lvl]], src_level_keys)
          if (!is.na(src_row) && src_row <= nrow(src_mat)) {
            tgt[match(lvl, local_levels), ] <- src_mat[src_row, ]
          }
        }
        if (nrow(src_mat) >= length(src_level_keys) + 1L) {
          tgt[nrow(tgt), ] <- src_mat[nrow(src_mat), ]
        }
      }
    }
    init_values[[paste0("E_factor_", j, "_raw")]] <- tgt
  }

  if (!is.null(params$E_segment)) {
    seg_mat <- cs2step_neural_to_r_array(params$E_segment)
    if (is.matrix(seg_mat) && nrow(seg_mat) >= 2L) {
      init_values[["E_segment_delta"]] <- as.numeric(seg_mat[2, ] - seg_mat[1, ])
    }
  }

  if (!is.null(params$M_cross)) {
    init_values <- cs_foundation_add_init_value(
      init_values = init_values,
      name = "M_cross_raw",
      value = cs2step_neural_to_r_array(params$M_cross)
    )
  }

  if (!is.null(params$W_resp_x) && length(local_x_feature_names) > 0L) {
    src_mat <- cs2step_neural_to_r_array(params$W_resp_x)
    tgt <- matrix(0, nrow = length(local_x_feature_names), ncol = ncol(src_mat))
    rownames(tgt) <- local_x_feature_names
    src_names <- group$x_feature_names %||% character(0)
    idx <- match(src_names, local_x_feature_names)
    ok <- which(!is.na(idx) & seq_along(src_names) <= nrow(src_mat))
    if (length(ok) > 0L) {
      tgt[idx[ok], ] <- src_mat[ok, , drop = FALSE]
    }
    init_values[["W_resp_x"]] <- tgt
  }

  init_values
}

cs_foundation_build_predictor <- function(fit,
                                          mode,
                                          names_list,
                                          factor_levels,
                                          metadata = NULL) {
  structure(
    list(
      model_type = "neural",
      mode = mode,
      encoder = list(
        factor_names = names(names_list),
        names_list = names_list,
        factor_levels = factor_levels,
        unknown_policy = "holdout"
      ),
      fit = fit,
      metadata = modifyList(
        list(
          timestamp = Sys.time(),
          cache_id = sprintf("foundation_adapt_%d", as.integer(stats::runif(1, 1, 1e9)))
        ),
        metadata %||% list()
      )
    ),
    class = "strategic_predictor"
  )
}

cs_foundation_pack_group <- function(group) {
  fit <- group$fit
  group$fit <- list(
    my_model = NULL,
    predict_pair = NULL,
    predict_single = NULL,
    theta_mean = if (!is.null(fit$theta_mean)) as.numeric(fit$theta_mean) else NULL,
    theta_var = if (!is.null(fit$theta_var)) as.numeric(fit$theta_var) else NULL,
    neural_model_info = cs2step_neural_pack_model_info(fit$neural_model_info, drop_params = TRUE),
    fit_metrics = fit$fit_metrics %||% fit$neural_model_info$fit_metrics %||% NULL
  )
  group
}

cs_foundation_unpack_group <- function(group,
                                       conda_env = "strategize_env",
                                       conda_env_required = TRUE,
                                       preload_params = FALSE) {
  bundle <- list(
    model_type = "neural",
    mode = group$mode,
    encoder = group$encoder,
    fit = list(
      theta_mean = group$fit$theta_mean,
      theta_var = group$fit$theta_var,
      neural_model_info = group$fit$neural_model_info,
      fit_metrics = group$fit$fit_metrics
    ),
    metadata = list(
      conda_env = conda_env,
      conda_env_required = conda_env_required
    )
  )
  predictor <- cs2step_unpack_predictor(
    bundle = bundle,
    conda_env = conda_env,
    conda_env_required = conda_env_required,
    preload_params = preload_params
  )
  group$fit <- predictor$fit
  group
}

#' Fit a pooled conjoint foundation model
#'
#' @param experiments List of experiment specifications. Each element must be a
#'   named list with at least \code{experiment_id}, \code{Y}, and \code{W}.
#' @param foundation_control Optional list controlling pooled training. Supported
#'   keys are \code{add_experiment_indicators}, \code{add_text_semantics},
#'   \code{text_embedding_fn}, and \code{neural_mcmc_control}.
#' @param conda_env Conda env name for the neural backend.
#' @param conda_env_required Require the conda env to exist.
#' @param cache_path Optional path to a cached foundation bundle.
#' @param cache_overwrite Logical; overwrite any existing cache at \code{cache_path}.
#' @param cache_compress Compression passed to \code{saveRDS()}.
#' @return An object of class \code{conjoint_foundation_model}.
#'
#' @details
#' Run \code{\link{build_backend}()} once before calling this function. The
#' neural foundation workflow relies on the same JAX backend as the package's
#' neural outcome model.
#'
#' Each element of \code{experiments} is a per-study specification with the
#' following contract.
#'
#' Required fields:
#' \describe{
#'   \item{\code{experiment_id}}{A unique study identifier.}
#'   \item{\code{Y}}{Outcome vector with \code{length(Y) == nrow(W)}.}
#'   \item{\code{W}}{A data frame or matrix of factor columns.}
#' }
#'
#' Optional fields:
#' \describe{
#'   \item{\code{mode}}{\code{"auto"}, \code{"pairwise"}, or \code{"single"}.
#'   Defaults to \code{"pairwise"} when \code{pair_id} is supplied and
#'   \code{"single"} otherwise.}
#'   \item{\code{pair_id}}{Required for pairwise studies. Each pair id must
#'   appear exactly twice.}
#'   \item{\code{profile_order}}{Optional within-pair ordering, typically
#'   \code{1}/\code{2}.}
#'   \item{\code{X}}{Optional numeric covariates aligned row-wise to
#'   \code{W}. Non-numeric columns are rejected.}
#'   \item{\code{respondent_id}, \code{respondent_task_id}}{Optional row-aligned
#'   respondent/task identifiers used when available for clustering and
#'   evaluation logic.}
#'   \item{\code{likelihood}}{\code{"auto"}, \code{"bernoulli"},
#'   \code{"categorical"}, or \code{"normal"}.}
#'   \item{\code{n_outcomes}}{Required only when forcing a categorical
#'   likelihood. In v1 it must match the number of observed classes in the
#'   study. Categorical outcomes are internally normalized to zero-based class
#'   ids before neural fitting.}
#'   \item{\code{names_list}, \code{p_list}}{Optional factor-level metadata used
#'   to define the local level universe for each factor.}
#'   \item{\code{canonical_factor_id}}{Optional named vector or list that forces
#'   cross-study sharing of factor identities when explicitly provided.}
#'   \item{\code{canonical_level_id}}{Optional named list that forces
#'   cross-study sharing of level identities when explicitly provided.}
#' }
#'
#' Pooled training groups experiments by compatible neural family:
#' \code{(mode, likelihood, n_outcomes)}. The returned object may therefore hold
#' multiple internal foundation groups when the input studies mix pairwise and
#' single designs or mix Bernoulli, categorical, and Gaussian outcomes.
#'
#' Schema sharing rules are conservative:
#' \itemize{
#'   \item explicit canonical ids force sharing;
#'   \item absent factors in a pooled schema are routed to holdout rows during
#'   training;
#'   \item raw text equality alone does not merge schema elements.
#' }
#'
#' The \code{foundation_control} list supports:
#' \describe{
#'   \item{\code{add_experiment_indicators}}{Whether to append experiment
#'   one-hot indicators to the pooled covariate matrix. Default \code{TRUE}.}
#'   \item{\code{add_text_semantics}}{Whether to add pooled semantic side
#'   features when \code{text_embedding_fn} is supplied. Default \code{TRUE}.}
#'   \item{\code{text_embedding_fn}}{Optional function that maps character input
#'   to numeric embeddings. It may accept a character vector and return a matrix
#'   with matching rows, or accept one string at a time and return a fixed-width
#'   numeric vector. These embeddings are side information, not the primary
#'   identity mechanism.}
#'   \item{\code{neural_mcmc_control}}{Optional list passed to the existing
#'   neural outcome backend. Defaults to a pooled SVI configuration using
#'   output-only uncertainty.}
#' }
#'
#' @examples
#' \dontrun{
#' library(strategize)
#'
#' build_backend(conda_env = "strategize_env")
#'
#' study_a <- list(
#'   experiment_id = "study_a",
#'   Y = c(1, 0, 0, 1),
#'   W = data.frame(
#'     price = c("Low", "Low", "High", "High"),
#'     message = c("Jobs", "Taxes", "Jobs", "Taxes")
#'   ),
#'   pair_id = c(1, 1, 2, 2),
#'   profile_order = c(1, 2, 1, 2),
#'   canonical_factor_id = c(price = "price", message = "message")
#' )
#'
#' study_b <- list(
#'   experiment_id = "study_b",
#'   Y = c(0, 1, 1, 0),
#'   W = data.frame(
#'     price = c("Low", "High", "Low", "High"),
#'     message = c("Jobs", "Jobs", "Taxes", "Taxes"),
#'     messenger = c("Local", "Local", "National", "National")
#'   ),
#'   pair_id = c(1, 1, 2, 2),
#'   profile_order = c(1, 2, 1, 2),
#'   canonical_factor_id = c(
#'     price = "price",
#'     message = "message",
#'     messenger = "messenger"
#'   )
#' )
#'
#' foundation_fit <- fit_conjoint_foundation_model(
#'   experiments = list(study_a, study_b),
#'   foundation_control = list(
#'     neural_mcmc_control = list(
#'       ModelDims = 32L,
#'       ModelDepth = 1L,
#'       subsample_method = "batch_vi",
#'       uncertainty_scope = "output",
#'       svi_steps = 100L
#'     )
#'   )
#' )
#' }
#'
#' @seealso \code{\link{adapt_conjoint_foundation_model}()},
#'   \code{\link{save_conjoint_foundation_bundle}()},
#'   \code{\link{load_conjoint_foundation_bundle}()},
#'   \code{\link{build_backend}()}
#' @export
fit_conjoint_foundation_model <- function(experiments,
                                          foundation_control = NULL,
                                          conda_env = "strategize_env",
                                          conda_env_required = TRUE,
                                          cache_path = NULL,
                                          cache_overwrite = FALSE,
                                          cache_compress = TRUE) {
  if (!is.null(cache_path)) {
    cache_path <- as.character(cache_path)
    if (length(cache_path) != 1L || !nzchar(cache_path)) {
      stop("'cache_path' must be a non-empty character path.", call. = FALSE)
    }
    if (!isTRUE(cache_overwrite) && file.exists(cache_path)) {
      return(load_conjoint_foundation_bundle(
        file = cache_path,
        conda_env = conda_env,
        conda_env_required = conda_env_required,
        preload_params = FALSE
      ))
    }
  }
  if (!is.list(experiments) || length(experiments) < 1L) {
    stop("'experiments' must be a non-empty list.", call. = FALSE)
  }

  control <- modifyList(cs_foundation_default_control(), foundation_control %||% list())
  experiments_norm <- lapply(seq_along(experiments), function(i) {
    cs_foundation_normalize_experiment(experiments[[i]], index = i)
  })

  group_keys <- vapply(experiments_norm, function(exp) {
    cs_foundation_group_key(exp$mode, exp$likelihood, exp$n_outcomes)
  }, character(1))
  experiment_groups <- split(experiments_norm, group_keys)

  groups_out <- lapply(names(experiment_groups), function(group_key) {
    exps <- experiment_groups[[group_key]]
    group_meta <- exps[[1]]
    registry <- cs_foundation_build_group_registry(exps)
    pooled <- cs_foundation_build_group_training_data(exps, registry, control)
    enc <- cs_encode_W_indices(
      W = pooled$W,
      names_list = pooled$names_list,
      unknown = "holdout",
      align = "by_name"
    )
    fit <- cs2step_eval_outcome_model_neural(
      Y = pooled$Y,
      W_idx = enc$W_idx,
      factor_levels = pooled$factor_levels,
      diff = identical(group_meta$mode, "pairwise"),
      pair_id = pooled$pair_id,
      profile_order = pooled$profile_order,
      X = pooled$X,
      respondent_id = pooled$respondent_id,
      respondent_task_id = pooled$respondent_task_id,
      likelihood_override = group_meta$likelihood,
      n_outcomes_override = if (identical(group_meta$likelihood, "categorical")) group_meta$n_outcomes else NULL,
      conda_env = conda_env,
      conda_env_required = conda_env_required,
      neural_mcmc_control = control$neural_mcmc_control %||% NULL
    )
    names_list_group <- pooled$names_list
    factor_levels_group <- pooled$factor_levels

    list(
      group_key = group_key,
      mode = group_meta$mode,
      likelihood = group_meta$likelihood,
      n_outcomes = as.integer(group_meta$n_outcomes),
      experiment_ids = vapply(exps, `[[`, character(1), "experiment_id"),
      encoder = list(
        factor_names = names(names_list_group),
        names_list = names_list_group,
        factor_levels = factor_levels_group,
        unknown_policy = "holdout"
      ),
      schema_registry = registry,
      x_feature_names = pooled$x_feature_names,
      x_schema = pooled$x_schema,
      text_registry = pooled$text_registry,
      fit = fit
    )
  })
  names(groups_out) <- names(experiment_groups)

  out <- structure(
    list(
      schema_version = 1L,
      model_type = "conjoint_foundation",
      groups = groups_out,
      metadata = list(
        created_at = Sys.time(),
        conda_env = conda_env,
        conda_env_required = conda_env_required,
        experiment_ids = vapply(experiments_norm, `[[`, character(1), "experiment_id"),
        grouping_note = "Experiments are pooled within compatible mode/likelihood families."
      )
    ),
    class = "conjoint_foundation_model"
  )

  if (!is.null(cache_path)) {
    save_conjoint_foundation_bundle(
      file = cache_path,
      foundation_model = out,
      overwrite = TRUE,
      compress = cache_compress
    )
  }
  out
}

cs_foundation_match_group <- function(foundation_model, mode, likelihood, n_outcomes) {
  key <- cs_foundation_group_key(mode, likelihood, n_outcomes)
  group <- foundation_model$groups[[key]] %||% NULL
  if (is.null(group)) {
    stop(
      sprintf(
        "No compatible foundation group found for mode='%s', likelihood='%s', n_outcomes=%d.",
        mode, likelihood, as.integer(n_outcomes)
      ),
      call. = FALSE
    )
  }
  group
}

#' Adapt a pooled conjoint foundation model to a single study
#'
#' @param foundation_model A fitted \code{conjoint_foundation_model}.
#' @param Y Outcome vector.
#' @param W Factor matrix/data.frame.
#' @param X Optional numeric covariates.
#' @param mode \code{"auto"}, \code{"pairwise"}, or \code{"single"}.
#' @param pair_id Optional pair identifiers.
#' @param profile_order Optional within-pair ordering.
#' @param experiment_id Optional experiment identifier for the adaptation study.
#' @param names_list Optional factor level names.
#' @param p_list Optional \code{p_list}.
#' @param respondent_id Optional respondent identifiers.
#' @param respondent_task_id Optional respondent-task identifiers.
#' @param likelihood Optional likelihood override.
#' @param n_outcomes Optional categorical outcome count.
#' @param canonical_factor_id Optional factor-level sharing ids.
#' @param canonical_level_id Optional level-level sharing ids.
#' @param neural_mcmc_control Optional list passed to the Bayesian neural backend.
#' @param foundation_adaptation_control Optional list controlling adaptation.
#'   Supported keys are \code{strict_schema_match}, \code{allow_extra_covariates},
#'   \code{use_text_semantics}, and \code{text_embedding_fn}.
#' @param conda_env Conda env name for the neural backend.
#' @param conda_env_required Require the conda env to exist.
#' @param cache_path Optional predictor cache path.
#' @param cache_overwrite Logical; overwrite any existing cache at \code{cache_path}.
#' @param cache_compress Compression passed to \code{saveRDS()}.
#' @return A \code{strategic_predictor}.
#'
#' @details
#' Adaptation reuses the package's existing Bayesian neural outcome model rather
#' than fitting a separate downstream architecture. Internally, this function:
#'
#' \enumerate{
#'   \item normalizes the target study into the same local schema representation
#'   used by the neural outcome model,
#'   \item finds the compatible foundation group matching
#'   \code{(mode, likelihood, n_outcomes)},
#'   \item builds warm-start values from the pooled foundation parameters, and
#'   \item runs the current Bayesian neural fit with those warm starts.
#' }
#'
#' Group matching is exact. Adaptation fails if the foundation object does not
#' contain a compatible internal family for the requested target study.
#'
#' The target-study data contract mirrors the per-experiment contract used by
#' \code{\link{fit_conjoint_foundation_model}()}:
#' \itemize{
#'   \item \code{length(Y) == nrow(W)};
#'   \item pairwise studies require \code{pair_id};
#'   \item \code{X}, when supplied, must be numeric and row-aligned to
#'   \code{W};
#'   \item categorical adaptation follows the same \code{n_outcomes} rule as
#'   pooled training.
#' }
#'
#' Schema reuse is partial by design:
#' \itemize{
#'   \item matched factors and levels inherit warm-started embeddings from the
#'   foundation fit;
#'   \item unmatched local schema elements fall back to local initialization;
#'   \item if \code{foundation_adaptation_control$strict_schema_match = TRUE},
#'   unmatched factors cause an error instead of falling back.
#' }
#'
#' The \code{foundation_adaptation_control} list supports:
#' \describe{
#'   \item{\code{strict_schema_match}}{Require every local factor to match a
#'   pooled foundation slot. Default \code{FALSE}.}
#'   \item{\code{allow_extra_covariates}}{Allow local covariates that were not
#'   present during pooled training. Extra columns are appended after the shared
#'   pooled covariate schema. Default \code{TRUE}.}
#'   \item{\code{use_text_semantics}}{Reuse semantic side features during
#'   adaptation when the foundation object includes them. Default
#'   \code{TRUE}.}
#'   \item{\code{text_embedding_fn}}{Optional embedding function used to rebuild
#'   semantic side information for the target study. When omitted, adaptation
#'   reuses the stored pooled semantic registry when available.}
#' }
#'
#' @examples
#' \dontrun{
#' library(strategize)
#'
#' build_backend(conda_env = "strategize_env")
#'
#' foundation_fit <- fit_conjoint_foundation_model(
#'   experiments = list(
#'     list(
#'       experiment_id = "study_a",
#'       Y = c(1, 0, 0, 1),
#'       W = data.frame(
#'         price = c("Low", "Low", "High", "High"),
#'         message = c("Jobs", "Taxes", "Jobs", "Taxes")
#'       ),
#'       pair_id = c(1, 1, 2, 2),
#'       profile_order = c(1, 2, 1, 2),
#'       canonical_factor_id = c(price = "price", message = "message")
#'     )
#'   )
#' )
#'
#' adapted_fit <- adapt_conjoint_foundation_model(
#'   foundation_model = foundation_fit,
#'   Y = c(1, 0, 0, 1),
#'   W = data.frame(
#'     price = c("Low", "High", "Low", "High"),
#'     message = c("Jobs", "Jobs", "Taxes", "Taxes")
#'   ),
#'   mode = "pairwise",
#'   pair_id = c(1, 1, 2, 2),
#'   profile_order = c(1, 2, 1, 2),
#'   experiment_id = "target_study",
#'   canonical_factor_id = c(price = "price", message = "message")
#' )
#'
#' predict(
#'   adapted_fit,
#'   newdata = list(
#'     W = data.frame(
#'       price = c("Low", "High"),
#'       message = c("Jobs", "Taxes")
#'     ),
#'     pair_id = c(1, 1),
#'     profile_order = c(1, 2)
#'   )
#' )
#' }
#'
#' @seealso \code{\link{fit_conjoint_foundation_model}()},
#'   \code{\link{build_backend}()}
#' @export
adapt_conjoint_foundation_model <- function(foundation_model,
                                            Y,
                                            W,
                                            X = NULL,
                                            mode = c("auto", "pairwise", "single"),
                                            pair_id = NULL,
                                            profile_order = NULL,
                                            experiment_id = "adaptation_target",
                                            names_list = NULL,
                                            p_list = NULL,
                                            respondent_id = NULL,
                                            respondent_task_id = NULL,
                                            likelihood = NULL,
                                            n_outcomes = NULL,
                                            canonical_factor_id = NULL,
                                            canonical_level_id = NULL,
                                            neural_mcmc_control = NULL,
                                            foundation_adaptation_control = NULL,
                                            conda_env = "strategize_env",
                                            conda_env_required = TRUE,
                                            cache_path = NULL,
                                            cache_overwrite = FALSE,
                                            cache_compress = TRUE) {
  if (!inherits(foundation_model, "conjoint_foundation_model")) {
    stop("'foundation_model' must be a conjoint_foundation_model.", call. = FALSE)
  }
  if (!is.null(cache_path)) {
    cache_path <- as.character(cache_path)
    if (length(cache_path) != 1L || !nzchar(cache_path)) {
      stop("'cache_path' must be a non-empty character path.", call. = FALSE)
    }
    if (!isTRUE(cache_overwrite) && file.exists(cache_path)) {
      return(load_strategic_predictor(
        cache_path,
        conda_env = conda_env,
        conda_env_required = conda_env_required
      ))
    }
  }

  experiment <- cs_foundation_normalize_experiment(
    experiment = list(
      experiment_id = experiment_id,
      Y = Y,
      W = W,
      X = X,
      mode = match.arg(mode),
      pair_id = pair_id,
      profile_order = profile_order,
      names_list = names_list,
      p_list = p_list,
      respondent_id = respondent_id,
      respondent_task_id = respondent_task_id,
      likelihood = likelihood,
      n_outcomes = n_outcomes,
      canonical_factor_id = canonical_factor_id,
      canonical_level_id = canonical_level_id
    ),
    index = 1L
  )
  group <- cs_foundation_match_group(
    foundation_model = foundation_model,
    mode = experiment$mode,
    likelihood = experiment$likelihood,
    n_outcomes = experiment$n_outcomes
  )

  adaptation_control <- modifyList(
    cs_foundation_default_adaptation_control(),
    foundation_adaptation_control %||% list()
  )
  local_map <- cs_foundation_build_local_factor_map(experiment)
  X_aug <- cs_foundation_build_adaptation_x(
    group = group,
    experiment = experiment,
    exp_map = local_map,
    adaptation_control = adaptation_control
  )
  local_names_list <- experiment$names_list
  enc <- cs_encode_W_indices(
    W = experiment$W,
    names_list = local_names_list,
    unknown = "error",
    align = "by_name"
  )
  local_x_feature_names <- if (!is.null(X_aug)) colnames(X_aug) else character(0)
  init_site_values <- cs_foundation_build_init_site_values(
    group = group,
    experiment = experiment,
    local_x_feature_names = local_x_feature_names,
    strict_schema_match = isTRUE(adaptation_control$strict_schema_match),
    conda_env = conda_env,
    conda_env_required = conda_env_required
  )

  fit_control <- neural_mcmc_control %||% list()
  fit_control <- modifyList(list(init_site_values = init_site_values), fit_control)

  fit <- cs2step_eval_outcome_model_neural(
    Y = experiment$Y,
    W_idx = enc$W_idx,
    factor_levels = experiment$factor_levels,
    diff = identical(experiment$mode, "pairwise"),
    pair_id = experiment$pair_id,
    profile_order = experiment$profile_order,
    X = X_aug,
    respondent_id = experiment$respondent_id,
    respondent_task_id = experiment$respondent_task_id,
    likelihood_override = experiment$likelihood,
    n_outcomes_override = if (identical(experiment$likelihood, "categorical")) experiment$n_outcomes else NULL,
    conda_env = conda_env,
    conda_env_required = conda_env_required,
    neural_mcmc_control = fit_control
  )

  out <- cs_foundation_build_predictor(
    fit = fit,
    mode = experiment$mode,
    names_list = local_names_list,
    factor_levels = experiment$factor_levels,
    metadata = list(
      call = match.call(),
      conda_env = conda_env,
      conda_env_required = conda_env_required,
      foundation_group_key = group$group_key,
      foundation_experiment_ids = group$experiment_ids,
      adaptation_experiment_id = experiment$experiment_id
    )
  )
  if (!is.null(cache_path)) {
    save_strategic_predictor(
      fit = out,
      file = cache_path,
      overwrite = TRUE,
      compress = cache_compress
    )
  }
  out
}

#' Save a conjoint foundation bundle
#'
#' @param file Path to save the bundle.
#' @param foundation_model A fitted \code{conjoint_foundation_model}.
#' @param overwrite Logical; overwrite any existing file.
#' @param compress Compression passed to \code{saveRDS()}.
#' @return The bundle path (invisibly).
#'
#' @details
#' Foundation bundles store the packed neural metadata and posterior summaries
#' for each internal foundation group, together with the pooled schema registry
#' and adaptation metadata. They do not store active JIT functions or a live
#' Python session.
#'
#' @examples
#' \dontrun{
#' tmp <- tempfile(fileext = ".rds")
#' save_conjoint_foundation_bundle(tmp, foundation_model = foundation_fit, overwrite = TRUE)
#' }
#'
#' @seealso \code{\link{load_conjoint_foundation_bundle}()}
#' @export
save_conjoint_foundation_bundle <- function(file,
                                            foundation_model,
                                            overwrite = FALSE,
                                            compress = TRUE) {
  if (!inherits(foundation_model, "conjoint_foundation_model")) {
    stop("'foundation_model' must be a conjoint_foundation_model.", call. = FALSE)
  }
  file <- as.character(file)
  if (length(file) != 1L || !nzchar(file)) {
    stop("'file' must be a non-empty path.", call. = FALSE)
  }
  if (file.exists(file) && !isTRUE(overwrite)) {
    stop("Bundle file already exists; set overwrite = TRUE to replace it.", call. = FALSE)
  }
  bundle <- foundation_model
  bundle$groups <- lapply(bundle$groups, cs_foundation_pack_group)
  class(bundle) <- c("conjoint_foundation_bundle", "list")
  dir.create(dirname(file), recursive = TRUE, showWarnings = FALSE)
  saveRDS(bundle, file = file, compress = compress)
  invisible(file)
}

#' Load a conjoint foundation bundle
#'
#' @param file Path to a bundle created by \code{save_conjoint_foundation_bundle()}.
#' @param conda_env Conda env name for the neural backend.
#' @param conda_env_required Require the conda env to exist.
#' @param preload_params Logical; reconstruct neural params immediately.
#' @return A \code{conjoint_foundation_model}.
#'
#' @details
#' Loading reconstructs the packed foundation object and, when
#' \code{preload_params = TRUE}, immediately materializes the neural parameter
#' arrays for each internal foundation group. Leave \code{preload_params} as
#' \code{FALSE} when you only need metadata or plan to adapt later.
#'
#' @examples
#' \dontrun{
#' foundation_fit <- load_conjoint_foundation_bundle("foundation_bundle.rds")
#' }
#'
#' @seealso \code{\link{save_conjoint_foundation_bundle}()}
#' @export
load_conjoint_foundation_bundle <- function(file,
                                            conda_env = "strategize_env",
                                            conda_env_required = TRUE,
                                            preload_params = FALSE) {
  file <- as.character(file)
  if (length(file) != 1L || !nzchar(file)) {
    stop("'file' must be a non-empty path.", call. = FALSE)
  }
  if (!file.exists(file)) {
    stop("Bundle file does not exist.", call. = FALSE)
  }
  bundle <- readRDS(file)
  if (!is.list(bundle) || is.null(bundle$groups)) {
    stop("Unrecognized conjoint foundation bundle format.", call. = FALSE)
  }
  bundle$groups <- lapply(bundle$groups, cs_foundation_unpack_group,
                          conda_env = conda_env,
                          conda_env_required = conda_env_required,
                          preload_params = preload_params)
  class(bundle) <- "conjoint_foundation_model"
  bundle
}
