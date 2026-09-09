cs_correctness_contract <- function() "strategize_correctness_20260909_v2"

cs_policy_design_diagnostics <- function(policy, p_list, n_oriented) {
  log_second <- log_max <- 0
  for (name in names(p_list)) {
    p <- as.numeric(p_list[[name]])
    pi <- as.numeric(policy[[name]][names(p_list[[name]])])
    if (length(pi) != length(p) || any(!is.finite(c(pi, p))) ||
        any(p < 0 | pi < 0) || abs(sum(pi) - 1) > 1e-7 ||
        any(pi[p == 0] > 0)) stop("Policy violates assignment support.", call. = FALSE)
    keep <- p > 0
    log_second <- log_second + log(sum(pi[keep]^2 / p[keep]))
    log_max <- log_max + log(max(pi[keep] / p[keep]))
  }
  data.frame(design_population_ess_fraction = exp(-log_second),
             design_absolute_ess = n_oriented * exp(-log_second),
             design_max_possible_weight = exp(log_max),
             design_d2 = log_second, design_dinf = log_max)
}

cs_policy_design_pass <- function(d, control) {
  all(is.finite(unlist(d))) &&
    d$design_population_ess_fraction >= control$design_ess_fraction_min &&
    d$design_absolute_ess >= control$design_abs_ess_min &&
    d$design_max_possible_weight <= control$design_max_weight
}

cs_crossfit_select_policy <- function(train_args, control, fold, n_oriented,
                                       fit = strategize) {
  adaptive <- isTRUE(control$adaptive_lambda)
  path <- if (adaptive) sort(unique(c(train_args$lambda, control$lambda_path))) else train_args$lambda
  candidates <- list()
  for (i in seq_along(path)) {
    train_args$lambda <- path[[i]]
    # Keep outcome fitting/screening randomness identical across penalties.
    # Neither candidate fitting nor selection receives heldout outcomes.
    set.seed(as.integer(control$seed + 1009L * fold))
    result <- do.call(fit, train_args)
    d <- cs_policy_design_diagnostics(cs_crossfit_q_extract_policy(result),
                                      train_args$p_list, n_oriented)
    pass <- cs_policy_design_pass(d, control)
    candidates[[i]] <- data.frame(fold = fold, lambda = path[[i]],
                                   design_constraints_pass = pass, d)
    if (isTRUE(pass) || !adaptive) break
  }
  list(result = result, candidates = do.call(rbind, candidates),
       info = c(list(selected_lambda = path[[i]], adaptive_lambda = adaptive,
                     lambda_selection_pass = if (adaptive) pass else NA,
                     lambda_selection_index = i,
                     lambda_selection_reason = if (!adaptive) "fixed_lambda" else
                       if (pass) "passed_design_constraints" else "no_lambda_passed",
                     design_constraints_pass = pass), as.list(d[1L, ])))
}

# Marginal probability contrasts from the SAME training-fold model used for
# learned-policy OPE. Common random numbers integrate other factors and the
# opponent under the assignment design; no heldout outcomes are used.
cs_crossfit_amce_policies <- function(result, p_list, control, fold, n_oriented) {
  n <- control$n_policy_draws
  a <- cs_crossfit_q_sample_policy(p_list, n, seed = control$seed + 7919L * fold)
  b <- cs_crossfit_q_sample_policy(p_list, n, seed = control$seed + 7919L * fold + 1L)
  utilities <- lapply(names(p_list), function(name) {
    u <- vapply(names(p_list[[name]]), function(level) {
      a[[name]] <- level
      mean((cs_crossfit_q_pair_predict(a, b, result, p_list) +
              1 - cs_crossfit_q_pair_predict(b, a, result, p_list)) / 2)
    }, numeric(1))
    stats::setNames(u - sum(u * p_list[[name]]), names(p_list[[name]]))
  })
  names(utilities) <- names(p_list)
  hard <- Map(function(u, p) {
    tied <- p > 0 & abs(u - max(u[p > 0])) <= 1e-12
    stats::setNames(as.numeric(tied) / sum(tied), names(u))
  }, utilities, p_list)
  info <- function(policy, policy_name, tau = NA_real_) {
    d <- cs_policy_design_diagnostics(policy, p_list, n_oriented)
    pass <- cs_policy_design_pass(d, control)
    list(policy = policy, info = c(list(
      policy_name = policy_name, adaptive_lambda = FALSE,
      lambda_selection_pass = NA, selected_lambda = NA_real_,
      amce_tau = tau, amce_selection_pass = pass,
      amce_selection_reason = if (pass) "passed_design_constraints" else "failed_design_constraints",
      design_constraints_pass = pass), as.list(d[1L, ])))
  }
  max_policy <- info(hard, "amce_max")
  candidates <- list()
  for (tau in sort(unique(control$amce_tau_grid), decreasing = TRUE)) {
    soft <- Map(function(u, p) {
      # Tilt the actual assignment distribution, so tau=0 is the reference.
      z <- tau * u
      q <- p * exp(z - max(z))
      q / sum(q)
    }, utilities, p_list)
    selected <- info(soft, "amce_soft", tau)
    candidates[[length(candidates) + 1L]] <- data.frame(
      fold = fold, tau = tau,
      design_constraints_pass = selected$info$design_constraints_pass)
    if (isTRUE(selected$info$design_constraints_pass)) break
  }
  list(amce_max = max_policy, amce_soft = selected,
       candidates = do.call(rbind, candidates))
}

cs_crossfit_selection_summary <- function(folds) {
  num <- function(name, f) {
    if (!name %in% names(folds) || any(!is.finite(folds[[name]]))) return(NA_real_)
    f(folds[[name]])
  }
  flag <- function(name) {
    if (!name %in% names(folds) || anyNA(folds[[name]])) return(NA)
    all(folds[[name]])
  }
  list(selected_lambda_min = num("selected_lambda", min),
       selected_lambda_median = num("selected_lambda", stats::median),
       selected_lambda_max = num("selected_lambda", max),
       selected_lambda_mean = num("selected_lambda", mean),
       amce_tau_min = num("amce_tau", min),
       amce_tau_median = num("amce_tau", stats::median),
       amce_tau_max = num("amce_tau", max),
       lambda_selection_failures = if ("lambda_selection_pass" %in% names(folds) && !anyNA(folds$lambda_selection_pass)) sum(!folds$lambda_selection_pass) else NA_integer_,
       adaptive_lambda = flag("adaptive_lambda"),
       lambda_selection_pass = flag("lambda_selection_pass"),
       design_constraints_pass = flag("design_constraints_pass"),
       amce_selection_pass = flag("amce_selection_pass"),
       design_population_ess_fraction_min = num("design_population_ess_fraction", min),
       design_absolute_ess_min = num("design_absolute_ess", min),
       design_max_possible_weight_max = num("design_max_possible_weight", max),
       design_d2_max = num("design_d2", max), design_dinf_max = num("design_dinf", max))
}

cs_crossfit_rbind_fill <- function(dfs) {
  fields <- unique(unlist(lapply(dfs, names)))
  do.call(rbind, lapply(dfs, function(d) {
    for (name in setdiff(fields, names(d))) d[[name]] <- NA
    d[, fields, drop = FALSE]
  }))
}

cs_crossfit_benchmark_summary <- function(folds, control) {
  if (!length(folds)) return(NULL)
  df <- cs_crossfit_rbind_fill(folds)
  rows <- lapply(split(df, paste(df$policy_name, df$estimator, sep = "::")), function(d) {
    n <- d$n_oriented
    out <- data.frame(policy_name = d$policy_name[[1L]], estimator = d$estimator[[1L]],
      Q_crossfit = stats::weighted.mean(d$Q_crossfit, n),
      Q_reference_crossfit = stats::weighted.mean(d$Q_reference_crossfit, n),
      Q_gain_crossfit = stats::weighted.mean(d$Q_gain_crossfit, n),
      n_oriented = sum(n), n_folds = nrow(d), ess = sum(d$ess),
      mean_ess_fraction = mean(d$ess_fraction), max_weight = max(d$max_weight),
      mean_weight = sum(d$weight_sum) / sum(n),
      weight_mean = sum(d$weight_sum) / sum(n), weight_sum_ratio = sum(d$weight_sum) / sum(n),
      hajek_denominator_ok = all(d$hajek_denominator_ok))
    cbind(out, as.data.frame(cs_crossfit_selection_summary(d)))
  })
  summary <- do.call(rbind, rows)
  list(summary = summary, headline = summary[summary$estimator == control$headline, ],
       folds = if (control$return_fold_results) df else NULL,
       contributions = if (control$return_fold_results)
         do.call(rbind, lapply(folds, attr, which = "contributions")) else NULL,
       provenance = "same_primary_training_folds_v2",
       control = list(tau_grid = control$amce_tau_grid, headline = control$headline))
}
