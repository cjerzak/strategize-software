suppressPackageStartupMessages({
  library(devtools)
  library(withr)
})

`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}

repo_root <- normalizePath(".", winslash = "/", mustWork = TRUE)
pkg_root <- file.path(repo_root, "strategize")

devtools::load_all(pkg_root, quiet = TRUE, export_all = FALSE, helpers = FALSE)
sys.source(file.path(pkg_root, "tests", "testthat", "helper-strategize.R"), envir = globalenv())

get_neural_model_info_local <- function(res) {
  model_info <- res$neural_model_info$ast
  if (is.null(model_info)) model_info <- res$neural_model_info$dag
  if (is.null(model_info)) model_info <- res$neural_model_info$ast0
  if (is.null(model_info)) model_info <- res$neural_model_info$dag0
  model_info
}

extract_average_case_pi_hat_local <- function(res) {
  pi_obj <- res$pi_star_point
  if (is.list(pi_obj) && length(pi_obj) == 1L && is.list(pi_obj[[1L]])) {
    pi_obj <- pi_obj[[1L]]
  }
  vapply(pi_obj, function(prob_vec) {
    if (!is.null(names(prob_vec)) && "1" %in% names(prob_vec)) {
      return(as.numeric(prob_vec[["1"]]))
    }
    as.numeric(prob_vec[[2L]])
  }, numeric(1))
}

extract_average_case_neural_mu_hat_local <- function(res, W) {
  model <- res$Y_models$my_model_ast_jnp
  if (is.null(model)) {
    model <- res$Y_models$my_model_dag_jnp
  }
  if (!is.function(model)) {
    stop("Neural average-case fit did not expose a prediction function.", call. = FALSE)
  }

  W_df <- as.data.frame(W, stringsAsFactors = FALSE)
  if (!is.null(names(res$p_list)) && !is.null(colnames(W_df))) {
    W_df <- W_df[, names(res$p_list), drop = FALSE]
  }

  W_num <- as.matrix(vapply(seq_along(res$p_list), function(d_) {
    level_names <- names(res$p_list[[d_]])
    match(as.character(W_df[[d_]]), level_names)
  }, numeric(nrow(W_df))))

  pred <- model(X_new = W_num)
  if (is.list(pred) && !is.null(pred$mu)) {
    return(as.numeric(pred$mu))
  }
  as.numeric(pred)
}

compute_binary_null_metrics_local <- function(y) {
  y <- as.numeric(y)
  y <- y[is.finite(y)]
  p_null <- mean(y)
  p_null <- min(max(p_null, 1e-6), 1 - 1e-6)
  list(
    log_loss = -mean(y * log(p_null) + (1 - y) * log(1 - p_null)),
    accuracy = max(mean(y), 1 - mean(y)),
    brier = mean((p_null - y) ^ 2)
  )
}

safe_num <- function(x) {
  if (is.null(x) || !length(x)) {
    return(NA_real_)
  }
  as.numeric(x[[1L]])
}

model_label <- function(residual_mode) {
  if (identical(residual_mode, "full_attn")) {
    "full_attn_residual"
  } else {
    "baseline_standard"
  }
}

run_pairwise_oos_benchmark <- function(residual_mode, depth = 4L) {
  withr::with_envvar(c(
    STRATEGIZE_NEURAL_FAST_MCMC = "true",
    STRATEGIZE_NEURAL_EVAL_FOLDS = "2",
    STRATEGIZE_NEURAL_EVAL_SEED = "123"
  ), {
    data <- generate_pairwise_performance_test_data(
      n_pairs = 1000L,
      n_factors = 3,
      n_levels = 2,
      seed = 20260327
    )
    data <- add_adversarial_structure(data, seed = 20260328)

    params <- default_strategize_params(fast = TRUE)
    params$outcome_model_type <- "neural"
    base_neural_control <- params$neural_mcmc_control %||% list()
    params$neural_mcmc_control <- modifyList(
      base_neural_control,
      list(
        subsample_method = "batch_vi",
        batch_size = 128L,
        ModelDims = 16L,
        ModelDepth = as.integer(depth),
        residual_mode = residual_mode
      )
    )

    p_list <- generate_test_p_list(data$W)
    res <- do.call(strategize, c(
      list(Y = data$Y, W = data$W, p_list = p_list),
      data[c(
        "pair_id",
        "respondent_id",
        "respondent_task_id",
        "profile_order",
        "competing_group_variable_respondent",
        "competing_group_variable_candidate",
        "competing_group_competition_variable_candidate"
      )],
      params
    ))

    info <- get_neural_model_info_local(res)
    metrics <- info$fit_metrics
    y_eval <- data$Y[data$profile_order == 1L]
    null_metrics <- compute_binary_null_metrics_local(y_eval)

    data.frame(
      benchmark = "pairwise_oos",
      model = model_label(residual_mode),
      residual_mode = residual_mode,
      depth = as.integer(depth),
      seed = 20260327L,
      n_eval = safe_num(metrics$n_eval),
      auc = safe_num(metrics$auc),
      log_loss = safe_num(metrics$log_loss),
      accuracy = safe_num(metrics$accuracy),
      brier = safe_num(metrics$brier),
      null_log_loss = safe_num(null_metrics$log_loss),
      null_accuracy = safe_num(null_metrics$accuracy),
      null_brier = safe_num(null_metrics$brier),
      stringsAsFactors = FALSE
    )
  })
}

run_average_case_recovery_benchmark <- function(residual_mode,
                                                seed,
                                                depth = 4L,
                                                svi_steps = 600L,
                                                eval_max_n = 1000L) {
  fixture <- generate_linear_average_case_fixture()
  withr::with_seed(seed, {
    res <- strategize(
      Y = fixture$Y,
      W = fixture$W,
      lambda = fixture$lambda,
      outcome_model_type = "neural",
      diff = FALSE,
      adversarial = FALSE,
      compute_se = FALSE,
      penalty_type = "L2",
      use_regularization = FALSE,
      use_optax = FALSE,
      force_gaussian = FALSE,
      nSGD = 1000L,
      nMonte_Qglm = 1000L,
      a_init_sd = 0.001,
      optim_type = "gd",
      neural_mcmc_control = list(
        subsample_method = "batch_vi",
        ModelDims = 64L,
        ModelDepth = as.integer(depth),
        residual_mode = residual_mode,
        qk_norm = FALSE,
        batch_size = 512L,
        optimizer = "adam",
        vi_guide = "auto_diagonal",
        svi_steps = as.integer(svi_steps),
        svi_num_draws = 25L,
        uncertainty_scope = "output",
        eval_enabled = TRUE,
        eval_n_folds = 2L,
        eval_seed = 123L,
        eval_max_n = as.integer(eval_max_n)
      )
    )

    pi_hat <- extract_average_case_pi_hat_local(res)
    Q_hat <- if (!is.null(res$Q_point_mEst) && all(is.finite(res$Q_point_mEst))) {
      as.numeric(res$Q_point_mEst)
    } else {
      as.numeric(res$Q_point)
    }
    pi_rel_err <- mean(
      abs(pi_hat - fixture$pi_star_true) / pmax(abs(fixture$pi_star_true), 1e-8)
    )
    Q_rel_err <- abs(Q_hat - fixture$trueQ) / pmax(abs(fixture$trueQ), 1e-8)
    mu_hat <- extract_average_case_neural_mu_hat_local(res, fixture$W)
    rmse_y <- sqrt(mean((mu_hat - fixture$Y) ^ 2))
    rmse_null <- sqrt(mean((mean(fixture$Y) - fixture$Y) ^ 2))
    rmse_mu_true <- sqrt(mean((mu_hat - fixture$mu_true) ^ 2))
    cor_mu <- suppressWarnings(stats::cor(mu_hat, fixture$mu_true))
    info <- get_neural_model_info_local(res)
    metrics <- info$fit_metrics

    data.frame(
      benchmark = "average_case_recovery",
      model = model_label(residual_mode),
      residual_mode = residual_mode,
      depth = as.integer(depth),
      seed = as.integer(seed),
      pi_rel_err = pi_rel_err,
      Q_rel_err = Q_rel_err,
      Q_hat = Q_hat,
      Q_true = fixture$trueQ,
      rmse_y = rmse_y,
      rmse_null = rmse_null,
      rmse_mu_true = rmse_mu_true,
      cor_mu = cor_mu,
      oos_n_eval = safe_num(metrics$n_eval),
      oos_rmse = safe_num(metrics$rmse),
      oos_mae = safe_num(metrics$mae),
      oos_nll = safe_num(metrics$nll),
      stringsAsFactors = FALSE
    )
  })
}

summarise_numeric_by_model <- function(df, benchmark_name) {
  subset_df <- df[df$benchmark == benchmark_name, , drop = FALSE]
  id_cols <- c("benchmark", "model", "residual_mode", "depth")
  metric_cols <- setdiff(names(subset_df), c(id_cols, "seed"))
  metric_cols <- metric_cols[vapply(metric_cols, function(col) is.numeric(subset_df[[col]]), logical(1))]

  split_df <- split(subset_df, subset_df$model)
  out <- lapply(split_df, function(chunk) {
    row <- chunk[1L, id_cols, drop = FALSE]
    for (col in metric_cols) {
      values <- chunk[[col]]
      row[[paste0(col, "_mean")]] <- mean(values, na.rm = TRUE)
      row[[paste0(col, "_sd")]] <- stats::sd(values, na.rm = TRUE)
    }
    row
  })
  do.call(rbind, out)
}

avg_seed_env <- Sys.getenv("STRATEGIZE_BENCH_AVG_SEEDS", "20260326")
avg_seeds <- as.integer(strsplit(avg_seed_env, ",", fixed = TRUE)[[1L]])
avg_seeds <- avg_seeds[is.finite(avg_seeds)]
if (!length(avg_seeds)) {
  avg_seeds <- 20260326L
}
avg_svi_steps <- suppressWarnings(as.integer(Sys.getenv("STRATEGIZE_BENCH_AVG_SVI_STEPS", "600")))
if (!is.finite(avg_svi_steps) || avg_svi_steps < 1L) {
  avg_svi_steps <- 600L
}
avg_eval_max_n <- suppressWarnings(as.integer(Sys.getenv("STRATEGIZE_BENCH_AVG_EVAL_MAX", "1000")))
if (!is.finite(avg_eval_max_n) || avg_eval_max_n < 1L) {
  avg_eval_max_n <- 1000L
}
residual_modes <- c("standard", "full_attn")

message("Running pairwise OOS depth-4 benchmark...")
pairwise_rows <- do.call(
  rbind,
  lapply(residual_modes, run_pairwise_oos_benchmark, depth = 4L)
)

message("Running average-case recovery depth-4 benchmark...")
average_rows <- do.call(
  rbind,
  lapply(residual_modes, function(mode) {
    do.call(
      rbind,
      lapply(avg_seeds, function(seed) {
        message(sprintf("  average-case residual_mode=%s seed=%d", mode, seed))
        run_average_case_recovery_benchmark(
          mode,
          seed = seed,
          depth = 4L,
          svi_steps = avg_svi_steps,
          eval_max_n = avg_eval_max_n
        )
      })
    )
  })
)

raw_results <- rbind(pairwise_rows, average_rows)
pairwise_summary <- summarise_numeric_by_model(raw_results, "pairwise_oos")
average_summary <- summarise_numeric_by_model(raw_results, "average_case_recovery")

raw_path <- file.path(repo_root, "tmp", "benchmark_full_attn_residual_depth4_raw.csv")
pairwise_summary_path <- file.path(repo_root, "tmp", "benchmark_full_attn_residual_depth4_pairwise_summary.csv")
average_summary_path <- file.path(repo_root, "tmp", "benchmark_full_attn_residual_depth4_average_summary.csv")
rds_path <- file.path(repo_root, "tmp", "benchmark_full_attn_residual_depth4_results.rds")

write.csv(raw_results, raw_path, row.names = FALSE)
write.csv(pairwise_summary, pairwise_summary_path, row.names = FALSE)
write.csv(average_summary, average_summary_path, row.names = FALSE)
saveRDS(
  list(
    raw_results = raw_results,
    pairwise_summary = pairwise_summary,
    average_summary = average_summary
  ),
  rds_path
)

message("")
message("Pairwise OOS summary:")
print(pairwise_summary, row.names = FALSE)
message("")
message("Average-case recovery summary:")
print(average_summary, row.names = FALSE)
message("")
message("Saved:")
message(raw_path)
message(pairwise_summary_path)
message(average_summary_path)
message(rds_path)
message(sprintf(
  "Average-case settings: seeds=%s; svi_steps=%d; eval_max_n=%d",
  paste(avg_seeds, collapse = ","),
  avg_svi_steps,
  avg_eval_max_n
))
