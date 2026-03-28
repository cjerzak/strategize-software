pkgload::load_all("/Users/cjerzak/Documents/strategize-software/strategize", quiet = TRUE)
source("/Users/cjerzak/Documents/strategize-software/strategize/tests/testthat/helper-strategize.R")

compute_binary_null_metrics <- function(y) {
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

get_neural_model_info <- function(res) {
  model_info <- res$neural_model_info$ast
  if (is.null(model_info)) {
    model_info <- res$neural_model_info$dag
  }
  if (is.null(model_info)) {
    model_info <- res$neural_model_info$ast0
  }
  if (is.null(model_info)) {
    model_info <- res$neural_model_info$dag0
  }
  model_info
}

withr::local_envvar(c(
  STRATEGIZE_NEURAL_FAST_MCMC = "true",
  STRATEGIZE_NEURAL_EVAL_FOLDS = "2",
  STRATEGIZE_NEURAL_EVAL_SEED = "123"
))

data <- generate_test_data(n = 40, seed = 123)
params <- default_strategize_params(fast = TRUE)
params$outcome_model_type <- "neural"
p_list <- generate_test_p_list(data$W)

res <- do.call(strategize, c(
  list(Y = data$Y, W = data$W, p_list = p_list),
  data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order")],
  params
))

info <- get_neural_model_info(res)
metrics <- info$fit_metrics
y_eval <- data$Y[data$profile_order == 1L]
null_metrics <- compute_binary_null_metrics(y_eval)

cat("metrics\n")
print(metrics[c(
  "likelihood", "n_eval", "auc", "log_loss", "accuracy",
  "brier", "eval_note", "n_folds", "seed"
)])
cat("null_metrics\n")
print(null_metrics)
cat("by_fold\n")
print(metrics$by_fold)
