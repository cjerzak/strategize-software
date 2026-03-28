cs_compute_auc <- function(y_true, y_score) {
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

cs_compute_binary_log_loss <- function(y_true, y_score, eps = 1e-12) {
  y_true <- as.numeric(y_true)
  y_score <- as.numeric(y_score)
  ok <- is.finite(y_true) & is.finite(y_score)
  y_true <- y_true[ok]
  y_score <- y_score[ok]
  if (!length(y_true)) return(NA_real_)
  p <- pmin(pmax(y_score, eps), 1 - eps)
  -mean(y_true * log(p) + (1 - y_true) * log(1 - p))
}

cs_compute_multiclass_log_loss <- function(y_true, prob_mat, eps = 1e-12) {
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

cs_binary_metric_diagnostics <- function(y_true, y_score, threshold = 0.5) {
  y_true <- as.numeric(y_true)
  y_score <- as.numeric(y_score)
  ok <- is.finite(y_true) & is.finite(y_score)
  y_true <- y_true[ok]
  y_score <- y_score[ok]

  quantile_names <- c("p05", "p25", "p50", "p75", "p95")
  empty_quantiles <- setNames(rep(NA_real_, length(quantile_names)), quantile_names)
  empty_confusion <- c(tn = NA_integer_, fp = NA_integer_, fn = NA_integer_, tp = NA_integer_)

  if (!length(y_true)) {
    return(list(
      truth_pred_correlation = NA_real_,
      y_mean = NA_real_,
      pred_mean = NA_real_,
      pred_sd = NA_real_,
      pred_quantiles = empty_quantiles,
      confusion_0_5 = empty_confusion
    ))
  }

  cor_val <- NA_real_
  if (length(y_true) > 1L) {
    y_sd <- suppressWarnings(stats::sd(y_true))
    pred_sd <- suppressWarnings(stats::sd(y_score))
    if (is.finite(y_sd) && y_sd > 0 && is.finite(pred_sd) && pred_sd > 0) {
      cor_val <- suppressWarnings(stats::cor(y_true, y_score))
    }
  }

  pred_quantiles <- stats::quantile(
    y_score,
    probs = c(0.05, 0.25, 0.5, 0.75, 0.95),
    na.rm = TRUE,
    names = FALSE,
    type = 7
  )
  names(pred_quantiles) <- quantile_names

  pred_class <- as.integer(y_score >= threshold)
  confusion_0_5 <- c(
    tn = sum(pred_class == 0L & y_true == 0L),
    fp = sum(pred_class == 1L & y_true == 0L),
    fn = sum(pred_class == 0L & y_true == 1L),
    tp = sum(pred_class == 1L & y_true == 1L)
  )

  list(
    truth_pred_correlation = cor_val,
    y_mean = mean(y_true),
    pred_mean = mean(y_score),
    pred_sd = if (length(y_score) > 1L) stats::sd(y_score) else 0,
    pred_quantiles = pred_quantiles,
    confusion_0_5 = confusion_0_5
  )
}

cs_compute_outcome_metrics <- function(y_eval, pred_eval, likelihood, threshold = 0.5) {
  binary_likelihood <- likelihood %in% c("bernoulli", "binomial")
  if (binary_likelihood) {
    y_eval <- as.numeric(y_eval)
    p <- as.numeric(pred_eval)
    keep <- is.finite(y_eval) & is.finite(p) & (y_eval %in% c(0, 1))
    y_eval <- y_eval[keep]
    p <- p[keep]
    metrics <- list(
      likelihood = likelihood,
      n_eval = length(y_eval),
      auc = cs_compute_auc(y_eval, p),
      log_loss = cs_compute_binary_log_loss(y_eval, p),
      accuracy = if (length(y_eval)) mean((p >= threshold) == y_eval) else NA_real_,
      brier = if (length(y_eval)) mean((p - y_eval) ^ 2) else NA_real_
    )
    return(c(metrics, cs_binary_metric_diagnostics(y_eval, p, threshold = threshold)))
  }

  if (identical(likelihood, "categorical")) {
    y_eval <- as.integer(y_eval)
    prob_mat <- as.matrix(pred_eval)
    keep <- !is.na(y_eval)
    y_eval <- y_eval[keep]
    prob_mat <- prob_mat[keep, , drop = FALSE]
    if (length(y_eval)) {
      log_loss <- cs_compute_multiclass_log_loss(y_eval, prob_mat)
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
  } else if (length(pred_sigma) != length(y_eval)) {
    pred_sigma <- rep(NA_real_, length(y_eval))
  }

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
    rmse = if (length(y_eval)) sqrt(mean((pred_mu - y_eval) ^ 2)) else NA_real_,
    mae = if (length(y_eval)) mean(abs(pred_mu - y_eval)) else NA_real_,
    nll = nll
  )
}
