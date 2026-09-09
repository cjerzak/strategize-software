# Select explicit interaction columns; never ask a screener to interact
# pairwise differences. All main effects remain in the subsequent refit.
cs_glm_screen_features <- function(main, interactions, y, family,
                                   n_folds = 3L, cluster = NULL,
                                   intercept = TRUE) {
  if (is.null(interactions) || !NCOL(interactions)) return(integer(0))
  main <- as.matrix(main)
  interactions <- as.matrix(interactions)
  if (!NCOL(interactions) || !glm_response_has_variation(y)) return(integer(0))
  x <- cbind(main, interactions)
  varying <- apply(x, 2L, function(z) all(is.finite(z)) && stats::sd(z) > 0)
  idx <- which(varying)
  if (length(idx) < 2L) return(which(varying[-seq_len(NCOL(main))]))
  if (is.null(cluster)) cluster <- seq_along(y)
  if (length(cluster) != length(y) || anyNA(cluster)) {
    stop("Interaction screening requires complete cluster IDs.", call. = FALSE)
  }
  groups <- unique(as.character(cluster))
  k <- min(as.integer(n_folds), length(groups))
  if (k < 3L) {
    # There are too few independent groups to select a penalty by CV. Retaining
    # the explicit features is safer than claiming a data-selected empty model.
    return(seq_len(NCOL(interactions)))
  }
  assignments <- sample(rep(seq_len(k), length.out = length(groups)))
  foldid <- assignments[match(as.character(cluster), groups)]
  fit <- glmnet::cv.glmnet(
    x = x[, idx, drop = FALSE], y = y, family = family,
    alpha = 1, foldid = foldid, intercept = intercept,
    standardize = TRUE, type.measure = "deviance"
  )
  beta <- as.numeric(stats::coef(fit, s = "lambda.min"))[-1L]
  selected <- idx[is.finite(beta) & abs(beta) > 1e-10]
  selected <- selected[selected > NCOL(main)] - NCOL(main)
  as.integer(selected)
}
