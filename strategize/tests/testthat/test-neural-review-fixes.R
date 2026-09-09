test_that("validation caps keep complete respondents in mixed observation layouts", {
  # Logical ordering: paired tasks first, then single-profile observations.
  cluster <- c(rep(paste0("pairs::", 1:8), each = 5), rep(paste0("single::", 1:8), each = 3))
  fold <- cs_make_stratified_folds(length(cluster), 3L, rep(0:1, 32), cluster, seed = 7L)$fold_id
  train <- which(fold != 1L)
  validation <- which(fold == 1L)
  set.seed(38)
  rng <- .Random.seed
  out <- neural_cap_validation_split(train, validation, 6L, cluster, seed = 8L)
  expect_identical(.Random.seed, rng)
  expect_length(intersect(cluster[out$train_idx], cluster[out$validation_idx]), 0L)
  expect_setequal(c(out$train_idx, out$validation_idx), seq_along(cluster))
  expect_gte(length(out$validation_idx), 6L)
  expect_lt(length(out$validation_idx), 6L + 5L)
  expect_identical(out, neural_cap_validation_split(train, validation, 6L, cluster, seed = 8L))
})

test_that("large validation clusters overshoot rather than leak", {
  out <- neural_cap_validation_split(1:3, 4:13, 2L, c(rep("train", 3), rep("validation", 10)))
  expect_identical(out$validation_idx, 4:13)
  expect_error(neural_cap_validation_split(1:3, 4:6, 2L, rep("same", 6)), "respondents must be disjoint")
  rows <- neural_cap_validation_split(1:3, 4:12, 2L)
  expect_length(rows$validation_idx, 2)
  expect_setequal(c(rows$train_idx, rows$validation_idx), 1:12)
})

test_that("singleton strata do not sample indices outside their stratum", {
  for (seed in 1:20) {
    folds <- cs_make_stratified_folds(8L, 3L, y = letters[1:8], cluster = LETTERS[1:8], seed = seed)
    expect_true(all(folds$fold_id %in% 1:3))
  }
})

test_that("finite validation metrics and budget completion are not convergence evidence", {
  diag <- neural_build_convergence_diagnostics(
    parameter_diagnostics = list(n_nonfinite = 0L), svi_loss_curve = c(2, 3),
    early_stopping = list(best_metric = .9), steps_completed = 2L, steps_planned = 44008L
  )
  expect_identical(diag$execution_status, "incomplete")
  expect_true(is.na(diag$converged))
  done <- neural_build_convergence_diagnostics(svi_loss_curve = c(3, 2), steps_completed = 2L, steps_planned = 2L)
  expect_identical(done$execution_status, "completed")
  expect_identical(done$optimization_status, "not_assessed")
  expect_true(is.na(done$converged))
})

test_that("selected and last checkpoint metrics stay distinct", {
  es <- neural_finalize_validation_metrics(list(
    best_metric = .71, best_step = 100L, validation_steps = c(100L, 200L),
    validation_loss_history = c(.71, .75), final_metric = .71
  ))
  expect_equal(es$last_metric, .75)
  expect_equal(es$final_metric, .75)
  expect_equal(es$selected_model_metric, .71)
  expect_identical(es$last_step, 200L)
  expect_identical(es$selected_model_step, 100L)
  expect_identical(neural_build_gradient_diagnostics()$gradient_status_scope, "finiteness_only")
})

test_that("replica-local family normalization is rejected before distributed training", {
  expect_error(strategize_dp_validate_training_control(list(universal_family_logprob_normalize = TRUE), TRUE),
               "replica-local batches")
  expect_no_error(strategize_dp_validate_training_control(list(universal_family_logprob_normalize = FALSE), TRUE))
  expect_no_error(strategize_dp_validate_training_control(list(universal_family_logprob_normalize = TRUE), FALSE))
})

test_that("unstacking restores exact weight shapes, gains and scalar gates", {
  p <- list(W_q_layers = array(seq_len(24), c(2, 3, 4)), RMS_q_layers = matrix(1:6, 2, 3),
            alpha_attn_layers = c(.04, .09), W_out = matrix(1:3, 3, 1))
  out <- neural_unstack_standard_transformer_layers(p, 2L)
  expect_equal(out$W_q_l2, p$W_q_layers[2, , ])
  expect_equal(out$RMS_q_l1, p$RMS_q_layers[1, ])
  expect_equal(out$alpha_attn_l2, .09)
  expect_identical(out$W_out, p$W_out)
  expect_false("W_q_layers" %in% names(out))
  expect_error(neural_unstack_standard_transformer_layers(p, 3L), "expected 3")
  singleton <- neural_unstack_standard_transformer_layers(list(W_ff2_layers = array(7, c(1, 1, 1))), 1L)
  expect_identical(dim(singleton$W_ff2_l1), c(1L, 1L))
})

test_that("token-family warm starts follow names when the target omits pair context", {
  params <- list(E_token_family = matrix(1:15, 5, 3))
  source <- c("factor_fused", "stage", "matchup", "choice", "separator")
  target <- c("factor_fused", "choice", "separator")
  mapped <- neural_remap_token_family_init(params, source, target)
  expect_identical(mapped$E_token_family, params$E_token_family[c(1, 4, 5), ])
  expect_error(neural_remap_token_family_init(params, source, c(target, "new")), "do not cover")
})

test_that("full-objective normalization accepts only supported policies", {
  validate <- function(value) validate_strategize_inputs(
    Y = c(1, 0, 1, 0), W = data.frame(x = c("a", "b", "a", "b")), lambda = .1,
    neural_mcmc_control = list(svi_objective_normalization = value)
  )
  expect_true(validate("per_observation"))
  expect_true(validate("none"))
  for (value in list("likelihood_only", NA_character_, TRUE, c("none", "per_observation"))) {
    expect_error(validate(value), "svi_objective_normalization")
  }
})

test_that("two-category ordinal heads retain one cutpoint per study", {
  raw <- matrix(c(-2, 1, 3), ncol = 1)
  expect_identical(cs2step_ordinal_thresholds_from_raw(raw), raw)
  probabilities <- cs2step_ordinal_prob_matrix(c(0, 0, 0), 2L,
                                              experiment_index = 0:2, ordinal_threshold_raw = raw)
  expect_equal(probabilities[, 1], plogis(raw[, 1]))
})
