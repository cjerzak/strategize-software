accounting_fixture <- function(parameters = 1e6, tokens = 10000) {
  list(stored_model_parameters = parameters, expected_global_tokens_per_update = tokens)
}

test_that("automatic budgets clamp doubles before converting to integers", {
  steps <- strategize:::neural_optimal_svi_steps
  expect_identical(steps(accounting_fixture(), "batch_vi"), 2000L)
  expect_identical(steps(accounting_fixture(), "batch"), 2000L)
  expect_identical(steps(accounting_fixture(), "full"), 1000L)
  expect_identical(steps(accounting_fixture(1, 1e10), "full"), 50L)
  for (shape in list(c(4096, 32), c(8192, 80))) {
    # Transformer weights alone already exceed int32. No allocation needed.
    count <- shape[[2]] * (4 * shape[[1]]^2 + 3 * shape[[1]] * round(3.75 * shape[[1]]))
    expect_gt(count, .Machine$integer.max)
    expect_no_warning(actual <- steps(accounting_fixture(count, 64 * 40), "batch_vi"))
    expect_identical(actual, 40000L)
  }
  expect_identical(steps(accounting_fixture(1e15, 1), "batch_vi"), 40000L)
  expect_identical(steps(accounting_fixture(1e308, 1), "batch_vi"), 40000L)
  expect_error(steps(NULL), "resolved model and token accounting")
  for (bad in list(NA_real_, Inf, 0, -1, numeric(0))) {
    expect_error(steps(accounting_fixture(bad)), "positive finite")
    expect_error(steps(accounting_fixture(tokens = bad)), "positive finite")
  }
  expect_error(steps(accounting_fixture(), min_steps_full = 1001), "bounds")
})

test_that("budget metadata explains the heuristic and applied cap", {
  details <- strategize:::neural_optimal_svi_steps(accounting_fixture(), return_info = TRUE)
  expect_equal(details$accounting_version, 2L)
  expect_equal(details$unclamped_steps, 2000)
  expect_equal(details$resolved_steps, 1000L)
  expect_equal(details$applied_bound, "maximum")
  expect_equal(details$budget_basis, "stored_model_parameters")
})

test_that("budgets honor training rows, global batches and actual mixed rounding", {
  accounting <- strategize:::neural_svi_budget_accounting
  parameters <- list(stored_model_parameters = 1e6, active_model_parameters = 6e5)
  tokens <- c(10, 20, 30, 40, 1000)
  a <- accounting(parameters, tokens, train_idx = 1:4, batch_size = 3, subsample_method = "batch_vi")
  expect_equal(a$expected_global_tokens_per_update, 75)
  expect_equal(a$n_training_observations, 4)
  expect_equal(accounting(parameters, tokens, 1:4, 3, "full")$expected_global_tokens_per_update, 100)
  mixed <- accounting(parameters, tokens, 1:4, 3, "batch_vi", mixed_pair_count = 2)
  expect_equal(mixed$expected_global_tokens_per_update, 2 * 15 + 35)
  expect_equal(mixed$expected_global_batch_size, 3)
  separate <- accounting(parameters, tokens, 1:4, 3, "batch_vi", mixed_pair_count = 2,
    mixed_sampling = "independent")
  expect_equal(separate$expected_global_tokens_per_update, 100)
  expect_equal(separate$expected_global_batch_size, 4)
  expect_equal(strategize:::neural_svi_mixed_batch_counts(1, 9, 4), c(pair = 1, single = 3))
  expect_error(strategize:::neural_svi_mixed_batch_counts(1, 9, 1), "pairwise and single")
  balanced <- list(obs_by_unit = list(c(1,2), 3, 4), units_by_study = list(c(1,2), 3))
  b <- accounting(parameters, tokens, 1:4, 8, "batch_vi", balanced_state = balanced)
  expect_equal(b$expected_global_tokens_per_update, 8 * mean(c(mean(c(15,30)), 40)))
  expect_equal(b$expected_global_batch_size, 8)
  loop <- list(enabled = TRUE, prelude_layers = 1, coda_layers = 1,
    iterations = 2, training_iterations = c(1,3), training_probabilities = c(.25,.75))
  recurrent <- accounting(parameters, tokens, 1:4, 3, "batch_vi", model_depth = 4, transformer_loop = loop)
  expect_equal(recurrent$expected_layer_passes, 7)
  expect_equal(recurrent$max_layer_passes, 8)
  expect_equal(recurrent$stored_model_parameters, a$stored_model_parameters)
  expect_equal(recurrent$expected_global_tokens_per_update, a$expected_global_tokens_per_update)
})

test_that("token accounting counts schema slots, masked context and separate towers", {
  count <- strategize:::neural_svi_tokens_per_observation
  info <- list(default_factor_order = 0:3, factor_order_by_experiment = list(0:1, c(0,2,3)),
    default_covariate_order = 0:2, covariate_order_by_experiment = list(0:1, 2L),
    covariate_names = letters[1:3], experiment_token_mode = "description",
    experiment_description_text = matrix(1, 2, 3),
    party_missing_index = 9L, resp_party_missing_index = 9L,
    context_present_masking = TRUE, cross_candidate_encoder_mode = "none",
    low_rank_interaction_rank = 0L)
  params <- c("E_covariate_fused_base", "W_experiment_text", "E_party", "E_resp_party", "E_rel")
  # Pair: 2*(choice + experiment + 2 covariates + 2 factors + party + relation + respondent).
  # Single with absent groups: choice + experiment + 1 covariate + 3 factors.
  expect_equal(count(info, params, c(TRUE,FALSE), 0:1, c(0,9), c(1,9), c(0,9)), c(18,6))
  towers <- c(params, "W_rc_r", "W_rc_c", "W_rc_out")
  expect_equal(count(info, towers, c(TRUE,FALSE), 0:1, c(0,9), c(1,9), c(0,9)), c(33,13))
  expect_equal(count(info, c(towers,"W_to_cand"), c(TRUE,FALSE), 0:1, c(0,9), c(1,9), c(0,9)), c(15,7))
  info$cross_candidate_encoder_mode <- "full"
  info$pairwise_antisymmetry <- "none"
  expect_equal(count(info, params, c(TRUE,FALSE), 0:1, c(0,9), c(1,9), c(0,9)), c(15,6))
  # Token budget maxima are allocation bounds, not data; changing them does not
  # create additional exposure. Missing covariate values retain their tokens.
  info$max_factor_tokens <- 1024L
  info$max_covariate_tokens <- 1024L
  expect_equal(count(info, params, c(TRUE,FALSE), 0:1, c(0,9), c(1,9), c(0,9)), c(15,6))
})

test_that("explicit steps, floors and saved automatic schedules remain stable", {
  resolve <- strategize:::neural_resolve_svi_budget
  automatic <- resolve("optimal", 10L, accounting = accounting_fixture(), subsample_method = "batch_vi")
  expect_equal(automatic$svi_steps, 2000L)
  resumed <- resolve("optimal", 10L, resume_budget = automatic, subsample_method = "batch_vi",
    accounting = accounting_fixture(1e15))
  expect_identical(resumed, automatic)
  expect_identical(resolve(7L, 3L, user_supplied_svi_steps = TRUE,
    output_only_mode = TRUE, likelihood = "normal", subsample_method = "batch_vi")$svi_steps, 7L)
  floored <- resolve("optimal", 10L, accounting = accounting_fixture(100),
    output_only_mode = TRUE, likelihood = "normal", subsample_method = "batch_vi")
  expect_equal(floored$svi_steps, 2000L)
  expect_equal(floored$svi_num_draws, 10L)
  expect_true(floored$applied_output_single_normal_batch_vi_floor)
  expect_identical(resolve("optimal", 10L, resume_budget = floored,
    output_only_mode = TRUE, likelihood = "normal", subsample_method = "batch_vi"), floored)
  legacy <- automatic
  legacy$accounting <- NULL
  expect_identical(resolve("optimal", 10L, resume_budget = legacy, subsample_method = "batch_vi"), legacy)
  expect_error(resolve("optimal", 10L, resume_budget = list(used_optimal = FALSE)), "Checkpoint")
})

test_that("gradient checkpoint counts are not narrowed to int32", {
  d <- strategize:::neural_append_gradient_checkpoint(NULL, 1L,
    list(grad_l2 = 3, grad_rms = 1, grad_max_abs = 2,
      grad_n_nonfinite = 2^32 + 1, grad_n_elements = 2^34))
  expect_equal(d$checkpoint_n_elements, 2^34)
  expect_equal(d$checkpoint_n_nonfinite, 2^32 + 1)
  expect_equal(d$gradient_status, "nonfinite")
})

test_that("distributed gradient counts survive the Python to R boundary", {
  skip_if_not_installed("reticulate")
  if (!reticulate::py_module_available("jax")) skip("JAX unavailable")
  reticulate::import_from_path("strategize_distributed",
    path = system.file("python", package = "strategize"), convert = FALSE)
  scope <- reticulate::py_run_string("
from types import SimpleNamespace
from strategize_distributed import Runtime
# Simulate compiled scalar output and static shapes without huge allocations.
runtime = Runtime.__new__(Runtime)
runtime._compile = lambda *a, **kw: lambda *a: dict(scale=1., ssq=1., nonfinite_hi=3, nonfinite_lo=19)
runtime.local_tree = lambda x: x
svi = SimpleNamespace(optim=SimpleNamespace(get_params=lambda _, params=SimpleNamespace(size=2**34): params))
stats = runtime.gradients(svi, SimpleNamespace(optim_state=None), {})
", local = TRUE)
  expect_identical(scope$stats$grad_n_elements, 2^34)
  expect_identical(scope$stats$grad_n_nonfinite, 3 * 2^32 + 19)
})
