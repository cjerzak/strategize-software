# =============================================================================
# Muon Optimizer Targeting Tests
# =============================================================================

muon_test_labels <- local({
  initialized <- FALSE

  function(params) {
    skip_on_cran()
    skip_if_no_jax()

    if (!initialized) {
      strategize:::initialize_jax(conda_env = "strategize_env", conda_env_required = TRUE)
      strategize:::neural_get_muon_dimension_numbers_callable(force_refresh = TRUE)
      reticulate::py_run_string(
        paste(
          "def _strategize_muon_test_labels(params):",
          "    dimnums_tree = _strategize_muon_dimnums(params)",
          "    out = {}",
          "    for name in params.keys():",
          "        out[str(name)] = 'muon' if dimnums_tree[str(name)] is not None else 'adam'",
          "    return out",
          sep = "\n"
        )
      )
      initialized <<- TRUE
    }

    labels <- reticulate::py_eval("_strategize_muon_test_labels")(params)
    reticulate::py_to_r(labels)
  }
})

muon_test_array <- function(ndim) {
  strategize:::initialize_jax(conda_env = "strategize_env", conda_env_required = TRUE)
  shape <- if (identical(as.integer(ndim), 2L)) list(2L, 2L) else list(2L)
  strategize:::strenv$jnp$ones(shape)
}

test_that("muon dimension-number tree hits intended matrix weights and excludes others", {
  skip_on_cran()
  skip_if_no_jax()
  strategize:::initialize_jax(conda_env = "strategize_env", conda_env_required = TRUE)

  if (!reticulate::py_has_attr(strategize:::strenv$optax, "contrib") ||
      !reticulate::py_has_attr(strategize:::strenv$optax$contrib, "muon")) {
    skip("optax.contrib.muon not available")
  }

  cases <- list(
    list(name = "W_q_l1", ndim = 2L, want = "muon"),
    list(name = "W_ff2_l3", ndim = 2L, want = "muon"),
    list(name = "W_q_cross", ndim = 2L, want = "muon"),
    list(name = "M_cross_raw", ndim = 2L, want = "muon"),
    # On-critical-path hidden MLPs now orthogonalized alongside the transformer FF
    # (previously fell to Muon's Adam sub-branch despite identical structure).
    list(name = "W_factor_fuse_1", ndim = 2L, want = "muon"),
    list(name = "W_factor_fuse_2", ndim = 2L, want = "muon"),
    list(name = "W_covariate_fuse_1", ndim = 2L, want = "muon"),
    list(name = "W_covariate_value_conditioner_1", ndim = 2L, want = "muon"),
    list(name = "W_rc_r", ndim = 2L, want = "muon"),
    # Output/unembedding heads are excluded from Muon (standard recipe): they need
    # Adam's per-coordinate scaling to calibrate output magnitude.
    list(name = "W_out", ndim = 2L, want = "adam"),
    list(name = "W_rc_out", ndim = 2L, want = "adam"),
    list(name = "b_out", ndim = 1L, want = "adam"),
    list(name = "RMS_attn_l1", ndim = 1L, want = "adam")
  )

  params <- setNames(
    lapply(cases, function(case) muon_test_array(case$ndim)),
    vapply(cases, `[[`, character(1), "name")
  )
  labels <- muon_test_labels(params)

  for (case in cases) {
    expect_identical(
      unname(labels[[case$name]]),
      case$want,
      info = sprintf("Expected %s to map to %s", case$name, case$want)
    )
  }

  expect_false(strategize:::neural_muon_targets_matrix_weight("W_q_l1", ndim = 1L))
})

test_that("muon dimension-number tree handles guide-location aliases but not guide scales", {
  skip_on_cran()
  skip_if_no_jax()
  strategize:::initialize_jax(conda_env = "strategize_env", conda_env_required = TRUE)

  if (!reticulate::py_has_attr(strategize:::strenv$optax, "contrib") ||
      !reticulate::py_has_attr(strategize:::strenv$optax$contrib, "muon")) {
    skip("optax.contrib.muon not available")
  }

  cases <- list(
    list(name = "W_q_l1_auto_loc", want = "muon"),
    list(name = "W_ff1_l2_base_auto_loc", want = "muon"),
    list(name = "W_ff2_l2_decentered_auto_loc", want = "muon"),
    list(name = "W_factor_fuse_1_auto_loc", want = "muon"),
    list(name = "M_cross_raw_auto_loc", want = "muon"),
    list(name = "W_out_auto_loc", want = "adam"),
    list(name = "W_q_l1_auto_scale", want = "adam"),
    list(name = "W_ff1_l2_base_auto_scale", want = "adam"),
    list(name = "W_out_auto_scale", want = "adam")
  )

  params <- setNames(
    replicate(length(cases), muon_test_array(2L), simplify = FALSE),
    vapply(cases, `[[`, character(1), "name")
  )
  labels <- muon_test_labels(params)

  for (case in cases) {
    expect_identical(
      unname(labels[[case$name]]),
      case$want,
      info = sprintf("Expected %s to map to %s", case$name, case$want)
    )
  }
})

test_that("muon rejects guides that flatten the matrix structure", {
  for (explicit in c(FALSE, TRUE)) {
    expect_error(strategize:::neural_resolve_svi_optimizer_tag(
      "muon", "auto_diagonal", explicit), "auto_diagonal.*incompatible")
  }
  expect_identical(strategize:::neural_resolve_svi_optimizer_tag("adam", "auto_diagonal"), "adam")
})

test_that("default Muon requires the full Optax API", {
  skip_if_not_installed("reticulate")
  testthat::local_mocked_bindings(py_has_attr = function(x, name) name %in% names(x), .package = "reticulate")
  env <- strategize:::strenv
  for (api in list(list(), list(contrib = list(muon = function(...) NULL)))) {
    old <- env$optax
    env$optax <- api
    tryCatch({
      expect_error(strategize:::neural_resolve_svi_optimizer_tag("muon", "auto_normal"), "requires optax")
      expect_error(strategize:::neural_resolve_svi_optimizer_tag("muon", "auto_normal", TRUE), "requires optax")
    }, finally = { env$optax <- old })
  }
})

test_that("direct dense neural SVI uses Muon when the optimizer is omitted", {
  skip_on_cran()
  skip_if_no_jax()
  withr::local_envvar(c(STRATEGIZE_NEURAL_SKIP_EVAL = "1"))
  W <- data.frame(feature = rep(c("A", "B"), 8L))
  names_list <- strategize:::cs2step_build_names_list(W)
  W_idx <- strategize:::cs2step_encode_W_indices(W, names_list = names_list, unknown = "error")
  fit <- strategize:::cs2step_eval_outcome_model_neural(
    Y = rep(c(1, 0, 0, 1), 4L), W_idx = W_idx, names_list = names_list,
    factor_levels = vapply(names_list, function(x) length(x[[1L]]), integer(1)),
    diff = TRUE, pair_id = rep(seq_len(8L), each = 2L), profile_order = rep(1:2, 8L),
    neural_mcmc_control = list(ModelDims = 8L, ModelDepth = 1L,
      subsample_method = "batch_vi", uncertainty_scope = "output", transformer_ffn = "swiglu",
      svi_steps = 2L, svi_num_draws = 1L, batch_size = 4L,
      early_stopping = FALSE, eval_enabled = FALSE, gradient_diagnostics = FALSE))
  diagnostics <- fit$neural_model_info$optimizer_diagnostics
  expect_identical(diagnostics$optimizer, "muon")
  expect_identical(diagnostics$muon_partition_status, "verified")
  expect_gt(diagnostics$muon_parameter_count, 0)
})
