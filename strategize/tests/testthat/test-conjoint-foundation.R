test_that("pooled foundation training and writing stubs point to preference.fm", {
  expect_error(
    fit_conjoint_foundation_model(experiments = list()),
    "preference.fm::fit_conjoint_foundation_model",
    fixed = TRUE
  )

  fake <- structure(list(groups = list()), class = "conjoint_foundation_model")
  expect_error(
    save_conjoint_foundation_bundle(tempfile(), fake),
    "preference.fm::save_conjoint_foundation_bundle",
    fixed = TRUE
  )
})

test_that("legacy RDS foundation bundles still load", {
  bundle <- structure(
    list(
      schema_version = 1L,
      model_type = "conjoint_foundation",
      groups = list(),
      metadata = list(created_at = Sys.time())
    ),
    class = c("conjoint_foundation_bundle", "list")
  )

  tmp <- tempfile(fileext = ".rds")
  saveRDS(bundle, tmp)
  loaded <- load_conjoint_foundation_bundle(tmp, preload_params = FALSE)

  expect_s3_class(loaded, "conjoint_foundation_model")
  expect_identical(loaded$groups, list())
})

test_that("checkpoint directory loader restores direct params without preference.fm", {
  skip_on_cran()
  skip_if_no_jax()
  orbax_available <- tryCatch({
    reticulate::use_condaenv("strategize_env", required = TRUE)
    reticulate::py_module_available("orbax.checkpoint")
  }, error = function(e) FALSE)
  skip_if_not(orbax_available, "orbax.checkpoint not available")

  strategize:::initialize_jax(conda_env = "strategize_env", conda_env_required = TRUE)
  ocp <- tryCatch(
    reticulate::import("orbax.checkpoint.experimental.v1", convert = FALSE),
    error = function(e) NULL
  )
  if (is.null(ocp) || !reticulate::py_has_attr(ocp, "save_pytree")) {
    ocp <- reticulate::import("orbax.checkpoint", convert = FALSE)
  }
  skip_if_not(
    reticulate::py_has_attr(ocp, "save_pytree") ||
      reticulate::py_has_attr(ocp, "PyTreeCheckpointer"),
    "unsupported orbax checkpoint API"
  )

  group_key <- strategize:::cs_foundation_universal_group_key()
  tmp <- tempfile()
  dir.create(tmp, recursive = TRUE)
  bundle <- structure(
    list(
      schema_version = 1L,
      model_type = "conjoint_foundation",
      groups = list(),
      metadata = list(created_at = Sys.time())
    ),
    class = c("conjoint_foundation_bundle", "list")
  )
  bundle$groups[[group_key]] <- list(
    group_key = group_key,
    mode = "universal",
    likelihood = "bernoulli",
    n_outcomes = 1L,
    supported_modes = "pairwise",
    supported_likelihoods = "bernoulli",
    supported_pairwise_context_modes = "stage_free",
    experiment_ids = "study_a",
    encoder = list(
      factor_names = "price",
      names_list = list(price = list(c("Low", "High"))),
      factor_levels = c(price = 2L),
      unknown_policy = "holdout"
    ),
    schema_registry = list(
      slot_table = data.frame(
        slot_name = "slot_001",
        slot_key = "canon::price",
        display_label = "price",
        stringsAsFactors = FALSE
      ),
      pooled_names_list = list(slot_001 = list(c("canon::low", "canon::high"))),
      slot_level_keys = list(slot_001 = c("canon::low", "canon::high")),
      slot_level_labels = list(slot_001 = c("Low", "High")),
      experiment_maps = list()
    ),
    x_feature_names = character(0),
    x_schema = list(
      base_x_names = character(0),
      experiment_indicator_names = character(0),
      semantic_feature_names = character(0),
      experiment_token_levels = "study_a"
    ),
    text_registry = NULL,
    token_control = list(),
    fit = list(
      theta_mean = NULL,
      theta_var = NULL,
      neural_model_info = list(
        params = NULL,
        param_names = "W_out",
        param_shapes = list(c(1L, 1L)),
        param_sizes = list(1L),
        param_offsets = list(0L),
        n_params = 1L
      ),
      fit_metrics = NULL
    )
  )
  saveRDS(bundle, file.path(tmp, "metadata.rds"))

  manifest <- list(
    schema_version = 1L,
    artifact_type = "conjoint_foundation_checkpoint",
    writer = list(package = "preference.fm", version = "0.0.1"),
    compatible = list(strategize_version = as.character(utils::packageVersion("strategize"))),
    arrays = list(
      format = "orbax_pytree",
      path = "arrays",
      groups = list(
        group_001 = list(
          group_key = group_key,
          params = list(W_out = list(shape = c(1L, 1L), dtype = "float32")),
          text_registry = list()
        )
      )
    )
  )
  jsonlite::write_json(
    manifest,
    file.path(tmp, "manifest.json"),
    auto_unbox = TRUE,
    null = "null"
  )
  tree <- list(
    groups = list(
      group_001 = list(
        params = list(
          W_out = strategize:::strenv$jnp$ones(reticulate::tuple(1L, 1L))
        )
      )
    )
  )
  if (reticulate::py_has_attr(ocp, "save_pytree")) {
    ocp$save_pytree(file.path(tmp, "arrays"), tree)
  } else {
    ocp$PyTreeCheckpointer()$save(file.path(tmp, "arrays"), item = tree)
  }

  loaded <- load_conjoint_foundation_bundle(tmp, preload_params = FALSE)
  expect_s3_class(loaded, "conjoint_foundation_model")
  expect_false(is.null(loaded$groups[[group_key]]$fit$neural_model_info$params$W_out))
  expect_equal(loaded$groups[[group_key]]$fit$theta_mean, 1)
})
