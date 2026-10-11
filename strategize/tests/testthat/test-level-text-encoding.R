test_that("level text encoding resolves, defaults to hybrid, and keeps legacy bundles global", {
  resolve <- strategize:::neural_resolve_level_text_encoding
  expect_identical(resolve(NULL), "hybrid")
  expect_identical(resolve("GLOBAL"), "global")
  expect_identical(resolve("factor_contrast"), "factor_contrast")
  expect_error(resolve("mean"), "'hybrid', 'factor_contrast' or 'global'")
  expect_false(strategize:::neural_level_text_contrast(list()))
  expect_true(strategize:::neural_level_text_contrast(list(level_text_encoding = "factor_contrast")))
  expect_false(strategize:::neural_level_text_contrast(list(level_text_encoding = "hybrid")))
  expect_identical(strategize:::neural_resolve_token_runtime_config(list(), list())$level_text_encoding, "hybrid")
  expect_identical(
    strategize:::neural_resolve_token_runtime_config(list(), list(level_text_encoding = "global"))$level_text_encoding,
    "global"
  )
})

test_that("level contrast keeps within-factor directions and zero rows", {
  contrast <- strategize:::neural_apply_level_text_contrast_r
  set.seed(2)
  shared <- rnorm(5)
  x <- rbind(shared + c(1, 0, 0, 0, 0), shared + c(0, 1, 0, 0, 0), shared + c(0, 0, 1, 0, 0), 0)
  out <- contrast(x)
  expect_equal(out[4L, ], rep(0, 5))
  expect_equal(sqrt(rowSums(out[1:3, ]^2)), rep(1, 3))
  expect_equal(colSums(out[1:3, ]), rep(0, 5), tolerance = 1e-12)
  # The shared template direction is gone: identical within-factor offsets give
  # identical contrasts regardless of the shared vector.
  expect_equal(contrast(rbind(3 * shared + c(1, 0, 0, 0, 0), 3 * shared + c(0, 1, 0, 0, 0), 3 * shared + c(0, 0, 1, 0, 0), 0)), out)
  two <- contrast(rbind(shared + 1, shared - 1, 0))
  expect_equal(two[1L, ], -two[2L, ])
  expect_equal(unname(contrast(rbind(shared, 0))), matrix(0, 2L, 5L))
})

test_that("JAX level contrast matches the R twin through the projection", {
  skip_on_cran()
  skip_if_no_jax()
  strategize:::initialize_jax()
  np <- reticulate::import("numpy", convert = FALSE)
  set.seed(3)
  x <- rbind(matrix(rnorm(12), 3L, 4L) + 2, 0)
  W <- matrix(rnorm(8), 4L, 2L)
  normalizer <- list(center = rep(0.3, 4), renormalize = TRUE)
  project <- strategize:::neural_project_text_matrix
  W_j <- strategize:::strenv$jnp$array(W, dtype = strategize:::strenv$dtj)
  got <- reticulate::py_to_r(np$asarray(project(x, W_j, normalizer = normalizer, contrast = TRUE)))
  want <- strategize:::neural_apply_level_text_contrast_r(
    strategize:::neural_apply_text_embedding_normalizer_r(x, normalizer)
  ) %*% W
  expect_equal(got, want, tolerance = 1e-5, ignore_attr = TRUE)
  legacy <- reticulate::py_to_r(np$asarray(project(x, W_j, normalizer = normalizer)))
  expect_equal(legacy, strategize:::neural_apply_text_embedding_normalizer_r(x, normalizer) %*% W,
               tolerance = 1e-5, ignore_attr = TRUE)
  # Hybrid: the global projection plus the contrast through its own weights.
  W2 <- matrix(rnorm(8), 4L, 2L)
  info <- list(text_embedding_normalizer = normalizer, level_text_encoding = "hybrid")
  params <- list(W_level_name_text = W_j, W_level_contrast_text = strategize:::strenv$jnp$array(W2, dtype = strategize:::strenv$dtj))
  hybrid <- reticulate::py_to_r(np$asarray(strategize:::neural_project_level_text(x, params, info)))
  expect_equal(hybrid, legacy + strategize:::neural_apply_level_text_contrast_r(
    strategize:::neural_apply_text_embedding_normalizer_r(x, normalizer)) %*% W2, tolerance = 1e-5, ignore_attr = TRUE)
  only <- reticulate::py_to_r(np$asarray(strategize:::neural_project_level_text(
    x, params[1L], modifyList(info, list(level_text_encoding = "factor_contrast")))))
  expect_equal(only, got, tolerance = 1e-5, ignore_attr = TRUE)
})

test_that("text pathway diagnostics describe the contrasted level input", {
  set.seed(4)
  level_text <- list(rbind(matrix(rnorm(15), 3L, 5L) + 3, 0), rbind(rnorm(5) + 3, 0), rbind(matrix(rnorm(10), 2L, 5L) + 3, 0))
  params <- list(W_level_name_text = matrix(rnorm(15), 5L, 3L))
  diag <- function(encoding) {
    out <- strategize:::neural_build_text_pathway_diagnostics(
      params, list(level_name_text = level_text, level_text_encoding = encoding)
    )
    expect_length(out$notes, 0L)
    out$pathway_summaries
  }
  contrast <- diag("factor_contrast")
  legacy <- diag("global")
  expect_identical(contrast$n_rows, 6L)
  expect_lt(contrast$input_constant_share, legacy$input_constant_share)
  params$W_level_contrast_text <- matrix(rnorm(15), 5L, 3L)
  hybrid <- diag("hybrid")
  expect_identical(hybrid$pathway, c("level_name", "level_contrast"))
  expect_equal(hybrid$input_constant_share, c(legacy$input_constant_share, contrast$input_constant_share))
})

test_that("runtime token patches keep the saved level text encoding", {
  make <- strategize:::neural_make_runtime_token_model_info
  saved <- list(level_text_encoding = "factor_contrast")
  runtime <- make(model_dims = 8L)
  expect_false("level_text_encoding" %in% names(runtime))
  for (name in names(runtime)) saved[[name]] <- runtime[[name]]
  expect_true(strategize:::neural_level_text_contrast(saved))
  expect_identical(make(model_dims = 8L, level_text_encoding = "GLOBAL")$level_text_encoding, "global")
})
