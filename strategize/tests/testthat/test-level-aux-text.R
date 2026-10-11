test_that("distinctive level text drops words shared by every level of a factor", {
  distinct <- strategize:::neural_level_distinctive_text
  expect_identical(
    distinct(c("Stronger than in Nicaragua", "Similar to Nicaragua", "Weaker than in Nicaragua")),
    c("Stronger than in", "Similar to", "Weaker than in")
  )
  expect_identical(
    distinct(c("Ist für höhere Zölle auf Waren", "Ist für niedrigere Zölle auf Waren")),
    c("höhere", "niedrigere")
  )
  expect_identical(distinct(c("10-35k: 15%", "10-35k: 25%")), c("15%", "25%"))
  # Nothing shared, an empty remainder, or duplicate remainders keep the level.
  expect_identical(distinct(c("Female", "Male")), c("Female", "Male"))
  expect_identical(distinct(c("Tax", "Tax rise")), c("Tax", "Tax rise"))
  expect_identical(distinct(c("a x b", "a x c b")), c("a x b", "a x c b"))
  expect_identical(distinct("Only level"), "Only level")
})

test_that("auxiliary level text is validated against the level text rows", {
  validate <- strategize:::neural_validate_level_aux_text
  level_text <- list(matrix(1, 3L, 4L), matrix(1, 2L, 4L))
  expect_silent(validate(NULL, level_text))
  expect_silent(validate(list(matrix(0, 3L, 4L), matrix(0, 2L, 4L)), level_text))
  expect_error(validate(list(matrix(0, 2L, 4L), matrix(0, 2L, 4L)), level_text), "rows of level_name_text")
  expect_error(validate(list(matrix(0, 3L, 4L)), level_text), "one matrix per factor")
})

test_that("runtime token patches replace or remove auxiliary rows with the level text", {
  make <- strategize:::neural_make_runtime_token_model_info
  patch <- function(saved, runtime) {
    for (name in names(runtime)) saved[[name]] <- runtime[[name]]
    saved
  }
  saved <- list(level_aux_text = list(matrix(1, 2L, 3L)))
  # A patch without level text leaves the saved rows alone.
  runtime <- make(model_dims = 8L)
  expect_false("level_aux_text" %in% names(runtime))
  expect_false(is.null(patch(saved, runtime)$level_aux_text))
  skip_on_cran()
  skip_if_no_jax()
  strategize:::initialize_jax()
  # New level text without auxiliary rows removes the stale training rows ...
  struct <- strategize:::neural_make_default_fused_structural_info(factor_names = "f", factor_levels = 3L)
  make <- function(...) {
    strategize:::neural_make_runtime_token_model_info(
      factor_struct_matrix = struct$factor_struct_matrix,
      level_struct_matrices = struct$level_struct_matrices,
      factor_struct_feature_names = struct$factor_struct_feature_names,
      level_struct_feature_names = struct$level_struct_feature_names,
      factor_name_text = matrix(0.5, 1L, 3L), ...
    )
  }
  level_text <- list(matrix(0.5, 3L, 3L))
  runtime <- make(model_dims = 8L, level_name_text = level_text)
  expect_true("level_aux_text" %in% names(runtime))
  expect_null(patch(saved, runtime)$level_aux_text)
  # ... and new auxiliary rows replace them.
  runtime <- make(model_dims = 8L, level_name_text = level_text, level_aux_text = list(matrix(0.25, 3L, 3L)))
  expect_identical(as.integer(strategize:::neural_dim2(patch(saved, runtime)$level_aux_text[[1L]])[[1L]]), 3L)
  expect_error(make(model_dims = 8L, level_name_text = level_text, level_aux_text = list(matrix(0, 2L, 3L))),
               "rows of level_name_text")
})

test_that("auxiliary projection adds the distinctive rows and keeps absent rows zero", {
  skip_on_cran()
  skip_if_no_jax()
  strategize:::initialize_jax()
  np <- reticulate::import("numpy", convert = FALSE)
  jnp <- strategize:::strenv$jnp
  set.seed(5)
  x <- rbind(matrix(rnorm(12), 3L, 4L) + 2, 0)
  aux <- rbind(matrix(rnorm(8), 2L, 4L), 0, 0)
  W <- matrix(rnorm(8), 4L, 2L)
  W_aux <- matrix(rnorm(8), 4L, 2L)
  normalizer <- list(center = rep(0.3, 4), renormalize = TRUE)
  info <- list(text_embedding_normalizer = normalizer, level_text_encoding = "global")
  params <- list(W_level_name_text = jnp$array(W, dtype = strategize:::strenv$dtj),
                 W_level_aux_text = jnp$array(W_aux, dtype = strategize:::strenv$dtj))
  norm_r <- strategize:::neural_apply_text_embedding_normalizer_r
  got <- reticulate::py_to_r(np$asarray(strategize:::neural_project_level_text(x, params, info, aux_matrix = aux)))
  want <- norm_r(x, normalizer) %*% W + norm_r(aux, normalizer) %*% W_aux
  expect_equal(got, want, tolerance = 1e-5, ignore_attr = TRUE)
  # Rows without auxiliary text receive no auxiliary contribution.
  expect_equal(got[3:4, ], (norm_r(x, normalizer) %*% W)[3:4, ], tolerance = 1e-5, ignore_attr = TRUE)
  # Without aux rows, or without the weight, the projection is unchanged.
  base <- reticulate::py_to_r(np$asarray(strategize:::neural_project_level_text(x, params[1L], info)))
  expect_equal(reticulate::py_to_r(np$asarray(strategize:::neural_project_level_text(x, params, info))),
               base, tolerance = 1e-6, ignore_attr = TRUE)
  expect_equal(reticulate::py_to_r(np$asarray(strategize:::neural_project_level_text(x, params[1L], info, aux_matrix = aux))),
               base, tolerance = 1e-6, ignore_attr = TRUE)
})

test_that("prediction-time schemas rebuild auxiliary level text or drop it", {
  embed <- function(x) t(vapply(x, function(s) c(nchar(s), utf8ToInt(substr(s, 1, 1)) / 100, 1), numeric(3)))
  names_list <- list(f = list(c("Stronger than in X", "Weaker than in X")))
  got <- strategize:::cs2step_neural_schema_level_aux_text(list(), names_list, text_dim = 3L, text_embedding_fn = embed)
  expect_identical(dim(got$f), c(3L, 3L))
  expect_equal(got$f[1:2, ], embed(c("Stronger", "Weaker")), ignore_attr = TRUE)
  expect_equal(unname(got$f[3L, ]), rep(0, 3))
  expect_warning(
    expect_null(strategize:::cs2step_neural_schema_level_aux_text(list(), names_list, text_dim = 3L)),
    "auxiliary level text"
  )
})
