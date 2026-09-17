test_that("recurrence controls are explicit, validated and legacy safe", {
  resolve <- strategize:::neural_resolve_transformer_loop
  expect_false(resolve()$enabled)
  expect_identical(resolve(TRUE, 8L), list(enabled = TRUE, iterations = 2L,
    prelude_layers = 1L, coda_layers = 1L, backprop_iterations = 0L))
  expect_identical(resolve(TRUE, 1L)$prelude_layers, 0L)
  expect_identical(resolve(TRUE, 1L)$coda_layers, 0L)
  expect_identical(resolve(TRUE, 2L)$coda_layers, 0L)
  expect_equal(strategize:::neural_transformer_effective_depth(resolve(TRUE,8L),8L),14)
  for (bad in list(0, -1, 1.5, NA_real_, Inf, "2", c(1,2)))
    expect_error(resolve(list(iterations = bad)), "integer")
  expect_error(resolve(list(backprop_iterations = -1)), "integer")
  expect_error(resolve(list(prelude_layers=1,coda_layers=1),2L), "recurrent core")
  expect_error(resolve(list(typo=2)), "Unknown")
  expect_error(resolve(list(2)), "named list")
  expect_error(resolve(list(enabled=NA)), "TRUE or FALSE")
  expect_error(strategize:::neural_transformer_loop_config(list(
    model_depth=3L,transformer_loop=TRUE,residual_mode="full_attn")), "standard")
  info <- strategize:::neural_make_transformer_model_info(3L,16L,4L,4L,transformer_loop=TRUE)
  key <- strategize:::neural_model_jit_cache_key(info)
  info$transformer_loop$iterations <- 4L
  expect_false(identical(key,strategize:::neural_model_jit_cache_key(info)))
  expect_true("looped_transformer_v1" %in% strategize::strategize_fm_backend()$capabilities)
})
