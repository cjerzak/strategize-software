test_that("recurrence controls are explicit, validated and legacy safe", {
  resolve <- strategize:::neural_resolve_transformer_loop
  expect_false(resolve()$enabled)
  expect_identical(resolve(TRUE, 8L), list(
    enabled = TRUE,
    iterations = 2L,
    prelude_layers = 1L,
    coda_layers = 1L,
    backprop_iterations = 0L,
    training_iterations = NULL,
    training_probabilities = NULL,
    reinjection = "scaled_add",
    normalize_core_gradients = FALSE
  ))
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
  sampled <- resolve(list(
    training_iterations = 1:4,
    training_probabilities = c(2, 5, 2, 1),
    reinjection = "RMS_GATED",
    normalize_core_gradients = TRUE
  ), 8L)
  expect_equal(sampled$training_probabilities, c(.2, .5, .2, .1))
  expect_identical(sampled$reinjection, "rms_gated")
  expect_equal(strategize:::neural_transformer_effective_depth(sampled, 8L), 15.2)
  expect_error(resolve(list(training_iterations = c(1, 1))), "unique positive")
  expect_error(resolve(list(training_probabilities = c(.5, .5))), "requires")
  expect_error(resolve(list(reinjection = "concat")), "scaled_add")
  expect_error(strategize:::neural_transformer_loop_config(list(
    model_depth=3L,transformer_loop=TRUE,residual_mode="full_attn")), "standard")
  info <- strategize:::neural_make_transformer_model_info(3L,16L,4L,4L,transformer_loop=TRUE)
  key <- strategize:::neural_model_jit_cache_key(info)
  info$transformer_loop$iterations <- 4L
  expect_false(identical(key,strategize:::neural_model_jit_cache_key(info)))
  expect_true("looped_transformer_v1" %in% strategize::strategize_fm_backend()$capabilities)
  expect_true("looped_transformer_v2" %in% strategize::strategize_fm_backend()$capabilities)
})

test_that("recurrent prediction diagnostics track NLL changes", {
  logits <- cbind(c(-1, 1), c(-2, 2), c(-.5, .5))
  out <- strategize:::neural_recurrent_prediction_diagnostics(logits, c(0, 1))
  expect_identical(out$iteration, 1:3)
  expect_gt(out$nll[1], out$nll[2])
  expect_lt(out$nll[1], out$nll[3])
  expect_equal(out$fraction_examples_improved[2], 1)
  expect_equal(out$fraction_examples_improved[3], 0)
})
