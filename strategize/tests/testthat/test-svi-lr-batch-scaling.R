test_that("square-root batch scaling of the SVI learning rate", {
  scale <- strategize:::neural_svi_lr_batch_scale
  expect_identical(scale(256, 1e6, "none", 64), 1)
  expect_equal(scale(256, 1e6, "sqrt", 64), 2)
  expect_equal(scale(16, 1e6, "sqrt", 64), 0.5)
  expect_equal(scale(64, 1e6, "sqrt", 64), 1)
  expect_equal(scale(1024, 100, "sqrt", 25), 2)  # batch capped at the observation count
  expect_identical(scale(NA, 100, "sqrt", 64), 1)
  expect_error(scale(64, 100, "linear", 64), "'none' or 'sqrt'")
  expect_error(scale(64, 100, "sqrt", 0), "positive finite")
})
