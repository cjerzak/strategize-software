library(testthat)

# Load helper functions directly
source(file.path("strategize", "R", "CS_HelperFxns.R"))

test_that("toSimplex returns a valid probability vector", {
  x <- c(0.1, -0.2, 0.3)
  s <- toSimplex(x)
  expect_equal(sum(s), 1, tolerance = 1e-7)
  expect_true(all(s >= 0))
})

test_that("ess_fxn computes effective sample size correctly", {
  w <- c(1, 1, 1, 1)
  expect_equal(ess_fxn(w), 4)

  w2 <- c(1, 0.5)
  expect_equal(ess_fxn(w2), sum(w2)^2 / sum(w2^2))
})

test_that("RescaleFxn rescales and recenters", {
  x <- c(-1, 0, 1)
  res <- RescaleFxn(x, estMean = 2, estSD = 3)
  expect_equal(res, x * 3 + 2)

  res_no_center <- RescaleFxn(x, estMean = 2, estSD = 3, center = FALSE)
  expect_equal(res_no_center, x * 3)
})

test_that("getSE handles missing values", {
  vals <- c(1, 2, 3, NA)
  expect_equal(getSE(vals), sqrt(var(vals, na.rm = TRUE) / 3))
})

