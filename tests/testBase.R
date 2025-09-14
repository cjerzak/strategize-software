options(error=NULL)
# devtools::install_github("cjerzak/strategize-software/strategize")
# strategize::build_backend()
library(testthat); library(strategize)
source(file.path("./Documents/strategize-software/strategize", "R", "CS_HelperFxns.R"))

# test of helper functions
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


# Test core strategize functionality
test_that("strategize returns a valid result", {
  skip_if_not_installed("reticulate")

  set.seed(123)
  n <- 500
  W <- matrix(rep(c("A", "B"), length.out = n), ncol = 1)
  Y <- rnorm(n)
  res <- strategize(
    Y = Y,
    W = W,
    lambda = 0.1,
    K = 1,
    nSGD = 1,
    force_gaussian = TRUE,
    nFolds_glm = 1L,
    nMonte_adversarial = 1L,
    nMonte_Qglm = 1L,
    compute_se = FALSE,
    conda_env_required = FALSE
  )
  expect_type(res, "list")
  expect_true("PiStar_point" %in% names(res))
})

# Test cross-validation functionality
test_that("cv_strategize selects lambda", {
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("jax"),
              "jax not available for cv_strategize tests")

  set.seed(123)
  n <- 80
  W <- matrix(rep(c("A", "B"), each = n/2), ncol = 1)
  Y <- rnorm(n)
  cv <- cv_strategize(
    Y = Y,
    W = W,
    lambda = 0.1,
    folds = 2L,
    respondent_id = 1:n,
    respondent_task_id = 1:n,
    K = 1,
    nSGD = 1,
    force_gaussian = TRUE,
    nMonte_adversarial = 1L,
    nMonte_Qglm = 1L,
    nFolds_glm = 1L,
    compute_se = FALSE,
    conda_env_required = FALSE
  )
  expect_type(cv, "list")
  expect_true("lambda" %in% names(cv))
  expect_equal(cv$lambda, 0.1)
})

