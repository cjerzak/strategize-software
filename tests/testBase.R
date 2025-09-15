{
options(error=NULL)
# install.packages("~/Documents/strategize-software/strategize",repos = NULL, type = "source",force = F)
# devtools::install_github("cjerzak/strategize-software/strategize")
# strategize::build_backend()
library(testthat); library(strategize)
source(file.path("~/Documents/strategize-software/strategize", "R", "CS_HelperFxns.R"))

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

# generate data for strategize test s
set.seed(1234321)
n <- 1000
W <- cbind(matrix(sample(c("A", "B"), n, replace = TRUE), ncol = 1),
           matrix(sample(c("A", "B"), n, replace = TRUE), ncol = 1),
           matrix(sample(c("A", "B"), n, replace = TRUE), ncol = 1))
colnames(W) <- c("V1","V2","V3")
respondent_id <-  c(seq_len(n/2),seq_len(n/2))
respondent_task_id <- c(seq_len(n/2),seq_len(n/2))
profile_order <- sample(1:2, n, replace = TRUE)
Y <- as.numeric(ave(
  drop((W == "B") %*% c(0.4, 0.2, 0.3)),                      # latent utility: weights for the three features
  respondent_task_id,                                         # pair each forced-choice task
  FUN = function(g) rank(g, ties.method = "random") == length(g) # winner within each pair
))

# Test core strategize functionality
# outcome_model_type <- "glm"
for(outcome_model_type in c("glm","neural")){
  test_that(sprintf("strategize returns a valid result [%s]", outcome_model_type), {
    res <- {strategize(
      Y = Y,
      W = W,
      lambda = 0.1,
      respondent_id = respondent_id,
      respondent_task_id = respondent_task_id,
      profile_order = profile_order,
      K = 1, # 
      nSGD = 10,
      outcome_model_type = outcome_model_type,
      force_gaussian = TRUE,
      nMonte_adversarial = 10L,
      nMonte_Qglm = 10L,
      compute_se = FALSE,
      conda_env_required = FALSE
    )}
    expect_type(res, "list")
    expect_true("pi_star_point" %in% names(res))
  })
  
  stop("XXX")
  
  # Test cross-validation functionality
  test_that(sprintf("cv_strategize selects lambda [%s]",outcome_model_type), {
    cv_res <- {cv_strategize(
      Y = Y,
      W = W,
      lambda_seq = c(0.01, 0.1),
      folds = 2L,
      respondent_id = respondent_id,
      respondent_task_id = respondent_task_id,
      profile_order = profile_order,
      K = 1,
      nSGD = 100,
      outcome_model_type = outcome_model_type,
      force_gaussian = TRUE,
      nMonte_adversarial = 10L,
      nMonte_Qglm = 10L,
      compute_se = FALSE,
      conda_env_required = FALSE
    )}
    expect_type(cv_res, "list")
    expect_true("lambda" %in% names(cv_res))
    expect_true(cv_res$lambda %in% lambda_seq)
    expect_true("CVInfo" %in% names(cv_res))
  })
}
}
