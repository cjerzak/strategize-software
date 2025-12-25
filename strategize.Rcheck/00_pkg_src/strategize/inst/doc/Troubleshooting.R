## ----setup, include = FALSE---------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  eval = FALSE
)

## -----------------------------------------------------------------------------
# library(strategize)
# build_backend(conda_env = "strategize_env")

## -----------------------------------------------------------------------------
# reticulate::conda_list()
# # Should show "strategize_env" in the list

## -----------------------------------------------------------------------------
# reticulate::install_miniconda()

## -----------------------------------------------------------------------------
# build_backend(conda_env = "strategize_env")

## -----------------------------------------------------------------------------
# reticulate::use_condaenv("strategize_env")
# reticulate::py_module_available("jax")
# # Should return TRUE

## -----------------------------------------------------------------------------
# Sys.setenv(RETICULATE_PYTHON = "/path/to/conda/envs/strategize_env/bin/python")

## -----------------------------------------------------------------------------
# # Add to ~/.Rprofile
# Sys.setenv(RETICULATE_PYTHON = "~/miniconda3/envs/strategize_env/bin/python")

## -----------------------------------------------------------------------------
# # Wrong: One row per choice task
# # Y has 100 elements (100 tasks)
# # W has 100 rows
# 
# # Correct: One row per profile
# # Y has 200 elements (100 tasks x 2 profiles)
# # W has 200 rows (each profile as separate row)
# 
# # If your data is in wide format, convert to long:
# library(tidyr)
# data_long <- pivot_longer(data_wide, cols = starts_with("profile"))

## -----------------------------------------------------------------------------
# # Auto-generate from data
# p_list <- create_p_list(W, uniform = TRUE)
# 
# # Or manually ensure levels match:
# unique(W$Gender)  # Check what levels exist
# p_list$Gender <- c(Male = 0.5, Female = 0.5)  # Match exactly

## -----------------------------------------------------------------------------
# # Check for NAs
# sum(is.na(Y))
# sum(is.na(W))
# 
# # Remove incomplete cases
# complete <- complete.cases(Y, W)
# Y <- Y[complete]
# W <- W[complete, ]

## -----------------------------------------------------------------------------
# result <- strategize(
#   Y = Y, W = W, lambda = 0.1,
#   nSGD = 500  # Increase from default 100
# )

## -----------------------------------------------------------------------------
# result <- strategize(
#   Y = Y, W = W, lambda = 0.1,
#   learning_rate_max = 0.0001  # Lower than default 0.001
# )

## -----------------------------------------------------------------------------
# result <- strategize(
#   Y = Y, W = W, lambda = 0.1,
#   nMonte_Qglm = 500  # Increase from default 100
# )

## -----------------------------------------------------------------------------
# result <- strategize(Y = Y, W = W, lambda = 0.01)  # Lower lambda

## -----------------------------------------------------------------------------
# # Check if factors actually predict Y
# summary(glm(Y ~ ., data = cbind(Y = Y, W), family = binomial))

## -----------------------------------------------------------------------------
# cv_result <- cv_strategize(
#   Y = Y, W = W,
#   lambda_seq = c(0.001, 0.01, 0.1, 0.5, 1.0)
# )

## -----------------------------------------------------------------------------
# result <- strategize(
#   Y = Y, W = W, lambda = 0.1,
#   nMonte_adversarial = 10,  # Reduce from default
#   nMonte_Qglm = 50          # Reduce from default
# )

## -----------------------------------------------------------------------------
# result <- strategize(Y = Y, W = W, lambda = 0.1, nSGD = 50)

## -----------------------------------------------------------------------------
# # Use subset for testing
# idx <- sample(nrow(W), 1000)
# result_test <- strategize(Y = Y[idx], W = W[idx, ], lambda = 0.1)

## -----------------------------------------------------------------------------
# # More Monte Carlo samples for tighter SEs
# result <- strategize(
#   Y = Y, W = W, lambda = 0.1,
#   nMonte_Qglm = 500,
#   compute_se = TRUE
# )

## -----------------------------------------------------------------------------
# table(W$Gender)  # Ensure adequate sample in each level

## -----------------------------------------------------------------------------
# result <- strategize(
#   Y = Y, W = W, lambda = 0.1,
#   adversarial = TRUE,
#   competing_group_variable_respondent = respondent_party,  # e.g., "Dem" or "Rep"
#   competing_group_variable_candidate = candidate_party     # Same for candidates
# )

## -----------------------------------------------------------------------------
# ?strategize
# ?cv_strategize

