## ----setup, include = FALSE---------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  eval = FALSE
)

## ----install------------------------------------------------------------------
# # Install package from GitHub
# devtools::install_github("cjerzak/strategize-software/strategize")
# 
# # Load the package
# library(strategize)
# 
# # Set up Python/JAX backend (required, takes ~5 min first time)
# # This creates a conda environment with JAX for gradient computation
# build_backend(conda_env = "strategize_env")

## ----data-prep----------------------------------------------------------------
# library(strategize)
# 
# # Load your conjoint data
# # Example structure:
# # - Y: binary outcome (0/1 for forced-choice)
# # - W: data.frame with factor columns
# 
# # Create baseline probability list (if using uniform randomization)
# p_list <- list(
#   Gender = c(Male = 0.5, Female = 0.5),
#   Age = c(Young = 0.33, Middle = 0.33, Old = 0.34),
#   Party = c(Dem = 0.5, Rep = 0.5)
# )
# 
# # Or auto-generate from your data:
# # p_list <- create_p_list(W, uniform = TRUE)

## ----run-analysis-------------------------------------------------------------
# # Run strategize to find optimal distribution
# result <- strategize(
#   Y = Y,
#   W = W,
#   lambda = 0.1,         # Regularization (higher = closer to baseline)
#   p_list = p_list,
#   pair_id = pair_id,    # Identifies forced-choice pairs
#   diff = TRUE,          # For forced-choice designs
#   nSGD = 100,           # Gradient descent iterations
#   compute_se = TRUE     # Compute standard errors
# )

## ----cv-----------------------------------------------------------------------
# cv_result <- cv_strategize(
#   Y = Y,
#   W = W,
#   lambda_seq = c(0.01, 0.1, 0.5, 1.0),
#   p_list = p_list,
#   pair_id = pair_id,
#   diff = TRUE,
#   folds = 3
# )
# 
# # Best lambda is automatically selected
# print(cv_result$lambda)

## ----interpret----------------------------------------------------------------
# # View optimal distribution
# result$pi_star_point
# # $k1
# # $k1$Gender
# #   Male Female
# #   0.35   0.65   <- Female candidates preferred (65% vs 50% baseline)
# # $k1$Age
# #   Young Middle    Old
# #    0.40   0.35   0.25   <- Younger candidates preferred
# 
# # Expected outcome under optimal strategy
# result$Q_point
# # [1] 0.58  <- 58% expected win rate (compared to ~50% under baseline)
# 
# # View confidence intervals (if compute_se = TRUE)
# result$pi_star_lb  # Lower bounds
# result$pi_star_ub  # Upper bounds

## ----visualize----------------------------------------------------------------
# # Plot optimal vs baseline distributions
# strategize.plot(
#   pi_star_list = result$pi_star_point,
#   pi_star_se_list = result$pi_star_se,
#   p_list = p_list,
#   main_title = "Optimal vs Baseline Distribution"
# )

## ----presets------------------------------------------------------------------
# # Quick test (not for inference)
# result <- strategize(Y = Y, W = W, lambda = 0.1, nSGD = 20, compute_se = FALSE)
# 
# # Standard analysis
# result <- strategize(Y = Y, W = W, lambda = 0.1, nSGD = 100, compute_se = TRUE)
# 
# # Publication quality
# result <- strategize(Y = Y, W = W, lambda = 0.1, nSGD = 500,
#                      nMonte_Qglm = 500, compute_se = TRUE)

## ----help---------------------------------------------------------------------
# # Function documentation
# ?strategize
# ?cv_strategize
# ?strategize.plot
# 
# # Report issues
# # https://github.com/cjerzak/strategize-software/issues

