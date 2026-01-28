## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

# Check if code should be evaluated
# By default, code is NOT executed. This ensures the vignette builds
# successfully during R CMD check and on systems without the conda environment.
#
# To enable code execution when building the vignette manually:
#   Sys.setenv(STRATEGIZE_RUN_VIGNETTE = "true")
#   rmarkdown::render("vignettes/MainVignette.Rmd")

CONDA_AVAILABLE <- FALSE

if (identical(Sys.getenv("STRATEGIZE_RUN_VIGNETTE"), "true")) {
  # User explicitly opted in to run code - check if conda env exists
  CONDA_AVAILABLE <- tryCatch({
    if (requireNamespace("reticulate", quietly = TRUE)) {
      conda_list <- reticulate::conda_list()
      "strategize" %in% conda_list$name || "strategize_env" %in% conda_list$name
    } else {
      FALSE
    }
  }, error = function(e) FALSE)
}

# Inform about execution status (only when not in check)
if (!CONDA_AVAILABLE && interactive()) {
  message("Note: Vignette code chunks are not executed by default. ",
          "To run code: Sys.setenv(STRATEGIZE_RUN_VIGNETTE='true') before rendering.")
}

## -----------------------------------------------------------------------------
# A small example: Suppose we have a forced-choice design with 1000
# respondent-profile observations, each profile has 3 factors.

# Create a simple data frame 'my_data_red' that includes:
#  - respondent/candidate group variables for adversarial scenarios
#  - factors that will populate FACTOR_MAT_
#  - an outcome Yobs
n <- 4000
my_data_red <- data.frame(
  # For adversarial grouping:
  R_Partisanship = sample(c("Democrat", "Republican"), n, replace = TRUE),
  Party.affiliation_clean = sample(c("Democrat", "Republican"), n, replace = TRUE),
  Party.competition = sample(c("Same", "Different"), n, replace = TRUE),
  
  # For indexing each respondent and task:
  respondentIndex = c(replicate(2,1:(n/2))),
  task = c(replicate(n,1)),
  profile_order = c(replicate((n/2),1),replicate((n/2),2)), # for every respondent and task, there is a profile order 
  
  # Some factors for the conjoint:
  Sex = sample(c("Male", "Female"), n, replace = TRUE),
  Age = sample(c("36-51","52-67","68-76"), n, replace = TRUE),
  Family = sample(c("Single (divorced)", "Single (never married)",
                    "Married (no child)", "Married (two children)"),
                  n, replace = TRUE)
)

# Suppose Yobs is our outcome (e.g., was this profile chosen?)
Yobs <- c(y_ <- rbinom(n/2, 1, prob = 0.5), 1 - y_)  # purely random for illustration

# Construct 'FACTOR_MAT_' from the factor columns we want to optimize over:
FACTOR_MAT_ <- my_data_red[, c("Sex","Age","Family")]

# (Optional) A cluster variable for variance computations:
cluster_var <- my_data_red$respondentIndex   # or something more meaningful

# Pair or choice-set IDs (if forced-choice):
pair_id <- paste0(my_data_red$respondentIndex, "_", my_data_red$task)

# Original assignment probability list (assume uniform or known):
p_list <- list(
  Sex = c("Female" = 0.5, "Male" = 0.5),
  Age = c("36-51" = 1/3, "52-67" = 1/3, "68-76" = 1/3),
  Family = c("Single (divorced)" = 0.25,
             "Single (never married)" = 0.25,
             "Married (no child)" = 0.25,
             "Married (two children)" = 0.25)
)

# If you want 'slates' of candidate features by party (for multi-stage or adversarial):
SlateList <- list(
  "Democratic" = p_list,
  "Republican" = p_list
)

# Illustrative hyperparameter settings 
X <- rbind(X <- matrix(rnorm(n/2 * 2), n/2, 2), X)  # optional respondent covariates
nSGD_TwoStep <- 200
nMonte_adversarial <- 5
regularizationType <- "KL"    # penalty type ("KL","L2","LogMaxProb", etc.)
conda_env <- "strategize"
GLOBAL_NFOLDS <- 3
nFolds_glm <- 3
lambda_MaxMin <- 0.1
LAMBDA_SEQ <- c(0.01, lambda_MaxMin)

## ----average-case, eval=CONDA_AVAILABLE---------------------------------------
# library(strategize)
# 
# AdversarialValue <- FALSE
# #if AdversarialValue = FALSE, use 'cv_strategize' for standard cross-validation and single-agent optimization.
# 
# res_avecase <- {cv_strategize(
#   # Core inputs:
#   Y = Yobs,
#   W = FACTOR_MAT_,
#   X = NULL,
# 
#   # Regularization or penalty inputs:
#   lambda = LAMBDA_SEQ,
#   penalty_type = regularizationType,
#   use_regularization = TRUE,
# 
#   # Clustering / grouping:
#   varcov_cluster_variable = cluster_var, # in applications, this could be respondent
#   competing_group_variable_respondent = NULL,
#   competing_group_variable_candidate = NULL,
#   competing_group_competition_variable_candidate = NULL,
# 
#   # IDs and factor specification:
#   pair_id = pair_id,
#   respondent_id = my_data_red$respondentIndex,
#   respondent_task_id = my_data_red$task,
#   profile_order = my_data_red$profile_order,
# 
#   # Probability lists:
#   p_list = p_list,
#   slate_list = SlateList,
# 
#   # Optimization controls:
#   K = 1L,
#   force_gaussian = FALSE,
#   nSGD = nSGD_TwoStep,
#   nMonte_adversarial = nMonte_adversarial,
#   compute_se = TRUE,
# 
#   # If we only want a 'difference' structure on the first iteration, for example:
#   diff = TRUE,
# 
#   # Cross-validation folds (only used if calling cv_strategize):
#   folds = GLOBAL_NFOLDS,
#   nFolds_glm = nFolds_glm,
# 
#   # Toggling between single-agent vs. adversarial:
#   adversarial = AdversarialValue,
# 
#   # Python environment management:
#   conda_env = conda_env,
#   conda_env_required = TRUE
# )}
# 
# # Inspect results:
# res_avecase$pi_star_point     # The learned factor-level distribution(s)
# res_avecase$Q_point_mEst     # The estimated performance or outcome
# 
# res_avecase$PiStar_se        # Standard errors (if compute_se=TRUE)
# res_avecase$Q_se_mEst        # Standard errors

## ----plotting, eval=CONDA_AVAILABLE-------------------------------------------
# oldpar <- par(mfrow=c(1,3))
# for(d in seq_along(p_list)){
#   old_probs <- p_list[[d]]
#   new_probs <- res_avecase$PiStar_point[[1]][[d]]  # if single cluster
# 
#   barplot( rbind(old_probs, new_probs), beside=TRUE,
#            col=c("gray","blue"), ylim=c(0,1),
#            main=names(p_list)[d], ylab="Prob.")
#   legend("topright", legend=c("Original", "Optimal"), fill=c("gray","blue"), cex=0.8)
# }
# par(oldpar)

## ----adversarial, eval=CONDA_AVAILABLE----------------------------------------
# AdversarialValue <- TRUE
# # if AdversarialValue = TRUE, use 'strategize' for a max-min problem;
# 
# res_adversarial <- {strategize(
#   # Core inputs:
#   Y = Yobs,
#   W = FACTOR_MAT_,
#   X = NULL,
# 
#   # Regularization or penalty inputs:
#   lambda = lambda_MaxMin,
#   penalty_type = regularizationType,
#   use_regularization = TRUE,
# 
#   # Clustering / grouping:
#   varcov_cluster_variable = cluster_var,
#   competing_group_variable_respondent = my_data_red$R_Partisanship,
#   competing_group_variable_candidate = my_data_red$Party.affiliation_clean,
#   competing_group_competition_variable_candidate = my_data_red$Party.competition,
# 
#   # IDs and factor specification:
#   pair_id = pair_id,
#   respondent_id = my_data_red$respondentIndex,
#   respondent_task_id = my_data_red$task,
#   profile_order = my_data_red$profile,
# 
#   # Probability lists:
#   p_list = p_list,
#   slate_list = SlateList,
# 
#   # Optimization controls:
#   K = 1L,
#   force_gaussian = FALSE,
#   nSGD = nSGD_TwoStep,
#   nMonte_adversarial = nMonte_adversarial,
#   compute_se = TRUE,
# 
#   # If we only want a 'difference' structure on the first iteration, for example:
#   diff = TRUE,
# 
#   # Cross-validation folds (only used if calling cv_strategize):
#   folds = GLOBAL_NFOLDS,
#   nFolds_glm = nFolds_glm,
# 
#   # Toggling between single-agent vs. adversarial:
#   adversarial = AdversarialValue,
# 
#   # Python environment management:
#   conda_env = conda_env,
#   conda_env_required = TRUE
# )}
# 
# # Inspect results:
# res_adversarial$PiStar_point$k1     # The learned factor-level distribution(s)
# res_adversarial$PiStar_point$k2     # The learned factor-level distribution(s)
# res_adversarial$Q_point_mEst     # The estimated performance or outcome
# 
# res_adversarial$PiStar_se$k1        # Standard errors (if compute_se=TRUE)
# res_adversarial$PiStar_se$k2        # Standard errors (if compute_se=TRUE)
# res_adversarial$Q_se_mEst        # Standard errors (if compute_se=TRUE)

## ----cluster-based, eval=CONDA_AVAILABLE--------------------------------------
# res_clust <- {cv_strategize(
#   # Core inputs:
#   Y = Yobs,
#   W = FACTOR_MAT_,
#   X = X,
# 
#   # Regularization or penalty inputs:
#   lambda = LAMBDA_SEQ,
#   penalty_type = regularizationType,
#   use_regularization = TRUE,
# 
#   # Clustering / grouping:
#   varcov_cluster_variable = cluster_var, # in applications, this could be respondent
#   competing_group_variable_respondent = NULL,
#   competing_group_variable_candidate = NULL,
#   competing_group_competition_variable_candidate = NULL,
# 
#   # IDs and factor specification:
#   pair_id = pair_id,
#   respondent_id = my_data_red$respondentIndex,
#   respondent_task_id = my_data_red$task,
#   profile_order = my_data_red$profile_order,
# 
#   # Probability lists:
#   p_list = p_list,
#   slate_list = SlateList,
# 
#   # Optimization controls:
#   K = 3L,
#   force_gaussian = FALSE,
#   nSGD = nSGD_TwoStep,
#   nMonte_adversarial = nMonte_adversarial,
#   compute_se = TRUE,
# 
#   # Cross-validation folds (only used if calling cv_strategize):
#   folds = GLOBAL_NFOLDS,
#   nFolds_glm = nFolds_glm,
# 
#   # Toggling between single-agent vs. adversarial:
#   adversarial = FALSE,
# 
#   # Python environment management:
#   conda_env = conda_env,
#   conda_env_required = TRUE
# )}
# 
# # Inspect results:
# res_clust$PiStar_point     # The learned factor-level distribution(s)
# res_clust$Q_point_mEst     # The estimated performance or outcome
# 
# res_clust$PiStar_se        # Standard errors (if compute_se=TRUE)
# res_clust$Q_se_mEst

