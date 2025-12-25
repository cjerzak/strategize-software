pkgname <- "strategize"
source(file.path(R.home("share"), "R", "examples-header.R"))
options(warn = 1)
base::assign(".ExTimings", "strategize-Ex.timings", pos = 'CheckExEnv')
base::cat("name\tuser\tsystem\telapsed\n", file=base::get(".ExTimings", pos = 'CheckExEnv'))
base::assign(".format_ptime",
function(x) {
  if(!is.na(x[4L])) x[1L] <- x[1L] + x[4L]
  if(!is.na(x[5L])) x[2L] <- x[2L] + x[5L]
  options(OutDec = '.')
  format(x[1L:3L], digits = 7L)
},
pos = 'CheckExEnv')

### * </HEADER>
library('strategize')

base::assign(".oldSearch", base::search(), pos = 'CheckExEnv')
base::assign(".old_wd", base::getwd(), pos = 'CheckExEnv')
cleanEx()
nameEx("build_backend")
### * build_backend

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: build_backend
### Title: Build the environment for 'strategize'. Creates a conda
###   environment in which JAX and NumPy are installed. Users may also
###   create such an environment themselves.
### Aliases: build_backend

### ** Examples

## Not run: 
##D # Create a conda environment named "strategize"
##D # and install the required Python packages (jax, numpy, etc.)
##D build_backend(conda_env = "strategize", conda = "auto")
##D 
##D # If you want to specify a particular conda path:
##D # build_backend(conda_env = "strategize", conda = "/usr/local/bin/conda")
## End(Not run)




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("build_backend", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("create_p_list")
### * create_p_list

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: create_p_list
### Title: Create Baseline Probability List from Data
### Aliases: create_p_list

### ** Examples

# Create sample factor matrix
W <- data.frame(
  Gender = c("Male", "Female", "Male", "Female"),
  Age = c("Young", "Old", "Young", "Old")
)

# Uniform probabilities (for balanced designs)
p_list_uniform <- create_p_list(W, uniform = TRUE)
print(p_list_uniform)
# $Gender
#   Male Female
#    0.5    0.5
# $Age
#   Young    Old
#     0.5    0.5

# Observed frequencies
p_list_observed <- create_p_list(W, uniform = FALSE)
print(p_list_observed)




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("create_p_list", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("cv_strategize")
### * cv_strategize

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: cv_strategize
### Title: Cross-validation for Optimal Stochastic Interventions in
###   Conjoint Analysis
### Aliases: cv_strategize

### ** Examples

## No test: 
# ================================================
# Cross-validation to select regularization lambda
# ================================================
set.seed(123)
n <- 400  # profiles (200 pairs)

# Generate factor matrix
W <- data.frame(
  Gender = sample(c("Male", "Female"), n, replace = TRUE),
  Age = sample(c("35", "50", "65"), n, replace = TRUE),
  Party = sample(c("Dem", "Rep"), n, replace = TRUE)
)

# Simulate outcome with true effects
latent <- 0.2 * (W$Gender == "Female") + 0.15 * (W$Age == "35")
prob <- plogis(latent)

# Create paired forced-choice structure
pair_id <- rep(1:(n/2), each = 2)
Y <- numeric(n)
for (p in unique(pair_id)) {
  idx <- which(pair_id == p)
  winner <- sample(idx, 1, prob = prob[idx])
  Y[idx] <- as.integer(seq_along(idx) == which(idx == winner))
}
profile_order <- rep(1:2, n/2)

# Cross-validate over lambda values
# Lower lambda = less regularization = further from baseline
cv_result <- cv_strategize(
  Y = Y,
  W = W,
  lambda_seq = c(0.01, 0.1, 0.5, 1.0),
  folds = 2,
  pair_id = pair_id,
  respondent_id = pair_id,
  profile_order = profile_order,
  diff = TRUE,
  nSGD = 50,
  compute_se = FALSE
)

# View CV results and selected lambda
print(cv_result$lambda)       # Optimal lambda
print(cv_result$CVInfo)       # Performance at each lambda
print(cv_result$pi_star_point) # Optimal distribution
print(cv_result$Q_point)       # Expected outcome
## End(No test)




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("cv_strategize", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("plot_best_response_curves")
### * plot_best_response_curves

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: plot_best_response_curves
### Title: Plot Dimension-by-Dimension Best-Response Curves from
###   Adversarial 'strategize()' Output
### Aliases: plot_best_response_curves

### ** Examples

## Not run: 
##D # =====================================================
##D # Visualize best-response curves in adversarial mode
##D # =====================================================
##D # First, fit an adversarial strategize model
##D set.seed(42)
##D n <- 400
##D 
##D # Generate data with party structure
##D W <- data.frame(
##D   Gender = sample(c("Male", "Female"), n, replace = TRUE),
##D   Age = sample(c("Young", "Middle", "Old"), n, replace = TRUE)
##D )
##D 
##D # Party affiliations for respondents and candidates
##D respondent_party <- sample(c("Dem", "Rep"), n/2, replace = TRUE)
##D candidate_party <- rep(c("Dem", "Rep"), n/2)
##D 
##D Y <- rbinom(n, 1, 0.5)  # Simplified outcome
##D 
##D # Fit adversarial model
##D adv_result <- strategize(
##D   Y = Y,
##D   W = W,
##D   lambda = 0.1,
##D   adversarial = TRUE,
##D   competing_group_variable_respondent = rep(respondent_party, each = 2),
##D   competing_group_variable_candidate = candidate_party,
##D   nSGD = 100
##D )
##D 
##D # Plot best-response curves for Gender dimension (d_ = 1)
##D # Shows how each party's optimal Gender distribution responds
##D # to changes in the other party's Gender distribution
##D plot_best_response_curves(
##D   res = adv_result,
##D   d_ = 1,  # Gender is first factor
##D   nPoints_br = 50,
##D   title = "Gender: Best-Response Curves",
##D   col_ast = "blue",   # Democrats
##D   col_dag = "red"     # Republicans
##D )
##D 
##D # Intersection point indicates Nash equilibrium for this dimension
## End(Not run)




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("plot_best_response_curves", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("strategize")
### * strategize

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: strategize
### Title: Estimate Optimal (or Adversarial) Stochastic Interventions for
###   Conjoint Experiments
### Aliases: strategize

### ** Examples

## No test: 
# ============================================
# Example 1: Basic single-agent optimization
# ============================================
# Generate synthetic conjoint data
set.seed(42)
n <- 400  # Number of profiles (200 pairs)

# Factor matrix: candidate attributes
W <- data.frame(
  Gender = sample(c("Male", "Female"), n, replace = TRUE),
  Age = sample(c("Young", "Middle", "Old"), n, replace = TRUE),
  Party = sample(c("Dem", "Rep"), n, replace = TRUE)
)

# Simulate outcome: Female + Young candidates preferred
latent <- 0.3 * (W$Gender == "Female") +
          0.2 * (W$Age == "Young") -
          0.1 * (W$Age == "Old")
prob <- plogis(latent)

# Paired forced-choice: within each pair, one wins
pair_id <- rep(1:(n/2), each = 2)
Y <- numeric(n)
for (p in unique(pair_id)) {
  idx <- which(pair_id == p)
  winner <- sample(idx, 1, prob = prob[idx])
  Y[idx] <- as.integer(seq_along(idx) == which(idx == winner))
}
profile_order <- rep(1:2, n/2)

# Baseline probabilities (uniform assignment)
p_list <- list(
  Gender = c(Male = 0.5, Female = 0.5),
  Age = c(Young = 1/3, Middle = 1/3, Old = 1/3),
  Party = c(Dem = 0.5, Rep = 0.5)
)

# Run strategize to find optimal distribution
# (requires conda environment with JAX - see build_backend())
result <- strategize(
  Y = Y,
  W = W,
  lambda = 0.1,
  pair_id = pair_id,
  respondent_id = pair_id,
  respondent_task_id = pair_id,
  profile_order = profile_order,
  p_list = p_list,
  diff = TRUE,
  nSGD = 50,
  compute_se = FALSE
)

# View optimal distribution
print(result$pi_star_point)

# View expected outcome under optimal strategy
print(result$Q_point)
## End(No test)




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("strategize", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("strategize.plot")
### * strategize.plot

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: strategize.plot
### Title: Plot Estimated Probabilities for Hypothetical Scenarios
### Aliases: strategize.plot

### ** Examples

# =============================================
# Visualize optimal vs baseline distributions
# =============================================
# This function works without JAX - just needs the result structure

# Create mock strategize result for plotting
# (In practice, use output from strategize())
pi_star_list <- list(k1 = list(
  Gender = c(Male = 0.35, Female = 0.65),
  Age = c(Young = 0.45, Middle = 0.30, Old = 0.25),
  Party = c(Dem = 0.40, Rep = 0.60)
))

pi_star_se_list <- list(k1 = list(
  Gender = c(Male = 0.04, Female = 0.04),
  Age = c(Young = 0.03, Middle = 0.03, Old = 0.03),
  Party = c(Dem = 0.05, Rep = 0.05)
))

# Baseline (original assignment) probabilities
p_list <- list(
  Gender = c(Male = 0.5, Female = 0.5),
  Age = c(Young = 0.33, Middle = 0.33, Old = 0.34),
  Party = c(Dem = 0.5, Rep = 0.5)
)

# Plot comparing optimal to baseline
strategize.plot(
  pi_star_list = pi_star_list,
  pi_star_se_list = pi_star_se_list,
  p_list = p_list,
  main_title = "Optimal vs Baseline Distribution",
  ticks_type = "assignmentProbs"  # Show baseline as reference ticks
)




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("strategize.plot", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("strategize_onestep")
### * strategize_onestep

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: strategize_onestep
### Title: Estimate an Optimal (or Adversarial) Stochastic Intervention for
###   Conjoint Analysis Using a One-Step M-estimation Approach
### Aliases: strategize_onestep

### ** Examples

## Not run: 
##D # ================================================
##D # One-step M-estimation approach
##D # ================================================
##D # This approach simultaneously estimates the outcome model and
##D # optimal distribution, rather than the two-step approach
##D 
##D set.seed(123)
##D n <- 400
##D 
##D # Generate factor matrix
##D W <- data.frame(
##D   Gender = sample(c("Male", "Female"), n, replace = TRUE),
##D   Age = sample(c("Young", "Middle", "Old"), n, replace = TRUE),
##D   Party = sample(c("Dem", "Rep"), n, replace = TRUE)
##D )
##D 
##D # Simulate outcome (Female and Young preferred)
##D latent <- 0.3 * (W$Gender == "Female") + 0.2 * (W$Age == "Young")
##D Y <- rbinom(n, 1, plogis(latent))
##D 
##D # Baseline probabilities (uniform)
##D p_list <- list(
##D   Gender = c(Male = 0.5, Female = 0.5),
##D   Age = c(Young = 1/3, Middle = 1/3, Old = 1/3),
##D   Party = c(Dem = 0.5, Rep = 0.5)
##D )
##D 
##D # Optional respondent covariates
##D X <- matrix(rnorm(n * 2), n, 2)
##D colnames(X) <- c("Income", "Education")
##D 
##D # Run one-step estimation
##D result <- strategize_onestep(
##D   W = W,
##D   Y = Y,
##D   X = X,
##D   p_list = p_list,
##D   nSGD = 100,
##D   penalty_type = "LogMaxProb",
##D   lambda_seq = c(0.01, 0.1),
##D   test_fraction = 0.3,
##D   quiet = TRUE
##D )
##D 
##D # View optimal distribution
##D print(result$pi_star_point)
##D 
##D # View expected outcome under optimal strategy
##D print(result$Q_point)
## End(Not run)




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("strategize_onestep", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("strategize_preset")
### * strategize_preset

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: strategize_preset
### Title: Get Recommended Parameter Settings
### Aliases: strategize_preset

### ** Examples

## No test: 
# Get standard settings
params <- strategize_preset("standard")
print(params)

# Use with strategize (hypothetically)
# result <- do.call(strategize, c(list(Y = Y, W = W, p_list = p_list), params))
## End(No test)




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("strategize_preset", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
### * <FOOTER>
###
cleanEx()
options(digits = 7L)
base::cat("Time elapsed: ", proc.time() - base::get("ptime", pos = 'CheckExEnv'),"\n")
grDevices::dev.off()
###
### Local variables: ***
### mode: outline-minor ***
### outline-regexp: "\\(> \\)?### [*]+" ***
### End: ***
quit('no')
