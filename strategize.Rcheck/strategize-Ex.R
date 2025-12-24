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
# A minimal example using hypothetical data
set.seed(123)
# Suppose Y is a binary forced choice outcome, W has several attributes (factors)
Y <- rbinom(200, size = 1, prob = 0.5)
W <- data.frame(
  Gender = sample(c("Male","Female"), 200, TRUE),
  Age    = sample(c("35","50","65"),  200, TRUE),
  Party  = sample(c("Dem","Rep"),     200, TRUE)
)

# Cross-validate over a range of lambda
lam_seq <- c(0, 0.001, 0.01, 0.1)
cv_fit <- cv_strategize(
  Y = Y, 
  W = W, 
  lambda_seq = lam_seq, 
  folds = 2
)

# Extract optimal lambda and final fit
print(cv_fit$lambda)
print(cv_fit$CVInfo)
print(names(cv_fit$pi_star_point))
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
##D # After fitting an adversarial strategize model:
##D adv_res <- strategize(
##D   Y = Yobs,
##D   W = W,
##D   adversarial = TRUE,
##D   ...
##D )
##D 
##D # Suppose dimension 1 is "Gender." Then to see each player's best response:
##D plot_best_response_curves(
##D   res    = adv_res,
##D   d_     = 1,
##D   nPoints_br= 41,         # can reduce or enlarge
##D   title  = "Gender Best-Response Curves",
##D   col_ast= "blue",
##D   col_dag= "red"
##D )
##D 
##D # The intersection (if shown) approximates an equilibrium for dimension 1.
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

## Not run: 
##D   # Suppose we have a forced-choice conjoint dataset with
##D   # factor matrix W, outcome Y, and baseline probabilities p_list
##D 
##D   # Basic usage: single agent optimizing expected outcome
##D   opt_result <- strategize(
##D       Y = Y,
##D       W = W,
##D       lambda = 0.1,
##D       p_list = p_list,
##D       adversarial = FALSE,         # No adversarial component
##D       penalty_type = "KL",         # Kullback-Leibler penalty
##D       nSGD = 200              # # of gradient descent iterations
##D   )
##D 
##D   # Inspect the learned distribution and performance
##D   print(opt_result$pi_star_point)
##D   print(opt_result$Q_point_mEst)
##D   print(opt_result$CVInfo)            # If cross-validation was used
##D 
##D   # Adversarial scenario with multi-stage structure
##D   # E.g., define 'competing_group_variable_respondent' for two parties' supporters
##D   adv_result <- strategize(
##D       Y = Y,
##D       W = W,
##D       lambda = 0.2,
##D       p_list = p_list,
##D       adversarial = TRUE,         # Solve zero-sum game across two sets of respondents
##D       competing_group_variable_respondent = partyID,
##D       nSGD = 300
##D   )
##D 
##D   # 'adv_result' now contains distributions for each party's candidate
##D   # that approximate a mixed-strategy Nash equilibrium
##D   print(adv_result$pi_star_point$k1)   # Party A distribution
##D   print(adv_result$pi_star_point$k2)   # Party B distribution
## End(Not run)




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

## Not run: 
##D # Example usage (assuming appropriate data structures)
##D # Note: p_list elements must have named levels
##D hypotheticalProbs <- list(k1 = list(
##D   Gender = c(Male = 0.4, Female = 0.6),
##D   Party = c(Dem = 0.3, Rep = 0.7)
##D ))
##D SEs <- list(k1 = list(
##D   Gender = c(Male = 0.05, Female = 0.05),
##D   Party = c(Dem = 0.06, Rep = 0.06)
##D ))
##D assignmentProbs <- list(
##D   Gender = c(Male = 0.5, Female = 0.5),
##D   Party = c(Dem = 0.5, Rep = 0.5)
##D )
##D strategize.plot(
##D   pi_star_list = hypotheticalProbs,
##D   pi_star_se_list = SEs,
##D   p_list = assignmentProbs,
##D   col_vec = c("blue"),
##D   main_title = "Example Plot"
##D )
## End(Not run)




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
##D # Suppose we have a forced-choice conjoint dataset (W, Y) and baseline probabilities p_list.
##D # We want to estimate an optimal distribution that maximizes average Y.
##D 
##D set.seed(123)
##D # X could be respondent covariates, if any
##D X <- matrix(rnorm(nrow(W)*2), nrow(W), 2)
##D 
##D result_one_step <- strategize_onestep(
##D   W = W,
##D   Y = Y,
##D   X = X,
##D   p_list = p_list,
##D   nSGD = 400,
##D   use_hajek = TRUE,
##D   penalty_type = "LogMaxProb",
##D   lambda_seq = c(0.01, 0.1),
##D   test_fraction = 0.3
##D )
##D 
##D # Inspect the estimated distribution over factor levels
##D str(result_one_step$pi_star_point)
##D 
##D # Evaluate estimated performance
##D print( result_one_step$Q_point )
## End(Not run)




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("strategize_onestep", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
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
