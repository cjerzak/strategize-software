#' Estimate Optimal (or Adversarial) Stochastic Interventions for Conjoint Experiments
#'
#' @description
#' \code{strategize} implements the core methods described in the accompanying paper
#' for learning an optimal or adversarial probability distribution over conjoint factor levels.
#' It is specifically designed for forced-choice conjoint settings (e.g., candidate-choice experiments)
#' and can accommodate scenarios in which a single agent optimizes its strategy in isolation,
#' or in which two (potentially adversarial) agents simultaneously optimize against each other.
#'
#' This function can be used to find the \emph{optimal stochastic intervention} for maximizing
#' an outcome of interest (e.g., vote choice, rating, or utility), possibly subject to a penalty
#' that keeps the learned distribution close to the original design distribution. It can also
#' incorporate institutional rules (e.g., primaries, multiple stages of choice) by specifying
#' additional arguments. Estimation can be done under standard generalized linear modeling assumptions
#' or more advanced approaches. The function returns estimates of the learned distribution and
#' the associated performance quantity (\eqn{Q(\boldsymbol{\pi}^\ast)}) along with optional inference based on
#' the (asymptotic) delta method.
#'
#' @usage
#' strategize(
#'   Y,
#'   W,
#'   X = NULL,
#'   lambda,
#'   varcov_cluster_variable = NULL,
#'   competing_group_variable_respondent = NULL,
#'   competing_group_variable_respondent_proportions = NULL,
#'   competing_group_variable_candidate = NULL,
#'   competing_group_competition_variable_candidate = NULL,
#'   pair_id = NULL,
#'   respondent_id = NULL,
#'   respondent_task_id = NULL,
#'   profile_order = NULL,
#'   p_list = NULL,
#'   slate_list = NULL,
#'   K = 1,
#'   nSGD = 100,
#'   diff = FALSE,
#'   adversarial = FALSE,
#'   adversarial_model_strategy = "four",
#'   include_stage_interactions = NULL,
#'   partial_pooling = NULL,
#'   partial_pooling_strength = 50,
#'   use_regularization = TRUE,
#'   force_gaussian = FALSE,
#'   a_init_sd = 0.001,
#'   outcome_model_type = "glm",
#'   neural_mcmc_control = NULL,
#'   penalty_type = "KL",
#'   compute_se = FALSE,
#'   se_method = c("full", "implicit"),
#'   conda_env = "strategize_env",
#'   conda_env_required = FALSE,
#'   conf_level = 0.90,
#'   nFolds_glm = 3L,
#'   folds = NULL,
#'   nMonte_adversarial = 5L,
#'   primary_pushforward = "mc",
#'   primary_strength = 1.0,
#'   primary_n_entrants = 1L,
#'   primary_n_field = 1L,
#'   nMonte_Qglm = 100L,
#'   learning_rate_max = 0.001,
#'   temperature = 0.5, 
#'   save_outcome_model = FALSE,
#'   presaved_outcome_model = FALSE,
#'   outcome_model_key = NULL,
#'   use_optax = FALSE,
#'   optim_type = "gd",
#'   optimism = "extragrad",
#'   optimism_coef = 1,
#'   compute_hessian = TRUE,
#'   hessian_max_dim = 50L
#' )
#'
#' @param Y A numeric or binary vector of observed outcomes, typically in \code{\{0,1\}} for forced-choice
#'   conjoint tasks, indicating whether the profile was selected. For instance, \code{Y = 1} if
#'   candidate A was chosen over candidate B, and \code{Y = 0} otherwise. The length must match
#'   the number of rows in \code{W}.
#'
#' @param W A matrix or data frame representing the assigned levels of each factor in a conjoint
#'   design (one column per factor). Each row corresponds to a single profile. For forced-choice
#'   tasks, a given respondent may have contributed multiple rows if you reshape pairwise choices
#'   into long format. If the experiment used multiple factors \eqn{D}, with each factor having
#'   \eqn{L_d} levels, \code{W} should capture all factor assignments accordingly.
#'
#' @param X An optional matrix or data frame of additional covariates, often respondent-level
#'   features (e.g., respondent demographics). If \code{K > 1}, \code{X} may be used internally to
#'   fit multi-cluster or multi-component outcome models, or to allow cluster-specific effect
#'   estimation for more granular insights. Defaults to \code{NULL}.
#'
#' @param lambda A numeric scalar or vector giving the regularization penalty (e.g., in Kullback-Leibler
#'   or L2 sense) used to shrink the learned probability distribution(s) of factor levels toward a
#'   baseline distribution \code{p_list}. Typically set via either domain knowledge or cross-validation.
#'
#' @param varcov_cluster_variable An optional vector of cluster identifiers (e.g., respondent IDs)
#'   used to form a robust variance-covariance estimate of the outcome model. If \code{NULL},
#'   the usual IID assumption is made. Defaults to \code{NULL}.
#'
#' @param competing_group_variable_respondent Optional variable marking competition group
#'   membership of respondents. Particularly relevant in adversarial settings
#'   (\code{adversarial = TRUE}) or multi-stage electoral settings, e.g., capturing the party of each
#'   respondent. Defaults to \code{NULL}.
#'
#' @param competing_group_variable_candidate Optional variable marking competition group
#'   membership of candidate profiles. Defaults to \code{NULL}.
#'
#' @param competing_group_competition_variable_candidate Optional variable indicating whether
#'   a candidate profile belongs to the competing group in adversarial settings. Defaults to \code{NULL}.
#'
#' @param competing_group_variable_respondent_proportions Optional numeric vector specifying
#'   the population proportions of each competing group. If \code{NULL}, proportions are estimated
#'   from the data. Useful when the sample proportions differ from the target population proportions.
#'   Defaults to \code{NULL}.
#'
#' @param pair_id A factor or numeric vector identifying the forced-choice pair. If each row of
#'   \code{W} is a single profile, \code{pair_id} groups the rows belonging to the same choice set.
#'   Defaults to \code{NULL}.
#'
#' @param respondent_id,respondent_task_id Another set of optional identifiers. \code{respondent_id}
#'   marks each respondent across tasks, while \code{respondent_task_id} can define unique IDs for
#'   repeated measurements from the same respondent across multiple tasks. Useful for advanced
#'   clustering or robust SEs. Defaults to \code{NULL}.
#'
#' @param profile_order If each forced-choice is shown with different ordering (e.g., `Candidate A`
#'   vs. `Candidate B`), \code{profile_order} can label each row accordingly. Helpful for ensuring
#'   consistent labeling of reference vs. opposing profiles. Defaults to \code{NULL}.
#'
#' @param p_list An optional list describing the baseline probability distribution over factor levels
#'   in \code{W}. Typically derived from the initial design distribution or uniform assignment
#'   distribution. If \code{NULL}, the function may assume uniform or attempt to estimate the
#'   distribution from \code{W}.
#'
#' @param slate_list An optional list (or lists) providing custom slates of candidate features
#'   (and their associated probabilities). Used in more advanced or adversarial setups where
#'   certain combinations must be included or excluded. If \code{NULL}, no special constraints
#'   beyond the usual factor-level distributions are applied.
#'
#' @param K Integer specifying the number of latent clusters for multi-component outcome models. If
#'   \code{K = 1}, no latent clustering is done. Defaults to \code{1}.
#'
#' @param nSGD Integer specifying the number of stochastic gradient descent (or gradient-based)
#'   iterations to use when learning the optimal distributions. Defaults to \code{100}.
#'
#' @param diff Logical indicating whether the outcome \code{Y} represents a first-difference or
#'   difference-based metric. In forced-choice contexts, typically \code{diff = FALSE}. Defaults
#'   to \code{FALSE}.
#'
#' @param adversarial Logical controlling whether to enable the max-min adversarial scenario. When
#'   \code{TRUE}, the function searches for a pair of distributions (one for each competing party
#'   or group) such that each party's distribution is optimal given the other party's distribution.
#'   Defaults to \code{FALSE}.
#'
#' @param adversarial_model_strategy Character string indicating whether to estimate
#'   \code{"four"} outcome models (primary + general for each group), \code{"two"} outcome models
#'   (one per group reused for both primary and general), or \code{"neural"} (Bayesian Transformer
#'   models with party tokens; defaults to a single pooled model across groups and stages. Set
#'   \code{neural_mcmc_control$n_bayesian_models = 2} to fit separate AST/DAG models). Defaults to
#'   \code{"four"}.
#'
#' @param include_stage_interactions Logical indicating whether to include stage (primary vs
#'   general) indicator and stage-by-factor interactions in the outcome model. When \code{NULL}
#'   (default), automatically set to \code{TRUE} when \code{adversarial_model_strategy = "two"}
#'   and \code{FALSE} otherwise. Including stage interactions allows a single pooled model to
#'   learn different response patterns for primary vs general election scenarios, which helps
#'   prevent pattern-matching equilibria where both parties converge to identical strategies.
#'
#' @param partial_pooling Logical indicating whether to partially pool (shrink) group-specific
#'   outcome model coefficients toward a shared average when
#'   \code{adversarial_model_strategy = "two"}. When \code{NULL} (default), automatically set to
#'   \code{TRUE} for the two-strategy adversarial case and \code{FALSE} otherwise.
#'
#' @param partial_pooling_strength Numeric scalar controlling the amount of shrinkage used for
#'   partial pooling in the two-strategy adversarial case. Interpreted as a pseudo-sample size:
#'   larger values increase pooling, smaller values preserve group differentiation. Defaults to
#'   \code{50}.
#'
#' @param use_regularization Logical indicating whether to regularize the outcome model (in addition
#'   to any penalty \code{lambda} on the distribution shift). This can help avoid overfitting in
#'   high-dimensional designs. Defaults to \code{TRUE}.
#'
#' @param force_gaussian Logical indicating whether to force a Gaussian-based outcome modeling
#'   approach, even if \code{Y} is binary or forced-choice. If \code{FALSE}, the function attempts
#'   to choose a more appropriate link (e.g., \code{"binomial"}). Defaults to \code{FALSE}.
#'
#' @param a_init_sd Numeric scalar specifying the standard deviation for random initialization
#'   of unconstrained parameters used in the gradient-based search over factor-level probabilities.
#'   Defaults to \code{0.001}.
#'
#' @param outcome_model_type Character string specifying the outcome model to use. Currently
#'   supports \code{"glm"} for generalized linear models or \code{"neural"} for a neural-network
#'   approximation. Defaults to \code{"glm"}.
#'
#' @param neural_mcmc_control Optional list overriding default MCMC settings used when
#'   \code{outcome_model_type = "neural"}. Named entries override the defaults in
#'   \code{CS_2Step_ModelOutcome_neural.R}. Set
#'   \code{neural_mcmc_control$uncertainty_scope = "output"} to compute delta-method
#'   uncertainty using only the output-layer parameters (default is \code{"all"}). In adversarial
#'   neural mode, set \code{neural_mcmc_control$n_bayesian_models = 2} to fit separate AST/DAG
#'   models (default is 1 for a single differential model). Use
#'   \code{neural_mcmc_control$ModelDims} and \code{neural_mcmc_control$ModelDepth} to override
#'   the Transformer hidden width and depth. Set
#'   \code{neural_mcmc_control$cross_candidate_encoder = "term"} (or \code{TRUE}) to include
#'   the opponent-dependent cross-candidate term in pairwise mode, or set
#'   \code{neural_mcmc_control$cross_candidate_encoder = "full"} to enable a full cross-encoder
#'   that jointly encodes both candidates. Use \code{"none"} (or \code{FALSE}) to disable.
#'   For variational inference (subsample_method = "batch_vi"), set
#'   \code{neural_mcmc_control$optimizer} to \code{"adam"} (numpyro.optim),
#'   \code{"adamw"} (AdamW), or \code{"adabelief"} (optax). Learning-rate decay is controlled by
#'   \code{neural_mcmc_control$svi_steps} (integer steps, or \code{"optimal"} for
#'   a scaling-law heuristic based on model/data size; for minibatched VI this
#'   also scales with \code{batch_size}) and
#'   \code{neural_mcmc_control$svi_lr_schedule} (default \code{"warmup_cosine"}), with optional
#'   \code{svi_lr_warmup_frac} and \code{svi_lr_end_factor}.
#'
#' @param penalty_type A character string specifying the type of penalty (e.g., \code{"KL"}, \code{"L2"},
#'   or \code{"LogMaxProb"}) used in the objective function for shifting the factor-level probabilities
#'   away from the baseline \code{p_list}. Defaults to \code{"KL"}.
#'
#' @param compute_se Logical indicating whether standard errors should be computed for the final
#'   estimates (via the delta method or related expansions). Defaults to \code{FALSE}.
#'
#' @param se_method Character string specifying the SE computation method when \code{compute_se = TRUE}.
#'   \code{"full"} differentiates through the full optimization trace (default). \code{"implicit"}
#'   uses implicit differentiation at the solution (adversarial equilibrium or non-adversarial optimum).
#'
#' @param conda_env A character string naming a Python conda environment that includes \pkg{jax},
#'   \pkg{optax}, and other dependencies. If not \code{NULL}, the function attempts to activate
#'   that environment. Defaults to \code{"strategize_env"}.
#'
#' @param conda_env_required Logical; if \code{TRUE}, raises an error if the environment given by
#'   \code{conda_env} cannot be activated. Otherwise, the function attempts to proceed with any
#'   available installation. Defaults to \code{FALSE}.
#'
#' @param conf_level Numeric in \eqn{(0,1)}, specifying the confidence level for intervals or
#'   credible bounds. Defaults to \code{0.90}.
#'
#' @param nFolds_glm Integer specifying the number of folds (default \code{3L}) for internal
#'   cross-validation used in certain outcome model or regularization steps. Defaults to \code{3L}.
#'
#' @param folds An optional user-supplied partitioning or CV scheme, overriding \code{nFolds_glm}.
#'   Defaults to \code{NULL}.
#'
#' @param nMonte_adversarial Integer specifying the number of Monte Carlo samples used in adversarial
#'   or max-min steps, e.g., sampling from the opposing candidate's distribution to approximate
#'   expected payoffs. Defaults to \code{5L}.
#'
#' @param primary_pushforward Character string controlling the primary-stage push-forward estimator.
#'   Use \code{"mc"} (default) for Monte Carlo sampling with per-draw primary winners, or
#'   \code{"linearized"} for the faster averaged-weight approximation, or
#'   \code{"multi"} for multi-candidate primaries.
#' @param primary_strength Numeric scalar controlling primary decisiveness. Values > 1 make
#'   primary outcomes more deterministic; values in (0, 1) make primaries more noisy. Defaults
#'   to 1.0 (neutral scaling).
#' @param primary_n_entrants Integer number of entrant candidates sampled per party in
#'   multi-candidate primaries (\code{primary_pushforward = "multi"}). Defaults to 1.
#' @param primary_n_field Integer number of field candidates sampled per party in
#'   multi-candidate primaries (\code{primary_pushforward = "multi"}). Defaults to 1.
#' 
#' @param nMonte_Qglm Integer specifying the number of Monte Carlo samples for evaluating or
#'   approximating the quantity of interest under certain outcomes or distributions. Defaults to
#'   \code{100L}.
#'
#' @param learning_rate_max Base learning rate for gradient-based optimizers. Defaults to
#'   \code{0.001}.
#'
#' @param temperature Numeric temperature parameter used in Gumbel-Softmax sampling to smooth
#'   the exploration of the probability simplex. Smaller values yield distributions closer to the
#'   argmax. Defaults to \code{0.5}.
#'
#' @param save_outcome_model Logical indicating whether to save the fitted outcome model to
#'   disk for reuse. Useful for large models or repeated runs. Defaults to \code{FALSE}.
#'
#' @param presaved_outcome_model Logical indicating whether to use a previously saved outcome
#'   model instead of re-fitting. Defaults to \code{FALSE}.
#'
#' @param outcome_model_key Optional character string to append to saved outcome model
#'   filenames. Useful for distinguishing between different model configurations or
#'   experimental runs. When provided, files are saved as
#'   \code{main_{group}_{round}_{key}.csv}. Defaults to \code{NULL}.
#'
#' @param use_optax Logical indicating whether to use the \href{https://github.com/google-deepmind/optax}{\code{optax}}
#'   library for gradient-based optimization in JAX (\code{TRUE}) or a built-in method (\code{FALSE}).
#'   Defaults to \code{FALSE}.
#'
#' @param optim_type A character string for choosing which optimizer or approach is used internally
#'   (e.g., \code{"gd"} for gradient descent). Defaults to \code{"gd"}.
#' @param optimism Character string controlling optimistic / extra-gradient updates for the gradient
#'   optimizer. Options: \code{"extragrad"} (default; classic Korpelevich extra-gradient),
#'   \code{"smp"} (stochastic mirror-prox: extra-gradient with weighted averaging of look-ahead points),
#'   \code{"ogda"} (optimistic gradient), \code{"rain"} (RAIN: Recursive Anchored Iteration with
#'   anchored extra-gradient and increasing quadratic anchor penalties), or \code{"none"}
#'   (standard updates). Only supported when
#'   \code{use_optax = FALSE}.
#' @param optimism_coef Numeric scalar controlling the magnitude of optimism adjustments. For
#'   \code{"ogda"}, this scales the optimistic correction term. For \code{"rain"}, this is the
#'   initial anchor weight \eqn{\lambda_0} used by RAIN; anchor weights grow multiplicatively by
#'   \eqn{(1+\gamma)} each outer stage.
#' @param rain_gamma Non-negative numeric scalar for the RAIN anchor-growth parameter \eqn{\gamma}.
#'   Larger values grow anchor weights faster. Only used when \code{optimism = "rain"}.
#' @param rain_eta Optional numeric scalar step size \eqn{\eta} for RAIN. If \code{NULL}, defaults
#'   to \code{learning_rate_max}. Only used when \code{optimism = "rain"}.
#' @param compute_hessian Logical. Whether to compute Hessian functions for equilibrium
#'   geometry analysis in adversarial mode. When \code{TRUE} (default), Hessian functions
#'   are JIT-compiled to enable \code{\link{check_hessian_geometry}} analysis. Set to
#'   \code{FALSE} to skip Hessian computation for faster execution.
#' @param hessian_max_dim Integer. Maximum number of parameters per player before
#'   automatically skipping Hessian computation. Defaults to \code{50L}. For problems
#'   with more parameters, Hessian computation is skipped to avoid memory/time overhead.
#'   The result will have \code{hessian_skipped_reason = "high_dimension"} in this case.
#'
#' @return A named \code{list} containing:
#' \describe{
#' \item{\code{pi_star_point}}{An estimate of the (possibly multi-cluster or adversarial)
#' optimal distribution(s) over the factor levels.
#'
#' Structure depends on parameters:
#' - If \code{adversarial = TRUE} and \code{K = 1}, returns a pair of distributions (e.g., maximin solutions).
#' - If \code{K > 1}, returns a list where each element corresponds to a cluster-optimal distribution.
#' - Otherwise, returns a single distribution.}
#'
#' \item{\code{pi_star_se}}{Standard errors for entries in \code{pi_star_point}. Mirrors the structure of \code{pi_star_point} (e.g., a pair of SEs if \code{adversarial = TRUE} and \code{K = 1}). Only present if \code{compute_se = TRUE}.}
#'
#' \item{\code{Q_point_mEst}}{Point estimate(s) of the optimized outcome (e.g., utility/vote share). Matches the structure of \code{pi_star_point}.}
#'
#' \item{\code{Q_se_mEst}}{Standard errors for \code{Q_point_mEst}. Only present if \code{compute_se = TRUE}.}
#'
#' \item{\code{pi_star_lb}, \code{pi_star_ub}}{Confidence bounds for \code{pi_star_point} (if \code{compute_se = TRUE} and a confidence level is provided).}
#'
#' \item{\code{outcome_model_view}}{Interpretable summaries of the fitted outcome models (by player and stage).
#' Includes main-effect tables and top interactions for AST/DAG primary/general submodels when available.}
#'
#' \item{\code{CVInfo}}{Cross-validation performance data (if applicable). Typically a \code{data.frame} or list.}
#'
#' \item{\code{estimationType}}{String indicating the approach used (e.g., \code{"TwoStep"} or \code{"OneStep"}).}
#'
#' \item{\code{...}}{Additional internal details (e.g., fitted models, optimization logs).}
#' }
#'
#' @details
#' \strong{Modeling the outcome:} Internally, \code{strategize} may fit a generalized linear model
#' or a more flexible approach (such as multi-cluster factorization) to learn the mapping from
#' factor-level assignments \code{W} (and optional covariates \code{X}) onto outcomes \code{Y}.
#' Once these outcome coefficients are estimated, the function uses gradient-based or closed-form
#' solutions to find the \emph{optimal stochastic intervention(s)}, i.e., new factor-level probability
#' distributions that maximize an expected outcome (or solve the max-min adversarial problem).
#'
#' \strong{Adversarial or strategic design:} When \code{adversarial = TRUE}, the function attempts to
#' solve a zero-sum game in which one agent (say, \dQuote{A}) chooses its distribution to maximize
#' vote share, while the other (\dQuote{B}) simultaneously chooses its distribution to minimize
#' \dQuote{A}'s vote share. In many settings, \code{competing_group_variable_respondent} and related
#' arguments help define which respondents belong to the \dQuote{A} or \dQuote{B} sub-electorate
#' (e.g., a primary). The final solution is a mixed-strategy Nash equilibrium, if it exists, for
#' the forced-choice environment. This can be used to compare or interpret real-world candidate
#' positioning in multi-stage elections.
#'
#' \strong{Regularization:} The argument \code{lambda} penalizes how far the learned distribution
#' strays from the baseline distribution \code{p_list}. This helps avoid overfitting in high-dimensional
#' designs. Different penalty types can be selected via \code{penalty_type}.
#'
#' \strong{Implementation details:} Under the hood, this function may rely on \pkg{jax} for automatic
#' differentiation. By default, it uses an internal gradient-based approach. If \code{use_optax = TRUE},
#' the \code{optax} library is used for optimization. The function can automatically detect or
#' load a \pkg{conda} environment if specified, though advanced users can pass \code{conda_env_required = TRUE}
#' to enforce that environment activation is mandatory.
#'
#' @seealso
#' \code{\link{cv_strategize}} for cross-validation across candidate values of \code{lambda}.
#' See also \code{\link{strategize_onestep}} for a function that implements a \dQuote{one-step}
#' approach to M-estimation of the same target quantity. 
#'
#' @examples
#' \donttest{
#' # ============================================
#' # Example 1: Basic single-agent optimization
#' # ============================================
#' # Generate synthetic conjoint data
#' set.seed(42)
#' n <- 400  # Number of profiles (200 pairs)
#'
#' # Factor matrix: candidate attributes
#' W <- data.frame(
#'   Gender = sample(c("Male", "Female"), n, replace = TRUE),
#'   Age = sample(c("Young", "Middle", "Old"), n, replace = TRUE),
#'   Party = sample(c("Dem", "Rep"), n, replace = TRUE)
#' )
#'
#' # Simulate outcome: Female + Young candidates preferred
#' latent <- 0.3 * (W$Gender == "Female") +
#'           0.2 * (W$Age == "Young") -
#'           0.1 * (W$Age == "Old")
#' prob <- plogis(latent)
#'
#' # Paired forced-choice: within each pair, one wins
#' pair_id <- rep(1:(n/2), each = 2)
#' Y <- numeric(n)
#' for (p in unique(pair_id)) {
#'   idx <- which(pair_id == p)
#'   winner <- sample(idx, 1, prob = prob[idx])
#'   Y[idx] <- as.integer(seq_along(idx) == which(idx == winner))
#' }
#' profile_order <- rep(1:2, n/2)
#'
#' # Baseline probabilities (uniform assignment)
#' p_list <- list(
#'   Gender = c(Male = 0.5, Female = 0.5),
#'   Age = c(Young = 1/3, Middle = 1/3, Old = 1/3),
#'   Party = c(Dem = 0.5, Rep = 0.5)
#' )
#'
#' # Run strategize to find optimal distribution
#' # (requires conda environment with JAX - see build_backend())
#' result <- strategize(
#'   Y = Y,
#'   W = W,
#'   lambda = 0.1,
#'   pair_id = pair_id,
#'   respondent_id = pair_id,
#'   respondent_task_id = pair_id,
#'   profile_order = profile_order,
#'   p_list = p_list,
#'   diff = TRUE,
#'   nSGD = 50,
#'   compute_se = FALSE
#' )
#'
#' # View optimal distribution
#' print(result$pi_star_point)
#'
#' # View expected outcome under optimal strategy
#' print(result$Q_point)
#' }
#'
#' @md
#' @export

strategize       <-          function(
                                            Y,
                                            W,
                                            X = NULL,
                                            lambda,
                                            varcov_cluster_variable = NULL,
                                            competing_group_variable_respondent = NULL,
                                            competing_group_variable_respondent_proportions = NULL,
                                            competing_group_variable_candidate = NULL,
                                            competing_group_competition_variable_candidate = NULL,
                                            pair_id = NULL,
                                            respondent_id = NULL,
                                            respondent_task_id = NULL,
                                            profile_order = NULL,
                                            p_list = NULL,
                                            slate_list = NULL,
                                            K = 1,
                                            nSGD = 100,
                                            diff = FALSE,
                                            adversarial = FALSE,
                                            adversarial_model_strategy = "four",
                                            include_stage_interactions = NULL,
                                            partial_pooling = NULL,
                                            partial_pooling_strength = 50,
                                            use_regularization = TRUE,
                                            force_gaussian = FALSE,
                                            a_init_sd = 0.001,
                                            outcome_model_type = "glm",
                                            neural_mcmc_control = NULL,
                                            penalty_type = "KL",
                                            compute_se = FALSE,
                                            se_method = c("full", "implicit"),
                                            conda_env = "strategize_env",
                                            conda_env_required = FALSE,
                                            conf_level = 0.90,
                                            nFolds_glm = 3L,
                                            folds = NULL, 
                                            nMonte_adversarial = 5L,
                                            primary_pushforward = "mc",
                                            primary_strength = 1.0,
                                            primary_n_entrants = 1L,
                                            primary_n_field = 1L,
                                            nMonte_Qglm = 100L,
                                            learning_rate_max = 0.001, 
                                            temperature = 0.5, 
                                            save_outcome_model = FALSE,
                                            presaved_outcome_model = FALSE,
                                            outcome_model_key = NULL,
                                            use_optax = FALSE,
  optim_type = "gd",
                                            optimism = "extragrad",
                                            optimism_coef = 1,
                                            rain_gamma = 0.05,
                                            rain_eta = NULL,
                                            compute_hessian = TRUE,
                                            hessian_max_dim = 50L){
  # [1.] ast then dag 
  #   ast is 1, based on sort(unique(competing_group_variable_candidate))[1]
  #   dag is 2, based on sort(unique(competing_group_variable_candidate))[2]
  # [2.] when simplex constrained with holdout, LAST entry is held out 
  
  message("-------------\nstrategize() call has begun...")

  # Input validation with clear error messages
  validate_strategize_inputs(
    Y = Y, W = W, X = X, lambda = lambda,
    p_list = p_list, K = K, pair_id = pair_id,
    profile_order = profile_order,
    adversarial = adversarial,
    adversarial_model_strategy = adversarial_model_strategy,
    partial_pooling = partial_pooling,
    partial_pooling_strength = partial_pooling_strength,
    competing_group_variable_respondent = competing_group_variable_respondent,
    competing_group_variable_candidate = competing_group_variable_candidate,
    competing_group_competition_variable_candidate = competing_group_competition_variable_candidate,
    outcome_model_type = outcome_model_type,
    neural_mcmc_control = neural_mcmc_control,
    penalty_type = penalty_type, diff = diff,
    primary_pushforward = primary_pushforward,
    primary_strength = primary_strength,
    primary_n_entrants = primary_n_entrants,
    primary_n_field = primary_n_field,
    rain_gamma = rain_gamma,
    rain_eta = rain_eta
  )
  if (isTRUE(adversarial) && is.null(competing_group_competition_variable_candidate)) {
    respondent_groups <- as.character(competing_group_variable_respondent)
    candidate_groups <- as.character(competing_group_variable_candidate)
    competing_group_competition_variable_candidate <- ifelse(
      candidate_groups == respondent_groups,
      "Same",
      "Different"
    )
  }
  primary_pushforward <- tolower(primary_pushforward)
  primary_strength <- as.numeric(primary_strength)
  primary_n_entrants <- ai(primary_n_entrants)
  primary_n_field <- ai(primary_n_field)
  adversarial_model_strategy <- match.arg(tolower(adversarial_model_strategy), c("two", "four", "neural"))
  optimism <- match.arg(optimism, c("none", "ogda", "extragrad", "smp", "rain"))
  se_method <- match.arg(se_method, c("full", "implicit"))
  if (use_optax && optimism != "none") {
    stop("Optimistic / extra-gradient updates are only available when use_optax = FALSE.")
  }
  if (outcome_model_type == "neural" && optim_type != "gd") {
    warning("Neural outcome models require gradient-based optimization; setting optim_type = \"gd\".")
    optim_type <- "gd"
  }

  n_bayesian_models <- 1L
  if (!is.null(neural_mcmc_control) &&
      !is.null(neural_mcmc_control$n_bayesian_models)) {
    n_bayesian_models <- as.integer(neural_mcmc_control$n_bayesian_models)
  }
  use_single_neural_model <- isTRUE(adversarial) &&
    outcome_model_type == "neural" &&
    adversarial_model_strategy == "neural" &&
    n_bayesian_models == 1L

  # Set default for include_stage_interactions: TRUE when adversarial_model_strategy == "two"
  if (is.null(include_stage_interactions)) {
    include_stage_interactions <- isTRUE(adversarial) && adversarial_model_strategy == "two"
  }
  if (is.null(partial_pooling)) {
    partial_pooling <- isTRUE(adversarial) && adversarial_model_strategy == "two"
  }
  partial_pooling_strength <- as.numeric(partial_pooling_strength)
  if (!is.finite(partial_pooling_strength) || partial_pooling_strength < 0) {
    stop("'partial_pooling_strength' must be a finite, non-negative numeric value.")
  }
  use_stage_models <- isTRUE(adversarial) &&
    (adversarial_model_strategy == "four" ||
       (adversarial_model_strategy == "two" && isTRUE(include_stage_interactions)))

  # Initialize Hessian-related variables with defaults (may be overwritten in adversarial mode)
  d2Q_da2_ast <- NULL
  d2Q_da2_dag <- NULL
  hessian_available <- FALSE
  hessian_skipped_reason <- if (!adversarial) "not_adversarial" else NULL
  n_params_per_player <- NA_integer_

  # define evaluation environment
  evaluation_environment <- environment()
  if (identical(outcome_model_type, "neural")) {
    strenv$neural_model_env <- evaluation_environment
  }
  
  # add colnames if not available 
  if(!is.null(X)){ if(is.null(colnames(X))){ colnames(X) <- paste0("V",1:ncol(X)) } }
  if(ncol(W) < 2){ use_regularization <- FALSE }

  # load in packages
  if(!"jnp" %in% ls(envir = strenv)) {
    message("Initializing computational environment...")  
    initialize_jax(conda_env = conda_env, conda_env_required = conda_env_required) 
  }
  
  # define compile fxn
  compile_fxn <- function(x, static_argnums=NULL){return(strenv$jax$jit(x, static_argnums=static_argnums))}
  #compile_fxn <- function(x, static_argnums=NULL){ return(x) } ; warning("!!!!\nTURNING COMPILE OFF FOR SANITY ESTABLISHMENT!\n!!!!");

  # setup pretty pi functions
  {
    RenamePiList <- function(pi_){
      for(k_ in 1:length(pi_)){
        pi_[[k_]] <- sapply(1:length(pi_[[k_]]),function(zer){
          names(pi_[[k_]][[zer]]) <- names_list[[zer]][[1]]; list(pi_[[k_]][[zer]]) })
        names( pi_[[k_]]) <- names( names_list )
      }
    return( pi_  ) }
    RejiggerPi <- function(pi_,isSE = F){
      update_these <- f2n(unique(names(regularization_adjust_hash)))
      for(k_ in 1:length(pi_)){
        updates_ <- pi_[[k_]]
        pi_[[k_]] <- p_list_PreRegularization
        pi_[[k_]][update_these] <- updates_
        if(isSE){
          pi_[[k_]][-update_these] <- lapply(pi_[[k_]][-update_these],
                                             function(rzer){rzer[]<-NA;return(rzer)})
        }
      }
    return( pi_ ) }
  }

  nMonte_adversarial <- ai( nMonte_adversarial )
  q_ave <- q_dag_ave <- pi_star_ave <- NULL
  my_model_ast_jnp <- my_model_ast0_jnp <- my_model_dag_jnp <- my_model_dag0_jnp <- NULL
  main_comp_mat <- shadow_comp_mat <- NULL
  dQ_da_ast <- dQ_da_dag <- NULL
  a_i_ast_optimized <- a_i_dag_optimized <- NULL
  pi_star_red_ast <- pi_star_red_dag <- NULL
  SLATE_VEC_dag_jnp <- SLATE_VEC_ast_jnp <- NULL
  w_orig <- W
  MaxMinType <- "TwoRoundSingle"

  MNtemp <- strenv$jnp$array( ifelse(!is.null(temperature), yes = temperature, no = 0.5)  ) 
  strenv$primary_strength <- strenv$jnp$array(primary_strength, strenv$dtj)
  strenv$adversarial_model_strategy <- adversarial_model_strategy
  glm_family = "gaussian"; glm_outcome_transform <- function(x){x} # identity function
  if(!force_gaussian){ 
    if(mean(unique(Y) %in% c(0,1)) == 1){ 
      glm_family = "binomial"; glm_outcome_transform <- strenv$jax$nn$sigmoid
    } }

  # ensure naming conventions are correct (i.e. in alignment with p_list if available)
  if(is.null(p_list) | is.null(names(p_list[[1]]))){
    names_list <- apply(w_orig,2,function(zer){ list(sort(names(table(as.factor(zer))),decreasing=F)) })
  }
  if(!is.null(p_list) & !is.null(names(p_list[[1]]))){
    names_list <- lapply(p_list,function(zer){ list(names(zer)) })
  }
  W <- sapply(1:ncol(W),function(zer){ match(W[,zer],names_list[[zer]][[1]]) })
  W <- as.matrix(W)
  if(is.null(colnames(w_orig))){
    if(!is.null(p_list) && !is.null(names(p_list))){
      colnames(W) <- names(p_list)
    } else {
      colnames(W) <- paste0("V", seq_len(ncol(W)))
    }
  } else {
    colnames(W) <- colnames(w_orig)
  }

  # get info about # of levels per factor
  # When p_list is provided, use its structure to ensure consistent dimensions across CV folds
  if(!is.null(p_list) & !is.null(names(p_list[[1]]))){
    factor_levels_full <- factor_levels <- sapply(p_list, length)
  } else {
    factor_levels_full <- factor_levels <- apply(W,2,function(zer){length(unique(zer))})
  }

  # model outcomes
  message("Initializing outcome models...")
  holdout_indicator  <-  1*(K == 1)
  if(K > 1 & !use_regularization){ warning("K > 1; Forcing regularization...");use_regularization <- T }
  use_regularization_ORIG <- use_regularization

  RoundsPool <- nRounds <- GroupsPool <- 1
  if(adversarial){
    GroupsPool <- sort(unique(competing_group_variable_candidate))
    if (adversarial_model_strategy == "four") {
      RoundsPool <- c(0, 1)
    } else {
      RoundsPool <- 1
    }
    nRounds <- length(RoundsPool)
  }
  group_sample_sizes <- NULL
  if (isTRUE(adversarial) && adversarial_model_strategy == "two") {
    group_sample_sizes <- rep(NA_real_, length(GroupsPool))
  }

  model_chunks <- c("vcov_OutcomeModel", "main_info", "interaction_info",
                    "interaction_info_PreRegularization", "REGRESSION_PARAMS_jax",
                    "regularization_adjust_hash", "main_dat", "my_mean",
                    "EST_INTERCEPT_tf", "my_model", "EST_COEFFICIENTS_tf",
                    "neural_model_info", "fit_metrics")

  for(Round_ in RoundsPool){
  for(GroupCounter in 1:length(GroupsPool)){
    if (use_single_neural_model && GroupCounter > 1L) {
      next
    }
    print(c(Round_, GroupCounter))
    use_regularization <- use_regularization_ORIG
    if(adversarial == F){ indi_ <- 1:length( Y )  }
    if(adversarial == T){
      if (adversarial_model_strategy %in% c("two", "neural")) {
        if (use_single_neural_model) {
          # Pooled Bayesian model across groups (single differential fit).
          indi_ <- which(competing_group_variable_candidate %in% GroupsPool)
        } else {
          # Pool primary + general tasks for each respondent group.
          indi_ <- which(competing_group_variable_respondent == GroupsPool[GroupCounter] &
                           competing_group_variable_candidate %in% GroupsPool)
        }
      } else {
        if(Round_ == 0){
          indi_ <- which( competing_group_variable_respondent == GroupsPool[GroupCounter] &
                      ( competing_group_competition_variable_candidate == "Same" &
                          competing_group_variable_candidate == GroupsPool[GroupCounter] ) )
          # cbind(competing_group_variable_respondent,competing_group_variable_candidate)[indi_,]
        }
        if(Round_ == 1){
          indi_ <- which( competing_group_variable_respondent == GroupsPool[GroupCounter] &
                            ( competing_group_competition_variable_candidate == "Different" &
                                competing_group_variable_candidate %in% GroupsPool) )
          # this group comes first
          indi_which_CandidateThisGroup <-
                                  which( competing_group_variable_respondent == GroupsPool[GroupCounter] &
                                   ( competing_group_competition_variable_candidate == "Different" &
                                       competing_group_variable_candidate == GroupsPool[GroupCounter]) )
          #cbind(competing_group_variable_respondent,competing_group_variable_candidate)[indi_,]
        }
      }
      if(is.null(competing_group_variable_respondent_proportions)){ 
        strenv$AstProp <- prop.table(table(competing_group_variable_respondent[
                      competing_group_variable_respondent %in% GroupsPool]))[1]
        strenv$DagProp <- prop.table(table(competing_group_variable_respondent[
                          competing_group_variable_respondent %in% GroupsPool]))[2]
      }
      if(!is.null(competing_group_variable_respondent_proportions)){ 
        strenv$AstProp <- c(competing_group_variable_respondent_proportions[GroupsPool[1]])
        strenv$DagProp <- c(competing_group_variable_respondent_proportions[GroupsPool[2]])
      }
    }

    # subset data
    W_ <- as.matrix(W[indi_,]); Y_ <- Y[indi_];
    varcov_cluster_variable_ <- varcov_cluster_variable[indi_]
    pair_id_ <- pair_id[ indi_ ]
    profile_order_ <- if (!is.null(profile_order)) profile_order[indi_] else NULL
    competing_group_competition_variable_candidate_ <- competing_group_competition_variable_candidate[ indi_ ]
    competing_group_variable_respondent_ <- competing_group_variable_respondent[ indi_ ]
    competing_group_variable_candidate_ <- competing_group_variable_candidate[ indi_ ]
    if (isTRUE(adversarial) && adversarial_model_strategy == "two") {
      if (isTRUE(diff) && !is.null(pair_id_) && length(pair_id_) > 0) {
        group_sample_sizes[GroupCounter] <- length(unique(pair_id_))
      } else {
        group_sample_sizes[GroupCounter] <- length(indi_)
      }
    }

    # run models with inputs: W_; Y_; varcov_cluster_variable_;
    if(outcome_model_type == "glm"){ initialize_ModelOutcome <- paste(deparse(generate_ModelOutcome),collapse="\n") } # linear w interactions
    if(outcome_model_type == "neural"){ initialize_ModelOutcome <- paste(deparse(generate_ModelOutcome_neural),collapse="\n")}  
    initialize_ModelOutcome <- gsub(initialize_ModelOutcome,pattern="function \\(\\)",replacement="")
    eval( parse( text = initialize_ModelOutcome ), envir = evaluation_environment )
    
    # define combined parameter vector & fxn for reextracting intercept & coefficient
    REGRESSION_PARAMS_jax <- strenv$jnp$concatenate(list(EST_INTERCEPT_tf,
                                                         EST_COEFFICIENTS_tf), 0L)

    # Create general coefficients version (with stage adjustments) if available
    # This is used when include_stage_interactions = TRUE with "two" strategy
    if (exists("EST_INTERCEPT_tf_general", inherits = TRUE) &&
        exists("EST_COEFFICIENTS_tf_general", inherits = TRUE)) {
      REGRESSION_PARAMS_jax_general <- strenv$jnp$concatenate(
        list(EST_INTERCEPT_tf_general, EST_COEFFICIENTS_tf_general), 0L
      )
      EST_INTERCEPT_tf_general_store <- EST_INTERCEPT_tf_general
      EST_COEFFICIENTS_tf_general_store <- EST_COEFFICIENTS_tf_general
    } else {
      # No stage interactions - general is same as base
      REGRESSION_PARAMS_jax_general <- REGRESSION_PARAMS_jax
      EST_INTERCEPT_tf_general_store <- EST_INTERCEPT_tf
      EST_COEFFICIENTS_tf_general_store <- EST_COEFFICIENTS_tf
    }

    gather_fxn <- compile_fxn(function(x){
      INTERCEPT_ <- strenv$jnp$expand_dims(strenv$jnp$take(x,0L),0L)
      COEFFS_ <- strenv$jnp$take(x, strenv$jnp$array( 1L:(strenv$jnp$shape(x)[[1]]-1L) ) )
      if(length(COEFFS_$shape)==0){
        COEFFS_ <- strenv$jnp$expand_dims(strenv$jnp$expand_dims(COEFFS_, 0L), 0L) 
      }
      return( list(INTERCEPT_, COEFFS_)) } ) 

    # rename as appropriate
    round_text <- ifelse( Round_==0, yes="0", no="")
    if( (doAst <- (GroupCounter == 1)  ) ){ # do ast
        tmp <- sapply(model_chunks,function(chunk_){ 
          eval(parse(text = sprintf("%s_ast%s_jnp <- %s",chunk_,round_text,chunk_)),envir = evaluation_environment)
        })
    }
    if( !doAst ){ # do dag
        tmp <- sapply(model_chunks,function(chunk_){
          eval(parse(text = sprintf("%s_dag%s_jnp <- %s",chunk_,round_text,chunk_)),envir = evaluation_environment) 
        })
     }
    if (isTRUE(adversarial) && adversarial_model_strategy == "two" && isTRUE(include_stage_interactions)) {
      if (doAst) {
        REGRESSION_PARAMS_jax_general_ast_jnp <- REGRESSION_PARAMS_jax_general
        EST_INTERCEPT_tf_general_ast_jnp <- EST_INTERCEPT_tf_general_store
        EST_COEFFICIENTS_tf_general_ast_jnp <- EST_COEFFICIENTS_tf_general_store
        if (exists("vcov_OutcomeModel_general", inherits = TRUE)) {
          vcov_OutcomeModel_general_ast_jnp <- vcov_OutcomeModel_general
        }
      } else {
        REGRESSION_PARAMS_jax_general_dag_jnp <- REGRESSION_PARAMS_jax_general
        EST_INTERCEPT_tf_general_dag_jnp <- EST_INTERCEPT_tf_general_store
        EST_COEFFICIENTS_tf_general_dag_jnp <- EST_COEFFICIENTS_tf_general_store
        if (exists("vcov_OutcomeModel_general", inherits = TRUE)) {
          vcov_OutcomeModel_general_dag_jnp <- vcov_OutcomeModel_general
        }
      }
    }
    # rm( tmp )
  }
  }

  has_model <- function(suffix) {
    params_name <- paste0("REGRESSION_PARAMS_jax_", suffix, "_jnp")
    if (!exists(params_name, inherits = TRUE)) {
      return(FALSE)
    }
    !is.null(get(params_name, inherits = TRUE))
  }

  copy_model_chunks <- function(source_suffix, target_suffix) {
    for (chunk in model_chunks) {
      src_name <- paste0(chunk, "_", source_suffix, "_jnp")
      dest_name <- paste0(chunk, "_", target_suffix, "_jnp")
      if (!exists(src_name, inherits = TRUE)) {
        next
      }
      if (exists(dest_name, inherits = TRUE)) {
        dest_val <- get(dest_name, inherits = TRUE)
        if (!is.null(dest_val)) {
          next
        }
      }
      assign(dest_name, get(src_name, inherits = TRUE), envir = evaluation_environment)
    }
  }

  if (isTRUE(adversarial) && adversarial_model_strategy %in% c("two", "neural")) {
    copy_model_chunks("ast", "ast0")
    copy_model_chunks("dag", "dag0")
  }

  for (suffix in c("ast0", "dag0", "ast", "dag")) {
    source_suffix <- if (suffix %in% c("dag", "dag0") && has_model("dag")) "dag" else "ast"
    copy_model_chunks(source_suffix, suffix)
  }

  # When using stage interactions with "two" strategy:
  # - ast0/dag0 (primary) use base coefficients (already set)
  # - ast/dag (general) use stage-adjusted coefficients
  if (isTRUE(adversarial) && adversarial_model_strategy == "two" && isTRUE(include_stage_interactions)) {
    message("Applying stage-adjusted coefficients for general election models (ast/dag)...")

    # Update ast model with general coefficients
    if (exists("REGRESSION_PARAMS_jax_general_ast_jnp", inherits = TRUE)) {
      REGRESSION_PARAMS_jax_ast_jnp <- REGRESSION_PARAMS_jax_general_ast_jnp
      EST_INTERCEPT_tf_ast_jnp <- EST_INTERCEPT_tf_general_ast_jnp
      EST_COEFFICIENTS_tf_ast_jnp <- EST_COEFFICIENTS_tf_general_ast_jnp
    }
    if (exists("vcov_OutcomeModel_general_ast_jnp", inherits = TRUE)) {
      vcov_OutcomeModel_ast_jnp <- vcov_OutcomeModel_general_ast_jnp
    }

    # Update dag model with general coefficients
    if (exists("REGRESSION_PARAMS_jax_general_dag_jnp", inherits = TRUE)) {
      REGRESSION_PARAMS_jax_dag_jnp <- REGRESSION_PARAMS_jax_general_dag_jnp
      EST_INTERCEPT_tf_dag_jnp <- EST_INTERCEPT_tf_general_dag_jnp
      EST_COEFFICIENTS_tf_dag_jnp <- EST_COEFFICIENTS_tf_general_dag_jnp
    }
    if (exists("vcov_OutcomeModel_general_dag_jnp", inherits = TRUE)) {
      vcov_OutcomeModel_dag_jnp <- vcov_OutcomeModel_general_dag_jnp
    }
  }
  if (isTRUE(adversarial) && adversarial_model_strategy == "two" && isTRUE(partial_pooling)) {
    pooled_loaded <- FALSE
    get_two_suffix <- function(group) {
      base <- if (!is.null(outcome_model_key)) {
        sprintf("%s_%s_%s", group, 1, outcome_model_key)
      } else {
        sprintf("%s_%s", group, 1)
      }
      sprintf("%s_two", base)
    }
    pooled_file_ast <- sprintf("./StrategizeInternals/coef_pooled_%s.rds",
                               get_two_suffix(GroupsPool[1]))
    pooled_file_dag <- sprintf("./StrategizeInternals/coef_pooled_%s.rds",
                               get_two_suffix(GroupsPool[2]))

    if (isTRUE(presaved_outcome_model) &&
        file.exists(pooled_file_ast) && file.exists(pooled_file_dag)) {
      pooled_ast <- readRDS(pooled_file_ast)
      pooled_dag <- readRDS(pooled_file_dag)
      if (!is.null(pooled_ast$regression_params_primary) &&
          !is.null(pooled_dag$regression_params_primary)) {
        message("Loading pooled outcome coefficients for two-strategy model...")
        REGRESSION_PARAMS_jax_ast0_jnp <- strenv$jnp$array(
          as.numeric(pooled_ast$regression_params_primary),
          dtype = strenv$dtj
        )
        REGRESSION_PARAMS_jax_dag0_jnp <- strenv$jnp$array(
          as.numeric(pooled_dag$regression_params_primary),
          dtype = strenv$dtj
        )
        if (isTRUE(include_stage_interactions) &&
            !is.null(pooled_ast$regression_params_general) &&
            !is.null(pooled_dag$regression_params_general)) {
          REGRESSION_PARAMS_jax_ast_jnp <- strenv$jnp$array(
            as.numeric(pooled_ast$regression_params_general),
            dtype = strenv$dtj
          )
          REGRESSION_PARAMS_jax_dag_jnp <- strenv$jnp$array(
            as.numeric(pooled_dag$regression_params_general),
            dtype = strenv$dtj
          )
        } else {
          REGRESSION_PARAMS_jax_ast_jnp <- REGRESSION_PARAMS_jax_ast0_jnp
          REGRESSION_PARAMS_jax_dag_jnp <- REGRESSION_PARAMS_jax_dag0_jnp
        }

        # Keep intercept/coeff vectors in sync after loading.
        parts_ast0 <- gather_fxn(REGRESSION_PARAMS_jax_ast0_jnp)
        EST_INTERCEPT_tf_ast0_jnp <- parts_ast0[[1]]
        EST_COEFFICIENTS_tf_ast0_jnp <- parts_ast0[[2]]

        parts_dag0 <- gather_fxn(REGRESSION_PARAMS_jax_dag0_jnp)
        EST_INTERCEPT_tf_dag0_jnp <- parts_dag0[[1]]
        EST_COEFFICIENTS_tf_dag0_jnp <- parts_dag0[[2]]

        parts_ast <- gather_fxn(REGRESSION_PARAMS_jax_ast_jnp)
        EST_INTERCEPT_tf_ast_jnp <- parts_ast[[1]]
        EST_COEFFICIENTS_tf_ast_jnp <- parts_ast[[2]]

        parts_dag <- gather_fxn(REGRESSION_PARAMS_jax_dag_jnp)
        EST_INTERCEPT_tf_dag_jnp <- parts_dag[[1]]
        EST_COEFFICIENTS_tf_dag_jnp <- parts_dag[[2]]

        pooled_loaded <- TRUE
      }
    }

    if (!pooled_loaded) {
      if (any(!is.finite(group_sample_sizes))) {
        group_sample_sizes <- vapply(GroupsPool, function(grp) {
          indi_pool <- which(competing_group_variable_respondent == grp &
                               competing_group_variable_candidate %in% GroupsPool)
          if (isTRUE(diff) && !is.null(pair_id) && length(indi_pool) > 0) {
            length(unique(pair_id[indi_pool]))
          } else {
            length(indi_pool)
          }
        }, numeric(1))
      }
      if (length(group_sample_sizes) == 2 && all(is.finite(group_sample_sizes))) {
        n_ast <- group_sample_sizes[1]
        n_dag <- group_sample_sizes[2]
        denom <- n_ast + n_dag

        pool_weight <- function(n) {
          if (!is.finite(n) || n <= 0) {
            return(0)
          }
          w <- partial_pooling_strength / (partial_pooling_strength + n)
          max(0, min(1, w))
        }

        if (denom > 0) {
          pool_params <- function(params_ast, params_dag) {
            params_ast_np <- strenv$np$array(params_ast)
            params_dag_np <- strenv$np$array(params_dag)
            strenv$np$divide(
              strenv$np$add(strenv$np$multiply(n_ast, params_ast_np),
                            strenv$np$multiply(n_dag, params_dag_np)),
              denom
            )
          }
          shrink_params <- function(params, pool_np, weight) {
            params_np <- strenv$np$array(params)
            strenv$np$add(strenv$np$multiply(1 - weight, params_np),
                          strenv$np$multiply(weight, pool_np))
          }

          message("Applying partial pooling to two-strategy outcome models...")
          pool_base_np <- pool_params(REGRESSION_PARAMS_jax_ast0_jnp,
                                      REGRESSION_PARAMS_jax_dag0_jnp)
          w_ast <- pool_weight(n_ast)
          w_dag <- pool_weight(n_dag)

          REGRESSION_PARAMS_jax_ast0_jnp <- strenv$jnp$array(
            shrink_params(REGRESSION_PARAMS_jax_ast0_jnp, pool_base_np, w_ast),
            dtype = strenv$dtj
          )
          REGRESSION_PARAMS_jax_dag0_jnp <- strenv$jnp$array(
            shrink_params(REGRESSION_PARAMS_jax_dag0_jnp, pool_base_np, w_dag),
            dtype = strenv$dtj
          )

          if (isTRUE(include_stage_interactions)) {
            pool_gen_np <- pool_params(REGRESSION_PARAMS_jax_ast_jnp,
                                       REGRESSION_PARAMS_jax_dag_jnp)
            REGRESSION_PARAMS_jax_ast_jnp <- strenv$jnp$array(
              shrink_params(REGRESSION_PARAMS_jax_ast_jnp, pool_gen_np, w_ast),
              dtype = strenv$dtj
            )
            REGRESSION_PARAMS_jax_dag_jnp <- strenv$jnp$array(
              shrink_params(REGRESSION_PARAMS_jax_dag_jnp, pool_gen_np, w_dag),
              dtype = strenv$dtj
            )
          } else {
            REGRESSION_PARAMS_jax_ast_jnp <- REGRESSION_PARAMS_jax_ast0_jnp
            REGRESSION_PARAMS_jax_dag_jnp <- REGRESSION_PARAMS_jax_dag0_jnp
          }

          # Keep intercept/coeff vectors in sync after pooling.
          parts_ast0 <- gather_fxn(REGRESSION_PARAMS_jax_ast0_jnp)
          EST_INTERCEPT_tf_ast0_jnp <- parts_ast0[[1]]
          EST_COEFFICIENTS_tf_ast0_jnp <- parts_ast0[[2]]

          parts_dag0 <- gather_fxn(REGRESSION_PARAMS_jax_dag0_jnp)
          EST_INTERCEPT_tf_dag0_jnp <- parts_dag0[[1]]
          EST_COEFFICIENTS_tf_dag0_jnp <- parts_dag0[[2]]

          parts_ast <- gather_fxn(REGRESSION_PARAMS_jax_ast_jnp)
          EST_INTERCEPT_tf_ast_jnp <- parts_ast[[1]]
          EST_COEFFICIENTS_tf_ast_jnp <- parts_ast[[2]]

          parts_dag <- gather_fxn(REGRESSION_PARAMS_jax_dag_jnp)
          EST_INTERCEPT_tf_dag_jnp <- parts_dag[[1]]
          EST_COEFFICIENTS_tf_dag_jnp <- parts_dag[[2]]

          if (isTRUE(save_outcome_model)) {
            dir.create('./StrategizeInternals',showWarnings=FALSE)
            pooled_ast <- list(
              regression_params_primary = as.numeric(strenv$np$array(REGRESSION_PARAMS_jax_ast0_jnp)),
              regression_params_general = as.numeric(strenv$np$array(REGRESSION_PARAMS_jax_ast_jnp))
            )
            pooled_dag <- list(
              regression_params_primary = as.numeric(strenv$np$array(REGRESSION_PARAMS_jax_dag0_jnp)),
              regression_params_general = as.numeric(strenv$np$array(REGRESSION_PARAMS_jax_dag_jnp))
            )
            saveRDS(pooled_ast, file = pooled_file_ast)
            saveRDS(pooled_dag, file = pooled_file_dag)
          }
        }
      }
    }
  }
  message("Done initializing outcome models & starting optimization sequence...")

  n_main_params <- nrow( main_info )
  if(is.null(p_list) & any(apply(W,2,function(zer){
    max(abs(prop.table(table(zer))-1/length(unique(zer))))})>0.1)){
    warning("Assignment probabilities don't seem uniform!")
  }
  if(is.null(p_list)){
    p_list <- p_list_full <- sapply(factor_levels,function(l_d){list(rep(1/l_d,times=l_d))})
    p_vec <- unlist(p_list_red <- sapply(factor_levels,function(l_d){rep(1/l_d,times=l_d-1)}))
  }
  if(!is.null(p_list)){
    if(any(names(p_list) != colnames(W))){  stop("p_list and W not aligned") }
    p_list_full <- p_list
    p_vec_full_PreRegularization <- p_list_full
    p_list_PreRegularization <- p_list
    p_vec <- unlist(lapply(p_list,function(zer){zer[-length(zer)]}))
    p_vec_full <- unlist(lapply(p_list,function(zer){zer}))
  }

  message("Defining some preliminary objects...")
  strenv$ParameterizationType <- ifelse( holdout_indicator == 0,
                                  yes = "Full", no = "Implicit" ) 
  #strenv$d_locator_full <- strenv$d_locator_use
  d_locator_full <- unlist(sapply(1:length(factor_levels),function(l_d){rep(l_d,times=factor_levels[l_d])}))
  d_locator <- unlist(sapply(1:length(factor_levels),function(l_d){rep(l_d,times=factor_levels[l_d]-1)}))
  strenv$nUniqueFactors <- as.integer(length(unique(as.vector(d_locator))))
  strenv$nUniqueLevelsByFactors <- as.integer( factor_levels )
  strenv$d_locator_use <- strenv$jnp$array(
                                    ifelse(strenv$ParameterizationType == "Implicit", 
                                           yes = list(d_locator), 
                                           no = list(d_locator_full))[[1]] ) 
  
  # p logic 
  p_vec_sum_prime <- unlist(tapply(1:length(p_vec),d_locator,function(er){
    sapply(er,function(re){sum(p_vec[er[!er %in% re]])}) }))
  p_vec_sum_prime_full <- unlist(tapply(1:length(p_vec_full),d_locator_full,function(er){
    sapply(er,function(re){sum(p_vec_full[er[!er %in% re]])}) }))
  main_indices_i0 <- strenv$jnp$array((ai(1:n_main_params-1L)),strenv$jnp$int32)
  inter_indices_i0 <- NULL; if( n_main_params != EST_COEFFICIENTS_tf$size){ 
    inter_indices_i0 <- strenv$jnp$array((ai(((n_main_params+1):EST_COEFFICIENTS_tf$size)-1L)),strenv$jnp$int32)
  }
  if(strenv$ParameterizationType == "Implicit"){ p_vec_use <- p_vec; p_vec_sum_prime_use <- p_vec_sum_prime }
  if(strenv$ParameterizationType == "Full"){ p_vec_use <- p_vec_full; p_vec_sum_prime_use <- p_vec_sum_prime_full }

  if(optim_type != "gd"){
    initialize_ExactSol <- paste(deparse(generate_ExactSol), collapse="\n")
    initialize_ExactSol <- gsub(initialize_ExactSol,pattern="function \\(\\)", replacement="")
    eval( parse( text = initialize_ExactSol ), envir = evaluation_environment )
    if(strenv$ParameterizationType == "Implicit"){ getPiStar_exact <- generate_ExactSolImplicit }
    if(strenv$ParameterizationType == "Full"){ getPiStar_exact <- generate_ExactSolExplicit }
  }

  # pi in constrained space using gradient ascent
  p_vec_tf <- strenv$jnp$array(as.matrix(p_vec_use),dtype = strenv$dtj)
  inv_learning_rate <- strenv$jnp$array(1., dtype = strenv$dtj)

  # LR updates, etc.
  GetInvLR <- compile_fxn(function(inv_learning_rate,grad_i){
    # WN grad
    #return( (strenv$jnp$add(inv_learning_rate,strenv$jnp$divide(strenv$jnp$sum(strenv$jnp$square(grad_i)), inv_learning_rate))) )

    # Adagrad-norm
    return( (strenv$jnp$add(inv_learning_rate,strenv$jnp$sum(strenv$jnp$square(grad_i)))))
  })
  GetUpdatedParameters <- compile_fxn(function(a_vec, grad_i, inv_learning_rate_i){
    return( strenv$jnp$add(a_vec, strenv$jnp$multiply(strenv$jnp$reciprocal(inv_learning_rate_i), grad_i)))
  })

  a_vec_init_mat <- as.matrix(unlist( lapply(p_list, function(zer){ c(compositions::alr( t((zer)))) }) ) )
  a_vec_init_ast <- strenv$jnp$array(a_vec_init_mat+rnorm(length(a_vec_init_mat),sd = a_init_sd), strenv$dtj)
  a_vec_init_dag <- strenv$jnp$array(a_vec_init_mat+rnorm(length(a_vec_init_mat),sd = a_init_sd*adversarial), strenv$dtj)
  
  LabelSmoothingFxn <- (function(x, epsilon = 0.01){
      return( (1 - epsilon) * x + epsilon / strenv$jnp$array( x$shape[[1]] )$astype(x$dtype) ) })
  a2Simplex <- compile_fxn(function(a_){
    exp_a_ <- strenv$jnp$exp(a_)
    aOnSimplex <- tapply(1:nrow(a_structure),a_structure$d,function(zer){
      OnSimplex_ <- strenv$jnp$divide(  strenv$jnp$take(exp_a_, n2int(as.matrix(zer-1L) )),
                        strenv$jnp$add(
                          strenv$OneTf_flat,strenv$jnp$sum(strenv$jnp$take(exp_a_, n2int(zer-1L) ) )))
      OnSimplex_ <- LabelSmoothingFxn( OnSimplex_ )
      return( list( OnSimplex_ ) ) })
    names(aOnSimplex) <- NULL
    return(  strenv$jnp$concatenate(aOnSimplex,0L) )
  })
  a2FullSimplex <- compile_fxn(function(a_){
    # assumes holdout category is LAST 
    exp_a_ <- strenv$jnp$exp( a_ )
    aOnSimplex <- tapply(1:nrow(a_structure_leftoutLdminus1),a_structure_leftoutLdminus1$d,function(zer){
      exp_a_zer <- strenv$jnp$concatenate(list(strenv$jnp$take(exp_a_, n2int(as.matrix(as.array(zer - 1L) ))),
                                  strenv$jnp$array(as.matrix(1.))), # last category is exp(0) = 1
                                  axis = 0L)
      OnSimplex_ <- strenv$jnp$divide(  exp_a_zer, strenv$jnp$sum(exp_a_zer))
      OnSimplex_ <- LabelSmoothingFxn( OnSimplex_ )
      return( list( OnSimplex_ ) ) })
    names( aOnSimplex ) <- NULL
    return( strenv$jnp$concatenate(aOnSimplex,0L)  )
  })
  strenv$OneTf_flat <- strenv$jnp$squeeze(strenv$OneTf <- strenv$jnp$array(matrix(1L), strenv$dtj)$flatten(), 0L)
  Neg2_tf <- strenv$jnp$array(-2., strenv$dtj)

  message("Defining Q functions..")
  a2Simplex_optim <- ifelse( holdout_indicator == 1 ,
                             yes = list(a2Simplex),
                             no = list(a2FullSimplex) )[[1]]
  pi_star_value_init_ast <- a2Simplex_optim( a_vec_init_ast ) # a_ = a_vec_init_ast
  pi_star_value_init_dag <- a2Simplex_optim( a_vec_init_dag )
  
  if(adversarial){ stopifnot(length(unique(competing_group_variable_respondent))==2) }
  if(adversarial){ stopifnot(length(unique(competing_group_variable_candidate))==2) }

  # define Q functions
  environment(getQStar_single) <- evaluation_environment
  getQStar_single <- compile_fxn( getQStar_single )

  # multiround material
  for(DisaggreateQ in ifelse(adversarial, yes = list(c(F,T)), no = list(F))[[1]]){
    # general specifications
    getQStar_diff_ <- paste(deparse(getQStar_diff_BASE),collapse="\n")
    getQStar_diff_ <- gsub(getQStar_diff_, pattern = "Q_DISAGGREGATE",
                           replacement = sprintf("T == %s", DisaggreateQ))
    getQStar_diff_ <- eval( parse( text = getQStar_diff_ ), envir = evaluation_environment )

    # specifications for case (getQStar_diff_MultiGroup getQStar_diff_SingleGroup)
    eval(parse(text = sprintf("getQStar_diff_%sGroup <- compile_fxn( getQStar_diff_ )", 
                              ifelse(DisaggreateQ, yes = "Multi", no = "Single") )))
  }

  # Pretty Pi function
  {
    length_full_simplex <- length( unique( unlist( w_orig ) ) )
    length_simplex_use <- sum(  factor_levels  )

    # setup pretty pi
    add_to_term <- 1*rev(!duplicated(rev(d_locator)))
    # d_locator + add_to_term - CONFIRM DROP ??
    pi_star_value_loc <- rep(NA, times = n_main_params)
    if(strenv$ParameterizationType == "Implicit"){
      pi_star_value_loc_shadow <- rep(NA,times=length(unique(d_locator)))
      atShadow <- atSpot <- 0; for(ra in 1:length(pi_star_value_loc)){
        isLast <- sum(d_locator[ra:length(d_locator)] %in% d_locator[ra])==1
        if(!isLast){
          atSpot <- atSpot + 1 ;pi_star_value_loc[ra] <- atSpot
        }
        if(isLast){
          atSpot <- atSpot + 1
          pi_star_value_loc[ra] <- atSpot

          # account for shadow component
          atShadow <- atShadow + 1
          atSpot <- atSpot + 1
          pi_star_value_loc_shadow[atShadow] <- atSpot
        }
      }

      # re-normalize - go back from pretty for q
      {
        split_vec <- rep(0,times = length_simplex_use )
        split_vec[pi_star_value_loc_shadow] <- 1
        split_vec <- rev(cumsum(rev(split_vec)))
        split_vec <- cumsum(!duplicated(split_vec))
      }
      main_comp_mat <- matrix(0, ncol = n_main_params, nrow = length_simplex_use)
      main_comp_mat <- strenv$jnp$array(sapply(1:length(pi_star_value_loc),function(zer){
        main_comp_mat[pi_star_value_loc[zer],zer] <- 1
        return( main_comp_mat[,zer] ) }),strenv$dtj)
      strenv$main_comp_mat <- main_comp_mat

      shadow_comp_mat <- matrix(0, ncol = n_main_params, nrow = length_simplex_use)
      shadow_comp_mat <- strenv$jnp$array( sapply(1:length(pi_star_value_loc_shadow),function(zer){
        shadow_comp_mat[pi_star_value_loc_shadow[zer],zer] <- 1
        return( shadow_comp_mat[,zer] ) }),strenv$dtj)
      strenv$shadow_comp_mat <- shadow_comp_mat
    }

    split_vec_full <- unlist(sapply(1:length(factor_levels),function(xz){
                           rep(xz,times=factor_levels[xz])} ))
    split_vec_use <- ifelse(strenv$ParameterizationType == "Implicit",
                            yes = list(split_vec), no = list(split_vec_full))[[1]]
  }

  strenv$getPrettyPi <- compile_fxn( getPrettyPi_R <- getPrettyPi, static_argnums = 1L )
  strenv$getPrettyPi_diff <- compile_fxn(getPrettyPi_diff_R <- ifelse(strenv$ParameterizationType=="Implicit",
                                  yes = list( getPrettyPi_R ),
                                  no = list(function(x, a=NULL,b=NULL,c=NULL,d=NULL){x}))[[1]], 
                                  static_argnums = 1L)
  strenv$a2Simplex_diff_use <- ifelse(strenv$ParameterizationType == "Implicit",
                               yes = list(a2Simplex),
                               no = list(a2FullSimplex))[[1]]

  ## get exact result
  pi_star_exact <- -10; if(optim_type %in% c("tryboth") & diff == F){
    pi_star_exact <- strenv$np$array(
                                    strenv$getPrettyPi(   getPiStar_exact( EST_COEFFICIENTS_tf )  ,
                                                           strenv$ParameterizationType,
                                                           strenv$d_locator_use,       
                                                           strenv$main_comp_mat,   
                                                           strenv$shadow_comp_mat
                                                           ) 
                                     )  
  }

  use_exact <- !( use_gd <- (any(pi_star_exact<0) | any(pi_star_exact>1)  | adversarial |  diff | 
    (abs(sum(pi_star_exact) - sum(unlist(p_list_full))) > 1e-5) ))
  if( use_gd ){

  # define GD function
  p_vec_full_jnp <- strenv$jnp$array( as.matrix( p_vec_full ) )
  SLATE_VEC_ast_jnp <- SLATE_VEC_dag_jnp <- p_vec_jnp <- strenv$jnp$array(   as.matrix(p_vec)   )
  if(!is.null(slate_list)){ 
    SLATE_VEC_ast_jnp <- strenv$jnp$array( as.matrix( unlist(lapply(slate_list[[1]],function(zer){
      return( zer[-length(zer)] )# last position holdout 
      })) ) )
    SLATE_VEC_dag_jnp <- strenv$jnp$array( as.matrix( unlist(lapply(slate_list[[2]],function(zer){
      return( zer[-length(zer)] )# last position holdout 
      })) ) )
    # mean( names(unlist(slate_list[[1]])) == names(unlist(slate_list[[2]])) ) # target of 1 
  }
  
  if(use_optax == T){
      
      for(sfx in c("ast", "dag")){
        # Lr schedule 
        LR_schedule <- strenv$optax$warmup_cosine_decay_schedule(
                                              warmup_steps =  max(c(4L,0.1*nSGD)),
                                              decay_steps = max(c(4L,0.9*nSGD)),
                                              init_value = learning_rate_max/100, 
                                              peak_value = learning_rate_max, 
                                              end_value =  learning_rate_max/100)
        # model partition + setup state
        assign(name_optimizer <- paste0("optax_optimizer_", sfx), 
               strenv$optax$chain(
                 #strenv$optax$scale(-1),strenv$optax$scale_by_rss(initial_accumulator_value = 0.01)  
                 strenv$optax$scale(-1), strenv$optax$adabelief(LR_schedule)
          ))
        assign(paste0("opt_state_", sfx), eval(parse(text=name_optimizer))$init(strenv$jnp$array(get(paste0("a_vec_init_", sfx)))))
        assign(paste0("jit_apply_updates_", sfx), compile_fxn(strenv$optax$apply_updates))
        assign(paste0("jit_update_", sfx), compile_fxn( eval(parse(text=name_optimizer))$update))
      }
  }

  message("Defining gd function...")
  # bring functions into env and compile as needed
  strenv$getMultinomialSamp <- strenv$jax$jit(
    getMultinomialSamp_R,
    static_argnums = 3L,
    static_argnames = c("ParameterizationType")
  )
  environment(getQPiStar_gd) <- evaluation_environment
  }

  # get jax seed into correct type
  jax_seed <- strenv$jax$random$PRNGKey( ai(runif(1,1,1000)) )

  # Obtain solution via exact calculation
  message("Starting optimization...")
  q_star_OUTER <- q_star_se_OUTER <- pi_star_se_list_OUTER <- pi_star_list_OUTER <- replicate(n = K, list())
  for(k_clust in 1:K){
  if(K > 1){
    message(sprintf("Optimizing cluster %s of %s...",k_clust, K))
    ################################################
    # WARNING: Operational only in average case mode
    EST_INTERCEPT_tf <- strenv$jnp$array(t( my_mean_full[1,k_clust] ) )
    EST_COEFFICIENTS_tf <- strenv$jnp$array(as.matrix( my_mean_full[-1,k_clust] ) )
    REGRESSION_PARAMS_jax <- strenv$jnp$array(strenv$jnp$concatenate(list(EST_INTERCEPT_tf,
                                                                          EST_COEFFICIENTS_tf),0L))
    REGRESSION_PARAMS_jax_ast_jnp <- REGRESSION_PARAMS_jax
    REGRESSION_PARAMS_jax_dag_jnp <- REGRESSION_PARAMS_jax_ast0_jnp <- REGRESSION_PARAMS_jax_ast_jnp

    # reset covariates
    vcov_OutcomeModel_ast_jnp <- vcov_OutcomeModel_dag_jnp <- vcov_OutcomeModel_ast0_jnp <- vcov_OutcomeModel_dag0_jnp <- vcov_OutcomeModel_jnp <- vcov_OutcomeModel_by_k[[ k_clust ]]
  }
    
  # exact approach
  if(use_exact){
    gc()
    message("Optimization type: Exact")
    FxnForJacobian <- function(  INPUT_  ){
      EST_INTERCEPT_tf_ <- INPUT_[[1]]
      EST_COEFFICIENTS_tf_  <- INPUT_[[2]]
      pi_star_full_exact <- pi_star_full <- strenv$getPrettyPi( pi_star_reduced <- 
                                                           getPiStar_exact(EST_COEFFICIENTS_tf_), 
                                                           strenv$ParameterizationType,
                                                           strenv$d_locator_use,       
                                                           strenv$main_comp_mat,   
                                                           strenv$shadow_comp_mat)
      q_star_exact <- q_star <- getQStar_single(pi_star_ast = pi_star_reduced,
                                                pi_star_dag = pi_star_reduced,
                                                EST_INTERCEPT_tf_ast = EST_INTERCEPT_tf_,
                                                EST_COEFFICIENTS_tf_ast = EST_COEFFICIENTS_tf_,
                                                EST_INTERCEPT_tf_dag = EST_INTERCEPT_tf_,
                                                EST_COEFFICIENTS_tf_dag = EST_COEFFICIENTS_tf_)
      results_vec <- strenv$jnp$concatenate(list(q_star, 
                                                 pi_star_full),0L)
      return( results_vec )
    }
    results_vec <- FxnForJacobian( list(EST_INTERCEPT_tf,EST_COEFFICIENTS_tf) )
    jacobian_mat <- strenv$jax$jacobian(FxnForJacobian, 0L)(  list(EST_INTERCEPT_tf,
                                                                   EST_COEFFICIENTS_tf) ) 

    # reshape jacobian and process results
    jacobian_mat_exact <- jacobian_mat <- cbind(
        strenv$np$array(strenv$jnp$squeeze(strenv$jnp$squeeze(strenv$jnp$squeeze(jacobian_mat[[1]],1L),1L))),
        strenv$np$array(strenv$jnp$squeeze(strenv$jnp$squeeze(strenv$jnp$squeeze(jacobian_mat[[2]],1L),2L))) )
    vcov_OutcomeModel_concat <- vcov_OutcomeModel_ast_jnp
    q_star_exact <- q_star <- strenv$np$array( strenv$jnp$take(results_vec, 0L) )
    pi_star_full <- strenv$np$array( strenv$jnp$take(results_vec,
                                                     strenv$jnp$array((1L:length(results_vec))[-c(1:3)] -1L)))
    pi_star_reduced_exact <- getPiStar_exact(EST_COEFFICIENTS_tf)
    pi_star_red_ast <- strenv$jnp$array(as.matrix(strenv$np$array(pi_star_reduced_exact)))
    pi_star_red_dag <- pi_star_red_ast
  }

  # define main Q function in different cases
  if(!adversarial & !diff){ QFXN <- getQStar_single }
  if(!adversarial & diff){ QFXN <- getQStar_diff_SingleGroup }
  if(adversarial & diff){ QFXN <- getQStar_diff_MultiGroup }
  if(use_gd){
    message("Optimization type: Gradient ascent")

    # perform main gd runs + inference
    # first do ave case analysis
    p_vec_full_ast_jnp <- p_vec_full_dag_jnp <- p_vec_full_jnp

    # initialize QMonte fxns 
    InitializeQMonteFxns_impl <- if (outcome_model_type == "neural") {
      if (adversarial && primary_pushforward == "multi") {
        InitializeQMonteFxns_MultiCandidate
      } else {
        InitializeQMonteFxns_MCSampling
      }
    } else if (adversarial && primary_pushforward == "mc") {
      InitializeQMonteFxns_MCSampling
    } else if (adversarial && primary_pushforward == "multi") {
      InitializeQMonteFxns_MultiCandidate
    } else {
      InitializeQMonteFxns
    }
    InitializeQMonteFxns_ <- paste(deparse(InitializeQMonteFxns_impl),collapse="\n")
    InitializeQMonteFxns_ <- gsub(InitializeQMonteFxns_, pattern = "function \\(\\)", replacement = "")
    InitializeQMonteFxns_ <- eval( parse( text = InitializeQMonteFxns_ ), envir = evaluation_environment )

    # setup gd functions dparams
    environment(FullGetQStar_) <- evaluation_environment # keep to avoid having to pass all subparameters like nMonte 
    #FullGetQStar_ <- compile_fxn(FullGetQStar_, static_argnums = (static_q <- c(18L:19L) - 1L))
    FullGetQStar_ <- compile_fxn(FullGetQStar_, static_argnums = (static_q <- NULL))
    dQ_da_ast <- compile_fxn(strenv$jax$value_and_grad(FullGetQStar_, argnums = 0L),
                             static_argnums = static_q)
    dQ_da_dag <- compile_fxn(strenv$jax$value_and_grad(FullGetQStar_, argnums = 1L),
                             static_argnums = static_q)

    # Create Hessian functions for equilibrium geometry analysis (conditionally)
    # Only for adversarial mode - Hessians verify Nash equilibrium saddle point geometry
    n_params_per_player <- as.integer(strenv$np$array(a_vec_init_ast$shape[[1]]))

    if (adversarial) {
      should_compute_hessian <- compute_hessian && (n_params_per_player <= hessian_max_dim)

      if (should_compute_hessian) {
        message(sprintf("Creating Hessian functions (D=%d parameters per player)", n_params_per_player))
        d2Q_da2_ast <- compile_fxn(strenv$jax$hessian(FullGetQStar_, argnums = 0L),
                                   static_argnums = static_q)
        d2Q_da2_dag <- compile_fxn(strenv$jax$hessian(FullGetQStar_, argnums = 1L),
                                   static_argnums = static_q)
        hessian_available <- TRUE
        hessian_skipped_reason <- NULL
      } else if (!compute_hessian) {
        message("Skipping Hessian computation (compute_hessian=FALSE)")
        hessian_available <- FALSE
        hessian_skipped_reason <- "user_disabled"
      } else {
        message(sprintf("Skipping Hessian computation (D=%d > hessian_max_dim=%d)",
                        n_params_per_player, hessian_max_dim))
        hessian_available <- FALSE
        hessian_skipped_reason <- "high_dimension"
      }
    }
    # For non-adversarial mode, defaults set at initialization are used:
    # hessian_available = FALSE, hessian_skipped_reason = "not_adversarial"

    # perform GD 
    q_with_pi_star_full <- getQPiStar_gd(
      REGRESSION_PARAMETERS_ast   = REGRESSION_PARAMS_jax_ast_jnp,   #  1
      REGRESSION_PARAMETERS_dag   = REGRESSION_PARAMS_jax_dag_jnp,   #  2
      REGRESSION_PARAMETERS_ast0  = REGRESSION_PARAMS_jax_ast0_jnp,  #  3
      REGRESSION_PARAMETERS_dag0  = REGRESSION_PARAMS_jax_dag0_jnp,  #  4
      P_VEC_FULL_ast              = p_vec_full_ast_jnp,              #  5
      P_VEC_FULL_dag              = p_vec_full_dag_jnp,              #  6
      SLATE_VEC_ast               = SLATE_VEC_ast_jnp,               #  7
      SLATE_VEC_dag               = SLATE_VEC_dag_jnp,               #  8
      LAMBDA                      = strenv$jnp$array(lambda),        #  9
      SEED                        = jax_seed,                        # 10
      functionList                = list(dQ_da_ast, 
                                         dQ_da_dag,
                                         QFXN),  # 11
      a_i_ast                     = a_vec_init_ast,                   # 12
      a_i_dag                     = a_vec_init_dag,                   # 13
      
      functionReturn              = TRUE,                             # 14
      gd_full_simplex             = TRUE,                             # 15
      quiet                       = FALSE,                            # 16
      optimism                    = optimism,                         # 17
      optimism_coef               = optimism_coef,                    # 18
      rain_gamma                  = rain_gamma,                       # 19
      rain_eta                    = rain_eta                          # 20
    )
    dQ_da_ast <- q_with_pi_star_full[[2]]$dQ_da_ast
    dQ_da_dag <- q_with_pi_star_full[[2]]$dQ_da_dag
    a_i_ast_optimized <- q_with_pi_star_full[[2]]$a_i_ast
    a_i_dag_optimized <- q_with_pi_star_full[[2]]$a_i_dag
    QFXN <- q_with_pi_star_full[[2]]$QFXN
    q_with_pi_star_full <- strenv$jnp$array(q_with_pi_star_full[[1]], strenv$dtj)
    
    if(!use_optax){
      inv_learning_rate_ast_vec <- unlist(  lapply(strenv$inv_learning_rate_ast_vec,
                                                   function(zer){ strenv$np$array(zer) }))
    }
    
    grad_mag_ast_vec <- unlist(  lapply(strenv$grad_mag_ast_vec,function(zer){
      strenv$np$array(strenv$jnp$sqrt( strenv$jnp$sum(strenv$jnp$square(strenv$jnp$array(zer,strenv$dtj))) ))  }) )
    try(suppressWarnings(plot( grad_mag_ast_vec, main = "Gradient Magnitude Evolution (ast)", log ="y")),T)
    try(points(lowess(grad_mag_ast_vec), cex = 2, type = "l",lwd = 2, col = "red"), T)
    
    if(adversarial){ 
      grad_mag_dag_vec <- try(unlist(  lapply(strenv$grad_mag_dag_vec,function(zer){
        strenv$np$array(strenv$jnp$sqrt( strenv$jnp$sum(strenv$jnp$square(strenv$jnp$array(zer,strenv$dtj))) )) }) ),T)
      try(suppressWarnings(plot( grad_mag_dag_vec , main = "Gradient Magnitude Evolution (dag)",log="y")),T)
      try(points(lowess(grad_mag_dag_vec), cex = 2, type = "l",lwd = 2, col = "red"), T)
    }
    
    loss_ast_vec <- strenv$np$array(strenv$jnp$stack(strenv$loss_ast_vec,0L))
    try(suppressWarnings(plot( loss_ast_vec, main = "Value (ast)", log ="y")),T)
    if(adversarial){ 
      loss_dag_vec <- strenv$np$array(strenv$jnp$stack(strenv$loss_dag_vec,0L))
      try(suppressWarnings(plot( loss_dag_vec, main = "Value (dag)", log ="y")),T)
    }
    
    pi_star_red <- getQPiStar_gd(
                        REGRESSION_PARAMETERS_ast = REGRESSION_PARAMS_jax_ast_jnp,
                        REGRESSION_PARAMETERS_dag = REGRESSION_PARAMS_jax_dag_jnp,
                        REGRESSION_PARAMETERS_ast0 = REGRESSION_PARAMS_jax_ast0_jnp,
                        REGRESSION_PARAMETERS_dag0 = REGRESSION_PARAMS_jax_dag0_jnp,
                        P_VEC_FULL_ast = p_vec_full_ast_jnp,
                        P_VEC_FULL_dag = p_vec_full_dag_jnp,
                        SLATE_VEC_ast = SLATE_VEC_ast_jnp, 
                        SLATE_VEC_dag = SLATE_VEC_dag_jnp,
                        LAMBDA = strenv$jnp$array(  lambda  ),
                        SEED   = jax_seed,
                        functionList = list(dQ_da_ast, dQ_da_dag,
                                            QFXN),
                        a_i_ast = a_vec_init_ast, 
                        a_i_dag = a_vec_init_dag, 
                        functionReturn  = FALSE,
                        gd_full_simplex = FALSE, 
                        quiet           = FALSE,
                        optimism        = optimism,                            # 
                        optimism_coef   = optimism_coef,
                        rain_gamma      = rain_gamma,
                        rain_eta        = rain_eta
                        )
    pi_star_red <- strenv$np$array(pi_star_red)[-c(1:3),]
    pi_star_red_ast <- strenv$jnp$array(as.matrix(  pi_star_red[1:(length(pi_star_red)/2)] ) )
    pi_star_red_dag <- strenv$jnp$array(as.matrix(  pi_star_red[-c(1:(length(pi_star_red)/2))]))

    q_star_gd <- q_star <- strenv$np$array(  q_with_pi_star_full )[1]
    # sanity check: 
    # strenv$np$array(  q_with_pi_star_full )[1]  - sum(strenv$np$array(  q_with_pi_star_full )[2:3]*c(strenv$AstProp, strenv$DagProp)) 
    pi_star_full_gd <- pi_star_full <- strenv$np$array( q_with_pi_star_full )[-c(1:3)]

    #  https://github.com/google/jax/issues/1696 
    jacobian_mat_gd <- jacobian_mat <- matrix(0, ncol = 4*REGRESSION_PARAMS_jax_ast_jnp$shape[[1]],
                                                 nrow = q_with_pi_star_full$shape[[1]])
    diag(jacobian_mat_gd) <- diag(jacobian_mat) <- 1
    if (is.null(dim(vcov_OutcomeModel_ast_jnp))) {
      vcov_OutcomeModel_concat <- rep(0, length(vcov_OutcomeModel_ast_jnp) * 4L)
    } else {
      vcov_OutcomeModel_concat <- matrix(0, nrow = nrow(vcov_OutcomeModel_ast_jnp) * 4L,
                                            ncol = nrow(vcov_OutcomeModel_ast_jnp) * 4L)
    }
    if(compute_se){
      message("Computing SEs...")
      # Preserve convergence history before jacrev re-runs getQPiStar_gd.
      convergence_cache <- list(
        grad_mag_ast_vec = strenv$grad_mag_ast_vec,
        grad_mag_dag_vec = strenv$grad_mag_dag_vec,
        loss_ast_vec = strenv$loss_ast_vec,
        loss_dag_vec = strenv$loss_dag_vec,
        inv_learning_rate_ast_vec = strenv$inv_learning_rate_ast_vec,
        inv_learning_rate_dag_vec = strenv$inv_learning_rate_dag_vec
      )
      # first, compute vcov
      if (is.null(dim(vcov_OutcomeModel_ast_jnp))) {
        vcov_OutcomeModel_concat <- c(vcov_OutcomeModel_ast_jnp,
                                      vcov_OutcomeModel_dag_jnp,
                                      vcov_OutcomeModel_ast0_jnp,
                                      vcov_OutcomeModel_dag0_jnp)
      } else {
        vcov_OutcomeModel_concat <- as.matrix( Matrix::bdiag( list(
                                            vcov_OutcomeModel_ast_jnp,
                                            vcov_OutcomeModel_dag_jnp,
                                            vcov_OutcomeModel_ast0_jnp,
                                            vcov_OutcomeModel_dag0_jnp  )  ) )
      }

      se_method_effective <- se_method
      if (se_method_effective == "full") {
        # jacfwd uses forward-mode automatic differentiation, which is more efficient for "tall" Jacobian matrices
        # jacrev uses reverse-mode, which is more efficient for "wide" Jacobian matrices.
        # For near-square matrices, jacfwd probably has an edge over jacrev.
        # note: do not jit compile as computation only used once (compilation induces overhead)
        jacobian_mat <- strenv$jax$jacrev(getQPiStar_gd, 0L:3L)(
                                    REGRESSION_PARAMS_jax_ast_jnp,
                                    REGRESSION_PARAMS_jax_dag_jnp,
                                    REGRESSION_PARAMS_jax_ast0_jnp,
                                    REGRESSION_PARAMS_jax_dag0_jnp,
                                    p_vec_full_ast_jnp,
                                    p_vec_full_dag_jnp,
                                    SLATE_VEC_ast_jnp, 
                                    SLATE_VEC_dag_jnp,
                                    strenv$jnp$array(  lambda  ),
                                    jax_seed,
                                    functionList = list(dQ_da_ast,
                                                        dQ_da_dag,
                                                        QFXN),
                                    functionReturn = F,
                                    a_i_ast = a_vec_init_ast, 
                                    a_i_dag = a_vec_init_dag, 
                                    gd_full_simplex = T,
                                    quiet = F
                                    )
        jacobian_mat_gd <- jacobian_mat <- lapply(jacobian_mat,function(l_){
          strenv$np$array( strenv$jnp$squeeze(strenv$jnp$squeeze(strenv$jnp$array(l_,strenv$dtj),1L),2L) ) })
        jacobian_mat_gd <- jacobian_mat <- do.call(cbind, jacobian_mat)
      } else {
        message("Computing SEs via implicit differentiation...")
        reshape_jac <- function(jac){
          strenv$jnp$reshape(jac, list(jac$shape[[1L]], -1L))
        }
        as_jac_list <- function(jac_list){
          if (is.list(jac_list)) {
            return(jac_list)
          }
          list(jac_list)
        }
        concat_jac <- function(jac_list){
          jac_list <- as_jac_list(jac_list)
          strenv$jnp$concatenate(lapply(jac_list, reshape_jac), axis = 1L)
        }

        StationarityFxn <- function(a_i_ast, a_i_dag,
                                    REGRESSION_PARAMETERS_ast,
                                    REGRESSION_PARAMETERS_dag,
                                    REGRESSION_PARAMETERS_ast0,
                                    REGRESSION_PARAMETERS_dag0,
                                    SEED_IN_LOOP){
          SEED_IN_LOOP <- strenv$jax$lax$stop_gradient(SEED_IN_LOOP)
          REGRESSION_PARAMETERS_ast <- gather_fxn(REGRESSION_PARAMETERS_ast)
          INTERCEPT_ast_ <- REGRESSION_PARAMETERS_ast[[1]]
          COEFFICIENTS_ast_ <- REGRESSION_PARAMETERS_ast[[2]]

          INTERCEPT_dag0_ <- INTERCEPT_ast0_ <- INTERCEPT_dag_ <- INTERCEPT_ast_
          COEFFICIENTS_dag0_ <- COEFFICIENTS_ast0_ <- COEFFICIENTS_dag_ <- COEFFICIENTS_ast_
          if(adversarial){
            REGRESSION_PARAMETERS_dag <- gather_fxn(REGRESSION_PARAMETERS_dag)
            INTERCEPT_dag_ <- REGRESSION_PARAMETERS_dag[[1]]
            COEFFICIENTS_dag_ <- REGRESSION_PARAMETERS_dag[[2]]
          }
          if(adversarial && isTRUE(use_stage_models)){
            REGRESSION_PARAMETERS_ast0 <- gather_fxn(REGRESSION_PARAMETERS_ast0)
            INTERCEPT_ast0_ <- REGRESSION_PARAMETERS_ast0[[1]]
            COEFFICIENTS_ast0_ <- REGRESSION_PARAMETERS_ast0[[2]]

            REGRESSION_PARAMETERS_dag0 <- gather_fxn(REGRESSION_PARAMETERS_dag0)
            INTERCEPT_dag0_ <- REGRESSION_PARAMETERS_dag0[[1]]
            COEFFICIENTS_dag0_ <- REGRESSION_PARAMETERS_dag0[[2]]
          }

          grad_ast <- dQ_da_ast(  a_i_ast, a_i_dag,
                                  INTERCEPT_ast_,  COEFFICIENTS_ast_,
                                  INTERCEPT_dag_,  COEFFICIENTS_dag_,
                                  INTERCEPT_ast0_, COEFFICIENTS_ast0_,
                                  INTERCEPT_dag0_, COEFFICIENTS_dag0_,
                                  p_vec_full_ast_jnp, p_vec_full_dag_jnp,
                                  SLATE_VEC_ast_jnp, SLATE_VEC_dag_jnp,
                                  strenv$jnp$array(  lambda  ),
                                  Q_SIGN_ <- strenv$jnp$array(1.),
                                  SEED_IN_LOOP
                                  )[[2]]
          grad_ast <- strenv$jnp$reshape(grad_ast, list(-1L))
          if(!adversarial){
            return(grad_ast)
          }
          grad_dag <- dQ_da_dag(  a_i_ast, a_i_dag,
                                  INTERCEPT_ast_,  COEFFICIENTS_ast_,
                                  INTERCEPT_dag_,  COEFFICIENTS_dag_,
                                  INTERCEPT_ast0_, COEFFICIENTS_ast0_,
                                  INTERCEPT_dag0_, COEFFICIENTS_dag0_,
                                  p_vec_full_ast_jnp, p_vec_full_dag_jnp,
                                  SLATE_VEC_ast_jnp, SLATE_VEC_dag_jnp,
                                  strenv$jnp$array(  lambda  ),
                                  Q_SIGN_ <- strenv$jnp$array(-1.),
                                  SEED_IN_LOOP
                                  )[[2]]
          grad_dag <- strenv$jnp$reshape(grad_dag, list(-1L))
          strenv$jnp$concatenate(list(grad_ast, grad_dag), 0L)
        }

        OutputFxn <- function(a_i_ast, a_i_dag,
                              REGRESSION_PARAMETERS_ast,
                              REGRESSION_PARAMETERS_dag,
                              REGRESSION_PARAMETERS_ast0,
                              REGRESSION_PARAMETERS_dag0,
                              SEED_IN_LOOP){
          SEED_IN_LOOP <- strenv$jax$lax$stop_gradient(SEED_IN_LOOP)
          REGRESSION_PARAMETERS_ast <- gather_fxn(REGRESSION_PARAMETERS_ast)
          INTERCEPT_ast_ <- REGRESSION_PARAMETERS_ast[[1]]
          COEFFICIENTS_ast_ <- REGRESSION_PARAMETERS_ast[[2]]

          INTERCEPT_dag0_ <- INTERCEPT_ast0_ <- INTERCEPT_dag_ <- INTERCEPT_ast_
          COEFFICIENTS_dag0_ <- COEFFICIENTS_ast0_ <- COEFFICIENTS_dag_ <- COEFFICIENTS_ast_
          if(adversarial){
            REGRESSION_PARAMETERS_dag <- gather_fxn(REGRESSION_PARAMETERS_dag)
            INTERCEPT_dag_ <- REGRESSION_PARAMETERS_dag[[1]]
            COEFFICIENTS_dag_ <- REGRESSION_PARAMETERS_dag[[2]]
          }
          if(adversarial && isTRUE(use_stage_models)){
            REGRESSION_PARAMETERS_ast0 <- gather_fxn(REGRESSION_PARAMETERS_ast0)
            INTERCEPT_ast0_ <- REGRESSION_PARAMETERS_ast0[[1]]
            COEFFICIENTS_ast0_ <- REGRESSION_PARAMETERS_ast0[[2]]

            REGRESSION_PARAMETERS_dag0 <- gather_fxn(REGRESSION_PARAMETERS_dag0)
            INTERCEPT_dag0_ <- REGRESSION_PARAMETERS_dag0[[1]]
            COEFFICIENTS_dag0_ <- REGRESSION_PARAMETERS_dag0[[2]]
          }

          pi_star_ast_full_simplex_ <- getPrettyPi( pi_star_ast_<-strenv$a2Simplex_diff_use(a_i_ast),
                                                    strenv$ParameterizationType,
                                                    strenv$d_locator_use,
                                                    strenv$main_comp_mat,
                                                    strenv$shadow_comp_mat)
          pi_star_dag_full_simplex_ <- getPrettyPi( pi_star_dag_<-strenv$a2Simplex_diff_use(a_i_dag),
                                                    strenv$ParameterizationType,
                                                    strenv$d_locator_use,
                                                    strenv$main_comp_mat,
                                                    strenv$shadow_comp_mat)

          if(glm_family=="gaussian"){
            pi_star_ast_f_all <- strenv$jnp$expand_dims(pi_star_ast_,0L)
            pi_star_dag_f_all <- strenv$jnp$expand_dims(pi_star_dag_,0L)
          }
          if(glm_family != "gaussian"){
            draw_ast <- draw_profile_samples(
              pi_star_ast_, nMonte_Qglm, SEED_IN_LOOP,
              MNtemp, strenv$ParameterizationType, strenv$d_locator_use,
              sampler = strenv$getMultinomialSamp
            )
            pi_star_ast_f_all <- draw_ast$samples
            SEED_IN_LOOP <- draw_ast$seed_next

            draw_dag <- draw_profile_samples(
              pi_star_dag_, nMonte_Qglm, SEED_IN_LOOP,
              MNtemp, strenv$ParameterizationType, strenv$d_locator_use,
              sampler = strenv$getMultinomialSamp
            )
            pi_star_dag_f_all <- draw_dag$samples
            SEED_IN_LOOP <- draw_dag$seed_next
          }

          if(!adversarial){
            q_star_f <- strenv$Vectorized_QMonteIter(
                                          pi_star_ast_f_all,  pi_star_dag_f_all,
                                          INTERCEPT_ast_, COEFFICIENTS_ast_,
                                          INTERCEPT_dag_, COEFFICIENTS_dag_)$mean(0L)
          }
          if(adversarial){
            n_q_samp <- as.integer(pi_star_ast_f_all$shape[[1L]])
            if (primary_pushforward == "multi") {
              samp_ast <- sample_pool_jax(
                pi_star_ast_, n_q_samp, primary_n_entrants, SEED_IN_LOOP,
                MNtemp, strenv$ParameterizationType, strenv$d_locator_use,
                sampler = strenv$getMultinomialSamp
              )
              TSAMP_ast_all <- samp_ast$samples
              SEED_IN_LOOP <- samp_ast$seed_next

              samp_dag <- sample_pool_jax(
                pi_star_dag_, n_q_samp, primary_n_entrants, SEED_IN_LOOP,
                MNtemp, strenv$ParameterizationType, strenv$d_locator_use,
                sampler = strenv$getMultinomialSamp
              )
              TSAMP_dag_all <- samp_dag$samples
              SEED_IN_LOOP <- samp_dag$seed_next

              samp_ast_field <- sample_pool_jax(
                SLATE_VEC_ast_jnp, n_q_samp, primary_n_field, SEED_IN_LOOP,
                MNtemp, strenv$ParameterizationType, strenv$d_locator_use,
                sampler = strenv$getMultinomialSamp
              )
              TSAMP_ast_PrimaryComp_all <- samp_ast_field$samples
              SEED_IN_LOOP <- samp_ast_field$seed_next

              samp_dag_field <- sample_pool_jax(
                SLATE_VEC_dag_jnp, n_q_samp, primary_n_field, SEED_IN_LOOP,
                MNtemp, strenv$ParameterizationType, strenv$d_locator_use,
                sampler = strenv$getMultinomialSamp
              )
              TSAMP_dag_PrimaryComp_all <- samp_dag_field$samples
              SEED_IN_LOOP <- samp_dag_field$seed_next
            } else {
              TSAMP_ast_all <- pi_star_ast_f_all
              TSAMP_dag_all <- pi_star_dag_f_all
              draw_ast_field <- draw_profile_samples(
                SLATE_VEC_ast_jnp, n_q_samp, SEED_IN_LOOP,
                MNtemp, strenv$ParameterizationType, strenv$d_locator_use,
                sampler = strenv$getMultinomialSamp
              )
              TSAMP_ast_PrimaryComp_all <- draw_ast_field$samples
              SEED_IN_LOOP <- draw_ast_field$seed_next

              draw_dag_field <- draw_profile_samples(
                SLATE_VEC_dag_jnp, n_q_samp, SEED_IN_LOOP,
                MNtemp, strenv$ParameterizationType, strenv$d_locator_use,
                sampler = strenv$getMultinomialSamp
              )
              TSAMP_dag_PrimaryComp_all <- draw_dag_field$samples
              SEED_IN_LOOP <- draw_dag_field$seed_next
            }

            Qres <- strenv$Vectorized_QMonteIter_MaxMin(
              TSAMP_ast_all, TSAMP_dag_all,
              TSAMP_ast_PrimaryComp_all, TSAMP_dag_PrimaryComp_all,
              a_i_ast, a_i_dag,
              INTERCEPT_ast_, COEFFICIENTS_ast_,
              INTERCEPT_dag_, COEFFICIENTS_dag_,
              INTERCEPT_ast0_, COEFFICIENTS_ast0_,
              INTERCEPT_dag0_, COEFFICIENTS_dag0_,
              p_vec_full_ast_jnp, p_vec_full_dag_jnp,
              strenv$jnp$array(  lambda  ), strenv$jnp$array(1.),
              strenv$jax$random$split(SEED_IN_LOOP, n_q_samp)
            )
            q_star_val <- Qres$q_ast$mean(0L)
            q_star_dims <- if (length(pi_star_ast_full_simplex_$shape) >= 2L) {
              list(1L, 1L)
            } else {
              list(1L)
            }
            q_star_val <- strenv$jnp$reshape(q_star_val, q_star_dims)
            q_star_f <- strenv$jnp$concatenate(list(q_star_val, q_star_val, q_star_val), 0L)
          }

          strenv$jnp$concatenate(list(q_star_f,
                                      pi_star_ast_full_simplex_,
                                      pi_star_dag_full_simplex_), 0L)
        }

        dF_da_list <- if (adversarial) {
          strenv$jax$jacrev(StationarityFxn, 0L:1L)
        } else {
          strenv$jax$jacrev(StationarityFxn, 0L)
        }
        dF_da_list <- dF_da_list(
          a_i_ast_optimized,
          a_i_dag_optimized,
          REGRESSION_PARAMS_jax_ast_jnp,
          REGRESSION_PARAMS_jax_dag_jnp,
          REGRESSION_PARAMS_jax_ast0_jnp,
          REGRESSION_PARAMS_jax_dag0_jnp,
          jax_seed
        )
        dF_dtheta_list <- strenv$jax$jacrev(StationarityFxn, 2L:5L)(
          a_i_ast_optimized,
          a_i_dag_optimized,
          REGRESSION_PARAMS_jax_ast_jnp,
          REGRESSION_PARAMS_jax_dag_jnp,
          REGRESSION_PARAMS_jax_ast0_jnp,
          REGRESSION_PARAMS_jax_dag0_jnp,
          jax_seed
        )
        dF_da_mat <- concat_jac(dF_da_list)
        dF_dtheta_mat <- concat_jac(dF_dtheta_list)

        da_dtheta <- strenv$jnp$linalg$solve(
          dF_da_mat,
          strenv$jnp$negative(dF_dtheta_mat)
        )

        g_a_list <- if (adversarial) {
          strenv$jax$jacrev(OutputFxn, 0L:1L)
        } else {
          strenv$jax$jacrev(OutputFxn, 0L)
        }
        g_a_list <- g_a_list(
          a_i_ast_optimized,
          a_i_dag_optimized,
          REGRESSION_PARAMS_jax_ast_jnp,
          REGRESSION_PARAMS_jax_dag_jnp,
          REGRESSION_PARAMS_jax_ast0_jnp,
          REGRESSION_PARAMS_jax_dag0_jnp,
          jax_seed
        )
        g_theta_list <- strenv$jax$jacrev(OutputFxn, 2L:5L)(
          a_i_ast_optimized,
          a_i_dag_optimized,
          REGRESSION_PARAMS_jax_ast_jnp,
          REGRESSION_PARAMS_jax_dag_jnp,
          REGRESSION_PARAMS_jax_ast0_jnp,
          REGRESSION_PARAMS_jax_dag0_jnp,
          jax_seed
        )
        g_a_mat <- concat_jac(g_a_list)
        g_theta_mat <- concat_jac(g_theta_list)

        jacobian_mat <- strenv$jnp$add(g_theta_mat, strenv$jnp$matmul(g_a_mat, da_dtheta))
        jacobian_mat_gd <- jacobian_mat <- as.matrix(strenv$np$array(jacobian_mat))
      }
      # plot(colMeans(abs(jacobian_mat_gd)))
      strenv$grad_mag_ast_vec <- convergence_cache$grad_mag_ast_vec
      strenv$grad_mag_dag_vec <- convergence_cache$grad_mag_dag_vec
      strenv$loss_ast_vec <- convergence_cache$loss_ast_vec
      strenv$loss_dag_vec <- convergence_cache$loss_dag_vec
      strenv$inv_learning_rate_ast_vec <- convergence_cache$inv_learning_rate_ast_vec
      strenv$inv_learning_rate_dag_vec <- convergence_cache$inv_learning_rate_dag_vec
    }
  }

  # the first three entries of output are:
  # Qhat_population, Qhat_ast, Qhat_dag
  if (is.null(dim(vcov_OutcomeModel_concat))) {
    vcov_diag <- as.numeric(vcov_OutcomeModel_concat)
    n_params <- ncol(jacobian_mat)
    if (length(vcov_diag) < n_params) {
      vcov_diag <- c(vcov_diag, rep(0, n_params - length(vcov_diag)))
    }
    if (length(vcov_diag) > n_params) {
      vcov_diag <- vcov_diag[seq_len(n_params)]
    }
    vcov_diag_out <- as.numeric((jacobian_mat ^ 2) %*% vcov_diag)
    vcov_PiStar <- diag(vcov_diag_out)
  } else {
    vcov_PiStar <- jacobian_mat %*% vcov_OutcomeModel_concat %*% t(jacobian_mat)
  }
  q_star <- as.matrix(   q_star  )
  q_star_se <- sqrt(  diag( vcov_PiStar )[1] )
  pi_star_numeric <- strenv$np$array( pi_star_full ) # - c(1:3) already extracted 

  # drop the q part
  if(diff == T){ 
    pi_star_se <- sqrt(  diag( vcov_PiStar )[-c(1:3)] )
  }
  if(diff == F){
    # CHECK HERE - CHECK 
    take_indices <- 1:length( pi_star_numeric )
    if(use_gd){ take_indices <- 1:(length(pi_star_numeric)/2 )  }
    pi_star_numeric <- pi_star_numeric[take_indices]
    pi_star_se <- sqrt(  diag( vcov_PiStar )[-c(1:3)][take_indices] )
    
    # setup pretty pi's
    pi_star_se_list <- pi_star_list <- list()
    pi_star_list$k1 <- (  split(pi_star_numeric, split_vec_use) ) # previously split_vec
    pi_star_se_list$k1 <- (  split(pi_star_se, split_vec_use) )
  }
  if(diff == T){
    pi_star_se_list <- pi_star_list <- list()
    pi_star_list$k1 <- split(pi_star_numeric[1:length(p_vec_full)], split_vec_use)
    pi_star_se_list$k1 <- split(pi_star_se[1:length(p_vec_full)], split_vec_use)

    # save jnp for later
    pi_star_vec_jnp <- strenv$jnp$array(as.matrix(pi_star_numeric[1:length(p_vec_full)]))
    pi_star_dag_vec_jnp <- strenv$jnp$array(as.matrix(pi_star_numeric[-c(1:length(p_vec_full))]))
    pi_star_list$k2 <- split(pi_star_numeric[-c(1:length(p_vec_full))], split_vec_use)
    pi_star_se_list$k2 <- split(pi_star_se[-c(1:length(p_vec_full))], split_vec_use)
  }

  # re-jig to account for regularization
  pi_star_list <- RejiggerPi(pi_ = pi_star_list, isSE = F  )
  pi_star_se_list <- RejiggerPi(pi_ = pi_star_se_list, isSE = T  )
  
  # append to outer list for K > 1 case
  pi_star_list_OUTER[[k_clust]] <- (pi_star_list <- RenamePiList(  pi_star_list  ))
  pi_star_se_list_OUTER[[k_clust]] <- (pi_star_se_list <- RenamePiList(  pi_star_se_list  ))
  q_star_OUTER[[k_clust]] <- q_star
  q_star_se_OUTER[[k_clust]] <- q_star_se
  } # end loop k in 1, ..., K

  # reset names for K > 1 case
  if(K > 1){
    pi_star_list <- lapply(pi_star_list_OUTER, function(l_){ l_$k1 })
    names( pi_star_list ) <- paste("k",  1:K, sep = "")

    pi_star_se_list <- lapply(pi_star_se_list_OUTER, function(l_){ l_$k1 })
    names( pi_star_se_list ) <- paste("k",  1:K, sep = "")

    q_star <- unlist( q_star_OUTER )
    q_star_se <- unlist( q_star_se_OUTER )
    names(q_star_se) <- names(q_star) <- paste("k",  1:K, sep = "")
  }

  for(sign_ in c(-1,1)){
    bound_ <- lapply(1:max(c(length(GroupsPool),K)),function(k_){
       l_ <- sapply(1:length(pi_star_list[[k_]]),function(zer){
          ret_ <- list( pi_star_list[[k_]][[zer]] + sign_*abs(qnorm((1-conf_level)/2))*pi_star_se_list[[k_]][[zer]] )
          names(ret_) <- names(pi_star_list[[k_]])[zer]
          return(    ret_   )   })
       return(l_) })
    names(bound_) <- paste("k",1:length(bound_),sep="")
    if(sign_ == -1){ lowerList <- bound_ }
    if(sign_ == 1){ upperList <- bound_ }
  }

  if(!diff){
    # Set _vec_jnp to full pi_star, but keep pi_star_red_ast/dag in reduced form for QFXN compatibility
    pi_star_dag_vec_jnp <- pi_star_vec_jnp <- pi_star_numeric
    p_vec_full_ast_jnp <- p_vec_full_dag_jnp <- p_vec_full
  }
  
  if(adversarial){
    names(pi_star_list) <- names(pi_star_list) <- GroupsPool
    if(compute_se){
      names(lowerList) <- names(upperList) <- GroupsPool
      names(pi_star_se_list) <- GroupsPool
    }
  }
  
  # append to strenv 
  strenv$dQ_da_ast          <- dQ_da_ast
  strenv$dQ_da_dag          <- dQ_da_dag
  strenv$QFXN               <- QFXN
  strenv$getQPiStar_gd      <- getQPiStar_gd

  # Build interpretable summaries for outcome models
  outcome_model_view <- (function() {
    to_scalar <- function(x) {
      if (is.null(x)) return(NA_real_)
      out <- tryCatch(as.numeric(x), error = function(e) NULL)
      if (!is.null(out) && length(out) > 0 && is.finite(out[1])) return(out[1])
      out <- tryCatch(as.numeric(strenv$np$array(x)), error = function(e) NA_real_)
      if (length(out) == 0) return(NA_real_)
      out[1]
    }

    to_numeric <- function(x) {
      if (is.null(x)) return(numeric(0))
      out <- tryCatch(as.numeric(x), error = function(e) NULL)
      if (!is.null(out) && length(out) > 0) return(out)
      out <- tryCatch(as.numeric(strenv$np$array(x)), error = function(e) numeric(0))
      out
    }

    flatten_levels <- function(x) {
      if (is.list(x) && length(x) == 1 && is.atomic(x[[1]])) return(as.character(x[[1]]))
      if (is.atomic(x)) return(as.character(x))
      as.character(unlist(x))
    }

    factor_names <- colnames(w_orig)
    if (is.null(factor_names)) {
      factor_names <- paste0("Factor", seq_len(ncol(w_orig)))
    }
    level_names <- lapply(names_list, flatten_levels)

    linkinv <- if (glm_family == "binomial") stats::plogis else function(x) x

    build_main_effects <- function(main_info, coef_vec, intercept, vcov) {
      if (is.null(main_info) || nrow(main_info) == 0) return(NULL)
      coef_vec <- to_numeric(coef_vec)
      n_main <- nrow(main_info)
      if (length(coef_vec) < n_main) return(NULL)
      coef_main <- coef_vec[seq_len(n_main)]
      base_pred <- if (is.na(intercept)) NA_real_ else linkinv(intercept)

      rows <- vector("list", length = sum(factor_levels))
      row_idx <- 1L
      for (d in seq_len(length(factor_levels))) {
        levels_d <- level_names[[d]]
        if (is.null(levels_d) || length(levels_d) != factor_levels[d]) {
          levels_d <- as.character(seq_len(factor_levels[d]))
        }
        for (l in seq_len(factor_levels[d])) {
          coef <- 0
          coef_se <- NA_real_
          is_baseline <- TRUE
          mi <- which(main_info$d == d & main_info$l == l)
          if (length(mi) > 0) {
            mi <- mi[1]
            coef_idx <- main_info$d_index[mi]
            if (!is.na(coef_idx) && coef_idx >= 1 && coef_idx <= length(coef_main)) {
              coef <- coef_main[coef_idx]
              is_baseline <- FALSE
              vcov_idx <- 1 + coef_idx
              if (!is.null(vcov) && length(dim(vcov)) == 2 &&
                  vcov_idx <= nrow(vcov)) {
                coef_se <- sqrt(vcov[vcov_idx, vcov_idx])
              }
            }
          }

          eta <- if (is.na(intercept)) NA_real_ else intercept + coef
          pred <- if (is.na(eta)) NA_real_ else linkinv(eta)
          delta <- if (is.na(pred) || is.na(base_pred)) NA_real_ else pred - base_pred
          odds_ratio <- if (glm_family == "binomial") exp(coef) else NA_real_

          rows[[row_idx]] <- data.frame(
            Factor = factor_names[d],
            Level = levels_d[l],
            Coef = coef,
            CoefSE = coef_se,
            Pred = pred,
            Delta = delta,
            OddsRatio = odds_ratio,
            IsBaseline = is_baseline,
            stringsAsFactors = FALSE
          )
          row_idx <- row_idx + 1L
        }
      }
      do.call(rbind, rows)
    }

    build_interactions <- function(interaction_info, coef_vec, n_main, top_n = 10L) {
      if (is.null(interaction_info) || nrow(interaction_info) == 0) return(NULL)
      coef_vec <- to_numeric(coef_vec)
      if (length(coef_vec) <= n_main) return(NULL)
      coef_inter <- coef_vec[(n_main + 1L):length(coef_vec)]
      n_inter <- min(nrow(interaction_info), length(coef_inter))
      if (n_inter <= 0) return(NULL)
      interaction_info <- interaction_info[seq_len(n_inter), , drop = FALSE]
      coef_inter <- coef_inter[seq_len(n_inter)]

      labels <- vapply(seq_len(n_inter), function(i) {
        d1 <- interaction_info$d[i]
        d2 <- interaction_info$dp[i]
        l1 <- interaction_info$l[i]
        l2 <- interaction_info$lp[i]
        f1 <- if (d1 <= length(factor_names)) factor_names[d1] else paste0("Factor", d1)
        f2 <- if (d2 <= length(factor_names)) factor_names[d2] else paste0("Factor", d2)
        lv1 <- if (!is.null(level_names[[d1]]) && length(level_names[[d1]]) >= l1) {
          level_names[[d1]][l1]
        } else {
          as.character(l1)
        }
        lv2 <- if (!is.null(level_names[[d2]]) && length(level_names[[d2]]) >= l2) {
          level_names[[d2]][l2]
        } else {
          as.character(l2)
        }
        sprintf("%s:%s x %s:%s", f1, lv1, f2, lv2)
      }, character(1))

      inter_df <- data.frame(
        Interaction = labels,
        Coef = coef_inter,
        AbsCoef = abs(coef_inter),
        stringsAsFactors = FALSE
      )
      inter_df <- inter_df[order(-inter_df$AbsCoef), , drop = FALSE]
      if (!is.null(top_n) && nrow(inter_df) > top_n) {
        inter_df <- inter_df[seq_len(top_n), , drop = FALSE]
      }
      inter_df$AbsCoef <- NULL
      inter_df
    }

    build_model_view <- function(suffix, stage, player, group_value) {
      mean_name <- paste0("my_mean_", suffix, "_jnp")
      if (!exists(mean_name, inherits = TRUE)) return(NULL)
      coef_vec <- get(mean_name, inherits = TRUE)
      main_info_name <- paste0("main_info_", suffix, "_jnp")
      inter_info_name <- paste0("interaction_info_", suffix, "_jnp")
      vcov_name <- paste0("vcov_OutcomeModel_", suffix, "_jnp")
      intercept_name <- paste0("EST_INTERCEPT_tf_", suffix, "_jnp")
      metrics_name <- paste0("fit_metrics_", suffix, "_jnp")

      main_info <- if (exists(main_info_name, inherits = TRUE)) {
        get(main_info_name, inherits = TRUE)
      } else {
        NULL
      }
      interaction_info <- if (exists(inter_info_name, inherits = TRUE)) {
        get(inter_info_name, inherits = TRUE)
      } else {
        NULL
      }
      vcov <- if (exists(vcov_name, inherits = TRUE)) {
        get(vcov_name, inherits = TRUE)
      } else {
        NULL
      }
      intercept <- if (exists(intercept_name, inherits = TRUE)) {
        to_scalar(get(intercept_name, inherits = TRUE))
      } else {
        NA_real_
      }
      fit_metrics <- if (exists(metrics_name, inherits = TRUE)) {
        get(metrics_name, inherits = TRUE)
      } else {
        NULL
      }
      baseline <- if (is.na(intercept)) NA_real_ else linkinv(intercept)
      n_main <- if (!is.null(main_info)) nrow(main_info) else 0L

      main_effects <- build_main_effects(main_info, coef_vec, intercept, vcov)
      top_interactions <- build_interactions(interaction_info, coef_vec, n_main)
      note <- NULL
      if (outcome_model_type != "glm" && (is.null(main_effects) || nrow(main_effects) == 0)) {
        note <- "Outcome model is non-linear; main-effect table not available."
      }

      list(
        stage = stage,
        player = player,
        group = group_value,
        outcome_model_type = outcome_model_type,
        glm_family = glm_family,
        link = ifelse(glm_family == "binomial", "logit", "identity"),
        intercept = intercept,
        baseline = baseline,
        fit_metrics = fit_metrics,
        main_effects = main_effects,
        top_interactions = top_interactions,
        note = note
      )
    }

    models <- list()
    if (isTRUE(adversarial)) {
      models[["ast_primary"]] <- build_model_view("ast0", "primary", "AST", GroupsPool[1])
      models[["ast_general"]] <- build_model_view("ast", "general", "AST", GroupsPool[1])
      models[["dag_primary"]] <- build_model_view("dag0", "primary", "DAG", GroupsPool[2])
      models[["dag_general"]] <- build_model_view("dag", "general", "DAG", GroupsPool[2])
    } else {
      models[["overall"]] <- build_model_view("ast", "single", "overall", NA)
    }
    models <- models[!vapply(models, is.null, logical(1))]
    list(
      metadata = list(
        outcome_model_type = outcome_model_type,
        glm_family = glm_family,
        holdout_level = ifelse(holdout_indicator == 1, "last_level", "none"),
        adversarial_model_strategy = adversarial_model_strategy
      ),
      models = models
    )
  })()

  get_neural_info <- function(name) {
    if (exists(name, inherits = TRUE)) {
      return(get(name, inherits = TRUE))
    }
    NULL
  }
  neural_model_info_out <- list(
    ast = get_neural_info("neural_model_info_ast_jnp"),
    ast0 = get_neural_info("neural_model_info_ast0_jnp"),
    dag = get_neural_info("neural_model_info_dag_jnp"),
    dag0 = get_neural_info("neural_model_info_dag0_jnp")
  )
  
  message("strategize() call has finished...\n-------------")
  return( 
          list(   "pi_star_point" = pi_star_list,
                  "pi_star_se" = pi_star_se_list,
                  
                  "Q_point" = q_star,
                  "Q_se"= q_star_se,
                  "Q_point_mEst" = q_star,
                  "Q_se_mEst"= q_star_se,
                  
                  "pi_star_vec" = pi_star_numeric,
                  "pi_star_red_ast" = pi_star_red_ast,
                  "pi_star_red_dag" = pi_star_red_dag,
                  "factor_levels" = factor_levels,
                  "pi_star_se_vec" = pi_star_se,
                  "pi_star_ave" = pi_star_ave,
                  "q_ave" = q_ave,
                  "q_dag_ave" = q_dag_ave,
                  "pi_star_lb" = lowerList,
                  "pi_star_ub" = upperList,
                  "Q_point" = c(q_star),
                  "penalty_type" = penalty_type,
                  "lambda" = lambda,
                  "p_vec_full" = p_vec_full,
                  "regularization_adjust_hash" = regularization_adjust_hash,
                  "p_list" = p_list,
                  "slate_list" = slate_list, 
                  "ParameterizationType" = strenv$ParameterizationType, 

                  # reconstruct q info
                  "QFXN" = QFXN,

                  'p_vec_full_ast_jnp' = p_vec_full_ast_jnp,
                  'p_vec_full_dag_jnp' = p_vec_full_dag_jnp,
                  'pi_star_ast_vec_jnp' = pi_star_vec_jnp,
                  'pi_star_dag_vec_jnp' = pi_star_dag_vec_jnp,
                  "est_intercept_jnp" = strenv$jnp$array(EST_INTERCEPT_tf),
                  "est_coefficients_jnp" = strenv$jnp$array(EST_COEFFICIENTS_tf),

                  "vcov_outcome_model" = vcov_OutcomeModel,
                  "vcov_outcome_model_concat" = vcov_OutcomeModel_concat, 
                  "jacobian_mat" = jacobian_mat, 
                  "optim_type" = optim_type,
                  "optimism" = optimism,
                  "optimism_coef" = optimism_coef,
                  "force_gaussian" = force_gaussian,
                  "used_regularization" = UsedRegularization,
                  "estimation_type" = "TwoStep",
                  "gather_fxn" = gather_fxn, 
                  "a_i_ast" = a_i_ast_optimized,
                  "a_i_dag" = a_i_dag_optimized,
                  
                  "ParameterizationType" = strenv$ParameterizationType,
                  "d_locator" = d_locator,  
                  "d_locator_use" = strenv$d_locator_use, 
                  "main_comp_mat" = main_comp_mat,
                  "shadow_comp_mat" = shadow_comp_mat, 
                  "getQPiStar_gd" = getQPiStar_gd, 
                  "FullGetQStar_" = FullGetQStar_, 
                  "dQ_da_dag" = dQ_da_dag, 
                  "dQ_da_ast" = dQ_da_ast, 
                  "REGRESSION_PARAMETERS_ast" = REGRESSION_PARAMS_jax_ast_jnp,
                  "REGRESSION_PARAMETERS_dag" = REGRESSION_PARAMS_jax_dag_jnp,
                  "REGRESSION_PARAMETERS_ast0"= REGRESSION_PARAMS_jax_ast0_jnp,
                  "REGRESSION_PARAMETERS_dag0"= REGRESSION_PARAMS_jax_dag0_jnp,
                  "P_VEC_FULL_ast" = p_vec_full_ast_jnp,
                  "P_VEC_FULL_dag" = p_vec_full_dag_jnp,
                  "SLATE_VEC_ast"  = SLATE_VEC_ast_jnp,
                  "SLATE_VEC_dag"  = SLATE_VEC_dag_jnp,
                  
                  "temperature" = temperature, 
                  "primary_strength" = primary_strength,
                  "primary_n_entrants" = primary_n_entrants,
                  "primary_n_field" = primary_n_field,
                  "adversarial_model_strategy" = adversarial_model_strategy,
                  "AstProp" = strenv$AstProp,   
                  "DagProp" = strenv$DagProp,   
                  "strenv" = strenv,
                  "Y_models" = list(
                    "my_model_ast_jnp"  = my_model_ast_jnp,
                    "my_model_ast0_jnp" = my_model_ast0_jnp,
                    "my_model_dag_jnp"  = my_model_dag_jnp,
                    "my_model_dag0_jnp" = my_model_dag0_jnp
                    ),
                  
                  "outcome_model_view" = outcome_model_view,
                  "neural_model_info" = neural_model_info_out,

                  # Hessian functions for equilibrium geometry analysis
                  "d2Q_da2_ast" = d2Q_da2_ast,
                  "d2Q_da2_dag" = d2Q_da2_dag,
                  "hessian_available" = hessian_available,
                  "hessian_skipped_reason" = hessian_skipped_reason,
                  "n_params_per_player" = n_params_per_player,

                  # Convergence history for diagnostics
                  "convergence_history" = tryCatch({
                    # Helper to safely convert JAX/Python objects to numeric
                    safe_to_numeric <- function(x) {
                      tryCatch({
                        strenv$np$array(x)
                      }, error = function(e) NA_real_)
                    }

                    list(
                      "grad_ast" = unlist(lapply(strenv$grad_mag_ast_vec, safe_to_numeric)),
                      "grad_dag" = unlist(lapply(strenv$grad_mag_dag_vec, safe_to_numeric)),
                      "loss_ast" = unlist(lapply(strenv$loss_ast_vec, safe_to_numeric)),
                      "loss_dag" = unlist(lapply(strenv$loss_dag_vec, safe_to_numeric)),
                      "inv_lr_ast" = unlist(lapply(strenv$inv_learning_rate_ast_vec, safe_to_numeric)),
                      "inv_lr_dag" = unlist(lapply(strenv$inv_learning_rate_dag_vec, safe_to_numeric)),
                      "nSGD" = nSGD,
                      "adversarial" = adversarial,
                      "primary_pushforward" = primary_pushforward,
                      "primary_strength" = primary_strength,
                      "primary_n_entrants" = primary_n_entrants,
                      "primary_n_field" = primary_n_field,
                      "adversarial_model_strategy" = adversarial_model_strategy,
                      "optimism" = optimism,
                      "optimism_coef" = optimism_coef,
                      "rain_gamma" = rain_gamma,
                      "rain_eta" = rain_eta,
                      "rain_lambda" = if (!is.null(strenv$rain_lambda_vec)) {
                        unlist(lapply(strenv$rain_lambda_vec, safe_to_numeric))
                      } else {
                        rep(NA_real_, nSGD)
                      }
                    )
                  }, error = function(e) {
                    # Fallback if conversion fails entirely
                    list(
                      "grad_ast" = rep(NA_real_, nSGD),
                      "grad_dag" = rep(NA_real_, nSGD),
                      "loss_ast" = rep(NA_real_, nSGD),
                      "loss_dag" = rep(NA_real_, nSGD),
                      "inv_lr_ast" = rep(NA_real_, nSGD),
                      "inv_lr_dag" = rep(NA_real_, nSGD),
                      "nSGD" = nSGD,
                      "adversarial" = adversarial,
                      "primary_pushforward" = primary_pushforward,
                      "primary_strength" = primary_strength,
                      "primary_n_entrants" = primary_n_entrants,
                      "primary_n_field" = primary_n_field,
                      "adversarial_model_strategy" = adversarial_model_strategy,
                      "optimism" = optimism,
                      "optimism_coef" = optimism_coef,
                      "rain_gamma" = rain_gamma,
                      "rain_eta" = rain_eta,
                      "rain_lambda" = rep(NA_real_, nSGD)
                    )
                  })
                  )  # end outout list 
          )
}
