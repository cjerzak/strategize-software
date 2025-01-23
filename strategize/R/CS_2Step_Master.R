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
#'   MaxMin = FALSE,
#'   use_regularization = FALSE,
#'   force_gaussian = FALSE,
#'   a_init_sd = 0,
#'   penalty_type = "KL",
#'   compute_se = TRUE,
#'   conda_env = NULL,
#'   conda_env_required = FALSE,
#'   conf_level = 0.90,
#'   nFolds_glm = 3L,
#'   folds = NULL,
#'   nMonte_MaxMin = 5L,
#'   nMonte_Qglm = 100L,
#'   use_optax = FALSE,
#'   optim_type = "gd"
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
#' @param competing_group_variable_respondent,competing_group_variable_candidate,
#'   competing_group_competition_variable_candidate Optional variables that mark competition group
#'   membership of respondents or candidate profiles. Particularly relevant in adversarial settings
#'   (\code{MaxMin = TRUE}) or multi-stage electoral settings, e.g., capturing the party of each
#'   respondent or candidate. Defaults to \code{NULL}.
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
#' @param slate_list An optional list (or lists) providing custom “slates” of candidate features
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
#' @param MaxMin Logical controlling whether to enable the max-min adversarial scenario. When
#'   \code{TRUE}, the function searches for a pair of distributions (one for each competing party
#'   or group) such that each party’s distribution is optimal given the other party’s distribution.
#'   Defaults to \code{FALSE}.
#'
#' @param use_regularization Logical indicating whether to regularize the outcome model (in addition
#'   to any penalty \code{lambda} on the distribution shift). This can help avoid overfitting in
#'   high-dimensional designs. Defaults to \code{FALSE}.
#'
#' @param force_gaussian Logical indicating whether to force a Gaussian-based outcome modeling
#'   approach, even if \code{Y} is binary or forced-choice. If \code{FALSE}, the function attempts
#'   to choose a more appropriate link (e.g., \code{"binomial"}). Defaults to \code{FALSE}.
#'
#' @param a_init_sd Numeric scalar specifying the standard deviation for random initialization
#'   of unconstrained parameters used in the gradient-based search over factor-level probabilities.
#'   Defaults to \code{0}.
#'
#' @param penalty_type A character string specifying the type of penalty (e.g., \code{"KL"}, \code{"L2"},
#'   or \code{"LogMaxProb"}) used in the objective function for shifting the factor-level probabilities
#'   away from the baseline \code{p_list}. Defaults to \code{"KL"}.
#'
#' @param compute_se Logical indicating whether standard errors should be computed for the final
#'   estimates (via the delta method or related expansions). Defaults to \code{TRUE}.
#'
#' @param conda_env A character string naming a Python conda environment that includes \pkg{jax},
#'   \pkg{optax}, and other dependencies. If not \code{NULL}, the function attempts to activate
#'   that environment. Defaults to \code{NULL}.
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
#' @param nMonte_MaxMin Integer specifying the number of Monte Carlo samples used in adversarial
#'   or max-min steps, e.g., sampling from the opposing candidate’s distribution to approximate
#'   expected payoffs. Defaults to \code{5L}.
#'
#' @param nMonte_Qglm Integer specifying the number of Monte Carlo samples for evaluating or
#'   approximating the quantity of interest under certain outcomes or distributions. Defaults to
#'   \code{100L}.
#'
#' @param use_optax Logical indicating whether to use the \href{https://github.com/deepmind/optax}{\code{optax}}
#'   library for gradient-based optimization in JAX (\code{TRUE}) or a built-in method (\code{FALSE}).
#'   Defaults to \code{FALSE}.
#'
#' @param jax_seed Integer seed for reproducible JAX-based computations. Defaults to
#'   \code{as.integer(Sys.time())}.
#'
#' @param optim_type A character string for choosing which optimizer or approach is used internally
#'   (e.g., \code{"default"}, \code{"SecondOrder"}, or \code{"tryboth"}). Defaults to \code{"tryboth"}.
#'
#' @return A named \code{list} containing:
#' \describe{
#' \item{\code{pi_star_point}}{An estimate of the (possibly multi-cluster or adversarial)
#' optimal distribution(s) over the factor levels.
#'
#' Structure depends on parameters:
#' - If \code{MaxMin = TRUE} and \code{K = 1}, returns a pair of distributions (e.g., maximin solutions).
#' - If \code{K > 1}, returns a list where each element corresponds to a cluster-optimal distribution.
#' - Otherwise, returns a single distribution.}
#'
#' \item{\code{pi_star_se}}{Standard errors for entries in \code{pi_star_point}. Mirrors the structure of \code{pi_star_point} (e.g., a pair of SEs if \code{MaxMin = TRUE} and \code{K = 1}). Only present if \code{compute_se = TRUE}.}
#'
#' \item{\code{Q_point_mEst}}{Point estimate(s) of the optimized outcome (e.g., utility/vote share). Matches the structure of \code{pi_star_point}.}
#'
#' \item{\code{Q_se_mEst}}{Standard errors for \code{Q_point_mEst}. Only present if \code{compute_se = TRUE}.}
#'
#' \item{\code{pi_star_lb}, \code{pi_star_ub}}{Confidence bounds for \code{pi_star_point} (if \code{compute_se = TRUE} and a confidence level is provided).}
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
#' \strong{Adversarial or strategic design:} When \code{MaxMin = TRUE}, the function attempts to
#' solve a zero-sum game in which one agent (say, \dQuote{A}) chooses its distribution to maximize
#' vote share, while the other (\dQuote{B}) simultaneously chooses its distribution to minimize
#' \dQuote{A}’s vote share. In many settings, \code{competing_group_variable_respondent} and related
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
#' @references
#' - Hainmueller, J., Hopkins, D. J., & Yamamoto, T. (2014). Causal Inference in Conjoint Analysis:
#'   Understanding Multidimensional Choices via Stated Preference Experiments. \emph{Political
#'   Analysis}, 22(1), 1–30.
#' - Egami, N., & Imai, K. (2019). Causal Interaction in Factorial Experiments:
#'   Application to Conjoint Analysis. \emph{Journal of the American Statistical Association}, 114(526), 529–540.
#' - Goplerud, M., & Titiunik, R. (2022). Analysis of High-Dimensional Factorial Experiments:
#'   Estimation of Interactive and Non-Interactive Effects. \emph{arXiv preprint arXiv:2207.XXXX}.
#' - (Paper Reference) A forthcoming or accompanying manuscript describing in detail the methods
#'   for \emph{optimal} or \emph{adversarial} stochastic interventions in conjoint settings.
#'
#' @seealso
#' \code{\link{cv_strategize}} for cross-validation across candidate values of \code{lambda}.
#' See also \code{\link{strategize_onestep}} for a function that implements a \dQuote{one-step}
#' approach to M-estimation of the same target quantity. 
#'
#' @examples
#' \dontrun{
#'   # Suppose we have a forced-choice conjoint dataset with
#'   # factor matrix W, outcome Y, and baseline probabilities p_list
#'
#'   # Basic usage: single agent optimizing expected outcome
#'   opt_result <- strategize(
#'       Y = Y,
#'       W = W,
#'       lambda = 0.1,
#'       p_list = p_list,
#'       MaxMin = FALSE,         # No adversarial component
#'       penalty_type = "KL",         # Kullback-Leibler penalty
#'       nSGD = 200              # # of gradient descent iterations
#'   )
#'
#'   # Inspect the learned distribution and performance
#'   print(opt_result$pi_star_point)
#'   print(opt_result$Q_point_mEst)
#'   print(opt_result$CVInfo)            # If cross-validation was used
#'
#'   # Adversarial scenario with multi-stage structure
#'   # E.g., define 'competing_group_variable_respondent' for two parties' supporters
#'   adv_result <- strategize(
#'       Y = Y,
#'       W = W,
#'       lambda = 0.2,
#'       p_list = p_list,
#'       MaxMin = TRUE,         # Solve zero-sum game across two sets of respondents
#'       competing_group_variable_respondent = partyID,
#'       nSGD = 300
#'   )
#'
#'   # 'adv_result' now contains distributions for each party's candidate
#'   # that approximate a mixed-strategy Nash equilibrium
#'   print(adv_result$pi_star_point$k1)   # Party A distribution
#'   print(adv_result$pi_star_point$k2)   # Party B distribution
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
                                            diff = F, MaxMin = F,
                                            use_regularization = F,
                                            force_gaussian = F,
                                            a_init_sd = 0.001,
                                            penalty_type = "KL",
                                            compute_se = T,
                                            conda_env = NULL,
                                            conda_env_required = F,
                                            conf_level = 0.90,
                                            nFolds_glm = 3L,
                                            folds = NULL, 
                                            nMonte_MaxMin = 5L,
                                            nMonte_Qglm = 100L,
                                            use_optax = F, 
                                            optim_type = "gd"){
  # [1.] ast then dag 
  #   ast is 1, based on sort(unique(competing_group_variable_candidate))[1]
  #   dag is 2, based on sort(unique(competing_group_variable_candidate))[2]
  # [2.] when simplex constrained with holdout, LAST entry is held out 
  
  message("-------------")
  message("strategize() call has begun...")
  
  # define evaluation environment 
  evaluation_environment <- environment()
  
  # add colnames if not available 
  if(!is.null(X)){ if(is.null(colnames(X))){ colnames(X) <- paste0("V",1:ncol(X)) } }

  # load in packages
  if(!"jnp" %in% ls(envir = strenv)) {
    message("Initializing computational environment...")  
    initialize_jax(conda_env = conda_env, conda_env_required = conda_env_required) 
  }
  
  # define compile fxn
  compile_fxn <- function(x){strenv$jax$jit(x)}
  #compile_fxn <- function(x){x} ; message("TURNING COMPILE OFF FOR DEBUGGIN!"); Sys.sleep(5L)

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

  nMonte_MaxMin <- ai( nMonte_MaxMin )
  q_ave <- q_dag_ave <- pi_star_ave <- NULL
  w_orig <- W
  MaxMinType <- "TwoRoundSingle"

  glm_family = "gaussian"; glm_outcome_transform <- function(x){x} # identity function
  if(!force_gaussian){ 
    if(mean(unique(Y) %in% c(0,1)) == 1){ 
      glm_family = "binomial"; glm_outcome_transform <- strenv$jax$nn$sigmoid
    } }
  MNtemp <- strenv$jnp$array( .5 ) 

  # ensure naming conventions are correct (i.e. in alignment with p_list if available)
  if(is.null(p_list) | is.null(names(p_list[[1]]))){
    names_list <- apply(w_orig,2,function(zer){ list(sort(names(table(as.factor(zer))),decreasing=F)) })
  }
  if(!is.null(p_list) & !is.null(names(p_list[[1]]))){
    names_list <- lapply(p_list,function(zer){ list(names(zer)) })
  }
  W <- sapply(1:ncol(W),function(zer){ match(W[,zer],names_list[[zer]][[1]]) })

  # get info about # of levels per factor
  factor_levels_full <- factor_levels <- apply(W,2,function(zer){length(unique(zer))})

  # obtain outcome models
  message("Initializing outcome models...")
  if(T == T){
    if(K > 1 & !use_regularization){ warning("K > 1; Forcing regularization...");use_regularization <- T }
    use_regularization_ORIG <- use_regularization

    RoundsPool <- nRounds <- GroupsPool <- 1
    if(MaxMin){
      nRounds <- length( RoundsPool <- c(0,1) ) 
      GroupsPool <- sort(unique(competing_group_variable_candidate))
    }

    for(Round_ in RoundsPool){
    for(GroupCounter in 1:length(GroupsPool)){
      print(c(Round_, GroupCounter))
      use_regularization <- use_regularization_ORIG
      if(MaxMin == F){ indi_ <- 1:length( Y )  }
      if(MaxMin == T){
        if(Round_ == 0){
          indi_ <- which( competing_group_variable_respondent == GroupsPool[GroupCounter] &
                      ( competing_group_competition_variable_candidate == "Same" &
                          competing_group_variable_candidate == GroupsPool[GroupCounter] ) )
        }
        if(Round_ == 1){
          indi_ <- which( competing_group_variable_respondent == GroupsPool[GroupCounter] &
                            ( competing_group_competition_variable_candidate == "Different" &
                                competing_group_variable_candidate %in% GroupsPool) )
        }
        AstProp <- prop.table(table(competing_group_variable_respondent[
                      competing_group_variable_respondent %in% GroupsPool]))[1]
        DagProp <- prop.table(table(competing_group_variable_respondent[
                          competing_group_variable_respondent %in% GroupsPool]))[2]
      }

      # subset data
      W_ <- W[indi_,]; Y_ <- Y[indi_];
      varcov_cluster_variable_ <- varcov_cluster_variable[indi_]
      pair_id_ <- pair_id[ indi_ ]

      # run models with inputs: W_; Y_; varcov_cluster_variable_;
      initialize_ModelOutcome <- paste(deparse(generate_ModelOutcome),collapse="\n")
      initialize_ModelOutcome <- gsub(initialize_ModelOutcome,pattern="function \\(\\)",replace="")
      eval( parse( text = initialize_ModelOutcome ), envir = evaluation_environment )
      
      # define combined parameter vector & fxn for reextracting intercept & coefficient 
      REGRESSION_PARAMS_jax <- strenv$jnp$concatenate(list(EST_INTERCEPT_tf, EST_COEFFICIENTS_tf), 0L)
      gather_fxn <- compile_fxn(function(x){
        return( list(strenv$jnp$expand_dims(strenv$jnp$take(x,0L),0L),  # INTERCEPT_ 
                     strenv$jnp$take(x, strenv$jnp$array( 1L:(strenv$jnp$shape(x)[[1]]-1L) ) ))) } ) # COEFFICIENTS 

      # rename as appropriate
      ret_chunks <- c("vcov_OutcomeModel", "main_info","interaction_info","interaction_info_PreRegularization",
            "REGRESSION_PARAMS_jax","regularization_adjust_hash","main_dat", "my_mean","EST_INTERCEPT_tf","my_model", "EST_COEFFICIENTS_tf")
      round_text <- ifelse( Round_==0, yes="0", no="")
      if( (doAst <- (GroupCounter == 1) | (MaxMin == F)) ){
          tmp <- sapply(ret_chunks,function(chunk_){ eval(parse(text = sprintf("%s_ast%s_jnp <- %s",chunk_,round_text,chunk_)),envir = evaluation_environment) })
          rm(tmp)
      }
      if( !doAst ){
          tmp <- sapply(ret_chunks,function(chunk_){ eval(parse(text = sprintf("%s_dag%s_jnp <- %s",chunk_,round_text,chunk_)),envir = evaluation_environment) })
          rm( tmp )
       }
    }
    }
  }

  for(suffix in c("ast0", "dag0", "dag")) {
    if(!paste0("REGRESSION_PARAMS_jax_", suffix, "_jnp") %in% ls()){
      assign(paste0("EST_INTERCEPT_tf_", suffix, "_jnp"), EST_INTERCEPT_tf_ast_jnp)
      assign(paste0("EST_COEFFICIENTS_tf_", suffix, "_jnp"), EST_COEFFICIENTS_tf_ast_jnp)
      assign(paste0("REGRESSION_PARAMS_jax_", suffix, "_jnp"), REGRESSION_PARAMS_jax_ast_jnp)
      assign(paste0("vcov_OutcomeModel_", suffix, "_jnp"), vcov_OutcomeModel_ast_jnp)
  } }
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
    if(any(names(p_list) != colnames(W))){stop("p_list and W not aligned")}
    p_list_full <- p_list
    p_vec_full_PreRegularization <- p_list_full
    p_list_PreRegularization <- p_list
    p_vec <- unlist(lapply(p_list,function(zer){zer[-length(zer)]}))
    p_vec_full <- unlist(lapply(p_list,function(zer){zer}))
  }

  ParameterizationType <- ifelse( holdout_indicator == 0, yes = "Full", no = "Implicit")
  n2int <- function(x){  strenv$jnp$array(x,strenv$jnp$int32)  }
  d_locator <- unlist(sapply(1:length(factor_levels),function(l_d){rep(l_d,times=factor_levels[l_d]-1)}))
  
  # p logic 
  p_vec_sum_prime <- unlist(tapply(1:length(p_vec),d_locator,function(er){
    sapply(er,function(re){sum(p_vec[er[!er %in% re]])}) }))
  d_locator_full <- unlist(sapply(1:length(factor_levels),function(l_d){rep(l_d,times=factor_levels[l_d])}))
  p_vec_sum_prime_full <- unlist(tapply(1:length(p_vec_full),d_locator_full,function(er){
    sapply(er,function(re){sum(p_vec_full[er[!er %in% re]])}) }))
  main_indices_i0 <- strenv$jnp$array((ai(1:n_main_params-1L)),strenv$jnp$int32)
  inter_indices_i0 <- strenv$jnp$array((ai(((n_main_params+1):EST_COEFFICIENTS_tf$size)-1L)),strenv$jnp$int32)
  if(ParameterizationType == "Implicit"){ p_vec_use <- p_vec; p_vec_sum_prime_use <- p_vec_sum_prime }
  if(ParameterizationType == "Full"){ p_vec_use <- p_vec_full; p_vec_sum_prime_use <- p_vec_sum_prime_full }

  if(optim_type != "gd"){
    message("Initializing manual exact solution code...")
    initialize_ExactSol <- paste(deparse(generate_ExactSol), collapse="\n")
    initialize_ExactSol <- gsub(initialize_ExactSol,pattern="function \\(\\)", replace="")
    eval( parse( text = initialize_ExactSol ), envir = evaluation_environment )
    if(ParameterizationType == "Implicit"){ getPiStar_exact <- generate_ExactSolImplicit }
    if(ParameterizationType == "Full"){ getPiStar_exact <- generate_ExactSolExplicit }
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
  a_vec_init_dag <- strenv$jnp$array(a_vec_init_mat+rnorm(length(a_vec_init_mat),sd = a_init_sd*MaxMin), strenv$dtj)
  
  LabelSmoothingFxn <- (function(x, epsilon = 0.01){
      return( (1 - epsilon) * x + epsilon / strenv$jnp$array( x$shape[[1]] )$astype(x$dtype) ) })
  a2Simplex <- compile_fxn(function(a_){
    exp_a_ <- strenv$jnp$exp(a_)
    aOnSimplex <- tapply(1:nrow(a_structure),a_structure$d,function(zer){
      OnSimplex_ <- strenv$jnp$divide(  strenv$jnp$take(exp_a_, n2int(as.matrix(zer-1L) )),
                        strenv$jnp$add(OneTf_flat,strenv$jnp$sum(strenv$jnp$take(exp_a_, n2int(zer-1L) ) )))
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
  OneTf_flat <- strenv$jnp$squeeze(OneTf <- strenv$jnp$array(matrix(1L), strenv$dtj)$flatten(), 0L)
  Neg2_tf <- strenv$jnp$array(-2., strenv$dtj)

  message("Defining Q functions..")
  a2Simplex_optim <- ifelse( holdout_indicator == 1 ,
                             yes = list(a2Simplex),
                             no = list(a2FullSimplex) )[[1]]
  pi_star_value_init_ast <- a2Simplex_optim( a_vec_init_ast ) # a_ = a_vec_init_ast
  pi_star_value_init_dag <- a2Simplex_optim( a_vec_init_dag )

  # define Q functions
  environment(getQStar_single) <- evaluation_environment
  getQStar_single <- compile_fxn( getQStar_single )

  # multiround material
  for(DisaggreateQ in ifelse(MaxMin, yes = list(c(F,T)), no = list(F))[[1]]){
    # general specifications
    getQStar_diff_ <- paste(deparse(getQStar_diff_BASE),collapse="\n")
    getQStar_diff_ <- gsub(getQStar_diff_, pattern = "Q_DISAGGREGATE", replace = sprintf("T == %s", DisaggreateQ))
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
    if(ParameterizationType == "Implicit"){
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

      shadow_comp_mat <- matrix(0, ncol = n_main_params, nrow = length_simplex_use)
      shadow_comp_mat <- strenv$jnp$array( sapply(1:length(pi_star_value_loc_shadow),function(zer){
        shadow_comp_mat[pi_star_value_loc_shadow[zer],zer] <- 1
        return( shadow_comp_mat[,zer] ) }),strenv$dtj)
    }

    split_vec_full <- unlist(sapply(1:length(factor_levels),function(xz){
                           rep(xz,times=factor_levels[xz])} ))
    split_vec_use <- ifelse(ParameterizationType == "Implicit",
                            yes = list(split_vec), no = list(split_vec_full))[[1]]
  }

  environment(getPrettyPi) <- evaluation_environment
  getPrettyPi <- compile_fxn( getPrettyPi )

  getPrettyPi_diff <- ifelse(ParameterizationType=="Implicit",
                                  yes = list(getPrettyPi),
                                  no = list(compile_fxn(function(x){x})))[[1]]
  a2Simplex_diff_use <- ifelse(ParameterizationType == "Implicit",
                               yes = list(a2Simplex),
                               no = list(a2FullSimplex))[[1]]

  ## get exact result
  pi_star_exact <- -10; if(optim_type %in% c("tryboth") & diff == F){
    pi_star_exact <- strenv$np$array(getPrettyPi(   getPiStar_exact( EST_COEFFICIENTS_tf )  )) # pi_star_value =
  }

  use_exact <- !( use_gd <- any(pi_star_exact<0) | any(pi_star_exact>1)  |
    (abs(sum(pi_star_exact) - sum(unlist(p_list_full))) > 1e-5) )
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
      LEARNING_RATE_MAX <- 0.01
      LR_schedule <- strenv$optax$warmup_cosine_decay_schedule(warmup_steps =  min(c(20L,nSGD)),decay_steps = max(c(21L,nSGD-100L)),
                                                        init_value = LEARNING_RATE_MAX/100, peak_value = LEARNING_RATE_MAX, end_value =  LEARNING_RATE_MAX/100)
    
      # model partition + setup state
      optax_optimizer <-  strenv$optax$chain(
         strenv$optax$scale(1),strenv$optax$scale_by_rss(initial_accumulator_value = 0.001)  
        #strenv$optax$scale(-1), strenv$optax$adabelief(LR_schedule)
        )
      
      for(sfx in c("ast", "dag")){
        assign(paste0("opt_state_", sfx), optax_optimizer$init(strenv$jnp$array(get(paste0("a_vec_init_", sfx)))))
        assign(paste0("jit_apply_updates_", sfx), compile_fxn(strenv$optax$apply_updates))
        assign(paste0("jit_update_", sfx), compile_fxn(optax_optimizer$update))
      }
  }

  message("Defining gd function...")
  # bring functions into env and compile as needed
  environment(getMultinomialSamp) <- evaluation_environment; getMultinomialSamp <- compile_fxn( getMultinomialSamp )
  environment(getPiStar_gd) <- evaluation_environment
  }

  # get jax seed into correct type
  jax_seed <- strenv$jnp$array( ai(runif(1,1,1000)) )

  # Obtain solution via exact calculation
  message("Starting optimization...")
  pi_star_se_list_OUTER <- pi_star_list_OUTER <- replicate(n = K, list())
  q_star_OUTER <- q_star_se_OUTER <- replicate(n = K, list())
  for(k_clust in 1:K){
  if(K > 1){
    message(sprintf("Optimizing cluster %s of %s...",k_clust, K))
    ################################################
    # WARNING: Operational only in average case mode
    EST_INTERCEPT_tf <- strenv$jnp$array(t( my_mean_full[1,k_clust] ) )
    EST_COEFFICIENTS_tf <- strenv$jnp$array(as.matrix( my_mean_full[-1,k_clust] ) )
    REGRESSION_PARAMS_jax <- strenv$jnp$array(strenv$jnp$concatenate(list(EST_INTERCEPT_tf, EST_COEFFICIENTS_tf),0L))
    REGRESSION_PARAMS_jax_ast_jnp <- REGRESSION_PARAMS_jax
    REGRESSION_PARAMS_jax_dag_jnp <- REGRESSION_PARAMS_jax_ast0_jnp <- REGRESSION_PARAMS_jax_ast_jnp

    # reset covariates
    vcov_OutcomeModel_ast_jnp <- vcov_OutcomeModel_dag_jnp <- vcov_OutcomeModel_ast0_jnp <- vcov_OutcomeModel_dag0_jnp <- vcov_OutcomeModel_jnp <- vcov_OutcomeModel_by_k[[ k_clust ]]
  }

  if(use_exact){
    message("Optimization type: Exact")
    FxnForJacobian <- function(  INPUT_  ){
      EST_INTERCEPT_tf_ <- INPUT_[[1]]
      EST_COEFFICIENTS_tf_  <- INPUT_[[2]]
      pi_star_full_exact <- pi_star_full <- getPrettyPi( pi_star_reduced <- getPiStar_exact(EST_COEFFICIENTS_tf_))
      q_star_exact <- q_star <- getQStar_single(pi_star_ast = pi_star_reduced,
                                                pi_star_dag = pi_star_reduced,
                                                EST_INTERCEPT_tf_ast = EST_INTERCEPT_tf_,
                                                EST_COEFFICIENTS_tf_ast = EST_COEFFICIENTS_tf_,
                                                EST_INTERCEPT_tf_dag = EST_INTERCEPT_tf_,
                                                EST_COEFFICIENTS_tf_dag = EST_COEFFICIENTS_tf_)
      results_vec <- strenv$jnp$concatenate(list(q_star, pi_star_full),0L)
      return( results_vec )
    }
    results_vec <- FxnForJacobian( list(EST_INTERCEPT_tf,EST_COEFFICIENTS_tf) )
    jacobian_mat <- strenv$jax$jacobian(FxnForJacobian, 0L)(  list(EST_INTERCEPT_tf,EST_COEFFICIENTS_tf) ) 

    # reshape jacobian and process results
    jacobian_mat_exact <- jacobian_mat <- cbind(
                          strenv$np$array(strenv$jnp$squeeze(strenv$jnp$squeeze(strenv$jnp$squeeze(jacobian_mat[[1]],1L),1L))),
                          strenv$np$array(strenv$jnp$squeeze(strenv$jnp$squeeze(strenv$jnp$squeeze(jacobian_mat[[2]],1L),2L))) )
    vcov_OutcomeModel_concat <- vcov_OutcomeModel_ast_jnp
    q_star_exact <- q_star <- strenv$np$array( strenv$jnp$take(results_vec, 0L) )
    pi_star_full <- strenv$np$array( strenv$jnp$take(results_vec, strenv$jnp$array((1L:length(results_vec))[-c(1:3)] -1L)))
  }

  # define main Q function in different cases
  if(!MaxMin & !diff){ QFXN <- getQStar_single }
  if(!MaxMin & diff){ QFXN <- getQStar_diff_SingleGroup }
  if(MaxMin & diff){ QFXN <- getQStar_diff_MultiGroup }

  if(use_gd){
    message("Optimization type: Gradient ascent")

    # perform main gd runs + inference
    # first do ave case analysis
    p_vec_full_ast_jnp <- p_vec_full_dag_jnp <- p_vec_full_jnp

    # initialize QMonte fxns 
    InitializeQMonteFxns_ <- paste(deparse(InitializeQMonteFxns),collapse="\n")
    InitializeQMonteFxns_ <- gsub(InitializeQMonteFxns_, pattern = "function \\(\\)", replace = "")
    InitializeQMonteFxns_ <- eval( parse( text = InitializeQMonteFxns_ ), envir = evaluation_environment )

    # setup gd functions dparams
    environment(FullGetQStar_) <- evaluation_environment; FullGetQStar_ <- compile_fxn(FullGetQStar_)
    dQ_da_ast <- compile_fxn(strenv$jax$grad(FullGetQStar_, argnums = 0L))
    dQ_da_dag <- compile_fxn(strenv$jax$grad(FullGetQStar_, argnums = 1L))
    
    # perform GD 
    # plot(strenv$np$array(REGRESSION_PARAMS_jax_ast_jnp), strenv$np$array(REGRESSION_PARAMS_jax_dag_jnp))
    q_with_pi_star_full <- getPiStar_gd(
                             REGRESSION_PARAMETERS_ast = REGRESSION_PARAMS_jax_ast_jnp,
                             REGRESSION_PARAMETERS_dag = REGRESSION_PARAMS_jax_dag_jnp,
                             REGRESSION_PARAMETERS_ast0 = REGRESSION_PARAMS_jax_ast0_jnp,
                             REGRESSION_PARAMETERS_dag0 = REGRESSION_PARAMS_jax_dag0_jnp,
                             P_VEC_FULL_ast = p_vec_full_ast_jnp,
                             P_VEC_FULL_dag = p_vec_full_dag_jnp,
                             SLATE_VEC_ast = SLATE_VEC_ast_jnp, 
                             SLATE_VEC_dag = SLATE_VEC_dag_jnp,
                             LAMBDA = strenv$jnp$array(  lambda  ),
                             BASE_SEED = jax_seed,
                             functionList = list(dQ_da_ast, dQ_da_dag,
                                                 QFXN, getMultinomialSamp),
                             a_i_ast = a_vec_init_ast, 
                             a_i_dag = a_vec_init_dag, 
                             functionReturn = T, 
                             gd_full_simplex = T, 
                             quiet = F)
    dQ_da_ast <- q_with_pi_star_full[[2]][[1]]
    dQ_da_dag <- q_with_pi_star_full[[2]][[2]]
    QFXN <- q_with_pi_star_full[[2]][[3]]
    getMultinomialSamp <- q_with_pi_star_full[[2]][[4]]
    q_with_pi_star_full <- strenv$jnp$array(q_with_pi_star_full[[1]], strenv$dtj)

    if(!use_optax){
      inv_learning_rate_ast_vec <- unlist(  lapply(inv_learning_rate_ast_vec,
                                                   function(zer){ strenv$np$array(zer) }))
      #try(plot( 1/inv_learning_rate_ast_vec , main = "Inv LR (ast)",log="y"),T)
    }

    grad_mag_dag_vec <- try(unlist(  lapply(grad_mag_dag_vec,function(zer){
      strenv$np$array(strenv$jnp$sqrt( strenv$jnp$sum(strenv$jnp$square(strenv$jnp$array(zer,strenv$dtj))) )) }) ),T)
    try(plot( grad_mag_dag_vec , main = "Gradient Magnitude Evolution (dag)",log="y"),T)
    try(points(lowess(grad_mag_dag_vec), cex = 2, type = "l",lwd = 2, col = "red"), T)
    
    grad_mag_ast_vec <- unlist(  lapply(grad_mag_ast_vec,function(zer){
      strenv$np$array(strenv$jnp$sqrt( strenv$jnp$sum(strenv$jnp$square(strenv$jnp$array(zer,strenv$dtj))) ))  }) )
    try(plot( grad_mag_ast_vec, main = "Gradient Magnitude Evolution (ast)", log ="y"),T)
    try(points(lowess(grad_mag_ast_vec), cex = 2, type = "l",lwd = 2, col = "red"), T)
    
    pi_star_red <- getPiStar_gd(
                        REGRESSION_PARAMETERS_ast = REGRESSION_PARAMS_jax_ast_jnp,
                        REGRESSION_PARAMETERS_dag = REGRESSION_PARAMS_jax_dag_jnp,
                        REGRESSION_PARAMETERS_ast0 = REGRESSION_PARAMS_jax_ast0_jnp,
                        REGRESSION_PARAMETERS_dag0 = REGRESSION_PARAMS_jax_dag0_jnp,
                        P_VEC_FULL_ast = p_vec_full_ast_jnp,
                        P_VEC_FULL_dag = p_vec_full_dag_jnp,
                        SLATE_VEC_ast = SLATE_VEC_ast_jnp, 
                        SLATE_VEC_dag = SLATE_VEC_dag_jnp,
                        LAMBDA = strenv$jnp$array(  lambda  ),
                        BASE_SEED = jax_seed,
                        functionList = list(dQ_da_ast, dQ_da_dag,
                                            QFXN, getMultinomialSamp),
                        a_i_ast = a_vec_init_ast, 
                        a_i_dag = a_vec_init_dag, 
                        functionReturn = F,
                        gd_full_simplex = F, 
                        quiet = F)
    pi_star_red <- strenv$np$array(pi_star_red)[-c(1:3),]
    pi_star_red_dag <- strenv$jnp$array(as.matrix(  pi_star_red[-c(1:(length(pi_star_red)/2))]))
    pi_star_red_ast <- strenv$jnp$array(as.matrix(  pi_star_red[1:(length(pi_star_red)/2)] ) )

    q_star_gd <- q_star <- strenv$np$array(  q_with_pi_star_full )[1]
    # sanity check: 
    # strenv$np$array(  q_with_pi_star_full )[1]  - sum(strenv$np$array(  q_with_pi_star_full )[2:3]*c(AstProp, DagProp)) 
    pi_star_full_gd <- pi_star_full <- strenv$np$array( q_with_pi_star_full )[-c(1:3)]

    #  https://github.com/google/jax/issues/1696 
    jacobian_mat_gd <- jacobian_mat <- matrix(0, ncol = 4*REGRESSION_PARAMS_jax_ast_jnp$shape[[1]],
                                                 nrow = q_with_pi_star_full$shape[[1]])
    diag(jacobian_mat_gd) <- diag(jacobian_mat) <- 1
    vcov_OutcomeModel_concat <- matrix(0, nrow = nrow(vcov_OutcomeModel_ast_jnp)*4,
                                          ncol = nrow(vcov_OutcomeModel_ast_jnp)*4)
    if(compute_se){
      message("Computing SEs...")
      # first, compute vcov
      vcov_OutcomeModel_concat <- as.matrix( Matrix::bdiag( list(
                                          vcov_OutcomeModel_ast_jnp,
                                          vcov_OutcomeModel_dag_jnp,
                                          vcov_OutcomeModel_ast0_jnp,
                                          vcov_OutcomeModel_dag0_jnp  )  ) ) 

      # jacfwd uses forward-mode automatic differentiation, which is more efficient for “tall” Jacobian matrices
      # jacrev uses reverse-mode, which is more efficient for “wide” Jacobian matrices.
      # For near-square matrices, jacfwd probably has an edge over jacrev.
      # note: do not jit compile as computation only used once (compilation induces overhead)
      jacobian_mat <- strenv$jax$jacrev(getPiStar_gd, 0L:3L)(
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
                                  functionList = list(dQ_da_ast, dQ_da_dag,
                                                      QFXN, getMultinomialSamp),
                                  functionReturn = F,
                                  a_i_ast = a_vec_init_ast, 
                                  a_i_dag = a_vec_init_dag, 
                                  gd_full_simplex = T, 
                                  quiet = F )
      jacobian_mat_gd <- jacobian_mat <- lapply(jacobian_mat,function(l_){
        strenv$np$array( strenv$jnp$squeeze(strenv$jnp$squeeze(strenv$jnp$array(l_,strenv$dtj),1L),2L) ) })
      jacobian_mat_gd <- jacobian_mat <- do.call(cbind, jacobian_mat)
      # plot(colMeans(abs(jacobian_mat_gd)))
    }
  }

  # the first three entries of output are:
  # Qhat_population, Qhat_ast, Qhat_dag
  vcov_PiStar <- jacobian_mat %*% vcov_OutcomeModel_concat %*% t(jacobian_mat)
  q_star <- as.matrix(   q_star  )
  q_star_se <- sqrt(  diag( vcov_PiStar )[1] )
  pi_star_numeric <- strenv$np$array( pi_star_full ) # - c(1:3) already extracted 

  # drop the q part
  if(diff == T){ pi_star_se <- sqrt(  diag( vcov_PiStar )[-c(1:3)] ) }
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
    bound_ <- lapply(1:K,function(k_){
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
    pi_star_dag_vec_jnp <- pi_star_vec_jnp <- pi_star_red_dag <- pi_star_red_ast <- pi_star_numeric
    p_vec_full_ast_jnp <- p_vec_full_dag_jnp <- p_vec_full
  }

  message("strategize() call has finished...")
  message("-------------")
  return( list(   "pi_star_point" = pi_star_list,
                  "pi_star_se" = pi_star_se_list,
                  "Q_point_mEst" = q_star,
                  "Q_se_mEst"= q_star_se,
                  "PiStar_vec" = pi_star_numeric,
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
                  "lambda" = lambda,
                  "p_vec_full" = p_vec_full,
                  "regularization_adjust_hash" = regularization_adjust_hash,
                  "p_list" = p_list,
                  "slate_list" = slate_list, 

                  # reconstruct q info
                  "Qfxn" = QFXN,

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
                  "force_gaussian" = force_gaussian,
                  "used_regularization" = UsedRegularization,
                  "estimation_type" = "TwoStep",
                  "Y_model" = my_model ) )
}
