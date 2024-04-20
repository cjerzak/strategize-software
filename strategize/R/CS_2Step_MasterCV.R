#' Implements...
#'
#' @usage
#'
#' cv.OptiConjoint(...)
#'
#' @param x Description
#'
#' @return `z` Description
#' @export
#'
#' @details `cv.OptiConjoint` implements...
#'
#' @examples
#'
#' # Perform analysis
#' cv.OptiConjoint <- OptiConjoint()
#'
#' print( cv.OptiConjoint )
#'
#' @export
#'
#' @md

cv.OptiConjoint       <-          function(
                                            Y,
                                            W,
                                            X = NULL,
                                            lambda_seq = NULL,
                                            lambda = NULL,
                                            folds = 2L,
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
                                            UseOptax = F,
                                            K = 1,
                                            nSGD = 100,
                                            diff = F, MaxMin = F,
                                            UseRegularization = F,
                                            OpenBrowser = F,
                                            ForceGaussianFamily = F,
                                            A_INIT_SD = 0.001,
                                            TypePen = "KL",
                                            ComputeSEs = T,
                                            conda_env = NULL,
                                            conda_env_required = F,
                                            confLevel = 0.90,
                                            nFolds_glm = 3L,
                                            nMonte_MaxMin = 5L,
                                            nMonte_Qglm = 100L,
                                            jax_seed = as.integer(Sys.time()),
                                            OptimType = "default"){
  # define evaluation environment
  evaluation_environment <- environment()

  # initialize environment
  {
    print2("Initializing environment...")
    if(!is.null(conda_env)){
      try(reticulate::use_condaenv(conda_env, required = conda_env_required), T)
    }

    # import computational modules
    jax <- reticulate::import("jax",as="jax")
    jnp <- reticulate::import("jax.numpy")
    py_gc <- reticulate::import("gc")

    # setup numerical precision for delta method
    dtj <- jnp$float64
    jax$config$update("jax_enable_x64", T)
  }

  # setup lamba 
  if(is.null(lambda_seq) & is.null(lambda)){
    lambda_seq <- 10^seq(-4, 0, length.out = 5) * sd(Y, na.rm = T)
  }
  if(is.null(lambda_seq) & !is.null(lambda)){ lambda_seq <- lambda }

  if(is.null(respondent_id)){ respondent_id <- 1:length(Y) }

  # CV sequence
  {
    print2("Starting CV sequence...")
    outsamp_results <- insamp_results <- matrix(nrow = 0, ncol = 4, dimnames = list(NULL, c("lambda","Qhat","Qse","selected")))

    # build cv splits - same for all lambda 
    all_tabs <- apply(W,2,table)
    ok_counter <- 0; ok <- F; while(!ok){ 
        ok_counter <- ok_counter + 1 
        
        # sample based on unique respondent-tasks 
        tmp_ <- (paste0(respondent_id, "_", respondent_task_id))
        indi_list <- sample(1:folds, size = length(unique(tmp_)), replace = T)
        names(indi_list) <- unique(tmp_)
        indi_list <- indi_list[tmp_]
        
        indi_list <- sapply(1:folds, function(f_){
          list(which(!indi_list %in% f_), # pos 1 is in
               which(indi_list %in% f_) ) # pos 2 is out 
        })
        split_tabs_in <- apply(indi_list,2,function(l_){ apply(W[l_[[1]],],2,table) })
        if(all(names(unlist(all_tabs)) == names(unlist(split_tabs_in)))){
          if(all(unlist(split_tabs_in) > 10)){ ok <- T }
        }
        if(ok_counter > 1000){stop("Stopping: Could not find split with > 10 observations per factor level.")}
    }
    
    lambda_counter <- 0; for(lambda__ in lambda_seq){
      lambda_counter <- lambda_counter + 1
      Qoptimized__ <- replicate(n = folds, list())

      # CV sequence
      q_vec_in <- q_vec_out <- c()
      for(split_ in c(1:folds)){
        out_indices <- indi_list[2,split_][[1]]; gc(); py_gc$collect()
        for(type_ in c(1,2)){ 
          # in sample optimization of pi*, evaluation on OOS coefficients 
          use_indices <- indi_list[type_,split_][[1]]
          nSGD_use <- ifelse(type_ == 1, yes = nSGD, no = 1L)
          Qoptimized__[[split_]][[type_]] <- OptiConjoint(
  
            # input data
            Y = Y[use_indices],
            W = W[use_indices,],
            X = ifelse(analysisType=="cluster", yes = list(X[use_indices,]), no = list(NULL))[[1]],
            varcov_cluster_variable = varcov_cluster_variable[use_indices],
            pair_id = pair_id[use_indices],
            respondent_id = respondent_id[ use_indices ],
            respondent_task_id = respondent_task_id[ use_indices ],
            profile_order = profile_order[ use_indices ],
            p_list = p_list,
            slate_list = slate_list, 
            UseOptax = UseOptax, 
            lambda = lambda__,
  
            # hyperparameters
            ComputeSEs = F, 
            nSGD = nSGD_use,
            TypePen = TypePen,
            K = K,
            ForceGaussianFamily = ForceGaussianFamily,
            UseRegularization = UseRegularization,
            OptimType = OptimType,
            A_INIT_SD = A_INIT_SD,
            nFolds_glm = nFolds_glm,
            diff = diff,
            MaxMin = MaxMin,
            conda_env = conda_env,
            conda_env_required = conda_env_required)
        }
        
        # out of sample test of pi* on new estimates 
        q_vec_in <- c(q_vec_in, Qoptimized__[[split_]][[1]]$Q_point)
        q_vec_out <- c(q_vec_out, unlist(Qoptimized__[[split_]][[2]]$Qfxn(
          "pi_star_ast" = Qoptimized__[[split_]][[1]]$pi_star_red_ast,
          "pi_star_dag" = Qoptimized__[[split_]][[1]]$pi_star_red_dag,
          "EST_INTERCEPT_tf_ast" = Qoptimized__[[split_]][[2]]$EST_INTERCEPT_jnp,
          "EST_COEFFICIENTS_tf_ast" = Qoptimized__[[split_]][[2]]$EST_COEFFICIENTS_jnp,
          "EST_INTERCEPT_tf_dag" = Qoptimized__[[split_]][[2]]$EST_INTERCEPT_jnp,
          "EST_COEFFICIENTS_tf_dag" = Qoptimized__[[split_]][[2]]$EST_COEFFICIENTS_jnp)$tolist()[[1]])
        )
      }
      outsamp_results <- as.data.frame(rbind(outsamp_results, c(lambda__, mean(q_vec_out), se(q_vec_out), 0)))
      insamp_results <- as.data.frame(rbind(insamp_results, c(lambda__, mean(q_vec_in), se(q_vec_in), 0)))
    }

    outsamp_results$l_bound <- outsamp_results$Qhat - (qStar_lambda <- 1) * outsamp_results$Qse
    outsamp_results$u_bound <- outsamp_results$Qhat + qStar_lambda * outsamp_results$Qse
    lambda__ <- lambda_seq[which_selected <- which.max(outsamp_results$Qhat)] # lambda.min rule
    #lambda__ <- lambda_seq[which_selected <- which(max(outsamp_results$Qhat <= outsamp_results$u_bound)[1]] # lambda.se rule
    outsamp_results$selected[which_selected] <- 1 
    print2("Done with CV sequence & starting final run with log(lambda) of %.2f...", log(lambda__))
  }

  # final output
  gc(); py_gc$collect(); Qoptimized_ <- OptiConjoint(
                              # input data
                              Y = Y,
                              W = W,
                              X = ifelse(K > 1, yes = list(X), no = list(NULL))[[1]],
                              nSGD = nSGD,
                              TypePen = TypePen,
                              varcov_cluster_variable = varcov_cluster_variable,
                              competing_group_variable_respondent = competing_group_variable_respondent,
                              competing_group_variable_candidate = competing_group_variable_candidate,
                              competing_group_competition_variable_candidate = competing_group_competition_variable_candidate,
                              pair_id = pair_id,
                              respondent_id = respondent_id,
                              respondent_task_id = respondent_task_id,
                              profile_order = profile_order,
                              p_list = p_list,
                              slate_list = slate_list, 
                              UseOptax = UseOptax, 
                              lambda = lambda__, # this lambda is the one chosen via CV

                              # hyperparameters
                              OptimType = OptimType,
                              ForceGaussianFamily = ForceGaussianFamily,
                              UseRegularization = UseRegularization,
                              A_INIT_SD = A_INIT_SD,
                              ComputeSEs = ComputeSEs,
                              K = K,
                              nMonte_MaxMin = nMonte_MaxMin,
                              nFolds_glm = nFolds_glm,
                              diff = diff,
                              MaxMin = MaxMin,
                              conda_env = conda_env,
                              conda_env_required = conda_env_required)
  print2("Done with strategic analysis!")
  return(  c( Qoptimized_,
            "lambda" = lambda__,
            "qStar_lambda" = qStar_lambda,
            "CVInfo" = list(outsamp_results )) )
}
