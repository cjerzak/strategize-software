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
    print("Initializing environment...")
    library(tensorflow); library(keras)
    if(!is.null(conda_env)){
      try(tensorflow::use_condaenv(conda_env,
                                   required = conda_env_required), T)
    }
    CPUDevice <- tf$config$list_physical_devices()[[1]]
    tf$config$set_visible_devices( CPUDevice )
    tf$config$set_soft_device_placement(T)
    dttf <- tf$float64
    try(tf$config$experimental$set_memory_growth(tf$config$list_physical_devices('GPU')[[1]], T),T)
    try(tfp <- tf_probability(),T)
    try(tfd <- tfp$distributions,T)
    print(tf$version$VERSION)

    # import computational modules
    jax <- tensorflow::import("jax",as="jax")
    jnp <- tensorflow::import("jax.numpy")
    py_gc <- reticulate::import("gc")

    # setup numerical precision for delta method
    dtj <- jnp$float64
    jax$config$update("jax_enable_x64", T)
  }

  if(is.null(lambda_seq) & is.null(lambda)){
    lambda_seq <- 10^seq(-4, 0, length.out = 5) * sd(Y, na.rm = T)
  }
  if(is.null(lambda_seq) & !is.null(lambda)){ lambda_seq <- lambda }

  # CV sequence
  {
    print("Starting CV sequence...")
    cv_info_orig <- outsamp_results <- replicate(length(lambda_seq),list())
    insamp_results <- c() ; lambda_counter <- 0;
    for(lambda__ in lambda_seq){
      gc(); py_gc$collect()
      print(lambda__)
      lambda_counter <- lambda_counter + 1
      Qoptimized__ <- replicate(n = folds, list())
      if(is.null(respondent_id)){ respondent_id <- 1:length(Y) }
      indi_list <- tapply( 1:length(Y), respondent_id,  c  )
      split_i1 <- sample(1:length(indi_list),length(indi_list)/folds)
      split_i2 <- (1:length(indi_list))[! 1:length(indi_list) %in% split_i1 ]
      indi_list <- list(unlist(indi_list[split_i1]),unlist(indi_list[split_i2]))

      # CV sequence
      for(split_ in c(1:folds)){
        Qoptimized__[[split_]] <- OptiConjoint(

          # input data
          Y = Y[indi_list[[split_]]],
          W = W[indi_list[[split_]],],
          X = ifelse(analysisType=="cluster", yes = list(X[indi_list[[split_]],]), no = list(NULL))[[1]],
          varcov_cluster_variable = varcov_cluster_variable[indi_list[[split_]]],
          pair_id = pair_id[indi_list[[split_]]],
          respondent_id = respondent_id[ indi_list[[split_]] ],
          respondent_task_id = respondent_task_id[ indi_list[[split_]] ],
          profile_order = profile_order[ indi_list[[split_]] ],
          p_list = p_list,
          lambda = lambda__,

          # hyperparameters
          ComputeSEs = F, # -> must be set to F for CV sequence
          nSGD = nSGD,
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
          conda_loc = conda_loc,
          conda_env_required = conda_env_required)
        if("try-error" %in% class(Qoptimized__[[split_]])){ stop("Failed in Qoptimized__[[split_]]!");browser() }
      }
      q_vec_in <- c(Qoptimized__[[1]]$Q_point, Qoptimized__[[2]]$Q_point)
      q_vec_cv <- c(Qoptimized__[[2]]$Qfxn(
        "pi_star_ast" = Qoptimized__[[1]]$pi_star_red_ast,
        "pi_star_dag" = Qoptimized__[[1]]$pi_star_red_dag,
        "EST_INTERCEPT_tf_ast" = Qoptimized__[[2]]$EST_INTERCEPT_jnp,
        "EST_COEFFICIENTS_tf_ast" = Qoptimized__[[2]]$EST_COEFFICIENTS_jnp,
        "EST_INTERCEPT_tf_dag" = Qoptimized__[[2]]$EST_INTERCEPT_jnp,
        "EST_COEFFICIENTS_tf_dag" = Qoptimized__[[2]]$EST_COEFFICIENTS_jnp)$tolist()[[1]],
        Qoptimized__[[1]]$Qfxn(
          "pi_star_ast" = Qoptimized__[[2]]$pi_star_red_ast,
          "pi_star_dag" = Qoptimized__[[2]]$pi_star_red_dag,
          "EST_INTERCEPT_tf_ast" = Qoptimized__[[1]]$EST_INTERCEPT_jnp,
          "EST_COEFFICIENTS_tf_ast" = Qoptimized__[[1]]$EST_COEFFICIENTS_jnp,
          "EST_INTERCEPT_tf_dag" = Qoptimized__[[1]]$EST_INTERCEPT_jnp,
          "EST_COEFFICIENTS_tf_dag" = Qoptimized__[[1]]$EST_COEFFICIENTS_jnp)$tolist()[[1]]
      )
      #plot(q_vec_cv,rev(q_vec_in));abline(a=0,b=1)
      lambda_results_ <- c(lambda__, mean(q_vec_cv), se(q_vec_cv))
      insamp_results <- rbind(insamp_results, c(lambda__, mean(q_vec_in), se(q_vec_in)))
      outsamp_results[[lambda_counter]] <- cv_info_orig[[lambda_counter]] <- list(lambda_results_,"")
      #lambda_results_ <- c(lambda__, Qoptimized_$Q_point_mEst, Qoptimized_$Q_se_mEst)
      #outsamp_results[[lambda_counter]] <- cv_info_orig[[lambda_counter]] <- list(lambda_results_,Qoptimized_)
      print(lambda_results_)
    }
    outsamp_results <- as.data.frame( t( do.call(cbind, lapply(outsamp_results,function(zer){zer[[1]]})) ) )
    colnames(insamp_results) <- colnames(outsamp_results) <- c("lambda","Q","Qse")
    # insight: in and out Q are dif when should be same - indexing off?
    #https://www.youtube.com/watch?v=AzwXNW6BYf0
    insamp_results <- as.data.frame( insamp_results )
    outsamp_results$l_bound <- outsamp_results$Q - (qStar_lambda <- 1) * outsamp_results$Qse
    outsamp_results$u_bound <- outsamp_results$Q + qStar_lambda * outsamp_results$Qse
    #plot(insamp_results$Q,outsamp_results$Q);abline(a=0,b=1)
    #plot(insamp_results$Q-outsamp_results$Q);abline(a=0,b=1)
    #plot(log(outsamp_results$lambda), outsamp_results$Q,pch = as.character(  rank(-outsamp_results$l_bound)) )
    lambda__ <- lambda_seq[which(max(outsamp_results$Q) <= outsamp_results$u_bound)[1]] # 1 se rule
    print("Done with CV sequence!")
  }

  # final output
  {
    print("Starting final OptiConjoint run...")
    gc(); py_gc$collect()
    Qoptimized_ <- OptiConjoint(

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
                                lambda = lambda__, # this lambda is chosen via CV

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
                                conda_loc = conda_loc,
                                conda_env_required = conda_env_required)
    print("Done with final OptiConjoint run!")
  }

  # RETURN
  return(  c( Qoptimized_,
            "lambda" = lambda__,
            "qStar_lambda" = qStar_lambda,
            "OptiConjointCVInfo" = list(outsamp_results )) )
}
