#' computeQ_TwoStep
#'
#' Implements the organizational record linkage algorithms of Jerzak and Libgober (2021).
#'
#' @usage
#'
#' computeQ_TwoStep(x, Y, by ...)
#'
#' @param x,Y data frames to be merged
#'
#' @return `z` The merged data frame.
#' @export
#'
#' @details `LinkOrgs` automatically processes the name text for each dataset (specified by `by`, `by.x`, and/or `by.Y`. Users may specify the following options:
#'
#' - Set `DistanceMeasure` to control algorithm for computing pairwise string distances. Options include "`osa`", "`jaccard`", "`jw`". See `?stringdist::stringdist` for all options. (Default is "`jaccard`")
#'
#' @examples
#'
#' #Create synthetic data
#' x_orgnames <- c("apple","oracle","enron inc.","mcdonalds corporation")
#' y_orgnames <- c("apple corp","oracle inc","enron","mcdonalds co")
#' x <- data.frame("orgnames_x"=x_orgnames)
#' Y <- data.frame("orgnames_y"=y_orgnames)
#'
#' # Perform merge
#' linkedOrgs <- LinkOrgs(x = x,
#'                        Y = Y,
#'                        by.x = "orgnames_x",
#'                        by.Y = "orgnames_y",
#'                        MaxDist = 0.6)
#'
#' print( linkedOrgs )
#'
#' @export
#'
#' @md

computeQ_TwoStep       <-          function(Y,
                                            W,
                                            lambda,
                                            varcov_cluster_variable = NULL,
                                            competing_respondent_group_variable = NULL,
                                            competing_candidate_group_variable = NULL,
                                            pair_id = NULL,
                                            p_list = NULL,
                                            full_dat = full_dat,
                                            kEst = 1,
                                            nSGD = 100,
                                            diff = F, MaxMin = F,
                                            UseRegularization = F,
                                            OpenBrowser = F,
                                            ForceGaussianFamily = F,
                                            A_INIT_SD = 0,
                                            TypePen = "KL",
                                            ComputeSEs = T,
                                            confLevel = 0.90,
                                            OptimType = "default"){

  # load in packages - may help memory bugs to load them in thru package
  if(T == T){
    library(tensorflow); library(keras)
    try(tensorflow::use_python(python = "/Users/cjerzak/miniforge3/bin/python", required = T),T)
    try(tensorflow::use_condaenv("tensorflow_m1",required = T, conda = "~/miniforge3/bin/conda"), T)
    try(tf$config$experimental$set_memory_growth(tf$config$list_physical_devices('GPU')[[1]], T),T)
    try(tfp <- tf_probability(),T)
    try(tfd <- tfp$distributions,T)
    print(tf$version$VERSION)

    # jax
    jax <- tensorflow::import("jax",as="jax")
    optax <- tensorflow::import("optax")
    jnp <- tensorflow::import("jax.numpy")
    tf2jax <- tensorflow::import("tf2jax",as="tf2jax")
    evaluation_environment <- environment()
  }

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
        p_list_PreRegularization[]
        pi_[[k_]][update_these] <- updates_
        if(isSE){
          pi_[[k_]][-update_these] <- lapply(pi_[[k_]][-update_these],
                                             function(rzer){rzer[]<-NA;return(rzer)})
        }
      }
    return( pi_ ) }
  }

  q_ave <- q_dag_ave <- pi_star_ave <- NULL
  if(OpenBrowser == T){ browser() }
  nSGD_orig <- nSGD
  ai <- as.integer
  w_orig <- W
  #usedRegularization <- F
  #MaxMinType <- "OneRoundDouble"
  #MaxMinType <- "OneRoundSingle"
  MaxMinType <- "TwoRoundSingle"

  glm_family = "gaussian"; glm_outcome_transform <- tf$identity
  if(!ForceGaussianFamily){
    if(mean(unique(Y) %in% c(0,1)) == 1){
      glm_family = "binomial";
      glm_outcome_transform <- tf$nn$sigmoid;
      #glm_outcome_transform <- function(x){return(   0.5*(1+tf$math$erf(x / sqrt(2)) ))  } #probit; glm_outcome_transform(1.2); pnorm(1.2)
      #if(diff == T){ glm_outcome_transform <- tf$identity }
  } }

  # ensure naming conventions are correct (i.e. in alignment with p_list if available)
  if(is.null(p_list) | is.null(names(p_list[[1]]))){
    names_list <- apply(w_orig,2,function(zer){list(sort(names(table(as.factor(zer))),decreasing=F))})
  }
  if(!is.null(p_list) & !is.null(names(p_list[[1]]))){
    names_list <- lapply(p_list,function(zer){list(names(zer))})
  }
  W <- sapply(1:ncol(W),function(zer){
    match(W[,zer],names_list[[zer]][[1]]) })

  # get info about # of levels per factor
  factor_levels_full <- factor_levels <- apply(W,2,function(zer){length(unique(zer))})

  if(diff == F){
    initialize_ModelOutcome_FindIt <- paste(deparse(generate_ModelOutcome_FindIt),collapse="\n")
    initialize_ModelOutcome_FindIt <- gsub(initialize_ModelOutcome_FindIt,pattern="function \\(\\)",replace="")
    eval( parse( text = initialize_ModelOutcome_FindIt ),envir = evaluation_environment )
  }

  # obtain outcome models
  if(T == T){
    UseRegularization_ORIG <- UseRegularization
    Rounds <- c(0,1)

    nRounds <- GroupsPool <- 1; RoundsPool <- 1
    if(MaxMin){
      nRounds <- 2
      RoundsPool <- c(0,1)
      GroupsPool <- sort(unique(competing_candidate_group_variable))
    }

    for(Round_ in RoundsPool){
    for(GroupCounter in 1:length(GroupsPool)){
      print(c(Round_, GroupCounter))
      # select data to use for each iteration
      #W_ <- W; Y_ <- Y; varcov_cluster_variable_ <- varcov_cluster_variable
      UseRegularization <- UseRegularization_ORIG
      if(MaxMin == F){ indi_ <- 1:length( Y )  }
      if(MaxMin == T){
        if(Round_ == 0){
          indi_ <- which( competing_respondent_group_variable == GroupsPool[ GroupCounter ] &
                      ( full_dat$Party.competition == "Same" &
                          competing_candidate_group_variable == GroupsPool[GroupCounter]) )
        }
        if(Round_ == 1){
          indi_ <- which( competing_respondent_group_variable == GroupsPool[ GroupCounter ] &
                            ( full_dat$Party.competition == "Different" &
                                competing_candidate_group_variable %in% GroupsPool) )
        }
        DagProp <- prop.table(table(competing_respondent_group_variable[competing_respondent_group_variable %in% GroupsPool]))[2]
      }

      # subset data
      W_ <- W[indi_,]; Y_ <- Y[indi_];
      varcov_cluster_variable_ <- varcov_cluster_variable[indi_]
      full_dat_ <- full_dat[ indi_ ,]
      pair_id_ <- pair_id[ indi_ ]
      #table(full_dat_$Party.affiliation_clean)
      #table(full_dat_$R_Partisanship)
      #table(table(pair_id_))

      # run models with inputs: W_; Y_; varcov_cluster_variable_; full_dat_
      initialize_ModelOutcome <- paste(deparse(generate_ModelOutcome),collapse="\n")
      initialize_ModelOutcome <- gsub(initialize_ModelOutcome,pattern="function \\(\\)",replace="")
      eval( parse( text = initialize_ModelOutcome ), envir = evaluation_environment )

      REGRESSION_PARAMS_jax <- jnp$array(tf$concat(list(EST_INTERCEPT_tf, EST_COEFFICIENTS_tf),0L))

      # rename as appropriate
      ret_chunks <- c("vcov_OutcomeModel", "main_info","interaction_info","interaction_info_PreRegularization",
            "REGRESSION_PARAMS_jax","regularization_adjust_hash","main_dat", "my_mean","EST_INTERCEPT_tf","my_model", "EST_COEFFICIENTS_tf")
      dag_condition <- (GroupCounter == 1) | (MaxMin == F)
      round_text <- ifelse( Round_==0,yes="0",no="")
      if( dag_condition ){
          tmp <- sapply(ret_chunks,function(chunk_){ eval(parse(text = sprintf("%s_dag%s = %s",chunk_,round_text,chunk_)),envir = evaluation_environment) })
          rm(tmp)
      }
      if( !dag_condition ){
          tmp <- sapply(ret_chunks,function(chunk_){ eval(parse(text = sprintf("%s_ast%s = %s",chunk_,round_text,chunk_)),envir = evaluation_environment) })
          rm(tmp)
       }
    }
    }
  }

  if(!"vcov_OutcomeModel_dag0" %in% ls()){
    vcov_OutcomeModel_dag0 <- vcov_OutcomeModel_dag
    EST_INTERCEPT_tf_dag0 <- EST_INTERCEPT_tf_dag
    EST_COEFFICIENTS_tf_dag0 <- EST_COEFFICIENTS_tf_dag
    REGRESSION_PARAMS_jax_dag0 <- REGRESSION_PARAMS_jax_dag
  }
  if(!"vcov_OutcomeModel_ast0" %in% ls()){
    vcov_OutcomeModel_ast0 <- vcov_OutcomeModel_dag
    EST_INTERCEPT_tf_ast0 <- EST_INTERCEPT_tf_dag
    EST_COEFFICIENTS_tf_ast0 <- EST_COEFFICIENTS_tf_dag
    REGRESSION_PARAMS_jax_ast0 <- REGRESSION_PARAMS_jax_dag
  }
  if(!"vcov_OutcomeModel_ast" %in% ls()){
    vcov_OutcomeModel_ast <- vcov_OutcomeModel_dag
    EST_INTERCEPT_tf_ast <- EST_INTERCEPT_tf_dag
    EST_COEFFICIENTS_tf_ast <- EST_COEFFICIENTS_tf_dag
    REGRESSION_PARAMS_jax_ast <- REGRESSION_PARAMS_jax_dag
  }
  print("Done fitting outcome models!...")

  n_main_params <- nrow( main_info )
  if(is.null(p_list) & any(apply(W,2,function(zer){
    max(abs(prop.table(table(zer))-1/length(unique(zer))))})>0.1)){
    stop("Must adjust for non-uniform assignment probability!")
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

  ParameterizationType <- ifelse( DiffType == "glm" | diff == F, yes = "Implicit", no = "Full")
  n2int <- function(x){tf$constant(x,tf$int32)}
  d_locator <- unlist(sapply(1:length(factor_levels),function(l_d){rep(l_d,times=factor_levels[l_d]-1)}))
  #length(d_locator); length(p_vec)
  p_vec_sum_prime <- unlist(tapply(1:length(p_vec),d_locator,function(er){
    sapply(er,function(re){sum(p_vec[er[!er %in% re]])}) }))
  d_locator_full <- unlist(sapply(1:length(factor_levels),function(l_d){rep(l_d,times=factor_levels[l_d])}))
  p_vec_sum_prime_full <- unlist(tapply(1:length(p_vec_full),d_locator_full,function(er){
    sapply(er,function(re){sum(p_vec_full[er[!er %in% re]])}) }))
  main_indices_i0 <- tf$constant((ai(1:n_main_params-1L)),tf$int32)
  inter_indices_i0 <- tf$constant((ai(((n_main_params+1):length(my_mean))-1L)),tf$int32)
  if(ParameterizationType == "Implicit"){ p_vec_use <- p_vec; p_vec_sum_prime_use <- p_vec_sum_prime }
  if(ParameterizationType == "Full"){ p_vec_use <- p_vec_full; p_vec_sum_prime_use <- p_vec_sum_prime_full }

  if(ParameterizationType == "Implicit"){
    print("Initializing manual exact solution code...")
    initialize_ExactSol <- paste(deparse(generate_ExactSol),collapse="\n")
    initialize_ExactSol <- gsub(initialize_ExactSol,pattern="function \\(\\)",replace="")
    eval( parse( text = initialize_ExactSol ),envir = evaluation_environment )
    getPiStar_exact <- generate_ExactSolImplicit
  }
  if(ParameterizationType == "Full"){
    print("Initializing manual exact solution code...")
    initialize_ExactSol <- paste(deparse(generate_ExactSol),collapse="\n")
    initialize_ExactSol <- gsub(initialize_ExactSol,pattern="function \\(\\)",replace="")
    eval( parse( text = initialize_ExactSol ),envir = evaluation_environment )
    getPiStar_exact <- generate_ExactSolExplicit
  }

  # pi in constrained space using gradient ascent
  p_vec_tf <- tf$constant(as.matrix(p_vec_use),tf$float32)
  inv_learning_rate <- tf$constant(1., dtype=tf$float32)
  #tf_function_ex <- function(x){x}
  tf_function_ex <- function(x){tf_function(x)}

  # initialize manual gradient updates
  print("Initializing manual gradient updates...")
  initialize_ManualDoUpdates <- paste(deparse(generate_ManualDoUpdates),collapse="\n")
  initialize_ManualDoUpdates <- gsub(initialize_ManualDoUpdates,pattern="function \\(\\)",replace="")
  eval( parse( text = initialize_ManualDoUpdates ),envir = evaluation_environment )

  # LR updates, etc.
  GetInvLR <- tf_function_ex(function(inv_learning_rate,grad_i){
    # WN grad
    #return( tf$stop_gradient(tf$add(inv_learning_rate,tf$divide(tf$reduce_sum(tf$square(grad_i)), inv_learning_rate))) )

    # Adagrad-norm
    return( tf$stop_gradient(tf$add(inv_learning_rate,tf$reduce_sum(tf$square(grad_i)))))
  })

  GetUpdatedParameters <- tf_function_ex(function(a_vec, grad_i, inv_learning_rate_i){
    return( tf$add(a_vec, tf$multiply(tf$math$reciprocal(inv_learning_rate_i),grad_i)))
  })

  a_vec_init_mat <- as.matrix(unlist( lapply(p_list, function(zer){ c(compositions::alr( t((zer)))) }) ) )
  a_vec_init_ast <- tf$constant(a_vec_init_mat+rnorm(length(a_vec_init_mat),sd=A_INIT_SD) ,tf$float32)
  a_vec_init_dag <- tf$constant(a_vec_init_mat+rnorm(length(a_vec_init_mat),sd = A_INIT_SD*MaxMin),tf$float32)
  a2Simplex <- tf_function_ex(function(a_){
    exp_a_ <- tf$exp(a_)
    aOnSimplex <- tapply(1:nrow(main_info),main_info$d,function(zer){
      tmp <- tf$divide(  tf$gather(exp_a_, n2int(zer-1L) ),
                        tf$add(OneTf_flat,tf$reduce_sum(tf$gather(exp_a_, n2int(zer-1L) ) )))
      if(length(dim(tmp)) == 1){ tmp <- tf$expand_dims( tmp, 0L ) }
      return( list( tmp ) ) })
    names(aOnSimplex) <- NULL
    return(  tf$concat(aOnSimplex,0L) )
  })
  a2FullSimplex <- tf_function_ex(function(a_){
    exp_a_ <- tf$exp(a_)
    aOnSimplex <- tapply(1:nrow(main_info_leftoutLdminus1),main_info_leftoutLdminus1$d,function(zer){
      exp_a_zer <- tf$concat(list(tf$gather(exp_a_, n2int(as.array(zer - 1L) )),
                                  as.matrix(1.)),axis = 0L)
      tmp <- tf$divide(  exp_a_zer, tf$reduce_sum(exp_a_zer))
      if(length(dim(tmp)) == 1){ tmp <- tf$expand_dims( tmp, 0L ) }
      return( list( tmp ) ) })
    names(aOnSimplex) <- NULL
    return(  tf$concat(aOnSimplex,0L) )
  })
  OneTf <- tf$constant(matrix(1L),tf$float32)
  OneTf_flat <- tf$constant(1L,tf$float32)
  Neg2_tf <- tf$constant(-2.,tf$float32)

  # Q functions
  print("Defining Q functions..")
  pi_star_value_init_ast <- a2Simplex( a_vec_init_ast )
  pi_star_value_init_dag <- a2Simplex( a_vec_init_dag )
  pi_star_value_init_ast <- a2Simplex( a_vec_init <- a_vec_init_ast )
  getQStar_single <- tf_function(function(pi_star,
                                   EST_COEFFICIENTS_tf,
                                   EST_INTERCEPT_tf){
    # coef info
    main_coef <- tf$gather(EST_COEFFICIENTS_tf, indices = main_indices_i0, axis = 0L)
    inter_coef <- tf$gather(EST_COEFFICIENTS_tf,indices = inter_indices_i0, axis = 0L)

    # get interaction info
    pi_dp <- tf$gather(pi_star, n2int(as.integer(interaction_info$dl_index_adj)-1L), axis=0L)
    pi_dpp <- tf$gather(pi_star, n2int(as.integer(interaction_info$dplp_index_adj)-1L), axis=0L)

    Qhat <- glm_outcome_transform(
        EST_INTERCEPT_tf + tf$matmul(tf$transpose(main_coef),pi_star) +
          tf$reduce_sum(tf$multiply(tf$multiply(inter_coef,pi_dp),pi_dpp),keepdims=T) )
    return( Qhat ) })
  getQStar_single_conv <- tf2jax$convert_functional(
                                        getQStar_single,
                                        pi_star = jnp$array(pi_star_value_init_ast),
                                        EST_INTERCEPT_tf = jnp$array(EST_INTERCEPT_tf),
                                        EST_COEFFICIENTS_tf = jnp$array(EST_COEFFICIENTS_tf))
  for(UseSinglePop in ifelse(MaxMin, yes = list(c(F,T)), no = list(T))[[1]]){
    # general specifications
    getQStar_diff_R <- function(pi_star_ast, pi_star_dag,
                                EST_COEFFICIENTS_tf_ast, EST_INTERCEPT_tf_ast,
                                EST_COEFFICIENTS_tf_dag, EST_INTERCEPT_tf_dag){
      # coef
      main_coef_ast <- tf$gather(EST_COEFFICIENTS_tf_ast, indices = main_indices_i0, axis = 0L)
      inter_coef_ast <- tf$gather(EST_COEFFICIENTS_tf_ast, indices = inter_indices_i0, axis = 0L)

      # get interaction info
      pi_ast_dp <- tf$gather(pi_star_ast, n2int(as.integer(interaction_info$dl_index_adj)-1L), axis=0L)
      pi_ast_dpp <- tf$gather(pi_star_ast, n2int(as.integer(interaction_info$dplp_index_adj)-1L), axis=0L)
      pi_ast_prod <- tf$multiply(pi_ast_dp, pi_ast_dpp)

      pi_dag_dp <- tf$gather(pi_star_dag, n2int(as.integer(interaction_info$dl_index_adj)-1L), axis=0L)
      pi_dag_dpp <- tf$gather(pi_star_dag, n2int(as.integer(interaction_info$dplp_index_adj)-1L), axis=0L)
      pi_dag_prod <- tf$multiply(pi_dag_dp, pi_dag_dpp)

      # combine
      Qhat_population <- Qhat_dag <- Qhat <- glm_outcome_transform( EST_INTERCEPT_tf_ast +
                                                                      tf$matmul(tf$transpose(main_coef_ast), pi_star_ast - pi_star_dag) +
                                                                      tf$reduce_sum(tf$multiply(inter_coef_ast, pi_ast_prod - pi_dag_prod), keepdims = T))

      # get dag value
      if( SINGLE_PROP_KEY ){
        main_coef_dag <- tf$gather(EST_COEFFICIENTS_tf_dag, indices = main_indices_i0, axis=0L)
        inter_coef_dag <- tf$gather(EST_COEFFICIENTS_tf_dag, indices = inter_indices_i0, axis=0L)
        Qhat_dag <- glm_outcome_transform( EST_INTERCEPT_tf_dag +
                                             tf$matmul(tf$transpose(main_coef_dag), pi_star_ast - pi_star_dag ) +
                                             tf$reduce_sum(tf$multiply(inter_coef_dag, pi_ast_prod - pi_dag_prod ), keepdims=T))
        # Pr(Win D_c Among All | R_c Opp) = Pr(Win D_c Among All | R_c Opp, R voters) Pr(R voters) +
        #Pr(Win D_c Among All | R_c Opp, D voters) Pr(D voters) +
        #Pr(Win D_c Among All | R_c Opp, I voters) Pr(I voters)
        Qhat_population <- Qhat*(1-DagProp) + Qhat_dag*DagProp
      }
      return( tf$concat(list(Qhat_population, Qhat, Qhat_dag),0L)  )
    }
    getQStar_diff_R <- paste(deparse(getQStar_diff_R),collapse="\n")
    getQStar_diff_R <- gsub(getQStar_diff_R, pattern = "SINGLE_PROP_KEY", replace = sprintf("T == !%s",UseSinglePop))
    getQStar_diff_R <- eval( parse( text = getQStar_diff_R ),envir = evaluation_environment )

    # specifications for case
    if(UseSinglePop){ getQStar_diff_R_multi <- getQStar_diff_R; name_ <-"Single" }
    if(!UseSinglePop){ getQStar_diff_R <- getQStar_diff_R; name_ <-"Multi" }
    eval(parse(text = sprintf("getQStar_diff_R_%sGroup <- getQStar_diff_R",name_)))
    eval(parse(text = sprintf("getQStar_diff_%sGroup <- tf_function( getQStar_diff_R_%sGroup )",name_,name_)))
    eval(parse(text = sprintf("getQStar_diff_%sGroup_conv <-  ( tf2jax$convert_functional(
                                          getQStar_diff_%sGroup,
                                          pi_star_ast = jnp$array(pi_star_value_init_ast),
                                          pi_star_dag = jnp$array(pi_star_value_init_dag),
                                          EST_INTERCEPT_tf_ast = jnp$array(EST_INTERCEPT_tf_ast),
                                          EST_COEFFICIENTS_tf_ast = jnp$array(EST_COEFFICIENTS_tf_ast),
                                          EST_INTERCEPT_tf_dag = jnp$array(EST_INTERCEPT_tf_dag),
                                          EST_COEFFICIENTS_tf_dag = jnp$array(EST_COEFFICIENTS_tf_dag)))",
                              name_,name_)))
  }

  # check work:
  try(getQStar_single( pi_star = pi_star_value_init_ast, EST_INTERCEPT_tf = EST_INTERCEPT_tf, EST_COEFFICIENTS_tf = EST_COEFFICIENTS_tf),T)
  try(getQStar_single_conv( pi_star =  jnp$array(pi_star_value_init_ast) , EST_INTERCEPT_tf = jnp$array(EST_INTERCEPT_tf), EST_COEFFICIENTS_tf = jnp$array(EST_COEFFICIENTS_tf)),T)

  # Pretty Pi function
  {
    length_full_simplex <- length(unique(unlist(w_orig)))
    length_simplex_use <- sum(factor_levels)

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
    main_comp_mat <- sapply(1:length(pi_star_value_loc),function(zer){
      main_comp_mat[pi_star_value_loc[zer],zer] <- 1
      return( main_comp_mat[,zer] ) })
    main_comp_mat <- tf$constant(main_comp_mat,tf$float32)

    shadow_comp_mat <- matrix(0, ncol = n_main_params, nrow = length_simplex_use)
    shadow_comp_mat <- sapply(1:length(pi_star_value_loc_shadow),function(zer){
      shadow_comp_mat[pi_star_value_loc_shadow[zer],zer] <- 1
      return( shadow_comp_mat[,zer] ) })
    shadow_comp_mat <- tf$constant(shadow_comp_mat,tf$float32)
    }

    split_vec_full <- unlist(sapply(1:length(factor_levels),function(xz){
      rep(xz,times=factor_levels[xz])} ))

    split_vec_use <- ifelse(ParameterizationType == "Implicit",
                            yes = list(split_vec), no = list(split_vec_full))[[1]]
  }
  getPrettyPi <- tf_function(function(pi_star_value){
    # NB: NO RENORMALIZATION IS DONE
    if(ParameterizationType == "Full"){
      #pi_star_full <- tapply(1:length(d_locator_full),d_locator_full,function(zer){tf$gather(pi_star_value,n2int(ai(zer-1L))) })
      pi_star_full <- pi_star_value
    }
    if(ParameterizationType == "Implicit"){
      pi_star_impliedTerms <- tapply(1:length(d_locator),d_locator,function(zer){
        pi_implied <- OneTf - tf$reduce_sum(tf$gather(pi_star_value,
                                                      n2int(ai(zer-1L)),0L),keepdims=T) })
      names(pi_star_impliedTerms) <- NULL
      pi_star_impliedTerms <- tf$concat(pi_star_impliedTerms,0L)

      pi_star_full <- tf$add(tf$matmul(main_comp_mat,pi_star_value),
                             tf$matmul(shadow_comp_mat,pi_star_impliedTerms))
    }

    return( pi_star_full )
  })
  getPrettyPi(pi_star_value_init_ast)
  getPrettyPi_conv <- tf2jax$convert_functional(getPrettyPi,jnp$array(pi_star_value_init_ast))
  getPrettyPi_conv_diff <- ifelse(ParameterizationType=="Implicit",
                                  yes = list(getPrettyPi_conv),
                                  no = list(jax$jit(function(x){x})))[[1]]
  getPrettyPi_conv_diff(pi_star_value_init_ast)

  a2Simplex_conv <- try(tf2jax$convert_functional(a2Simplex,jnp$array(a_vec_init_ast)),T)
  a2FullSimplex_conv <- try(tf2jax$convert_functional(a2FullSimplex,jnp$array(a_vec_init_ast)),T)
  a2Simplex_conv_diff_use <- ifelse(ParameterizationType == "Implicit",
                               yes = list(a2Simplex_conv),
                               no = list(a2FullSimplex_conv))[[1]]
  GetInvLR_conv <- tf2jax$convert_functional(GetInvLR,jnp$array(1.), a_vec_init_ast)
  GetUpdatedParameters_conv <- tf2jax$convert_functional(GetUpdatedParameters,
                                                         a_vec_init_ast,a_vec_init_ast, jnp$array(1.))

  ## get exact result
  pi_star_exact <- -10
  if(OptimType %in% c("tryboth") & diff == F){
    pi_star_exact <- as.numeric(getPrettyPi(getPiStar_exact()))
  }

  use_gd <- any(pi_star_exact<0) | any(pi_star_exact>1)  |
    (abs(sum(pi_star_exact) - sum(unlist(p_list_full))) > 1e-5)
  use_exact <- !use_gd
  if( use_gd ){
  if(!diff){
    getFixedEntries <- tf_function(function(EST_COEFFICIENTS_tf){
        main_coef <- tf$gather(EST_COEFFICIENTS_tf, indices = main_indices_i0, axis=0L)

        if(T == T){
        inter_coef <- tf$gather(EST_COEFFICIENTS_tf, indices = inter_indices_i0, axis=0L)
        # term 2 fix contribution
        term2_FC <- sapply(1:n_main_params,function(main_comp){
          interaction_info_red <- interaction_info[
            (ind1<-(interaction_info$d_adj %in% main_info[main_comp,]$d_adj &
                      interaction_info$l %in% main_info[main_comp,]$l)) |
              (ind2<-(interaction_info$dp_adj %in% main_info[main_comp,]$d_adj &
                        interaction_info$lp %in% main_info[main_comp,]$l ) ),]
          id_d <- apply(interaction_info_red[,c("d_adj","l")],1,function(zer){paste(zer,collapse="_")})
          id_dp <- apply(interaction_info_red[,c("dp_adj","lp")],1,function(zer){paste(zer,collapse="_")})
          id_ <- ifelse(!(interaction_info_red$d_adj %in% main_info[main_comp,]$d_adj),
                        yes = id_d, no = id_dp)
          id_main <- apply(main_info[,c("d_adj","l")],1,function(zer){paste(zer,collapse="_")})
          which_inter <- which(ind1|ind2)
          inter_into_main_0i <- n2int(ai(sapply(id_,function(zer){which(id_main %in% zer)})-1L))

          if(nrow(interaction_info_red)>0){
            inter_coef_ <- tf$gather(inter_coef,indices = n2int(ai(which_inter-1L)), axis = 0L)
          }

          # expand dimensions in the length == 1 case
          if(length(which_inter) == 1){
            inter_coef_ <- tf$expand_dims(inter_coef_,0L)
            inter_into_main <- tf$expand_dims(inter_into_main_0i,0L)
          }
          return( list("inter_coef_" = inter_coef_,
                       "indices_on_a_simplex_for_inter_prob" = inter_into_main_0i) )
        })
        for(jf in 1:length(term2_FC[1,])){
          eval(parse(text=sprintf("term2_FC_a%s <- term2_FC[1,][[jf]]",jf)))
          eval(parse(text=sprintf("term2_FC_b%s <- term2_FC[2,][[jf]]",jf)))
        }

        # term 4 fix contribution
        term4_FC <- sapply(1:n_main_params,function(main_comp){
          which_d <- which(d_locator[main_comp] == d_locator)
          sum_p <- tf$expand_dims(tf$reduce_sum(tf$gather(p_vec_tf,
                                                          indices = n2int(ai(which_d-1L)), axis=0L),keepdims=F),0L)
          return(   list("sum_p"=sum_p,
                         "indices_for_sum_pi"=n2int(as.matrix(ai(which_d-1L))))  )
        })
        term4_FC_a <- tf$concat(term4_FC[1,],0L)
        for(jf in 1:length(term4_FC[2,])){
          eval(parse(text = sprintf("term4_FC_b%s = term4_FC[2,][[jf]]",jf)))
        }
        add_text <- c(paste("term2_FC_a",1:n_main_params,sep=""),
                      paste("term2_FC_b",1:n_main_params,sep=""),
                      paste("term4_FC_b",1:n_main_params,sep=""))
        add_text <- sapply(add_text,function(zer){sprintf("'%s'=%s",zer,zer)})
        add_text <- paste(add_text,collapse=",")
        eval(parse(text = sprintf("l_res <- list(
                      'main_coef'=main_coef,
                      'inter_coef'=inter_coef,
                      'term4_FC_a'=term4_FC_a,
                      %s)",add_text)))
        }
        return( l_res )
      })
    getFixedEntries_conv <- tf2jax$convert_functional(getFixedEntries,jnp$array(EST_COEFFICIENTS_tf))
    fe <- getFixedEntries( EST_COEFFICIENTS_tf ) # needed for function initialization
    getFixedEntries_conv( jnp$array(EST_COEFFICIENTS_tf) ) # needed for function initialization
  }

  # define GD function
  #REGRESSION_PARAMS_jax_dag <- REGRESSION_PARAMS_jax <- jnp$array(tf$concat(list(EST_INTERCEPT_tf, EST_COEFFICIENTS_tf),0L))
  #if(MaxMin & MaxMinType == "OneRoundSingle"){
    #EST_INTERCEPT_tf_dag <- EST_INTERCEPT_tf;EST_COEFFICIENTS_tf_dag <- EST_COEFFICIENTS_tf}
  #if(MaxMin & MaxMinType == "OneRoundDouble"){
    #REGRESSION_PARAMS_jax_ast <- jnp$array(tf$concat(list(EST_INTERCEPT_tf_ast,EST_COEFFICIENTS_tf_ast),0L))
    #REGRESSION_PARAMS_jax_dag <- jnp$array(tf$concat(list(EST_INTERCEPT_tf_dag,EST_COEFFICIENTS_tf_dag),0L)) }
  gather_conv <- tf2jax$convert_functional(tf_function(function(x){
      INTERCEPT_ <- tf$expand_dims(tf$gather(x,0L),1L)
      COEFFICIENTS_ <- tf$gather(x,ai(1L:(length(x)-1L)))
      list(INTERCEPT_, COEFFICIENTS_)}), x = REGRESSION_PARAMS_jax)
  if(diff == F){
    initialize_GD_WithExactGradients <- paste(deparse(generate_GD_WithExactGradients),collapse="\n")
    initialize_GD_WithExactGradients <- gsub(initialize_GD_WithExactGradients,pattern="function \\(\\)",replace="")
    eval( parse( text = initialize_GD_WithExactGradients ),envir = evaluation_environment )
  }

  p_vec_jnp <- jnp$array(   as.matrix(p_vec)   )
  p_vec_full_jnp <- jnp$array( as.matrix( p_vec_full ) )
  lambda_jnp <-  jnp$array(  lambda  )
  {
      UseOptax <- F
      # model partition + setup state
      optax_optimizer_ast <-  optax$chain(
        #optax$scale(0.01),optax$scale_by_adam() #optax$adam( learning_rate = LR_schedule, eps_root = 1e-6)
        optax$scale(1),optax$scale_by_rss(initial_accumulator_value = 0.001)  )
      opt_state_ast <- optax_optimizer_ast$init(jnp$array( a_vec_init_ast ))
      jit_apply_updates_ast <- jax$jit(optax$apply_updates)
      jit_update_ast <- jax$jit(optax_optimizer_ast$update)

      # model partition + setup state
      optax_optimizer_jax <-  optax$chain(
        #optax$scale(0.01),optax$scale_by_adam() #optax$adam( learning_rate = LR_schedule, eps_root = 1e-6)
        optax$scale(1),optax$scale_by_rss(initial_accumulator_value = 0.001)  )
      opt_state_dag <- optax_optimizer_jax$init(jnp$array( a_vec_init_dag ))
      jit_apply_updates_dag <- jax$jit(optax$apply_updates)
      jit_update_dag <- jax$jit(optax_optimizer_jax$update)
  }

    print("Defining gd function...")
  {
    getPiStar_gd <-  function(REGRESSION_PARAMETERS_ast,
                              REGRESSION_PARAMETERS_dag,
                              REGRESSION_PARAMETERS_ast0,
                              REGRESSION_PARAMETERS_dag0,
                              P_VEC_FULL_ast,
                              P_VEC_FULL_dag,
                              LAMBDA){
        REGRESSION_PARAMETERS_ast <- gather_conv(REGRESSION_PARAMETERS_ast)
        INTERCEPT_ast_ <- REGRESSION_PARAMETERS_ast[[1]]
        COEFFICIENTS_ast_ <- REGRESSION_PARAMETERS_ast[[2]]

        INTERCEPT_dag0_ <- INTERCEPT_ast0_ <- INTERCEPT_dag_ <- INTERCEPT_ast_
        COEFFICIENTS_dag0_ <- COEFFICIENTS_ast0_ <- COEFFICIENTS_dag_ <- COEFFICIENTS_ast_
        if( MaxMin ){
          REGRESSION_PARAMETERS_dag <- gather_conv(REGRESSION_PARAMETERS_dag)
          INTERCEPT_dag_ <- REGRESSION_PARAMETERS_dag[[1]]
          COEFFICIENTS_dag_ <- REGRESSION_PARAMETERS_dag[[2]]
        }
        if(nRounds > 1){
          REGRESSION_PARAMETERS_ast0 <- gather_conv(REGRESSION_PARAMETERS_ast0)
          INTERCEPT_ast0_ <- REGRESSION_PARAMETERS_ast0[[1]]
          COEFFICIENTS_ast0_ <- REGRESSION_PARAMETERS_ast0[[2]]

          REGRESSION_PARAMETERS_dag0 <- gather_conv(REGRESSION_PARAMETERS_dag0)
          INTERCEPT_dag0_ <- REGRESSION_PARAMETERS_dag0[[1]]
          COEFFICIENTS_dag0_ <- REGRESSION_PARAMETERS_dag0[[2]]
        }

        a_i_ast <- jnp$array( a_vec_init_ast )
        a_i_dag <- jnp$array( a_vec_init_dag )

        # gradient descent iterations
        grad_mag_dag_vec <<- grad_mag_ast_vec <<- rep(NA, times = nSGD)
        goOn <- F; i<-0; maxIter<-nSGD;
        while(goOn == F){
          i<-i+1;
          if(i == 1){
            {
              FullGetQStar_ <- jax$jit(  function(a_i_ast,
                                                      a_i_dag,
                                                      INTERCEPT_ast_, COEFFICIENTS_ast_,
                                                      INTERCEPT_dag_, COEFFICIENTS_dag_,
                                                      INTERCEPT_ast0_, COEFFICIENTS_ast0_,
                                                      INTERCEPT_dag0_, COEFFICIENTS_dag0_,
                                                      P_VEC_FULL_ast_,
                                                      P_VEC_FULL_dag_,
                                                      LAMBDA_,Q_SIGN){
                pi_star_full_i_ast <- getPrettyPi_conv_diff( pi_star_i_ast <- a2Simplex_conv_diff_use( a_i_ast ))
                pi_star_full_i_dag <- getPrettyPi_conv_diff( pi_star_i_dag <- a2Simplex_conv_diff_use( a_i_dag ))

                if(diff){
                  q__ <- ifelse(MaxMin,
                         yes = list(getQStar_diff_MultiGroup_conv),
                         no = list(getQStar_diff_SingleGroup_conv))[[1]](
                    pi_star_ast =  pi_star_i_ast,
                    pi_star_dag = pi_star_i_dag,
                    EST_INTERCEPT_tf_ast = INTERCEPT_ast_,
                    EST_COEFFICIENTS_tf_ast = COEFFICIENTS_ast_,
                    EST_INTERCEPT_tf_dag = INTERCEPT_dag_,
                    EST_COEFFICIENTS_tf_dag = COEFFICIENTS_dag_)
                  q_max <- q__ <- jnp$take(q__,0L)
                }
                if(!diff){
                  q__ <- getQStar_single_conv(pi_star = pi_star_i_ast,
                                    EST_INTERCEPT_tf = INTERCEPT_ast_,
                                    EST_COEFFICIENTS_tf = COEFFICIENTS_ast_)
                  q_max <- q__ <- jnp$take(q__,0L)
                }

                if(MaxMin){
                q_ast <- q__
                q_dag <- jnp$subtract(jnp$array(1),q__)
                cond_UseDag <- jnp$multiply(jnp$array(0.5),jnp$subtract(jnp$array(1.), Q_SIGN))
                cond_UseAst <- jnp$multiply(jnp$array(0.5),jnp$add(jnp$array(1.), Q_SIGN))

                q_max <- jnp$add(
                  jnp$multiply(cond_UseDag, q_dag),
                  jnp$multiply(cond_UseAst, q_ast)
                )
                if(nRounds > 1){
                  q0_ast <- getQStar_diff_SingleGroup_conv(
                                pi_star_ast =  pi_star_i_ast,
                                pi_star_dag = p_vec_jnp,
                                EST_INTERCEPT_tf_ast = INTERCEPT_ast0_,
                                EST_COEFFICIENTS_tf_ast = COEFFICIENTS_ast0_,
                                EST_INTERCEPT_tf_dag = INTERCEPT_ast0_,
                                EST_COEFFICIENTS_tf_dag = COEFFICIENTS_ast0_)
                  q0_dag <- getQStar_diff_SingleGroup_conv(
                                      pi_star_ast = pi_star_i_dag,
                                      pi_star_dag = p_vec_jnp,
                                      EST_INTERCEPT_tf_ast = INTERCEPT_dag0_,
                                      EST_COEFFICIENTS_tf_ast = COEFFICIENTS_dag0_,
                                      EST_INTERCEPT_tf_dag = INTERCEPT_dag0_,
                                      EST_COEFFICIENTS_tf_dag = COEFFICIENTS_dag0_)
                  q0_dag <- jnp$take(q0_dag,0L)
                  q0_ast <- jnp$take(q0_ast,0L)
                  q0 <- jnp$add(
                    jnp$multiply(cond_UseDag, q0_dag),
                    jnp$multiply(cond_UseAst, q0_ast)
                  )
                  # one utility function - leads to same strategy
                  #q_max <- q0

                  #q_max <- q_max

                  # maximize product
                  #q_max <- jnp$multiply(q_max,q0)

                  # find best profile for base that reaches 0.5 general threshold
                  q_max <- jnp$multiply( jax$nn$softplus(jnp$subtract(q_max,jnp$array(0.5))), q0)

                  # find best profile for general that reaches 0.5 primary threshold
                  #q_max <- jnp$multiply( jax$nn$softplus(jnp$subtract(q_max,jnp$array(0.5))), q0)
                }
                }
                if(TypePen == "L1"){
                  var_pen_ast__ <- jnp$multiply(LAMBDA_, jnp$negative(jnp$sum(jnp$abs(  jnp$subtract(  pi_star_full_i_ast, P_VEC_FULL_ast_ )  ))))
                  var_pen_dag__ <- jnp$multiply(LAMBDA_, jnp$negative(jnp$sum(jnp$abs(  jnp$subtract(  pi_star_full_i_dag, P_VEC_FULL_dag_ )  ))))
                }
                if(TypePen == "L2"){
                  var_pen_ast__ <- jnp$multiply(LAMBDA_,jnp$negative(jnp$sum(jnp$square(  jnp$subtract(  pi_star_full_i_ast, P_VEC_FULL_ast_ )  ))))
                  var_pen_dag__ <- jnp$multiply(LAMBDA_,jnp$negative(jnp$sum(jnp$square(  jnp$subtract(  pi_star_full_i_dag, P_VEC_FULL_dag_ )  ))))
                }
                if(TypePen == "LInfinity"){
                  var_pen_ast__ <- tapply(1:length(split_vec_full), split_vec_full, function(zer){
                    list( jnp$max(jnp$take(pi_star_full_i_ast, indices = jnp$array( as.integer(zer-1L)),axis = 0L)))})
                  names(var_pen_ast__)<-NULL ; var_pen_ast__ <- jnp$sum( jnp$stack(var_pen_ast__))

                  var_pen_dag__ <- tapply(1:length(split_vec_full), split_vec_full, function(zer){
                    list( jnp$max(jnp$take(pi_star_full_i_dag, indices = jnp$array( as.integer(zer-1L)),axis = 0L)))})
                  names(var_pen_dag__)<-NULL ; var_pen_dag__ <- jnp$sum( jnp$stack(var_pen_dag__))

                  var_pen_ast__ <- jnp$multiply(LAMBDA_,jnp$negative(var_pen_ast__))
                  var_pen_dag__ <- jnp$multiply(LAMBDA_,jnp$negative(var_pen_dag__))
                }
                if(TypePen == "KL"){
                  var_pen_ast__ <- jnp$multiply(LAMBDA_,jnp$negative(jnp$sum(jnp$multiply(P_VEC_FULL_ast_, jnp$subtract(jnp$log(P_VEC_FULL_ast_),jnp$log(pi_star_full_i_ast))))))
                  var_pen_dag__ <- jnp$multiply(LAMBDA_,jnp$negative(jnp$sum(jnp$multiply(P_VEC_FULL_dag_, jnp$subtract(jnp$log(P_VEC_FULL_dag_),jnp$log(pi_star_full_i_dag))))))
                }

                ret_ <- jnp$add( jnp$add( q_max, var_pen_ast__), var_pen_dag__)
                return( ret_ )
              } )
              dQ_da_ast <- jax$jit(jax$grad(FullGetQStar_, argnums = jnp$array(0L)))
              dQ_da_dag <- jax$jit(jax$grad(FullGetQStar_, argnums = jnp$array(1L)))
            }

            FullGetQStar_(a_i_ast = a_i_ast, a_i_dag = a_i_dag,
                           INTERCEPT_ast_ = INTERCEPT_ast_, COEFFICIENTS_ast_ = COEFFICIENTS_ast_,
                           INTERCEPT_dag_ = INTERCEPT_dag_, COEFFICIENTS_dag_ = COEFFICIENTS_dag_,
                           INTERCEPT_ast0_ = INTERCEPT_ast0_, COEFFICIENTS_ast0_ = COEFFICIENTS_ast0_,
                           INTERCEPT_dag0_ = INTERCEPT_dag0_, COEFFICIENTS_dag0_ = COEFFICIENTS_dag0_,
                           P_VEC_FULL_ast_ = P_VEC_FULL_ast, P_VEC_FULL_dag_ = P_VEC_FULL_dag,
                           LAMBDA_ = LAMBDA, Q_SIGN = jnp$array(1.))
            FullGetQStar_(a_i_ast = a_i_ast, a_i_dag = a_i_dag,
                              INTERCEPT_ast_ = INTERCEPT_ast_, COEFFICIENTS_ast_ = COEFFICIENTS_ast_,
                              INTERCEPT_dag_ = INTERCEPT_dag_, COEFFICIENTS_dag_ = COEFFICIENTS_dag_,
                              INTERCEPT_ast0_ = INTERCEPT_ast0_, COEFFICIENTS_ast0_ = COEFFICIENTS_ast0_,
                              INTERCEPT_dag0_ = INTERCEPT_dag0_, COEFFICIENTS_dag0_ = COEFFICIENTS_dag0_,
                              P_VEC_FULL_ast_ = P_VEC_FULL_ast, P_VEC_FULL_dag_ = P_VEC_FULL_dag,
                              LAMBDA_ = LAMBDA, Q_SIGN = jnp$array(-1.))

            init_dQ_da_ast <- dQ_da_ast(
                                a_i_ast, a_i_dag,
                                INTERCEPT_ast_,  COEFFICIENTS_ast_,
                                INTERCEPT_dag_,  COEFFICIENTS_dag_,
                                INTERCEPT_ast0_, COEFFICIENTS_ast0_,
                                INTERCEPT_dag0_, COEFFICIENTS_dag0_,
                                P_VEC_FULL_ast, P_VEC_FULL_dag,
                                LAMBDA, jnp$array(1.) )
            inv_learning_rate_da_ast <- jnp$maximum(jnp$array(0.001), jnp$multiply(10, jnp$square(jnp$linalg$norm(init_dQ_da_ast))))

            init_dQ_da_dag <- dQ_da_dag( a_i_ast, a_i_dag,
                                         INTERCEPT_ast_,  COEFFICIENTS_ast_,
                                         INTERCEPT_dag_,  COEFFICIENTS_dag_,
                                         INTERCEPT_ast0_, COEFFICIENTS_ast0_,
                                         INTERCEPT_dag0_, COEFFICIENTS_dag0_,
                                         P_VEC_FULL_ast, P_VEC_FULL_dag,
                                         LAMBDA, jnp$array(-1.) )
            inv_learning_rate_da_dag <- jnp$maximum(jnp$array(0.001), jnp$multiply(10,  jnp$square(jnp$linalg$norm( init_dQ_da_dag ))))
            #plot(init_dQ_da_ast$to_py(),init_dQ_da_dag$to_py());abline(a=0,b=1)
            #plot(init_dQ_da_ast$to_py()-init_dQ_da_dag$to_py());abline(h = 0)
          }

          # da_dag updates (min step)
          if( i %% 1 == 0 & MaxMin ){
            grad_i_dag <- dQ_da_dag(  a_i_ast, a_i_dag,
                                      INTERCEPT_ast_,  COEFFICIENTS_ast_,
                                      INTERCEPT_dag_,  COEFFICIENTS_dag_,
                                      INTERCEPT_ast0_, COEFFICIENTS_ast0_,
                                      INTERCEPT_dag0_, COEFFICIENTS_dag0_,
                                      P_VEC_FULL_ast, P_VEC_FULL_dag,
                                      LAMBDA, jnp$array(-1.) )
            if(!UseOptax){
              inv_learning_rate_da_dag <-  jax$lax$stop_gradient(GetInvLR_conv(inv_learning_rate_da_dag, grad_i_dag))
              a_i_dag <- GetUpdatedParameters_conv(a_vec = a_i_dag, grad_i = grad_i_dag,
                                                   inv_learning_rate_i = jnp$sqrt(inv_learning_rate_da_dag))
          }

            if(UseOptax){
              updates_and_opt_state_dag <- jit_update_dag( updates = grad_i_dag, state = opt_state_dag, params = a_i_dag)
              opt_state_dag <- updates_and_opt_state_dag[[2]]
              a_i_dag <- jit_apply_updates_dag(params = grad_i_dag, updates = updates_and_opt_state_dag[[1]])
            }

            grad_mag_dag_vec[i] <<- list(jnp$linalg$norm( grad_i_dag ))
          }

          # da updates (max step)
          if( i %% 1 == 0 | (!MaxMin) ){
            grad_i_ast <- dQ_da_ast( a_i_ast, a_i_dag,
                                     INTERCEPT_ast_,  COEFFICIENTS_ast_,
                                     INTERCEPT_dag_,  COEFFICIENTS_dag_,
                                     INTERCEPT_ast0_, COEFFICIENTS_ast0_,
                                     INTERCEPT_dag0_, COEFFICIENTS_dag0_,
                                     P_VEC_FULL_ast, P_VEC_FULL_dag,
                                     LAMBDA, jnp$array(1.) )
            if(!UseOptax){
              inv_learning_rate_da_ast <-  jax$lax$stop_gradient( GetInvLR_conv(inv_learning_rate_da_ast, grad_i_ast) )
              a_i_ast <- GetUpdatedParameters_conv(a_vec = a_i_ast, grad_i = grad_i_ast,
                                inv_learning_rate_i = jnp$sqrt(inv_learning_rate_da_ast))
            }

            #plot(grad_i$to_py(),grad_i_dag$to_py());abline(a=0,b=1)
            if(UseOptax){
              updates_and_opt_state_ast <- jit_update_ast( updates = grad_i_ast, state = opt_state_ast, params = a_i_ast)
              opt_state_ast <- updates_and_opt_state_ast[[2]]
              a_i_ast <- jit_apply_updates_ast(params = grad_i_ast, updates = updates_and_opt_state_ast[[1]])
            }

            grad_mag_ast_vec[i] <<- list( jnp$linalg$norm( grad_i_ast ) )
          }

          if(i >= maxIter){goOn <- T}
        }

        # save output
        {
          pi_star_ast_full_simplex_ <- getPrettyPi_conv( pi_star_ast_ <- a2Simplex_conv_diff_use( a_i_ast ) )
          pi_star_dag_full_simplex_ <- getPrettyPi_conv( pi_star_dag_ <- a2Simplex_conv_diff_use( a_i_dag ))
          if(diff){
            q_star_ <- ifelse(MaxMin,
                            yes = list(getQStar_diff_MultiGroup_conv),
                            no = list(getQStar_diff_SingleGroup_conv))[[1]](
                                   pi_star_ast = pi_star_ast_,
                                   pi_star_dag = pi_star_dag_,
                                   EST_INTERCEPT_tf_ast = INTERCEPT_ast_, EST_COEFFICIENTS_tf_ast = COEFFICIENTS_ast_,
                                   EST_INTERCEPT_tf_dag = INTERCEPT_dag_, EST_COEFFICIENTS_tf_dag = COEFFICIENTS_dag_)
          }
          if(!diff){
            q_star_ <- getQStar_single_conv(pi_star = pi_star_ast_,
                            EST_INTERCEPT_tf = INTERCEPT_ast_,
                            EST_COEFFICIENTS_tf = COEFFICIENTS_ast_)
          }
          if( gd_full_simplex == T){ ret_array <- jnp$concatenate(list( q_star_, pi_star_ast_full_simplex_, pi_star_dag_full_simplex_ ) ) }
          if( gd_full_simplex == F){ ret_array <- jnp$concatenate(list( q_star_, pi_star_ast_, pi_star_dag_ ) ) }
          # plot(a_i_dag$to_py(),a_i$to_py());abline(a=0,b=1)
          # plot(pi_star_full_simplex_$to_py(),pi_star_dag_full_simplex_$to_py());abline(a=0,b=1)
          return( ret_array  ) # ret_array$shape
        }
    }
  }

  # get initial learning rate for gd result
  #if(!diff){nSGD <- 1; inv_learning_rate <- tf$constant(getPiStar_gd(REGRESSION_PARAMETERS = REGRESSION_PARAMS_jax),tf$float32)}
  nSGD <- nSGD_orig
  }

  # Obtain solution via exact calculation
  print("Starting optimization...")
  if(use_exact){
    results_vec_list <- replicate(length(unlist(p_list_full))+1,list()) # + 1 for intercept
    dx_vars <- list( EST_INTERCEPT_tf, EST_COEFFICIENTS_tf )
    with(tf$GradientTape(persistent = F) %as% tape, {
      tape$watch(  dx_vars   )
      pi_star_full_exact <- pi_star_full <- getPrettyPi( pi_star_reduced <- getPiStar_exact())
      q_star_exact <- q_star <- getQStar_single(pi_star = pi_star_reduced,
                                         EST_INTERCEPT_tf = EST_INTERCEPT_tf,
                                         EST_COEFFICIENTS_tf = EST_COEFFICIENTS_tf)
      results_vec <- tf$concat(list(q_star, pi_star_full),0L)
      for(ia in 1:length(results_vec_list)){
        results_vec_list[[ia]] <- tf$reshape(tf$gather(results_vec,n2int(ai(ia-1L)),axis=0L),shape=list())
      }
    })

    # automatic jacobian from tf
    {
    jacobian_time <- system.time(jacobian_mat <- tape$jacobian(results_vec, dx_vars ))
    jacobian_mat_exact <- jacobian_mat <- cbind(
                          as.matrix(tf$squeeze(tf$squeeze(tf$squeeze(jacobian_mat[[1]],1L),1L))),
                          as.matrix(tf$squeeze(tf$squeeze(tf$squeeze(jacobian_mat[[2]],1L),2L)))
                        )
    vcov_OutcomeModel_concat <- vcov_OutcomeModel_ast
    }
  }

  if(use_gd){
    # perform main gd runs + inference
    nSGD <- nSGD_orig

    # first do ave case analysis
    p_vec_full_ast_jnp <- p_vec_full_dag_jnp <- p_vec_full_jnp
    if(T == F){
      p_vec_full_ast_jnp <- p_vec_full_dag_jnp <- p_vec_full_jnp
      MaxMin <- F; gd_full_simplex <- T
      q_with_pi_star_full_ast_ <- getPiStar_gd(
                                           REGRESSION_PARAMETERS_ast = REGRESSION_PARAMS_jax_ast,
                                           REGRESSION_PARAMETERS_dag = REGRESSION_PARAMS_jax_ast,
                                           REGRESSION_PARAMETERS_ast0 = REGRESSION_PARAMS_jax_ast,
                                           REGRESSION_PARAMETERS_dag0 = REGRESSION_PARAMS_jax_ast,
                                           P_VEC_FULL_ast = p_vec_full_ast_jnp,
                                           P_VEC_FULL_dag = p_vec_full_ast_jnp,
                                           LAMBDA = lambda_jnp)$to_py()
      q_with_pi_star_full_dag_ <- getPiStar_gd(
                                           REGRESSION_PARAMETERS_ast = REGRESSION_PARAMS_jax_dag,
                                           REGRESSION_PARAMETERS_dag = REGRESSION_PARAMS_jax_dag,
                                           REGRESSION_PARAMETERS_ast0 = REGRESSION_PARAMS_jax_dag,
                                           REGRESSION_PARAMETERS_dag0 = REGRESSION_PARAMS_jax_dag,
                                           P_VEC_FULL_ast = p_vec_full_dag_jnp,
                                           P_VEC_FULL_dag = p_vec_full_dag_jnp,
                                           LAMBDA = lambda_jnp)$to_py()
      q_ave <- q_with_pi_star_full_ast_[1]; q_dag_ave <- q_with_pi_star_full_dag_[1]
      MaxMin <- T; pi_star_ave <- list()
      pi_star_ave$k1 <- split(p_vec_full_ast <- q_with_pi_star_full_ast_[-c(1:3)][c(1:length(p_vec_full))], split_vec_use)
      pi_star_ave$k2 <- split(p_vec_full_dag <- q_with_pi_star_full_dag_[-c(1:3)][c(1:length(p_vec_full))], split_vec_use)
      pi_star_ave <- RenamePiList( RejiggerPi(pi_ = pi_star_ave, isSE = F  ) )
      p_vec_full_ast_jnp <- jnp$array( as.matrix( p_vec_full_ast ) )
      p_vec_full_dag_jnp <- jnp$array( as.matrix( p_vec_full_dag ) )
      # p_vec_full_jnp$shape; p_vec_full_jnp$shape
      # plot(p_vec_full_dag,p_vec_full_ast);abline(a=0,b=1)
    }

    # do true iterative optimization
    LAMBDA_ITERATIVE <- lambda_jnp
    #LAMBDA_ITERATIVE <- jnp$array(0.1)
    # optimize q for pi* then for pi_dag and so forth, change iterative
    #plot( REGRESSION_PARAMS_jax$to_py(),REGRESSION_PARAMS_jax_dag$to_py() );abline(a=0,b=1)
    #plot(p_vec_full_dot_jnp$to_py(), p_vec_full_dag_jnp$to_py());abline(a=0,b=1)
    #MaxMin <- T;
    gd_full_simplex <- T
    gd_time <- system.time( q_with_pi_star_full <- getPiStar_gd(
                             REGRESSION_PARAMETERS_ast = REGRESSION_PARAMS_jax_ast,
                             REGRESSION_PARAMETERS_dag = REGRESSION_PARAMS_jax_dag,
                             REGRESSION_PARAMETERS_ast0 = REGRESSION_PARAMS_jax_ast0,
                             REGRESSION_PARAMETERS_dag0 = REGRESSION_PARAMS_jax_dag0,
                             P_VEC_FULL_ast = p_vec_full_ast_jnp,
                             P_VEC_FULL_dag = p_vec_full_dag_jnp,
                             LAMBDA = LAMBDA_ITERATIVE) )
    q_with_pi_star_full <- tf$constant(q_with_pi_star_full, tf$float32)
    # as.matrix(q_with_pi_star_full)[1:3]
    print("Time GD: ");print(gd_time)

    grad_mag_ast_vec <- unlist(  lapply(grad_mag_ast_vec,function(zer){
      ifelse(is.na(zer),no = as.numeric(tf$sqrt( tf$reduce_sum(tf$square(tf$constant(zer,tf$float32))) )), yes = NA) }) )
    try(plot( grad_mag_ast_vec , main = "Gradient Magnitude Evolution (ast)"),T)
    grad_mag_dag_vec <- try(unlist(  lapply(grad_mag_dag_vec,function(zer){
      ifelse(is.na(zer),no=as.numeric(tf$sqrt( tf$reduce_sum(tf$square(tf$constant(zer,tf$float32))) )),yes=NA) }) ),T)
    try(plot( grad_mag_dag_vec , main = "Gradient Magnitude Evolution (dag)"),T)

    gd_full_simplex <- F
    pi_star_red <- getPiStar_gd(
                        REGRESSION_PARAMETERS_ast = REGRESSION_PARAMS_jax_ast,
                        REGRESSION_PARAMETERS_dag = REGRESSION_PARAMS_jax_dag,
                        REGRESSION_PARAMETERS_ast0 = REGRESSION_PARAMS_jax_ast0,
                        REGRESSION_PARAMETERS_dag0 = REGRESSION_PARAMS_jax_dag0,
                        P_VEC_FULL_ast = p_vec_full_ast_jnp,
                        P_VEC_FULL_dag = p_vec_full_dag_jnp,
                        LAMBDA = LAMBDA_ITERATIVE)
    pi_star_red <- (pi_star_red$to_py()[-c(1:3),])
    pi_star_red_dag <- jnp$array(as.matrix(pi_star_red[-c(1:(length(pi_star_red)/2))]))
    pi_star_red_ast <- jnp$array(as.matrix(  pi_star_red[1:(length(pi_star_red)/2)] ) )

    q_star_gd <- q_star <- as.numeric(  q_with_pi_star_full )[1]
    pi_star_full_gd <- pi_star_full <- as.numeric( q_with_pi_star_full )[-c(1:3)]
    #plot(pi_star_full_gd[1:(length(pi_star_full_gd)/2)],pi_star_full_gd[-c(1:(length(pi_star_full_gd)/2))]); abline(a=0,b=1)
    #plot(pi_star_full_gd[1:(length(pi_star_full_gd)/2)]-pi_star_full_gd[-c(1:(length(pi_star_full_gd)/2))]); abline(h=0)

    # see https://github.com/google/jax/issues/1696 for debugging help
    jacobian_time <- ""
    jacobian_mat_gd <- jacobian_mat <- matrix(0,
                              ncol = 4*REGRESSION_PARAMS_jax_ast$shape[[1]],
                              nrow = q_with_pi_star_full$shape$as_list()[1])
    diag(jacobian_mat_gd) <- diag(jacobian_mat) <- 1
    vcov_OutcomeModel_concat <- matrix(0,
                                       nrow = nrow(vcov_OutcomeModel_dag)*4,
                                       ncol = nrow(vcov_OutcomeModel_dag)*4)
    if(ComputeSEs){
      gd_full_simplex <- T
      jacobian_time <- system.time(jacobian_mat <-
                          jax$jacobian(getPiStar_gd,0L:3L)(
                                          REGRESSION_PARAMS_jax_ast,
                                          REGRESSION_PARAMS_jax_dag,
                                          REGRESSION_PARAMS_jax_ast0,
                                          REGRESSION_PARAMS_jax_dag0,
                                          p_vec_full_ast_jnp,
                                          p_vec_full_dag_jnp,
                                          LAMBDA_ITERATIVE))
      jacobian_mat_gd <- jacobian_mat <- lapply(jacobian_mat,function(l_){
        as.matrix( tf$squeeze(tf$squeeze(tf$constant(l_,tf$float32),1L),2L) ) })
      jacobian_mat_gd <- jacobian_mat <- do.call(cbind, jacobian_mat)
      vcov_OutcomeModel_concat <- list(
                     vcov_OutcomeModel_ast  ,
                     vcov_OutcomeModel_dag  ,
                     vcov_OutcomeModel_ast0 ,
                     vcov_OutcomeModel_dag0  )
      vcov_OutcomeModel_concat <- Matrix::bdiag( vcov_OutcomeModel_concat )
      vcov_OutcomeModel_concat <- as.matrix(  vcov_OutcomeModel_concat )
      #jacobian_mat_gd <- jacobian_mat <- as.matrix( tf$squeeze(tf$squeeze(tf$constant(jacobian_mat,tf$float32),1L),2L) )
      print("Time Jacobian of GD Solution: ");print(jacobian_time)
    }
    # hist(c(jacobian_mat));image(jacobian_mat)
    # summary(lm(as.numeric(sol_gd)~as.numeric(sol_exact)))
  }

  # print time of jacobian calculation
  # remember, the first three entries of output are:
  # Qhat_population, Qhat, Qhat_dag
  print( jacobian_time )
  vcov_PiStar <- jacobian_mat %*% vcov_OutcomeModel_concat %*% t(jacobian_mat)
  q_star <- as.matrix(   q_star  )
  q_star_se <- sqrt(  diag( vcov_PiStar )[1] )
  pi_star_numeric <- as.numeric( pi_star_full )

  # drop the q part
  if(diff == T){
    pi_star_se <- sqrt(  diag( vcov_PiStar )[-c(1:3)] )
  }
  if(diff == F){
    take_indices <- 1:length( pi_star_numeric )
    if(use_gd){ take_indices <- 1:(length(pi_star_numeric)/2+1)  }
    pi_star_numeric <- pi_star_numeric[take_indices]
    pi_star_se <- sqrt(  diag( vcov_PiStar )[-1][take_indices] )
  }

  # setup pretty pi's
  if( diff == F ){
    pi_star_se_list <- pi_star_list <- list()
    pi_star_list$k1 <- (  split(pi_star_numeric, split_vec) )
    pi_star_se_list$k1 <- (  split(pi_star_se, split_vec) )
  }

  if( diff == T ){
    pi_star_se_list <- pi_star_list <- list()
    pi_star_list$k1 <- split(pi_star_numeric[1:length(p_vec_full)], split_vec_use)
    pi_star_se_list$k1 <- split(pi_star_se[1:length(p_vec_full)], split_vec_use)

    # save jnp for later
    pi_star_vec_jnp <- jnp$array(as.matrix(pi_star_numeric[1:length(p_vec_full)]))
    {
      pi_star_dag_vec_jnp <- jnp$array(as.matrix(pi_star_numeric[-c(1:length(p_vec_full))]))
      pi_star_list$k2 <- split(pi_star_numeric[-c(1:length(p_vec_full))], split_vec_use)
      pi_star_se_list$k2 <- split(pi_star_se[-c(1:length(p_vec_full))], split_vec_use)
      #plot(unlist(pi_star_list$k1),unlist(pi_star_list$k2));abline(a=0,b=1)
    }
    #  plot (pi_star_numeric[1:length(p_vec_full)] - pi_star_numeric[-c(1:length(p_vec_full))] )
  }

  # re-jig to account for regularization
  pi_star_list <- RejiggerPi(pi_ = pi_star_list, isSE = F  )
  pi_star_se_list <- RejiggerPi(pi_ = pi_star_se_list, isSE = T  )

  # checks
  #plot(abs(pi_star_true-as.numeric(1-getPiStar_exact())));abline(a=0,b=1);points(pi_star_se,col="red")
  #plot(abs(pi_star_true-unlist(lapply(pi_star_list,function(zer){zer[-1]}))));abline(a=0,b=1);points(pi_star_se,col="red"); points(pi_star_se,col="red")

  # add names
  pi_star_list <- RenamePiList(  pi_star_list  )
  pi_star_se_list <- RenamePiList(  pi_star_se_list  )

  for(sign_ in c(-1,1)){
    bound_ <- lapply(1:kEst,function(k_){
       l_ <- sapply(1:length(pi_star_list[[k_]]),function(zer){
          ret_ <- list( pi_star_list[[k_]][[zer]] + sign_*abs(qnorm((1-confLevel)/2))*pi_star_se_list[[k_]][[zer]] )
          names(ret_) <- names(pi_star_list[[k_]])[zer]
          return(    ret_   )   })
       return(l_) })
    if(sign_ == -1){ lowerList <- bound_ }
    if(sign_ == 1){ upperList <- bound_ }
  }

  names(upperList) <- names(lowerList) <- paste("k",1:length(lowerList),sep="")

  if(!diff){
    pi_star_red_dag <- pi_star_red_ast <- pi_star_numeric
    pi_star_dag_vec_jnp <- pi_star_vec_jnp <- pi_star_red_ast
    p_vec_full_ast_jnp <- p_vec_full_dag_jnp <- p_vec_full
  }
  return( list(
                  "PiStar_point" = pi_star_list,
                  "PiStar_se" = pi_star_se_list,
                  "Q_point_mEst" = c(q_star),
                  "Q_se_mEst"= q_star_se,
                  "PiStar_vec" = pi_star_numeric,
                  "pi_star_red_ast" = pi_star_red_ast,
                  "pi_star_red_dag" = pi_star_red_dag,
                  "factor_levels" = factor_levels,
                  "PiSEStar_vec" = pi_star_se,
                  "pi_star_ave" = pi_star_ave,
                  "q_ave" = q_ave, "q_dag_ave" = q_dag_ave,
                  "PiStar_lb" = lowerList,
                  "PiStar_ub" = upperList,
                  "Q_point" = c(q_star),
                  "lambda" = lambda,
                  "p_vec_full" = p_vec_full,
                  "regularization_adjust_hash" = regularization_adjust_hash,
                  "p_list" = p_list,

                  # reconstruct q info
                  "Qfxn" = ifelse(MaxMin,
                                  yes = list(getQStar_diff_MultiGroup_conv),
                                  no = list(getQStar_diff_SingleGroup_conv))[[1]],

                  'p_vec_full_ast_jnp' = p_vec_full_ast_jnp,
                  'p_vec_full_dag_jnp' = p_vec_full_dag_jnp,
                  'pi_star_ast_vec_jnp' = pi_star_vec_jnp,
                  'pi_star_dag_vec_jnp' = pi_star_dag_vec_jnp,
                  "EST_INTERCEPT_jnp" = jnp$array(EST_INTERCEPT_tf),
                  "EST_COEFFICIENTS_jnp" = jnp$array(EST_COEFFICIENTS_tf),

                  "vcov_OutcomeModel" = vcov_OutcomeModel,
                  "OptimType" = OptimType,
                  "ForceGaussianFamily" = ForceGaussianFamily,
                  "UsedRegularization" = UsedRegularization,

                  "model" = my_model  ) )
}
