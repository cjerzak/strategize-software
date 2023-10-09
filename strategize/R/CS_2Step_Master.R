#' Implements...
#'
#' @usage
#'
#' OptiConjoint(...)
#'
#' @param x Description
#'
#' @return `z` Description
#' @export
#'
#' @details `OptiConjoint` implements...
#'
#' @examples
#'
#' # Perform analysis
#' OptiConjoint_analysis <- OptiConjoint()
#'
#' print( OptiConjoint_analysis )
#'
#' @export
#'
#' @md

OptiConjoint       <-          function(
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
                                            K = 1,
                                            nSGD = 100,
                                            diff = F, MaxMin = F,
                                            UseRegularization = F,
                                            OpenBrowser = F,
                                            ForceGaussianFamily = F,
                                            A_INIT_SD = 0,
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

  # load in packages - may help memory bugs to load them in thru package
  print("Initializing computational environment...")
  # conda_env <- "tensorflow_m1"; conda_env_required <- T
  {
    if(!is.null(conda_env)){
      try(reticulate::use_condaenv(conda_env,
                                   required = conda_env_required), T)
    }
    JaxKey <- function(int_){ jax$random$PRNGKey(int_)}

    # import computational modules
    jax <- reticulate::import("jax",as="jax")
    oryx <- reticulate::import("oryx")
    jnp <- reticulate::import("jax.numpy")
    np <- reticulate::import("numpy")
    py_gc <- reticulate::import("gc")
    optax <- reticulate::import("optax")

    # setup numerical precision for delta method
    dtj <- jnp$float64
    jax$config$update("jax_enable_x64", T)
  }
  print("Done initializing computational environment!")

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

  nMonte_MaxMin <- as.integer( nMonte_MaxMin )
  q_ave <- q_dag_ave <- pi_star_ave <- NULL
  if(OpenBrowser == T){ browser() }
  nSGD_orig <- nSGD
  ai <- as.integer
  w_orig <- W
  #usedRegularization <- F
  MaxMinType <- "TwoRoundSingle"

  glm_family = "gaussian";
  if(!ForceGaussianFamily){
    if(mean(unique(Y) %in% c(0,1)) == 1){
      glm_family = "binomial";
    }
  }

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

  #initialize_ModelOutcome_FindIt <- paste(deparse(generate_ModelOutcome_FindIt),collapse="\n")
  #initialize_ModelOutcome_FindIt <- gsub(initialize_ModelOutcome_FindIt,pattern="function \\(\\)",replace="")
  #eval( parse( text = initialize_ModelOutcome_FindIt ),envir = evaluation_environment )

  # obtain outcome models
  print("Initializing outcome models...")
  if(T == T){
    if(K > 1 & !UseRegularization){
      warning("K > 1; Forcing regularization...");UseRegularization <- T
    }
    UseRegularization_ORIG <- UseRegularization
    Rounds <- c(0,1)

    nRounds <- GroupsPool <- 1; RoundsPool <- 1
    if(MaxMin){
      nRounds <- 2
      RoundsPool <- c(0,1)
      GroupsPool <- sort(unique(competing_group_variable_candidate))
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
          indi_ <- which( competing_group_variable_respondent == GroupsPool[ GroupCounter ] &
                      ( competing_group_competition_variable_candidate == "Same" &
                          competing_group_variable_candidate == GroupsPool[GroupCounter]) )
        }
        if(Round_ == 1){
          indi_ <- which( competing_group_variable_respondent == GroupsPool[ GroupCounter ] &
                            ( competing_group_competition_variable_candidate == "Different" &
                                competing_group_variable_candidate %in% GroupsPool) )
        }
        DagProp <- prop.table(table(competing_group_variable_respondent[competing_group_variable_respondent %in% GroupsPool]))[2]
      }

      # subset data
      W_ <- W[indi_,]; Y_ <- Y[indi_];
      varcov_cluster_variable_ <- varcov_cluster_variable[indi_]
      pair_id_ <- pair_id[ indi_ ]
      #table(full_dat_$Party.affiliation_clean)
      #table(full_dat_$R_Partisanship)
      #table(table(pair_id_))

      # run models with inputs: W_; Y_; varcov_cluster_variable_;
      initialize_ModelOutcome <- paste(deparse(generate_ModelOutcome),collapse="\n")
      initialize_ModelOutcome <- gsub(initialize_ModelOutcome,pattern="function \\(\\)",replace="")
      eval( parse( text = initialize_ModelOutcome ), envir = evaluation_environment )

      REGRESSION_PARAMS_jax <- jnp$concatenate(list(EST_INTERCEPT_tf, EST_COEFFICIENTS_tf),0L)

      # rename as appropriate
      ret_chunks <- c("vcov_OutcomeModel", "main_info","interaction_info","interaction_info_PreRegularization",
            "REGRESSION_PARAMS_jax","regularization_adjust_hash","main_dat", "my_mean","EST_INTERCEPT_tf","my_model", "EST_COEFFICIENTS_tf")
      dag_condition <- (GroupCounter == 1) | (MaxMin == F)
      round_text <- ifelse( Round_==0, yes="0", no="")
      if( dag_condition ){
          tmp <- sapply(ret_chunks,function(chunk_){ eval(parse(text = sprintf("%s_dag%s <- %s",chunk_,round_text,chunk_)),envir = evaluation_environment) })
          rm(tmp)
      }
      if( !dag_condition ){
          tmp <- sapply(ret_chunks,function(chunk_){ eval(parse(text = sprintf("%s_ast%s <- %s",chunk_,round_text,chunk_)),envir = evaluation_environment) })
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
  print("Done initializing outcome models!...")

  # setup glm transform in tf / jax
  print("Starting optimization sequence...")
  glm_outcome_transform <- function(x){x} # identity function
  if(!ForceGaussianFamily){
    if(mean(unique(Y) %in% c(0,1)) == 1){
      # sigmoid transform
      glm_outcome_transform <- jax$nn$sigmoid

      # identity transform
      #if(diff == T){ glm_outcome_transform <- function(x){x} }
    }
  }

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

  #ParameterizationType <- ifelse( DiffType == "glm" | diff == F, yes = "Implicit", no = "Full")
  ParameterizationType <- ifelse( holdout_indicator == 0, yes = "Full", no = "Implicit")
  n2int <- function(x){  jnp$array(x,jnp$int32)  }
  d_locator <- unlist(sapply(1:length(factor_levels),function(l_d){rep(l_d,times=factor_levels[l_d]-1)}))
  #length(d_locator); length(p_vec)
  p_vec_sum_prime <- unlist(tapply(1:length(p_vec),d_locator,function(er){
    sapply(er,function(re){sum(p_vec[er[!er %in% re]])}) }))
  d_locator_full <- unlist(sapply(1:length(factor_levels),function(l_d){rep(l_d,times=factor_levels[l_d])}))
  p_vec_sum_prime_full <- unlist(tapply(1:length(p_vec_full),d_locator_full,function(er){
    sapply(er,function(re){sum(p_vec_full[er[!er %in% re]])}) }))
  main_indices_i0 <- jnp$array((ai(1:n_main_params-1L)),jnp$int32)
  inter_indices_i0 <- jnp$array((ai(((n_main_params+1):EST_COEFFICIENTS_tf$size)-1L)),jnp$int32)
  if(ParameterizationType == "Implicit"){ p_vec_use <- p_vec; p_vec_sum_prime_use <- p_vec_sum_prime }
  if(ParameterizationType == "Full"){ p_vec_use <- p_vec_full; p_vec_sum_prime_use <- p_vec_sum_prime_full }

  if(OptimType != "gd"){
  if(ParameterizationType == "Implicit"){
    print("Initializing manual exact solution code... (Implicit mode)")
    initialize_ExactSol <- paste(deparse(generate_ExactSol),collapse="\n")
    initialize_ExactSol <- gsub(initialize_ExactSol,pattern="function \\(\\)",replace="")
    eval( parse( text = initialize_ExactSol ),envir = evaluation_environment )
    getPiStar_exact <- generate_ExactSolImplicit
  }
  if(ParameterizationType == "Full"){
    print("Initializing manual exact solution code... (Full mode)")
    initialize_ExactSol <- paste(deparse(generate_ExactSol),collapse="\n")
    initialize_ExactSol <- gsub(initialize_ExactSol,pattern="function \\(\\)",replace="")
    eval( parse( text = initialize_ExactSol ),envir = evaluation_environment )
    getPiStar_exact <- generate_ExactSolExplicit
  }
  }

  # pi in constrained space using gradient ascent
  p_vec_tf <- jnp$array(as.matrix(p_vec_use),dtype = dtj)
  inv_learning_rate <- jnp$array(1., dtype = dtj)
  #compile_fxn <- function(x){x}
  compile_fxn <- function(x){jax$jit(x)}

  # initialize manual gradient updates
  #print("Initializing manual gradient updates...") # Dereciated
  #initialize_ManualDoUpdates <- paste(deparse(generate_ManualDoUpdates),collapse="\n")
  #initialize_ManualDoUpdates <- gsub(initialize_ManualDoUpdates,pattern="function \\(\\)",replace="")
  #eval( parse( text = initialize_ManualDoUpdates ),envir = evaluation_environment )

  # LR updates, etc.
  GetInvLR <- compile_fxn(function(inv_learning_rate,grad_i){
    # WN grad
    #return( (jnp$add(inv_learning_rate,jnp$divide(jnp$sum(jnp$square(grad_i)), inv_learning_rate))) )

    # Adagrad-norm
    return( (jnp$add(inv_learning_rate,jnp$sum(jnp$square(grad_i)))))
  })

  GetUpdatedParameters <- compile_fxn(function(a_vec, grad_i, inv_learning_rate_i){
    return( jnp$add(a_vec, jnp$multiply(jnp$reciprocal(inv_learning_rate_i),grad_i)))
  })

  a_vec_init_mat <- as.matrix(unlist( lapply(p_list, function(zer){ c(compositions::alr( t((zer)))) }) ) )
  a_vec_init_ast <- jnp$array(a_vec_init_mat+rnorm(length(a_vec_init_mat),sd=A_INIT_SD), dtj)
  a_vec_init_dag <- jnp$array(a_vec_init_mat+rnorm(length(a_vec_init_mat),sd = A_INIT_SD*MaxMin), dtj)
  a2Simplex <- compile_fxn(function(a_){
    exp_a_ <- jnp$exp(a_)
    aOnSimplex <- tapply(1:nrow(a_structure),a_structure$d,function(zer){
      tmp <- jnp$divide(  jnp$take(exp_a_, n2int(as.matrix(zer-1L) )),
                        jnp$add(OneTf_flat,jnp$sum(jnp$take(exp_a_, n2int(zer-1L) ) )))
      return( list( tmp ) ) })
    names(aOnSimplex) <- NULL
    return(  jnp$concatenate(aOnSimplex,0L) )
  })
  a2FullSimplex <- compile_fxn(function(a_){
    exp_a_ <- jnp$exp( a_ )
    aOnSimplex <- tapply(1:nrow(a_structure_leftoutLdminus1),a_structure_leftoutLdminus1$d,function(zer){
      exp_a_zer <- jnp$concatenate(list(jnp$take(exp_a_, n2int(as.matrix(as.array(zer - 1L) ))),
                                  jnp$array(as.matrix(1.))), # last category is exp(0) = 1
                                  axis = 0L)
      tmp <- jnp$divide(  exp_a_zer, jnp$sum(exp_a_zer))
      return( list( tmp ) ) })
    names( aOnSimplex ) <- NULL
    return(  jnp$expand_dims(jnp$stack(aOnSimplex,0L),1L) )
  })
  OneTf <- jnp$array(matrix(1L),dtj)
  OneTf_flat <- jnp$array(1L,dtj)
  Neg2_tf <- jnp$array(-2.,dtj)

  # Q functions
  print("Defining Q functions..")
  a2Simplex_optim <- ifelse( holdout_indicator == 1 ,
                             yes = list(a2Simplex),
                             no = list(a2FullSimplex) )[[1]]
  pi_star_value_init_dag <- a2Simplex_optim( a_ = a_vec_init_dag )
  pi_star_value_init_ast <- a2Simplex_optim( a_vec_init <- a_vec_init_ast )

  # define Q functions
  getQStar_single <- compile_fxn( getQStar_single )

  # multiround material
  {
  for(UseSinglePop in ifelse(MaxMin, yes = list(c(F,T)), no = list(T))[[1]]){
    # general specifications
    getQStar_diff_R <- paste(deparse(getQStar_diff_R),collapse="\n")
    getQStar_diff_R <- gsub(getQStar_diff_R, pattern = "SINGLE_PROP_KEY", replace = sprintf("T == !%s",UseSinglePop))
    getQStar_diff_R <- eval( parse( text = getQStar_diff_R ),envir = evaluation_environment )

    # specifications for case
    if(UseSinglePop){ getQStar_diff_R_multi <- getQStar_diff_R; name_ <-"Single" }
    if(!UseSinglePop){ getQStar_diff_R <- getQStar_diff_R; name_ <-"Multi" }
    eval(parse(text = sprintf("getQStar_diff_R_%sGroup <- getQStar_diff_R",name_)))
    eval(parse(text = sprintf("getQStar_diff_%sGroup <- compile_fxn( getQStar_diff_R_%sGroup )",name_,name_)))
  }
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
    main_comp_mat <- sapply(1:length(pi_star_value_loc),function(zer){
      main_comp_mat[pi_star_value_loc[zer],zer] <- 1
      return( main_comp_mat[,zer] ) })
    main_comp_mat <- jnp$array(main_comp_mat,dtj)

    shadow_comp_mat <- matrix(0, ncol = n_main_params, nrow = length_simplex_use)
    shadow_comp_mat <- sapply(1:length(pi_star_value_loc_shadow),function(zer){
      shadow_comp_mat[pi_star_value_loc_shadow[zer],zer] <- 1
      return( shadow_comp_mat[,zer] ) })
    shadow_comp_mat <- jnp$array(shadow_comp_mat,dtj)
    }

    split_vec_full <- unlist(sapply(1:length(factor_levels),function(xz){
      rep(xz,times=factor_levels[xz])} ))

    split_vec_use <- ifelse(ParameterizationType == "Implicit",
                            yes = list(split_vec), no = list(split_vec_full))[[1]]
  }

  environment(getPrettyPi) <- environment()
  getPrettyPi <- compile_fxn( getPrettyPi )

  getPrettyPi_diff <- ifelse(ParameterizationType=="Implicit",
                                  yes = list(getPrettyPi),
                                  no = list(jax$jit(function(x){x})))[[1]]
  a2Simplex_diff_use <- ifelse(ParameterizationType == "Implicit",
                               yes = list(a2Simplex),
                               no = list(a2FullSimplex))[[1]]

  ## get exact result
  pi_star_exact <- -10
  if(OptimType %in% c("tryboth") & diff == F){
    pi_star_exact <- np$array(getPrettyPi(getPiStar_exact(EST_COEFFICIENTS_tf)))
  }

  use_exact <- !( use_gd <- any(pi_star_exact<0) | any(pi_star_exact>1)  |
    (abs(sum(pi_star_exact) - sum(unlist(p_list_full))) > 1e-5) )
  if( use_gd ){
  if(!diff){
    getFixedEntries <- compile_fxn(function(EST_COEFFICIENTS_tf){
        main_coef <- jnp$take(EST_COEFFICIENTS_tf, indices = main_indices_i0, axis=0L)

        if(T == T){
        inter_coef <- jnp$take(EST_COEFFICIENTS_tf, indices = inter_indices_i0, axis=0L)
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
            inter_coef_ <- jnp$take(inter_coef,indices = n2int(ai(which_inter-1L)), axis = 0L)
          }

          # expand dimensions in the length == 1 case
          if(length(which_inter) == 1){
            inter_coef_ <- jnp$expand_dims(inter_coef_,0L)
            inter_into_main <- jnp$expand_dims(inter_into_main_0i,0L)
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
          sum_p <- jnp$expand_dims(jnp$sum(jnp$take(p_vec_tf,
                                                          indices = n2int(ai(which_d-1L)), axis=0L),keepdims=F),0L)
          return(   list("sum_p"=sum_p,
                         "indices_for_sum_pi"=n2int(as.matrix(ai(which_d-1L))))  )
        })
        term4_FC_a <- jnp$concatenate(term4_FC[1,],0L)
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
    fe <- getFixedEntries( EST_COEFFICIENTS_tf ) # needed for function initialization
  }

  # define GD function
  #REGRESSION_PARAMS_jax_dag <- REGRESSION_PARAMS_jax <- jnp$array(jnp$concatenate(list(EST_INTERCEPT_tf, EST_COEFFICIENTS_tf),0L))
  gather_fxn <- compile_fxn(function(x){
      INTERCEPT_ <- jnp$expand_dims(jnp$take(x,0L),0L)
      COEFFICIENTS_ <- jnp$take(x, jnp$array( jnp$arange( jnp$shape(x)[[1]] ) ) )
      list(INTERCEPT_, COEFFICIENTS_)})
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
    # bring functions into env and compile as needed
    getMultinomialSamp <<- jax$jit( getMultinomialSamp )

    environment(getPiStar_gd) <- environment()
  }

  # get initial learning rate for gd result
  nSGD <- nSGD_orig
  }

  # get jax seed into correct type
  jax_seed <- jnp$array(as.integer(c(jax_seed)))

  # Obtain solution via exact calculation
  print("Starting optimization...")
  pi_star_se_list_OUTER <- pi_star_list_OUTER <- replicate(n = K, list())
  q_star_OUTER <- q_star_se_OUTER <- replicate(n = K, list())
  for(k_clust in 1:K){
  if(K > 1){
    print(sprintf("Optimizing cluster %s of %s...",k_clust, K))
    ################################################
    # WARNING: Operational only in average case mode
    # reset means
    EST_INTERCEPT_tf <- jnp$array(t( my_mean_full[1,k_clust] ) )
    EST_COEFFICIENTS_tf <- jnp$array(as.matrix( my_mean_full[-1,k_clust] ) )
    REGRESSION_PARAMS_jax <- jnp$array(jnp$concatenate(list(EST_INTERCEPT_tf, EST_COEFFICIENTS_tf),0L))
    REGRESSION_PARAMS_jax_ast <- REGRESSION_PARAMS_jax
    REGRESSION_PARAMS_jax_dag <- REGRESSION_PARAMS_jax_ast0 <- REGRESSION_PARAMS_jax_ast

    # reset covariates
    vcov_OutcomeModel_ast <- vcov_OutcomeModel_dag <- vcov_OutcomeModel_ast0 <- vcov_OutcomeModel_dag0 <- vcov_OutcomeModel <- vcov_OutcomeModel_by_k[[ k_clust ]]
  }

  if(use_exact){
    print("Optimization type: Exact solution")
    #results_vec_list <- replicate(length(unlist(p_list_full))+1,list()) # + 1 for intercept
    FxnForJacobian <- function(INPUT_){
      EST_INTERCEPT_tf_ <- INPUT_[[1]]
      EST_COEFFICIENTS_tf_  <- INPUT_[[2]]
      pi_star_full_exact <- pi_star_full <- getPrettyPi( pi_star_reduced <- getPiStar_exact(EST_COEFFICIENTS_tf_))
      q_star_exact <- q_star <- getQStar_single(pi_star = pi_star_reduced,
                                                EST_INTERCEPT_tf = EST_INTERCEPT_tf_,
                                                EST_COEFFICIENTS_tf = EST_COEFFICIENTS_tf_)
      results_vec <- jnp$concatenate(list(q_star, pi_star_full),0L)
      return( results_vec )
    }
    results_vec <- FxnForJacobian(list(EST_INTERCEPT_tf,EST_COEFFICIENTS_tf))
    jacobian_mat <- jax$jacobian(FxnForJacobian,0L)( (INPUT_  <- list(EST_INTERCEPT_tf,EST_COEFFICIENTS_tf)))

    # old code
    # for(ia in 1:length(results_vec_list)){results_vec_list[[ia]] <- jnp$reshape(jnp$take(results_vec,n2int(ai(ia-1L)),axis=0L),shape=list()) }

    # reshape jacobian and process results
    jacobian_mat_exact <- jacobian_mat <- cbind(
                          np$array(jnp$squeeze(jnp$squeeze(jnp$squeeze(jacobian_mat[[1]],1L),1L))),
                          np$array(jnp$squeeze(jnp$squeeze(jnp$squeeze(jacobian_mat[[2]],1L),2L)))
                         )
    vcov_OutcomeModel_concat <- vcov_OutcomeModel_ast
    q_star_exact <- q_star <- np$array( jnp$take(results_vec, 0L) )
    pi_star_full <- np$array( jnp$take(results_vec, jnp$array(1L:(length(results_vec)-1L))) )
  }

  if(use_gd){
    print("Optimization type: Gradient ascent")

    # perform main gd runs + inference
    nSGD <- nSGD_orig

    # first do ave case analysis
    p_vec_full_ast_jnp <- p_vec_full_dag_jnp <- p_vec_full_jnp

    # do true iterative optimization
    # optimize q for pi* then for pi_dag and so forth, change iterative
    # plot( REGRESSION_PARAMS_jax$to_py(),REGRESSION_PARAMS_jax_dag$to_py() );abline(a=0,b=1)
    # plot(p_vec_full_dot_jnp$to_py(), p_vec_full_dag_jnp$to_py());abline(a=0,b=1)

    # check the multinomial sampling
    # plot(getMultinomialSamp(p_vec_jnp, baseSeed = jnp$array(55L))$to_py())
    gd_full_simplex <- T
    gd_time <- system.time( q_with_pi_star_full <- getPiStar_gd(
                             REGRESSION_PARAMETERS_ast = REGRESSION_PARAMS_jax_ast,
                             REGRESSION_PARAMETERS_dag = REGRESSION_PARAMS_jax_dag,
                             REGRESSION_PARAMETERS_ast0 = REGRESSION_PARAMS_jax_ast0,
                             REGRESSION_PARAMETERS_dag0 = REGRESSION_PARAMS_jax_dag0,
                             P_VEC_FULL_ast = p_vec_full_ast_jnp,
                             P_VEC_FULL_dag = p_vec_full_dag_jnp,
                             LAMBDA = lambda_jnp,
                             BASE_SEED = jax_seed ) )
    q_with_pi_star_full <- jnp$array(q_with_pi_star_full, dtj)
    # as.matrix(q_with_pi_star_full)[1:3]
    print("Time GD: ");print(gd_time)

    grad_mag_ast_vec <- unlist(  lapply(grad_mag_ast_vec,function(zer){
      ifelse(is.na(zer),no = np$array(jnp$sqrt( jnp$sum(jnp$square(jnp$array(zer,dtj))) )), yes = NA) }) )
    try(plot( grad_mag_ast_vec, main = "Gradient Magnitude Evolution (ast)", log =""),T)
    try(points(lowess(grad_mag_ast_vec), cex = 2, type = "l",lwd = 2, col = "red"), T)
    grad_mag_dag_vec <- try(unlist(  lapply(grad_mag_dag_vec,function(zer){
      ifelse(is.na(zer),no = np$array(jnp$sqrt( jnp$sum(jnp$square(jnp$array(zer,dtj))) )),yes=NA) }) ),T)
    try(plot( grad_mag_dag_vec , main = "Gradient Magnitude Evolution (dag)"),T)
    try(points(lowess(grad_mag_dag_vec), cex = 2, type = "l",lwd = 2, col = "red"), T)

    gd_full_simplex <- F
    browser()
    pi_star_red <- getPiStar_gd(
                        REGRESSION_PARAMETERS_ast = REGRESSION_PARAMS_jax_ast,
                        REGRESSION_PARAMETERS_dag = REGRESSION_PARAMS_jax_dag,
                        REGRESSION_PARAMETERS_ast0 = REGRESSION_PARAMS_jax_ast0,
                        REGRESSION_PARAMETERS_dag0 = REGRESSION_PARAMS_jax_dag0,
                        P_VEC_FULL_ast = p_vec_full_ast_jnp,
                        P_VEC_FULL_dag = p_vec_full_dag_jnp,
                        LAMBDA = lambda_jnp,
                        BASE_SEED = jax_seed)
    pi_star_red <- (pi_star_red$to_py()[-c(1:3),])
    pi_star_red_dag <- jnp$array(as.matrix(pi_star_red[-c(1:(length(pi_star_red)/2))]))
    pi_star_red_ast <- jnp$array(as.matrix(  pi_star_red[1:(length(pi_star_red)/2)] ) )

    q_star_gd <- q_star <- np$array(  q_with_pi_star_full )[1]
    pi_star_full_gd <- pi_star_full <- np$array( q_with_pi_star_full )[-c(1:3)]
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
                                          lambda_jnp,
                                          jax_seed))
      jacobian_mat_gd <- jacobian_mat <- lapply(jacobian_mat,function(l_){
        as.matrix( jnp$squeeze(jnp$squeeze(jnp$array(l_,dtj),1L),2L) ) })
      jacobian_mat_gd <- jacobian_mat <- do.call(cbind, jacobian_mat)
      vcov_OutcomeModel_concat <- list(
                     vcov_OutcomeModel_ast  ,
                     vcov_OutcomeModel_dag  ,
                     vcov_OutcomeModel_ast0 ,
                     vcov_OutcomeModel_dag0  )
      vcov_OutcomeModel_concat <- Matrix::bdiag( vcov_OutcomeModel_concat )
      vcov_OutcomeModel_concat <- as.matrix(  vcov_OutcomeModel_concat )
      #jacobian_mat_gd <- jacobian_mat <- as.matrix( jnp$squeeze(jnp$squeeze(jnp$array(jacobian_mat,dtj),1L),2L) )
      print("Time Jacobian of GD Solution: ");print(jacobian_time)
    }
    # hist(c(jacobian_mat));image(jacobian_mat)
    # summary(lm(np$array(sol_gd)~np$array(sol_exact)))
  }

  # print time of jacobian calculation
  # remember, the first three entries of output are:
  # Qhat_population, Qhat, Qhat_dag
  vcov_PiStar <- jacobian_mat %*% vcov_OutcomeModel_concat %*% t(jacobian_mat)
  q_star <- as.matrix(   q_star  )
  q_star_se <- sqrt(  diag( vcov_PiStar )[1] )
  pi_star_numeric <- np$array( pi_star_full )
  #trueQ;print(civ<-c(q_star - abs(qnorm((1-ConfLevel)/2))*q_star_se,q_star + abs(qnorm((1-ConfLevel)/2))*q_star_se))
  # 1*(  civ[1] <= trueQ & civ[2] >= trueQ)

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
  #plot(abs(pi_star_true-np$array(1-getPiStar_exact())));abline(a=0,b=1);points(pi_star_se,col="red")
  #plot(abs(pi_star_true-unlist(lapply(pi_star_list,function(zer){zer[-1]}))));abline(a=0,b=1);points(pi_star_se,col="red"); points(pi_star_se,col="red")

  # add names
  pi_star_list <- RenamePiList(  pi_star_list  )
  pi_star_se_list <- RenamePiList(  pi_star_se_list  )

  # append to outer list for K > 1 case
  pi_star_list_OUTER[[k_clust]] <- pi_star_list
  pi_star_se_list_OUTER[[k_clust]] <- pi_star_se_list
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

  return( list(   "PiStar_point" = pi_star_list,
                  "PiStar_se" = pi_star_se_list,
                  "Q_point_mEst" = q_star,
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
                                  yes = list(getQStar_diff_MultiGroup),
                                  no = list(getQStar_diff_SingleGroup))[[1]],

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
                  "estimationType" = "TwoStep",
                  "Y_model" = my_model ) )
}
