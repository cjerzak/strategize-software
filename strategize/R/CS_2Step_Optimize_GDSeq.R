getQPiStar_gd <-  function(REGRESSION_PARAMETERS_ast,
                          REGRESSION_PARAMETERS_dag,
                          REGRESSION_PARAMETERS_ast0,
                          REGRESSION_PARAMETERS_dag0,
                          P_VEC_FULL_ast,
                          P_VEC_FULL_dag,
                          SLATE_VEC_ast, 
                          SLATE_VEC_dag,
                          LAMBDA,
                          SEED,
                          functionList,
                          a_i_ast,  # initial value, ast 
                          a_i_dag,  # initial value, dag 
                          functionReturn = T,
                          gd_full_simplex, 
                          quiet = TRUE
                          ){
  # throw fxns into env 
  dQ_da_ast <- functionList[[1]]
  dQ_da_dag <- functionList[[2]]
  QFXN <- functionList[[3]]

  REGRESSION_PARAMETERS_ast <- gather_fxn(REGRESSION_PARAMETERS_ast)
  INTERCEPT_ast_ <- REGRESSION_PARAMETERS_ast[[1]]
  COEFFICIENTS_ast_ <- REGRESSION_PARAMETERS_ast[[2]]


  INTERCEPT_dag0_ <- INTERCEPT_ast0_ <- INTERCEPT_dag_ <- INTERCEPT_ast_
  COEFFICIENTS_dag0_ <- COEFFICIENTS_ast0_ <- COEFFICIENTS_dag_ <- COEFFICIENTS_ast_
  if( adversarial ){
    REGRESSION_PARAMETERS_dag <- gather_fxn(REGRESSION_PARAMETERS_dag)
    INTERCEPT_dag_ <- REGRESSION_PARAMETERS_dag[[1]]
    COEFFICIENTS_dag_ <- REGRESSION_PARAMETERS_dag[[2]]
  }
  if(nRounds > 1 & adversarial){
    REGRESSION_PARAMETERS_ast0 <- gather_fxn(REGRESSION_PARAMETERS_ast0)
    INTERCEPT_ast0_ <- REGRESSION_PARAMETERS_ast0[[1]]
    COEFFICIENTS_ast0_ <- REGRESSION_PARAMETERS_ast0[[2]]

    REGRESSION_PARAMETERS_dag0 <- gather_fxn(REGRESSION_PARAMETERS_dag0)
    INTERCEPT_dag0_ <- REGRESSION_PARAMETERS_dag0[[1]]
    COEFFICIENTS_dag0_ <- REGRESSION_PARAMETERS_dag0[[2]]
  }

  # gradient descent iterations
  strenv$grad_mag_ast_vec <- strenv$grad_mag_dag_vec <- rep(NA, times = nSGD)
  strenv$loss_ast_vec <- strenv$loss_dag_vec <- rep(NA, times = nSGD)
  strenv$inv_learning_rate_ast_vec <- strenv$inv_learning_rate_dag_vec <- rep(NA, times = nSGD)
  goOn <- F; i<-0
  INIT_MIN_GRAD_ACCUM <- strenv$jnp$array(0.1)
  while(goOn == F){
    if ((i <- i + 1) < 5 | i %in% unique(ceiling(c(0.25, 0.5, 0.75, 1) * nSGD))) { 
      message(sprintf("SGD Iteration: %s of %s", i, nSGD) ) 
    }

    # do dag updates ("min" step)
    if( i %% 1 == 0 & adversarial ){
      
      # dQ_da_dag built off FullGetQStar_
      SEED   <- strenv$jax$random$split(SEED)[[1L]]
      grad_i_dag <- dQ_da_dag(  a_i_ast, a_i_dag,                    #1,2
                                INTERCEPT_ast_,  COEFFICIENTS_ast_,  #3,4
                                INTERCEPT_dag_,  COEFFICIENTS_dag_,  #5,6
                                INTERCEPT_ast0_, COEFFICIENTS_ast0_, #7,8
                                INTERCEPT_dag0_, COEFFICIENTS_dag0_, #9,10
                                P_VEC_FULL_ast, P_VEC_FULL_dag,      #11,12
                                SLATE_VEC_ast, SLATE_VEC_dag,        #13,14
                                LAMBDA,                              #15
                                Q_SIGN_ <- strenv$jnp$array(-1.),    #16
                                SEED                                 #17
                                )
      strenv$loss_dag_vec[i] <- list(grad_i_dag[[1]]); grad_i_dag <- grad_i_dag[[2]]
      
      if(i == 1){
        inv_learning_rate_da_dag <- strenv$jnp$maximum(INIT_MIN_GRAD_ACCUM, 10*strenv$jnp$square(strenv$jnp$linalg$norm( grad_i_dag )))
      }

      if(!use_optax){
        inv_learning_rate_da_dag <-  strenv$jax$lax$stop_gradient(GetInvLR(inv_learning_rate_da_dag, grad_i_dag))
        a_i_dag <- GetUpdatedParameters(a_vec = a_i_dag, 
                                        grad_i = grad_i_dag,
                                        inv_learning_rate_i = strenv$jnp$sqrt(inv_learning_rate_da_dag))
      }
      if(use_optax){
        updates_and_opt_state_dag <- jit_update_dag( updates = grad_i_dag, 
                                                     state = opt_state_dag, 
                                                     params = a_i_dag)
        opt_state_dag <- updates_and_opt_state_dag[[2]]
        a_i_dag <- jit_apply_updates_dag(params = a_i_dag,  
                                         updates = updates_and_opt_state_dag[[1]])
      }

      strenv$grad_mag_dag_vec[i] <- list(strenv$jnp$linalg$norm( grad_i_dag ))
      if(!use_optax){ strenv$inv_learning_rate_dag_vec[i] <- list( inv_learning_rate_da_dag ) }
    }

    # do updates ("max" step)
    if( i %% 1 == 0 | (!adversarial) ){
      SEED   <- strenv$jax$random$split(SEED)[[1L]]
      grad_i_ast <- dQ_da_ast( a_i_ast, a_i_dag,
                               INTERCEPT_ast_,  COEFFICIENTS_ast_,
                               INTERCEPT_dag_,  COEFFICIENTS_dag_,
                               INTERCEPT_ast0_, COEFFICIENTS_ast0_,
                               INTERCEPT_dag0_, COEFFICIENTS_dag0_,
                               P_VEC_FULL_ast, P_VEC_FULL_dag,
                               SLATE_VEC_ast, SLATE_VEC_dag,
                               LAMBDA, 
                               (Q_SIGN_ <- strenv$jnp$array(1.)),
                               SEED
                               )
      strenv$loss_ast_vec[i] <- list(grad_i_ast[[1]]); grad_i_ast <- grad_i_ast[[2]]
      
      if(i==1){ inv_learning_rate_da_ast <- strenv$jnp$maximum(INIT_MIN_GRAD_ACCUM, 10*strenv$jnp$square(strenv$jnp$linalg$norm(grad_i_ast)))  }
      if(!use_optax){
        inv_learning_rate_da_ast <-  strenv$jax$lax$stop_gradient( GetInvLR(inv_learning_rate_da_ast, grad_i_ast) )
        a_i_ast <- GetUpdatedParameters(a_vec = a_i_ast, 
                                        grad_i = grad_i_ast,
                                        inv_learning_rate_i = strenv$jnp$sqrt(inv_learning_rate_da_ast))
      }

      if(use_optax){
        updates_and_opt_state_ast <- jit_update_ast( updates = grad_i_ast, 
                                                     state = opt_state_ast, 
                                                     params = a_i_ast)
        opt_state_ast <- updates_and_opt_state_ast[[2]]
        a_i_ast <- jit_apply_updates_ast(params = a_i_ast, 
                                         updates = updates_and_opt_state_ast[[1]])
      }

      strenv$grad_mag_ast_vec[i] <- list( strenv$jnp$linalg$norm( grad_i_ast ) )
      if(!use_optax){ strenv$inv_learning_rate_ast_vec[i] <- list( inv_learning_rate_da_ast ) }
    }
    if(i >= nSGD){goOn <- T}
  }

  message("Saving output from gd...")
  {
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
      pi_star_ast_f_all <- strenv$jax$vmap(function(s_){ 
                                    strenv$getMultinomialSamp(pi_star_ast_, 
                                                              MNtemp, 
                                                              s_, 
                                                              strenv$ParameterizationType, 
                                                              strenv$d_locator_use
                                                              )},
                                in_axes = list(0L))(strenv$jax$random$split( strenv$jax$random$PRNGKey( 30L + jax_seed ), 
                                                                             nMonte_Qglm) )
      pi_star_dag_f_all <- strenv$jax$vmap(function(s_){ 
                                    strenv$getMultinomialSamp(pi_star_dag_, 
                                                              MNtemp, 
                                                              s_, 
                                                              strenv$ParameterizationType, 
                                                              strenv$d_locator_use
                                                              )}, 
                               in_axes = list(0L))( strenv$jax$random$split( strenv$jax$random$PRNGKey( 400L + jax_seed ), 
                                                                             nMonte_Qglm) )
    }

    q_star_f <- strenv$Vectorized_QMonteIter(pi_star_ast_f_all,  pi_star_dag_f_all,
                                       INTERCEPT_ast_, COEFFICIENTS_ast_,
                                       INTERCEPT_dag_, COEFFICIENTS_dag_)$mean(0L)

    # setup results for returning
    ret_array <- strenv$jnp$concatenate(ifelse(gd_full_simplex, 
                                               yes = list(list( q_star_f, 
                                                                pi_star_ast_full_simplex_, 
                                                                pi_star_dag_full_simplex_ )), 
                                               no  = list(list( q_star_f, 
                                                                pi_star_ast_, 
                                                                pi_star_dag_ )))[[1]])
    if( functionReturn ){ 
                            ret_array <- list(ret_array,
                                               list("dQ_da_ast" = dQ_da_ast, 
                                                    "dQ_da_dag" = dQ_da_dag,
                                                    "QFXN" = QFXN, 
                                                    "getMultinomialSamp" = strenv$getMultinomialSamp,
                                                    "a_i_ast" = a_i_ast, 
                                                    "a_i_dag" = a_i_dag
                                               ) )
    }
    return( ret_array )
  }
}
