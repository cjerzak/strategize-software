getQStar_single <- function(pi_star_ast, pi_star_dag,
                             EST_INTERCEPT_tf_ast, EST_COEFFICIENTS_tf_ast,
                             EST_INTERCEPT_tf_dag, EST_COEFFICIENTS_tf_dag){
  # note: here, dag ignored 
  
  # coef info
  main_coef <- strenv$jnp$expand_dims( strenv$jnp$take(EST_COEFFICIENTS_tf_ast, indices = main_indices_i0, axis = 0L),1L)
  inter_coef <- strenv$jnp$expand_dims( strenv$jnp$take(EST_COEFFICIENTS_tf_ast,indices = inter_indices_i0, axis = 0L), 1L)

  # get interaction info
  pi_dp <- strenv$jnp$take(pi_star_ast, n2int( ai(interaction_info$dl_index_adj)-1L), axis=0L)
  pi_dpp <- strenv$jnp$take(pi_star_ast, n2int( ai(interaction_info$dplp_index_adj)-1L), axis=0L)

  Qhat <- strenv$jnp$squeeze(
                glm_outcome_transform( EST_INTERCEPT_tf_ast + 
                                   strenv$jnp$matmul( main_coef$transpose(), pi_star_ast) +
                                   strenv$jnp$matmul( inter_coef$transpose(), pi_dp*pi_dpp ) ) , 1L)

  return( strenv$jnp$concatenate( list(Qhat, Qhat, Qhat), 0L)  ) # to keep sizes consistent with diff case 
}

getQStar_diff_BASE <- function(pi_star_ast, pi_star_dag,
                               EST_INTERCEPT_tf_ast, EST_COEFFICIENTS_tf_ast,
                               EST_INTERCEPT_tf_dag, EST_COEFFICIENTS_tf_dag){
  
  # coef
  main_coef_ast <- strenv$jnp$expand_dims(strenv$jnp$take(EST_COEFFICIENTS_tf_ast, indices = main_indices_i0, axis = 0L),1L)
  inter_coef_ast <- strenv$jnp$expand_dims(strenv$jnp$take(EST_COEFFICIENTS_tf_ast, indices = inter_indices_i0, axis = 0L),1L)

  # get interaction info
  pi_ast_dp <- strenv$jnp$take(pi_star_ast, n2int( ai(interaction_info$dl_index_adj)-1L), axis=0L)
  pi_ast_dpp <- strenv$jnp$take(pi_star_ast, n2int( ai(interaction_info$dplp_index_adj)-1L), axis=0L)

  pi_dag_dp <- strenv$jnp$take(pi_star_dag, n2int( ai(interaction_info$dl_index_adj)-1L), axis=0L)
  pi_dag_dpp <- strenv$jnp$take(pi_star_dag, n2int( ai(interaction_info$dplp_index_adj)-1L), axis=0L)

  # combine
  DELTA_pi_star <- pi_star_ast - pi_star_dag
  DELTA_pi_star_prod <- pi_ast_dp * pi_ast_dpp - pi_dag_dp * pi_dag_dpp

  Qhat_ast_among_ast <- glm_outcome_transform( 
            EST_INTERCEPT_tf_ast + 
            strenv$jnp$matmul(main_coef_ast$transpose(), DELTA_pi_star) + 
            strenv$jnp$matmul( inter_coef_ast$transpose(),  DELTA_pi_star_prod ) )
  
  if( !Q_DISAGGREGATE ){ Qhat_population <- Qhat_ast_among_dag <- Qhat_ast_among_ast }
  if( Q_DISAGGREGATE ){
    main_coef_dag <- strenv$jnp$expand_dims( strenv$jnp$take(EST_COEFFICIENTS_tf_dag, indices = main_indices_i0, axis=0L), 1L)
    inter_coef_dag <- strenv$jnp$expand_dims( strenv$jnp$take(EST_COEFFICIENTS_tf_dag, indices = inter_indices_i0, axis=0L), 1L)
    
    Qhat_ast_among_dag <- glm_outcome_transform( 
              EST_INTERCEPT_tf_dag + 
              strenv$jnp$matmul( main_coef_dag$transpose(), DELTA_pi_star ) +
              strenv$jnp$matmul( inter_coef_dag$transpose(), DELTA_pi_star_prod ) )
    
    # Pr( Ast | Ast Voter) * Pr(Ast Voters) +  Pr( Ast | Dag Voter) * Pr(Dag Voters)
    Qhat_population <- Qhat_ast_among_ast * strenv$jnp$array(AstProp) +  Qhat_ast_among_dag * strenv$jnp$array(DagProp)
  }
  return( strenv$jnp$concatenate( list(Qhat_population, Qhat_ast_among_ast, Qhat_ast_among_dag), 0L)  )
}

FullGetQStar_ <- function(a_i_ast,
                          a_i_dag,
                          INTERCEPT_ast_, COEFFICIENTS_ast_,
                          INTERCEPT_dag_, COEFFICIENTS_dag_,
                          INTERCEPT_ast0_, COEFFICIENTS_ast0_,
                          INTERCEPT_dag0_, COEFFICIENTS_dag0_,
                          P_VEC_FULL_ast_, P_VEC_FULL_dag_,
                          SLATE_VEC_ast_, SLATE_VEC_dag_,
                          LAMBDA_,
                          Q_SIGN,
                          SEED_IN_LOOP){
  pi_star_full_i_ast <- getPrettyPi_diff( pi_star_i_ast <- a2Simplex_diff_use( a_i_ast ))
  pi_star_full_i_dag <- getPrettyPi_diff( pi_star_i_dag <- a2Simplex_diff_use( a_i_dag ))

  if(!MaxMin){
    # NOTE: When diff == F, dag not used  
    q_max <- q__ <- strenv$jnp$take(QFXN(pi_star_ast =  pi_star_i_ast,
                  pi_star_dag = pi_star_i_dag,
                  EST_INTERCEPT_tf_ast = INTERCEPT_ast_,
                  EST_COEFFICIENTS_tf_ast = COEFFICIENTS_ast_,
                  EST_INTERCEPT_tf_dag = INTERCEPT_dag_,
                  EST_COEFFICIENTS_tf_dag = COEFFICIENTS_dag_),0L)
  }

  if(MaxMin){
    # setup conditions 
    cond_UseAst <- strenv$jnp$multiply(strenv$jnp$array(0.5),strenv$jnp$add(strenv$jnp$array(1.), Q_SIGN))

    # sample main competitor features
    TSAMP_ast_all <- strenv$jax$vmap(function(s_){ getMultinomialSamp(pi_star_i_ast, MNtemp, s_)},in_axes = 0L)(
        strenv$jax$random$split(strenv$jax$random$PRNGKey(SEED_IN_LOOP + 199L),nMonte_MaxMin) )
    TSAMP_dag_all <- strenv$jax$vmap(function(s_){ getMultinomialSamp(pi_star_i_dag, MNtemp, s_)},in_axes = list(0L))(
        strenv$jax$random$split(strenv$jax$random$PRNGKey(SEED_IN_LOOP + 299L),nMonte_MaxMin) )

    # sample primary competitor features uniformly or using slates 
    TSAMP_ast_PrimaryComp_all <- strenv$jax$vmap(function(s_){ getMultinomialSamp(SLATE_VEC_ast_, MNtemp, s_)},in_axes = list(0L))(
      strenv$jax$random$split(strenv$jax$random$PRNGKey(SEED_IN_LOOP + 399L),nMonte_MaxMin) )
    TSAMP_dag_PrimaryComp_all <- strenv$jax$vmap(function(s_){ getMultinomialSamp(SLATE_VEC_dag_, MNtemp, s_)},in_axes = list(0L))(
      strenv$jax$random$split(strenv$jax$random$PRNGKey(SEED_IN_LOOP + 499L),nMonte_MaxMin) )
    
    # compute electoral analysis 
    VectorizedQMonteLoop_res <- Vectorized_QMonteIter_MaxMin(
                        TSAMP_ast_all, TSAMP_dag_all,
                        TSAMP_ast_PrimaryComp_all, TSAMP_dag_PrimaryComp_all,
                        a_i_ast,
                        a_i_dag,
                        INTERCEPT_ast_, COEFFICIENTS_ast_,
                        INTERCEPT_dag_, COEFFICIENTS_dag_,
                        INTERCEPT_ast0_, COEFFICIENTS_ast0_,
                        INTERCEPT_dag0_, COEFFICIENTS_dag0_,
                        P_VEC_FULL_ast_,
                        P_VEC_FULL_dag_,
                        LAMBDA_,
                        Q_SIGN,
                        SEED_IN_LOOP)
    
    # sanity checks 
    # VectorizedQMonteLoop_res[[1]][[1]]
    # VectorizedQMonteLoop_res[[1]][[2]]
    # VectorizedQMonteLoop_res[[1]][[3]]
    
    # VectorizedQMonteLoop_res[[2]][[1]]$aval
    # VectorizedQMonteLoop_res[[2]][[2]]
    # VectorizedQMonteLoop_res[[2]][[3]]$aval
    
    # sample only for the conditioning? 
    # Sum Quantity Given Condition / (ep + sum( Sum Condition))
    GeneralVoteShareAstAmongAst_Given_AstWinsAstPrimary <- strenv$jnp$sum(VectorizedQMonteLoop_res[[1]][[1]]) / 
                                            (strenv$jnp$array(0.001) + strenv$jnp$sum(VectorizedQMonteLoop_res[[1]][[2]]))
    GeneralVoteShareDagAmongAst_Given_AstWinsAstPrimary <- strenv$jnp$sum( 1 - VectorizedQMonteLoop_res[[1]][[1]]) / 
                                            (strenv$jnp$array(0.001) + strenv$jnp$sum(VectorizedQMonteLoop_res[[1]][[2]]))
    
    GeneralVoteShareAstAmongDag_Given_DagWinsDagPrimary <- strenv$jnp$sum(VectorizedQMonteLoop_res[[2]][[1]]) / 
                                            (strenv$jnp$array(0.001) + strenv$jnp$sum(VectorizedQMonteLoop_res[[2]][[2]]))
    GeneralVoteShareDagAmongDag_Given_DagWinsDagPrimary <- strenv$jnp$sum(1 - VectorizedQMonteLoop_res[[2]][[1]]) / 
                                             (strenv$jnp$array(0.001) + strenv$jnp$sum(VectorizedQMonteLoop_res[[2]][[2]]))
    
    PrAstWinsAstPrimary <- strenv$jnp$mean(VectorizedQMonteLoop_res[[1]][[3]])
    PrDagWinsDagPrimary <- strenv$jnp$mean(VectorizedQMonteLoop_res[[2]][[3]])

    q_max_ast <- GeneralVoteShareAstAmongAst_Given_AstWinsAstPrimary * PrAstWinsAstPrimary * strenv$jnp$array(AstProp) + 
                    GeneralVoteShareAstAmongDag_Given_DagWinsDagPrimary * PrDagWinsDagPrimary * strenv$jnp$array(DagProp)
    q_max_dag <- GeneralVoteShareDagAmongAst_Given_AstWinsAstPrimary * PrAstWinsAstPrimary * strenv$jnp$array(AstProp) + 
                    GeneralVoteShareDagAmongDag_Given_DagWinsDagPrimary * PrDagWinsDagPrimary * strenv$jnp$array(DagProp)
    
    if(T == F){
    AveProdAst_AstOpti <- strenv$jnp$sum(VectorizedQMonteLoop_res[[1]][[1]]) / 
                        (strenv$jnp$array(0.001) + strenv$jnp$sum(VectorizedQMonteLoop_res[[1]][[2]]))
    AveProdDag_AstOpti <- strenv$jnp$sum(VectorizedQMonteLoop_res[[2]][[1]]) / 
                        (strenv$jnp$array(0.001) + strenv$jnp$sum(VectorizedQMonteLoop_res[[2]][[2]]))
    AveProdAst_DagOpti <- strenv$jnp$sum(VectorizedQMonteLoop_res[[3]][[1]]) /
                        (strenv$jnp$array(0.001) + strenv$jnp$sum(VectorizedQMonteLoop_res[[3]][[2]]))
    AveProdDag_DagOpti <- strenv$jnp$sum(VectorizedQMonteLoop_res[[4]][[1]]) /
                        (strenv$jnp$array(0.001) + strenv$jnp$sum(VectorizedQMonteLoop_res[[4]][[2]]))
    }

    # quantity to maximize for ast and dag respectively 
    q_max <-  cond_UseAst * q_max_ast  + (1-cond_UseAst) * q_max_dag 
  }

  # regularization
  {
    if(penalty_type %in% c("L1","L2")){
      PenFxn <- ifelse(penalty_type == "L1", yes = list(strenv$jnp$abs), no = list(strenv$jnp$square))[[1]]
      var_pen_ast__ <- strenv$jnp$multiply(LAMBDA_, strenv$jnp$negative(strenv$jnp$sum(PenFxn( pi_star_full_i_ast - P_VEC_FULL_ast_ )  )))
      var_pen_dag__ <- strenv$jnp$multiply(LAMBDA_, strenv$jnp$negative(strenv$jnp$sum(PenFxn( pi_star_full_i_dag - P_VEC_FULL_dag_ )  )))
    }
    if(penalty_type == "LInfinity"){
      var_pen_ast__ <- tapply(1:length(split_vec_full), split_vec_full, function(zer){
        list( strenv$jnp$max(strenv$jnp$take(pi_star_full_i_ast, indices = strenv$jnp$array( ai(zer-1L)),axis = 0L)))})
      names(var_pen_ast__)<-NULL ; var_pen_ast__ <- strenv$jnp$negative(LAMBDA_*strenv$jnp$sum( strenv$jnp$stack(var_pen_ast__)))

      var_pen_dag__ <- tapply(1:length(split_vec_full), split_vec_full, function(zer){
        list( strenv$jnp$max(strenv$jnp$take(pi_star_full_i_dag, indices = strenv$jnp$array( ai(zer-1L)),axis = 0L)))})
      names(var_pen_dag__)<-NULL ; var_pen_dag__ <- strenv$jnp$negative(LAMBDA_*strenv$jnp$sum( strenv$jnp$stack(var_pen_dag__)))
    }
    if(penalty_type == "KL"){
      var_pen_ast__ <- strenv$jnp$multiply(LAMBDA_,strenv$jnp$negative(strenv$jnp$sum(P_VEC_FULL_ast_ * (strenv$jnp$log(P_VEC_FULL_ast_) - strenv$jnp$log(pi_star_full_i_ast)))))
      var_pen_dag__ <- strenv$jnp$multiply(LAMBDA_,strenv$jnp$negative(strenv$jnp$sum(P_VEC_FULL_dag_ * (strenv$jnp$log(P_VEC_FULL_dag_) - strenv$jnp$log(pi_star_full_i_dag)))))
    }
  }

  if( MaxMin ){ myLoss <- q_max + var_pen_ast__ + var_pen_dag__ } 
  if( !MaxMin ){ myLoss <- q_max + var_pen_ast__ }
   
  return( myLoss )
}