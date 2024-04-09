getQStar_single <- function(pi_star,
                            EST_INTERCEPT_tf,
                            EST_COEFFICIENTS_tf
                            ){
  # coef info
  main_coef <- jnp$take(EST_COEFFICIENTS_tf, indices = main_indices_i0, axis = 0L)
  inter_coef <- jnp$take(EST_COEFFICIENTS_tf,indices = inter_indices_i0, axis = 0L)

  # get interaction info
  pi_dp <- jnp$take(pi_star, n2int(as.integer(interaction_info$dl_index_adj)-1L), axis=0L)
  pi_dpp <- jnp$take(pi_star, n2int(as.integer(interaction_info$dplp_index_adj)-1L), axis=0L)

  MainTermContrib <- jnp$matmul(jnp$transpose(main_coef),pi_star)
  InterTermContrib <- jnp$sum(jnp$multiply(jnp$multiply(inter_coef,pi_dp),pi_dpp))
  Qhat <- glm_outcome_transform(
    jnp$add( jnp$add(EST_INTERCEPT_tf, MainTermContrib), InterTermContrib)
  )
  return( Qhat ) }

getQStar_diff_BASE <- function(pi_star_ast, pi_star_dag,
                               EST_INTERCEPT_tf_ast, EST_COEFFICIENTS_tf_ast,
                               EST_INTERCEPT_tf_dag, EST_COEFFICIENTS_tf_dag){
  # coef
  main_coef_ast <- jnp$take(EST_COEFFICIENTS_tf_ast, indices = main_indices_i0, axis = 0L)
  inter_coef_ast <- jnp$take(EST_COEFFICIENTS_tf_ast, indices = inter_indices_i0, axis = 0L)

  # get interaction info
  pi_ast_dp <- jnp$take(pi_star_ast, n2int(as.integer(interaction_info$dl_index_adj)-1L), axis=0L)
  pi_ast_dpp <- jnp$take(pi_star_ast, n2int(as.integer(interaction_info$dplp_index_adj)-1L), axis=0L)
  pi_ast_prod <- jnp$multiply(pi_ast_dp, pi_ast_dpp)

  pi_dag_dp <- jnp$take(pi_star_dag, n2int(as.integer(interaction_info$dl_index_adj)-1L), axis=0L)
  pi_dag_dpp <- jnp$take(pi_star_dag, n2int(as.integer(interaction_info$dplp_index_adj)-1L), axis=0L)
  pi_dag_prod <- jnp$multiply(pi_dag_dp, pi_dag_dpp)

  # combine
  DELTA_pi_star <- jnp$subtract(pi_star_ast, pi_star_dag)
  DELTA_pi_star_prod <- jnp$subtract(pi_ast_prod, pi_dag_prod)

  Qhat_ast <- glm_outcome_transform( jnp$add(
    jnp$add(EST_INTERCEPT_tf_ast,
            jnp$matmul(jnp$transpose(main_coef_ast), DELTA_pi_star)),
    jnp$sum(jnp$multiply(inter_coef_ast, DELTA_pi_star_prod), keepdims = T)))
  if( !Q_DISAGGREGATE ){ Qhat_population <- Qhat_dag <- Qhat_ast }

  if( Q_DISAGGREGATE ){
    main_coef_dag <- jnp$take(EST_COEFFICIENTS_tf_dag, indices = main_indices_i0, axis=0L)
    inter_coef_dag <- jnp$take(EST_COEFFICIENTS_tf_dag, indices = inter_indices_i0, axis=0L)
    DELTA_ast_dag <- jnp$subtract(  pi_star_ast,  pi_star_dag )
    DELTA_ast_dag_prod <- jnp$subtract(  pi_ast_prod, pi_dag_prod )
    Qhat_dag <- glm_outcome_transform( jnp$add(  jnp$add(
      EST_INTERCEPT_tf_dag,
      jnp$matmul(jnp$transpose(main_coef_dag), DELTA_ast_dag )),
      jnp$sum(jnp$multiply(inter_coef_dag, DELTA_ast_dag_prod ), keepdims=T)) )
    # Pr(Win D_c Among All | R_c Opp) = 
    #   Pr(Win D_c Among All | R_c Opp, R voters) Pr(R voters) +
    #   Pr(Win D_c Among All | R_c Opp, D voters) Pr(D voters) +
    #   Pr(Win D_c Among All | R_c Opp, I voters) Pr(I voters) # dropped 
    Qhat_population <- Qhat_ast * AstProp +  Qhat_dag * DagProp
  }
  return( jnp$concatenate(list(Qhat_population, Qhat_ast, Qhat_dag),0L)  )
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

  cond_UseAst <- jnp$multiply(jnp$array(0.5),jnp$add(jnp$array(1.), Q_SIGN))
  cond_UseDag <- jnp$multiply(jnp$array(0.5),jnp$subtract(jnp$array(1.), Q_SIGN))

  if(!MaxMin){
    if(!diff){
      q__ <- QFXN(pi_star = pi_star_i_ast,
                  EST_INTERCEPT_tf = INTERCEPT_ast_,
                  EST_COEFFICIENTS_tf = COEFFICIENTS_ast_
                  )
      q_max <- q__ <- jnp$take(q__,0L)
    }
    if(diff){
      q__ <- QFXN(pi_star_ast =  pi_star_i_ast,
                  pi_star_dag = pi_star_i_dag,
                  EST_INTERCEPT_tf_ast = INTERCEPT_ast_,
                  EST_COEFFICIENTS_tf_ast = COEFFICIENTS_ast_,
                  EST_INTERCEPT_tf_dag = INTERCEPT_dag_,
                  EST_COEFFICIENTS_tf_dag = COEFFICIENTS_dag_
                  )
      q_max <- q__ <- jnp$take(q__,0L)
    }
  }

  if(MaxMin){
    ProductDagGroup_sum_AstOpti <- ProductAstGroup_sum_AstOpti <- jnp$array(0.)
    ProductAstGroup_sum_DagOpti <- ProductDagGroup_sum_DagOpti <- jnp$array(0.)
    lax_cond_indicator_AstWinsPrimary <- lax_cond_indicator_DagWinsPrimary <- jnp$array(1.)
    lax_cond_indicator_counter_DagWinsPrimary <- jnp$array(0.)
    lax_cond_indicator_counter_AstWinsPrimary <- jnp$array(0.)

    # sample main competitor features
    TSAMP_ast_all <- jnp$concatenate(sapply(1:nMonte_MaxMin, function(monti_i){
      SEED_IN_LOOP_i <- jnp$array(as.integer(monti_i)) * SEED_IN_LOOP
      return( TSAMP_ast <- getMultinomialSamp(pi_star_i_ast, jnp$add(jnp$array(19L),SEED_IN_LOOP_i)) )
      }), 1L)
    TSAMP_dag_all <- jnp$concatenate(sapply(1:nMonte_MaxMin, function(monti_i){
      SEED_IN_LOOP_i <- jnp$array(as.integer(monti_i)) * SEED_IN_LOOP
      return( TSAMP_dag <- getMultinomialSamp(pi_star_i_dag, jnp$add(jnp$array(29L),SEED_IN_LOOP_i))  ) 
      }), 1L)

    # sample primary competitor features uniformly or using slates 
    TSAMP_ast_PrimaryComp_all <- jnp$concatenate(sapply(1:nMonte_MaxMin, function(monti_i){
      SEED_IN_LOOP_i <- jnp$array(as.integer(monti_i)) * SEED_IN_LOOP
      return( TSAMP_ast_PrimaryComp <- getMultinomialSamp(SLATE_VEC_ast_, jnp$add(jnp$array(399L),SEED_IN_LOOP_i)) )  
    }), 1L)
    TSAMP_dag_PrimaryComp_all <- jnp$concatenate(sapply(1:nMonte_MaxMin, function(monti_i){
      SEED_IN_LOOP_i <- jnp$multiply(jnp$array(as.integer(monti_i)),SEED_IN_LOOP)
      return( TSAMP_dag_PrimaryComp <- getMultinomialSamp(SLATE_VEC_dag_, jnp$add(jnp$array(499L),SEED_IN_LOOP_i)) ) 
    }), 1L)

    VectorizedQMonteLoop_res <- VectorizedQMonteLoop_optimize(
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
    
    # Sum Quantity Given Condition / (ep + sum( Sum Condition))
    AveProdAst_AstOpti <- jnp$sum(VectorizedQMonteLoop_res[[1]][[1]]) / 
                        (jnp$array(0.001) + jnp$sum(VectorizedQMonteLoop_res[[1]][[2]]))
    AveProdDag_AstOpti <- jnp$sum(VectorizedQMonteLoop_res[[2]][[1]]) / 
                        (jnp$array(0.001) + jnp$sum(VectorizedQMonteLoop_res[[2]][[2]]))
    AveProdAst_DagOpti <- jnp$sum(VectorizedQMonteLoop_res[[3]][[1]]) /
                        (jnp$array(0.001) + jnp$sum(VectorizedQMonteLoop_res[[3]][[2]]))
    AveProdDag_DagOpti <- jnp$sum(VectorizedQMonteLoop_res[[4]][[1]]) /
                        (jnp$array(0.001) + jnp$sum(VectorizedQMonteLoop_res[[4]][[2]]))

    # quantity to maximize
    q_max <-  cond_UseDag * ( AveProdAst_DagOpti + AveProdDag_DagOpti ) + 
                cond_UseAst * ( AveProdAst_AstOpti + AveProdDag_AstOpti )
  }

  # regularization
  {
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
  }

  return( q_max + var_pen_ast__ + var_pen_dag__ )
}
