getPiStar_gd <-  function(REGRESSION_PARAMETERS_ast,
                          REGRESSION_PARAMETERS_dag,
                          REGRESSION_PARAMETERS_ast0,
                          REGRESSION_PARAMETERS_dag0,
                          P_VEC_FULL_ast,
                          P_VEC_FULL_dag,
                          LAMBDA,
                          BASE_SEED){
  REGRESSION_PARAMETERS_ast <- gather_fxn(REGRESSION_PARAMETERS_ast)
  INTERCEPT_ast_ <- REGRESSION_PARAMETERS_ast[[1]]
  COEFFICIENTS_ast_ <- REGRESSION_PARAMETERS_ast[[2]]

  INTERCEPT_dag0_ <- INTERCEPT_ast0_ <- INTERCEPT_dag_ <- INTERCEPT_ast_
  COEFFICIENTS_dag0_ <- COEFFICIENTS_ast0_ <- COEFFICIENTS_dag_ <- COEFFICIENTS_ast_
  if( MaxMin ){
    REGRESSION_PARAMETERS_dag <- gather_fxn(REGRESSION_PARAMETERS_dag)
    INTERCEPT_dag_ <- REGRESSION_PARAMETERS_dag[[1]]
    COEFFICIENTS_dag_ <- REGRESSION_PARAMETERS_dag[[2]]
  }
  if(nRounds > 1 & MaxMin){
    REGRESSION_PARAMETERS_ast0 <- gather_fxn(REGRESSION_PARAMETERS_ast0)
    INTERCEPT_ast0_ <- REGRESSION_PARAMETERS_ast0[[1]]
    COEFFICIENTS_ast0_ <- REGRESSION_PARAMETERS_ast0[[2]]

    REGRESSION_PARAMETERS_dag0 <- gather_fxn(REGRESSION_PARAMETERS_dag0)
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
                                            LAMBDA_,
                                            Q_SIGN,
                                            SEED_IN_LOOP){
          pi_star_full_i_ast <- getPrettyPi_diff( pi_star_i_ast <- a2Simplex_diff_use( a_i_ast ))
          pi_star_full_i_dag <- getPrettyPi_diff( pi_star_i_dag <- a2Simplex_diff_use( a_i_dag ))

          if(!MaxMin){
            if(!diff){
              q__ <- getQStar_single(pi_star = pi_star_i_ast,
                                          EST_INTERCEPT_tf = INTERCEPT_ast_,
                                          EST_COEFFICIENTS_tf = COEFFICIENTS_ast_)
              q_max <- q__ <- jnp$take(q__,0L)
            }
            if(diff){
              q__ <- ifelse(MaxMin,
                            yes = list(getQStar_diff_MultiGroup),
                            no = list(getQStar_diff_SingleGroup))[[1]](
                              pi_star_ast =  pi_star_i_ast,
                              pi_star_dag = pi_star_i_dag,
                              EST_INTERCEPT_tf_ast = INTERCEPT_ast_,
                              EST_COEFFICIENTS_tf_ast = COEFFICIENTS_ast_,
                              EST_INTERCEPT_tf_dag = INTERCEPT_dag_,
                              EST_COEFFICIENTS_tf_dag = COEFFICIENTS_dag_)
              q_max <- q__ <- jnp$take(q__,0L)
            }
          }

          if(MaxMin){
            ProductAstGroup_sum_AstOpti <- jnp$array(0.)
            ProductDagGroup_sum_AstOpti <- jnp$array(0.)
            ProductAstGroup_sum_DagOpti <- jnp$array(0.)
            ProductDagGroup_sum_DagOpti <- jnp$array(0.)
            lax_cond_indicator_DagWinsPrimary <- jnp$array(1.)
            lax_cond_indicator_AstWinsPrimary <- jnp$array(1.)
            lax_cond_indicator_counter_DagWinsPrimary <- jnp$array(0.)
            lax_cond_indicator_counter_AstWinsPrimary <- jnp$array(0.)

            for( monti_i in 1:nMonte_MaxMin){

                SEED_IN_LOOP_i <- jnp$multiply(jnp$array(as.integer(monti_i)),SEED_IN_LOOP)
                TSAMP_ast <- getMultinomialSamp(pi_star_i_ast, baseSeed = jnp$add(jnp$array(1L),SEED_IN_LOOP_i))
                TSAMP_dag <- getMultinomialSamp(pi_star_i_dag, baseSeed = jnp$add(jnp$array(2L),SEED_IN_LOOP_i))

                # sample primary competitor features uniformly
                #TSAMP_ast_PrimaryComp <- getMultinomialSamp(p_vec_jnp, baseSeed = jnp$add(jnp$array(3L),SEED_IN_LOOP_i))
                #TSAMP_dag_PrimaryComp <- getMultinomialSamp(p_vec_jnp, baseSeed = jnp$add(jnp$array(4L),SEED_IN_LOOP_i))

                # sample primary competitor features from same distribution (stopping gradient)
                TSAMP_ast_PrimaryComp <- jax$lax$stop_gradient(getMultinomialSamp(pi_star_i_ast, baseSeed = jnp$add(jnp$array(3L),SEED_IN_LOOP_i)))
                TSAMP_dag_PrimaryComp <- jax$lax$stop_gradient(getMultinomialSamp(pi_star_i_dag, baseSeed = jnp$add(jnp$array(4L),SEED_IN_LOOP_i)))

                GeneralVoteShareResults_AstReferenced <- ifelse(MaxMin,
                              yes = list(getQStar_diff_MultiGroup),
                              no = list(getQStar_diff_SingleGroup))[[1]](
                                pi_star_ast =  TSAMP_ast,
                                pi_star_dag = TSAMP_dag,
                                #pi_star_ast =  pi_star_i_ast, pi_star_dag = pi_star_i_dag,
                                EST_INTERCEPT_tf_ast = INTERCEPT_ast_,
                                EST_COEFFICIENTS_tf_ast = COEFFICIENTS_ast_,
                                EST_INTERCEPT_tf_dag = INTERCEPT_dag_,
                                EST_COEFFICIENTS_tf_dag = COEFFICIENTS_dag_)
                GeneralVoteShareAstAmongAst <- jnp$take(GeneralVoteShareResults_AstReferenced,1L)
                GeneralVoteShareAstAmongDag <- jnp$take(GeneralVoteShareResults_AstReferenced,2L)

                cond_UseDag <- jnp$multiply(jnp$array(0.5),jnp$subtract(jnp$array(1.), Q_SIGN))
                cond_UseAst <- jnp$multiply(jnp$array(0.5),jnp$add(jnp$array(1.), Q_SIGN))

                # primary stage analysis
                {
                  PrimaryVoteShareAstAmongAst <- getQStar_diff_SingleGroup(
                    pi_star_ast =  TSAMP_ast,
                    pi_star_dag = TSAMP_ast_PrimaryComp,
                    EST_INTERCEPT_tf_ast = INTERCEPT_ast0_,
                    EST_COEFFICIENTS_tf_ast = COEFFICIENTS_ast0_,
                    EST_INTERCEPT_tf_dag = INTERCEPT_ast0_,
                    EST_COEFFICIENTS_tf_dag = COEFFICIENTS_ast0_)
                  PrimaryVoteShareDagAmongDag <- getQStar_diff_SingleGroup(
                    pi_star_ast = TSAMP_dag,
                    pi_star_dag = TSAMP_dag_PrimaryComp,
                    EST_INTERCEPT_tf_ast = INTERCEPT_dag0_,
                    EST_COEFFICIENTS_tf_ast = COEFFICIENTS_dag0_,
                    EST_INTERCEPT_tf_dag = INTERCEPT_dag0_,
                    EST_COEFFICIENTS_tf_dag = COEFFICIENTS_dag0_)
                  PrimaryVoteShareDagAmongDag <- jnp$take(PrimaryVoteShareDagAmongDag,0L)
                  PrimaryVoteShareAstAmongAst <- jnp$take(PrimaryVoteShareAstAmongAst,0L)

                  lax_cond_indicator_DagWinsPrimary <- jax$lax$cond(
                    pred = jnp$greater(PrimaryVoteShareDagAmongDag,jnp$array(0.5)),
                    true_fun = function(){ jnp$array(1.) },
                    false_fun = function(){ jnp$array(0.)} )
                  lax_cond_indicator_AstWinsPrimary <- jax$lax$cond(
                    pred = jnp$greater(PrimaryVoteShareAstAmongAst,jnp$array(0.5)),
                    true_fun = function(){ jnp$array(1.) },
                    false_fun = function(){ jnp$array(0.)} )

                  lax_cond_indicator_counter_AstWinsPrimary <- jnp$add(
                                                        lax_cond_indicator_counter_AstWinsPrimary,
                                                        lax_cond_indicator_AstWinsPrimary)
                  lax_cond_indicator_counter_DagWinsPrimary <- jnp$add(
                                                        lax_cond_indicator_counter_DagWinsPrimary,
                                                        lax_cond_indicator_DagWinsPrimary)
                }

                # combine all information together
                ProductAstGroup_AstOpti <- jnp$multiply(lax_cond_indicator_AstWinsPrimary,
                                              jnp$multiply(GeneralVoteShareAstAmongAst,
                                                     PrimaryVoteShareAstAmongAst))
                ProductDagGroup_AstOpti <- jnp$multiply(lax_cond_indicator_DagWinsPrimary,
                                              jnp$multiply(GeneralVoteShareAstAmongDag,
                                                      PrimaryVoteShareDagAmongDag))
                ProductAstGroup_DagOpti <- jnp$multiply(lax_cond_indicator_AstWinsPrimary,
                                              jnp$multiply(jnp$subtract(jnp$array(1.),GeneralVoteShareAstAmongAst),
                                                      PrimaryVoteShareAstAmongAst))
                ProductDagGroup_DagOpti <- jnp$multiply(lax_cond_indicator_DagWinsPrimary,
                                              jnp$multiply(jnp$subtract(jnp$array(1.),GeneralVoteShareAstAmongDag),
                                                      PrimaryVoteShareDagAmongDag))

                # cumulative sums
                ProductAstGroup_sum_AstOpti <- jnp$add(ProductAstGroup_sum_AstOpti,ProductAstGroup_AstOpti)
                ProductDagGroup_sum_AstOpti <- jnp$add(ProductDagGroup_sum_AstOpti,ProductDagGroup_AstOpti)
                ProductAstGroup_sum_DagOpti <- jnp$add(ProductAstGroup_sum_DagOpti,ProductAstGroup_DagOpti)
                ProductDagGroup_sum_DagOpti <- jnp$add(ProductDagGroup_sum_DagOpti,ProductDagGroup_DagOpti)
              }
            AveProdAst_AstOpti <- jnp$divide(ProductAstGroup_sum_AstOpti,
                                       jnp$add(jnp$array(0.001),
                                          lax_cond_indicator_counter_AstWinsPrimary))
            AveProdDag_AstOpti <- jnp$divide(ProductDagGroup_sum_AstOpti,
                                       jnp$add(jnp$array(0.001),
                                          lax_cond_indicator_counter_DagWinsPrimary))
            AveProdAst_DagOpti <- jnp$divide(ProductAstGroup_sum_DagOpti,
                                       jnp$add(jnp$array(0.001),
                                          lax_cond_indicator_counter_AstWinsPrimary))
            AveProdDag_DagOpti <- jnp$divide(ProductDagGroup_sum_DagOpti,
                                        jnp$add(jnp$array(0.001),
                                            lax_cond_indicator_counter_DagWinsPrimary))

            # quantity to maximize
            q_max <- jnp$add(
                  jnp$multiply(cond_UseDag,jnp$add( AveProdAst_DagOpti, AveProdDag_DagOpti )),
                  jnp$multiply(cond_UseAst,jnp$add( AveProdAst_AstOpti, AveProdDag_AstOpti ))
                )
           }

            # regularization here
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

          ret_ <- jnp$add( jnp$add( q_max, var_pen_ast__), var_pen_dag__)
          return( ret_ )
        }
        )
        dQ_da_ast <- jax$jit(jax$grad(FullGetQStar_, argnums = jnp$array(0L)))
        dQ_da_dag <- jax$jit(jax$grad(FullGetQStar_, argnums = jnp$array(1L)))
      }
    }

    # da_dag updates (min step)
    if( i %% 1 == 0 & MaxMin ){
      grad_i_dag <- dQ_da_dag(  a_i_ast, a_i_dag,
                                INTERCEPT_ast_,  COEFFICIENTS_ast_,
                                INTERCEPT_dag_,  COEFFICIENTS_dag_,
                                INTERCEPT_ast0_, COEFFICIENTS_ast0_,
                                INTERCEPT_dag0_, COEFFICIENTS_dag0_,
                                P_VEC_FULL_ast, P_VEC_FULL_dag,
                                LAMBDA, jnp$array(-1.),
                                jnp$add(BASE_SEED,jnp$array(as.integer( i) ) ) )
      if(i == 1){
        inv_learning_rate_da_dag <- jnp$maximum(jnp$array(0.001), jnp$multiply(10,  jnp$square(jnp$linalg$norm( grad_i_dag ))))
      }

      if(!UseOptax){
        inv_learning_rate_da_dag <-  jax$lax$stop_gradient(GetInvLR(inv_learning_rate_da_dag, grad_i_dag))
        a_i_dag <- GetUpdatedParameters(a_vec = a_i_dag, grad_i = grad_i_dag,
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
                               LAMBDA, jnp$array(1.),
                               jnp$add(jnp$array(2L),jnp$add(BASE_SEED,jnp$array(as.integer( i) ) ) )  )
      if(i==1){ inv_learning_rate_da_ast <- jnp$maximum(jnp$array(0.001), jnp$multiply(10, jnp$square(jnp$linalg$norm(grad_i_ast))))  }
      if(!UseOptax){
        inv_learning_rate_da_ast <-  jax$lax$stop_gradient( GetInvLR(inv_learning_rate_da_ast, grad_i_ast) )
        a_i_ast <- GetUpdatedParameters(a_vec = a_i_ast, grad_i = grad_i_ast,
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
    pi_star_ast_full_simplex_ <- getPrettyPi( pi_star_ast_ <- a2Simplex_diff_use( a_i_ast ) )
    pi_star_dag_full_simplex_ <- getPrettyPi( pi_star_dag_ <- a2Simplex_diff_use( a_i_dag ))
    doMonte_Q <- ifelse(glm_family == "gaussian", yes = F, no = T)
    nMonte_Q <- ifelse(glm_family == "gaussian", yes = 1, no = nMonte_Qglm)
    q_star_f <- jnp$array(0.)
    for(monti_ii in 1:nMonte_Q){
      if( !doMonte_Q ){
        pi_star_ast_f <- pi_star_ast_
        pi_star_dag_f <- pi_star_dag_
      }
      if( doMonte_Q ){
        SEED_IN_LOOP_ii <- jnp$multiply(jnp$array(as.integer(monti_ii)),jnp$array(345155L))
        TSAMP_ast <- getMultinomialSamp(pi_star_ast_, baseSeed = jnp$add(jnp$array(100L),SEED_IN_LOOP_ii))
        TSAMP_dag <- getMultinomialSamp(pi_star_dag_, baseSeed = jnp$add(jnp$array(201L),SEED_IN_LOOP_ii))
        pi_star_ast_f <- TSAMP_ast
        pi_star_dag_f <- TSAMP_dag
      }

      if( diff ){
        q_star_ <- ifelse(MaxMin,
                          yes = list(getQStar_diff_MultiGroup),
                          no = list(getQStar_diff_SingleGroup))[[1]](
                            pi_star_ast = pi_star_ast_f,
                            pi_star_dag = pi_star_dag_f,
                            EST_INTERCEPT_tf_ast = INTERCEPT_ast_, EST_COEFFICIENTS_tf_ast = COEFFICIENTS_ast_,
                            EST_INTERCEPT_tf_dag = INTERCEPT_dag_, EST_COEFFICIENTS_tf_dag = COEFFICIENTS_dag_)
      }
      if( !diff  ){
        q_star_ <- getQStar_single(pi_star = pi_star_ast_f,
                                        EST_INTERCEPT_tf = INTERCEPT_ast_,
                                        EST_COEFFICIENTS_tf = COEFFICIENTS_ast_)
      }
      q_star_f <- jnp$add(q_star_f,q_star_)
    }
    q_star_f <- jnp$divide(q_star_f, nMonte_Q)

    if( gd_full_simplex == T){ ret_array <- jnp$concatenate(list( q_star_, pi_star_ast_full_simplex_, pi_star_dag_full_simplex_ ) ) }
    if( gd_full_simplex == F){ ret_array <- jnp$concatenate(list( q_star_, pi_star_ast_, pi_star_dag_ ) ) }
    # plot(a_i_dag$to_py(),a_i$to_py());abline(a=0,b=1)
    # plot(pi_star_full_simplex_$to_py(),pi_star_dag_full_simplex_$to_py());abline(a=0,b=1)
    return( ret_array  ) # ret_array$shape
  }
}
