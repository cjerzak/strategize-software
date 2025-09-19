InitializeQMonteFxns_VSept19 <- function(){
  strenv$Vectorized_QMonteIter_MaxMin <- compile_fxn( strenv$jax$vmap(
    (strenv$QMonteIter_MaxMin <- compile_fxn(function(
                         TSAMP_ast, TSAMP_dag,
                         TSAMP_ast_PrimaryComp, TSAMP_dag_PrimaryComp,
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
                         SEED_IN_LOOP){
    GVShareResults_AstReferenced <- getQStar_diff_MultiGroup(
                                                  TSAMP_ast, #pi_star_ast
                                                  TSAMP_dag, # pi_star_dag
                                                  INTERCEPT_ast_, # EST_INTERCEPT_tf_ast
                                                  COEFFICIENTS_ast_, # EST_COEFFICIENTS_tf_ast
                                                  INTERCEPT_dag_, # EST_INTERCEPT_tf_dag
                                                  COEFFICIENTS_dag_) #EST_COEFFICIENTS_tf_dag
    GVShareAstAmongAst <- strenv$jnp$take(GVShareResults_AstReferenced,1L)
    GVShareAstAmongDag <- strenv$jnp$take(GVShareResults_AstReferenced,2L)
  
    # primary stage analysis
    {
      PrimaryVoteShareAstAmongAst <- strenv$jnp$take(getQStar_diff_SingleGroup(
        pi_star_ast =  TSAMP_ast,
        pi_star_dag = TSAMP_ast_PrimaryComp,
        EST_INTERCEPT_tf_ast = INTERCEPT_ast0_,
        EST_COEFFICIENTS_tf_ast = COEFFICIENTS_ast0_,
        EST_INTERCEPT_tf_dag = INTERCEPT_ast0_,
        EST_COEFFICIENTS_tf_dag = COEFFICIENTS_ast0_),0L)
      #plot(strenv$np$array( TSAMP_ast$val ) - strenv$np$array( TSAMP_ast_PrimaryComp$val ) ,
           #strenv$np$array( PrimaryVoteShareAstAmongAst$val ) )
      
      PrimaryVoteShareDagAmongDag <- strenv$jnp$take(getQStar_diff_SingleGroup(
        pi_star_ast = TSAMP_dag,
        pi_star_dag = TSAMP_dag_PrimaryComp,
        EST_INTERCEPT_tf_ast = INTERCEPT_dag0_,
        EST_COEFFICIENTS_tf_ast = COEFFICIENTS_dag0_,
        EST_INTERCEPT_tf_dag = INTERCEPT_dag0_,
        EST_COEFFICIENTS_tf_dag = COEFFICIENTS_dag0_),0L)
      #plot(strenv$np$array( TSAMP_dag$val$to_concrete_value() )-strenv$np$array( TSAMP_dag_PrimaryComp$val ) ,
           #strenv$np$array( PrimaryVoteShareDagAmongDag$val$to_concrete_value() ) )

      # win based on relaxed sample from bernoulli using oryx (depreciated)
      #Indicator_AstWinsPrimary <- strenv$oryx$distributions$RelaxedBernoulli(temperature = MNtemp, probs = PrimaryVoteShareAstAmongAst )$sample(seed = SEED_IN_LOOP + 9992L)
      #Indicator_DagWinsPrimary <- strenv$oryx$distributions$RelaxedBernoulli(temperature = MNtemp, probs = PrimaryVoteShareDagAmongDag )$sample(seed = SEED_IN_LOOP + 153L)
      
      # win based on relaxed sample from bernoulli using base JAX
      # (ast)
      eps <- 1e-8
      p_ast      <- strenv$jnp$clip(PrimaryVoteShareAstAmongAst, eps, 1 - eps)
      logits_ast <- strenv$jnp$log(p_ast) - strenv$jnp$log1p(-p_ast)
      u_ast      <- strenv$jax$random$uniform(SEED_IN_LOOP, shape = strenv$jnp$shape(p_ast), minval = eps, maxval = 1 - eps)
      Indicator_AstWinsPrimary <- strenv$jax$nn$sigmoid((logits_ast + strenv$jnp$log(u_ast) - strenv$jnp$log1p(-u_ast)) / MNtemp)
      SEED_IN_LOOP   <- strenv$jax$random$split(SEED_IN_LOOP)[[1L]]
      # (dag)
      p_dag      <- strenv$jnp$clip(PrimaryVoteShareDagAmongDag, eps, 1 - eps)
      logits_dag <- strenv$jnp$log(p_dag) - strenv$jnp$log1p(-p_dag)
      u_dag      <- strenv$jax$random$uniform(SEED_IN_LOOP, shape = strenv$jnp$shape(p_dag), minval = eps, maxval = 1 - eps)
      Indicator_DagWinsPrimary <- strenv$jax$nn$sigmoid((logits_dag + strenv$jnp$log(u_dag) - strenv$jnp$log1p(-u_dag)) / MNtemp)
      SEED_IN_LOOP   <- strenv$jax$random$split(SEED_IN_LOOP)[[1L]]
    }
  
    # combine all information together
    return(list( "AmongAst"=list("GVShareAstAmongAst"=GVShareAstAmongAst, 
                                  "Indicator_AstWinsPrimary"=Indicator_AstWinsPrimary, 
                                  "PrimaryVoteShareAstAmongAst"=PrimaryVoteShareAstAmongAst),
                 "AmongDag"=list("GVShareAstAmongDag"=GVShareAstAmongDag, 
                                  "Indicator_DagWinsPrimary"=Indicator_DagWinsPrimary, 
                                  "PrimaryVoteShareDagAmongDag"=PrimaryVoteShareDagAmongDag)) ) 
  })), 
  in_axes = eval(parse(text = paste("list(0L,0L,0L,0L,", # vectorize over TSAMPs and SEED
                      paste(rep("NULL,",times = 15-1), collapse=""), "0L",  ")",sep = "") ))))

  strenv$Vectorized_QMonteIter <- compile_fxn( strenv$jax$vmap( (strenv$QMonteIter <- compile_fxn( 
                          function(pi_star_ast_f, 
                                   pi_star_dag_f,
                                   INTERCEPT_ast_,
                                   COEFFICIENTS_ast_,
                                   INTERCEPT_dag_,
                                   COEFFICIENTS_dag_){
    # note: when diff == F, dag is ignored 
    q_star_ <- QFXN(pi_star_ast_f, 
                    pi_star_dag_f, 
                    INTERCEPT_ast_,  
                    COEFFICIENTS_ast_, 
                    INTERCEPT_dag_,  
                    COEFFICIENTS_dag_)
    return( q_star_ )
  })), in_axes = list(0L,0L,NULL,NULL,NULL,NULL)))
  
}

InitializeQMonteFxns <- function(){
  
  # Helper: population general-election value for a single (t, u) pair
  Qpop_pair <- compile_fxn(function(t, u,
                                    INTERCEPT_ast_, COEFFICIENTS_ast_,
                                    INTERCEPT_dag_, COEFFICIENTS_dag_){
    strenv$jnp$take(
      getQStar_diff_MultiGroup(
        t, u,
        INTERCEPT_ast_,  COEFFICIENTS_ast_,
        INTERCEPT_dag_,  COEFFICIENTS_dag_
      ),
      0L  # population component
    )
  })
  
  # Helper: primary head-to-head win prob κ_A(t, t') within A's primary
  kappa_pair_A <- compile_fxn(function(t, t_prime,
                                       INTERCEPT_ast0_, COEFFICIENTS_ast0_){
    strenv$jnp$take(
      getQStar_diff_SingleGroup(
        t,            # entrant
        t_prime,      # field draw
        INTERCEPT_ast0_, COEFFICIENTS_ast0_,
        INTERCEPT_ast0_, COEFFICIENTS_ast0_
      ),
      0L
    )
  })
  
  # Helper: primary head-to-head win prob κ_B(u, u') within B's primary
  kappa_pair_B <- compile_fxn(function(u, u_prime,
                                       INTERCEPT_dag0_, COEFFICIENTS_dag0_){
    strenv$jnp$take(
      getQStar_diff_SingleGroup(
        u,
        u_prime,
        INTERCEPT_dag0_, COEFFICIENTS_dag0_,
        INTERCEPT_dag0_, COEFFICIENTS_dag0_
      ),
      0L
    )
  })
  
  # === Core Monte-Carlo evaluator that respects the paper’s push-forward ===
  strenv$QMonteIter_MaxMin <- compile_fxn(function(
    TSAMP_ast, TSAMP_dag,                        # [nA, ·], [nB, ·] entrants
    TSAMP_ast_PrimaryComp, TSAMP_dag_PrimaryComp,# [nA',·],[nB',·]  field
    a_i_ast, a_i_dag,                            # (unused here; kept for signature)
    INTERCEPT_ast_, COEFFICIENTS_ast_,
    INTERCEPT_dag_, COEFFICIENTS_dag_,
    INTERCEPT_ast0_, COEFFICIENTS_ast0_,
    INTERCEPT_dag0_, COEFFICIENTS_dag0_,
    P_VEC_FULL_ast_, P_VEC_FULL_dag_,           # (unused here)
    LAMBDA_, Q_SIGN, SEED_IN_LOOP               # (unused here)
  ){
    
    # ---- Primary κ matrices ----
    # κ_A[i,j] = Pr_A_primary( t_i beats t'_j )
    kA <- strenv$jax$vmap(function(t_i){
      strenv$jax$vmap(function(tp_j){
        kappa_pair_A(t_i, tp_j, INTERCEPT_ast0_, COEFFICIENTS_ast0_)
      }, in_axes = 0L)(TSAMP_ast_PrimaryComp)
    }, in_axes = 0L)(TSAMP_ast)                             # [nA, nA']
    
    # κ_B[s,r] = Pr_B_primary( u_s beats u'_r )
    kB <- strenv$jax$vmap(function(u_s){
      strenv$jax$vmap(function(up_r){
        kappa_pair_B(u_s, up_r, INTERCEPT_dag0_, COEFFICIENTS_dag0_)
      }, in_axes = 0L)(TSAMP_dag_PrimaryComp)
    }, in_axes = 0L)(TSAMP_dag)                             # [nB, nB']
    
    # Averages for push-forward weights
    kA_mean_over_field   <- kA$mean(1L)  # [nA]   E_field[κ | entrant]
    kB_mean_over_field   <- kB$mean(1L)  # [nB]
    kA_mean_over_entrant <- kA$mean(0L)  # [nA']  E_entrant[κ | field]
    kB_mean_over_entrant <- kB$mean(0L)  # [nB']
    
    # ---- General-election blocks for the four primary outcomes (2×2) ----
    # C(t_i, u_s)
    C_tu <- strenv$jax$vmap(function(t_i){
      strenv$jax$vmap(function(u_s){
        Qpop_pair(t_i, u_s,
                  INTERCEPT_ast_, COEFFICIENTS_ast_,
                  INTERCEPT_dag_, COEFFICIENTS_dag_)
      }, in_axes = 0L)(TSAMP_dag)
    }, in_axes = 0L)(TSAMP_ast)                           # [nA, nB]
    
    # C(t_i, u'_r)
    C_tu_field <- strenv$jax$vmap(function(t_i){
      strenv$jax$vmap(function(up_r){
        Qpop_pair(t_i, up_r,
                  INTERCEPT_ast_, COEFFICIENTS_ast_,
                  INTERCEPT_dag_, COEFFICIENTS_dag_)
      }, in_axes = 0L)(TSAMP_dag_PrimaryComp)
    }, in_axes = 0L)(TSAMP_ast)                    # [nA, nB']
    
    # C(t'_j, u_s)
    C_field_u <- strenv$jax$vmap(function(tp_j){
      strenv$jax$vmap(function(u_s){
        Qpop_pair(tp_j, u_s,
                  INTERCEPT_ast_, COEFFICIENTS_ast_,
                  INTERCEPT_dag_, COEFFICIENTS_dag_)
      }, in_axes = 0L)(TSAMP_dag)
    }, in_axes = 0L)(TSAMP_ast_PrimaryComp)         # [nA', nB]
    
    # C(t'_j, u'_r)
    C_field_field <- strenv$jax$vmap(function(tp_j){
      strenv$jax$vmap(function(up_r){
        Qpop_pair(tp_j, up_r,
                  INTERCEPT_ast_, COEFFICIENTS_ast_,
                  INTERCEPT_dag_, COEFFICIENTS_dag_)
      }, in_axes = 0L)(TSAMP_dag_PrimaryComp)
    }, in_axes = 0L)(TSAMP_ast_PrimaryComp)   # [nA', nB']
    
    # ---- Push-forward mixture weights (no sampling; fully differentiable) ----
    one <- strenv$OneTf_flat
    W1 <- strenv$jnp$outer(kA_mean_over_field,             kB_mean_over_field)             # [nA,  nB]
    W2 <- strenv$jnp$outer(kA_mean_over_field,             one - kB_mean_over_entrant)     # [nA,  nB']
    W3 <- strenv$jnp$outer(one - kA_mean_over_entrant,     kB_mean_over_field)             # [nA', nB]
    W4 <- strenv$jnp$outer(one - kA_mean_over_entrant,     one - kB_mean_over_entrant)     # [nA', nB']
    
    # ---- Expected general-election vote share for A ----
    E1 <- (C_tu          * W1)$mean()   # A entrant vs B entrant
    E2 <- (C_tu_field    * W2)$mean()   # A entrant vs B field
    E3 <- (C_field_u     * W3)$mean()   # A field   vs B entrant
    E4 <- (C_field_field * W4)$mean()   # A field   vs B field
    
    q_ast <- E1 + E2 + E3 + E4
    q_dag <- one - q_ast  # zero-sum
    
    return(list("q_ast" = q_ast, "q_dag" = q_dag))
  })
  
  # Batch wrapper kept for API compatibility
  strenv$Vectorized_QMonteIter_MaxMin <- compile_fxn(function(
    TSAMP_ast_all, TSAMP_dag_all,
    TSAMP_ast_PrimaryComp_all, TSAMP_dag_PrimaryComp_all,
    a_i_ast, a_i_dag,
    INTERCEPT_ast_, COEFFICIENTS_ast_,
    INTERCEPT_dag_, COEFFICIENTS_dag_,
    INTERCEPT_ast0_, COEFFICIENTS_ast0_,
    INTERCEPT_dag0_, COEFFICIENTS_dag0_,
    P_VEC_FULL_ast_, P_VEC_FULL_dag_,
    LAMBDA_, Q_SIGN, SEED_IN_LOOP
  ){
    strenv$QMonteIter_MaxMin(
      TSAMP_ast_all, TSAMP_dag_all,
      TSAMP_ast_PrimaryComp_all, TSAMP_dag_PrimaryComp_all,
      a_i_ast, a_i_dag,
      INTERCEPT_ast_, COEFFICIENTS_ast_,
      INTERCEPT_dag_, COEFFICIENTS_dag_,
      INTERCEPT_ast0_, COEFFICIENTS_ast0_,
      INTERCEPT_dag0_, COEFFICIENTS_dag0_,
      P_VEC_FULL_ast_, P_VEC_FULL_dag_,
      LAMBDA_, Q_SIGN, SEED_IN_LOOP
    )
  })
  
  # ---- Non-adversarial MC evaluator (unchanged) ----
  strenv$Vectorized_QMonteIter <- compile_fxn(
    strenv$jax$vmap( (strenv$QMonteIter <- compile_fxn(
      function(pi_star_ast_f, 
               pi_star_dag_f,
               INTERCEPT_ast_,
               COEFFICIENTS_ast_,
               INTERCEPT_dag_,
               COEFFICIENTS_dag_){
        q_star_ <- QFXN(pi_star_ast_f, 
                        pi_star_dag_f, 
                        INTERCEPT_ast_,  
                        COEFFICIENTS_ast_, 
                        INTERCEPT_dag_,  
                        COEFFICIENTS_dag_)
        return(q_star_)
      })),
    in_axes = list(0L,0L,NULL,NULL,NULL,NULL)))
}

