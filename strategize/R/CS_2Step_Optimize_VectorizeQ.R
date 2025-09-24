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
  TEMP_PUSHF <- 1 # <<<<-------- highier values indicate 
  kappa_pair <- compile_fxn(function(v, v_prime,
                                       INTERCEPT_0_, COEFFICIENTS_0_){
    strenv$jnp$take(
      getQStar_diff_SingleGroup(
        v,            # entrant
        v_prime,      # field draw
        INTERCEPT_0_, TEMP_PUSHF * COEFFICIENTS_0_,
        INTERCEPT_0_, TEMP_PUSHF * COEFFICIENTS_0_
      ), 0L)
  })
  
  # === Core Monte-Carlo evaluator that respects the push-forward ===
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
        kappa_pair(t_i, tp_j, INTERCEPT_ast0_, COEFFICIENTS_ast0_)
      }, in_axes = 0L)(TSAMP_ast_PrimaryComp)
    }, in_axes = 0L)(TSAMP_ast)                             # [nA, nA']
    
    # κ_B[s,r] = Pr_B_primary( u_s beats u'_r )
    kB <- strenv$jax$vmap(function(u_s){
      strenv$jax$vmap(function(up_r){
        kappa_pair(u_s, up_r, INTERCEPT_dag0_, COEFFICIENTS_dag0_)
      }, in_axes = 0L)(TSAMP_dag_PrimaryComp)
    }, in_axes = 0L)(TSAMP_dag)                             # [nB, nB']
    
    # Averages for push-forward weights
    kA_mean_over_field   <- kA$mean(axis=1L)  # [nA]   E_field[κ | entrant]
    kB_mean_over_field   <- kB$mean(axis=1L)  # [nB]
    kA_mean_over_entrant <- kA$mean(axis=0L)  # [nA']  E_entrant[κ | field]
    kB_mean_over_entrant <- kB$mean(axis=0L)  # [nB']

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
    
    # ---- Push-forward mixture weights ----
    { 
    one <- strenv$OneTf_flat 
    W1 <- strenv$jnp$outer(kA_mean_over_field,           kB_mean_over_field)             # [nA,  nB]
    W2 <- strenv$jnp$outer(kA_mean_over_field,           one - kB_mean_over_entrant)     # [nA,  nB']
    W3 <- strenv$jnp$outer(one - kA_mean_over_entrant,   kB_mean_over_field)             # [nA', nB]
    W4 <- strenv$jnp$outer(one - kA_mean_over_entrant,   one - kB_mean_over_entrant)     # [nA', nB']
    
    # ---- Expected general-election vote share for A ----
    E1 <- (C_tu          * W1)$mean()   # A entrant vs B entrant
    E2 <- (C_tu_field    * W2)$mean()   # A entrant vs B field
    E3 <- (C_field_u     * W3)$mean()   # A field   vs B entrant
    E4 <- (C_field_field * W4)$mean()   # A field   vs B field
    }
    
    # Exploit linearity of expectation and the mixture structure: 
    # the pushforward decomposes into four weighted blocks (E1--E4 )
    # corresponding to the primary outcomes (entrant vs. entrant, entrant vs. field, etc.), 
    # each approximated directly over samples
    (q_ast <- E1 + E2 + E3 + E4)
    q_dag <- one  - q_ast  # zero-sum
    
    return(list("q_ast" = q_ast, 
                "q_dag" = q_dag))
  })
  
  # Batch wrapper 
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
  
  # ---- Non-adversarial MC evaluator ----
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

