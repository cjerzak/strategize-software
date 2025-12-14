
InitializeQMonteFxns_new1 <- function(){
  
  # --- Helper: population general-election value for a single (t, u) pair ---
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
  
  # --- Helper: primary head-to-head win prob κ_A(t, t') within A's primary ---
  TEMP_PUSHF <- 1
  kappa_pair <- compile_fxn(function(v, v_prime,
                                     INTERCEPT_0_, COEFFICIENTS_0_){
    strenv$jnp$take(
      getQStar_diff_SingleGroup(
        v,            # entrant
        v_prime,      # field draw
        INTERCEPT_0_, TEMP_PUSHF * COEFFICIENTS_0_,
        INTERCEPT_0_, TEMP_PUSHF * COEFFICIENTS_0_
      ),
      0L)
  })
  
  # --- Binary-concrete (straight-through) sampler for 2-class logits ---
  # Returns y in R^2; forward is hard one-hot, backward uses softmax.
  sample2_st <- compile_fxn(function(logits2, tau, key){
    g <- strenv$jax$random$gumbel(key, logits2$shape)
    y_soft <- strenv$jax$nn$softmax((logits2 + g) / tau, axis = -1L)
    # hard one-hot in forward pass
    idx <- strenv$jnp$argmax(logits2 + g, axis = -1L)
    y_hard <- strenv$jax$nn$one_hot(idx, 2L, dtype = strenv$dtj)
    # straight-through
    y <- strenv$jax$lax$stop_gradient(y_hard - y_soft) + y_soft
    y
  })
  
  # --- Core Monte Carlo evaluator (no linear decomposition; simulate primaries) ---
  strenv$QMonteIter_MaxMin <- compile_fxn(function(
    TSAMP_ast, TSAMP_dag,                        # [n, ·] entrants (MC draws)
    TSAMP_ast_PrimaryComp, TSAMP_dag_PrimaryComp,# [n, ·] field draws (MC)
    a_i_ast, a_i_dag,                            # (unused here; kept for signature)
    INTERCEPT_ast_, COEFFICIENTS_ast_,
    INTERCEPT_dag_, COEFFICIENTS_dag_,
    INTERCEPT_ast0_, COEFFICIENTS_ast0_,
    INTERCEPT_dag0_, COEFFICIENTS_dag0_,
    P_VEC_FULL_ast_, P_VEC_FULL_dag_,           # (unused here)
    LAMBDA_, Q_SIGN, SEED_IN_LOOP               # only SEED_IN_LOOP used
  ){
    
    # Shapes: assume same n across all four inputs (as produced upstream)
    n_draws <- TSAMP_ast$shape[[1]]

    # --- κ for each draw (pairwise entrant vs. its paired field) ---
    # kA[r] = Pr_A_primary( t_i[r] beats t'_r[r] )
    kA_vec <- strenv$jax$vmap(function(t_i, t_prime){
      kappa_pair(t_i, t_prime, INTERCEPT_ast0_, COEFFICIENTS_ast0_)
    }, in_axes = list(0L, 0L))(TSAMP_ast, TSAMP_ast_PrimaryComp)      # [n]
    
    # kB[r] = Pr_B_primary( u_s[r] beats u'_r[r] )
    kB_vec <- strenv$jax$vmap(function(u_s, u_prime){
      kappa_pair(u_s, u_prime, INTERCEPT_dag0_, COEFFICIENTS_dag0_)
    }, in_axes = list(0L, 0L))(TSAMP_dag, TSAMP_dag_PrimaryComp)      # [n]
    
    # --- Sample primary winners (straight-through binary-concrete) ---
    eps <- strenv$jnp$array(1e-6, dtype = strenv$dtj)
    
    # A-primary logits for class {entrant, field}
    logpA <- strenv$jnp$log(strenv$jnp$clip(kA_vec, eps, 1. - eps))
    logqA <- strenv$jnp$log1p(-strenv$jnp$clip(kA_vec, eps, 1. - eps))
    logitsA <- strenv$jnp$stack(list(logpA, logqA), axis = 1L)        # [n,2]
    
    # B-primary logits for class {entrant, field}
    logpB <- strenv$jnp$log(strenv$jnp$clip(kB_vec, eps, 1. - eps))
    logqB <- strenv$jnp$log1p(-strenv$jnp$clip(kB_vec, eps, 1. - eps))
    logitsB <- strenv$jnp$stack(list(logpB, logqB), axis = 1L)        # [n,2]
    
    # Split seed for A and B primaries
    keyAB <- strenv$jax$random$split(SEED_IN_LOOP, 2L)
    keyA  <- keyAB[[1L]]; keyB <- keyAB[[2L]]
    
    # One gumbel tensor per primary (vectorized across draws)
    # (We let sample2_st draw its own gumbels using these keys & shapes.)
    # Draw winners (y[:,0] = choose entrant; y[:,1] = choose field)
    yA <- sample2_st(logitsA, MNtemp, keyA)                           # [n,2]
    yB <- sample2_st(logitsB, MNtemp, keyB)                           # [n,2]
    
    # Convert winners into nominee feature-vectors (convex combination)
    # winner_A[r] = yA[r,0]*t_i[r] + yA[r,1]*t'_r[r]  (straight-through)
    wA0 <- strenv$jnp$expand_dims(strenv$jnp$take(yA, 0L, axis = 1L), 1L)  # [n,1]
    wA1 <- strenv$jnp$expand_dims(strenv$jnp$take(yA, 1L, axis = 1L), 1L)  # [n,1]
    nomA <- strenv$jnp$expand_dims(wA0,1L) * TSAMP_ast + strenv$jnp$expand_dims(wA1,1L) * TSAMP_ast_PrimaryComp                 # [n,·]
    
    wB0 <- strenv$jnp$expand_dims(strenv$jnp$take(yB, 0L, axis = 1L), 1L)  # [n,1]
    wB1 <- strenv$jnp$expand_dims(strenv$jnp$take(yB, 1L, axis = 1L), 1L)  # [n,1]
    nomB <- strenv$jnp$expand_dims(wB0,1L) * TSAMP_dag + strenv$jnp$expand_dims(wB1,1L) * TSAMP_dag_PrimaryComp                 # [n,·]
    
    # --- General election for the (simulated) nominees ---
    # c[r] = Pr_A_wins_general( nomA[r], nomB[r] )
    c_vec <- strenv$jax$vmap(function(t_win, u_win){
      Qpop_pair(t_win, u_win,
                INTERCEPT_ast_, COEFFICIENTS_ast_,
                INTERCEPT_dag_, COEFFICIENTS_dag_)
    }, in_axes = list(0L, 0L))(nomA, nomB)                              # [n]
    
    # Monte Carlo estimate of A's expected general-election win prob
    q_ast <- c_vec$mean(axis = 0L)
    q_dag <- strenv$OneTf_flat - q_ast  # zero-sum
    
    return(list("q_ast" = q_ast,
                "q_dag" = q_dag))
  })
  
  # --- Batch wrapper (kept for compatibility) ---
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
  
  # --- Non-adversarial MC evaluator (unchanged) ---
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

InitializeQMonteFxns_new0 <- function(){
  
  # -- Helper: population general-election value for a single (t, u) pair --
  Qpop_pair <- compile_fxn(function(t, u,
                                    INTERCEPT_ast_, COEFFICIENTS_ast_,
                                    INTERCEPT_dag_, COEFFICIENTS_dag_){
    # getQStar_diff_MultiGroup returns [Q_pop, Q_A|A, Q_A|B]; take population component
    strenv$jnp$take(
      getQStar_diff_MultiGroup(
        t, u,
        INTERCEPT_ast_,  COEFFICIENTS_ast_,
        INTERCEPT_dag_,  COEFFICIENTS_dag_
      ),
      0L
    )
  })
  
  # -- Helper: primary head-to-head win prob κ_A(t, t') within A's primary --
  #    (same form for B, but with its own primary-electorate parameters)
  kappa_pair <- compile_fxn(function(v, v_prime,
                                     INTERCEPT_0_, COEFFICIENTS_0_){
    strenv$jnp$take(
      getQStar_diff_SingleGroup(
        v, v_prime,
        INTERCEPT_0_, COEFFICIENTS_0_,
        INTERCEPT_0_, COEFFICIENTS_0_
      ),
      0L
    )
  })
  
  # -- Helper: sample a relaxed Bernoulli (Binary-Concrete / Gumbel-Sigmoid)
  #    s ~ RelaxedBernoulli(p, tau); returns s \in (0,1) with pathwise gradients.
  sample_relaxed_bernoulli <- compile_fxn(function(prob, key, temperature){
    eps <- strenv$jnp$array(1e-6, strenv$dtj)
    p   <- strenv$jnp$clip(prob, eps, strenv$OneTf_flat - eps)
    u   <- strenv$jax$random$uniform(key, shape=list(), minval=eps, maxval=strenv$OneTf_flat - eps)
    logit_p <- strenv$jnp$log(p) - strenv$jnp$log(strenv$OneTf_flat - p)
    lgu     <- strenv$jnp$log(u) - strenv$jnp$log(strenv$OneTf_flat - u)  # logistic noise
    strenv$jax$nn$sigmoid( (logit_p + lgu) / temperature )
  })
  
  # -- One Monte Carlo step: take one banked entrant & one field draw per party,
  #    sample primary winners (relaxed), and evaluate general-election value.
  one_mc_step <- compile_fxn(function(t_i, t_field, u_s, u_field,
                                      keyA, keyB,
                                      INTERCEPT_ast_, COEFFICIENTS_ast_,
                                      INTERCEPT_dag_, COEFFICIENTS_dag_,
                                      INTERCEPT_ast0_, COEFFICIENTS_ast0_,
                                      INTERCEPT_dag0_, COEFFICIENTS_dag0_){
    
    # Primary win probabilities for the paired head-to-heads
    kA <- kappa_pair(t_i, t_field, INTERCEPT_ast0_, COEFFICIENTS_ast0_)
    kB <- kappa_pair(u_s, u_field, INTERCEPT_dag0_, COEFFICIENTS_dag0_)
    
    # Relaxed (differentiable) winner indicators
    sA <- sample_relaxed_bernoulli(kA, keyA, MNtemp)  # MNtemp is the same temperature used for profile sampling
    sB <- sample_relaxed_bernoulli(kB, keyB, MNtemp)
    
    # Broadcast to profile vectors and form nominees as convex combinations
    sA <- strenv$jnp$expand_dims(sA, 0L)
    sB <- strenv$jnp$expand_dims(sB, 0L)
    t_nom <- sA * t_i + (strenv$OneTf_flat - sA) * t_field
    u_nom <- sB * u_s + (strenv$OneTf_flat - sB) * u_field
    
    # General election expected A share against B
    Qpop_pair(t_nom, u_nom,
              INTERCEPT_ast_, COEFFICIENTS_ast_,
              INTERCEPT_dag_, COEFFICIENTS_dag_)
  })
  
  # === Core Monte Carlo evaluator (institution-aware), no linearity tricks ===
  # Expects matching banks of samples for entrants and fields on both sides.
  strenv$QMonteIter_MaxMin <- compile_fxn(function(
    TSAMP_ast, TSAMP_dag,                        # [M, ·], [M, ·] entrants   (A, B)
    TSAMP_ast_PrimaryComp, TSAMP_dag_PrimaryComp,# [M, ·], [M, ·] field banks (A, B)
    a_i_ast, a_i_dag,                            # (unused; kept for signature parity)
    INTERCEPT_ast_, COEFFICIENTS_ast_,           # general-election A
    INTERCEPT_dag_, COEFFICIENTS_dag_,           # general-election B
    INTERCEPT_ast0_, COEFFICIENTS_ast0_,         # A primary electorate
    INTERCEPT_dag0_, COEFFICIENTS_dag0_,         # B primary electorate
    P_VEC_FULL_ast_, P_VEC_FULL_dag_,            # (unused here; reserved)
    LAMBDA_, Q_SIGN, SEED_IN_LOOP                # (Q_SIGN unused here; SEED used)
  ){
    
    # All four banks are constructed with the same Monte count M upstream
    M <- TSAMP_ast$shape[[1]]
    
    # Keys for relaxed Bernoulli sampling at the primary stage
    keys <- strenv$jax$random$split(SEED_IN_LOOP, 2L * M + 1L)
    # First M for A, next M for B (drop the leftover)
    idxA <- strenv$jnp$array( ai(1:M) - 1L )
    idxB <- strenv$jnp$array( ai((M+1):(2L*M)) - 1L )
    keysA <- strenv$jnp$take(keys, idxA, axis=0L)
    keysB <- strenv$jnp$take(keys, idxB, axis=0L)
    
    # Vmap the single-step simulator across the M paired draws
    q_vec <- strenv$jax$vmap(function(t_i, t_field, u_s, u_field, keyA, keyB){
      one_mc_step(
        t_i, t_field, u_s, u_field, keyA, keyB,
        INTERCEPT_ast_, COEFFICIENTS_ast_,
        INTERCEPT_dag_, COEFFICIENTS_dag_,
        INTERCEPT_ast0_, COEFFICIENTS_ast0_,
        INTERCEPT_dag0_, COEFFICIENTS_dag0_
      )
    }, in_axes = list(0L, 0L, 0L, 0L, 0L, 0L))(
      TSAMP_ast, TSAMP_ast_PrimaryComp,
      TSAMP_dag, TSAMP_dag_PrimaryComp,
      keysA, keysB
    )
    
    # Aggregate Monte Carlo estimates
    q_ast <- q_vec$mean()                 # expected A share
    q_dag <- strenv$OneTf_flat - q_ast    # zero-sum complement
    
    return(list("q_ast" = q_ast, "q_dag" = q_dag))
  })
  
  # Batch wrapper (unchanged interface)
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

InitializeQMonteFxns_linearized <- function(){
  
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
  TEMP_PUSHF <- 1
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
      }, in_axes = 0L)( strenv$jax$lax$stop_gradient( TSAMP_ast_PrimaryComp ) )
    }, in_axes = 0L)(TSAMP_ast)                             # [nA, nA']
    
    # κ_B[s,r] = Pr_B_primary( u_s beats u'_r )
    kB <- strenv$jax$vmap(function(u_s){
      strenv$jax$vmap(function(up_r){
        kappa_pair(u_s, up_r, INTERCEPT_dag0_, COEFFICIENTS_dag0_)
      }, in_axes = 0L)( strenv$jax$lax$stop_gradient(TSAMP_dag_PrimaryComp) )
    }, in_axes = 0L)(TSAMP_dag)                             # [nB, nB']
    
    #sanity checks 
    # hist(strenv$np$array(TSAMP_ast))
    # hist(strenv$np$array(TSAMP_ast_PrimaryComp))
    # causalimages::image2(strenv$np$array(kA))
    # View(strenv$np$array(kA))
    
    # Averages for push-forward weights
    kA_mean_over_field   <- kA$mean(axis=1L)  # [nA]   E_field[κ | entrant]
    kB_mean_over_field   <- kB$mean(axis=1L)  # [nB]
    kA_mean_over_entrant <- kA$mean(axis=0L)  # [nA']  E_entrant[κ | field]
    kB_mean_over_entrant <- kB$mean(axis=0L)  # [nB']
    
    # ---- General-election blocks for the four primary outcomes (2×2) ----
    {
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
    }
    
    # ---- Push-forward mixture weights ----
    one <- strenv$OneTf_flat 
    { 
      W1 <- strenv$jnp$outer(kA_mean_over_field,           kB_mean_over_field)             # [nA,  nB]
      W2 <- strenv$jnp$outer(kA_mean_over_field,           one - kB_mean_over_entrant)     # [nA,  nB']
      W3 <- strenv$jnp$outer(one - kA_mean_over_entrant,   kB_mean_over_field)             # [nA', nB]
      W4 <- strenv$jnp$outer(one - kA_mean_over_entrant,   one - kB_mean_over_entrant)     # [nA', nB']
      
      if(FALSE){
        # entrant wins its primary against H opponents
        H <- 3
        kA_win <- strenv$jnp$power(kA_mean_over_field,H)    # [nA]
        kB_win <- strenv$jnp$power(kB_mean_over_field,H)    # [nB]
        
        # field ends up nominee if the entrant does NOT beat the field pack
        # (use the analogous hardening for the field)
        A_field_wins <- 1. - (strenv$jnp$power(kA_mean_over_entrant, H))  # [nA']
        B_field_wins <- 1. - (strenv$jnp$power(kB_mean_over_entrant, H))  # [nB']
  
        W1 <- strenv$jnp$outer(kA_win,          kB_win)         # entrant vs entrant
        W2 <- strenv$jnp$outer(kA_win,          B_field_wins)   # entrant vs field
        W3 <- strenv$jnp$outer(A_field_wins,    kB_win)         # field   vs entrant
        W4 <- strenv$jnp$outer(A_field_wins,    B_field_wins)   # field   vs field
      }
      
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
    q_ast <- E1 + E2 + E3 + E4
    q_dag <- one - q_ast  # zero-sum
    
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

InitializeQMonteFxns <- function(){
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

        # win based on relaxed sample from bernoulli using base JAX
        Indicator_AstWinsPrimary <- strenv$jax$nn$sigmoid(
          (strenv$jnp$log(PrimaryVoteShareAstAmongAst / (1 - PrimaryVoteShareAstAmongAst)) +
             strenv$jax$random$gumbel(SEED_IN_LOOP)) / MNtemp)
        SEED_IN_LOOP   <- strenv$jax$random$split(SEED_IN_LOOP)[[1L]]
        Indicator_DagWinsPrimary <- strenv$jax$nn$sigmoid(
          (strenv$jnp$log(PrimaryVoteShareDagAmongDag / (1 - PrimaryVoteShareDagAmongDag)) +
             strenv$jax$random$gumbel(SEED_IN_LOOP)) / MNtemp)
        SEED_IN_LOOP   <- strenv$jax$random$split(SEED_IN_LOOP)[[1L]]
      }
      
      # primaries 
      PrAstWinsAstPrimary <- PrimaryVoteShareAstAmongAst$mean()
      PrAstLosesAstPrimary <- 1-PrAstWinsAstPrimary
      PrDagWinsDagPrimary <- PrimaryVoteShareDagAmongDag$mean()
      PrDagLosesDagPrimary <- 1-PrDagWinsDagPrimary
      
      # generals 
      # Compute  average vote shares in the subset of simulations where candidate wins its primary.
      # Specifically, sum of votes in that subset divided by (sum of total votes in that subset + epsilon)
      
      # among ast 
      Indicator_AstWinsPrimary <- Indicator_AstWinsPrimary # soft or hard 
      GVShareAstAmongAst_Given_AstWinsAstPrimary <- 
        ( GVShareAstAmongAst*Indicator_AstWinsPrimary )$sum()/ 
        ( (ep_<-0.01) + Indicator_AstWinsPrimary$sum() )
      GVShareDagAmongAst_Given_AstWinsAstPrimary <-  1 - GVShareAstAmongAst_Given_AstWinsAstPrimary
      
      # among dag 
      Indicator_DagWinsPrimary <- Indicator_DagWinsPrimary
      GVShareAstAmongDag_Given_DagWinsDagPrimary <- 
        (  GVShareAstAmongDag*Indicator_DagWinsPrimary )$sum() / 
        ( ep_ + Indicator_DagWinsPrimary$sum() )
      GVShareDagAmongDag_Given_DagWinsDagPrimary <- 1 - GVShareAstAmongDag_Given_DagWinsDagPrimary
      
      if(T == T){
        # compute expected value 
        GVShareDagAmongAst <- 1-GVShareAstAmongAst
        GVShareDagAmongDag <- 1-GVShareAstAmongDag
        
        Indicator_DagLosesPrimary <- 1-Indicator_DagWinsPrimary
        Indicator_AstLosesPrimary <- 1-Indicator_AstWinsPrimary
        
        # 2 by 2 AMONG AST 
        {
          # 1 ast 
          E_VoteShare_Ast_AmongAst_Given_AstWinsPrimary_DagWinsPrimary <-
            ((GVShareAstAmongAst)*
               (event_ <- (Indicator_AstWinsPrimary*Indicator_DagWinsPrimary) ))$sum() / 
            (0.001+(event_))$sum()
          
          # 2 ast 
          E_VoteShare_Ast_AmongAst_Given_AstLosesPrimary_DagLosesPrimary <-
            ((GVShareAstAmongAst)*
               ( event_ <- (Indicator_AstLosesPrimary*Indicator_DagLosesPrimary) ))$sum() / 
            (0.001+(event_))$sum()
          
          # 3 ast
          E_VoteShare_Ast_AmongAst_Given_AstLosesPrimary_DagWinsPrimary <-
            ((GVShareAstAmongAst)*
               (event_ <- (Indicator_AstLosesPrimary*Indicator_DagWinsPrimary) ))$sum() / 
            (0.001+(event_))$sum()
          
          # 4 ast
          E_VoteShare_Ast_AmongAst_Given_AstWinsPrimary_DagLosesPrimary <-
            ((GVShareAstAmongAst)*
               (event_ <- (Indicator_AstWinsPrimary*Indicator_DagLosesPrimary) ))$sum() / 
            (0.001+(event_))$sum()
          E_VoteShare_Dag_AmongAst_Given_AstWinsPrimary_DagWinsPrimary <- 1-E_VoteShare_Ast_AmongAst_Given_AstWinsPrimary_DagWinsPrimary
          E_VoteShare_Dag_AmongAst_Given_AstLosesPrimary_DagLosesPrimary <- 1-E_VoteShare_Ast_AmongAst_Given_AstLosesPrimary_DagLosesPrimary
          E_VoteShare_Dag_AmongAst_Given_AstLosesPrimary_DagWinsPrimary <- 1-E_VoteShare_Ast_AmongAst_Given_AstLosesPrimary_DagWinsPrimary
          E_VoteShare_Dag_AmongAst_Given_AstWinsPrimary_DagLosesPrimary <- 1-E_VoteShare_Ast_AmongAst_Given_AstWinsPrimary_DagLosesPrimary
        }
        
        # 2 by 2 among dag 
        {
          # 1 
          E_VoteShare_Ast_AmongDag_Given_AstWinsPrimary_DagWinsPrimary <-
            ((GVShareAstAmongDag)*
               (event_ <- (Indicator_AstWinsPrimary*Indicator_DagWinsPrimary) ))$sum() / 
            (0.001+(event_))$sum()
          
          # 2
          E_VoteShare_Ast_AmongDag_Given_AstLosesPrimary_DagLosesPrimary <-
            ((GVShareAstAmongDag)*
               ( event_ <- (Indicator_AstLosesPrimary*Indicator_DagLosesPrimary) ))$sum() / 
            (0.001+(event_))$sum()
          
          # 3 - 
          E_VoteShare_Ast_AmongDag_Given_AstLosesPrimary_DagWinsPrimary <-
            ((GVShareAstAmongDag)*
               (event_ <- (Indicator_AstLosesPrimary*Indicator_DagWinsPrimary) ))$sum() / 
            (0.001+(event_))$sum()
          
          # 4
          E_VoteShare_Ast_AmongDag_Given_AstWinsPrimary_DagLosesPrimary <-
            ( (GVShareAstAmongDag) * (event_ <- (Indicator_AstWinsPrimary*Indicator_DagLosesPrimary) ))$sum() / 
            (0.001+(event_))$sum()
          E_VoteShare_Dag_AmongDag_Given_AstWinsPrimary_DagWinsPrimary <- 1-E_VoteShare_Ast_AmongDag_Given_AstWinsPrimary_DagWinsPrimary
          E_VoteShare_Dag_AmongDag_Given_AstLosesPrimary_DagLosesPrimary <- 1-E_VoteShare_Ast_AmongDag_Given_AstLosesPrimary_DagLosesPrimary
          E_VoteShare_Dag_AmongDag_Given_AstLosesPrimary_DagWinsPrimary <- 1-E_VoteShare_Ast_AmongDag_Given_AstLosesPrimary_DagWinsPrimary
          E_VoteShare_Dag_AmongDag_Given_AstWinsPrimary_DagLosesPrimary <- 1-E_VoteShare_Ast_AmongDag_Given_AstWinsPrimary_DagLosesPrimary
        }
        
        E_VoteShare_Ast <- (
          E_VoteShare_Ast_AmongAst_Given_AstWinsPrimary_DagWinsPrimary * PrAstWinsAstPrimary * PrDagWinsDagPrimary * strenv$AstProp +
            0*E_VoteShare_Ast_AmongAst_Given_AstLosesPrimary_DagLosesPrimary * PrAstLosesAstPrimary * PrDagLosesDagPrimary * strenv$AstProp +
            E_VoteShare_Ast_AmongAst_Given_AstWinsPrimary_DagLosesPrimary * PrAstWinsAstPrimary * PrDagLosesDagPrimary * strenv$AstProp +
            0*E_VoteShare_Ast_AmongAst_Given_AstLosesPrimary_DagWinsPrimary * PrAstLosesAstPrimary * PrDagWinsDagPrimary * strenv$AstProp +
            
            E_VoteShare_Ast_AmongDag_Given_AstWinsPrimary_DagWinsPrimary * PrAstWinsAstPrimary * PrDagWinsDagPrimary * strenv$DagProp +
            0*E_VoteShare_Ast_AmongDag_Given_AstLosesPrimary_DagLosesPrimary * PrAstLosesAstPrimary * PrDagLosesDagPrimary * strenv$DagProp +
            E_VoteShare_Ast_AmongDag_Given_AstWinsPrimary_DagLosesPrimary * PrAstWinsAstPrimary * PrDagLosesDagPrimary * strenv$DagProp +
            0*E_VoteShare_Ast_AmongDag_Given_AstLosesPrimary_DagWinsPrimary * PrAstLosesAstPrimary * PrDagWinsDagPrimary * strenv$DagProp
        )
        
        E_VoteShare_Dag <- (
          E_VoteShare_Dag_AmongAst_Given_AstWinsPrimary_DagWinsPrimary * PrAstWinsAstPrimary * PrDagWinsDagPrimary * strenv$AstProp +
            0*E_VoteShare_Dag_AmongAst_Given_AstLosesPrimary_DagLosesPrimary * PrAstLosesAstPrimary * PrDagLosesDagPrimary * strenv$AstProp +
            0*E_VoteShare_Dag_AmongAst_Given_AstWinsPrimary_DagLosesPrimary * PrAstWinsAstPrimary * PrDagLosesDagPrimary * strenv$AstProp +
            E_VoteShare_Dag_AmongAst_Given_AstLosesPrimary_DagWinsPrimary * PrAstLosesAstPrimary * PrDagWinsDagPrimary * strenv$AstProp +
            
            E_VoteShare_Dag_AmongDag_Given_AstWinsPrimary_DagWinsPrimary * PrAstWinsAstPrimary * PrDagWinsDagPrimary * strenv$DagProp +
            0*E_VoteShare_Dag_AmongDag_Given_AstLosesPrimary_DagLosesPrimary * PrAstLosesAstPrimary * PrDagLosesDagPrimary * strenv$DagProp +
            0*E_VoteShare_Dag_AmongDag_Given_AstWinsPrimary_DagLosesPrimary * PrAstWinsAstPrimary * PrDagLosesDagPrimary * strenv$DagProp +
            E_VoteShare_Dag_AmongDag_Given_AstLosesPrimary_DagWinsPrimary * PrAstLosesAstPrimary * PrDagWinsDagPrimary * strenv$DagProp
        )
      }
      
      # quantity to maximize for ast and dag respectively 
      return(list("q_ast" = E_VoteShare_Ast, 
                  "q_dag" = E_VoteShare_Dag))
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



