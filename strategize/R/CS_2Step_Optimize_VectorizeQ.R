InitializeQMonteFxns <- function(){ 
  # The linearized version exploits:                                                                                                                          
  # 1. Independence of A and B primaries                                                                                                                      
  # 2. Linearity of expectation                                                                                                                               
  # 3. Precomputation of all C combinations 
  
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
    
    # ---- Push-forward mixture weights (profile-specific) ----
    # Eq. (PrimaryPushforward) implies weights depend on each profile's primary win rate.
    one <- strenv$OneTf_flat
    {
      nA <- strenv$jnp$array(kA$shape[[1L]])
      nA_field <- strenv$jnp$array(kA$shape[[2L]])
      nB <- strenv$jnp$array(kB$shape[[1L]])
      nB_field <- strenv$jnp$array(kB$shape[[2L]])

      # Per-sample nominee weights (entrant vs field)
      wA_e <- kA_mean_over_field / nA
      wA_f <- (one - kA_mean_over_entrant) / nA_field
      wB_e <- kB_mean_over_field / nB
      wB_f <- (one - kB_mean_over_entrant) / nB_field

      # Broadcast weights over the four primary outcome blocks
      wA_e_col <- strenv$jnp$expand_dims(wA_e, 1L)
      wA_f_col <- strenv$jnp$expand_dims(wA_f, 1L)
      wB_e_row <- strenv$jnp$expand_dims(wB_e, 0L)
      wB_f_row <- strenv$jnp$expand_dims(wB_f, 0L)

      E1 <- strenv$jnp$sum(C_tu * wA_e_col * wB_e_row)            # A entrant vs B entrant
      E2 <- strenv$jnp$sum(C_tu_field * wA_e_col * wB_f_row)      # A entrant vs B field
      E3 <- strenv$jnp$sum(C_field_u * wA_f_col * wB_e_row)       # A field   vs B entrant
      E4 <- strenv$jnp$sum(C_field_field * wA_f_col * wB_f_row)   # A field   vs B field
    }

    # Exploit linearity of expectation and the mixture structure:
    # the pushforward decomposes into four weighted blocks (E1--E4)
    # corresponding to the primary outcomes (entrant vs. entrant, entrant vs. field, etc.)
    # Weights w1+w2+w3+w4 = 1 by construction, ensuring proper normalization
    q_ast <- E1 + E2 + E3 + E4
    q_dag <- one - q_ast  # zero-sum

    # Wrap scalars in expand_dims for consistency with MC mode's array return
    # This ensures q_ast.mean(0) works correctly in getQPiStar_gd
    return(list("q_ast" = strenv$jnp$expand_dims(q_ast, 0L),
                "q_dag" = strenv$jnp$expand_dims(q_dag, 0L)))
  })

  # Optional pushforward self-test (runs once when enabled)
  if (isTRUE(strenv$UNIT_TEST_PUSHFORWARD) && !isTRUE(strenv$UNIT_TEST_PUSHFORWARD_DONE)) {
    strenv$UNIT_TEST_PUSHFORWARD_DONE <- TRUE

    dim_pi <- if (exists("p_vec_tf", inherits = TRUE)) {
      as.integer(strenv$jnp$shape(p_vec_tf)[[1]])
    } else {
      as.integer(main_indices_i0$size)
    }

    if (dim_pi >= 1L) {
      nA_samp <- 2L
      nB_samp <- 2L
      nA_field_samp <- 2L
      nB_field_samp <- 2L

      make_mat <- function(offset, n_rows) {
        vals <- seq(0.05 + offset, 0.85 + offset, length.out = n_rows * dim_pi)
        strenv$jnp$array(matrix(vals, nrow = n_rows, byrow = TRUE), dtype = strenv$dtj)
      }

      TSAMP_ast <- make_mat(0.00, nA_samp)
      TSAMP_dag <- make_mat(0.02, nB_samp)
      TSAMP_ast_PrimaryComp <- make_mat(0.04, nA_field_samp)
      TSAMP_dag_PrimaryComp <- make_mat(0.06, nB_field_samp)

      coef_len <- if (exists("EST_COEFFICIENTS_tf_ast_jnp", inherits = TRUE)) {
        as.integer(EST_COEFFICIENTS_tf_ast_jnp$size)
      } else {
        as.integer(strenv$np$array(strenv$jnp$max(main_indices_i0))) + 1L
      }
      coef_len <- max(1L, coef_len)

      COEF <- strenv$jnp$reshape(strenv$jnp$linspace(-0.2, 0.2, coef_len),
                                 list(coef_len, 1L))
      INT <- strenv$jnp$array(0.1, dtype = strenv$dtj)

      res <- strenv$QMonteIter_MaxMin(
        TSAMP_ast, TSAMP_dag, TSAMP_ast_PrimaryComp, TSAMP_dag_PrimaryComp,
        INT, INT,
        INT, COEF, INT, COEF,
        INT, COEF, INT, COEF,
        INT, INT, INT, 1.0, INT
      )

      # Rebuild kA/kB and the four C blocks
      kA <- strenv$jax$vmap(function(t_i){
        strenv$jax$vmap(function(tp_j){
          kappa_pair(t_i, tp_j, INT, COEF)
        }, in_axes = 0L)(TSAMP_ast_PrimaryComp)
      }, in_axes = 0L)(TSAMP_ast)

      kB <- strenv$jax$vmap(function(u_s){
        strenv$jax$vmap(function(up_r){
          kappa_pair(u_s, up_r, INT, COEF)
        }, in_axes = 0L)(TSAMP_dag_PrimaryComp)
      }, in_axes = 0L)(TSAMP_dag)

      C_tu <- strenv$jax$vmap(function(t_i){
        strenv$jax$vmap(function(u_s){
          Qpop_pair(t_i, u_s, INT, COEF, INT, COEF)
        }, in_axes = 0L)(TSAMP_dag)
      }, in_axes = 0L)(TSAMP_ast)

      C_tu_field <- strenv$jax$vmap(function(t_i){
        strenv$jax$vmap(function(up_r){
          Qpop_pair(t_i, up_r, INT, COEF, INT, COEF)
        }, in_axes = 0L)(TSAMP_dag_PrimaryComp)
      }, in_axes = 0L)(TSAMP_ast)

      C_field_u <- strenv$jax$vmap(function(tp_j){
        strenv$jax$vmap(function(u_s){
          Qpop_pair(tp_j, u_s, INT, COEF, INT, COEF)
        }, in_axes = 0L)(TSAMP_dag)
      }, in_axes = 0L)(TSAMP_ast_PrimaryComp)

      C_field_field <- strenv$jax$vmap(function(tp_j){
        strenv$jax$vmap(function(up_r){
          Qpop_pair(tp_j, up_r, INT, COEF, INT, COEF)
        }, in_axes = 0L)(TSAMP_dag_PrimaryComp)
      }, in_axes = 0L)(TSAMP_ast_PrimaryComp)

      kA_mean_over_field <- kA$mean(axis=1L)
      kB_mean_over_field <- kB$mean(axis=1L)
      kA_mean_over_entrant <- kA$mean(axis=0L)
      kB_mean_over_entrant <- kB$mean(axis=0L)

      nA <- strenv$jnp$array(kA$shape[[1L]])
      nA_field <- strenv$jnp$array(kA$shape[[2L]])
      nB <- strenv$jnp$array(kB$shape[[1L]])
      nB_field <- strenv$jnp$array(kB$shape[[2L]])

      wA_e <- kA_mean_over_field / nA
      wA_f <- (strenv$OneTf_flat - kA_mean_over_entrant) / nA_field
      wB_e <- kB_mean_over_field / nB
      wB_f <- (strenv$OneTf_flat - kB_mean_over_entrant) / nB_field

      # Explicit 4D expectation over primaries (no aggregation)
      inv_norm <- 1.0 / (nA * nB * nA_field * nB_field)
      kA_4 <- strenv$jnp$reshape(kA, list(nA, 1L, nA_field, 1L))
      kB_4 <- strenv$jnp$reshape(kB, list(1L, nB, 1L, nB_field))
      C_tu_4 <- strenv$jnp$reshape(C_tu, list(nA, nB, 1L, 1L))
      C_tu_field_4 <- strenv$jnp$reshape(C_tu_field, list(nA, 1L, 1L, nB_field))
      C_field_u_4 <- strenv$jnp$reshape(C_field_u, list(1L, nB, nA_field, 1L))
      C_field_field_4 <- strenv$jnp$reshape(C_field_field, list(1L, 1L, nA_field, nB_field))

      q_direct <- inv_norm * strenv$jnp$sum(
        kA_4 * kB_4 * C_tu_4 +
        kA_4 * (strenv$OneTf_flat - kB_4) * C_tu_field_4 +
        (strenv$OneTf_flat - kA_4) * kB_4 * C_field_u_4 +
        (strenv$OneTf_flat - kA_4) * (strenv$OneTf_flat - kB_4) * C_field_field_4
      )

      tol <- 1e-5
      if (!strenv$np$allclose(strenv$np$array(res$q_ast), strenv$np$array(q_direct),
                              rtol = tol, atol = tol)) {
        stop("Pushforward test failed: explicit expectation mismatch")
      }
      if (!strenv$np$allclose(strenv$np$array(strenv$jnp$sum(wA_e) + strenv$jnp$sum(wA_f)),
                              strenv$np$array(strenv$OneTf_flat), rtol = tol, atol = tol) ||
          !strenv$np$allclose(strenv$np$array(strenv$jnp$sum(wB_e) + strenv$jnp$sum(wB_f)),
                              strenv$np$array(strenv$OneTf_flat), rtol = tol, atol = tol)) {
        stop("Pushforward test failed: weight normalization")
      }
      if (!strenv$np$allclose(strenv$np$array(res$q_ast + res$q_dag),
                              strenv$np$array(strenv$OneTf_flat), rtol = tol, atol = tol)) {
        stop("Pushforward test failed: zero-sum")
      }
    }
  }
  
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

InitializeQMonteFxns_MCSampling <- function(){
  #The default version uses:
  #  1. Monte Carlo integration over profile draws
  #  2. Exact inner expectations over primary outcomes
  
  RelaxedBernoulli <- compile_fxn(function(p, seed, temp){
    eps <- 1e-6
    p <- strenv$jnp$clip(p, eps, 1.0 - eps)
    logit_p <- strenv$jnp$log(p) - strenv$jnp$log1p(-p)
    g1 <- strenv$jax$random$gumbel(seed)
    seed <- strenv$jax$random$split(seed)[[1L]]
    g2 <- strenv$jax$random$gumbel(seed)
    seed <- strenv$jax$random$split(seed)[[1L]]
    z <- strenv$jax$nn$sigmoid((logit_p + g1 - g2) / temp)
    list(z, seed)
  })
  
  Qpop_pair <- compile_fxn(function(t, u,
                                    INTERCEPT_ast_, COEFFICIENTS_ast_,
                                    INTERCEPT_dag_, COEFFICIENTS_dag_){
    strenv$jnp$take(
      getQStar_diff_MultiGroup(
        t, u,
        INTERCEPT_ast_, COEFFICIENTS_ast_,
        INTERCEPT_dag_, COEFFICIENTS_dag_
      ),
      0L
    )
  })
  
  kappa_pair <- compile_fxn(function(v, v_prime,
                                    INTERCEPT_0_, COEFFICIENTS_0_){
    strenv$jnp$take(
      getQStar_diff_SingleGroup(
        v,
        v_prime,
        INTERCEPT_0_, COEFFICIENTS_0_,
        INTERCEPT_0_, COEFFICIENTS_0_
      ),
      0L
    )
  })
  
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
      kA <- kappa_pair(TSAMP_ast, TSAMP_ast_PrimaryComp, INTERCEPT_ast0_, COEFFICIENTS_ast0_)
      kB <- kappa_pair(TSAMP_dag, TSAMP_dag_PrimaryComp, INTERCEPT_dag0_, COEFFICIENTS_dag0_)
      
      one <- strenv$OneTf_flat

      C_tt <- Qpop_pair(TSAMP_ast, TSAMP_dag,
                        INTERCEPT_ast_, COEFFICIENTS_ast_,
                        INTERCEPT_dag_, COEFFICIENTS_dag_)
      C_tf <- Qpop_pair(TSAMP_ast, TSAMP_dag_PrimaryComp,
                        INTERCEPT_ast_, COEFFICIENTS_ast_,
                        INTERCEPT_dag_, COEFFICIENTS_dag_)
      C_ft <- Qpop_pair(TSAMP_ast_PrimaryComp, TSAMP_dag,
                        INTERCEPT_ast_, COEFFICIENTS_ast_,
                        INTERCEPT_dag_, COEFFICIENTS_dag_)
      C_ff <- Qpop_pair(TSAMP_ast_PrimaryComp, TSAMP_dag_PrimaryComp,
                        INTERCEPT_ast_, COEFFICIENTS_ast_,
                        INTERCEPT_dag_, COEFFICIENTS_dag_)

      q_ast <- kA * kB * C_tt +
               kA * (one - kB) * C_tf +
               (one - kA) * kB * C_ft +
               (one - kA) * (one - kB) * C_ff
      list("q_ast" = q_ast,
           "q_dag" = one - q_ast)
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
