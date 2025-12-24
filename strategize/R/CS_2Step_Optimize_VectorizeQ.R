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
    
    # ---- Push-forward mixture weights ----
    # Fix: Use consistent scalar mixture probabilities to ensure proper normalization
    # Previously used different marginals (kA_mean_over_field vs kA_mean_over_entrant)
    # which created inconsistent weights that don't sum to 1
    one <- strenv$OneTf_flat
    {
      # Compute scalar mixture probabilities (averaged over all samples)
      # P(A entrant wins primary) = E_{entrant, field}[kappa_A(entrant, field)]
      P_A_entrant_wins <- kA$mean()  # scalar: overall prob A's entrant wins
      P_B_entrant_wins <- kB$mean()  # scalar: overall prob B's entrant wins
      P_A_field_wins <- one - P_A_entrant_wins
      P_B_field_wins <- one - P_B_entrant_wins

      # Scalar mixture weights for the four quadrants (sum to 1)
      w1 <- P_A_entrant_wins * P_B_entrant_wins      # A entrant vs B entrant
      w2 <- P_A_entrant_wins * P_B_field_wins        # A entrant vs B field
      w3 <- P_A_field_wins * P_B_entrant_wins        # A field   vs B entrant
      w4 <- P_A_field_wins * P_B_field_wins          # A field   vs B field

      # ---- Expected general-election vote share for A ----
      # Each block uses unweighted mean of C values, then weighted by mixture probability
      E1 <- C_tu$mean()          * w1   # A entrant vs B entrant
      E2 <- C_tu_field$mean()    * w2   # A entrant vs B field
      E3 <- C_field_u$mean()     * w3   # A field   vs B entrant
      E4 <- C_field_field$mean() * w4   # A field   vs B field
    }

    # Exploit linearity of expectation and the mixture structure:
    # the pushforward decomposes into four weighted blocks (E1--E4)
    # corresponding to the primary outcomes (entrant vs. entrant, entrant vs. field, etc.)
    # Weights w1+w2+w3+w4 = 1 by construction, ensuring proper normalization
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

InitializeQMonteFxns_MCSampling <- function(){
  #The default version uses:                                                                                                                                 
  #  1. Monte Carlo integration with soft indicators                                                                                                           
  # 2. Conditional expectations computed on-the-fly                                                                                                           
  # 3. Gumbel-softmax for differentiable sampling                                                                                                             
  
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
        
        # Full push-forward: include all four primary outcome scenarios
        # Previously some terms were zeroed with 0* which ignored scenarios where entrants lose
        E_VoteShare_Ast <- (
          # Among Ast voters (weighted by AstProp)
          E_VoteShare_Ast_AmongAst_Given_AstWinsPrimary_DagWinsPrimary * PrAstWinsAstPrimary * PrDagWinsDagPrimary * strenv$AstProp +
            E_VoteShare_Ast_AmongAst_Given_AstLosesPrimary_DagLosesPrimary * PrAstLosesAstPrimary * PrDagLosesDagPrimary * strenv$AstProp +
            E_VoteShare_Ast_AmongAst_Given_AstWinsPrimary_DagLosesPrimary * PrAstWinsAstPrimary * PrDagLosesDagPrimary * strenv$AstProp +
            E_VoteShare_Ast_AmongAst_Given_AstLosesPrimary_DagWinsPrimary * PrAstLosesAstPrimary * PrDagWinsDagPrimary * strenv$AstProp +

          # Among Dag voters (weighted by DagProp)
            E_VoteShare_Ast_AmongDag_Given_AstWinsPrimary_DagWinsPrimary * PrAstWinsAstPrimary * PrDagWinsDagPrimary * strenv$DagProp +
            E_VoteShare_Ast_AmongDag_Given_AstLosesPrimary_DagLosesPrimary * PrAstLosesAstPrimary * PrDagLosesDagPrimary * strenv$DagProp +
            E_VoteShare_Ast_AmongDag_Given_AstWinsPrimary_DagLosesPrimary * PrAstWinsAstPrimary * PrDagLosesDagPrimary * strenv$DagProp +
            E_VoteShare_Ast_AmongDag_Given_AstLosesPrimary_DagWinsPrimary * PrAstLosesAstPrimary * PrDagWinsDagPrimary * strenv$DagProp
        )

        E_VoteShare_Dag <- (
          # Among Ast voters (weighted by AstProp)
          E_VoteShare_Dag_AmongAst_Given_AstWinsPrimary_DagWinsPrimary * PrAstWinsAstPrimary * PrDagWinsDagPrimary * strenv$AstProp +
            E_VoteShare_Dag_AmongAst_Given_AstLosesPrimary_DagLosesPrimary * PrAstLosesAstPrimary * PrDagLosesDagPrimary * strenv$AstProp +
            E_VoteShare_Dag_AmongAst_Given_AstWinsPrimary_DagLosesPrimary * PrAstWinsAstPrimary * PrDagLosesDagPrimary * strenv$AstProp +
            E_VoteShare_Dag_AmongAst_Given_AstLosesPrimary_DagWinsPrimary * PrAstLosesAstPrimary * PrDagWinsDagPrimary * strenv$AstProp +

          # Among Dag voters (weighted by DagProp)
            E_VoteShare_Dag_AmongDag_Given_AstWinsPrimary_DagWinsPrimary * PrAstWinsAstPrimary * PrDagWinsDagPrimary * strenv$DagProp +
            E_VoteShare_Dag_AmongDag_Given_AstLosesPrimary_DagLosesPrimary * PrAstLosesAstPrimary * PrDagLosesDagPrimary * strenv$DagProp +
            E_VoteShare_Dag_AmongDag_Given_AstWinsPrimary_DagLosesPrimary * PrAstWinsAstPrimary * PrDagLosesDagPrimary * strenv$DagProp +
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
