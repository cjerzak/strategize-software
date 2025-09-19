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
      #plot(strenv$np$array( TSAMP_dag$val$to_concrete_value() )-strenv$np$array( TSAMP_dag_PrimaryComp$val ) ,
           #strenv$np$array( PrimaryVoteShareDagAmongDag$val$to_concrete_value() ) )

      # win based on relaxed sample from bernoulli using oryx (depreciated)
      #Indicator_AstWinsPrimary <- strenv$oryx$distributions$RelaxedBernoulli(
              #temperature = MNtemp, probs = PrimaryVoteShareAstAmongAst )$sample(seed = SEED_IN_LOOP + 9992L)
      #Indicator_DagWinsPrimary <- strenv$oryx$distributions$RelaxedBernoulli(
              #temperature = MNtemp, probs = PrimaryVoteShareDagAmongDag )$sample(seed = SEED_IN_LOOP + 153L)
      
      # win based on relaxed sample from bernoulli using base JAX
      # (ast)
      eps <- 0.01 
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
