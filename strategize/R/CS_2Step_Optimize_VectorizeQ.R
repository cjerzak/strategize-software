InitializeQMonteFxns <- function(){
  Vectorized_QMonteIter_MaxMin <- compile_fxn( jax$vmap(
    (QMonteIter_MaxMin <- compile_fxn(function(TSAMP_ast, TSAMP_dag,
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
    GeneralVoteShareResults_AstReferenced <- getQStar_diff_MultiGroup(TSAMP_ast, #pi_star_ast
                                                  TSAMP_dag, # pi_star_dag
                                                  INTERCEPT_ast_, # EST_INTERCEPT_tf_ast
                                                  COEFFICIENTS_ast_, # EST_COEFFICIENTS_tf_ast
                                                  INTERCEPT_dag_, # EST_INTERCEPT_tf_dag
                                                  COEFFICIENTS_dag_) #EST_COEFFICIENTS_tf_dag
    GeneralVoteShareAstAmongAst <- jnp$take(GeneralVoteShareResults_AstReferenced,1L)
    GeneralVoteShareAstAmongDag <- jnp$take(GeneralVoteShareResults_AstReferenced,2L)
  
    # primary stage analysis
    {
      PrimaryVoteShareAstAmongAst <- jnp$take(getQStar_diff_SingleGroup(
        pi_star_ast =  TSAMP_ast,
        pi_star_dag = TSAMP_ast_PrimaryComp,
        EST_INTERCEPT_tf_ast = INTERCEPT_ast0_,
        EST_COEFFICIENTS_tf_ast = COEFFICIENTS_ast0_,
        EST_INTERCEPT_tf_dag = INTERCEPT_ast0_,
        EST_COEFFICIENTS_tf_dag = COEFFICIENTS_ast0_),0L)
      
      PrimaryVoteShareDagAmongDag <- jnp$take(getQStar_diff_SingleGroup(
        pi_star_ast = TSAMP_dag,
        pi_star_dag = TSAMP_dag_PrimaryComp,
        EST_INTERCEPT_tf_ast = INTERCEPT_dag0_,
        EST_COEFFICIENTS_tf_ast = COEFFICIENTS_dag0_,
        EST_INTERCEPT_tf_dag = INTERCEPT_dag0_,
        EST_COEFFICIENTS_tf_dag = COEFFICIENTS_dag0_),0L)
  
      lax_cond_indicator_AstWinsPrimary <- jax$lax$cond(
        pred = jnp$greater(PrimaryVoteShareAstAmongAst,jnp$array(0.5)),
        true_fun = function(){ jnp$array(1.) },
        false_fun = function(){ jnp$array(0.)} )
      lax_cond_indicator_DagWinsPrimary <- jax$lax$cond(
        pred = jnp$greater(PrimaryVoteShareDagAmongDag,jnp$array(0.5)),
        true_fun = function(){ jnp$array(1.) },
        false_fun = function(){ jnp$array(0.)} )
    }
  
    # combine all information together
    return(list( list(GeneralVoteShareAstAmongAst, lax_cond_indicator_AstWinsPrimary, PrimaryVoteShareAstAmongAst),
                 list(GeneralVoteShareAstAmongDag, lax_cond_indicator_DagWinsPrimary, PrimaryVoteShareDagAmongDag)) ) 
  })), 
  in_axes = eval(parse(text = paste("list(0L,0L,0L,0L,",
                      paste(rep("NULL,",times = 15-1), collapse=""), "NULL",  ")",sep = "")))))

  Vectorized_QMonteIter <- compile_fxn( jax$vmap( (QMonteIter <- compile_fxn( function(pi_star_ast_f, pi_star_dag_f,
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

  return( list("QMonteIter"=QMonteIter,
               "QMonteIter_MaxMin"=QMonteIter_MaxMin,
               "Vectorized_QMonteIter"=Vectorized_QMonteIter,
               "Vectorized_QMonteIter_MaxMin"=Vectorized_QMonteIter_MaxMin) )
  
}
