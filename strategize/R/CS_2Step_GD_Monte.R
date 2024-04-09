QMonteIter_optimize <- function(TSAMP_ast, TSAMP_dag,
                       TSAMP_ast_PrimaryComp, TSAMP_dag_PrimaryComp,

                       #
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
  GeneralVoteShareResults_AstReferenced <- QFXN(TSAMP_ast, #pi_star_ast
                                                TSAMP_dag, # pi_star_dag
                                                INTERCEPT_ast_, # EST_INTERCEPT_tf_ast
                                                COEFFICIENTS_ast_, # EST_COEFFICIENTS_tf_ast
                                                INTERCEPT_dag_, # EST_INTERCEPT_tf_dag
                                                COEFFICIENTS_dag_ #EST_COEFFICIENTS_tf_dag
  )
  GeneralVoteShareAstAmongAst <- jnp$take(GeneralVoteShareResults_AstReferenced,1L)
  GeneralVoteShareAstAmongDag <- jnp$take(GeneralVoteShareResults_AstReferenced,2L)

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
    PrimaryVoteShareAstAmongAst <- jnp$take(PrimaryVoteShareAstAmongAst,0L)
    PrimaryVoteShareDagAmongDag <- jnp$take(PrimaryVoteShareDagAmongDag,0L)

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
  ProductAstGroup_AstOpti <- jnp$multiply(lax_cond_indicator_AstWinsPrimary,
                                  GeneralVoteShareAstAmongAst * PrimaryVoteShareAstAmongAst)
  ProductDagGroup_AstOpti <- jnp$multiply(lax_cond_indicator_DagWinsPrimary,
                                  GeneralVoteShareAstAmongDag * PrimaryVoteShareDagAmongDag)
  ProductAstGroup_DagOpti <- jnp$multiply(lax_cond_indicator_AstWinsPrimary,
                                          (1. - GeneralVoteShareAstAmongAst)*PrimaryVoteShareAstAmongAst)
  ProductDagGroup_DagOpti <- jnp$multiply(lax_cond_indicator_DagWinsPrimary,
                                          (1. - GeneralVoteShareAstAmongDag)*PrimaryVoteShareDagAmongDag)

  return(list( list(ProductAstGroup_AstOpti, lax_cond_indicator_AstWinsPrimary),
               list(ProductDagGroup_AstOpti, lax_cond_indicator_DagWinsPrimary),
               list(ProductAstGroup_DagOpti, lax_cond_indicator_AstWinsPrimary),
               list(ProductDagGroup_DagOpti, lax_cond_indicator_DagWinsPrimary) ) )
}

QMonteIter <- function(pi_star_ast_f, pi_star_dag_f,
                       INTERCEPT_ast_,
                       COEFFICIENTS_ast_,
                       INTERCEPT_dag_,
                       COEFFICIENTS_dag_){
  if( diff ){
    q_star_ <- QFXN(pi_star_ast_f, #pi_star_ast
                    pi_star_dag_f, #pi_star_dag
                    INTERCEPT_ast_,  #EST_INTERCEPT_tf_ast
                    COEFFICIENTS_ast_, #EST_COEFFICIENTS_tf_ast
                    INTERCEPT_dag_,  #EST_INTERCEPT_tf_dag
                    COEFFICIENTS_dag_) #EST_COEFFICIENTS_tf_dag
  }
  if( !diff  ){
    q_star_ <- QFXN(pi_star_ast_f, #pi_star
                    INTERCEPT_ast_, # EST_INTERCEPT_tf
                    COEFFICIENTS_ast_) # EST_COEFFICIENTS_tf =
  }
  return( q_star_ )
}
