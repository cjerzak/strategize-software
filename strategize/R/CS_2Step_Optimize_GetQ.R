getQStar_single <- function(pi_star_ast, pi_star_dag,
                            EST_INTERCEPT_tf_ast, EST_COEFFICIENTS_tf_ast,
                            EST_INTERCEPT_tf_dag, EST_COEFFICIENTS_tf_dag){
  # note: here, dag ignored 
  # coef info
  main_coef <- strenv$jnp$expand_dims( strenv$jnp$take(EST_COEFFICIENTS_tf_ast, 
                                                       indices = main_indices_i0, 
                                                       axis = 0L),1L)
  if(!is.null(inter_indices_i0)){ 
    inter_coef <- strenv$jnp$expand_dims( strenv$jnp$take(EST_COEFFICIENTS_tf_ast,
                                                          indices = inter_indices_i0, 
                                                          axis = 0L), 1L)
  
    # get interaction info
    pi_dp <- strenv$jnp$take(pi_star_ast, 
                             n2int( ai(interaction_info$dl_index_adj)-1L), axis=0L)
    pi_dpp <- strenv$jnp$take(pi_star_ast, 
                              n2int( ai(interaction_info$dplp_index_adj)-1L), axis=0L)
    Qhat <-  glm_outcome_transform( EST_INTERCEPT_tf_ast + 
                                      strenv$jnp$matmul( main_coef$transpose(), 
                                                         pi_star_ast) +
                                      strenv$jnp$matmul( inter_coef$transpose(), 
                                                         pi_dp*pi_dpp ) )
  }

  if(is.null(inter_indices_i0)){ 
    Qhat <-  glm_outcome_transform( EST_INTERCEPT_tf_ast + 
                          strenv$jnp$matmul( main_coef$transpose(),  pi_star_ast)  )
  }
  
  if( length(Qhat$shape) == 3L ) {
    Qhat <- Qhat$squeeze(2L)
  }
  return( strenv$jnp$concatenate( list(Qhat, 
                                       Qhat, 
                                       Qhat), 0L)  ) # to keep sizes consistent with diff case 
}

getQStar_diff_BASE <- function(pi_star_ast, pi_star_dag,
                               EST_INTERCEPT_tf_ast, EST_COEFFICIENTS_tf_ast,
                               EST_INTERCEPT_tf_dag, EST_COEFFICIENTS_tf_dag){

  # coef
  main_coef_ast <- strenv$jnp$expand_dims(strenv$jnp$take(EST_COEFFICIENTS_tf_ast, 
                                   indices = main_indices_i0, axis = 0L),1L)
  DELTA_pi_star <- (pi_star_ast - pi_star_dag)
  
  if(!is.null(inter_indices_i0)){ 
  inter_coef_ast <- strenv$jnp$expand_dims(strenv$jnp$take(EST_COEFFICIENTS_tf_ast, 
                                                           indices = inter_indices_i0, 
                                                           axis = 0L),1L)

  # get interaction info
  pi_ast_dp <- strenv$jnp$take(pi_star_ast, n2int( ai(interaction_info$dl_index_adj)-1L), axis=0L)
  pi_ast_dpp <- strenv$jnp$take(pi_star_ast, n2int( ai(interaction_info$dplp_index_adj)-1L), axis=0L)

  pi_dag_dp <- strenv$jnp$take(pi_star_dag, n2int( ai(interaction_info$dl_index_adj)-1L), axis=0L)
  pi_dag_dpp <- strenv$jnp$take(pi_star_dag, n2int( ai(interaction_info$dplp_index_adj)-1L), axis=0L)
  DELTA_pi_star_prod <- pi_ast_dp * pi_ast_dpp - pi_dag_dp * pi_dag_dpp
  
  Qhat_ast_among_ast <- glm_outcome_transform( 
            EST_INTERCEPT_tf_ast + 
            strenv$jnp$matmul(main_coef_ast$transpose(), DELTA_pi_star) + 
            strenv$jnp$matmul( inter_coef_ast$transpose(),  DELTA_pi_star_prod ) )
  }

  if(is.null(inter_indices_i0)){ 
    Qhat_ast_among_ast <- glm_outcome_transform( 
      EST_INTERCEPT_tf_ast +  strenv$jnp$matmul(main_coef_ast$transpose(), DELTA_pi_star)  )
  }
  
  if( !Q_DISAGGREGATE ){ Qhat_population <- Qhat_ast_among_dag <- Qhat_ast_among_ast }
  if( Q_DISAGGREGATE ){
    main_coef_dag <- strenv$jnp$expand_dims( strenv$jnp$take(EST_COEFFICIENTS_tf_dag, 
                                 indices = main_indices_i0, axis=0L), 1L)
    if(!is.null(inter_indices_i0)){ 
      inter_coef_dag <- strenv$jnp$expand_dims( strenv$jnp$take(EST_COEFFICIENTS_tf_dag, 
                                                                indices = inter_indices_i0, axis=0L), 1L)
      Qhat_ast_among_dag <- glm_outcome_transform( 
                EST_INTERCEPT_tf_dag + 
                strenv$jnp$matmul( main_coef_dag$transpose(), DELTA_pi_star ) +
                strenv$jnp$matmul( inter_coef_dag$transpose(), DELTA_pi_star_prod ) )
    }
    if(is.null(inter_indices_i0)){ 
      Qhat_ast_among_dag <- glm_outcome_transform( 
        EST_INTERCEPT_tf_dag +  strenv$jnp$matmul( main_coef_dag$transpose(), DELTA_pi_star ) )
    }
  
    # Pr( Ast | Ast Voter) * Pr(Ast Voters) +  Pr( Ast | Dag Voter) * Pr(Dag Voters)
    Qhat_population <- Qhat_ast_among_ast * strenv$jnp$array(strenv$AstProp) +  
                                Qhat_ast_among_dag * strenv$jnp$array(strenv$DagProp)
  }
  return( strenv$jnp$concatenate( list(Qhat_population, 
                                       Qhat_ast_among_ast, 
                                       Qhat_ast_among_dag), 0L)  )
}

FullGetQStar_VSept19_ <- function(a_i_ast,                                #1 
                          a_i_dag,                                #2 
                          INTERCEPT_ast_, COEFFICIENTS_ast_,      #3,4       
                          INTERCEPT_dag_, COEFFICIENTS_dag_,      #5,6 
                          INTERCEPT_ast0_, COEFFICIENTS_ast0_,    #7,8
                          INTERCEPT_dag0_, COEFFICIENTS_dag0_,    #9,10
                          P_VEC_FULL_ast_, P_VEC_FULL_dag_,       #11,12
                          SLATE_VEC_ast_, SLATE_VEC_dag_,         #13,14
                          LAMBDA_,                                #15
                          Q_SIGN,                                 #16 
                          SEED_IN_LOOP                            #17
                          ){
  message("Get pretty pi in FullGetQStar_...")
  pi_star_full_i_ast <- strenv$getPrettyPi_diff( pi_star_i_ast<-strenv$a2Simplex_diff_use(a_i_ast), 
                                                 strenv$ParameterizationType,
                                                 strenv$d_locator_use,       
                                                 strenv$main_comp_mat,   
                                                 strenv$shadow_comp_mat  )
  pi_star_full_i_dag <- strenv$getPrettyPi_diff( pi_star_i_dag<-strenv$a2Simplex_diff_use(a_i_dag),
                                                 strenv$ParameterizationType,
                                                 strenv$d_locator_use,       
                                                 strenv$main_comp_mat,   
                                                 strenv$shadow_comp_mat  )

  if(!adversarial){
    # NOTE: When diff == F, dag not used  
    q_max <- q__ <- strenv$jnp$take(QFXN(pi_star_ast =  pi_star_i_ast,
                  pi_star_dag = pi_star_i_dag,
                  EST_INTERCEPT_tf_ast = INTERCEPT_ast_,
                  EST_COEFFICIENTS_tf_ast = COEFFICIENTS_ast_,
                  EST_INTERCEPT_tf_dag = INTERCEPT_dag_,
                  EST_COEFFICIENTS_tf_dag = COEFFICIENTS_dag_),0L)
  }

  if(adversarial){
    message("Setup conditions...")
    # setup conditions 
    indicator_UseAst <- 0.5*(1 + Q_SIGN)
    
    #strenv$getMultinomialSamp_DEPRECIATED <-  getMultinomialSamp_R_DEPRECIATED
    #TSAMP_ast_all_DEPRECIATED <- strenv$jax$vmap(function(s_){ 
      #strenv$getMultinomialSamp_DEPRECIATED(pi_star_i_ast, MNtemp, s_,strenv$ParameterizationType, (strenv$np$array( strenv$d_locator_use) ) 
        #)},in_axes = 0L)(
        #strenv$jax$random$split(SEED_IN_LOOP, nMonte_adversarial) )
  
    # sample main competitor features
    TSAMP_ast_all <- strenv$jax$vmap(function(s_){ 
                    strenv$getMultinomialSamp(
                                     pi_star_i_ast, 
                                     MNtemp, 
                                     s_,
                                     strenv$ParameterizationType, 
                                     strenv$jnp$array(strenv$d_locator_use)
                                     )},in_axes = 0L)(
                          strenv$jax$random$split(SEED_IN_LOOP, nMonte_adversarial) )
    SEED_IN_LOOP   <- strenv$jax$random$split(SEED_IN_LOOP)[[1L]]
    TSAMP_dag_all <- strenv$jax$vmap(function(s_){ 
                    strenv$getMultinomialSamp(
                                     pi_star_i_dag, 
                                     MNtemp, 
                                     s_,
                                     strenv$ParameterizationType,
                                     strenv$jnp$array(strenv$d_locator_use)
                                     )},in_axes = list(0L))(
                          strenv$jax$random$split(SEED_IN_LOOP,
                                                  nMonte_adversarial) )
    SEED_IN_LOOP   <- strenv$jax$random$split(SEED_IN_LOOP)[[1L]]

    # sample primary competitor features uniformly or using slates 
    TSAMP_ast_PrimaryComp_all <- strenv$jax$vmap(function(s_){ 
                  strenv$getMultinomialSamp(SLATE_VEC_ast_, 
                                            MNtemp, 
                                            s_,
                                            strenv$ParameterizationType,
                                            strenv$jnp$array(strenv$d_locator_use)
                                            )},in_axes = list(0L))(
                            strenv$jax$random$split(SEED_IN_LOOP,
                                                    nMonte_adversarial) )
    SEED_IN_LOOP   <- strenv$jax$random$split(SEED_IN_LOOP)[[1L]]
    TSAMP_dag_PrimaryComp_all <- strenv$jax$vmap(function(s_){ 
                  strenv$getMultinomialSamp(SLATE_VEC_dag_, 
                                            MNtemp, 
                                            s_,
                                            strenv$ParameterizationType,
                                            strenv$jnp$array(strenv$d_locator_use)
                                            )},in_axes = list(0L))(
                          strenv$jax$random$split(SEED_IN_LOOP,
                                                  nMonte_adversarial) )
    SEED_IN_LOOP   <- strenv$jax$random$split(SEED_IN_LOOP)[[1L]]

    # compute electoral analysis 
    message("Run strenv$Vectorized_QMonteIter_MaxMin")
    QMonteRes <- strenv$Vectorized_QMonteIter_MaxMin(
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
                        strenv$jax$random$split(SEED_IN_LOOP,
                                                nMonte_adversarial)
                        )
    SEED_IN_LOOP   <- strenv$jax$random$split(SEED_IN_LOOP)[[1L]]
    
    # primaries 
    PrAstWinsAstPrimary <- QMonteRes$AmongAst$PrimaryVoteShareAstAmongAst$mean()
    PrAstLosesAstPrimary <- 1-PrAstWinsAstPrimary
    PrDagWinsDagPrimary <- QMonteRes$AmongDag$PrimaryVoteShareDagAmongDag$mean()
    PrDagLosesDagPrimary <- 1-PrDagWinsDagPrimary
    
    # generals 
    # Compute  average vote shares in the subset of simulations where candidate wins its primary.
    # Specifically, sum of votes in that subset divided by (sum of total votes in that subset + epsilon)
    
    # among ast 
    Indicator_AstWinsPrimary <- QMonteRes$AmongAst$Indicator_AstWinsPrimary # soft or hard 
    GVShareAstAmongAst_Given_AstWinsAstPrimary <- 
                  ( QMonteRes$AmongAst$GVShareAstAmongAst*Indicator_AstWinsPrimary )$sum()/ 
                              ( (ep_<-0.01) + Indicator_AstWinsPrimary$sum() )
    GVShareDagAmongAst_Given_AstWinsAstPrimary <-  1 - GVShareAstAmongAst_Given_AstWinsAstPrimary
    
    # among dag 
    Indicator_DagWinsPrimary <- QMonteRes$AmongDag$Indicator_DagWinsPrimary
    GVShareAstAmongDag_Given_DagWinsDagPrimary <- 
                    (  QMonteRes$AmongDag$GVShareAstAmongDag*Indicator_DagWinsPrimary )$sum() / 
                              ( ep_ + Indicator_DagWinsPrimary$sum() )
    GVShareDagAmongDag_Given_DagWinsDagPrimary <- 1 - GVShareAstAmongDag_Given_DagWinsDagPrimary

    if(T == T){
      # compute expected value 
      QMonteRes$AmongAst$GVShareDagAmongAst <- 1-QMonteRes$AmongAst$GVShareAstAmongAst
      QMonteRes$AmongDag$GVShareDagAmongDag <- 1-QMonteRes$AmongDag$GVShareAstAmongDag
      
      Indicator_DagLosesPrimary <- 1-Indicator_DagWinsPrimary
      Indicator_AstLosesPrimary <- 1-Indicator_AstWinsPrimary
      
      # 2 by 2 AMONG AST 
      {
      # 1 ast 
      E_VoteShare_Ast_AmongAst_Given_AstWinsPrimary_DagWinsPrimary <-
                ((QMonteRes$AmongAst$GVShareAstAmongAst)*
                  (event_ <- (Indicator_AstWinsPrimary*Indicator_DagWinsPrimary) ))$sum() / 
                (0.001+(event_))$sum()
      
      # 2 ast 
      E_VoteShare_Ast_AmongAst_Given_AstLosesPrimary_DagLosesPrimary <-
        ((QMonteRes$AmongAst$GVShareAstAmongAst)*
           ( event_ <- (Indicator_AstLosesPrimary*Indicator_DagLosesPrimary) ))$sum() / 
        (0.001+(event_))$sum()
      
      # 3 ast
      E_VoteShare_Ast_AmongAst_Given_AstLosesPrimary_DagWinsPrimary <-
        ((QMonteRes$AmongAst$GVShareAstAmongAst)*
           (event_ <- (Indicator_AstLosesPrimary*Indicator_DagWinsPrimary) ))$sum() / 
        (0.001+(event_))$sum()
      
      # 4 ast
      E_VoteShare_Ast_AmongAst_Given_AstWinsPrimary_DagLosesPrimary <-
        ((QMonteRes$AmongAst$GVShareAstAmongAst)*
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
          ((QMonteRes$AmongDag$GVShareAstAmongDag)*
             (event_ <- (Indicator_AstWinsPrimary*Indicator_DagWinsPrimary) ))$sum() / 
          (0.001+(event_))$sum()
        
        # 2
        E_VoteShare_Ast_AmongDag_Given_AstLosesPrimary_DagLosesPrimary <-
          ((QMonteRes$AmongDag$GVShareAstAmongDag)*
             ( event_ <- (Indicator_AstLosesPrimary*Indicator_DagLosesPrimary) ))$sum() / 
          (0.001+(event_))$sum()
        
        # 3 - 
        E_VoteShare_Ast_AmongDag_Given_AstLosesPrimary_DagWinsPrimary <-
          ((QMonteRes$AmongDag$GVShareAstAmongDag)*
             (event_ <- (Indicator_AstLosesPrimary*Indicator_DagWinsPrimary) ))$sum() / 
          (0.001+(event_))$sum()
        
        # 4
        E_VoteShare_Ast_AmongDag_Given_AstWinsPrimary_DagLosesPrimary <-
          ((QMonteRes$AmongDag$GVShareAstAmongDag)*
             (event_ <- (Indicator_AstWinsPrimary*Indicator_DagLosesPrimary) ))$sum() / 
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
      q_max_ast <- E_VoteShare_Ast
      q_max_dag <- E_VoteShare_Dag
    }
    #TRUE_ <- compute_vote_share_R(pi_R = mean(strenv$np$array( TSAMP_dag_all$to_concrete_value() )[,1,1]), pi_D = mean(strenv$np$array( TSAMP_ast_all )[,1,1]))
    # q_max_ast+q_max_dag

    # quantity to maximize for ast and dag respectively 
    q_max <-  indicator_UseAst * q_max_ast  + (1-indicator_UseAst) * q_max_dag 
  }

  # regularization
  {
    message("Apply regularization")
    if(penalty_type %in% c("L1","L2")){
      PenFxn <- ifelse(penalty_type == "L1", 
                       yes = list(strenv$jnp$abs),
                       no = list(strenv$jnp$square))[[1]]
      var_pen_ast__ <- LAMBDA_ * strenv$jnp$negative(strenv$jnp$sum(PenFxn( pi_star_full_i_ast - P_VEC_FULL_ast_ )  ))
      var_pen_dag__ <- LAMBDA_ * strenv$jnp$negative(strenv$jnp$sum(PenFxn( pi_star_full_i_dag - P_VEC_FULL_dag_ )  ))
    }
    if(penalty_type == "LInfinity"){
      var_pen_ast__ <- tapply(1:length(split_vec_full), split_vec_full, function(zer){
        list( strenv$jnp$max(strenv$jnp$take(pi_star_full_i_ast, 
                                             indices = strenv$jnp$array( ai(zer-1L)),axis = 0L)))})
      names(var_pen_ast__)<-NULL ; var_pen_ast__ <- strenv$jnp$negative(LAMBDA_*
                                                          strenv$jnp$sum( strenv$jnp$stack(var_pen_ast__)))

      var_pen_dag__ <- tapply(1:length(split_vec_full), split_vec_full, function(zer){
        list( strenv$jnp$max(strenv$jnp$take(pi_star_full_i_dag, 
                                             indices = strenv$jnp$array( ai(zer-1L)),axis = 0L)))})
      names(var_pen_dag__)<-NULL ; var_pen_dag__ <- strenv$jnp$negative(LAMBDA_*
                                                    strenv$jnp$sum( strenv$jnp$stack(var_pen_dag__)))
    }
    if(penalty_type == "KL"){
      var_pen_ast__ <- LAMBDA_*strenv$jnp$negative(strenv$jnp$sum(P_VEC_FULL_ast_ * (strenv$jnp$log(P_VEC_FULL_ast_) - strenv$jnp$log(pi_star_full_i_ast))))
      var_pen_dag__ <- LAMBDA_*strenv$jnp$negative(strenv$jnp$sum(P_VEC_FULL_dag_ * (strenv$jnp$log(P_VEC_FULL_dag_) - strenv$jnp$log(pi_star_full_i_dag))))
    }
  }
  
  message("Combine results + return")
  if( adversarial ){ myMaximize <- q_max + indicator_UseAst * var_pen_ast__  + 
                                       (1-indicator_UseAst) * var_pen_dag__  } 
  if( !adversarial ){ myMaximize <- q_max + var_pen_ast__ }
  return( myMaximize )
}


FullGetQStar_ <- function(a_i_ast,                                #1 
                          a_i_dag,                                #2 
                          INTERCEPT_ast_, COEFFICIENTS_ast_,      #3,4       
                          INTERCEPT_dag_, COEFFICIENTS_dag_,      #5,6 
                          INTERCEPT_ast0_, COEFFICIENTS_ast0_,    #7,8
                          INTERCEPT_dag0_, COEFFICIENTS_dag0_,    #9,10
                          P_VEC_FULL_ast_, P_VEC_FULL_dag_,       #11,12
                          SLATE_VEC_ast_, SLATE_VEC_dag_,         #13,14
                          LAMBDA_,                                #15
                          Q_SIGN,                                 #16 
                          SEED_IN_LOOP                            #17
){
  
  # Map logits -> simplex (respecting ParameterizationType)
  pi_star_full_i_ast <- strenv$getPrettyPi_diff( pi_star_i_ast<-strenv$a2Simplex_diff_use(a_i_ast), 
                                                 strenv$ParameterizationType,
                                                 strenv$d_locator_use,       
                                                 strenv$main_comp_mat,   
                                                 strenv$shadow_comp_mat  )
  pi_star_full_i_dag <- strenv$getPrettyPi_diff( pi_star_i_dag<-strenv$a2Simplex_diff_use(a_i_dag),
                                                 strenv$ParameterizationType,
                                                 strenv$d_locator_use,       
                                                 strenv$main_comp_mat,   
                                                 strenv$shadow_comp_mat  )
  
  # Average-case path (unchanged)
  if(!adversarial){
    q_vec <- QFXN(pi_star_ast =  pi_star_i_ast,
                  pi_star_dag =  pi_star_i_dag,
                  EST_INTERCEPT_tf_ast = INTERCEPT_ast_,
                  EST_COEFFICIENTS_tf_ast = COEFFICIENTS_ast_,
                  EST_INTERCEPT_tf_dag = INTERCEPT_dag_,
                  EST_COEFFICIENTS_tf_dag = COEFFICIENTS_dag_)
    q_max <- strenv$jnp$take(q_vec, 0L)
  }
  
  # Adversarial path: institution-aware push-forward (four-quadrant mixture)
  if(adversarial){
    
    # Draw policy samples
    TSAMP_ast_all <- strenv$jax$vmap(function(s_){ 
      strenv$getMultinomialSamp(pi_star_i_ast, MNtemp, s_, strenv$ParameterizationType, strenv$jnp$array(strenv$d_locator_use))
    }, in_axes = list(0L))(strenv$jax$random$split(SEED_IN_LOOP, nMonte_adversarial))
    SEED_IN_LOOP   <- strenv$jax$random$split(SEED_IN_LOOP)[[1L]]
    
    TSAMP_dag_all <- strenv$jax$vmap(function(s_){ 
      strenv$getMultinomialSamp(pi_star_i_dag, MNtemp, s_, strenv$ParameterizationType, strenv$jnp$array(strenv$d_locator_use))
    }, in_axes = list(0L))(strenv$jax$random$split(SEED_IN_LOOP, nMonte_adversarial))
    SEED_IN_LOOP   <- strenv$jax$random$split(SEED_IN_LOOP)[[1L]]
    
    # Draw field (“slate”) samples
    TSAMP_ast_PrimaryComp_all <- strenv$jax$vmap(function(s_){ 
      strenv$getMultinomialSamp(SLATE_VEC_ast_, MNtemp, s_, strenv$ParameterizationType, strenv$jnp$array(strenv$d_locator_use))
    }, in_axes = list(0L))(strenv$jax$random$split(SEED_IN_LOOP, nMonte_adversarial))
    SEED_IN_LOOP   <- strenv$jax$random$split(SEED_IN_LOOP)[[1L]]
    
    TSAMP_dag_PrimaryComp_all <- strenv$jax$vmap(function(s_){ 
      strenv$getMultinomialSamp(SLATE_VEC_dag_, MNtemp, s_, strenv$ParameterizationType, strenv$jnp$array(strenv$d_locator_use))
    }, in_axes = list(0L))(strenv$jax$random$split(SEED_IN_LOOP, nMonte_adversarial))
    SEED_IN_LOOP   <- strenv$jax$random$split(SEED_IN_LOOP)[[1L]]
    
    # Evaluate institutional objective (push-forward over nominees)
    QMonteRes <- strenv$Vectorized_QMonteIter_MaxMin(
      TSAMP_ast_all, TSAMP_dag_all,
      TSAMP_ast_PrimaryComp_all, TSAMP_dag_PrimaryComp_all,
      a_i_ast, a_i_dag,
      INTERCEPT_ast_,  COEFFICIENTS_ast_,
      INTERCEPT_dag_,  COEFFICIENTS_dag_,
      INTERCEPT_ast0_, COEFFICIENTS_ast0_,
      INTERCEPT_dag0_, COEFFICIENTS_dag0_,
      P_VEC_FULL_ast_, P_VEC_FULL_dag_,
      LAMBDA_, Q_SIGN, SEED_IN_LOOP
    )
    
    q_max_ast <- QMonteRes$q_ast
    q_max_dag <- QMonteRes$q_dag
    
    # Choose which side we’re optimizing in this call
    indicator_UseAst <- strenv$jnp$multiply(0.5, strenv$jnp$add(strenv$jnp$array(1.), Q_SIGN))
    q_max <- strenv$jnp$add(
      strenv$jnp$multiply(indicator_UseAst,                    q_max_ast),
      strenv$jnp$multiply(strenv$jnp$subtract(1., indicator_UseAst), q_max_dag)
    )
  }
  
  # ---- Regularization (unchanged), applied to the player being updated ----
  if(penalty_type %in% c("L1","L2")){
    PenFxn <- ifelse(penalty_type == "L1", 
                     yes = list(strenv$jnp$abs),
                     no = list(strenv$jnp$square))[[1]]
    var_pen_ast__ <- LAMBDA_ * strenv$jnp$negative(strenv$jnp$sum(PenFxn( pi_star_full_i_ast - P_VEC_FULL_ast_ )  ))
    var_pen_dag__ <- LAMBDA_ * strenv$jnp$negative(strenv$jnp$sum(PenFxn( pi_star_full_i_dag - P_VEC_FULL_dag_ )  ))
  } else if(penalty_type == "LInfinity"){
    var_pen_ast__ <- tapply(1:length(split_vec_full), split_vec_full, function(zer){
      list( strenv$jnp$max(strenv$jnp$take(pi_star_full_i_ast, indices = strenv$jnp$array( ai(zer-1L)),axis = 0L)) )})
    names(var_pen_ast__)<-NULL ; var_pen_ast__ <- strenv$jnp$negative(LAMBDA_*strenv$jnp$sum(strenv$jnp$stack(var_pen_ast__)))
    var_pen_dag__ <- tapply(1:length(split_vec_full), split_vec_full, function(zer){
      list( strenv$jnp$max(strenv$jnp$take(pi_star_full_i_dag, indices = strenv$jnp$array( ai(zer-1L)),axis = 0L)) )})
    names(var_pen_dag__)<-NULL ; var_pen_dag__ <- strenv$jnp$negative(LAMBDA_*strenv$jnp$sum(strenv$jnp$stack(var_pen_dag__)))
  } else { # "KL" default
    var_pen_ast__ <- LAMBDA_*strenv$jnp$negative(strenv$jnp$sum(P_VEC_FULL_ast_ * (strenv$jnp$log(P_VEC_FULL_ast_) - strenv$jnp$log(pi_star_full_i_ast))))
    var_pen_dag__ <- LAMBDA_*strenv$jnp$negative(strenv$jnp$sum(P_VEC_FULL_dag_ * (strenv$jnp$log(P_VEC_FULL_dag_) - strenv$jnp$log(pi_star_full_i_dag))))
  }
  
  indicator_UseAst <- strenv$jnp$multiply(0.5, strenv$jnp$add(strenv$jnp$array(1.), Q_SIGN))
  myMaximize <- strenv$jnp$add(
    q_max,
    strenv$jnp$add(
      strenv$jnp$multiply(indicator_UseAst,                          var_pen_ast__),
      strenv$jnp$multiply(strenv$jnp$subtract(strenv$jnp$array(1.), indicator_UseAst), var_pen_dag__)
    ))
  
  return( myMaximize )
}

