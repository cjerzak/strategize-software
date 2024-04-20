generate_GD_WithExactGradients <- function(){
  # NOTE: USE WITH diff == F
  messy_gd_update_text <- sprintf("main_coef = fe$main_coef,
                              inter_coef = fe$inter_coef, %s, term4_FC_a = fe$term4_FC_a",
                                  paste(paste(
                                    paste(fc2a <- paste("term2_FC_a",1:n_main_params,sep = ""),paste("fe$",fc2a,sep=""),sep="="),
                                    paste(fc2b <- paste("term2_FC_b",1:n_main_params,sep = ""),paste("fe$",fc2b,sep=""),sep="="),
                                    paste(fc4b <- paste("term4_FC_b",1:n_main_params,sep = ""),paste("fe$",fc4b,sep=""),sep="="),
                                    sep=","),collapse=","))
  getPiStar_gd <- function(REGRESSION_PARAMETERS_ast,
                           
                           # parameters for forward compatability 
                           REGRESSION_PARAMETERS_dag,
                           REGRESSION_PARAMETERS_ast0,
                           REGRESSION_PARAMETERS_dag0,
                           P_VEC_FULL_ast,
                           P_VEC_FULL_dag,
                           SLATE_VEC_ast, 
                           SLATE_VEC_dag,
                           LAMBDA,
                           BASE_SEED,
                           functionList,
                           functionReturn = T, quiet = T){
    REGRESSION_PARAMETERS_ast <- gather_fxn(REGRESSION_PARAMETERS_ast)
    INTERCEPT_ast_ <- INTERCEPT_ <- REGRESSION_PARAMETERS_ast[[1]]
    COEFFICIENTS_ast_ <- COEFFICIENTS_ <- REGRESSION_PARAMETERS_ast[[2]]

    a_i <- jnp$array( a_vec_init_ast )
    inv_learning_rate_i <- jnp$array(1.) # initial 
    fe <- getFixedEntries (  COEFFICIENTS_  ) # previously used getFixedEntries_conv

    # gradient descent iterations
    grad_mag_vec <<- rep(NA,times=nSGD)
    goOn <- F; i<-0;maxIter<-nSGD;while(goOn == F){ 
    i<-i+1;

    # compute gradient
    browser()
    state_i <- eval(parse( text = sprintf("doUpdate_simp(a_vec = a_i,
                                 inv_learning_rate = inv_learning_rate_i,%s)",messy_gd_update_text) ))

    # save for next iteration
    a_i <- state_i[[1]]; inv_learning_rate_i <- state_i[[2]]
    grad_mag_vec[i] <<- L2_grad <- list(state_i[[3]])

    if(i >= maxIter){goOn <- T}
    }

    if(nSGD == 1){return(sqrt(  sum(np$array(jnp$array(state_i[[3]], jnp$float32) )^2) ))}
    if(nSGD != 1){
      pi_star_ <- a2Simplex_( a_i )
      q_star_ <- getQStar(pi_star = pi_star_, EST_INTERCEPT_tf = INTERCEPT_, EST_COEFFICIENTS_tf = COEFFICIENTS_)
      pi_star_full_simplex_ <- getPrettyPi( pi_star_ )
      return( jnp$concatenate(list( q_star_, pi_star_full_simplex_ ) ) )
    }
  }

}
