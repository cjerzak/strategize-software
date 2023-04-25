generate_GD_WithExactGradients <- function(){

  # NOTE: USE WITH diff == F

  messy_gd_update_text <- sprintf("main_coef = fe$main_coef,
                              inter_coef = fe$inter_coef, %s, term4_FC_a = fe$term4_FC_a",
                                  paste(paste(
                                    paste(fc2a <- paste("term2_FC_a",1:n_main_params,sep = ""),paste("fe$",fc2a,sep=""),sep="="),
                                    paste(fc2b <- paste("term2_FC_b",1:n_main_params,sep = ""),paste("fe$",fc2b,sep=""),sep="="),
                                    paste(fc4b <- paste("term4_FC_b",1:n_main_params,sep = ""),paste("fe$",fc4b,sep=""),sep="="),
                                    sep=","),collapse=","))
  doUpdate_conv  <-   eval(parse(text=sprintf("tf2jax$convert_functional(doUpdate_simp, a_vec = a_vec_init,
                                                    inv_learning_rate = inv_learning_rate,
                                                    %s)",messy_gd_update_text)))
  getPiStar_gd <- function(REGRESSION_PARAMETERS){
    REGRESSION_PARAMETERS <- gather_conv(REGRESSION_PARAMETERS)
    INTERCEPT_dag_ <- INTERCEPT_ <- REGRESSION_PARAMETERS[[1]]
    COEFFICIENTS_dag_ <- COEFFICIENTS_ <- REGRESSION_PARAMETERS[[2]]

    a_i <- jnp$array( a_vec_init )
    a_i_dag <- jnp$array( a_vec_init )
    inv_learning_rate_i <- jnp$array(1.) #jnp$array( inv_learning_rate )
    fe <- getFixedEntries_conv(  COEFFICIENTS_  )

    # gradient descent iterations
    grad_mag_vec <<- rep(NA,times=nSGD)
    goOn <- F; i<-0;maxIter<-nSGD;while(goOn == F){ i<-i+1;
    # CHECK GRADIENT AGAINST TRUTH
    if(T == F){
      getTrueGrad <- function(){
        a_var <- tf$Variable(a_i,trainable = T, dtype = tf$float32)
        p_vec_full <- tf$constant(as.matrix(unlist(p_list_full)),tf$float32)
        with(tf$GradientTape(persistent = T) %as% tape, {
          tape$watch(  a_var   )
          a_var_simplex <- a2Simplex(a_var)
          a_var_pretty_simplex <- getPrettyPi(a_var_simplex)
          obj_ <- getQStar(a_var_simplex) - lambda*tf$reduce_sum(tf$square(a_var_pretty_simplex-p_vec_full))
        })
        true_grad_i <- tape$gradient(obj_,a_var)
      }
      true_grad_i <- getTrueGrad()
      plot(as.numeric(true_grad_i),as.numeric(state_i[[3]]));abline(a=0,b=1)
      plot(as.numeric(true_grad_i)-as.numeric(state_i[[3]]))
      summary(lm(as.numeric(true_grad_i)[1:15]~as.numeric(state_i[[3]])[1:15]))
      plot(as.numeric(true_grad_i)[1:15],as.numeric(state_i[[3]])[1:15])
      plot(as.numeric(true_grad_i)[1:15]/as.numeric(state_i[[3]])[1:15])
      plot(as.numeric(true_grad_i)/as.numeric(state_i[[3]]))
      plot(as.numeric(true_grad_i)-as.numeric(state_i[[3]]))
    }

    # compute gradient
    state_i <- eval(parse( text = sprintf("doUpdate_conv(a_vec = a_i,
                                 inv_learning_rate = inv_learning_rate_i,%s)",messy_gd_update_text) ))

    # save for next iteration
    a_i <- state_i[[1]]; inv_learning_rate_i <- state_i[[2]]
    grad_mag_vec[i] <<- L2_grad <- list(state_i[[3]])

    if(i >= maxIter){goOn <- T}
    }

    if(nSGD == 1){return(sqrt(  sum((as.numeric(tf$constant(state_i[[3]],tf$float32) ))^2) ))}
    if(nSGD != 1){
      pi_star_ <- a2Simplex_conv( a_i )
      q_star_ <- getQStar_conv(pi_star = pi_star_, EST_INTERCEPT_tf = INTERCEPT_, EST_COEFFICIENTS_tf = COEFFICIENTS_)
      pi_star_full_simplex_ <- getPrettyPi_conv( pi_star_ )
      return( jnp$concatenate(list( q_star_, pi_star_full_simplex_ ) ) )
    }
  }

}
