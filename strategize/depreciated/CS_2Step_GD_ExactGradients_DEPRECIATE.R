generate_ManualDoUpdates <- function(){
   #########################
   ## DEPRECIATE 
   #########################
    a2Grad <- jax$jit(function( a_ ){
      exp_a_ <- jnp$exp(  a_  )
      aGrad <- tapply(1:nrow(main_info),main_info$d,function(zer){
        exp_gathered <- jnp$take(exp_a_, n2int(zer-1L))
        sum_exp_a <- jnp$add( OneTf_flat, jnp$sum(exp_gathered) )
        tmp <- jnp$multiply(jnp$divide( exp_gathered,  jnp$square(sum_exp_a) ),
                           sum_exp_a - exp_gathered)
        if(length(unlist(tmp$shape)) == 0){ tmp <- jnp$expand_dims( tmp, 0L ) }
        return( list( tmp ) ) })
      names(aGrad) <- NULL
      return( jnp$concatenate(aGrad,0L) )
    })
    a2Term4 <- jax$jit(function(a_){
      exp_a_ <- jnp$exp( a_ )
      comp_ <- tapply(1:nrow(main_info),main_info$d,function(zer){
        exp_a_gathered <- jnp$take(exp_a_, n2int(zer-1L))
        sum_exp_a <- jnp$add(OneTf_flat, jnp$sum( exp_a_gathered ))
        tmp <- jnp$divide( exp_a_gathered, jnp$square( sum_exp_a ) )
        if(length(unlist(tmp$shape)) == 0){ tmp <- jnp$expand_dims( tmp, 0L ) }
        return( list( tmp ) ) })
      names(comp_) <- NULL
      return( jnp$concatenate(comp_,0L) )
    })
    doUpdate <- jax$jit(doUpdate_r <- function( a_vec, inv_learning_rate,
                                                main_coef,inter_coef,
                                                term2_FC_a, term2_FC_b,
                                                term4_FC_a, term4_FC_b
    ){
      dydx_a_simplex <- a2Grad(  a_vec )
      a_simplex <- a2Simplex( a_vec )
      exp_a <- jnp$exp( a_vec )

      # TERM 1 DERIVATIVE CONTRIBUTION
      term1_self <- jnp$multiply(dydx_a_simplex, main_coef) #done
      term1_other <- tapply(1:nrow(main_info),main_info$d,function(zer){
        exp_a_gathered <- jnp$take(exp_a, n2int(ai(zer-1L)))
        main_coef_gathered <- jnp$take(main_coef, n2int(ai(zer-1L)))
        sum_a_squared <- jnp$square(  jnp$add(OneTf_flat, jnp$sum(exp_a_gathered) ))
        tmp <- jnp$concatenate(sapply(1:length(zer), function(drop_indi){
          index_other_drop_self <- n2int((1:length(zer)-1L)[-drop_indi])
          beta_other <- jnp$take(main_coef_gathered, index_other_drop_self,axis=0L)
          exp_other <- jnp$take(exp_a_gathered, index_other_drop_self,axis=0L)
          exp_self <- jnp$take(exp_a_gathered,n2int(ai(drop_indi-1L)),axis=0L)
          return( jnp$sum(
            jnp$multiply(beta_other,jnp$negative(jnp$divide(jnp$multiply(exp_other,exp_self),sum_a_squared))),keepdims=T) )
        }), 0L)
        if(length(unlist(tmp$shape)) == 0){ tmp <- jnp$expand_dims( tmp, 1L ) }
        return( list( tmp ) ) })
      names(term1_other) <- NULL
      term1_other <- jnp$concatenate(term1_other, 0L)
      term1 <- jnp$add(term1_self, term1_other)

      # TERM 2 DERIVATIVE CONTRIBUTION
      term2_FC_self <- jnp$concatenate(sapply(1:length(a_vec),function(xer){
        gathered_coef_values <- jnp$expand_dims(jnp$take(term2_FC_a,n2int(ai(xer - 1L)),axis=0L)$flat_values,1L)
        gathered_a_simplex_indices <- jnp$take(term2_FC_b,n2int(ai(xer-1L)),axis=0L)
        inter_prob_ <-  jnp$take(a_simplex, indices = gathered_a_simplex_indices, axis = 0L)
        return(  jnp$expand_dims(jnp$expand_dims(
          jnp$sum(jnp$multiply(gathered_coef_values, inter_prob_)),0L), 1L))
      }),0L)
      term2_self <- jnp$multiply(dydx_a_simplex, term2_FC_self) # done
      term2_other <- tapply(1:nrow(main_info),main_info$d,function(zer){
        exp_a_gathered <- jnp$take(exp_a, n2int(ai(zer-1L)))
        sum_a <- jnp$add(  OneTf_flat, jnp$sum(exp_a_gathered)  )
        sum_a_squared <- jnp$square(  sum_a )
        gamma_other_gathered <- jnp$take(term2_FC_a, n2int(ai(zer-1L)),axis=0L)

        tmp <- jnp$concatenate(sapply(1:length(zer), function(drop_indi){
          gamma_other_rag <- jnp$take(gamma_other_gathered, n2int(ai(1:length(zer)-1L)[-drop_indi]),axis=0L)
          gathered_a_simplex_indices_others <- jnp$take(term2_FC_b, n2int(ai(zer-1L)[-drop_indi]),axis=0L)
          if(isFlat <-"flat_values" %in% names(gamma_other_rag)){ gamma_other <- gamma_other_rag$flat_values }
          if(!isFlat){ gamma_other <- gamma_other_rag }
          if(length(zer) <= 2){ segment_ids <- jnp$zeros(jnp$shape(gamma_other),dtype=jnp$int32) }
          if(length(zer) > 2){
            row_splits_to_segment_ids <- function(row_splits) {
              segment_ids <- integer()

              for(i in 1:(length(row_splits) - 1)) {
                segment_ids <- c(segment_ids, rep(i - 1, times = row_splits[i + 1] - row_splits[i]))
              }

              return(segment_ids) }

            segment_ids <- jnp$array(row_splits_to_segment_ids(gamma_other_rag$row_splits),jnp$int32 ) # tf$ragged$row_splits_to_segment_ids
           }
          if("flat_values" %in% names(gathered_a_simplex_indices_others)){
            gathered_a_simplex_indices_others <- gathered_a_simplex_indices_others$flat_values
          }
          inter_prob_others_of_others <-  jnp$take(a_simplex, indices = gathered_a_simplex_indices_others, axis = 0L)
          if("flat_values" %in% names(inter_prob_others_of_others)){
            inter_prob_others_of_others <- inter_prob_others_of_others$flat_values
          }
          PROD_interaction_others_of_others_prob <- jnp$multiply(jnp$expand_dims(gamma_other,1L),inter_prob_others_of_others)

          exp_other <- jnp$take(exp_a_gathered,n2int((1:length(zer)-1L)[-drop_indi]))
          exp_self <- jnp$take(exp_a_gathered,n2int(ai(drop_indi-1L)))
          prod_other_self <- jnp$negative(jnp$divide(jnp$multiply(exp_other,exp_self),sum_a_squared))
          prod_other_self <- jnp$take(prod_other_self,segment_ids,axis=0L)
          if(length(unlist(prod_other_self$shape))==1){prod_other_self <- jnp$expand_dims(prod_other_self,1L)}
          return( jnp$sum(jnp$multiply(prod_other_self,PROD_interaction_others_of_others_prob),keepdims=T) )
        }), 0L)
        if(length(unlist(tmp$shape)) == 0){ tmp <- jnp$expand_dims( tmp, 1L ) }
        return( list( tmp ) ) })
      names(term2_other) <- NULL
      term2_other <- jnp$concatenate(term2_other,0L)
      term2 <- jnp$add(term2_self, term2_other)

      # TERM 3 DERIVATIVE CONTRIBUTION
      term3_self <- jnp$multiply(Neg2_tf,jnp$multiply(jnp$multiply(lambda,dydx_a_simplex),
                                                    jnp$subtract(a_simplex, p_vec_tf) ))#done
      term3_other <- tapply(1:nrow(main_info),main_info$d,function(zer){
        p_gathered <- jnp$take(p_vec_tf, n2int(ai(zer-1L)))
        exp_a_gathered <- jnp$take(exp_a, n2int(ai(zer-1L)))
        sum_a_squared <- jnp$square( sum_a <- jnp$add(OneTf_flat,jnp$sum(exp_a_gathered)) ) 
        tmp <- jnp$concatenate(sapply(1:length(zer), function(drop_indi){
          p_other <- jnp$take(p_gathered, n2int((1:length(zer)-1L)[-drop_indi]))
          exp_other <- jnp$take(exp_a_gathered, n2int((1:length(zer)-1L)[-drop_indi]))
          exp_self <- jnp$take(exp_a_gathered,n2int(ai(drop_indi-1L)))
          jnp$sum(jnp$negative(jnp$divide(exp_other*exp_self,sum_a_squared))*(exp_other/sum_a - p_other),
                        keepdims=T) }), 0L)
        if(length(unlist(tmp$shape)) == 0){ tmp <- jnp$expand_dims( tmp, 1L ) }
        return( list( tmp ) ) })
      names(term3_other) <- NULL
      term3_other <- jnp$concatenate(term3_other,0L)
      term3_other <- jnp$multiply(Neg2_tf,jnp$multiply(lambda, term3_other))
      term3 <- jnp$add(term3_self, term3_other)

      # TERM 4 DERIVATIVE CONTRIBUTION
      term4_FC_ <- jnp$expand_dims(jnp$concatenate(sapply(1:length(a_vec),function(xer){
        gathered_sum_p <- jnp$expand_dims(jnp$take(term4_FC_a,indices = n2int(ai(xer-1L)),axis=0L),0L)
        gathered_a_simplex_indices <- jnp$take(term4_FC_b,indices = n2int(ai(xer-1L)),axis=0L)$flat_values
        sum_pi <- jnp$expand_dims(jnp$sum(jnp$take(a_simplex,indices = gathered_a_simplex_indices,axis=0L)),0L)
        return( jnp$subtract(sum_pi,  gathered_sum_p) )
      }),0L),1L)
      term4 <- jnp$multiply(jnp$multiply(Neg2_tf,lambda),jnp$multiply(a2Term4(a_vec),term4_FC_)) # done

      # COMBINE ALL GRADIENT TERMS
      grad_i <- jnp$add(jnp$add(term1, term2), jnp$add(term3, term4))

      # update
      inv_learning_rate_i <-  GetInvLR(inv_learning_rate, grad_i)

      # update parameters
      a_vec_updated <- GetUpdatedParameters(a_vec = a_vec,
                                            grad_i = grad_i,
                                            inv_learning_rate_i = jnp$sqrt( inv_learning_rate_i) )

      return( list(a_vec_updated, inv_learning_rate_i,grad_i) )
    })

    # what is the difference between doUpdate and doUpdate_simp
  eval(parse(text = sprintf("doUpdate_simp <- jax$jit(doUpdate_simp_ <- function( a_vec, inv_learning_rate,
                                                     main_coef,inter_coef,
                                                     %s,term4_FC_a
                                                     ){
    dydx_a_simplex <- a2Grad(  a_vec )
    # a_simplex <- a2Simplex( a_vec )# old 
    a_simplex <- a2Simplex_optim( a_vec ) # new 
    exp_a <- jnp$exp( a_vec )

    print2('TERM 1 DERIVATIVE CONTRIBUTION')
    term1_self <- jnp$multiply(dydx_a_simplex, main_coef) #done
    term1_other <- tapply(1:nrow(main_info),main_info$d,function(zer){
        exp_a_gathered <- jnp$take(exp_a, n2int(ai(zer-1L)))
        main_coef_gathered <- jnp$take(main_coef, n2int(ai(zer-1L)))
        sum_a_squared <- jnp$square(  jnp$add(OneTf_flat, jnp$sum(exp_a_gathered) ))
        tmp <- jnp$stack(sapply(1:length(zer), function(drop_indi){
          index_other_drop_self <- n2int((1:length(zer)-1L)[-drop_indi])
          beta_other <- jnp$take(main_coef_gathered, index_other_drop_self,axis=0L)
          exp_other <- jnp$take(exp_a_gathered, index_other_drop_self,axis=0L)
          exp_self <- jnp$take(exp_a_gathered,n2int(ai(drop_indi-1L)),axis=0L)
          return( list(jnp$sum(
            jnp$multiply(beta_other,jnp$negative(jnp$divide(jnp$multiply(exp_other,exp_self),sum_a_squared))),keepdims=T) ))
        }), 0L)
        if(length(unlist(tmp$shape)) == 1){ tmp <- jnp$expand_dims( tmp, 1L ) }
        return( list( tmp ) ) })
    names(term1_other) <- NULL
    term1_other <- jnp$concatenate(term1_other, 0L)
    term1 <- jnp$add(jnp$expand_dims(term1_self,1L), term1_other)

    print2('TERM 2 DERIVATIVE CONTRIBUTION')
    term2_FC_self <- jnp$concatenate(sapply(1:n_main_params,function(xer){
        gathered_coef_values <- eval(parse(text=sprintf('term2_FC_a%%s',xer)))
        gathered_a_simplex_indices <- eval(parse(text=sprintf('term2_FC_b%%s',xer)))
        inter_prob_ <-  jnp$take(a_simplex, indices = gathered_a_simplex_indices, axis = 0L)
        return(  jnp$expand_dims(jnp$expand_dims(
          jnp$sum(jnp$multiply(gathered_coef_values, inter_prob_)),0L), 1L) )
      }),0L)
    term2_self <- jnp$multiply(jnp$expand_dims(dydx_a_simplex,1L), term2_FC_self) # done

    term2_other <- tapply(1:nrow(main_info),main_info$d,function(zer){
      exp_a_gathered <- jnp$take(exp_a, n2int(ai(zer-1L)))
      sum_a <- jnp$add(  OneTf_flat, jnp$sum(exp_a_gathered)  )
      sum_a_squared <- jnp$square(  sum_a )
      gamma_other_gathered <- eval(parse(text = paste('list(',
              paste(paste('term2_FC_a',zer,sep=''),collapse=','),')',collapse='')))

      tmp <- jnp$concatenate(sapply(1:length(zer), function(drop_indi){
          if(length(zer) == 1){ tmp_return <- jnp$zeros(c(1L,1L)) }
          if(length(zer)>1){
          # gamma part
          gamma_other <- jnp$concatenate(gamma_other_gathered[-drop_indi],0L)
          gamma_other <- jnp$expand_dims(gamma_other,1L) # experimental

          # pi part
          gathered_a_simplex_indices_others <- eval(parse(text=paste('list(',paste(paste('term2_FC_b',zer[-drop_indi],sep=''),collapse=','),')',collapse='')))
          segment_ids <- jnp$array(jnp$concatenate(sapply(1:length(gathered_a_simplex_indices_others),function(ra){
            list(jnp$ones(jnp$shape(gathered_a_simplex_indices_others[[ra]]))*(ra-1L)) }),0L),jnp$int32)
          gathered_a_simplex_indices_others <- jnp$concatenate(gathered_a_simplex_indices_others,0L)
          inter_prob_others_of_others <-  jnp$take(a_simplex, indices = gathered_a_simplex_indices_others, axis = 0L)
          PROD_interaction_others_of_others_prob <- jnp$multiply(gamma_other,inter_prob_others_of_others)

          exp_other <- jnp$take(exp_a_gathered,n2int((1:length(zer)-1L)[-drop_indi]))
          exp_self <- jnp$take(exp_a_gathered,n2int(ai(drop_indi-1L)))
          prod_other_self <- jnp$negative(jnp$divide(jnp$multiply(exp_other,exp_self),sum_a_squared))
          prod_other_self <- jnp$expand_dims(prod_other_self,0L) # new 
          prod_other_self <- jnp$take(prod_other_self,segment_ids,axis=0L)
          if(length(unlist(prod_other_self$shape))==1){prod_other_self <- jnp$expand_dims(prod_other_self,1L)}
          tmp_return <- jnp$sum(jnp$multiply(prod_other_self,PROD_interaction_others_of_others_prob),keepdims=T)
        }

          return( tmp_return )
        }), 0L)
        if(length(unlist(tmp$shape)) == 0){ tmp <- jnp$expand_dims( tmp, 1L ) }
      return( list( tmp ) ) })
    names(term2_other) <- NULL
    term2_other <- jnp$concatenate(term2_other,0L)
    term2 <- jnp$add(term2_self, term2_other)
    
    print2('TERM 3 DERIVATIVE CONTRIBUTION')
    term3_self <- jnp$multiply(Neg2_tf,jnp$multiply(jnp$multiply(lambda,jnp$expand_dims(dydx_a_simplex,1L)),
                                                  jnp$subtract(a_simplex, p_vec_tf) ))#done
    term3_other <- tapply(1:nrow(main_info),main_info$d,function(zer){
      p_gathered <- jnp$take(p_vec_tf, n2int(ai(zer-1L)))
      exp_a_gathered <- jnp$take(exp_a, n2int(ai(zer-1L)))
      sum_a <- jnp$add(OneTf_flat,jnp$sum(exp_a_gathered))
      sum_a_squared <- jnp$square( sum_a  )
      tmp <- jnp$stack(sapply(1:length(zer), function(drop_indi){
        p_other <- jnp$take(p_gathered, n2int((1:length(zer)-1L)[-drop_indi]))
        exp_other <- jnp$take(exp_a_gathered, n2int((1:length(zer)-1L)[-drop_indi]))
        exp_self <- jnp$take(exp_a_gathered,n2int(ai(drop_indi-1L)))
        list( jnp$sum(jnp$negative(jnp$divide(exp_other*exp_self,sum_a_squared))*(exp_other/sum_a - p_other),
                      keepdims=T) )
        }), 0L)
      if(length(unlist(tmp$shape)) == 1){ tmp <- jnp$expand_dims( tmp, 1L ) }
      return( list( tmp ) ) })
    names(term3_other) <- NULL
    term3_other <- jnp$concatenate(term3_other,0L)
    term3_other <- jnp$multiply(Neg2_tf,jnp$multiply(lambda, term3_other))
    term3 <- jnp$add(term3_self, term3_other)

    print2('TERM 4 DERIVATIVE CONTRIBUTION')
    term4_FC_ <- jnp$expand_dims(jnp$concatenate(sapply(1:n_main_params,function(xer){
      # sum p
      gathered_sum_p <- jnp$expand_dims(jnp$take(term4_FC_a,indices = n2int(ai(xer-1L)),axis=0L),0L)

      # sum pi
      gathered_a_simplex_indices <- eval(parse(text=paste('term4_FC_b',xer,sep='')))
      sum_pi <- jnp$expand_dims(jnp$sum(jnp$take(a_simplex,indices = gathered_a_simplex_indices,axis=0L)),0L)

      # output
      return( jnp$subtract(sum_pi,  gathered_sum_p) )
    }),0L),1L)
    term4 <- jnp$multiply(jnp$multiply(Neg2_tf,lambda),jnp$multiply(jnp$expand_dims(a2Term4(a_vec),1L),term4_FC_)) # done

    print2('COMBINE ALL GRADIENT TERMS')
    grad_i <- jnp$add(jnp$add(term1, term2), jnp$add(term3, term4))

    # update
    inv_learning_rate_i <-  GetInvLR(inv_learning_rate, grad_i)

    # update parameters
    a_vec_updated <- GetUpdatedParameters(a_vec = a_vec,
                                          grad_i = grad_i,
                                          inv_learning_rate_i = inv_learning_rate_i)

    return( list(a_vec_updated, inv_learning_rate_i, grad_i) )
  })
  ",
                            paste(paste(paste("term2_FC_a",1:n_main_params,sep = ""),
                                        paste("term2_FC_b",1:n_main_params,sep = ""),
                                        paste("term4_FC_b",1:n_main_params,sep = ""),sep=","),collapse=",")
  )))
}
