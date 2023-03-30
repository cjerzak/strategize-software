#' generate_ManualDoUpdates
#'
#' Implements the organizational record linkage algorithms of Jerzak and Libgober (2021).
#'
#' @usage
#'
#' generate_ManualDoUpdates(x, y, by ...)
#'
#' @param x,y data frames to be merged
#'
#' @return `z` The merged data frame.
#' @export
#'
#' @details `LinkOrgs` automatically processes the name text for each dataset (specified by `by`, `by.x`, and/or `by.y`. Users may specify the following options:
#'
#' - Set `DistanceMeasure` to control algorithm for computing pairwise string distances. Options include "`osa`", "`jaccard`", "`jw`". See `?stringdist::stringdist` for all options. (Default is "`jaccard`")
#'
#' @examples
#'
#' #Create synthetic data
#' x_orgnames <- c("apple","oracle","enron inc.","mcdonalds corporation")
#' y_orgnames <- c("apple corp","oracle inc","enron","mcdonalds co")
#' x <- data.frame("orgnames_x"=x_orgnames)
#' y <- data.frame("orgnames_y"=y_orgnames)
#'
#' # Perform merge
#' linkedOrgs <- LinkOrgs(x = x,
#'                        y = y,
#'                        by.x = "orgnames_x",
#'                        by.y = "orgnames_y",
#'                        MaxDist = 0.6)
#'
#' print( linkedOrgs )
#'
#' @export
#'
#' @md
#'

generate_ManualDoUpdates <- function(){

    a2Grad <- tf_function_ex(function(a_){
      exp_a_ <- tf$exp(  a_  )
      aGrad <- tapply(1:nrow(main_info),main_info$d,function(zer){
        exp_gathered <- tf$gather(exp_a_, n2int(zer-1L))
        sum_exp_a <- tf$add( OneTf_flat, tf$reduce_sum(exp_gathered) )
        tmp <- tf$multiply(tf$divide( exp_gathered,  tf$square(sum_exp_a) ),
                           sum_exp_a - exp_gathered)
        if(length(dim(tmp)) == 1){ tmp <- tf$expand_dims( tmp, 0L ) }
        return( list( tmp ) ) })
      names(aGrad) <- NULL
      return( tf$concat(aGrad,0L) )
    })
    a2Term4 <- tf_function_ex(function(a_){
      exp_a_ <- tf$exp( a_ )
      comp_ <- tapply(1:nrow(main_info),main_info$d,function(zer){
        exp_a_gathered <- tf$gather(exp_a_, n2int(zer-1L))
        sum_exp_a <- tf$add(OneTf_flat, tf$reduce_sum( exp_a_gathered ))
        tmp <- tf$divide( exp_a_gathered, tf$square( sum_exp_a ) )
        if(length(dim(tmp)) == 1){ tmp <- tf$expand_dims( tmp, 0L ) }
        return( list( tmp ) ) })
      names(comp_) <- NULL
      return( tf$concat(comp_,0L) )
    })
    doUpdate <- tf_function_ex(doUpdate_r <- function( a_vec, inv_learning_rate,
                                                       main_coef,inter_coef,
                                                       term2_FC_a, term2_FC_b,
                                                       term4_FC_a, term4_FC_b
    ){
      dydx_a_simplex <- a2Grad(  a_vec )
      a_simplex <- a2Simplex( a_vec )
      exp_a <- tf$exp( a_vec )

      # TERM 1 DERIVATIVE CONTRIBUTION
      term1_self <- tf$multiply(dydx_a_simplex, main_coef) #done
      term1_other <- tapply(1:nrow(main_info),main_info$d,function(zer){
        exp_a_gathered <- tf$gather(exp_a, n2int(ai(zer-1L)))
        main_coef_gathered <- tf$gather(main_coef, n2int(ai(zer-1L)))
        sum_a_squared <- tf$square(  tf$add(OneTf_flat, tf$reduce_sum(exp_a_gathered) ))
        tmp <- tf$concat(sapply(1:length(zer), function(drop_indi){
          index_other_drop_self <- n2int((1:length(zer)-1L)[-drop_indi])
          beta_other <- tf$gather(main_coef_gathered, index_other_drop_self,axis=0L)
          exp_other <- tf$gather(exp_a_gathered, index_other_drop_self,axis=0L)
          exp_self <- tf$gather(exp_a_gathered,n2int(ai(drop_indi-1L)),axis=0L)
          return( tf$reduce_sum(
            tf$multiply(beta_other,tf$negative(tf$divide(tf$multiply(exp_other,exp_self),sum_a_squared))),keepdims=T) )
        }), 0L)
        if(length(dim(tmp)) == 1){ tmp <- tf$expand_dims( tmp, 1L ) }
        return( list( tmp ) ) })
      names(term1_other) <- NULL
      term1_other <- tf$concat(term1_other, 0L)
      term1 <- tf$add(term1_self, term1_other)

      # TERM 2 DERIVATIVE CONTRIBUTION
      term2_FC_self <- tf$concat(sapply(1:length(a_vec),function(xer){
        gathered_coef_values <- tf$expand_dims(tf$gather(term2_FC_a,n2int(ai(xer - 1L)),axis=0L)$flat_values,1L)
        gathered_a_simplex_indices <- tf$gather(term2_FC_b,n2int(ai(xer-1L)),axis=0L)
        inter_prob_ <-  tf$gather(a_simplex, indices = gathered_a_simplex_indices, axis = 0L)
        return(  tf$expand_dims(tf$expand_dims(
          tf$reduce_sum(tf$multiply(gathered_coef_values, inter_prob_)),0L), 1L))
      }),0L)
      term2_self <- tf$multiply(dydx_a_simplex, term2_FC_self) # done
      term2_other <- tapply(1:nrow(main_info),main_info$d,function(zer){
        exp_a_gathered <- tf$gather(exp_a, n2int(ai(zer-1L)))
        sum_a <- tf$add(  OneTf_flat, tf$reduce_sum(exp_a_gathered)  )
        sum_a_squared <- tf$square(  sum_a )
        gamma_other_gathered <- tf$gather(term2_FC_a, n2int(ai(zer-1L)),axis=0L)

        tmp <- tf$concat(sapply(1:length(zer), function(drop_indi){
          gamma_other_rag <- tf$gather(gamma_other_gathered, n2int(ai(1:length(zer)-1L)[-drop_indi]),axis=0L)
          gathered_a_simplex_indices_others <- tf$gather(term2_FC_b, n2int(ai(zer-1L)[-drop_indi]),axis=0L)
          if(isFlat <-"flat_values" %in% names(gamma_other_rag)){ gamma_other <- gamma_other_rag$flat_values }
          if(!isFlat){ gamma_other <- gamma_other_rag }
          if(length(zer) <= 2){ segment_ids <- tf$zeros(tf$shape(gamma_other),dtype=tf$int32) }
          if(length(zer) > 2){ segment_ids <- tf$cast(tf$ragged$row_splits_to_segment_ids(gamma_other_rag$row_splits),tf$int32) }
          if("flat_values" %in% names(gathered_a_simplex_indices_others)){
            gathered_a_simplex_indices_others <- gathered_a_simplex_indices_others$flat_values
          }
          inter_prob_others_of_others <-  tf$gather(a_simplex, indices = gathered_a_simplex_indices_others, axis = 0L)
          if("flat_values" %in% names(inter_prob_others_of_others)){
            inter_prob_others_of_others <- inter_prob_others_of_others$flat_values
          }
          PROD_interaction_others_of_others_prob <- tf$multiply(tf$expand_dims(gamma_other,1L),inter_prob_others_of_others)

          exp_other <- tf$gather(exp_a_gathered,n2int((1:length(zer)-1L)[-drop_indi]))
          exp_self <- tf$gather(exp_a_gathered,n2int(ai(drop_indi-1L)))
          prod_other_self <- tf$negative(tf$divide(tf$multiply(exp_other,exp_self),sum_a_squared))
          prod_other_self <- tf$gather(prod_other_self,segment_ids,axis=0L)
          if(length(dim(prod_other_self))==1){prod_other_self <- tf$expand_dims(prod_other_self,1L)}
          return( tf$reduce_sum(tf$multiply(prod_other_self,PROD_interaction_others_of_others_prob),keepdims=T) )
        }), 0L)
        if(length(dim(tmp)) == 1){ tmp <- tf$expand_dims( tmp, 1L ) }
        return( list( tmp ) ) })
      names(term2_other) <- NULL
      term2_other <- tf$concat(term2_other,0L)
      term2 <- tf$add(term2_self, term2_other)

      # TERM 3 DERIVATIVE CONTRIBUTION
      term3_self <- tf$multiply(Neg2_tf,tf$multiply(tf$multiply(lambda,dydx_a_simplex),
                                                    tf$subtract(a_simplex, p_vec_tf) ))#done
      term3_other <- tapply(1:nrow(main_info),main_info$d,function(zer){
        p_gathered <- tf$gather(p_vec_tf, n2int(ai(zer-1L)))
        exp_a_gathered <- tf$gather(exp_a, n2int(ai(zer-1L)))
        sum_a <- tf$add(OneTf_flat,tf$reduce_sum(exp_a_gathered))
        sum_a_squared <- tf$square( sum_a  )
        tmp <- tf$concat(sapply(1:length(zer), function(drop_indi){
          p_other <- tf$gather(p_gathered, n2int((1:length(zer)-1L)[-drop_indi]))
          exp_other <- tf$gather(exp_a_gathered, n2int((1:length(zer)-1L)[-drop_indi]))
          exp_self <- tf$gather(exp_a_gathered,n2int(ai(drop_indi-1L)))
          tf$reduce_sum(tf$negative(tf$divide(exp_other*exp_self,sum_a_squared))*(exp_other/sum_a - p_other),
                        keepdims=T) }), 0L)
        if(length(dim(tmp)) == 1){ tmp <- tf$expand_dims( tmp, 1L ) }
        return( list( tmp ) ) })
      names(term3_other) <- NULL
      term3_other <- tf$concat(term3_other,0L)
      term3_other <- tf$multiply(Neg2_tf,tf$multiply(lambda, term3_other))
      term3 <- tf$add(term3_self, term3_other)

      # TERM 4 DERIVATIVE CONTRIBUTION
      term4_FC_ <- tf$expand_dims(tf$concat(sapply(1:length(a_vec),function(xer){
        gathered_sum_p <- tf$expand_dims(tf$gather(term4_FC_a,indices = n2int(ai(xer-1L)),axis=0L),0L)
        gathered_a_simplex_indices <- tf$gather(term4_FC_b,indices = n2int(ai(xer-1L)),axis=0L)$flat_values
        sum_pi <- tf$expand_dims(tf$reduce_sum(tf$gather(a_simplex,indices = gathered_a_simplex_indices,axis=0L)),0L)
        return( tf$subtract(sum_pi,  gathered_sum_p) )
      }),0L),1L)
      term4 <- tf$multiply(tf$multiply(Neg2_tf,lambda),tf$multiply(a2Term4(a_vec),term4_FC_)) # done

      # COMBINE ALL GRADIENT TERMS
      grad_i <- tf$add(tf$add(term1, term2), tf$add(term3, term4))

      # update
      inv_learning_rate_i <-  GetInvLR(inv_learning_rate, grad_i)

      # update parameters
      a_vec_updated <- GetUpdatedParameters(a_vec = a_vec,
                                            grad_i = grad_i,
                                            inv_learning_rate_i = tf$sqrt( inv_learning_rate_i) )

      return( list(a_vec_updated, inv_learning_rate_i,grad_i) )
    })


  eval(parse(text = sprintf("doUpdate_simp <- tf_function_ex(doUpdate_simp_ <- function( a_vec, inv_learning_rate,
                                                     main_coef,inter_coef,
                                                     %s,term4_FC_a
                                                     ){
    dydx_a_simplex <- a2Grad(  a_vec )
    a_simplex <- a2Simplex( a_vec )
    exp_a <- tf$exp( a_vec )

    # TERM 1 DERIVATIVE CONTRIBUTION
    term1_self <- tf$multiply(dydx_a_simplex, main_coef) #done
    term1_other <- tapply(1:nrow(main_info),main_info$d,function(zer){
        exp_a_gathered <- tf$gather(exp_a, n2int(ai(zer-1L)))
        main_coef_gathered <- tf$gather(main_coef, n2int(ai(zer-1L)))
        sum_a_squared <- tf$square(  tf$add(OneTf_flat, tf$reduce_sum(exp_a_gathered) ))
        tmp <- tf$concat(sapply(1:length(zer), function(drop_indi){
          index_other_drop_self <- n2int((1:length(zer)-1L)[-drop_indi])
          beta_other <- tf$gather(main_coef_gathered, index_other_drop_self,axis=0L)
          exp_other <- tf$gather(exp_a_gathered, index_other_drop_self,axis=0L)
          exp_self <- tf$gather(exp_a_gathered,n2int(ai(drop_indi-1L)),axis=0L)
          return( tf$reduce_sum(
            tf$multiply(beta_other,tf$negative(tf$divide(tf$multiply(exp_other,exp_self),sum_a_squared))),keepdims=T) )
        }), 0L)
        if(length(dim(tmp)) == 1){ tmp <- tf$expand_dims( tmp, 1L ) }
        return( list( tmp ) ) })
    names(term1_other) <- NULL
    term1_other <- tf$concat(term1_other, 0L)
    term1 <- tf$add(term1_self, term1_other)

    # TERM 2 DERIVATIVE CONTRIBUTION
    term2_FC_self <- tf$concat(sapply(1:n_main_params,function(xer){
        gathered_coef_values <- eval(parse(text=sprintf('term2_FC_a%%s',xer)))
        gathered_a_simplex_indices <- eval(parse(text=sprintf('term2_FC_b%%s',xer)))
        inter_prob_ <-  tf$gather(a_simplex, indices = gathered_a_simplex_indices, axis = 0L)
        return(  tf$expand_dims(tf$expand_dims(
          tf$reduce_sum(tf$multiply(gathered_coef_values, inter_prob_)),0L), 1L) )
      }),0L)
    term2_self <- tf$multiply(dydx_a_simplex, term2_FC_self) # done

    term2_other <- tapply(1:nrow(main_info),main_info$d,function(zer){
      exp_a_gathered <- tf$gather(exp_a, n2int(ai(zer-1L)))
      sum_a <- tf$add(  OneTf_flat, tf$reduce_sum(exp_a_gathered)  )
      sum_a_squared <- tf$square(  sum_a )
      gamma_other_gathered <- eval(parse(text = paste('list(',
              paste(paste('term2_FC_a',zer,sep=''),collapse=','),')',collapse='')))

      tmp <- tf$concat(sapply(1:length(zer), function(drop_indi){
          if(length(zer) == 1){ tmp_return <- tf$zeros(c(1L,1L)) }
          if(length(zer)>1){
          # gamma part
          gamma_other <- tf$concat(gamma_other_gathered[-drop_indi],0L)

          # pi part
          gathered_a_simplex_indices_others <- eval(parse(text=paste('list(',paste(paste('term2_FC_b',zer[-drop_indi],sep=''),collapse=','),')',collapse='')))
          segment_ids <- tf$cast(tf$concat(sapply(1:length(gathered_a_simplex_indices_others),function(ra){
            list(tf$ones(tf$shape(gathered_a_simplex_indices_others[[ra]]))*(ra-1L)) }),0L),tf$int32)
          gathered_a_simplex_indices_others <- tf$concat(gathered_a_simplex_indices_others,0L)
          inter_prob_others_of_others <-  tf$gather(a_simplex, indices = gathered_a_simplex_indices_others, axis = 0L)
          PROD_interaction_others_of_others_prob <- tf$multiply(gamma_other,inter_prob_others_of_others)

          exp_other <- tf$gather(exp_a_gathered,n2int((1:length(zer)-1L)[-drop_indi]))
          exp_self <- tf$gather(exp_a_gathered,n2int(ai(drop_indi-1L)))
          prod_other_self <- tf$negative(tf$divide(tf$multiply(exp_other,exp_self),sum_a_squared))
          prod_other_self <- tf$gather(prod_other_self,segment_ids,axis=0L)
          if(length(dim(prod_other_self))==1){prod_other_self <- tf$expand_dims(prod_other_self,1L)}
          tmp_return <- tf$reduce_sum(tf$multiply(prod_other_self,PROD_interaction_others_of_others_prob),keepdims=T)
        }

          return( tmp_return )
        }), 0L)
        if(length(dim(tmp)) == 1){ tmp <- tf$expand_dims( tmp, 1L ) }
      return( list( tmp ) ) })
    names(term2_other) <- NULL
    term2_other <- tf$concat(term2_other,0L)
    term2 <- tf$add(term2_self, term2_other)

    # TERM 3 DERIVATIVE CONTRIBUTION
    term3_self <- tf$multiply(Neg2_tf,tf$multiply(tf$multiply(lambda,dydx_a_simplex),
                                                  tf$subtract(a_simplex, p_vec_tf) ))#done
    term3_other <- tapply(1:nrow(main_info),main_info$d,function(zer){
      p_gathered <- tf$gather(p_vec_tf, n2int(ai(zer-1L)))
      exp_a_gathered <- tf$gather(exp_a, n2int(ai(zer-1L)))
      sum_a <- tf$add(OneTf_flat,tf$reduce_sum(exp_a_gathered))
      sum_a_squared <- tf$square( sum_a  )
      tmp <- tf$concat(sapply(1:length(zer), function(drop_indi){
        p_other <- tf$gather(p_gathered, n2int((1:length(zer)-1L)[-drop_indi]))
        exp_other <- tf$gather(exp_a_gathered, n2int((1:length(zer)-1L)[-drop_indi]))
        exp_self <- tf$gather(exp_a_gathered,n2int(ai(drop_indi-1L)))
        tf$reduce_sum(tf$negative(tf$divide(exp_other*exp_self,sum_a_squared))*(exp_other/sum_a - p_other),
                      keepdims=T) }), 0L)
      if(length(dim(tmp)) == 1){ tmp <- tf$expand_dims( tmp, 1L ) }
      return( list( tmp ) ) })
    names(term3_other) <- NULL
    term3_other <- tf$concat(term3_other,0L)
    term3_other <- tf$multiply(Neg2_tf,tf$multiply(lambda, term3_other))
    term3 <- tf$add(term3_self, term3_other)

    # TERM 4 DERIVATIVE CONTRIBUTION
    term4_FC_ <- tf$expand_dims(tf$concat(sapply(1:n_main_params,function(xer){
      # sum p
      gathered_sum_p <- tf$expand_dims(tf$gather(term4_FC_a,indices = n2int(ai(xer-1L)),axis=0L),0L)

      # sum pi
      gathered_a_simplex_indices <- eval(parse(text=paste('term4_FC_b',xer,sep='')))
      sum_pi <- tf$expand_dims(tf$reduce_sum(tf$gather(a_simplex,indices = gathered_a_simplex_indices,axis=0L)),0L)

      # output
      return( tf$subtract(sum_pi,  gathered_sum_p) )
    }),0L),1L)
    term4 <- tf$multiply(tf$multiply(Neg2_tf,lambda),tf$multiply(a2Term4(a_vec),term4_FC_)) # done

    # COMBINE ALL GRADIENT TERMS
    grad_i <- tf$add(tf$add(term1, term2), tf$add(term3, term4))

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
