if(!diff){
  getFixedEntries <- compile_fxn(function(EST_COEFFICIENTS_tf){
    main_coef <- jnp$take(EST_COEFFICIENTS_tf, indices = main_indices_i0, axis=0L)
    
    {
      inter_coef <- jnp$take(EST_COEFFICIENTS_tf, indices = inter_indices_i0, axis=0L)
      # term 2 fix contribution
      term2_FC <- sapply(1:n_main_params,function(main_comp){
        interaction_info_red <- interaction_info[
          (ind1<-(interaction_info$d_adj %in% main_info[main_comp,]$d_adj &
                    interaction_info$l %in% main_info[main_comp,]$l)) |
            (ind2<-(interaction_info$dp_adj %in% main_info[main_comp,]$d_adj &
                      interaction_info$lp %in% main_info[main_comp,]$l ) ),]
        id_d <- apply(interaction_info_red[,c("d_adj","l")],1,function(zer){paste(zer,collapse="_")})
        id_dp <- apply(interaction_info_red[,c("dp_adj","lp")],1,function(zer){paste(zer,collapse="_")})
        id_ <- ifelse(!(interaction_info_red$d_adj %in% main_info[main_comp,]$d_adj),
                      yes = id_d, no = id_dp)
        id_main <- apply(main_info[,c("d_adj","l")],1,function(zer){paste(zer,collapse="_")})
        which_inter <- which(ind1|ind2)
        inter_into_main_0i <- n2int(ai(sapply(id_,function(zer){which(id_main %in% zer)})-1L))
        
        if(nrow(interaction_info_red)>0){
          inter_coef_ <- jnp$take(inter_coef,indices = n2int(ai(which_inter-1L)), axis = 0L)
        }
        
        # expand dimensions in the length == 1 case
        if(length(which_inter) == 1){
          inter_coef_ <- jnp$expand_dims(inter_coef_,0L)
          inter_into_main <- jnp$expand_dims(inter_into_main_0i,0L)
        }
        return( list("inter_coef_" = inter_coef_,
                     "indices_on_a_simplex_for_inter_prob" = inter_into_main_0i) )
      })
      for(jf in 1:length(term2_FC[1,])){
        eval(parse(text=sprintf("term2_FC_a%s <- term2_FC[1,][[jf]]",jf)))
        eval(parse(text=sprintf("term2_FC_b%s <- term2_FC[2,][[jf]]",jf)))
      }
      
      # term 4 fix contribution
      term4_FC <- sapply(1:n_main_params,function(main_comp){
        which_d <- which(d_locator[main_comp] == d_locator)
        sum_p <- jnp$expand_dims(jnp$sum(jnp$take(p_vec_tf,
                                                  indices = n2int(ai(which_d-1L)), axis=0L),keepdims=F),0L)
        return(   list("sum_p"=sum_p,
                       "indices_for_sum_pi"=n2int(as.matrix(ai(which_d-1L))))  )
      })
      term4_FC_a <- jnp$concatenate(term4_FC[1,],0L)
      for(jf in 1:length(term4_FC[2,])){
        eval(parse(text = sprintf("term4_FC_b%s = term4_FC[2,][[jf]]",jf)))
      }
      add_text <- c(paste("term2_FC_a",1:n_main_params,sep=""),
                    paste("term2_FC_b",1:n_main_params,sep=""),
                    paste("term4_FC_b",1:n_main_params,sep=""))
      add_text <- sapply(add_text,function(zer){sprintf("'%s'=%s",zer,zer)})
      add_text <- paste(add_text,collapse=",")
      eval(parse(text = sprintf("l_res <- list(
                      'main_coef'=main_coef,
                      'inter_coef'=inter_coef,
                      'term4_FC_a'=term4_FC_a,
                      %s)",add_text)))
    }
    return( l_res )
  })
  fe <- getFixedEntries( EST_COEFFICIENTS_tf ) # needed for function initialization
}