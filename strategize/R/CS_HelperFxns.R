ess_fxn <- function(wz){ sum(wz)^2 / sum(wz^2)}

toSimplex = function(x){
  x[x>22] <- 22; x[x< -22] <- -22
  sim_x = exp(x)/sum(exp(x))
  if(any(is.nan(sim_x))){browser()}
  return(sim_x)
}

RescaleFxn <- function(x, estMean=NULL, estSD=NULL, center=T){
  if(center == T){ x <- x *  estSD + estMean }
  if(center == F){ x <- x *  estSD }
  return( x )   }

NA20 <- function(zer){zer[is.na(zer)]<-0;zer}

rzip_tf <- (rzip<-function(l1,l2){  fl<-list(); for(aia in 1:length(l1)){ fl[[aia]] <- list(l1[[aia]], l2[[aia]]) }; return( fl  ) })

getSE <- function(er){ sqrt( var(er,na.rm=T) /  length(na.omit(er)) )  }

logPrWGivenPi_fxn = function(doc_indices,pi_mat,terms_posterior,log_=F,
                          doc_indices_u = NULL,d_indices_u = NULL,documents_list_=NULL){
  if(is.null(doc_indices_u)){
    doc_indices_u = unlist(doc_indices,recursive = F)
    d_indices_u = unlist(sapply(1:length(doc_indices),
                                function(se){list(rep(se,length(doc_indices[[se]])))}))
  }
  row.names(terms_posterior) <- colnames( terms_posterior ) <- row.names(pi_mat) <- colnames(pi_mat) <- NULL
  terms_posterior = as.matrix( terms_posterior )
  pr_w_position_given_theta = try(colSums(terms_posterior[,doc_indices_u] * pi_mat[,d_indices_u]),T)
  pr_w_given_theta = c(tapply(1:length(d_indices_u),d_indices_u,function(indi_){ sum(log( pr_w_position_given_theta[indi_] ))}))
  return( pr_w_given_theta)
}

se <- function(.){sqrt(1/length(.) * var(.))}

getMultinomialSamp <- function(pi_star_value, baseSeed){
  {
    # define d locator
    d_locator_use <- ifelse(ParameterizationType == "Implicit",
                            yes = list(d_locator), no = list(d_locator_full))[[1]]

    # get t samp
    T_star_samp_reduced <- tapply(1:length(d_locator_use),d_locator_use,function(zer){
      pi_selection <- jnp$take(pi_star_value, jnp$array(n2int(zer <- ai(  zer  ) - 1L)),0L)

      # add additional entry if implicit t ype
      if(ParameterizationType == "Implicit"){
        if(length(zer) == 1){ pi_selection <- jnp$expand_dims(pi_selection,0L) }
        pi_implied <- jnp$expand_dims(jnp$expand_dims(jnp$subtract(jnp$array(1.), jnp$sum(pi_selection)),0L),0L)
        pi_selection <- jnp$concatenate(list(pi_implied,pi_selection))
      }

      TDist <- oryx$distributions$RelaxedOneHotCategorical(
        probs = jnp$transpose(pi_selection),# row = simplex entry
        temperature = jnp$array(0.5))
      TSamp <- TDist$sample(size = 1L, seed = JaxKey( jnp$add(jnp$take(jnp$array(zer),0L),baseSeed)))
      TSamp <- jnp$transpose( TSamp )

      # if implicit, drop a term to keep correct shapes
      print("CONFIRM THAT DROPPING THE FIRST TERM IS CORRECT HERE IN getMultinomialSamp")
      #if(ParameterizationType == "Implicit"){ TSamp <- jnp$take(TSamp,jnp$array(ai(0L:(length(zer)-1L)),axis=0L) } #drop last entry
      if(ParameterizationType == "Implicit"){ TSamp <- jnp$take(TSamp,jnp$array(ai(1L:length(zer))),axis=0L) } #drop first entry
      if(length(zer) == 1){TSamp <- jnp$expand_dims(TSamp, 1L)}
      return (  TSamp   )
    })
    names(T_star_samp_reduced) <- NULL # drop name to allow concatenation
    T_star_samp_reduced <-  jnp$concatenate(unlist(T_star_samp_reduced),0L)
  }

  return( T_star_samp_reduced )
}

getPrettyPi <- function(pi_star_value){
  if(ParameterizationType == "Full"){
    #pi_star_full <- tapply(1:length(d_locator_full),d_locator_full,function(zer){jnp$take(pi_star_value,n2int(ai(zer-1L))) })
    pi_star_full <- pi_star_value
  }
  if(ParameterizationType == "Implicit"){
    pi_star_impliedTerms <- tapply(1:length(d_locator),d_locator,function(zer){
            pi_implied <- jnp$subtract(OneTf, jnp$sum(jnp$take(pi_star_value,
                                             n2int(ai(zer-1L)),0L)))
    })

    names(pi_star_impliedTerms) <- NULL
    pi_star_impliedTerms <- jnp$concatenate(pi_star_impliedTerms,0L)

    pi_star_full <- jnp$add(jnp$matmul(main_comp_mat, pi_star_value),
                            jnp$matmul(shadow_comp_mat, pi_star_impliedTerms))
  }

  return( pi_star_full )
}
