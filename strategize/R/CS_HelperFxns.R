print2 <- function(text, quiet = F){
  if(!quiet){ print( sprintf("[%s] %s" ,format(Sys.time(), "%Y-%m-%d %H:%M:%S"),text) ) }
}

ess_fxn <- function(wz){ sum(wz)^2 / sum(wz^2)}

toSimplex = function(x){
  x[x>22] <- 22; x[x< -22] <- -22
  sim_x = exp(x)/sum(exp(x))
  if(any(is.nan(sim_x))){browser()}
  return(sim_x)
}

ai <- as.integer

RescaleFxn <- function(x, estMean=NULL, estSD=NULL, center=T){
  return(  x*estSD + ifelse(center, yes = estMean, no = 0) ) 
}

NA20 <- function(zer){zer[is.na(zer)]<-0;zer}

getSE <- function(er){ sqrt( var(er,na.rm=T) /  length(na.omit(er)) )  }

se <- function(.){sqrt(1/length(.) * var(.))}

getMultinomialSamp <- function(pi_value, temperature, jax_seed){
  # define d locator
  d_locator_use <- ifelse(ParameterizationType == "Implicit",
                          yes = list(d_locator), no = list(d_locator_full))[[1]]

  # get t samp
  T_star_samp <- tapply(1:length(d_locator_use),d_locator_use,function(zer){
    pi_selection <- jnp$take(pi_value, jnp$array(n2int(zer <- ai(  zer  ) - 1L)),0L)

    # add additional entry if implicit t ype
    if(ParameterizationType == "Implicit"){
      if(length(zer) == 1){ pi_selection <- jnp$expand_dims(pi_selection,0L) }
      pi_implied <- jnp$expand_dims(jnp$expand_dims(jnp$subtract(jnp$array(1.), jnp$sum(pi_selection)),0L),0L)
      
      # add holdout 
      # pi_selection <- jnp$concatenate(list(pi_implied, pi_selection)) # add FIRST entry 
      pi_selection <- jnp$concatenate(list(pi_selection, pi_implied)) # add LAST entry 
    }

    TSamp <- oryx$distributions$RelaxedOneHotCategorical(
      probs = pi_selection$transpose(),
      temperature = temperature)$sample(size = 1L, seed = jax_seed)$transpose()

    # if implicit, drop a term to keep correct shapes
    #if(ParameterizationType == "Implicit"){ TSamp <- jnp$take(TSamp,jnp$array(ai(1L:length(zer))),axis=0L) } #drop FIRST entry
    if(ParameterizationType == "Implicit"){ TSamp <- jnp$take(TSamp,jnp$array(ai(0L:(length(zer)-1L))),axis=0L) } #drop LAST entry
    
    if(length(zer) == 1){TSamp <- jnp$expand_dims(TSamp, 1L)}
    return (  TSamp   )
  })
  names(T_star_samp) <- NULL # drop name to allow concatenation
  return( T_star_samp <-  jnp$concatenate(unlist(T_star_samp),0L) ) 
}

getPrettyPi <- function( pi_star_value ){
  if(ParameterizationType == "Full"){
    #pi_star_full <- tapply(1:length(d_locator_full),d_locator_full,function(zer){jnp$take(pi_star_value,n2int(ai(zer-1L))) })
    pi_star_full <- pi_star_value
  }
  if(ParameterizationType == "Implicit"){
    pi_star_impliedTerms <- tapply(1:length(d_locator),d_locator,function(zer){
            pi_implied <- jnp$subtract(OneTf, jnp$sum(jnp$take(pi_star_value,
                                             n2int(ai(zer-1L)),0L))) })

    names(pi_star_impliedTerms) <- NULL
    pi_star_impliedTerms <- jnp$concatenate(pi_star_impliedTerms,0L)

    pi_star_full <- jnp$expand_dims(jnp$add(jnp$matmul(main_comp_mat, pi_star_value)$flatten(),
                            jnp$matmul(shadow_comp_mat, pi_star_impliedTerms)$flatten()),1L)
  }

  return( pi_star_full )
}

computeQ_conjoint_internal <- function(FactorsMat_internal,
                                       Yobs_internal,
                                       FactorsMat_internal_mapped = NULL,
                                       hypotheticalProbList_internal,
                                       assignmentProbList_internal,
                                       log_pr_w_internal = NULL,
                                       hajek = T, knownNormalizationFactor = NULL,
                                       computeLB  = F){
  if(is.null(log_pr_w_internal)){
    log_pr_w_internal <- log( sapply(1:ncol(FactorsMat_internal),function(ze){
      (assignmentProbList_internal[[ze]][ FactorsMat_internal[,ze] ]) }) )
    if(all(class(log_pr_w_internal) == "numeric")){ log_pr_w_internal <- sum(log_pr_w_internal)}
    if(any(class(log_pr_w_internal) != "numeric")){ log_pr_w_internal = rowSums(log_pr_w_internal)}
  }

  # new probability
  if(!is.null(FactorsMat_internal_mapped)){FactorsMat_internal_mapped[] <- log(unlist(hypotheticalProbList_internal)[FactorsMat_internal_mapped])}
  if(is.null(FactorsMat_internal_mapped)){FactorsMat_internal_mapped <- log( sapply(1:ncol(FactorsMat_internal),function(ze){hypotheticalProbList_internal[[ze]][ FactorsMat_internal[,ze] ]  })  )}
  if(any(class(FactorsMat_internal_mapped) != "numeric")){ log_pr_w_new <- rowSums(FactorsMat_internal_mapped)}
  if(all(class(FactorsMat_internal_mapped) == "numeric")){ log_pr_w_new <- sum(FactorsMat_internal_mapped)}
  my_wts <- exp(  log_pr_w_new   - log_pr_w_internal  )
  sum_raw_wts <- sum( my_wts )
  if(hajek == T){
    #NOTE: Hajek == mean(Yobs_internal*my_wts) / mean( my_wts ), with my_wts unnormalized
    if(is.null(knownNormalizationFactor)){  my_wts <- my_wts / sum_raw_wts }
    if(!is.null(knownNormalizationFactor)){  my_wts <- my_wts / knownNormalizationFactor }
    if(computeLB == F){ Qest = sum(Yobs_internal * my_wts )  }
    if(computeLB == T){
      minValue     <- min(Yobs_internal)
      Yobs_nonZero <- Yobs_internal + (abs(minValue) + 1)*(minValue <= 0)
      #Qest <- exp(1/length(Yobs_internal)*sum(log(Yobs_nonZero)+log(my_wts)))
      Qest <- sum(log(Yobs_nonZero)+log(my_wts))
    }
  }
  if(hajek == F){
    if(computeLB == F){ Qest <- mean(Yobs_internal * my_wts )   }
    if(computeLB == T){
      minValue <- min(Yobs_internal)
      Yobs_nonZero <- Yobs_internal + (abs(minValue) + 1)*(minValue <= 0)
      Qest <- mean(log(Yobs_nonZero)+log(my_wts))
    }
  }

  return(list("Qest"=Qest,
              "Q_wts"=my_wts,
              "Yobs"=Yobs_internal,
              "Q_wts_raw_sum" = sum_raw_wts,
              "log_pr_w_new"=log_pr_w_new,
              "log_PrW"=log_pr_w_internal))
}

vec2list_noTransform <- function(vec_){ return( split(vec_,f = splitIndices)) }


computeQse_conjoint <- function(FactorsMat, Yobs,
                                hypotheticalProbList,
                                assignmentProbList,
                                FactorsMat_internal_mapped = NULL,
                                log_pr_w = NULL,
                                hajek = T,
                                knownNormalizationFactor = NULL,
                                knownSigma2 = NULL,
                                hypotheticalN = NULL,
                                returnLog = T,
                                log_treatment_combs=NULL){

  if(is.null(log_treatment_combs)){
    log_treatment_combs  <- sum(log(
      sapply(1:ncol(FactorsMat),function(ze){
        length(assignmentProbList[[ze]]) }) ))
  }

  if(is.null(log_pr_w)){
    log_pr_w <- log(
      sapply(1:ncol(FactorsMat),function(ze){
        (assignmentProbList[[ze]][ FactorsMat[,ze] ]) })
    )
    if(all(class(log_pr_w) == "numeric")){ log_pr_w <- sum(log_pr_w)}
    if(any(class(log_pr_w) != "numeric")){ log_pr_w <- rowSums(log_pr_w)}
  }


  # Perform weighting to obtain bound for E_pi[c_t]
  {
    if(!is.null(FactorsMat_internal_mapped)){FactorsMat_internal_mapped[] <- log(unlist(hypotheticalProbList)[FactorsMat_internal_mapped])}
    if(is.null(FactorsMat_internal_mapped)){FactorsMat_internal_mapped <- log( sapply(1:ncol(FactorsMat),function(ze){hypotheticalProbList[[ze]][ FactorsMat[,ze] ]  })  )}
    if(any(class(FactorsMat_internal_mapped) != "numeric")){ log_pr_w_new <- rowSums(FactorsMat_internal_mapped)}
    if(all(class(FactorsMat_internal_mapped) == "numeric")){ log_pr_w_new <- sum(FactorsMat_internal_mapped)}

    my_wts = exp(log_pr_w_new   - log_pr_w  )
    if(hajek == T){
      if(is.null(knownNormalizationFactor)){  my_wts <- my_wts / sum(my_wts)}
      if(!is.null(knownNormalizationFactor)){  my_wts <- my_wts / knownNormalizationFactor}
      scaleFactor = sum(Yobs^2 * my_wts )
    }
    if(hajek == F){ scaleFactor <- mean(Yobs^2 * my_wts )   }
  }

  # Compute max prob (take maximum prob. of each Multinomial)
  log_maxProb <- sum(log(
    sapply(1:ncol(FactorsMat),function(ze){
      max(hypotheticalProbList[[ze]]) })
  ))

  if(!is.null(knownSigma2)){ sigma2_hat <- knownSigma2 }
  if(is.null(knownSigma2)){ sigma2_hat <- var(Yobs) }

  # Combine terms to get VE and EV
  logN <- ifelse(is.null(hypotheticalN), yes = log(length(Yobs)), no = log(hypotheticalN))
  upperBound_se_VE_log = (log(scaleFactor) + log_treatment_combs + log_maxProb - logN)
  upperBound_se_EV_log = (log(sigma2_hat) + log_treatment_combs + log_maxProb - logN)
  upperBound_se_ <- 0.5*matrixStats::logSumExp(c(upperBound_se_EV_log,upperBound_se_VE_log))#0.5 for sqrt

  # log scale is used in optimization to improve numerical stability
  if(returnLog == F){upperBound_se_ <- exp(upperBound_se_) }
  return( upperBound_se_ )
}

strenv <- new.env( parent = emptyenv() )