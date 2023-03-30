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
    #if(any(class(log_pr_w) != "numeric")){ log_pr_w <- rowsums(log_pr_w)}
    if(any(class(log_pr_w) != "numeric")){ log_pr_w <- rowSums(log_pr_w)}
  }


  # Perform weighting to obtain bound for E_pi[c_t]
  {
    if(!is.null(FactorsMat_internal_mapped)){FactorsMat_internal_mapped[] <- log(unlist(hypotheticalProbList)[FactorsMat_internal_mapped])}
    if(is.null(FactorsMat_internal_mapped)){FactorsMat_internal_mapped <- log( sapply(1:ncol(FactorsMat),function(ze){hypotheticalProbList[[ze]][ FactorsMat[,ze] ]  })  )}
    #if(any(class(FactorsMat_internal_mapped) != "numeric")){ log_pr_w_new <- rowsums(FactorsMat_internal_mapped)}
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
  logN <- log(length(Yobs))
  if(!is.null(hypotheticalN)){logN <- log(hypotheticalN)}
  upperBound_se_VE_log = (log(scaleFactor) + log_treatment_combs + log_maxProb - logN)
  upperBound_se_EV_log = (log(sigma2_hat) + log_treatment_combs + log_maxProb - logN)
  upperBound_se_ <- 0.5*matrixStats::logSumExp(c(upperBound_se_EV_log,upperBound_se_VE_log))#0.5 for sqrt

  # log scale is used in optimization to improve numerical stability
  if(returnLog == F){upperBound_se_ <- exp(upperBound_se_) }
  return( upperBound_se_ )
}
