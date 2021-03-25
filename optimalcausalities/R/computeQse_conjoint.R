computeQse_conjoint <- function(FactorsMat, Yobs,
                                hypotheticalProbList,
                                assignmentProbList,
                                log_pr_w = NULL,
                                hajek = T,
                                returnLog = T,
                                log_treatment_combs=NULL){

  if(is.null(log_treatment_combs)){
  log_treatment_combs  <- sum(log(
      sapply(1:ncol(FactorsMat),function(ze){
        length(assignmentProbList[[ze]]) }) ))
  }

  if(is.null(log_pr_w)){
    log_pr_w = rowSums(log(
      sapply(1:ncol(FactorsMat),function(ze){
        (assignmentProbList[[ze]][ FactorsMat[,ze] ]) })
    ))#fast match here
  }


  # Perform weighting to obtain bound for E_theta[c_t]
  {
    log_pr_new = rowSums(log(
      sapply(1:ncol(FactorsMat),function(ze){
      (hypotheticalProbList[[ze]][ FactorsMat[,ze] ] ) })
    ))

    my_wts = exp(log_pr_new   - log_pr_w  )
    if(hajek == T){ my_wts <- my_wts / sum(my_wts);scaleFactor = sum(Yobs^2 * my_wts )   }
    if(hajek == F){ scaleFactor <- mean(Yobs^2 * my_wts )   }
  }

  # Compute max prob (take maximum prob. of each Multinomial)
  log_maxProb <- sum(log(
    sapply(1:ncol(FactorsMat),function(ze){
      max(hypotheticalProbList[[ze]]) })
  ))

  # Combine terms to get VE and EV
  upperBound_se_VE_log = (log(scaleFactor) + log_treatment_combs + log_maxProb - log(length(Yobs)))
  upperBound_se_EV_log = (log(var(Yobs)) + log_treatment_combs + log_maxProb - log(length(Yobs)))
  upperBound_se_ <- 0.5*matrixStats::logSumExp(c(upperBound_se_EV_log,upperBound_se_VE_log))#0.5 for sqrt

  # log scale is used in optimization to improve numerical stability
  if(returnLog == F){upperBound_se_ <- exp(upperBound_se_) }
  return( upperBound_se_ )
}
