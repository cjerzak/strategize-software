#' computeQse_conjoint
#'
#' Implements ...
#'
#' @param dfm 'document-feature matrix'. A list ...

#' @return A list consiting of \itemize{
#'   \item Items.
#' }
#'
#' @section References:
#' \itemize{
#' \item Kosuke, Rohit, Connor.  Working Paper.
#' }
#'
#' @examples
#' #set seed
#' set.seed(1)
#'
#' #Geneate data
#' x <- rnorm(100)
#'
#' @export


computeQse_conjoint <- function(FactorsMat, Yobs,
                                hypotheticalProbList,
                                assignmentProbList,
                                log_pr_w = NULL,
                                hajek = T,
                                returnLog = T,
                                log_treatment_combs){
  low_maxProb <- sum(log(
    sapply(1:ncol(FactorsMat),function(ze){
      max(assignmentProbList[[ze]]) })
  ))

  if(is.null(log_pr_w)){
    log_pr_w = rowSums(log(
      sapply(1:ncol(FactorsMat),function(ze){
        (assignmentProbList[[ze]][ FactorsMat[,ze] ]) })
    ))#fast match here
  }
  my_vec <- FactorsMat[,ze]
  my_vec[] <- 1
  my_vec <- as.numeric(my_vec)
  log_pr_new = rowSums(log(
    sapply(1:ncol(FactorsMat),function(ze){
    (hypotheticalProbList[[ze]][ FactorsMat[,ze] ] ) })
  ))

  my_wts = exp(log_pr_new   - log_pr_w  )
  if(hajek == T){ my_wts <- my_wts / sum(my_wts);scaleFactor = sum(Yobs^2 * my_wts )   }
  if(hajek == F){ scaleFactor <- mean(Yobs^2 * my_wts )   }

  upperBound_se_VE_log = (log(scaleFactor) + log_treatment_combs + low_maxProb - log(length(Yobs)))
  upperBound_se_EV_log = (log(var(Yobs)) + log_treatment_combs + low_maxProb - log(length(Yobs)))
  upperBound_se_ <- 0.5*matrixStats::logSumExp(c(upperBound_se_EV_log,upperBound_se_VE_log))#0.5 for sqrt

  if(returnLog == F){upperBound_se_ <- exp(upperBound_se_) }
  return( upperBound_se_ )
}
