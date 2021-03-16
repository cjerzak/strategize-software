#' computeQse_lda
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

computeQse_lda = function(THETA__,INDICES_, DOC_INDICES_U, D_INDICES_U,
                            PI_MAT_INPUT,MARGINAL_BOUNDS,DOC_LIST,
                            MODAL_DOC_LEN,
                            TERMS_MAT_INPUT,LOG_TREATCOMBS,YOBS,
                            returnLog = T,LOG_PR_W=NULL){
    logW = log(apply( term_mat * c(THETA__),2,sum))
    logW = logW[names(MARGINAL_BOUNDS)];
    logW_counter = logW; logW_counter[] <- 0
    my_tmp = rep(NA,times=MODAL_DOC_LEN);log_maxProb <- 0;for(fa in 1:MODAL_DOC_LEN){
      which_max = which.max(logW[logW_counter < MARGINAL_BOUNDS])
      my_tmp[fa] <- names(which_max)
      log_maxProb <- log_maxProb + logW[names(which_max)]
      logW_counter[names(which_max)] <- logW_counter[names(which_max)] + 1
    }
    #table(my_tmp); MARGINAL_BOUNDS[names(table(my_tmp))]
    THETA_MAT_ = PI_MAT_INPUT; THETA_MAT_[] <- THETA__
    NUM__ = PrWGivenPi_fxn(doc_indices     = DOC_LIST[INDICES_],
                           pi_mat = THETA_MAT_[,INDICES_],
                           terms_posterior = TERMS_MAT_INPUT,
                           doc_indices_u = DOC_INDICES_U,
                           d_indices_u = D_INDICES_U, log_=T)
    if(is.null(LOG_PR_W)){
      LOG_PR_W <- PrWGivenPi_fxn(doc_indices     = DOC_LIST[INDICES_],
                             pi_mat = PI_MAT_INPUT[,INDICES_],
                             terms_posterior = TERMS_MAT_INPUT,
                             doc_indices_u = DOC_INDICES_U,
                             d_indices_u = D_INDICES_U, log_=T)
    }
    MY_WTS__ = prop.table(exp(NUM__ - LOG_PR_W  ))
    scaleFactor = sum(YOBS[INDICES_]^2 * MY_WTS__)
    upperBound_se_VE_log = (log(scaleFactor) + LOG_TREATCOMBS + log_maxProb - log(length(INDICES_)))
    upperBound_se_EV_log = (var(YOBS[INDICES_]) + LOG_TREATCOMBS + log_maxProb - log(length(INDICES_)))
    upperBound_se_ <- 0.5*matrixStats::logSumExp(c(upperBound_se_EV_log,upperBound_se_VE_log))
    if(returnLog == F){upperBound_se_ <- exp(upperBound_se_) }
    return( upperBound_se_ )
  }

