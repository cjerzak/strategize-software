#' computeQ_lda
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
#' \item Kosuke Imai, Rohit, Connor
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


computeQ_lda = function( hypotheticalTopicProportion = NULL,
                      n_fold = 3,Yobs,
                      topicProportions,
                      documents_list,
                      wordTopicDistributions,
                      se_ub = sd(Yobs)/10,
                      split1_indices=NULL,
                      split2_indices=NULL,
                      computeThetaSEs = T, findMax = T,
                      nboot = 10,trim_q=1,
                      maxWt = 1e10,maxWt_hajek = NULL,
                      alphaLevel=0.05, openBrowser=F){

  # Initial setup
  {
    nWords = length(unique(unlist(lapply(documents_list,unique))))
    ModalDocLength = density(unlist(lapply(documents_list,length)))
    ModalDocLength = round(ModalDocLength$x[which.max(ModalDocLength$y)],0L)
    zerosVec_ <- unique(unlist(lapply(documents_list,function(ze){unique((ze))})))
    zerosVec <- rep(0,times=length( zerosVec_ )); names(zerosVec) <-zerosVec_
    marginalB <- lapply(documents_list,function(ze){
      my_tab <- prop.table(table( (ze )))
      zerosVec[names(my_tab)] <- my_tab
      return(zerosVec)
    })
    marginalB <- do.call(rbind,marginalB)
    marginalB = apply(marginalB,2,max) + 0.5*apply(marginalB,2,sd)
    marginalB = round(marginalB*ModalDocLength,0L)
    marginalB[is.na(marginalB)] <- 0
    prW_log_tmp <- sapply(1:length(documents_list),function(ze){
        sum(log(colSums(wordTopicDistributions[,documents_list[[ze]]] * topicProportions[,ze])))
      })
    ##https://socialsciences.mcmaster.ca/magee/761_762/other%20material/M-estimation.pdf
    ##m estimation theory
    log_treatCombs_fromEntropy <- - mean(prW_log_tmp) #negative per example log liklihood
    #combinations with replacement, with less over-counting
    {
      N_fxn <- function(t1,t2){ choose(t1 + t2 - 1, t2 - 1) }
      A__ <- 0 #sum of lower bounds
      b_vec <- marginalB;a_vec <- rep(0,times=length(b_vec))
      k__ <- ModalDocLength
      #choose(nWords+ModalDocLength-1,nWords-1)==choose(nWords+ModalDocLength-1,ModalDocLength)
      Component_S_dim0 <- sum( (-1)^0*N_fxn(k__- A__ - 1*0,  nWords))
      Component_S_dim1 <- sum( (-1)^1 * sapply(1:nWords,function(ze){
        N_fxn(k__ - A__ - (1 + b_vec[ze] - a_vec[ze]),  nWords)
      }) )
      PairsGrid <- t(combn(1:nWords,2))
      Component_S_dim2 <- sum( (-1)^2 * apply(PairsGrid,1,function(zee){
        N_fxn(  k__ - A__ -
                  (1+b_vec[zee[1]] - a_vec[zee[1]]) -
                  (1+b_vec[zee[2]] - a_vec[zee[2]]),
                nWords)
      }))
      log_treatCombs_fromCombinatorics <- log( Component_S_dim0 + Component_S_dim1 + Component_S_dim2)
    }
    log_treatCombs <- min(c(log_treatCombs_fromCombinatorics,log_treatCombs_fromEntropy))
  }

  if(!is.null(hypotheticalTopicProportion)){
    if( all( colnames(wordTopicDistributions) !=  1:ncol(wordTopicDistributions))){ #for fast execution
      documents_list = lapply(documents_list,function(ze){ sapply(ze, function(zee){which(colnames(wordTopicDistributions) %in% zee )})})
      colnames(wordTopicDistributions) <- 1:ncol(wordTopicDistributions)
    }
    Qest_raw <- computeQ_lda_internal(theta   = hypotheticalTopicProportion,
                                      Yobs     = Yobs,
                                      pi_mat    = topicProportions,
                                      doc_words = documents_list,
                                      term_mat  = wordTopicDistributions,
                                      smoothWts = smoothWts, trim_q = trim_q,
                                      maxWt = maxWt, maxWt_hajek = maxWt_hajek)
    Q_wts <- Qest_raw$Q_wts_hajek
    wts_raw <- Qest_raw$Q_wts_hajek
    Qest_raw = Qest_raw$Qhat*(1-hajek) + Qest_raw$Qhat_hajek*(hajek)

    SE_obs = exp(computeQse_lda(THETA__ = hypotheticalTopicProportion,
                              INDICES_ = 1:length(Yobs),
                              DOC_INDICES_U = NULL,
                              D_INDICES_U = NULL,
                              TERMS_MAT_INPUT = wordTopicDistributions,
                              PI_MAT_INPUT = topicProportions,
                              DOC_LIST = documents_list,
                              MODAL_DOC_LEN = ModalDocLength,
                              MARGINAL_BOUNDS = marginalB,
                              LOG_PR_W = NULL,
                              LOG_TREATCOMBS=log_treatCombs,
                              YOBS = Yobs,returnLog=T))
    RETURN_LIST <-  list("Q_point"= Qest_raw,
                          "Q_se" = SE_obs,
                          "Q_wts" = Q_wts)
  }

if(is.null(hypotheticalTopicProportion)){
  if(is.null(split1_indices)){
    DENOTE_SPLIT = sample(1:2,length(Yobs),prob=c(1,1),replace=T)
    split1_indices <- which(DENOTE_SPLIT==1)
    split2_indices <- which(DENOTE_SPLIT==2)
  }
  smoothWts = F; hajek = T;tol= 10^(-3)
  boot_max_lower <- boot_max_lower <- boot_max_upper <- optim_max_SEs_mEst <- list()
  Q_interval <- maxVal_est <- boot_max_lower <- boot_max_upper <- boot_max  <- list()

  # Columns coerced to integers for fast execution
  if(all(colnames(wordTopicDistributions) != 1:ncol(wordTopicDistributions))){
    documents_list = lapply(documents_list,function(ze){ sapply(ze, function(zee){which(colnames(wordTopicDistributions) %in% zee )})})
    colnames(wordTopicDistributions) <- 1:ncol(wordTopicDistributions)
  }

  doc_indices_u_split1 = unlist(documents_list[split1_indices],recursive = F)
  L_tmp = documents_list[split1_indices]
  d_indices_u_split1 = unlist(sapply(1:length(L_tmp), function(se){list(rep(se,length(L_tmp[[se]])))}))

  doc_indices_u_split2 = unlist(documents_list[split2_indices],recursive = F)
  L_tmp = documents_list[split2_indices]
  d_indices_u_split2 = unlist(sapply(1:length(L_tmp), function(se){list(rep(se,length(L_tmp[[se]])))})); rm(L_tmp)

  RET_MAT_MIN <- RET_MAT_MAX <- NULL;

  #find mean topic proportion + set up delta method transformation
  {
    MeanTopicProportion <- rowMeans(topicProportions)
    initVec = c(0,rev(c(compositions::alr(t(rev( MeanTopicProportion ) )))))[-1]
    toSimplex_f = function(theta_){toSimplex(c(0,theta_))}
    my_list = sapply(1:(nrow(topicProportions)-1),function(k_){
      sprintf("~exp(x%s)/(exp(0)+%s)", k_, paste( paste(paste("exp(x",1:(nrow(topicProportions)-1),sep=""),")",sep=''),collapse="+"))})
    my_list = sapply(my_list,function(ze){as.formula(ze)})
    entry1_v = 0;  RETURN_FILLER = ""
  }

  # for delta method
  initilizer = function(){return(initVec)}

  log_pr_w <- computeQ_lda_internal(theta  = MeanTopicProportion,
                           Yobs     = Yobs,
                           pi_mat    = topicProportions,
                           doc_words = documents_list,
                           term_mat  = wordTopicDistributions,
                           smoothWts = smoothWts, trim_q = trim_q,maxWt = maxWt, maxWt_hajek = maxWt_hajek)$log_pr_w

  # write out obj fxn
  {
    PEN_VEC = NA
    minThis_max  = function(theta_,INDICES,PEN_VALUE,DOC_INDICES_U, D_INDICES_U){
      theta_ = toSimplex_f(theta_)
      rez_ = computeQ_lda_internal(theta  = theta_,
                          Yobs     = Yobs[INDICES],
                          pi_mat    = topicProportions[,INDICES],
                          doc_words = documents_list[INDICES],
                          log_pr_w = log_pr_w[INDICES],
                          term_mat  = wordTopicDistributions,
                          doc_indices_u = DOC_INDICES_U,
                          d_indices_u = D_INDICES_U,
                          smoothWts = smoothWts, trim_q = trim_q,maxWt = maxWt, maxWt_hajek = maxWt_hajek)
      return( -rez_$Qhat_hajek ) }
    if(!findMax){
      minThis_max =  function(theta_,INDICES,PEN_VALUE,DOC_INDICES_U, D_INDICES_U){
        theta_ = toSimplex_f(theta_)
        rez_ = computeQ_lda_internal(theta   = theta_,
                            Yobs     = Yobs[INDICES],
                            pi_mat    = topicProportions[,INDICES],
                            doc_words = documents_list[INDICES],
                            log_pr_w = log_pr_w[INDICES],
                            term_mat  = wordTopicDistributions,
                            smoothWts = smoothWts,
                            doc_indices_u = DOC_INDICES_U,
                            d_indices_u = D_INDICES_U,
                            trim_q = trim_q,maxWt = maxWt, maxWt_hajek = maxWt_hajek)
        return(  rez_$Qhat_hajek )
      }
    }
  }

  # Variance bound setup
  {
    clip2=function(ze){ze}

    if(openBrowser){browser()}
    myRho <- NULL;logSE_LB <- -Inf;logSE_UB = log(se_ub)#log(sd(Yobs)* (1/length(split2_indices)^0.25))
    initVec_empiricalMean <- initVec;
    initVec_flat <- initVec;initVec_flat[] <- 0
    logSE_meanPi <- computeQse_lda(THETA__ = toSimplex_f(initVec_empiricalMean),
                                 INDICES_ = split1_indices,
                                 DOC_INDICES_U = doc_indices_u_split1,
                                 D_INDICES_U = d_indices_u_split1,
                                 TERMS_MAT_INPUT = wordTopicDistributions,
                                 PI_MAT_INPUT = topicProportions,
                                 DOC_LIST = documents_list,
                                 MODAL_DOC_LEN = ModalDocLength,
                                 MARGINAL_BOUNDS = marginalB,
                                 LOG_PR_W = log_pr_w,
                                 LOG_TREATCOMBS=log_treatCombs,
                                 YOBS = Yobs,returnLog=T)
    logSE_flatPi <- computeQse_lda(THETA__ = toSimplex_f(initVec_flat),
                                 INDICES_ = split1_indices,
                                 DOC_INDICES_U = doc_indices_u_split1,
                                 D_INDICES_U = d_indices_u_split1,
                                 TERMS_MAT_INPUT = wordTopicDistributions,
                                 PI_MAT_INPUT = topicProportions,
                                 DOC_LIST = documents_list,
                                 MODAL_DOC_LEN = ModalDocLength,
                                 MARGINAL_BOUNDS = marginalB,
                                 LOG_TREATCOMBS=log_treatCombs,
                                 LOG_PR_W = log_pr_w,
                                 YOBS = Yobs,returnLog=T)
    if(logSE_flatPi < logSE_UB){initVec <- initVec_flat}
    if(logSE_meanPi < logSE_UB){initVec <- initVec_empiricalMean}
    if(logSE_meanPi > logSE_UB & logSE_flatPi > logSE_UB){stop("LDA model not regularized enough\n
                                                           Initial values violate SE bound!")}
    #initVec[] <- 0;
    my_ep = 0.005#
    LB_VEC <- c(logSE_LB, rep(my_ep,times = 1+length(initVec)))
    UB_VEC <- c(logSE_UB, rep(1-my_ep,times = 1+length(initVec)))

    # COBYLA algorithm
    {
      nloptr_sol <- optim_max_raw <- ((rsolnp_results <- nloptr::nloptr(x0 = initVec ,
                                                                        eval_f =  function(ze){
                                                                          my_value = minThis_max(clip2(ze),
                                                                                                 INDICES = c(split1_indices),
                                                                                                 DOC_INDICES_U = doc_indices_u_split1,
                                                                                                 D_INDICES_U = d_indices_u_split1, PEN_VAL = 0)
                                                                          return(my_value)},
                                                                        #opts = list("algorithm"="NLOPT_LN_AUGLAG","ftol_abs" = 0.0001,"local_opts"=list("algorithm"="NLOPT_LN_COBYLA")),
                                                                        opts = list("algorithm"="NLOPT_LN_COBYLA"),
                                                                        lb = rep(-2,length(initVec)),
                                                                        ub = rep(2,length(initVec)),
                                                                        eval_g_ineq = function(theta_){
                                                                          theta__ = toSimplex_f(theta_)
                                                                          upperBound_variance_log <- computeQse_lda(THETA__ = theta__,
                                                                                                                    INDICES_ = split1_indices,
                                                                                                                    DOC_INDICES_U = doc_indices_u_split1,
                                                                                                                    D_INDICES_U = d_indices_u_split1,
                                                                                                                    TERMS_MAT_INPUT = wordTopicDistributions,
                                                                                                                    PI_MAT_INPUT = topicProportions,
                                                                                                                    DOC_LIST = documents_list,
                                                                                                                    MODAL_DOC_LEN = ModalDocLength,
                                                                                                                    MARGINAL_BOUNDS = marginalB,
                                                                                                                    LOG_TREATCOMBS=log_treatCombs,
                                                                                                                    LOG_PR_W = log_pr_w,
                                                                                                                    YOBS = Yobs,returnLog=T)
                                                                          constrainThis_ <- c(upperBound_variance_log, theta__)
                                                                          lessThan0_vec <- c(LB_VEC - constrainThis_,
                                                                                             constrainThis_ - UB_VEC) #<= 0
                                                                          lessThan0_vec = lessThan0_vec[!is.infinite(lessThan0_vec)]
                                                                          return( lessThan0_vec  )  } )))
      rsolnp_results$pars <- rsolnp_results$solution
      optim_max_raw <- rsolnp_results$pars
      rsolnp_results$convergence <- 0
      initVec <- rsolnp_results$solution
    }

    # augmented lagrangian optimization
    {
      optim_max_raw <- clip2((rsolnp_results <- Rsolnp::solnp(pars = initVec ,
                                                              fun =  function(ze){
                                                                my_value = minThis_max(clip2(ze),
                                                                                       INDICES = c(split1_indices),
                                                                                       DOC_INDICES_U = doc_indices_u_split1,#doc_indices_u_split1,
                                                                                       D_INDICES_U = d_indices_u_split1,
                                                                                       PEN_VAL = 0)
                                                              },
                                                              control=list(rho=myRho,tol=tol,trace = 1),
                                                              ineqfun = function(theta_){
                                                                theta__ = toSimplex_f(theta_)
                                                                upperBound_variance_log <- computeQse_lda(THETA__ = theta__,
                                                                                                          INDICES_ = split1_indices,
                                                                                                          DOC_INDICES_U = doc_indices_u_split1,
                                                                                                          D_INDICES_U = d_indices_u_split1,
                                                                                                          TERMS_MAT_INPUT = wordTopicDistributions,
                                                                                                          PI_MAT_INPUT = topicProportions,
                                                                                                          DOC_LIST = documents_list,
                                                                                                          MODAL_DOC_LEN = ModalDocLength,
                                                                                                          MARGINAL_BOUNDS = marginalB,
                                                                                                          LOG_PR_W = log_pr_w,
                                                                                                          LOG_TREATCOMBS=log_treatCombs,
                                                                                                          YOBS = Yobs,returnLog=T)
                                                                return( c(upperBound_variance_log, theta__)  )  },
                                                              ineqLB = LB_VEC,  ineqUB = UB_VEC))$pars)
    }
    mySE = exp(computeQse_lda(THETA__ = toSimplex_f(rsolnp_results$pars),
                              INDICES_ = split1_indices,
                              DOC_INDICES_U = doc_indices_u_split1,
                              D_INDICES_U = d_indices_u_split1,
                              TERMS_MAT_INPUT = wordTopicDistributions,
                              PI_MAT_INPUT = topicProportions,
                              DOC_LIST = documents_list,
                              MODAL_DOC_LEN = ModalDocLength,
                              MARGINAL_BOUNDS = marginalB,
                              LOG_PR_W = log_pr_w,
                              LOG_TREATCOMBS=log_treatCombs,
                              YOBS = Yobs,returnLog=T))
    if(rsolnp_results$convergence != 0){warning(sprintf("Convergence not established at tol = %.6f",tol))}
    optim_max <- toSimplex_f(optim_max_raw)

    print(sprintf("InSamp Value: %.3f, OutSamp Value: %.3f",
                  minThis_max(clip2(optim_max_raw), INDICES = c(split1_indices),
                              DOC_INDICES_U = doc_indices_u_split1,
                              D_INDICES_U = d_indices_u_split1, PEN_VAL = 0),
                  minThis_max(clip2(optim_max_raw), INDICES = c(split2_indices),
                              DOC_INDICES_U = doc_indices_u_split2,
                              D_INDICES_U = d_indices_u_split2, PEN_VAL = 0)))
    optim_max_raw = list("optim_max_raw"=optim_max_raw,"par"=optim_max_raw)
    }

  # asymptotic se's
  if(computeThetaSEs == T){
    library(geex); ex_eeFUN_max <- function(data){
      function(theta){ with(data, {
        DATA_SPLIT_USE = split1_indices
        my_grad = ((c(rootSolve::gradient(f = function(x){
          theta__ <- toSimplex_f(x)
          logSE__ <- computeQse_lda(THETA__ = theta__,
                                    INDICES_ = split1_indices,
                                    DOC_INDICES_U = doc_indices_u_split1,
                                    D_INDICES_U = d_indices_u_split1,
                                    TERMS_MAT_INPUT = wordTopicDistributions,
                                    PI_MAT_INPUT = topicProportions,
                                    DOC_LIST = documents_list,
                                    LOG_PR_W = log_pr_w,
                                    MODAL_DOC_LEN = ModalDocLength,
                                    MARGINAL_BOUNDS = marginalB,
                                    LOG_TREATCOMBS=log_treatCombs,
                                    YOBS = Yobs,returnLog=T)
          constrainThese_ <- c(logSE__, theta__)
          tmp_mat = cbind(LB_VEC - constrainThese_, constrainThese_ - UB_VEC )
          LowerUpperCloserToViolated = apply(tmp_mat,1,which.max)
          LagrangianTerm <- c(rsolnp_results$lagrange)*
            sapply(1:length(LowerUpperCloserToViolated),function(ze){ tmp_mat[ze,LowerUpperCloserToViolated[ze]] })
          mainContrib <- minThis_max(x,
                                     INDICES = DATA_SPLIT_USE,
                                     DOC_INDICES_U = doc_indices_u_split1,
                                     D_INDICES_U = d_indices_u_split1,
                                     PEN_VAL)
          TOTAL_OBJ = mainContrib  +  sum(LagrangianTerm)
          return( TOTAL_OBJ )   }, x = theta)))) ;
        my_grad } ) }}
    mEst_max <- m_estimate(
      estFUN = ex_eeFUN_max, data  = data.frame(),
      compute_roots = F, roots = optim_max_raw$par)
    m_mean_max = attributes(mEst_max)$estimates
    optim_max_mEst <- toSimplex(c(entry1_v,m_mean_max))
    m_cov_max  = attributes(mEst_max)$vcov

    optim_max_SEs_mEst = msm::deltamethod(my_list,
                                          mean = m_mean_max,
                                          cov  = m_cov_max) #cov is inverse of ?negative? hessian
  }

  Qest_raw = computeQ_lda_internal(theta   = optim_max,
                          Yobs     = Yobs[split1_indices],
                          pi_mat    = topicProportions[,split1_indices],
                          doc_words = documents_list[split1_indices],
                          term_mat  = wordTopicDistributions,
                          smoothWts = smoothWts, trim_q = trim_q,
                          maxWt = maxWt, maxWt_hajek = maxWt_hajek)
  names(Qest_raw$Q_wts_hajek) <- split1_indices
  wts_raw <- Qest_raw$Q_wts_hajek
  Qest_raw = Qest_raw$Qhat*(1-hajek) + Qest_raw$Qhat_hajek*(hajek)

  Qest_split = computeQ_lda_internal(theta   = optim_max,
                            Yobs     = Yobs[split2_indices],
                            pi_mat    = topicProportions[,split2_indices],
                            doc_words = documents_list[split2_indices],
                            term_mat  = wordTopicDistributions,
                            smoothWts = smoothWts, trim_q = trim_q,
                            maxWt = maxWt, maxWt_hajek = maxWt_hajek)
  names(Qest_split$Q_wts_hajek) <- split2_indices
  hist(Qest_split$Q_wts_hajek,main = "Histogram of Weights")
  wts_split <- Qest_split$Q_wts_hajek
  Qest_split = Qest_split$Qhat*(1-hajek) + Qest_split$Qhat_hajek*(hajek)

  splitForSEObs <- "1"
  SE_obs = exp(computeQse_lda(THETA__ = optim_max,
                              INDICES_ = eval(parse(text=sprintf("split%s_indices",splitForSEObs))),
                              DOC_INDICES_U = eval(parse(text=sprintf("doc_indices_u_split%s",splitForSEObs))),
                              D_INDICES_U = eval(parse(text=sprintf("d_indices_u_split%s",splitForSEObs))),
                              TERMS_MAT_INPUT = wordTopicDistributions,
                              PI_MAT_INPUT = topicProportions,
                              DOC_LIST = documents_list,
                              MODAL_DOC_LEN = ModalDocLength,
                              MARGINAL_BOUNDS = marginalB,
                              LOG_PR_W = log_pr_w,
                              LOG_TREATCOMBS=log_treatCombs,
                              YOBS = Yobs,returnLog=T))

  splitForSEObs <- "1"
  SE_split = exp(computeQse_lda(THETA__ = optim_max,
                                INDICES_ = eval(parse(text=sprintf("split%s_indices",splitForSEObs))),
                                DOC_INDICES_U = eval(parse(text=sprintf("doc_indices_u_split%s",splitForSEObs))),
                                D_INDICES_U = eval(parse(text=sprintf("d_indices_u_split%s",splitForSEObs))),
                                TERMS_MAT_INPUT = wordTopicDistributions,
                                PI_MAT_INPUT = topicProportions,
                                DOC_LIST = documents_list,
                                MODAL_DOC_LEN = ModalDocLength,
                                MARGINAL_BOUNDS = marginalB,
                                LOG_TREATCOMBS=log_treatCombs,
                                LOG_PR_W = log_pr_w,
                                YOBS = Yobs,returnLog=T))
  Q_interval <- c(Qest_raw - 1.64*SE_obs,Qest_raw + 1.64*SE_obs)
  Q_interval_split <- c(Qest_split - 1.64*SE_split,Qest_split + 1.64*SE_split)

  names(optim_max_SEs_mEst) <- NULL
  RETURN_LIST = list("ThetaStar"=c(optim_max),
                            "ThetaStar_se" = c(NA,optim_max_SEs_mEst),
                            "Q_point"=Qest_raw,
                            "Q_se" = SE_obs,
                            "Q_interval" = Q_interval,
                            "Q_point_split"=Qest_split,
                            "Q_se_split" = SE_split,
                            "Q_interval_split" = Q_interval_split,
                            "Q_wts_raw"=wts_raw,
                            "Q_wts_split"=wts_split)
}
  return(RETURN_LIST)
}

