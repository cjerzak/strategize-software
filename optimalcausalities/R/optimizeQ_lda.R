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


optimizeQ_lda = function(DATA_SPLIT1,DATA_SPLIT2=NULL,  DTM_MAT,
                      n_fold = 3,YOBS, PI_MAT,DOC_LIST,TERMS_MAT, SE_UB = sd(YOBS)/10,
                      nboot = 10,trim_q=1,
                      maxWt = 1e10,maxWt_hajek = NULL,
                      computeSEs = T, doMax = T,alphaLevel=0.05, openBrowser=F){
  if(is.null(DATA_SPLIT2)){DATA_SPLIT2 <- DATA_SPLIT1}
  smoothWts = F; hajek = T;tol= 10^(-3)
  boot_max_lower <- boot_max_lower <- boot_max_upper <- optim_max_SEs_mEst <- list()
  Q_interval <- maxVal_est <- boot_max_lower <- boot_max_upper <-cv_average_vec <- boot_max <- cv_path <- list()

  # Columns coerced to integers for fast execution
  if(all(colnames(TERMS_MAT) != 1:ncol(TERMS_MAT))){
    DOC_LIST = lapply(DOC_LIST,function(ze){ sapply(ze, function(zee){which(colnames(TERMS_MAT) %in% zee )})})
    colnames(TERMS_MAT) <- 1:ncol(TERMS_MAT)
  }

  doc_indices_u_split1 = unlist(DOC_LIST[DATA_SPLIT1],recursive = F)
  L_tmp = DOC_LIST[DATA_SPLIT1]
  d_indices_u_split1 = unlist(sapply(1:length(L_tmp), function(se){list(rep(se,length(L_tmp[[se]])))}))

  doc_indices_u_split2 = unlist(DOC_LIST[DATA_SPLIT2],recursive = F)
  L_tmp = DOC_LIST[DATA_SPLIT2]
  d_indices_u_split2 = unlist(sapply(1:length(L_tmp), function(se){list(rep(se,length(L_tmp[[se]])))})); rm(L_tmp)

  RET_MAT_MIN <- RET_MAT_MAX <- NULL;

  #find mean topic proportion + set up delta method transformation
  {
    MeanTopicProportion <- rowMeans(PI_MAT)
    initVec = c(0,rev(c(compositions::alr(t(rev( MeanTopicProportion ) )))))[-1]
    toSimplex_f = function(theta_){toSimplex(c(0,theta_))}
    my_list = sapply(1:(nrow(PI_MAT)-1),function(k_){
      sprintf("~exp(x%s)/(exp(0)+%s)", k_, paste( paste(paste("exp(x",1:(nrow(PI_MAT)-1),sep=""),")",sep=''),collapse="+"))})
    my_list = sapply(my_list,function(ze){as.formula(ze)})
    entry1_v = 0;  RETURN_FILLER = ""
  }

  # for delta method
  initilizer = function(){return(initVec)}

  log_pr_w <- computeQ_lda(theta  = MeanTopicProportion,
                           Yobs     = YOBS,
                           pi_mat    = PI_MAT,
                           doc_words = DOC_LIST,
                           term_mat  = TERMS_MAT,
                           smoothWts = smoothWts, trim_q = trim_q,maxWt = maxWt, maxWt_hajek = maxWt_hajek)$log_pr_w

  # write out obj fxn
  {
    PEN_VEC = NA
    minThis_max  = function(theta_,INDICES,PEN_VALUE,DOC_INDICES_U, D_INDICES_U){
      theta_ = toSimplex_f(theta_)
      rez_ = computeQ_lda(theta  = theta_,
                          Yobs     = YOBS[INDICES],
                          pi_mat    = PI_MAT[,INDICES],
                          doc_words = DOC_LIST[INDICES],
                          log_pr_w = log_pr_w[INDICES],
                          term_mat  = TERMS_MAT,
                          doc_indices_u = DOC_INDICES_U,
                          d_indices_u = D_INDICES_U,
                          smoothWts = smoothWts, trim_q = trim_q,maxWt = maxWt, maxWt_hajek = maxWt_hajek)
      return( -rez_$Qhat_hajek ) }
    if(!doMax){
      minThis_max =  function(theta_,INDICES,PEN_VALUE,DOC_INDICES_U, D_INDICES_U){
        theta_ = toSimplex_f(theta_)
        rez_ = computeQ_lda(theta   = theta_,
                            Yobs     = YOBS[INDICES],
                            pi_mat    = PI_MAT[,INDICES],
                            doc_words = DOC_LIST[INDICES],
                            log_pr_w = log_pr_w[INDICES],
                            term_mat  = TERMS_MAT,
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
    #clip2=function(ze){ze[ze< -4] <- -4; ze[ze > 4] <- 4; ze}
    nWords = length(unique(unlist(lapply(DOC_LIST,unique))))
    ModalDocLength = density(unlist(lapply(DOC_LIST,length)))
    ModalDocLength = round(ModalDocLength$x[which.max(ModalDocLength$y)],0L)
    usedWords = unique(unlist(lapply(DOC_LIST,function(ze){names(ze[!duplicated(ze)])})))
    marginalB = t(apply(DTM_MAT[rowSums(DTM_MAT)>0,usedWords],1,function(ze){ze/sum(ze)}))
    marginalB = apply(marginalB,2,max) #+ 0.1*apply(marginalB,2,sd)
    marginalB = round(marginalB*ModalDocLength,0L)
    marginalB[is.na(marginalB)] <- 0
    prW_log_tmp = PrWGivenPi_fxn(doc_indices = DOC_LIST, pi_mat = PI_MAT, terms_posterior = TERMS_MAT, log_=T)
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

    myRho <- NULL;logSE_LB <- -Inf;logSE_UB = log(SE_UB)#log(sd(Yobs)* (1/length(DATA_SPLIT1)^0.25))
    initVec_empiricalMean <- initVec;
    initVec_flat <- initVec;initVec_flat[] <- 0
    logSE_meanPi <- computeQse_lda(THETA__ = toSimplex_f(initVec_empiricalMean),
                                 INDICES_ = DATA_SPLIT1,
                                 DOC_INDICES_U = doc_indices_u_split1,
                                 D_INDICES_U = d_indices_u_split1,
                                 TERMS_MAT_INPUT = TERMS_MAT,
                                 PI_MAT_INPUT = PI_MAT,
                                 DOC_LIST = DOC_LIST,
                                 MODAL_DOC_LEN = ModalDocLength,
                                 MARGINAL_BOUNDS = marginalB,
                                 LOG_TREATCOMBS=log_treatCombs,
                                 YOBS = Yobs,log =T)
    logSE_flatPi <- computeQse_lda(THETA__ = toSimplex_f(initVec_flat),
                                 INDICES_ = DATA_SPLIT1,
                                 DOC_INDICES_U = doc_indices_u_split1,
                                 D_INDICES_U = d_indices_u_split1,
                                 TERMS_MAT_INPUT = TERMS_MAT,
                                 PI_MAT_INPUT = PI_MAT,
                                 DOC_LIST = DOC_LIST,
                                 MODAL_DOC_LEN = ModalDocLength,
                                 MARGINAL_BOUNDS = marginalB,
                                 LOG_TREATCOMBS=log_treatCombs,
                                 YOBS = Yobs,log=T)
    if(logSE_flatPi < logSE_UB){initVec <- initVec_flat}
    if(logSE_meanPi < logSE_UB){initVec <- initVec_empiricalMean}
    if(logSE_meanPi > logSE_UB & logSE_flatPi > logSE_UB){stop("LDA model not regularized enough\n
                                                           Initial values violate SE bound!")}
    #initVec[] <- 0;
    my_ep = 0.005#
    LB_VEC <- c(logSE_LB, rep(my_ep,times = 1+length(initVec)))
    UB_VEC <- c(logSE_UB, rep(1-my_ep,times = 1+length(initVec)))
    if(openBrowser){browser()}
    if(T == T){
      print("Doing warm start")
      nloptr_sol <- optim_max_raw <- ((rsolnp_results <- nloptr::nloptr(x0 = initVec ,
                                                                        eval_f =  function(ze){
                                                                          my_value = minThis_max(clip2(ze),
                                                                                                 INDICES = c(DATA_SPLIT1),
                                                                                                 DOC_INDICES_U = doc_indices_u_split1,
                                                                                                 D_INDICES_U = d_indices_u_split1, PEN_VAL = 0)
                                                                          return(my_value)},
                                                                        #opts = list("algorithm"="NLOPT_LD_SLSQP", "ftol_abs" = 0.001),
                                                                        opts = list("algorithm"="NLOPT_LN_AUGLAG","ftol_abs" = 0.0001,
                                                                                    "local_opts"=list("algorithm"="NLOPT_LN_COBYLA")),
                                                                        #opts = list("algorithm"="NLOPT_LN_COBYLA",
                                                                        lb = rep(-2,length(initVec)),
                                                                        ub = rep(2,length(initVec)),
                                                                        eval_g_ineq = function(theta_){
                                                                          theta__ = toSimplex_f(theta_)
                                                                          upperBound_variance_log <- computeQse_lda(THETA__ = theta__,
                                                                                                                    INDICES_ = DATA_SPLIT1,
                                                                                                                    DOC_INDICES_U = doc_indices_u_split1,
                                                                                                                    D_INDICES_U = d_indices_u_split1,
                                                                                                                    TERMS_MAT_INPUT = TERMS_MAT,
                                                                                                                    PI_MAT_INPUT = PI_MAT,
                                                                                                                    DOC_LIST = DOC_LIST,
                                                                                                                    MODAL_DOC_LEN = ModalDocLength,
                                                                                                                    MARGINAL_BOUNDS = marginalB,
                                                                                                                    LOG_TREATCOMBS=log_treatCombs,
                                                                                                                    YOBS = Yobs,log=T)
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
      print( initVec )
      optim_max_raw <- clip2((rsolnp_results <- Rsolnp::solnp(pars = initVec ,
                                                              fun =  function(ze){
                                                                my_value = minThis_max(clip2(ze),
                                                                                       INDICES = c(DATA_SPLIT1),
                                                                                       DOC_INDICES_U = doc_indices_u_split1,#doc_indices_u_split1,
                                                                                       D_INDICES_U = d_indices_u_split1,
                                                                                       PEN_VAL = 0)
                                                                if(T == F){ # CHECK WORK
                                                                  my_value3 = minThis_max(clip2(ze),
                                                                                          INDICES = c(DATA_SPLIT2),
                                                                                          DOC_INDICES_U = doc_indices_u_split2,#doc_indices_u_split1,
                                                                                          D_INDICES_U = d_indices_u_split2,
                                                                                          PEN_VAL = 0); print( c(my_value,my_value3) )
                                                                }
                                                                my_value
                                                              },
                                                              control=list(rho=myRho,tol=tol,trace = 1),
                                                              ineqfun = function(theta_){
                                                                theta__ = toSimplex_f(theta_)
                                                                upperBound_variance_log <- computeQse_lda(THETA__ = theta__,
                                                                                                          INDICES_ = DATA_SPLIT1,
                                                                                                          DOC_INDICES_U = doc_indices_u_split1,
                                                                                                          D_INDICES_U = d_indices_u_split1,
                                                                                                          TERMS_MAT_INPUT = TERMS_MAT,
                                                                                                          PI_MAT_INPUT = PI_MAT,
                                                                                                          DOC_LIST = DOC_LIST,
                                                                                                          MODAL_DOC_LEN = ModalDocLength,
                                                                                                          MARGINAL_BOUNDS = marginalB,
                                                                                                          LOG_TREATCOMBS=log_treatCombs,
                                                                                                          YOBS = Yobs,log=T)
                                                                return( c(upperBound_variance_log, theta__)  )  },
                                                              ineqLB = LB_VEC,  ineqUB = UB_VEC))$pars)
    }
    mySE = exp(computeQse_lda(THETA__ = toSimplex_f(rsolnp_results$pars),
                              INDICES_ = DATA_SPLIT1,
                              DOC_INDICES_U = doc_indices_u_split1,
                              D_INDICES_U = d_indices_u_split1,
                              TERMS_MAT_INPUT = TERMS_MAT,
                              PI_MAT_INPUT = PI_MAT,
                              DOC_LIST = DOC_LIST,
                              MODAL_DOC_LEN = ModalDocLength,
                              MARGINAL_BOUNDS = marginalB,
                              LOG_TREATCOMBS=log_treatCombs,
                              YOBS = Yobs,log=T))
    if(mySE > exp(logSE_UB)){ stop("LDA Regularization Not Strong Enough, \n Variance Bound Not Satisfied at Initialization") }
    if(rsolnp_results$convergence != 0){warning(sprintf("Convergence not established at tol = %.6f",tol))}
    optim_max <- toSimplex_f(optim_max_raw)

    print(sprintf("InSamp Value: %.3f, OutSamp Value: %.3f",
                  minThis_max(clip2(optim_max_raw), INDICES = c(DATA_SPLIT1),
                              DOC_INDICES_U = doc_indices_u_split1,
                              D_INDICES_U = d_indices_u_split1, PEN_VAL = 0),
                  minThis_max(clip2(optim_max_raw), INDICES = c(DATA_SPLIT2),
                              DOC_INDICES_U = doc_indices_u_split2,
                              D_INDICES_U = d_indices_u_split2, PEN_VAL = 0)))
    optim_max_raw = list("optim_max_raw"=optim_max_raw,"par"=optim_max_raw)
    }

  # asymptotic se's
  if(computeSEs == T){
    library(geex); ex_eeFUN_max <- function(data){
      function(theta){ with(data, {
        DATA_SPLIT_USE = DATA_SPLIT1
        my_grad = ((c(rootSolve::gradient(f = function(x){
          theta__ <- toSimplex_f(x)
          logSE__ <- computeQse_lda(THETA__ = theta__,
                                    INDICES_ = DATA_SPLIT1,
                                    DOC_INDICES_U = doc_indices_u_split1,
                                    D_INDICES_U = d_indices_u_split1,
                                    TERMS_MAT_INPUT = TERMS_MAT,
                                    PI_MAT_INPUT = PI_MAT,
                                    DOC_LIST = DOC_LIST,
                                    MODAL_DOC_LEN = ModalDocLength,
                                    MARGINAL_BOUNDS = marginalB,
                                    LOG_TREATCOMBS=log_treatCombs,
                                    YOBS = Yobs,log=T)
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

  Qest_raw = computeQ_lda(theta   = optim_max,
                          Yobs     = YOBS[DATA_SPLIT1],
                          pi_mat    = PI_MAT[,DATA_SPLIT1],
                          doc_words = DOC_LIST[DATA_SPLIT1],
                          term_mat  = TERMS_MAT,
                          smoothWts = smoothWts, trim_q = trim_q,
                          maxWt = maxWt, maxWt_hajek = maxWt_hajek)
  names(Qest_raw$Q_wts_hajek) <- DATA_SPLIT1
  wts_raw <- Qest_raw$Q_wts_hajek
  Qest_raw = Qest_raw$Qhat*(1-hajek) + Qest_raw$Qhat_hajek*(hajek)

  Qest_split = computeQ_lda(theta   = optim_max,
                            Yobs     = YOBS[DATA_SPLIT2],
                            pi_mat    = PI_MAT[,DATA_SPLIT2],
                            doc_words = DOC_LIST[DATA_SPLIT2],
                            term_mat  = TERMS_MAT,
                            smoothWts = smoothWts, trim_q = trim_q,
                            maxWt = maxWt, maxWt_hajek = maxWt_hajek)
  names(Qest_split$Q_wts_hajek) <- DATA_SPLIT2
  hist(Qest_split$Q_wts_hajek,main = "Histogram of Weights")
  wts_split <- Qest_split$Q_wts_hajek
  Qest_split = Qest_split$Qhat*(1-hajek) + Qest_split$Qhat_hajek*(hajek)

  splitForSEObs <- "1"
  SE_obs = exp(computeQse_lda(THETA__ = optim_max,
                              INDICES_ = eval(parse(text=sprintf("DATA_SPLIT%s",splitForSEObs))),
                              DOC_INDICES_U = eval(parse(text=sprintf("doc_indices_u_split%s",splitForSEObs))),
                              D_INDICES_U = eval(parse(text=sprintf("d_indices_u_split%s",splitForSEObs))),
                              TERMS_MAT_INPUT = TERMS_MAT,
                              PI_MAT_INPUT = PI_MAT,
                              DOC_LIST = DOC_LIST,
                              MODAL_DOC_LEN = ModalDocLength,
                              MARGINAL_BOUNDS = marginalB,
                              LOG_TREATCOMBS=log_treatCombs,
                              YOBS = Yobs,log=T))

  splitForSEObs <- "1"
  SE_split = exp(computeQse_lda(THETA__ = optim_max,
                                INDICES_ = eval(parse(text=sprintf("DATA_SPLIT%s",splitForSEObs))),
                                DOC_INDICES_U = eval(parse(text=sprintf("doc_indices_u_split%s",splitForSEObs))),
                                D_INDICES_U = eval(parse(text=sprintf("d_indices_u_split%s",splitForSEObs))),
                                TERMS_MAT_INPUT = TERMS_MAT,
                                PI_MAT_INPUT = PI_MAT,
                                DOC_LIST = DOC_LIST,
                                MODAL_DOC_LEN = ModalDocLength,
                                MARGINAL_BOUNDS = marginalB,
                                LOG_TREATCOMBS=log_treatCombs,
                                YOBS = Yobs,log=T))
  Q_interval <- c(Qest_raw - 1.64*SE_obs,Qest_raw + 1.64*SE_obs)
  Q_interval_split <- c(Qest_split - 1.64*SE_split,Qest_split + 1.64*SE_split)


  names(optim_max_SEs_mEst) <- NULL
  OPTIMALITY_RESULTS = list("OptimalTopicAllocation"=c(optim_max),
                            "OptimalTopicAllocation_lb_boot"=c(boot_max_lower[1:length(optim_max)]),
                            "OptimalTopicAllocation_ub_boot"=c(boot_max_upper[1:length(optim_max)]),
                            "OptimalTopicAllocation_MEstimation_se" = c(NA,optim_max_SEs_mEst),
                            "AverageOutcomeAtOptimal"=Qest_raw,
                            "AverageOutcomeAtOptimal_split"=Qest_split,
                            "Q_interval" = Q_interval,
                            "Q_interval_split" = Q_interval_split,
                            "SE_Q" = SE_obs,
                            "SE_Q_split" = SE_split,
                            "Q_wts_raw"=wts_raw,
                            "Q_wts_split"=wts_split,
                            "CV_diagnostics"=cv_average_vec,
                            "CV_path" = cv_path)
  return(OPTIMALITY_RESULTS)
}

