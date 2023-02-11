computeQ_lda_internal <- function(pi=NULL,
                         term_mat,
                         Yobs,
                         doc_words,
                         dtm=NULL,
                         pi_mat=NULL,
                         alpha_mat=NULL,
                         log_pr_w=NULL,
                         computeSE = F,
                         trim_q=1,quiet=T,iters=100,
                         smoothWts = F,
                         TreatFxn = NULL,
                         maxWt = 1e10,
                         maxWt_hajek = NULL,
                         term_mat_TRUE = NULL, # for simulation purposes only
                         doc_indices_u = NULL, # for computational speedups only
                         d_indices_u = NULL,# for computational speedups only
                         diagnostics = F){
  if( all( colnames(term_mat) !=  1:ncol(term_mat))){ #for fast execution
    doc_words = lapply(doc_words,function(ze){ sapply(ze, function(zee){which(colnames(term_mat) %in% zee )})})
    colnames(term_mat) <- 1:ncol(term_mat)
  }
  tauhat_noTrim_vec <- Q_maxWts <- Q_vec <- tauhat_vec <- rep(NA,times=1)
  Q_vec_hajek <- tauhat_vec_hajek <- tauhat_vec_aug <- Q_vec_aug <- tauhat_vec
  Q_vec_hajek_VAR <- Q_vec_aug <- tauhat_vec_aug <- Q_vec
  Qhat_double <- Qhat_hajek_double <- c()
  Qhat_double2 <- Qhat_hajek_double2 <- c()
  n_iters = 1; intOutPi = F
  if(is.null(pi_mat)){intOutPi = T; n_iters=iters}
  for(ijack in 1:n_iters){
    if(!quiet){if(ijack == 1 | (ijack / n_iters) %in% seq(0,1,by=0.10)){print(sprintf("Complete frac: %s",round(ijack/n_iters,1) ))}}
    ndocs = length(doc_words)
    if(is.null(TreatFxn)){
      pi_mat = replicate(pi,n=ndocs)
    }
    if(!is.null(TreatFxn)){
      pi_mat = TreatFxn(alpha_mat,pi)
    }

    #calculate conditionals
    {
      if(is.null(term_mat_TRUE)){term_mat_USE <- term_mat}
      log_pr_w_given_pi   = logPrWGivenPi_fxn(doc_indices     = doc_words,
                                              pi_mat          = pi_mat,
                                              terms_posterior = term_mat_USE,
                                              doc_indices_u   = doc_indices_u,
                                              d_indices_u     = d_indices_u)


      if(intOutPi){pi_mat = apply(alpha_mat, 2, function(ae){gtools::rdirichlet(1,ae)})}
      if(is.null(log_pr_w)){
        log_pr_w                = logPrWGivenPi_fxn(doc_indices          = doc_words,
                                                 pi_mat          = pi_mat,
                                                 terms_posterior = term_mat,
                                                 doc_indices_u = doc_indices_u,
                                                 d_indices_u = d_indices_u)
      }
    }

    # Perform weighting estimation
    {
      {
        Q_wts_noTrim_uniformDenom <- length(log_pr_w_given_pi)*exp(log_pr_w_given_pi)/sum(exp(log_pr_w_given_pi))
        Q_wts_uniformDenom <- length(log_pr_w_given_pi)*exp(trim_fxn(log_pr_w_given_pi-matrixStats::logSumExp(log_pr_w_given_pi),trim_q))
        Q_wts_uniformDenom[Q_wts_uniformDenom>maxWt] <- maxWt
        Q_uniformDenom =  mean(Yobs *  Q_wts_uniformDenom,na.rm=T)#(Q_wts/sum(Q_wts))
        Q_uniformDenom_noTrim =  mean(Yobs *  Q_wts_noTrim_uniformDenom,na.rm=T)#(Q_wts/sum(Q_wts))
      }
      {
        Q_wts_noTrim <- exp(log_pr_w_given_pi-log_pr_w)
        if(!smoothWts){Q_wts <- trim_fxn(exp(log_pr_w_given_pi-log_pr_w),  trim_q)}
        if(smoothWts){Q_wts <- trim_fxn(SRatFxn(log_pr_w_given_pi ,  log_pr_w ,100),trim_q)}
      }
      Q_wts[Q_wts>maxWt]<-maxWt
      prodQ = Yobs *  Q_wts
      prodQ_good = (prodQ)[!is.na(prodQ)]
      Q_wts_good = (Q_wts)[!is.na(prodQ)]
      Q_wts_norm <- hajekNorm(Q_wts_good,maxWt_hajek)
      Q_vec_hajek[ijack] <- Q_hajek <- sum(Yobs * Q_wts_norm)

      Q_vec[ijack] = mean(prodQ,na.rm=T)
      Q_maxWts[ijack] = wtPen(Q_wts)
      {
        Q_vec[ijack] = mean(prodQ,na.rm=T)
        Qhat = mean(prodQ,na.rm=T)
        Q_noTrim = Yobs *  Q_wts_noTrim
        Qhat_noTrim = mean(Q_noTrim,na.rm=T)
      }
    }

    # Perform augmented weighting
    if(!is.null(dtm)){
      # Model for the outcome
      if(T == F){
        library(glmnet)
        my_glmnet = cv.glmnet(x=as.matrix(dtm),y=as.matrix(Yobs),nfolds = 3)
        #my_coef = unique(c(1,which(abs(coef(my_glmnet,s="lambda.min")[-1,])>0)))
        #my_lm = lm(Yobs~as.matrix(dtm[,my_coef]))
        Yobs_pred = c(predict(my_glmnet,s="lambda.min",newx=as.matrix(dtm)))
        nd_vec = rowSums(dtm)
        coef_glmnet = c(as.matrix(coef(my_glmnet,s="lambda.min")))
        pi_draw_ = pi_draw[,1]# ASSUMES ALL SAME NEW PI
        prW_theoretical = colSums( pi1_draw_ *term_mat )
        #plot(nd_vec[1]*prW_theoretical,prW_empirical);abline(a=0,b=1)
        mySum = sum(prW_theoretical * coef_glmnet[-1])
        Yobs_pred = sapply(1:length(nd_vec),function(nd_){ 1*coef_glmnet[1]+nd_vec[nd_] * mySum })
        #Yobs_pred1_mc =rowMeans(replicate(2,c(predict(my_glmnet,s="lambda.min",newx=as.matrix(LDAdraw(PI_MAT = pi1_draw,
        Q_hat_double[ijack] = mean(Q_wts * (Yobs-Yobs_pred) + Yobs_pred1)
        Q_hat_hajek_double[ijack] = sum(Q_wts_norm * (Yobs-Yobs_pred) + 1/length(Yobs)*Yobs_pred)
      }

      #estimate Y(pi1) using estimated pi_i's as linear predictors (preferred)
      {
        lm_dat_est = cbind(Yobs,t(pi_mat))
        colnames(lm_dat_est)[-1] <- 1:nrow(pi_mat)
        my_lm_est = lm(Yobs~.,as.data.frame(lm_dat_est))
        newdata1_est = t(lm_dat_est[1,]);newdata1_est[-1] <- pi1
        Yobs_pred_topics = predict(my_lm_est)
        Y1_hat_topics <- predict(my_lm_est,newdata = as.data.frame(newdata1_est))
        EY1_hat_double2[ijack] = mean(Q1_wts * (Yobs-Yobs_pred_topics) + Y1_hat_topics)
        EY1_hat_hajek_double2[ijack] = sum(Q1_wts_norm * (Yobs-Yobs_pred_topics) + 1/length(Yobs)*Y1_hat_topics)
      }
    }
  }
  logMinProbSupported <- diagnostics_ratio <- diagnostics_value <- list()
  logMaxRatioSupported <- NA
  if(diagnostics == T){
    NumList <- DemList <- list(); min_vec <- max_vec <- c()
    nDim = ncol(term_mat)
    logMinProbSupported = nDim*min(log(term_mat))
    logProbWords = log(colSums( term_mat * pi_draw[,1]))
    logProbWordGivenU = t(log(term_mat))
    logMaxRatioSupported = nDim*max(logProbWords - logProbWordGivenU)
  }

  Qhat_se <- NULL
  if(computeSE){
    browser()
  }

  return( list(Q_mean     = mean(Q_vec,na.rm=T),
               Q_maxWts   = mean(Q_maxWts),
               Q_mean_aug = mean(Q_vec_aug,na.rm = T),
               Q_mean_hajek     = mean(Q_vec_hajek,na.rm=T),
               Q_wts = Q_wts,
               Q_wts_hajek = Q_wts_norm,
               Qhat = Qhat,
               Qhat_se = Qhat_se,
               Qhat_noTrim = Qhat_noTrim,
               Qhat_double = mean(Qhat_double,na.rm=T),
               Qhat_hajek_double = mean(Qhat_hajek_double,na.rm=T),
               Qhat_double2 = mean(Qhat_double2,na.rm=T),
               Qhat_hajek_double2 = mean(Qhat_hajek_double2,na.rm=T),
               Qhat_hajek     = mean(Q_vec_hajek,na.rm=T),
               Qhat_uniform = mean(Q_uniformDenom,na.rm=T),
               Qhat_uniformDenom = mean(Q_uniformDenom,na.rm=T),
               Qhat_uniformDenom_noTrim = mean(Q_uniformDenom_noTrim,na.rm=T),
               logMinProbSupported = logMinProbSupported,
               log_pr_w = log_pr_w,
               Q_vec = Q_vec,
               logMaxRatioSupported = logMaxRatioSupported))
}





