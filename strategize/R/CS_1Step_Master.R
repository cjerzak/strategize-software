#' Implements...
#'
#' @usage
#'
#' OneStep.OptiConjoint(...)
#'
#' @param x Description
#'
#' @return `z` Description
#' @export
#'
#' @details `OneStep.OptiConjoint` Description
#'
#' - Description
#'
#' @examples
#'
#' # Analysis
#' OptiConjoint_analysis <- OneStep.OptiConjoint()
#'
#' print( OptiConjoint_analysis )
#'
#' @export
#'
#' @md

OneStep.OptiConjoint <- function(
                              W,
                              Y,
                              X = NULL,
                              K = 1,
                              warmStart = F,
                              automatic_scaling = T,
                              p_list = NULL,
                              hypotheticalProbList = NULL,
                              pi_init_vec = NULL,
                              constrain_ub = NULL,
                              nLambda = 10,
                              penaltyType = "LogMaxProb",
                              testFraction = 0.5,
                              log_PrW = NULL,
                              LEARNING_RATE_BASE = 0.01,
                              cycle_width  = 50,
                              cycle_number = 4,
                              nSGD = 500,
                              nEpoch = NULL,
                              X_factorized = NULL,
                              momentum = 0.99,
                              nFullCycles_noRestarts = 1,
                              optim_method = "tf",
                              sg_method = NULL,
                              forceSEs = F,
                              clipAT_factor = 100000,
                              adaptiveMomentum = F,
                              PenaltyType = "L2",
                              knownNormalizationFactor = NULL,
                              split1_indices=NULL,
                              split2_indices=NULL,
                              openBrowser = F,
                              useHajekInOptimization = T, findMax = T,quiet = T,
                              lambda_seq = NULL,
                              lambda_coef = 0.0001,
                              nFolds = 3, batch_size = 50,
                              confLevel = 0.90,
                              conda_env = NULL,
                              conda_env_required = F,
                              hypotheticalNInVarObj=NULL){
  # load in packages
  {
    # conda_env <- "tensorflow_m1" ; conda_env_required <- T
    if(!is.null(conda_env)){
      try(reticulate::use_condaenv(conda_env, required = conda_env_required), T)
    }
    jax <- reticulate::import("jax")
    oryx <- reticulate::import("tensorflow_probability.substrates.jax") #
    optax <- reticulate::import("optax")
    jnp <- reticulate::import("jax.numpy")
    eq <- reticulate::import("equinox")
    np <- reticulate::import("numpy")
    py_gc <- reticulate::import("gc")
    piSEtype  = "automatic"
  }

  # define evaluation environment
  evaluation_environment <- environment()

  # initial processing
  {
  if(is.null(X)){ X <- cbind(rnorm(length(Y)),rnorm(length(Y))) }
  if(automatic_scaling == T){  X <- scale ( X  )  }

  useHajek <- T
  if(any(unlist(lapply(p_list,class)) == "table")){
    for(ij in 1:length(p_list)){
      n_ <- names(p_list[[ij]] )
      p_list[[ij]] <- as.vector(p_list[[ij]])
      names(p_list[[ij]]) <- n_
    } }
  penaltyProbList_unlisted <- unlist(p_list)
  if(findMax == F){ Y <- -1 * Y }

  # SCALE Y
  mean_Y <- 0; sd_Y <- 1
  if(automatic_scaling == T){
    mean_Y <- mean(Y); sd_Y <- sd(Y)
    Y <-   (Y -  mean_Y) / sd_Y
  }

  FactorsMat_numeric <- sapply(1:ncol(W),function(ze){ match(W[,ze], names(p_list[[ze]]))  })
  }

  ### case 1 - the new multinomial probabilities ARE specified
  if(!is.null(hypotheticalProbList)){
    Qhat_all <- computeQ_conjoint_internal(FactorsMat_internal = FactorsMat_numeric,
                                       Yobs_internal = Y,
                                       log_pr_w_internal = NULL,
                                       knownNormalizationFactor = knownNormalizationFactor,
                                       assignmentProbList_internal = p_list,
                                       hypotheticalProbList_internal = hypotheticalProbList,
                                       hajek = useHajek)
    SE_Q_all <- computeQse_conjoint(FactorsMat = FactorsMat_numeric,
                                Yobs = Y,
                                # hypotheticalN = NULL,
                                # log_PrW = NULL,
                                # log_treatment_combs = NULL,
                                hajek = useHajek,
                                knownNormalizationFactor = knownNormalizationFactor,
                                assignmentProbList = p_list,
                                returnLog = F,
                                hypotheticalProbList = hypotheticalProbList)
    Q_interval_all <- c(Qhat_all$Qest - abs(qnorm((1-confLevel)/2))*SE_Q_all,
                        Qhat_all$Qest_all + abs(qnorm((1-confLevel)/2))*SE_Q_all)
    return(    list("Q_point_all" = RescaleFxn(Qhat_all$Qest, estMean = mean_Y, estSD = sd_Y),
                    "Q_se_all" = RescaleFxn(SE_Q_all, estMean = mean_Y, estSD = sd_Y,center=F),
                    "Q_interval_all" = RescaleFxn(Q_interval_all, estMean = mean_Y, estSD = sd_Y),
                    "Q_wts_all" = Qhat_all$Q_wts,
                    "Q_wts_raw_sum_all" = Qhat_all$Q_wts_raw_sum,
                    "log_pr_w_new_all"=Qhat_all$log_pr_w_new,
                    "log_pr_w_all"=Qhat_all$log_PrW) )
  }#end !is.null(hypotheticalProbList)

  #### case 2 - the new multinomial probabilities ARE NOT specified
  if(is.null(hypotheticalProbList)){

    # setup for m estimation
    forceHajek <- T
    zStar <- abs(qnorm((1-confLevel)/2))
    varHat <- mean( (Y - (muHat <- mean(Y)) )^2   )
    n_target <- ifelse(is.null(hypotheticalNInVarObj), yes = length(  split1_indices  ), no = hypotheticalNInVarObj)

    # define number of treatment combinations
    treatment_combs <- exp(log_treatment_combs  <- sum(log(sapply(1:ncol(W),function(ze){ length(p_list[[ze]]) }) )))

    # initialize quantities
    marginalProb_m <- seList <- seList_automatic <- m_se_Q <- seList_manual <- lowerList <- upperList <- NULL
    PrXd_vec <- PrXdGivenClust_se <- PrXdGivenClust_mat <- NULL
    if(is.null(split1_indices)){
      split_ <- rep(1,times=length(Y))
      if(is.null(testFraction)){ testFraction <- 0.5 }
      split_[sample(1:length(split_), round(length(Y)*testFraction))] <- 2
      split1_indices = which(split_ == 1); split2_indices = which(split_ == 2)
      if(length(lambda_seq) == 1){
        warning("NO SAMPLE SPLITTING, AS LAMBDA IS FIXED")
        split1_indices <- split2_indices <- 1:length(Y)
      }
    }
    print(c(length(split1_indices),length(split2_indices)))

    # execute splits
    FactorsMat1 <- W[split1_indices,];FactorsMat1_numeric <- FactorsMat_numeric[split1_indices,]
    FactorsMat2 <- W[split2_indices,];FactorsMat2_numeric <- FactorsMat_numeric[split2_indices,]

    Yobs_split1 <- Y[split1_indices]
    log_pr_w_split1 <- log_PrW[split1_indices]
    sigma2_hat_split1 <- var( Yobs_split1)

    Yobs_split2 <- Y[split2_indices]
    log_pr_w_split2 <- log_PrW[split2_indices]
    sigma2_hat_split2 <- var( Yobs_split2)

    # obtain pr(w) so we don't need to recompute it at every step
    if( is.null(log_PrW)  ){
      log_PrW    <-   as.vector(computeQ_conjoint_internal(FactorsMat_internal = FactorsMat_numeric,
                                                            Yobs_internal=Y,
                                                            hypotheticalProbList_internal = p_list,
                                                            assignmentProbList_internal = p_list,
                                                            hajek = useHajek)$log_PrW)
    }

    # INITIALIZE M ESTIMATION
    print("Initialize M-Estimation...")
    initialMtext <- paste(deparse(initialize_m),collapse="\n")
    initialMtext <- gsub(initialMtext,pattern="function \\(\\)",replace="")
    eval(parse( text = initialMtext ),envir = evaluation_environment)

    # get initial pi values
    print("Initialize pi values...")
    {
      if(is.null(pi_init_vec)){
        TARGET_EPSILON_PI <- 0.01  #/ exp( length( p_list ) )
        CLUSTER_EPSILON <- 0.1

        pi_init_type <- "runif"
        pi_init_list = replicate(nFolds+1,replicate(K,lapply(p_list,function(ze){
          systemit_init <- tmp <- ((c(rev(c(compositions::alr(t(rev(ze))))))))
          p_val <- prop.table(exp(ze))
          bounds_seq <- seq(0.001,0.25,length.out=100)
          epsilon_penalty_seq <- sapply(bounds_seq,function(b_){
            max( replicate(100,{
              {
                if(pi_init_type == "runif"){ log_pi <- tmp + runif(length(tmp),min = -b_,  max = b_)}
                if(pi_init_type == "rnorm"){ log_pi <- tmp + rnorm(length(tmp),mean = 0, sd = b_) }
                log_pi <- c(0,log_pi)
                pi_val <- prop.table(exp(log_pi))
              }
              #epsilon_penalty <- max(abs(pi_val  - p_val))
              epsilon_penalty <- max( (pi_val  - p_val) )
              }))})
          b_ <- bounds_seq[ which.min(abs(epsilon_penalty_seq - TARGET_EPSILON_PI)) ]
          if(pi_init_type == "rnorm"){ return_value <- as.matrix(tmp + rnorm(length(tmp), mean = 0, sd = b_))}
          if(pi_init_type == "runif"){ return_value <- as.matrix(tmp + runif(length(tmp), min = -b_, max = b_))}
          attr(return_value,"random_") <- rep(b_,length(return_value))
          attr(return_value,"sys_") <- systemit_init
          return(return_value)
        })))
        pi_init_vec <- unlist(  pi_init_list[,1,1]  )
      }
      names(pi_init_vec) <- NULL
    }

    # get mapped transformations
    {
      probsIndex_mapped <- unlist(p_list)
      probsIndex_mapped[] <- 1:length( probsIndex_mapped )
      probsIndex_mapped <- vec2list_noTransform(probsIndex_mapped)
      FactorsMat1_mapped <- sapply(1:ncol(FactorsMat1_numeric),function(ze){
        probsIndex_mapped[[ze]][ FactorsMat1_numeric[,ze] ]  })
      row.names(FactorsMat1_numeric) <- colnames(FactorsMat1_numeric) <- row.names(FactorsMat1_mapped) <- colnames(FactorsMat1_mapped) <- NULL
    }

    # main cross-validation routine
    optim_max_hajek_list <- sapply(1:(length(lambda_seq)+1),function(er){
      list(matrix(NA,nrow = length( pi_init_vec), ncol = nFolds)) })
    if ( length(lambda_seq) == 1 ){
      LAMBDA_selected <- LAMBDA_ <- lambda_seq;

      FactorsMat_numeric_IN <- FactorsMat1_numeric <- FactorsMat2_numeric <- FactorsMat_numeric
      Yobs_IN <- Yobs_split1 <- Yobs_split2 <- Y
      log_pr_w_IN <- log_pr_w_split1 <- log_pr_w_split2 <- log_PrW
    }
    lambda_seq <- sort(lambda_seq, decreasing = T)

    # optimization
    {
        # helper functions
        if(is.null(X)){X<-matrix(rnorm(length(Y)*max(2,K)),nrow=length(Y),ncol=max(2,K))}
        FactorsMat_numeric <- sapply(1:ncol(FactorsMat_numeric <- W),function(zer){
          match(FactorsMat_numeric[,zer], names(p_list[[zer]])) })
        FactorsMat_numeric_0Indexed <- FactorsMat_numeric - 1L

        performance_matrix_out <- performance_matrix_in <- matrix(NA, ncol = length(lambda_seq), nrow = nFolds)
        holdBasis_traintv <- sample(1:length(split1_indices) %% nFolds+1)
        REGULARIZATION_LAMBDA_SEQ_ORIG <- lambda_seq

        if(length(lambda_seq) > 1){trainIndicator_pool <- c(1,0); if(nFolds == 1){stop("ERROR: SET nFolds > 1!")}}
        if(length(lambda_seq) == 1){warning(sprintf("NO CV SELCTION OF LAMBDA, FORCING LAMBDA = %.5f|",lambda_seq)); trainIndicator_pool <- 0}

        # Build Model
        print("Building...")
        buildText <- paste(deparse(ml_build),collapse="\n")
        buildText <- gsub(buildText,pattern="function \\(\\)",replace="")
        eval(parse( text = buildText ),envir = evaluation_environment)

        # Train Model + Perform CV
        print("Training...")
        trainText <- paste(deparse(ml_train),collapse="\n")
        trainText <- gsub(trainText,pattern="function \\(\\)",replace="")
        eval(parse( text = trainText ),envir = evaluation_environment)

        # obtain the pi's
        hypotheticalProbList <- getPiList( ModelList_object[[1]] );names(hypotheticalProbList) <- paste("k",1:K,sep="")
        FinalProbList <- hypotheticalProbList
        optim_max_hajek_full <- na.omit(  unlist(getPiList(ModelList_object[[1]], simplex=F)) )

        ClassProbs <- NULL; if(K > 1){ ClassProbs <- as.array(  getClassProb(X) ) }
        performance_mat <- as.data.frame(
                          cbind("lambda" = REGULARIZATION_LAMBDA_SEQ_ORIG,
                                "Qhat" = colMeans(performance_matrix_out),
                                "Qse" = apply(performance_matrix_out,2,getSE)
                                #"Q_in" = colMeans(performance_matrix_in), "Qse_in" = apply(performance_matrix_in,2,getSE))
                                ))
        LAMBDA_ <- LAMBDA <- LAMBDA_selected
    }

    optim_max_hajek_split1 <- optim_max_hajek_full
    if(length(REGULARIZATION_LAMBDA_SEQ_ORIG)>1){
    qStar <- 1
    par(mar=c(5,5,5,1))
    plot( log(performance_mat$lambda), main = sprintf("%s Fold CV",nFolds),cex.main = 2,
          performance_mat$Q_in,xlab = "log(lambda)", ylab="Value", col="gray",type = 'b',
          ylim = summary(c(performance_mat$Q_in - qStar*performance_mat$Qse_in,
                           performance_mat$Q_in + qStar*performance_mat$Qse_in,
                           performance_mat$Q_out - qStar*performance_mat$Qse_out,
                           performance_mat$Q_out + qStar*performance_mat$Qse_out
                             ))[c(1,6)])
    abline(h=mean(Y),lty = 2, col= "gray",lwd=2)
    for(iaj in 1:nrow(performance_mat)){
      points( c(log(performance_mat$lambda[iaj]),log(performance_mat$lambda[iaj])),
            c(performance_mat$Q_in[iaj] -qStar*performance_mat$Qse_in[iaj],
              performance_mat$Q_in[iaj] +qStar*performance_mat$Qse_in[iaj]), type = "l", lwd = 2,col = "gray" )
    }
    points( log(performance_mat$lambda), performance_mat$Q_out,type = "b",pch = "O")
    for(iaj in 1:nrow(performance_mat)){
      points( c(log(performance_mat$lambda[iaj]),log(performance_mat$lambda[iaj])),
              c(performance_mat$Q_out[iaj] -qStar*performance_mat$Qse_out[iaj],
                performance_mat$Q_out[iaj] +qStar*performance_mat$Qse_out[iaj]),type = "l", lwd = 2,col = "black" )
    }
    abline(v=log(  performance_mat$lambda[which.max(performance_mat$LB_empirical_out)]),lty=2,lwd=0.5 )
    }

    # save and store results
    {
      # save results - split 1
      hypotheticalProbList_full <- hypotheticalProbList_split1 <- hypotheticalProbList

      # Get values from tf
      Qhat_tf <- jnp$array( getQ_fxn( ModelList_object,  split2_indices  ), jnp$float32)
      Qhat <- np$array( Qhat_tf )
      Qhat_split1 <- Qhat_all <- Q_interval_split2 <- Q_interval_split1 <- Q_se_exact <- NULL
      Qhat_split2<-list();Qhat_split2$Qest <- np$array(Qhat_tf)
    }

    # get SEs
    seText <- paste(deparse(get_se),collapse="\n")
    seText <- gsub(seText,pattern="function \\(\\)",replace="")
    eval(parse( text = seText ),evaluation_environment)

    #plot(unlist(seList_manual),unlist(seList_automatic));abline(a=0,b=1)
    #plot(VarCov_n_automatic[,1],VarCov_n_manual[,1]);abline(a=0,b=1)
    #plot(VarCov_n_automatic[,5],VarCov_n_manual[,5]);abline(a=0,b=1)
    #image(cor(VarCov_n_automatic,VarCov_n_manual))

    return( list("PiStar_point"=hypotheticalProbList_full,
               "PiStar_se" = seList,
               "ClassProbProjCoefs" = ClassProbProjCoefs,
               "ClassProbProjCoefs_se" = ClassProbProjCoefs_se,
               "PiStar_lb"=lowerList,
               "PiStar_ub"=upperList,
               "Q_point"= RescaleFxn(Qhat_split1$Qest, estMean = mean_Y, estSD = sd_Y,center=T),
               "Q_point_mEst"= RescaleFxn(Qhat_split2$Qest, estMean = mean_Y, estSD = sd_Y,center=T),
               "Q_se_exact" = RescaleFxn(Q_se_exact, estMean = mean_Y, estSD = sd_Y,center=F),
               "Q_se_mEst" = RescaleFxn(m_se_Q, estMean = mean_Y, estSD = sd_Y,center=F),
               "Q_interval_split1" = RescaleFxn(Q_interval_split1, estMean = mean_Y, estSD = sd_Y,center=T),
               "Q_interval_split2" = RescaleFxn(Q_interval_split2, estMean = mean_Y, estSD = sd_Y,center=T),
               "Q_wts" = Qhat_split2$Q_wts,
               "Q_point_split2"=Qhat_split2$Qest,
               "Q_wts_raw_sum" = Qhat_split2$Q_wts_raw_sum,
               "Q_wts_raw_sum_split2" = Qhat_split2$Q_wts_raw_sum,
               "Q_wts_split2" = Qhat_split2$Q_wts,
               "split1_indices" = split1_indices,
               "split2_indices" = split2_indices,
               "ClassProbsXobs" = ClassProbsXobs,
               "VarCov_ProbClust" = VarCov_ProbClust,
               "MarginalClusterProbEvolution"=marginalProb_m,
               "pi_init_next" = optim_max_hajek_full,
               "CVInfo" = performance_mat,
               "optim_max_hajek_list" = optim_max_hajek_list,
               "PrXd_vec" = PrXd_vec,
               "X_factorized_complete"=ifelse(("X_factorized_complete" %in% ls()),yes=list(X_factorized_complete),no=list(NULL)),
               "PrXdGivenClust" = PrXdGivenClust_mat,
               "estimationType" = "OneStep",
               "PrXdGivenClust_se" = PrXdGivenClust_se,
               "Output.Description"=c("")) )
  }
}


