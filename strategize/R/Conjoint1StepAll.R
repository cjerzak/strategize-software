#' computeQ_OneStep
#'
#' Implements...
#'
#' @usage
#'
#' computeQ_OneStep(x, y, by ...)
#'
#' @param x,y data frames to be merged
#'
#' @return `z` The merged data frame.
#' @export
#'
#' @details `LinkOrgs` automatically processes the name text for each dataset (specified by `by`, `by.x`, and/or `by.y`. Users may specify the following options:
#'
#' - Set `DistanceMeasure` to control algorithm for computing pairwise string distances. Options include "`osa`", "`jaccard`", "`jw`". See `?stringdist::stringdist` for all options. (Default is "`jaccard`")
#'
#' @examples
#'
#' #Create synthetic data
#' x_orgnames <- c("apple","oracle","enron inc.","mcdonalds corporation")
#' y_orgnames <- c("apple corp","oracle inc","enron","mcdonalds co")
#' x <- data.frame("orgnames_x"=x_orgnames)
#' y <- data.frame("orgnames_y"=y_orgnames)
#'
#' # Perform merge
#' linkedOrgs <- LinkOrgs(x = x,
#'                        y = y,
#'                        by.x = "orgnames_x",
#'                        by.y = "orgnames_y",
#'                        MaxDist = 0.6)
#'
#' print( linkedOrgs )
#'
#' @export
#'
#' @md

strategize_OneStep <- function(FactorsMat,
                              Yobs,
                              X = NULL,
                              kClust = 1,
                              warmStart = F,
                              automatic_scaling = T,
                              assignmentProbList = NULL,
                              hypotheticalProbList = NULL,
                              pi_init_vec = NULL,
                              constrain_ub = NULL,
                              nLambda = 10,
                              useVariational = F,
                              penaltyType = "LogMaxProb",
                              testFraction = 0.5,
                              log_pr_w = NULL,
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
                              piSEtype  = "automatic",
                              openBrowser = F,
                              useHajekInOptimization = T, findMax = T,quiet = T,
                              LAMBDA_SEQ = NULL,
                              COEF_LAMBDA = 0.0001,
                              nFolds = 3, batch_size = 50,
                              confLevel = 0.90,
                              hypotheticalNInVarObj=NULL){
  # load in packages
  {
    # tensorflow
    library(tensorflow); library(keras)
    try(tensorflow::use_python(python = "/Users/cjerzak/miniforge3/bin/python", required = T),T)
    try(tensorflow::use_condaenv("tensorflow_m1",required = T, conda = "~/miniforge3/bin/conda"), T)
    try(tf$config$experimental$set_memory_growth(tf$config$list_physical_devices('GPU')[[1]], T),T)
    try(tfp <- tf_probability(),T)
    try(tfd <- tfp$distributions,T)
    tf$random$set_seed(as.integer(runif(1,0,100000)))# critical - MUST SET SEED FOR ALL RANDOM GENERATION TO WORK
    print(tf$version$VERSION)

    # jax
    jax <- tensorflow::import("jax",as="jax")
    oryx <- tensorflow::import("oryx")
    jnp <- tensorflow::import("jax.numpy")
    tf2jax <- tensorflow::import("tf2jax",as="tf2jax")
  }

  # define evaluation environment
  evaluation_environment <- environment()

  # initial processing
  {
  if(is.null(X)){ X <- cbind(rnorm(length(Yobs)),rnorm(length(Yobs))) }
  if(automatic_scaling == T){  X <- scale ( X  )  }

  useHajek <- T
  if(openBrowser){browser()}
  if(any(unlist(lapply(assignmentProbList,class)) == "table")){
    for(ij in 1:length(assignmentProbList)){
      n_ <- names(assignmentProbList[[ij]] )
      assignmentProbList[[ij]] <- as.vector(assignmentProbList[[ij]])
      names(assignmentProbList[[ij]]) <- n_
    } }
  penaltyProbList_unlisted <- unlist(assignmentProbList)
  if(findMax == F){ Yobs <- -1 * Yobs }

  # SCALE Yobs
  mean_Y <- 0; sd_Y <- 1
  if(automatic_scaling == T){
  mean_Y <- mean(Yobs); sd_Y <- sd(Yobs)
  Yobs <-   (Yobs -  mean_Y) / sd_Y
  }

  FactorsMat_numeric <- sapply(1:ncol(FactorsMat),function(ze){ match(FactorsMat[,ze], names(assignmentProbList[[ze]]))  })
  }

  ### case 1 - the new multinomial probabilities ARE specified
  if(!is.null(hypotheticalProbList)){
    Qhat_all <- computeQ_conjoint_internal(FactorsMat_internal = FactorsMat_numeric,
                                       Yobs_internal = Yobs,
                                       log_pr_w_internal = NULL,
                                       knownNormalizationFactor = knownNormalizationFactor,
                                       assignmentProbList_internal = assignmentProbList,
                                       hypotheticalProbList_internal = hypotheticalProbList,
                                       hajek = useHajek)
    SE_Q_all <- computeQse_conjoint(FactorsMat=FactorsMat_numeric,
                                Yobs=Yobs,
                                hypotheticalN = NULL,
                                log_pr_w = NULL,log_treatment_combs = NULL,
                                hajek = useHajek,
                                knownNormalizationFactor = knownNormalizationFactor,
                                assignmentProbList=assignmentProbList, returnLog = F,
                                hypotheticalProbList=hypotheticalProbList)
    Q_interval_all <- c(Qhat_all$Qest - abs(qnorm((1-confLevel)/2))*SE_Q_all,
                        Qhat_all$Qest_all + abs(qnorm((1-confLevel)/2))*SE_Q_all)
    return(    list("Q_point_all" = RescaleFxn(Qhat_all$Qest, estMean = mean_Y, estSD = sd_Y),
                    "Q_se_all" = RescaleFxn(SE_Q_all, estMean = mean_Y, estSD = sd_Y,center=F),
                    "Q_interval_all" = RescaleFxn(Q_interval_all, estMean = mean_Y, estSD = sd_Y),
                    "Q_wts_all" = Qhat_all$Q_wts,
                    "Q_wts_raw_sum_all" = Qhat_all$Q_wts_raw_sum,
                    "log_pr_w_new_all"=Qhat_all$log_pr_w_new,
                    "log_pr_w_all"=Qhat_all$log_pr_w) )
  }#end !is.null(hypotheticalProbList)

  #### case 2 - the new multinomial probabilities ARE NOT specified
  if(is.null(hypotheticalProbList)){

    # setup for m estimation
    {
      forceHajek <- T
      zStar <- abs(qnorm((1-confLevel)/2))
      varHat <- mean( (Yobs - (muHat <- mean(Yobs)) )^2   )
      n_target <- ifelse(is.null(hypotheticalNInVarObj), yes = length(  split1_indices  ), no = hypotheticalNInVarObj)
    }

    # define number of treatment combinations
    treatment_combs <- exp(log_treatment_combs  <- sum(log(sapply(1:ncol(FactorsMat),function(ze){ length(assignmentProbList[[ze]]) }) )))

    # initialize quantities
    marginalProb_m <- seList <- seList_automatic <- m_se_Q <- seList_manual <- lowerList <- upperList <- NULL
    PrXd_vec <- PrXdGivenClust_se <- PrXdGivenClust_mat <- NULL
    if(is.null(split1_indices)){
      split_ <- rep(1,times=length(Yobs))
      if(is.null(testFraction)){ testFraction <- 0.5 }
      split_[sample(1:length(split_), round(length(Yobs)*testFraction))] <- 2
      split1_indices = which(split_ == 1); split2_indices = which(split_ == 2)
      if(length(LAMBDA_SEQ) == 1){
        warning("NO SAMPLE SPLITTING, AS LAMBDA IS FIXED")
        split1_indices <- split2_indices <- 1:length(Yobs)
      }
    }
    print(c(length(split1_indices),length(split2_indices)))

    # execute splits
    FactorsMat1 <- FactorsMat[split1_indices,];FactorsMat1_numeric <- FactorsMat_numeric[split1_indices,]
    FactorsMat2 <- FactorsMat[split2_indices,];FactorsMat2_numeric <- FactorsMat_numeric[split2_indices,]

    Yobs_split1 <- Yobs[split1_indices]
    log_pr_w_split1 <- log_pr_w[split1_indices]
    sigma2_hat_split1 <- var( Yobs_split1)

    Yobs_split2 <- Yobs[split2_indices]
    log_pr_w_split2 <- log_pr_w[split2_indices]
    sigma2_hat_split2 <- var( Yobs_split2)

    # obtain pr(w) so we don't need to recompute it at every step
    if(is.null(log_pr_w)){
      log_pr_w    <-   as.vector(computeQ_conjoint_internal(FactorsMat_internal = FactorsMat_numeric,
                                                            Yobs_internal=Yobs,
                                                            hypotheticalProbList_internal = assignmentProbList,
                                                            assignmentProbList_internal = assignmentProbList,
                                                            hajek = useHajek)$log_pr_w)
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
        normType <- "alr"
        #normType <- "ilr"
        print(sprintf("Using %s prob. norm.",toupper(normType)))
        TARGET_EPSILON_PI <- 0.01  #/ exp( length( assignmentProbList ) )
        CLUSTER_EPSILON <- 0.1

        if(useVariational == T){ pi_init_type <- "rnorm" }
        if(useVariational == F){ pi_init_type <- "runif" }
        pi_init_list = replicate(nFolds+1,replicate(kClust,lapply(assignmentProbList,function(ze){
          if(normType == "alr"){systemit_init <- tmp <- ((c(rev(c(compositions::alr(t(rev(ze))))))))}
          if(normType == "ilr"){systemit_init <- tmp <- ((c((c(compositions::ilr(t((ze))))))))}
          p_val <- prop.table(exp(ze))
          bounds_seq <- seq(0.001,0.25,length.out=100)
          epsilon_penalty_seq <- sapply(bounds_seq,function(b_){
            max( replicate(100,{
              {
                if(pi_init_type == "runif"){ log_pi <- tmp + runif(length(tmp),min = -b_,  max = b_)}
                if(pi_init_type == "rnorm"){ log_pi <- tmp + rnorm(length(tmp),mean = 0, sd = b_) }
                if(normType == "alr"){
                  log_pi <- c(0,log_pi)
                  pi_val <- prop.table(exp(log_pi))
                }
                if(normType == "ilr"){
                  pi_val <- f2n(compositions::ilrInv(t((log_pi))))
                }
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
      probsIndex_mapped <- unlist(assignmentProbList)
      probsIndex_mapped[] <- 1:length( probsIndex_mapped )
      probsIndex_mapped <- vec2list_noTransform(probsIndex_mapped)
      FactorsMat1_mapped <- sapply(1:ncol(FactorsMat1_numeric),function(ze){
        probsIndex_mapped[[ze]][ FactorsMat1_numeric[,ze] ]  })
      row.names(FactorsMat1_numeric) <- colnames(FactorsMat1_numeric) <- row.names(FactorsMat1_mapped) <- colnames(FactorsMat1_mapped) <- NULL
    }

    # main cross-validation routine
    optim_max_hajek_list <- sapply(1:(length(LAMBDA_SEQ)+1),function(er){
      list(matrix(NA,nrow = length( pi_init_vec), ncol = nFolds)) })
    if ( length(LAMBDA_SEQ) == 1){
      LAMBDA_selected <- LAMBDA_ <- LAMBDA_SEQ;

      FactorsMat_numeric_IN <- FactorsMat1_numeric <- FactorsMat2_numeric <- FactorsMat_numeric
      Yobs_IN <- Yobs_split1 <- Yobs_split2 <- Yobs
      log_pr_w_IN <- log_pr_w_split1 <- log_pr_w_split2 <- log_pr_w
    }
    LAMBDA_SEQ <- sort(LAMBDA_SEQ, decreasing = T)

    # optimization
    {
        # helper functions
        if(is.null(X)){X<-matrix(rnorm(length(Yobs)*max(2,kClust)),nrow=length(Yobs),ncol=max(2,kClust))}
        FactorsMat_numeric <- sapply(1:ncol(FactorsMat_numeric <- FactorsMat),function(zer){
          match(FactorsMat_numeric[,zer], names(assignmentProbList[[zer]])) })
        FactorsMat_numeric_0Indexed <- FactorsMat_numeric - 1L

        performance_matrix_out <- performance_matrix_in <- matrix(NA, ncol = length(LAMBDA_SEQ), nrow = nFolds)
        holdBasis_traintv <- sample(1:length(split1_indices) %% nFolds+1)
        REGULARIZATION_LAMBDA_SEQ_ORIG <- LAMBDA_SEQ

        if(length(LAMBDA_SEQ) > 1){trainIndicator_pool <- c(1,0); if(nFolds == 1){stop("ERROR: SET nFolds > 1!")}}
        if(length(LAMBDA_SEQ) == 1){warning(sprintf("NO CV SELCTION OF LAMBDA, FORCING LAMBDA = %.5f|",LAMBDA_SEQ)); trainIndicator_pool <- 0}

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
        hypotheticalProbList <- getPiList();names(hypotheticalProbList) <- paste("k",1:kClust,sep="")
        FinalProbList <- hypotheticalProbList
        optim_max_hajek_full <- na.omit(  unlist(getPiList(simplex=F)) )
        ClassProbs <- as.array(  getClassProb(X) )
        performance_mat <- as.data.frame(
                          cbind("lambda" = REGULARIZATION_LAMBDA_SEQ_ORIG,
                                "Q_out" = colMeans(performance_matrix_out),
                                "Qse_out" = apply(performance_matrix_out,2,getSE),
                                "Q_in" = colMeans(performance_matrix_in),
                                "Qse_in" = apply(performance_matrix_in,2,getSE)) )
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
    abline(h=mean(Yobs),lty = 2, col= "gray",lwd=2)
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
      Qhat_tf <- tf$constant( getQ_fxn(  split2_indices  ), tf$float32)
      Qhat <- as.numeric( Qhat_tf )
      Qhat_split1 <- Qhat_all <- Q_interval_split2 <- Q_interval_split1 <- Q_se_exact <- NULL
      Qhat_split2<-list();Qhat_split2$Qest <- as.numeric(Qhat_tf)
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
               "performance_seq_mat" = performance_mat,
               "optim_max_hajek_list" = optim_max_hajek_list,
               "PrXd_vec" = PrXd_vec,
               "X_factorized_complete"=ifelse(("X_factorized_complete" %in% ls()),yes=list(X_factorized_complete),no=list(NULL)),
               "PrXdGivenClust" = PrXdGivenClust_mat,
               "estimationType" = "OneStep",
               "PrXdGivenClust_se" = PrXdGivenClust_se,
               "Output.Description"=c("")) )
  }
}


