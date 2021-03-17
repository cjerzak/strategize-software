#' computeQ_conjoint
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

optimizeQ_conjoint <- function(FactorsMat, Yobs,
                               assignmentProbList, se_ub, INDICES_SPLIT1=NULL, INDICES_SPLIT2=NULL,
                               computeSEs = F,
                               hajek = T,doMax=T,quiet=T){
  if(is.null(INDICES_SPLIT1)){
    DENOTE_SPLIT = sample(1:2,length(Yobs),prob=c(1,1),replace=T)
    INDICES_SPLIT1 <- which(DENOTE_SPLIT==1)
    INDICES_SPLIT2 <- which(DENOTE_SPLIT==2)
  }
    FactorsMat_numeric <- sapply(1:ncol(FactorsMat),function(ze){
      match(FactorsMat[,ze], names(assignmentProbList[[ze]]))
    })
    FactorsMat1 <- FactorsMat[INDICES_SPLIT1,];FactorsMat1_numeric <- FactorsMat_numeric[INDICES_SPLIT1,]
    FactorsMat2 <- FactorsMat[INDICES_SPLIT2,];FactorsMat2_numeric <- FactorsMat_numeric[INDICES_SPLIT2,]

    log_treatment_combs  <- sum(log(
      sapply(1:ncol(FactorsMat),function(ze){
        length(assignmentProbList[[ze]]) })
    ))

    # get denominator so we don't need to recompute it
    low_pr_w    <-   as.vector(computeQ_conjoint(FactorsMat = FactorsMat, Yobs=Yobs,
                                  hypotheticalProbList = assignmentProbList,
                                  assignmentProbList = assignmentProbList,
                                  hajek = T)$log_pr_w)


    # tricky initializations + simplex forcings
    all_names = unlist(lapply(assignmentProbList,function(ze){names(ze)}))
    theta_init = unlist(lapply(assignmentProbList,function(ze){c(rev(c(compositions::alr(t(rev(ze))))))}))
    theta_init_target = unlist(lapply(assignmentProbList,function(ze){c(0,rev(c(compositions::alr(t(rev(ze))))))}))
    add0_after_indices <- as.factor(unlist(sapply(1:length(assignmentProbList),function(ze){
      rep(ze,times=length(assignmentProbList[[ze]])-1)
    })))
    add0_after_indices <- which(!duplicated(add0_after_indices)) - 1
    add0_after_indices <- add0_after_indices[-1]

    log_se_ub <- log(se_ub)
    my_ep = 0.005#
    UB_VEC <- LB_VEC <- unlist(assignmentProbList)
    UB_VEC[] <- 1 - my_ep
    LB_VEC[] <- my_ep
    LB_VEC <- c(-Inf, LB_VEC)
    UB_VEC <- c(log_se_ub, UB_VEC)


    vec_<-theta_init
    splitIndices = as.factor(unlist(sapply(1:length(assignmentProbList),function(ze){
      rep(ze,times=length(assignmentProbList[[ze]]))
    })))
    myRho <- NULL;tol <- 0.001;
    vec2list <- function(vec_){
      vec_a <- rep(vec_, (1:2)[(1:length(vec_) %in% add0_after_indices) + 1])
      vec_a[add0_after_indices + 1:length(add0_after_indices)] <- 0
      vec_a <- c(0,vec_a)
      names(vec_a) <- all_names
      list_ <- split(vec_a,f = splitIndices)
      list_ <- lapply(list_,toSimplex) #bring to simplex
      return(list_)
    }
    vec2list_noTransform <- function(vec_){return( split(vec_,f = splitIndices) )}; #test: sum(abs(unlist(vec2list(theta_init))-unlist(assignmentProbList)))
    optim_max_hajek <- (rsolnp_results <- Rsolnp::solnp(pars = theta_init,
                                        fun =  function(theta_){
                                          Qhat <- computeQ_conjoint(FactorsMat = FactorsMat1_numeric,
                                                            Yobs=Yobs[INDICES_SPLIT1],
                                                            log_pr_w = low_pr_w[INDICES_SPLIT1],
                                                            assignmentProbList = assignmentProbList,
                                                            hypotheticalProbList = vec2list(theta_),
                                                            hajek = T)$Qest; if(doMax == T){Qhat <- -1*Qhat} #remember, solnp minimizes
                                          return( Qhat )
                                        }
                                        ,control=list(rho=myRho,tol=tol,trace = !quiet),
                                        ineqfun = function(theta_){
                                              hypoProbsList__ <- vec2list(theta_)
                                              log_Qse <- computeQse_conjoint(FactorsMat=FactorsMat1_numeric,
                                                                          Yobs=Yobs[INDICES_SPLIT1],
                                                                          log_pr_w = low_pr_w[INDICES_SPLIT1],
                                                                          assignmentProbList=assignmentProbList,
                                                                          log_treatment_combs = log_treatment_combs,
                                                                          hypotheticalProbList=hypoProbsList__)
                                              return( c(log_Qse,unlist(hypoProbsList__)) )
                                        },
                                        ineqLB = LB_VEC,ineqUB = UB_VEC))$pars
    hypotheticalProbList <- vec2list(optim_max_hajek)
    names(hypotheticalProbList) <- names( assignmentProbList )

    Qhat <- computeQ_conjoint(FactorsMat = FactorsMat1,
                      Yobs=Yobs[INDICES_SPLIT1],
                      log_pr_w = low_pr_w[INDICES_SPLIT1],
                      assignmentProbList = assignmentProbList,
                      hypotheticalProbList = hypotheticalProbList, hajek = T)
    Qhat_split <- computeQ_conjoint(FactorsMat = FactorsMat2,
                              Yobs=Yobs[INDICES_SPLIT2],
                              log_pr_w = low_pr_w[INDICES_SPLIT2],
                              assignmentProbList = assignmentProbList,
                              hypotheticalProbList = hypotheticalProbList, hajek = T)
    SE_Q <- computeQse_conjoint(FactorsMat=FactorsMat1,
                        Yobs=Yobs[INDICES_SPLIT1],
                        log_pr_w = low_pr_w[INDICES_SPLIT1],log_treatment_combs = log_treatment_combs,
                        assignmentProbList=assignmentProbList, returnLog = F,
                        hypotheticalProbList=hypotheticalProbList)
    SE_Q_split <- computeQse_conjoint(FactorsMat=FactorsMat2,
                               Yobs=Yobs[INDICES_SPLIT2],
                               log_pr_w = low_pr_w[INDICES_SPLIT2],log_treatment_combs = log_treatment_combs,
                               assignmentProbList=assignmentProbList, returnLog = F,
                               hypotheticalProbList=hypotheticalProbList)
    Q_interval <- c(Qhat$Qest - 1.64*SE_Q,  Qhat$Qest + 1.64*SE_Q)
    Q_interval_split <- c(Qhat_split$Qest - 1.64*Qse_split, Qhat_split$Qest + 1.64*Qse_split)

    if(computeSEs == T){
      #INDICES_mEst <- INDICES_SPLIT2; FactorsMat_ <- FactorsMat2_numeric
      INDICES_mEst <- INDICES_SPLIT1; FactorsMat_ <- FactorsMat1_numeric
      library(geex); ex_eeFUN_max <- function(data){
        function(theta){ with(data, {
          my_grad = ((c(rootSolve::gradient(f = function(x){
            hypotheticalProbList_ <- vec2list(x)
            logSE__ <-  computeQse_conjoint(FactorsMat=FactorsMat_,
                                            Yobs=Yobs[INDICES_mEst],
                                            log_pr_w = low_pr_w[INDICES_mEst],
                                            log_treatment_combs = log_treatment_combs,
                                            assignmentProbList=assignmentProbList, returnLog = F,
                                            hypotheticalProbList=hypotheticalProbList_)
            constrainThese_ <- c(logSE__,unlist(hypotheticalProbList_))

            tmp_mat = cbind(LB_VEC - constrainThese_, constrainThese_ - UB_VEC )
            LowerUpperCloserToViolated = apply(tmp_mat,1,which.max)
            LagrangianTerm <- c(rsolnp_results$lagrange)*
              sapply(1:length(LowerUpperCloserToViolated),function(ze){ tmp_mat[ze,LowerUpperCloserToViolated[ze]] })
            mainContrib <- computeQ_conjoint(FactorsMat = FactorsMat_,
                                             Yobs=Yobs[INDICES_mEst],
                                             log_pr_w = low_pr_w[INDICES_mEst],
                                             assignmentProbList = assignmentProbList,
                                             hypotheticalProbList = hypotheticalProbList, hajek = T)$Qest
            if(doMax == T){mainContrib <- -1*mainContrib} #remember, we're minimizing the final objective function
            TOTAL_OBJ = mainContrib  +  sum(LagrangianTerm)
            return( TOTAL_OBJ )   }, x = theta)))) ;
          print(head(my_grad));my_grad } ) }}
      mEst_max <- m_estimate(
        estFUN = ex_eeFUN_max, data  = data.frame(),
        compute_roots = F, roots = optim_max_hajek)
      m_mean_max = attributes(mEst_max)$estimates
      m_cov_max  = attributes(mEst_max)$vcov

      #save.image("~/Downloads/tmpWorkspace.rdata")

      vec_ <- 1:length(m_mean_max)
      vec_a <- rep(vec_, (1:2)[(1:length(vec_) %in% add0_after_indices) + 1])
      vec_a[add0_after_indices + 1:length(add0_after_indices)] <- "x"
      vec_a <- c("x",vec_a)
      transformation_list <- strsplit(paste(vec_a,collapse=" "),split= "x")[[1]][-1]
      transformation_list <- sapply(transformation_list,function(ze){
        zee <- strsplit(ze,split = ' ')[[1]]
        zee <- zee[zee!=""]
        my_string <- sapply(zee,function(k__){
        sprintf("~exp(x%s)/(exp(0)+%s)",
                k__,
                paste( paste(paste("exp(x",zee,sep=""),")",sep=''),collapse="+"))
        })
        return( my_string)
        })
      transformation_list <- unlist(transformation_list)
      transformation_list = sapply(transformation_list,function(ze){as.formula(ze)})
      transformation_list <- sapply(transformation_list,list)
      names(transformation_list) <- paste("x",1:length(transformation_list),sep="")
      optim_max_SEs_mEst = msm::deltamethod(transformation_list,
                                            mean = m_mean_max,
                                            cov  = m_cov_max)
      plot(optim_max_SEs_mEst,log="y")
      vec_a <- rep(optim_max_SEs_mEst, (1:2)[(1:length(optim_max_SEs_mEst) %in% add0_after_indices) + 1])
      vec_a[add0_after_indices + 1:length(add0_after_indices)] <- NA
      optim_max_SEs_mEst <- c(NA,vec_a)
      ensure0t1 <- function(ze){ ze[ze<0] <- 0; ze[ze>1] <- 1;ze}
      lowerList <- vec2list_noTransform(ensure0t1(unlist(hypotheticalProbList) - 1.68*optim_max_SEs_mEst))
      upperList <- vec2list_noTransform(ensure0t1(unlist(hypotheticalProbList) + 1.68*optim_max_SEs_mEst))
      seList <- vec2list_noTransform(optim_max_SEs_mEst)
      names(upperList) <- names(lowerList) <- names(seList) <- names( assignmentProbList )
    }


    OPTIMALITY_RESULTS = list("OptimalProbabilities"=hypotheticalProbList,
                              "OptimalProbabilities_se" = lowerList,
                              "OptimalProbabilities_lb"=NULL,
                              "OptimalProbabilities_ub"=upperList,
                              "AverageOutcomeAtOptimal"=Qhat$Qest,
                              "AverageOutcomeAtOptimal_split"=Qhat_split$Qest,
                              "Q_interval" = Q_interval,
                              "Q_interval_split" = Q_interval_split,
                              "SE_Q" = Qse,
                              "SE_Q_split" = Qse_split,
                              "Q_wts_raw" = Qhat$Q_wts,
                              "Q_wts_split" = Qhat_split$Q_wts)
    return(OPTIMALITY_RESULTS)

}
