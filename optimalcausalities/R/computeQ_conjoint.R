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

computeQ_conjoint <- function(FactorsMat,
                              Yobs,
                              assignmentProbList,
                              hypotheticalProbList = NULL,
                              se_ub = NULL,
                              split1_indices=NULL, split2_indices=NULL,
                              computeThetaSEs = F, openBrowser = F,
                              hajek = T,findMax=T,quiet=T,
                              optimizeLB = T,box_epsilon=0.01){
  FactorsMat_numeric <- sapply(1:ncol(FactorsMat),function(ze){
    match(FactorsMat[,ze], names(assignmentProbList[[ze]]))
  })
  if(!is.null(hypotheticalProbList)){
    Qhat <- computeQ_conjoint_internal(FactorsMat_internal = FactorsMat_numeric,
                                       Yobs_internal=Yobs,
                                       log_pr_w_internal = NULL,
                                       assignmentProbList_internal = assignmentProbList,
                                       hypotheticalProbList_internal = hypotheticalProbList,
                                       hajek = T)
    SE_Q <- computeQse_conjoint(FactorsMat=FactorsMat_numeric,
                                Yobs=Yobs,
                                log_pr_w = NULL,log_treatment_combs = NULL,
                                assignmentProbList=assignmentProbList, returnLog = F,
                                hypotheticalProbList=hypotheticalProbList)
    Q_interval <- c(Qhat$Qest - 1.64*SE_Q,  Qhat$Qest + 1.64*SE_Q)
    RETURN_LIST <-   list("Q_point" = Qhat$Qest,
                          "Q_se" = SE_Q,
                          "Q_interval" = Q_interval,
                          "Q_wts" = Qhat$Q_wts,
                          "log_pr_w_new"=Qhat$log_pr_w_new,
                          "log_pr_w"=Qhat$log_pr_w)
  }

  if(is.null(hypotheticalProbList)){
  seList <- lowerList <- upperList <- NULL
  if(is.null(split1_indices)){
    DENOTE_SPLIT = sample(1:2,length(Yobs),prob=c(1,1),replace=T)
    split1_indices <- which(DENOTE_SPLIT==1)
    split2_indices <- which(DENOTE_SPLIT==2)
  }
    FactorsMat1 <- FactorsMat[split1_indices,];FactorsMat1_numeric <- FactorsMat_numeric[split1_indices,]
    FactorsMat2 <- FactorsMat[split2_indices,];FactorsMat2_numeric <- FactorsMat_numeric[split2_indices,]

    log_treatment_combs  <- sum(log(
      sapply(1:ncol(FactorsMat),function(ze){
        length(assignmentProbList[[ze]]) })
    ))

    # get denominator so we don't need to recompute it
    low_pr_w    <-   as.vector(computeQ_conjoint_internal(FactorsMat_internal = FactorsMat_numeric,
                                  Yobs_internal=Yobs,
                                  hypotheticalProbList_internal = assignmentProbList,
                                  assignmentProbList_internal = assignmentProbList,
                                  hajek = T)$log_pr_w)


    # tricky initializations + simplex forcings
    all_names = unlist(lapply(assignmentProbList,function(ze){names(ze)}))
    theta_init = unlist(lapply(assignmentProbList,function(ze){c(rev(c(compositions::alr(t(rev(ze))))))}))

    log_se_ub <- log(se_ub)
    UB_VEC <- LB_VEC <- unlist(assignmentProbList)
    UB_VEC[] <- 1 - box_epsilon
    LB_VEC[] <- box_epsilon
    LB_VEC <- c(-Inf, LB_VEC)
    UB_VEC <- c(log_se_ub, UB_VEC)

    splitIndices = as.factor(unlist(sapply(1:length(assignmentProbList),function(ze){
      rep(ze,times=length(assignmentProbList[[ze]]))
    })))
    zeros_vec <- rep(0,times=length(unlist(assignmentProbList)));
    nonZero_indices <- lapply(1:length(assignmentProbList),function(ze){!duplicated(rep(ze,times=length(assignmentProbList[[ze]])))})
    nonZero_indices <- unlist(nonZero_indices)
    nonZero_indices <- which(!nonZero_indices)
    vec2list <- function(vec_){
      zeros_vec[nonZero_indices] <- vec_
      return( lapply(split(zeros_vec,f = splitIndices),toSimplex) )
    }
    vec2list_noTransform <- function(vec_){
      return( split(vec_,f = splitIndices))
    }
    #system.time(replicate(1000,vec2list(theta_init)))
    #lapply(assignmentProbList,length)[[1]]
    #vec2list(theta_init)[[1]] - toSimplex(c(0,theta_init[1:6]))
    #sum(abs(unlist(vec2list(theta_init))-unlist(assignmentProbList))) # test: should be close to 0
    if(openBrowser){browser()}

    # COBYLA algorithm
    if(T == F){
        nloptr_sol <- ((cobyla_results <- nloptr::nloptr(x0 = theta_init ,
                                                                          eval_f =  function(theta_){
                                                                              Qhat <- computeQ_conjoint_internal(FactorsMat_internal = FactorsMat1_numeric,
                                                                                                        Yobs_internal = Yobs[split1_indices],
                                                                                                        log_pr_w_internal = low_pr_w[split1_indices],
                                                                                                        assignmentProbList_internal = assignmentProbList,
                                                                                                        hypotheticalProbList_internal = vec2list(theta_),
                                                                                                        computeLB = optimizeLB,
                                                                                                        hajek = T)$Qest; if(findMax == T){Qhat <- -1*Qhat} #remember, solnp minimizes
                                                                              #print( Qhat )
                                                                              return( Qhat )},
                                                                          #opts = list("algorithm"="NLOPT_LN_AUGLAG","ftol_abs" = 0.0001,"local_opts"=list("algorithm"="NLOPT_LN_COBYLA")),
                                                                          opts = list("algorithm"="NLOPT_LN_COBYLA","ftol_abs" = 0.0001),
                                                                          lb = rep(-2,length(theta_init)),
                                                                          ub = rep(2,length(theta_init)),
                                                                          eval_g_ineq = function(theta_){
                                                                            hypoProbsList__ <- vec2list(theta_)
                                                                            log_Qse <- computeQse_conjoint(FactorsMat=FactorsMat1_numeric,
                                                                                                           Yobs=Yobs[split1_indices],
                                                                                                           log_pr_w = low_pr_w[split1_indices],
                                                                                                           returnLog = T,
                                                                                                           assignmentProbList=assignmentProbList,
                                                                                                           log_treatment_combs = log_treatment_combs,
                                                                                                           hypotheticalProbList=hypoProbsList__)
                                                                            #print( log_Qse )
                                                                            constrainThis_ <- c(log_Qse,unlist(hypoProbsList__))
                                                                            lessThan0_vec <- c(LB_VEC - constrainThis_,
                                                                                               constrainThis_ - UB_VEC) #<= 0
                                                                            lessThan0_vec = lessThan0_vec[!is.infinite(lessThan0_vec)]
                                                                            return( lessThan0_vec  )  } )))
        theta_init <- cobyla_results$solution
        cobyla_results$message
      }

    # augmented lagrangian
    {
    myDelta <- 0.001;  myRho <- 1;  tol <- 0.00002;
    optim_max_hajek <- (rsolnp_results <- Rsolnp::solnp(pars = theta_init,
                                        fun =  function(theta_){
                                          Qhat <- computeQ_conjoint_internal(FactorsMat_internal = FactorsMat1_numeric,
                                                            Yobs_internal=Yobs[split1_indices],
                                                            log_pr_w_internal = low_pr_w[split1_indices],
                                                            assignmentProbList_internal = assignmentProbList,
                                                            hypotheticalProbList_internal = vec2list(theta_),
                                                            computeLB = optimizeLB,
                                                            hajek = T)$Qest; if(findMax == T){Qhat <- -1*Qhat} #remember, solnp minimizes
                                          #if(runif(1)<0.1){print( Qhat )}
                                          #print( Qhat )
                                          return( Qhat )
                                        }
                                        ,control=list(rho=myRho,tol=tol, delta = myDelta,trace = !quiet),
                                        ineqfun = function(theta_){
                                              hypoProbsList__ <- vec2list(theta_)
                                              log_Qse <- computeQse_conjoint(FactorsMat=FactorsMat1_numeric,
                                                                          Yobs=Yobs[split1_indices],
                                                                          log_pr_w = low_pr_w[split1_indices],
                                                                          returnLog = T,
                                                                          assignmentProbList=assignmentProbList,
                                                                          log_treatment_combs = log_treatment_combs,
                                                                          hypotheticalProbList = hypoProbsList__)
                                              #if(runif(1)<0.1){print( log_Qse )}
                                              return( c(log_Qse,unlist(hypoProbsList__)) )
                                        },
                                        ineqLB = LB_VEC,ineqUB = UB_VEC))$pars
      if(rsolnp_results$convergence == 0){
        print(sprintf("Successful convergence within feasible region!",rsolnp_results$convergence))
      }
    if(rsolnp_results$convergence != 0){
      print(sprintf("Convergence key: %i, did not converge within feasible region!",rsolnp_results$convergence))
    }
    }
    hypotheticalProbList <- vec2list(optim_max_hajek)
    hypotheticalProbList <- sapply(1:length(hypotheticalProbList),function(ze){
      names(hypotheticalProbList[[ze]]) <- names( assignmentProbList[[ze]] )
      return( hypotheticalProbList[[ze]]  )   })
    names(hypotheticalProbList) <- names( assignmentProbList )

    Qhat <- computeQ_conjoint_internal(FactorsMat_internal = FactorsMat1_numeric,
                        Yobs_internal = Yobs[split1_indices],
                        log_pr_w_internal = low_pr_w[split1_indices],
                        assignmentProbList_internal = assignmentProbList,
                        hypotheticalProbList_internal = hypotheticalProbList, hajek = T)
    #Qhat$Qest
    SE_Q <- computeQse_conjoint(FactorsMat=FactorsMat1_numeric,
                          Yobs=Yobs[split1_indices],
                          log_pr_w = low_pr_w[split1_indices],log_treatment_combs = log_treatment_combs,
                          assignmentProbList=assignmentProbList, returnLog = F,
                          hypotheticalProbList=hypotheticalProbList)
    Qhat_split <- computeQ_conjoint_internal(FactorsMat_internal = FactorsMat2_numeric,
                              Yobs_internal = Yobs[split2_indices],
                              log_pr_w_internal = low_pr_w[split2_indices],
                              assignmentProbList_internal = assignmentProbList,
                              hypotheticalProbList_internal = hypotheticalProbList, hajek = T)
    SE_Q_split <- computeQse_conjoint(FactorsMat=FactorsMat2_numeric,
                               Yobs=Yobs[split2_indices],
                               log_pr_w = low_pr_w[split2_indices],log_treatment_combs = log_treatment_combs,
                               assignmentProbList=assignmentProbList, returnLog = F,
                               hypotheticalProbList=hypotheticalProbList)
    Q_interval <- c(Qhat$Qest - 1.64*SE_Q,  Qhat$Qest + 1.64*SE_Q)
    Q_interval_split <- c(Qhat_split$Qest - 1.64*SE_Q_split, Qhat_split$Qest + 1.64*SE_Q_split)

    if(computeThetaSEs == T){
      #INDICES_mEst <- split2_indices; FactorsMat_ <- FactorsMat2_numeric
      INDICES_mEst <- split1_indices; FactorsMat_ <- FactorsMat1_numeric
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
            mainContrib <- computeQ_conjoint_internal(FactorsMat_internal = FactorsMat_,
                                             Yobs_internal = Yobs[INDICES_mEst],
                                             log_pr_w_internal = low_pr_w[INDICES_mEst],
                                             assignmentProbList_internal = assignmentProbList,
                                             hypotheticalProbList_internal = hypotheticalProbList,
                                             computeLB = optimizeLB,
                                             hajek = T)$Qest
            if(findMax == T){mainContrib <- -1*mainContrib} #remember, we're minimizing the final objective function
            TOTAL_OBJ = mainContrib  +  sum(LagrangianTerm)
            return( TOTAL_OBJ )   }, x = theta)))) ;
          #print(head(my_grad));
          my_grad } ) }}
      mEst_max <- m_estimate(
        estFUN = ex_eeFUN_max, data  = data.frame(),
        compute_roots = F, roots = optim_max_hajek)
      m_mean_max = attributes(mEst_max)$estimates
      m_cov_max  = attributes(mEst_max)$vcov

      transformation_list <- zeros_vec;transformation_list[] <- "x"
      transformation_list[nonZero_indices] <- 1:length(m_mean_max)
      transformation_list <- strsplit(paste(transformation_list,collapse=" "),split= "x")[[1]][-1]
      transformation_list <- sapply(transformation_list,function(ze){
        zee <- strsplit(ze,split = ' ')[[1]]
        zee <- zee[zee!=""]
        my_string <- sapply(zee,function(k__){
        sprintf("~exp(x%s)/(exp(0)+%s)", k__,
                paste( paste(paste("exp(x",zee,sep=""),")",sep=''),collapse="+"))
        })
        return( my_string)
        })
      transformation_list <- unlist(transformation_list)
      transformation_list <- sapply(transformation_list,function(ze){as.formula(ze)})
      transformation_list <- sapply(transformation_list,list)
      names(transformation_list) <- paste("x",1:length(transformation_list),sep="")
      optim_max_SEs_mEst_vec <- optim_max_SEs_mEst <- msm::deltamethod(transformation_list,
                                            mean = m_mean_max,
                                            cov  = m_cov_max)
      optim_max_SEs_mEst_ <- zeros_vec;optim_max_SEs_mEst_[]<-NA
      optim_max_SEs_mEst_[nonZero_indices] <- optim_max_SEs_mEst
      optim_max_SEs_mEst <- optim_max_SEs_mEst_
      ensure0t1 <- function(ze){ ze[ze<0] <- 0; ze[ze>1] <- 1;ze}
      lowerList <- vec2list_noTransform(ensure0t1(unlist(hypotheticalProbList) - 1.68*optim_max_SEs_mEst))
      upperList <- vec2list_noTransform(ensure0t1(unlist(hypotheticalProbList) + 1.68*optim_max_SEs_mEst))
      names(optim_max_SEs_mEst) <- names(unlist(hypotheticalProbList))
      seList <- vec2list_noTransform(optim_max_SEs_mEst)
      names(upperList) <- names(lowerList) <- names(seList) <- names( assignmentProbList )
    }

    RETURN_LIST = list("ThetaStar_point"=hypotheticalProbList,
                              "ThetaStar_se" = seList,
                              "ThetaStar_lb"=lowerList,
                              "ThetaStar_ub"=upperList,
                              "Q_point"=Qhat$Qest,
                              "Q_se" = SE_Q,
                              "Q_interval" = Q_interval,
                              "Q_wts" = Qhat$Q_wts,
                              "Q_point_split"=Qhat_split$Qest,
                              "Q_se_split" = SE_Q_split,
                              "Q_interval_split" = Q_interval_split,
                              "Q_wts_split" = Qhat_split$Q_wts,
                        "Output.Description"=c(""))
  }
  return(RETURN_LIST)
}
