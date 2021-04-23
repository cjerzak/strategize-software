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
                              assignmentProbList = NULL,
                              hypotheticalProbList = NULL,
                              se_ub = NULL,
                              forceSEs = F,
                              knownNormalizationFactor = NULL,
                              split1_indices=NULL, split2_indices=NULL,
                              computeThetaSEs = F, openBrowser = F,
                              useHajek = T,findMax=T,quiet=T,
                              uniformInitialization = F,
                              optimizeBound = F,
                              logConstraint = T,
                              warmStart = F,
                              control = list(delta = NULL,
                                             rho = NULL,
                                             tol = NULL,
                                             n.restart = 1,
                                             outer.iter = NULL),
                              box_epsilon=0.01,hypotheticalN=NULL){
  library(Rfast)
  hajekForSEs <- T
  if(is.null(control$outer.iter)){control$outer.iter <- 25}
  FactorsMat_numeric <- sapply(1:ncol(FactorsMat),function(ze){
    match(FactorsMat[,ze], names(assignmentProbList[[ze]]))  })
  if(!is.null(hypotheticalProbList)){
    Qhat <- computeQ_conjoint_internal(FactorsMat_internal = FactorsMat_numeric,
                                       Yobs_internal=Yobs,
                                       log_pr_w_internal = NULL,
                                       knownNormalizationFactor = knownNormalizationFactor,
                                       assignmentProbList_internal = assignmentProbList,
                                       hypotheticalProbList_internal = hypotheticalProbList,
                                       hajek = useHajek)
    SE_Q <- computeQse_conjoint(FactorsMat=FactorsMat_numeric,
                                Yobs=Yobs,hypotheticalN = hypotheticalN,
                                log_pr_w = NULL,log_treatment_combs = NULL,
                                hajek = useHajek,
                                knownNormalizationFactor = knownNormalizationFactor,
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
    split_ <- rep(1,times=length(Yobs))
    split_[sample(1:length(split_), round(length(Yobs)/2))] <- 2
    split1_indices = which(split_ == 1); split2_indices = which(split_ == 2)
  }
    FactorsMat1 <- FactorsMat[split1_indices,];FactorsMat1_numeric <- FactorsMat_numeric[split1_indices,]
    FactorsMat2 <- FactorsMat[split2_indices,];FactorsMat2_numeric <- FactorsMat_numeric[split2_indices,]

    log_treatment_combs  <- sum(log(
      sapply(1:ncol(FactorsMat),function(ze){
        length(assignmentProbList[[ze]]) })
    ))
    treatment_combs <- exp(log_treatment_combs)

    # get denominator so we don't need to recompute it
    low_pr_w    <-   as.vector(computeQ_conjoint_internal(FactorsMat_internal = FactorsMat_numeric,
                                  Yobs_internal=Yobs,
                                  hypotheticalProbList_internal = assignmentProbList,
                                  assignmentProbList_internal = assignmentProbList,
                                  hajek = useHajek)$log_pr_w)

    # global split 1 data
    Yobs_split1 <- Yobs[split1_indices]
    sigma2_hat_split1 <- var( Yobs_split1)
    log_pr_w_split1 <- low_pr_w[split1_indices]
    n_split1 <- length(split1_indices)
    zStar <- 1.67

    # tricky initializations + simplex forcings
    all_names = unlist(lapply(assignmentProbList,function(ze){names(ze)}))
    theta_init = unlist(lapply(assignmentProbList,function(ze){c(rev(c(compositions::alr(t(rev(ze))))))}))
    if(uniformInitialization == T){
      theta_init <- unlist(lapply(assignmentProbList,function(ze){c(rev(c(compositions::alr(t(rep(1/length(ze),times=length(ze)) )))))}))
    }
    names(theta_init) <- NULL

    log_se_ub <- log(se_ub)
    UB_VEC <- LB_VEC <- unlist(assignmentProbList)
    UB_VEC[] <- 1 - box_epsilon
    LB_VEC[] <- box_epsilon
    if(logConstraint){if(!optimizeBound){LB_VEC <- c(-Inf, LB_VEC); UB_VEC <- c(log_se_ub, UB_VEC)};constrainLog = T;print("constraining log(se)")}
    if(!logConstraint){if(!optimizeBound){LB_VEC <- c(-Inf, LB_VEC); UB_VEC <- c(se_ub, UB_VEC)};constrainLog = F;print("constraining se not log(se)")}

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
    vec2list_noTransform0 <- function(vec_){# adds 0 entries
      zeros_vec[nonZero_indices] <- vec_
      return( split(zeros_vec,f = splitIndices) )   }
    vec2list_noTransform <- function(vec_){
      return( split(vec_,f = splitIndices)) }
    #system.time(replicate(1000,vec2list(theta_init)))
    #lapply(assignmentProbList,length)[[1]]
    #vec2list(theta_init)[[1]] - toSimplex(c(0,theta_init[1:6]))
    #sum(abs(unlist(vec2list(theta_init))-unlist(assignmentProbList))) # test: should be close to 0

    # get mapped transformations
    probsIndex_mapped <- unlist(assignmentProbList)
    probsIndex_mapped[] <- 1:length( probsIndex_mapped )
    probsIndex_mapped <- vec2list_noTransform(probsIndex_mapped)
    FactorsMat1_mapped <- sapply(1:ncol(FactorsMat1_numeric),function(ze){
      probsIndex_mapped[[ze]][ FactorsMat1_numeric[,ze] ]  })
    row.names(FactorsMat1_mapped) <- colnames(FactorsMat1_mapped) <- NULL
    row.names(FactorsMat1_numeric) <- colnames(FactorsMat1_numeric) <- NULL


    M_grad_fxn <- function(list_theta){
      M_grad <- unlist( sapply(1:length(list_theta),function(k__){
        theta_forGrad_minusk <- list_theta[-k__]
        theta_forGrad_k <- list_theta[[k__]]

        MaxProdConst_minusk  <- prod(  sapply(1:length(theta_forGrad_minusk),function(raa){
          max(theta_forGrad_minusk[[raa]])   }) )

        MaxProb <- max(theta_forGrad_k);
        MaxProb_whichmax <- which.max(theta_forGrad_k)
        MaxProb_grad <- MaxProdConst_minusk * sapply(2:length(theta_forGrad_k), function(factor_iter){
          if(theta_forGrad_k[factor_iter] != MaxProb){softMaxGrad_ <- -theta_forGrad_k[MaxProb_whichmax]*(theta_forGrad_k[factor_iter]) }
          if(theta_forGrad_k[factor_iter] == MaxProb){softMaxGrad_ <- theta_forGrad_k[factor_iter]*(1-theta_forGrad_k[factor_iter]) }
          softMaxGrad_
        })
        return( MaxProb_grad  )
        })  )
      M_grad <- vec2list_noTransform0(M_grad)
    }

    gradFxn <- function(thetae, dynamicD = T){
        list_thetae <- vec2list(thetae)

        if(dynamicD == F){
          D_value <- theta_forSumWts
        }
        if(dynamicD == T){
          log_pr_new_tmp  <- rowSums(log( sapply(1:ncol(FactorsMat1_numeric),function(ze){
            list_thetae[[ze]][ FactorsMat1_numeric[,ze] ]  })  ))
          my_wts_ = exp(log_pr_new_tmp   - log_pr_w_split1 )
          D_value <-  sum(my_wts_)

          D_grad <- rowSums(  sapply(1:nrow(FactorsMat1_numeric),function(iii){
            DContrib_iii <- unlist( sapply(1:length(list_thetae),function(ze){
              theta_forGrad_minusk <- list_thetae[-ze]
              theta_forGrad_k <- list_thetae[[ze]]
              FactorsMat1_numeric_iii <- FactorsMat1_numeric[iii,]
              FactorsMat_minusk <- FactorsMat1_numeric_iii[-ze]
              FactorsMat_k <- FactorsMat1_numeric_iii[ze]
              prodConst_minusK  <- prod(  sapply(1:length(theta_forGrad_minusk),function(raa){
                theta_forGrad_minusk[[raa]][FactorsMat_minusk[raa]]   }) )

              softMaxGradComp <- sapply(2:length(theta_forGrad_k), function(factor_iter){
                if(FactorsMat_k != factor_iter){softMaxGradComp_ <- -theta_forGrad_k[FactorsMat_k]*(theta_forGrad_k[factor_iter]) }
                if(FactorsMat_k == factor_iter){softMaxGradComp_ <- theta_forGrad_k[factor_iter]*(1-theta_forGrad_k[factor_iter]) }
                softMaxGradComp_
              })

              # q gradient
              D_contrib_grad <-  prodConst_minusK * softMaxGradComp / exp(log_pr_w_split1[iii])
            }))
          }) )
          D_grad <- vec2list_noTransform0(D_grad)
        }

        M_grad <- M_grad_fxn(list_thetae)
        M_value <- prod(sapply(1:length(list_thetae),function(ze){
          max(list_thetae[[ze]]) }))

        GradContrib <- rowSums(  sapply(1:nrow(FactorsMat1_numeric),function(iii){
          # obtain iii's data
          FactorsMat1_numeric_iii <- FactorsMat1_numeric[iii,]
          Y_iii    <- Yobs_split1[iii]
          pr_w_iii <-  exp(log_pr_w_split1[iii])

          unlist( sapply(1:length(list_thetae),function(k_){
            FactorsMat_iii_minusk <- FactorsMat1_numeric_iii[-k_]
            FactorsMat_iii_k      <- FactorsMat1_numeric_iii[k_]

            D_grad_k <- D_grad[[k_]][-1]
            M_grad_k <- M_grad[[k_]][-1]
            theta_forGrad_minusk <- list_thetae[-k_]
            theta_forGrad_k      <- list_thetae[[k_]]

            prodConst_minusk  <- prod(  sapply(1:length(theta_forGrad_minusk),function(raa){
              theta_forGrad_minusk[[raa]][FactorsMat_iii_minusk[raa]]   }) )
            prodConst_        <- prodConst_minusk*theta_forGrad_k[FactorsMat_iii_k]

            softMaxGradComp <- sapply(2:length(theta_forGrad_k), function(factor_iter){
              if(FactorsMat_iii_k != factor_iter){softMaxGradComp_ <- - theta_forGrad_k[FactorsMat_iii_k]*(theta_forGrad_k[factor_iter]) }
              if(FactorsMat_iii_k == factor_iter){softMaxGradComp_ <- theta_forGrad_k[factor_iter]*(1-theta_forGrad_k[factor_iter]) }
              return( softMaxGradComp_ )    })

            # q gradient
            N_q_value <- Y_iii * prodConst_ / pr_w_iii
            N_q_grad  <- Y_iii * prodConst_minusk * softMaxGradComp / pr_w_iii

            justQ <- T
            if(justQ == F){
            # y^2 weighting gradients
            N_v_value <- Y_iii^2 * prodConst_ / pr_w_iii
            V_value   <- N_v_value / D_value
            N_v_grad  <- Y_iii^2 * prodConst_minusk * softMaxGradComp  / pr_w_iii

            # Aggregate
            if(dynamicD == T){
              GRAD_T1   <- (D_grad_k*N_q_value - D_value*N_q_grad) / D_value^2
              GRAD_T2   <- zStar * 1/n_split1^2 * sigma2_hat_split1 * treatment_combs * M_grad_k
              V_grad    <- (D_grad_k*N_v_value - D_value*N_v_grad) / D_value^2 # quotent rule
              GRAD_T3   <- zStar * 1/n_split1 * treatment_combs * (M_grad_k*V_value + M_value*V_grad) # product rule
            }

            if(dynamicD == F){
              GRAD_T1  <- N_q_grad / D_value
              GRAD_T2  <- zStar * 1/n_split1^2 * sigma2_hat_split1*treatment_combs * M_grad_k
              V_grad   <- (D_grad_k*N_v_value - D_value*N_v_grad)/D_value^2
              GRAD_T3  <- zStar*1/n_split1*treatment_combs/D_value*(M_grad_k*N_v_value + M_value*N_v_grad)
            }

            if(!findMax){final_grad <-  GRAD_T1 + GRAD_T2 + GRAD_T3}
            if( findMax){final_grad <-  -1*GRAD_T1 + GRAD_T2 + GRAD_T3}
            }
            #final_grad <-  GRAD_T2 + GRAD_T3

            if(justQ == T){
            GRAD_T1   <- (D_grad_k*N_q_value - D_value*N_q_grad) / D_value^2
            if(findMax == F){ final_grad <- GRAD_T1 }
            if(findMax == T){ final_grad <- -GRAD_T1 }
            }

            return( final_grad )
          }) )
        } ) )
        GradContrib
      }
    if(T == F){
      qFxn <- function(thetaz_){
        list_thetaz <- vec2list(thetaz_)
        Qhat <- computeQ_conjoint_internal(FactorsMat_internal = FactorsMat1_numeric,
                                           Yobs_internal = Yobs_split1,
                                           knownNormalizationFactor = knownNormalizationFactor,
                                           log_pr_w_internal = log_pr_w_split1,
                                           assignmentProbList_internal = assignmentProbList,
                                           #hypotheticalProbList_internal = list_thetaz,
                                           hajek = useHajek)$Qest;
        Qse <- computeQse_conjoint(FactorsMat=FactorsMat1_numeric,
                                   Yobs=Yobs_split1,
                                   log_pr_w=log_pr_w_split1,
                                   returnLog=F,
                                   knownNormalizationFactor = knownNormalizationFactor,
                                   hajek = useHajek,
                                   knownSigma2 = sigma2_hat_split1,
                                   hypotheticalN = hypotheticalN,
                                   assignmentProbList=assignmentProbList,
                                   log_treatment_combs = log_treatment_combs,
                                   hypotheticalProbList=list_thetaz)
        if(findMax == F){minThis_ <-  1*(Qhat + zStar*Qse)} # min ub
        if(findMax == T){minThis_ <- -1*(Qhat - zStar*Qse)} # max lb
        print(c(Qhat,Qse))
        #minThis_ <- -Qhat
        return( minThis_ )}

      if(T == F){
        checkDeriv <- replicate(10,{
          newTheta      <- theta_init+rnorm(length(theta_init),sd=0.1)
          realGrad      <- c(maxLik::numericGradient(f  = qFxn, t0 = newTheta))
          predictedGrad <- gradFxn(newTheta)
          plot(predictedGrad, realGrad);abline(a=0,b=1)
          cor(predictedGrad, realGrad)
        })
      }
      #https://www.systutorials.com/docs/linux/man/3-nlopt/
      optim_max_hajek <- ((rsolnp_results <- nloptr::nloptr(x0 = theta_init ,
                                                            eval_f =  qFxn,
                                                            #eval_grad_f = function(arr){gradFxn(arr,dynamicD=T)},
                                                            #opts = list("algorithm"="NLOPT_G_MLSL_LDS","ftol_abs" = 0.0001,"local_opts"=list("algorithm"="NLOPT_LN_COBYLA")),
                                                            #opts = list("algorithm"="NLOPT_GD_STOGO","ftol_abs" = 0.00001),
                                                            #opts = list("algorithm"="NLOPT_LD_SLSQP","ftol_abs" = 0.00001)
                                                            opts = list("algorithm"="NLOPT_LN_NELDERMEAD",check_derivatives = F,check_derivatives_print = "all","ftol_abs"=0.0000001),
                                                            lb = rep(-4,length(theta_init)),
                                                            ub = rep(4,length(theta_init))
                                                            )))
      optim_max_hajek <- rsolnp_results$solution
      rsolnp_results$pars <- rsolnp_results$solution
      theta_init <- rsolnp_results$solution
      rsolnp_results$convergence <- 0
      print("warning: assuming convergence!")
    }

    if(openBrowser){browser()}

    minThis_fxn <- function(theta_){
      hypoProbsList__ <- vec2list(theta_)
      Qhat_ <- computeQ_conjoint_internal(FactorsMat_internal = FactorsMat1_numeric,
                                          Yobs_internal = Yobs_split1,
                                          #FactorsMat_internal_mapped = FactorsMat1_mapped,
                                          log_pr_w_internal = log_pr_w_split1,
                                          hypotheticalProbList_internal = hypoProbsList__,
                                          knownNormalizationFactor = knownNormalizationFactor,
                                          hajek = useHajek)$Qest;
      if(is.na(Qhat_)){browser()}
      if(!findMax){minThis_ <- Qhat_} #remember, solnp minimizes
      if( findMax){minThis_ <- -1*Qhat_} #remember, solnp minimizes
      if(optimizeBound){
        Qse_ <- computeQse_conjoint(FactorsMat=FactorsMat1_numeric,
                                    Yobs=Yobs_split1,
                                    log_pr_w = log_pr_w_split1,
                                    returnLog = F,
                                    hajek = useHajek,
                                    #FactorsMat_internal_mapped = FactorsMat1_mapped,
                                    knownNormalizationFactor = knownNormalizationFactor,
                                    hypotheticalN = hypotheticalN,
                                    knownSigma2 = sigma2_hat_split1,
                                    assignmentProbList=assignmentProbList,
                                    log_treatment_combs = log_treatment_combs,
                                    hypotheticalProbList = hypoProbsList__)
        if(findMax == T){minThis_ <- (Qhat_ + 1.96*Qse_)} #remember, solnp minimizes
        if(findMax == T){minThis_ <- -1*(Qhat_ - 1.96*Qse_)} #remember, solnp minimizes
      }
      return( minThis_ )
    }

    if(T == F){
      par(mar=c(4,5,1,1))
      plot(-maxLik::numericGradient(minThis_fxn, t0=theta_init),
           gradFxn(theta_init),cex.lab = 2,
         xlab = "Numerical Gradient",ylab = "Analytical Gradient");abline(a=0,b=1)
    }

    if(warmStart){
      #nloptr::nloptr.print.options()
      #https://www.systutorials.com/docs/linux/man/3-nlopt/
      optim_max_hajek <- ((rsolnp_results <- nloptr::nloptr(x0 = theta_init ,
                                                                          eval_f =  minThis_fxn,
                                                                          #eval_grad_f = function(ze){gradFxn(ze,dynamicD=T)},
                                                                          #opts = list("algorithm"="NLOPT_LN_AUGLAG","ftol_abs" = 0.0001,"local_opts"=list("algorithm"="NLOPT_LN_COBYLA")),
                                                                          opts = list("algorithm"="NLOPT_LN_AUGLAG",maxtime = 100,
                                                                                      "local_opts"=list("algorithm"="NLOPT_LN_NELDERMEAD",maxtime = 10)),
                                                                          #opts = list("algorithm"="NLOPT_LN_AUGLAG", maxeval=2000,maxtime = 100, ftol_abs = 0,check_derivatives = T, "local_opts"=list("algorithm"="NLOPT_GD_STOGO",maxeval=2000)),
                                                                          #opts = list("algorithm"="NLOPT_LN_COBYLA","ftol_abs" = 0.0001),
                                                                          #opts = list("algorithm"="NLOPT_LN_NELDERMEAD",check_derivatives = F,check_derivatives_print = "all","ftol_abs"=0.0000001),
                                                                          #lb = rep(-2,length(theta_init)),  ub = rep(2,length(theta_init)),
                                                                          eval_g_ineq = function(theta_){
                                                                            hypoProbsList__ <- vec2list(theta_)
                                                                            Qse_quantity <- computeQse_conjoint(FactorsMat=FactorsMat1_numeric,
                                                                                                           Yobs = Yobs_split1,
                                                                                                           log_pr_w = log_pr_w_split1,
                                                                                                           returnLog = constrainLog,
                                                                                                           knownNormalizationFactor = knownNormalizationFactor,
                                                                                                           hajek = useHajek,
                                                                                                           FactorsMat_internal_mapped = FactorsMat1_mapped,
                                                                                                           knownSigma2 = sigma2_hat_split1,
                                                                                                           hypotheticalN = hypotheticalN,
                                                                                                           assignmentProbList = assignmentProbList,
                                                                                                           log_treatment_combs = log_treatment_combs,
                                                                                                           hypotheticalProbList=hypoProbsList__)
                                                                            constrainThis_ <- c(Qse_quantity,unlist(hypoProbsList__))
                                                                            lessThan0_vec <- c(LB_VEC - constrainThis_,
                                                                                               constrainThis_ - UB_VEC) #<= 0
                                                                            lessThan0_vec = lessThan0_vec[!is.infinite(lessThan0_vec)]
                                                                            return( lessThan0_vec  )  } )))
        soptim_max_hajek <- rsolnp_results$solution
        rsolnp_results$pars <- rsolnp_results$solution
        theta_init <- rsolnp_results$solution
        rsolnp_results$convergence <- 0
        print("warning: assuming convergence!")
      }

    # augmented lagrangian
    #if(optimizer == "augmentedLagrangian"){
    {
    sd_scale <- 0.01
    if(is.null(control$n.restarts)){control$n.restarts<-1;sd_scale <- 0}
    optim_max_hajek <- (rsolnp_results <- Rsolnp::gosolnp(pars = theta_init, n.restarts = control$n.restarts, n.sim = 100,
                                        LB = rep(-5,times=length(theta_init)), UB = rep(5,times=length(theta_init)),
                                        distr = rep(3,times=length(theta_init)),
                                        distr.opt = sapply(theta_init,function(ze){ list(list("mean"=ze,"sd"=sd_scale*abs(ze)))  }),
                                        fun = minThis_fxn,
                                        control=list(rho=control$rho,
                                                     outer.iter=control$outer.iter,
                                                     inner.iter=control$inner.iter,
                                                     tol=control$tol,
                                                     delta = control$delta,
                                                     trace = !quiet),
                                        ineqfun = function(theta_){
                                              hypoProbsList__ <- vec2list(theta_)
                                              if(optimizeBound){
                                                constrainThis_vec_ <- c(unlist(hypoProbsList__))
                                              }
                                              if(!optimizeBound){
                                                Qse_quant <- computeQse_conjoint(FactorsMat = FactorsMat1_numeric,
                                                                          Yobs = Yobs_split1,
                                                                          log_pr_w = log_pr_w_split1,
                                                                          returnLog = constrainLog,
                                                                          hajek = useHajek,
                                                                          #FactorsMat_internal_mapped = FactorsMat1_mapped,
                                                                          knownNormalizationFactor = knownNormalizationFactor,
                                                                          hypotheticalN = hypotheticalN,
                                                                          knownSigma2 = sigma2_hat_split1,
                                                                          assignmentProbList=assignmentProbList,
                                                                          log_treatment_combs = log_treatment_combs,
                                                                          hypotheticalProbList = hypoProbsList__)
                                                constrainThis_vec_ <- c(Qse_quant, unlist(hypoProbsList__))
                                              }
                                              return( constrainThis_vec_ )
                                        },
                                        ineqLB = LB_VEC,  ineqUB = UB_VEC))$pars

      }

    # save results
    hypotheticalProbList <- vec2list(optim_max_hajek)
    hypotheticalProbList <- sapply(1:length(hypotheticalProbList),function(ze){
      names(hypotheticalProbList[[ze]]) <- names( assignmentProbList[[ze]] )
      return( list(hypotheticalProbList[[ze]]  ) )  })
    names(hypotheticalProbList) <- names( assignmentProbList )

    # Compute SEs
    Qhat <- computeQ_conjoint_internal(FactorsMat_internal = FactorsMat1_numeric,
                                       Yobs_internal = Yobs_split1,
                                       knownNormalizationFactor = knownNormalizationFactor,
                                       log_pr_w_internal = log_pr_w_split1,
                                       assignmentProbList_internal = assignmentProbList,
                                       hypotheticalProbList_internal = hypotheticalProbList, hajek = useHajek)
    #Qhat$Qest
    SE_Q <- computeQse_conjoint(FactorsMat=FactorsMat1_numeric,
                                Yobs=Yobs_split1,hypotheticalN = hypotheticalN,
                                knownNormalizationFactor = knownNormalizationFactor,
                                log_pr_w = log_pr_w_split1,log_treatment_combs = log_treatment_combs,
                                assignmentProbList=assignmentProbList, returnLog = F, hajek = useHajek,
                                hypotheticalProbList=hypotheticalProbList)
    Qhat_split <- computeQ_conjoint_internal(FactorsMat_internal = FactorsMat2_numeric,
                                             Yobs_internal = Yobs[split2_indices],
                                             log_pr_w_internal = low_pr_w[split2_indices],
                                             assignmentProbList_internal = assignmentProbList,
                                             knownNormalizationFactor = knownNormalizationFactor,
                                             hypotheticalProbList_internal = hypotheticalProbList, hajek = T)
    SE_Q_split <- computeQse_conjoint(FactorsMat=FactorsMat2_numeric,
                                      Yobs=Yobs[split2_indices],hypotheticalN = hypotheticalN,
                                      knownNormalizationFactor = knownNormalizationFactor,
                                      log_pr_w = low_pr_w[split2_indices],log_treatment_combs = log_treatment_combs,
                                      assignmentProbList=assignmentProbList, returnLog = F, hajek = useHajek,
                                      hypotheticalProbList=hypotheticalProbList)
    Q_interval <- c(Qhat$Qest - 1.64*SE_Q,  Qhat$Qest + 1.64*SE_Q)
    Q_interval_split <- c(Qhat_split$Qest - 1.64*SE_Q_split, Qhat_split$Qest + 1.64*SE_Q_split)

    # experimental check
    optimalityCriterion <- NULL;if(T == F){
    optimalityCriterion <- mean( ((replicate(1000,{
        theta_ <- runif(length(rsolnp_results$pars),-2,2)
        hypo_ <- vec2list(theta_)
        log_pr_w_new <- rowSums(log(
          sapply(1:ncol(FactorsMat1_numeric),function(ze){
            hypo_[[ze]][ FactorsMat1_numeric[,ze] ]  })
        ))
      my_wts = exp(  log_pr_w_new   - log_pr_w_split1  )
      minThis <-  -( sum(Yobs_split1*my_wts) - Qhat$Qest*sum(my_wts))
    }))) >= 0 )
    }

    withinFeasible <- all(unlist(hypotheticalProbList) - UB_VEC[2] <= 0) &
                          all(LB_VEC[2] - unlist(hypotheticalProbList)  <= 0) &
                                        log(SE_Q) <= UB_VEC[1]
    accept <- rsolnp_results$convergence == 0 & withinFeasible
    reject <- rsolnp_results$convergence != 0 | !withinFeasible
    if(forceSEs){reject <- F; accept <- T}
    if(accept){
      print(sprintf("Successful convergence within feasible region!",rsolnp_results$convergence))
    }
    if(reject){
      print(sprintf("Convergence key: %i, did not converge within feasible region!",rsolnp_results$convergence))
    }

    if(computeThetaSEs == T & reject){
        print("warning: no convergence, asymptotic SEs not valid and not reported!")
    }
    if(computeThetaSEs == T & accept){
      #INDICES_mEst <- split2_indices; FactorsMat_ <- FactorsMat2_numeric
      INDICES_mEst <- split1_indices; n_ <- length(INDICES_mEst)
      lagrange_listed    <- vec2list_noTransform( rsolnp_results$lagrange[-1] )
      slack_listed       <- vec2list_noTransform( rsolnp_results$ineqx0[-1] )
      library(geex); ex_eeFUN_max <- function(data){
          Yobs_DATA <- c(data[,1]) #Yobs
          log_pr_w_DATA <- c(data[,2]) #log_pr_w
          FactorsMat_ <- (data[,-c(1:2)])
          function(theta){
            # arrange components of theta
            theta_forYMean <- theta[1]
            theta_forYVar <- theta[2]
            theta_forSumWts <- theta[3]
            theta_forGrad <- theta[-c(1:3)]

            # sum Y mean comp
              sum_Ymean_comp <- sum( Yobs_DATA  -   theta_forYMean )

            # sum Yvar comp
              sum_YVar_comp <- sum( (Yobs_DATA  -   theta_forYMean)^2 - theta_forYVar )

            # sum wts comp
              {
                hypotheticalProbList__ <- vec2list( theta_forGrad )
                Qhat__sumComp <- computeQ_conjoint_internal(FactorsMat_internal = FactorsMat_,
                                                     Yobs_internal = Yobs_DATA,
                                                     knownNormalizationFactor = theta_forSumWts,
                                                     log_pr_w_internal = log_pr_w_DATA,
                                                     assignmentProbList_internal = assignmentProbList,
                                                     hypotheticalProbList_internal = hypotheticalProbList__,
                                                     hajek = hajekForSEs)
                sum_comp <- sum(Qhat__sumComp$Q_wts_raw_sum  -   theta_forSumWts )
              }

            # gradient comp
            theta_forGrad_listed  <- vec2list(theta_forGrad)
            GradContribMain <- unlist( sapply(1:length(theta_forGrad_listed),function(k__){
              # grab data
                theta_forGrad_minusk <- theta_forGrad_listed[-k__]
                theta_forGrad_k      <- theta_forGrad_listed[[k__]]
                FactorsMat_minusk    <- FactorsMat_[,-k__]
                FactorsMat_k         <- FactorsMat_[[k__]]

                # calculate pr_theta constants
                prodConstMinusk_  <- prod(  sapply(1:length(theta_forGrad_minusk),function(raa){
                  theta_forGrad_minusk[[raa]][FactorsMat_minusk[,raa]]   }) )
                prodConst_ <- prodConstMinusk_ * theta_forGrad_k[FactorsMat_k]

                # softmax gradient
                softMaxGrad <- sapply(2:length(theta_forGrad_k), function(factor_iter){
                  if(FactorsMat_k != factor_iter){softMaxGradComp_ <- -theta_forGrad_k[FactorsMat_k]*(theta_forGrad_k[factor_iter]) }
                  if(FactorsMat_k == factor_iter){softMaxGradComp_ <- theta_forGrad_k[factor_iter]*(1-theta_forGrad_k[factor_iter]) }
                  softMaxGradComp_
                })

                # q gradient
                pr_w_DATA <- exp(log_pr_w_DATA)
                Q_contrib_grad <- (Yobs_DATA * prodConstMinusk_ * softMaxGrad / pr_w_DATA) / theta_forSumWts
                if( findMax){Q_contrib_grad <- -1*Q_contrib_grad} #remember,

                #v variance gradients
                WtUnNormedVar_grad <- (Yobs_DATA^2 * prodConstMinusk_ * softMaxGrad / pr_w_DATA)
                WtUnNormedVar <- (Yobs_DATA^2 * prodConst_ / pr_w_DATA )

                {
                  MaxProdConst_minusk  <- prod(  sapply(1:length(theta_forGrad_minusk),function(raa){
                    max(theta_forGrad_minusk[[raa]])   }) )
                  MaxProb_k <- max(theta_forGrad_k);
                  MaxProb <- MaxProdConst_minusk*MaxProb_k
                  MaxProb_whichmax <- which.max(theta_forGrad_k)
                  MaxProb_grad <- MaxProdConst_minusk * sapply(2:length(theta_forGrad_k), function(factor_iter){
                    if(theta_forGrad_k[factor_iter] != MaxProb_k){softMaxGrad_ <- -theta_forGrad_k[MaxProb_whichmax]*(theta_forGrad_k[factor_iter]) }
                    if(theta_forGrad_k[factor_iter] == MaxProb_k){softMaxGrad_ <- theta_forGrad_k[factor_iter]*(1-theta_forGrad_k[factor_iter]) }
                    softMaxGrad_
                  })
                }

                Var_contrib_grad1 <- 1/n_^2 * theta_forYVar * exp(log_treatment_combs) * MaxProb_grad
                Var_contrib_grad2 <- 1/n_ * exp(log_treatment_combs) / theta_forSumWts *
                                    ( MaxProb * WtUnNormedVar_grad +
                                        MaxProb_grad *  WtUnNormedVar)
                lagrangian_contrib_varBound <- rsolnp_results$lagrange[1]*(-1*Var_contrib_grad1 - 1*Var_contrib_grad2)
                lagrangian_contrib_epBound <- 1/n_*( -lagrange_listed[[k__]][1] * softMaxGrad +
                                                       lagrange_listed[[k__]][2] * softMaxGrad)
                # LAMBDA * ( 1-ep - theta_k ) AND  LAMBDA * ( theta_k - ep  )
                final_grad <- Q_contrib_grad + lagrangian_contrib_varBound #+ lagrangian_contrib_epBound
                return( final_grad )
              }) )

            # checks
            if(T == F){
              Q__ <- function(theta___){
                hypotheticalProbList__ <- vec2list( theta___ )
                CM <- computeQ_conjoint_internal(FactorsMat_internal = FactorsMat_,
                                                 Yobs_internal = Yobs_DATA,
                                                 knownNormalizationFactor = theta_forSumWts,
                                                 log_pr_w_internal = log_pr_w_DATA,
                                                 assignmentProbList_internal = assignmentProbList,
                                                 hypotheticalProbList_internal = hypotheticalProbList__,
                                                 hajek = hajekForSEs)$Qest
                CV <- computeQ_conjoint_internal(FactorsMat_internal = FactorsMat_,
                                                 Yobs_internal = Yobs_DATA^2,
                                                 knownNormalizationFactor = theta_forSumWts,
                                                 log_pr_w_internal = log_pr_w_DATA,
                                                 assignmentProbList_internal = assignmentProbList,
                                                 hypotheticalProbList_internal = hypotheticalProbList__,
                                                 hajek = hajekForSEs)$Qest
                maxProb_ <- prod(  unlist(  lapply(hypotheticalProbList__,max) ) )
                Lagrangian_Var <-  rsolnp_results$lagrange[1]*(1/n_*se_ub -
                                    (1/n_*(1/n_*theta_forYVar * exp(log_treatment_combs) * maxProb_ +
                                          exp(log_treatment_combs) * maxProb_* CV ) + rsolnp_results$ineqx0[1])  )
                Lagrangian_Bounds <-   sum(  sapply(1:length(lagrange_listed),function(ze){
                    lagrange_listed[[ze]] * (hypotheticalProbList__[[ze]] - slack_listed[[ze]])
                          }) )
                - CM + Lagrangian_Var# +  Lagrangian_Bounds
              }
              numericalGrad <- maxLik::numericGradient(Q__, theta_forGrad)
              plot(numericalGrad,GradContribMain);abline(a=0,b=1)

              MaxProb_fxn <- function(ze){
                prod(  sapply(ze, function(er){
                  max(  exp(c(0,er))/sum(exp(c(0,er))) ) })) }
              maxLik::numericGradient(MaxProb_fxn,theta_forGrad)
            }

            to0_vec <- c(sum_comp,
                         sum_Ymean_comp,
                         sum_YVar_comp,
                         GradContribMain)
            return( to0_vec )
            #end function(theta)
          }
      }
        mEst_roots <- as.vector(c(mean(Yobs[INDICES_mEst]),
                                  var(Yobs[INDICES_mEst]),
                                  Qhat$Q_wts_raw_sum,
                                  optim_max_hajek))
        mEst_max <- m_estimate(
          estFUN = ex_eeFUN_max,
          data <- as.data.frame( cbind("Yobs" = Yobs[INDICES_mEst],
                                   "low_pr_w" = low_pr_w[INDICES_mEst],
                                   FactorsMat_numeric[INDICES_mEst, ] )),
          compute_roots = F,
          roots = mEst_roots,
          root_control = setup_root_control(start = mEst_roots))

      m_mean_max = attributes(mEst_max)$estimates[-c(1:3)]
      m_cov_max  = attributes(mEst_max)$vcov[-c(1:3),-c(1:3)]
      #attributes(mEst_max)$sandwich_components

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
                              "split1_indices" = split1_indices,
                              "split2_indices" = split2_indices,
                            "optimalityCriterion" = optimalityCriterion,
                       "convergence"= rsolnp_results$convergence,
                        "Output.Description"=c(""))
  }
  return(RETURN_LIST)
}
