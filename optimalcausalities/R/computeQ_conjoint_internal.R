computeQ_conjoint_internal <- function(FactorsMat_internal, Yobs_internal,
                              hypotheticalProbList_internal,
                              assignmentProbList_internal,
                              log_pr_w_internal = NULL,
                              hajek = T,
                              computeLB  = F){
    if(is.null(log_pr_w)){
      log_pr_w = rowSums(log(
        sapply(1:ncol(FactorsMat),function(ze){
        (assignmentProbList[[ze]][ FactorsMat[,ze] ]) })
        ))
    }
    log_pr_w_new <- rowSums(log(
      sapply(1:ncol(FactorsMat),function(ze){
        hypotheticalProbList[[ze]][ FactorsMat[,ze] ]  })
    ))
    my_wts = exp(  log_pr_w_new   - log_pr_w  )
    if(hajek == T){
      my_wts <- my_wts / sum(my_wts);
      if(computeLB == F){ Qest = sum(Yobs * my_wts )  }
      if(computeLB == T){
        minValue <- min(Yobs)
        Yobs_nonZero <- Yobs + (abs(minValue) + 1)*(minValue <= 0)
        #Qest <- exp(1/length(Yobs)*sum(log(Yobs_nonZero)+log(my_wts)))
        Qest <- sum(log(Yobs_nonZero)+log(my_wts))
      }
    }
    if(hajek == F){
      if(computeLB == F){ Qest <- mean(Yobs * my_wts )   }
      if(computeLB == T){
        minValue <- min(Yobs)
        Yobs_nonZero <- Yobs + (abs(minValue) + 1)*(minValue <= 0)
        Qest <- mean(log(Yobs_nonZero)+log(my_wts))
      }
    }

    return(list("Qest"=Qest,
                "Q_wts"=my_wts,
                "log_pr_w_new"=log_pr_w_new,
                "log_pr_w"=log_pr_w))
}
