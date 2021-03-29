computeQ_conjoint_internal <- function(FactorsMat_internal, Yobs_internal,
                              hypotheticalProbList_internal,
                              assignmentProbList_internal,
                              log_pr_w_internal = NULL,
                              hajek = T,
                              computeLB  = F){
    if(is.null(log_pr_w_internal)){
      log_pr_w_internal = rowSums(log(
        sapply(1:ncol(FactorsMat_internal),function(ze){
        (assignmentProbList_internal[[ze]][ FactorsMat_internal[,ze] ]) })
        ))
    }
    log_pr_w_new <- rowSums(log(
      sapply(1:ncol(FactorsMat_internal),function(ze){
        hypotheticalProbList_internal[[ze]][ FactorsMat_internal[,ze] ]  })
    ))
    my_wts = exp(  log_pr_w_new   - log_pr_w_internal  )
    if(hajek == T){
      my_wts <- my_wts / sum(my_wts);
      if(computeLB == F){ Qest = sum(Yobs_internal * my_wts )  }
      if(computeLB == T){
        minValue <- min(Yobs_internal)
        Yobs_nonZero <- Yobs_internal + (abs(minValue) + 1)*(minValue <= 0)
        #Qest <- exp(1/length(Yobs_internal)*sum(log(Yobs_nonZero)+log(my_wts)))
        Qest <- sum(log(Yobs_nonZero)+log(my_wts))
      }
    }
    if(hajek == F){
      if(computeLB == F){ Qest <- mean(Yobs_internal * my_wts )   }
      if(computeLB == T){
        minValue <- min(Yobs_internal)
        Yobs_nonZero <- Yobs_internal + (abs(minValue) + 1)*(minValue <= 0)
        Qest <- mean(log(Yobs_nonZero)+log(my_wts))
      }
    }

    return(list("Qest"=Qest,
                "Q_wts"=my_wts,
                "log_pr_w_new"=log_pr_w_new,
                "log_pr_w"=log_pr_w_internal))
}
