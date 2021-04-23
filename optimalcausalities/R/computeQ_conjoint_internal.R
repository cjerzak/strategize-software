computeQ_conjoint_internal <- function(FactorsMat_internal,
                              Yobs_internal,
                              FactorsMat_internal_mapped = NULL,
                              hypotheticalProbList_internal,
                              assignmentProbList_internal,
                              log_pr_w_internal = NULL,
                              hajek = T, knownNormalizationFactor = NULL,
                              computeLB  = F){
    if(is.null(log_pr_w_internal)){
      log_pr_w_internal <- log( sapply(1:ncol(FactorsMat_internal),function(ze){
          (assignmentProbList_internal[[ze]][ FactorsMat_internal[,ze] ]) }) )
      if(all(class(log_pr_w_internal) == "numeric")){ log_pr_w_internal <- sum(log_pr_w_internal)}
      if(any(class(log_pr_w_internal) != "numeric")){ log_pr_w_internal = rowsums(log_pr_w_internal)}
    }

    # new probability
    if(!is.null(FactorsMat_internal_mapped)){FactorsMat_internal_mapped[] <- log(unlist(hypotheticalProbList_internal)[FactorsMat_internal_mapped])}
    if(is.null(FactorsMat_internal_mapped)){FactorsMat_internal_mapped <- log( sapply(1:ncol(FactorsMat_internal),function(ze){hypotheticalProbList_internal[[ze]][ FactorsMat_internal[,ze] ]  })  )}
    if(any(class(FactorsMat_internal_mapped) != "numeric")){ log_pr_w_new <- rowsums(FactorsMat_internal_mapped)}
    if(all(class(FactorsMat_internal_mapped) == "numeric")){ log_pr_w_new <- sum(FactorsMat_internal_mapped)}
    my_wts <- exp(  log_pr_w_new   - log_pr_w_internal  )
    sum_raw_wts <- sum( my_wts )
    if(hajek == T){
      if(is.null(knownNormalizationFactor)){  my_wts <- my_wts / sum_raw_wts }
      if(!is.null(knownNormalizationFactor)){  my_wts <- my_wts / knownNormalizationFactor }
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
                "Q_wts_raw_sum" = sum_raw_wts,
                "log_pr_w_new"=log_pr_w_new,
                "log_pr_w"=log_pr_w_internal))
}
