computeQ_conjoint_internal <- function(FactorsMat, Yobs,
                              hypotheticalProbList,
                              assignmentProbList,
                              log_pr_w = NULL,
                              hajek = T){
    if(is.null(log_pr_w)){
      log_pr_w = rowSums(log(
        sapply(1:ncol(FactorsMat),function(ze){
        (assignmentProbList[[ze]][ FactorsMat[,ze] ]) })
        ))#fast match here
    }
    log_pr_new <- rowSums(log(
      sapply(1:ncol(FactorsMat),function(ze){
        hypotheticalProbList[[ze]][ FactorsMat[,ze] ]  })
    ))
    my_wts = exp(  log_pr_new   - log_pr_w  )
    if(hajek == T){ my_wts <- my_wts / sum(my_wts);Qest = sum(Yobs * my_wts )   }
    if(hajek == F){ Qest <- mean(Yobs * my_wts )   }

    return(list("Qest"=Qest,
                "Qest_se"=NULL,
                "Q_wts"=my_wts,
                "log_pr_w"=log_pr_w))
}
