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

strategize.plot <- function(hypotheticalProbList=NULL, SEsList = NULL,assignmentProbList=NULL,col.main = "black",zStar = 1, xlim = NULL,
                            ticksType = "assignmentProbs",
                            col_vec = NULL, plotNames = T, plotCIs  = T, widths_vec, heights_vec,mainTitle="",margins_vec=NULL,add = F,pch = 20){
  assignmentProbList_ <- sapply(1:length(assignmentProbList),function(ze){
    names(assignmentProbList[[ze]]) <- gsub(names(assignmentProbList[[ze]]),pattern=names(assignmentProbList)[ze],replace="")
    names(assignmentProbList[[ze]]) <- gsub(names(assignmentProbList[[ze]]),pattern="\\.",replace="")
    assignmentProbList[[ze]]
  })
  ncols <- 3
  nrows <- ceiling(length(assignmentProbList_)/3)
  par(mfrow = c(nrows,ncols))
  for(d_ in 1:length(assignmentProbList_)){
    ordering_d <- order(unlist(lapply(strsplit(names(pd_ <- assignmentProbList_[[d_]]),split=""),function(zer){zer[1]})),
                        decreasing = T)

    if(dataAsset == "ono"){
      if( names(assignmentProbList)[d_] == "Experience.in.public.office"){ordering_d <- c(2,1,4,3) }
      if( names(assignmentProbList)[d_] == "Age"){ordering_d <- rev(ordering_d) }
      if( names(assignmentProbList)[d_] == "Favorability.rating.among.the.public"){ordering_d <- rev(ordering_d) }
    }

    # get pretty names
    prettyFactorNames <- paste(gsub(names(assignmentProbList)[d_],pattern="RefFeat",replace=""),sep="")
    prettyFactorNames[grepl(prettyFactorNames,pattern="Reason")] <- "Immigration Reason"
    prettyFactorNames[grepl(prettyFactorNames,pattern="Trips")] <- "Prior Exposure to US"
    prettyFactorNames[grepl(prettyFactorNames,pattern="Country")] <- "Country of Origin"
    prettyFactorNames <- gsub(prettyFactorNames,pattern="\\.",replace=" ")
    prettyFactorNames[grepl(prettyFactorNames,pattern="Experience in public")] <- "Experience"
    prettyFactorNames[grepl(prettyFactorNames,pattern="Salient personal")] <- "Personality"
    prettyFactorNames[grepl(prettyFactorNames,pattern="Party affiliation")] <- "Party"
    prettyFactorNames[grepl(prettyFactorNames,pattern="Policy area of expertise")] <- "Expertise"
    prettyFactorNames[grepl(prettyFactorNames,pattern="Position on national security")] <- "Security"
    prettyFactorNames[grepl(prettyFactorNames,pattern="Position on immig")] <- "Immigration"
    prettyFactorNames[grepl(prettyFactorNames,pattern="Position on abortion")] <- "Abortion"
    prettyFactorNames[grepl(prettyFactorNames,pattern="Position on government")] <- "Spending"
    prettyFactorNames[grepl(prettyFactorNames,pattern="Favorability rating")] <- "Favorability"
    prettyFactorNames[grepl(prettyFactorNames,pattern="Country of Origin")] <- "Origin"
    prettyFactorNames[grepl(prettyFactorNames,pattern="Immigration Reason")] <- "Reason"
    prettyFactorNames[grepl(prettyFactorNames,pattern="WorkExperience")] <- "Experience"
    prettyFactorNames[grepl(prettyFactorNames,pattern="WorkPlans")] <- "Plans"
    prettyFactorNames[grepl(prettyFactorNames,pattern="Prior Exposure")] <- "Exposure"
    prettyFactorNames[grepl(prettyFactorNames,pattern="Prior Exposure")] <- "Exposure"
    prettyFactorNames[grepl(prettyFactorNames,pattern="Party competition")] <- "Competition"
    prettyFactorNames[grepl(prettyFactorNames,pattern="Party.competition")] <- "Competition"

    # preliminary
    zStar <- 1; yScale <- 0.2
    k_EST <- length(hypotheticalProbList)
    p_d <- assignmentProbList[[d_]][ordering_d]
    ypos_grid <- as.data.frame(  expand.grid("k"=1:k_EST, "l"=1:length(pd_)) )
    ypos_grid$ypos <- ( 1.5*(ypos_grid$l) + (yScale)*(ypos_grid$k - 1)/1)

    # start plot
    par(mar=c(3.5,17,2,1))
    #ylim <- c(0.8,length(pd_)+0.4*sqrt(k_EST))
    ylim <- c(0.8,max(ypos_grid$ypos)+0.20)
    plot(pd_[ordering_d], 1:length(pd_),
         ylim = ylim,
         main = prettyFactorNames,
         cex.main = 2,cex = 0,
         xlim = ifelse(is.null(xlim),yes=list(c(0,1)),no=list(xlim))[[1]],
         yaxt="n",xlab ="",ylab="")

    for(l_ in 1:length(pd_)){
      shadowk_ <- 0
      for(k_ in k_EST:1){
        shadowk_ <- shadowk_ + 1
        pi_kd <- hypotheticalProbList[[k_]][[d_]][ordering_d]
        se_kd <- SEsList[[k_]][[d_]][ordering_d]

        if(is.null(col_vec)){ col_ <- k_ }
        if(!is.null(col_vec)){ col_ <- col_vec[k_] }
        y_loc <- ypos_grid[ypos_grid$l == l_ & ypos_grid$k == shadowk_ , "ypos"]
        if(ticksType == "assignmentProbs"){ points(p_d[l_], y_loc, pch="|",col = "gray",cex=1.5)}
        if(ticksType == "zero"){ points(0, y_loc, pch="|",col = "gray",cex=1.5) }
        if(ticksType == "none"){   }
        points(c( min(xlim[2],pi_kd[l_]+zStar*se_kd[l_]),
                  max(xlim[1],pi_kd[l_]-zStar*se_kd[l_] )),
               c(y_loc,
                 y_loc),lwd=2,type = "l",col = col_)
        points(pi_kd[l_],y_loc,pch=19,col = col_,cex = 1.5)
      }
    }

    # names
    my_names <- names(assignmentProbList_[[d_]][ordering_d])
    my_names <- sapply(my_names,function(zer){
      zer <- gsub(zer,pattern="RefFeatCountry",replace="")
      zer <- gsub(zer,pattern="RefFeatJob",replace="")
      zer <- gsub(zer,pattern="RefFeatEducation",replace="")
      zer <- gsub(zer,pattern="RefFeatReason",replace="") })
    my_names <- gsub(my_names, pattern= "\\_",replace=" \\& ")
    my_names <- gsub(my_names,pattern="Reduce deficit through tax",replace="Tax")
    my_names <- gsub(my_names,pattern="Reduce deficit through spending",replace="Spending")
    my_names <- gsub(my_names,pattern="Opposes giving guest worker",replace="Opposes guest worker")
    my_names <- gsub(my_names,pattern="Favors giving guest worker",replace="Favors guest worker")
    if(plotNames == T & ! add){
      axis(2,
           #at = 1:length(p_d)+((yScale<-0.2)*(k_EST-1)/1)*0.5,
           at = tapply(ypos_grid$ypos, ypos_grid$l,mean),
           labels = my_names, tick = F,
           col = "black",lwd=1,
           las = 2,cex.axis = 1.25)
    }
  }
}

