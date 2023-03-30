#' generate_ModelOutcome_FindIt
#'
#' Implements the organizational record linkage algorithms of Jerzak and Libgober (2021).
#'
#' @usage
#'
#' generate_ModelOutcome_FindIt(x, y, by ...)
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
#'

generate_ModelOutcome_FindIt <- function(){
  {
    if(T == F){
      library(FindIt)
      for(col_ in colnames(w_orig)){ full_dat[,col_] <- as.factor(full_dat[,col_]) }
      CausalANOVA_factors <- colnames(w_orig)
      DiffType <- "ANOVA"
      full_dat$pair_id <- paste(full_dat$respondentIndex, full_dat$task, sep = "_")
      MaxMinIters <- 1:c(1+MaxMin)
      for(MaxMinIter in MaxMinIters){
        if(MaxMin == F){ indi_ <- 1:nrow( full_dat )  }
        if(MaxMin == T){
          W_indi_ast <- (which(full_dat$R_Partisanship == "Democrat"))
          W_indi_dag <- (which(full_dat$R_Partisanship == "Republican"))
          DagProp <- length(W_indi_dag)/(length(W_indi_ast)+length(W_indi_dag))
          indi_ <- ifelse(MaxMinIter == 1, yes = list(W_indi_dag), no = list(W_indi_ast))[[1]]
        }
        my_model <- CausalANOVA(eval(parse(text = sprintf("selected~%s",
                                                          paste(CausalANOVA_factors,collapse="+")))),
                                data = full_dat[indi_,], nway = 2, diff = T,
                                family = glm_family,
                                select.prob = F,
                                #cluster = varcov_cluster_variable[indi_],
                                pair = full_dat$pair_id[indi_], collapse = F)
        renamed_list <- (  sapply(1:length(names_list),function(zer){
          paste(names(names_list)[zer],names_list[[zer]][[1]],sep="") }) )
        main_terms_anova <- unlist(main_terms_anova_list <- my_model$coefs[1:ncol(w_orig)])
        inter_terms_anova <- my_model$coefs[-c(1:ncol(w_orig))]
        if(sum(is.na(unlist(my_model$coefs)))>0){print("NAs in COEF");browser()}
        #inter_terms_anova <- unlist(lapply(inter_terms_anova,function(zer){zer[is.na(zer)] <- 0; (zer)  }),recursive = F)

        # main info
        #setdiff(unlist(renamed_list),names(main_terms_anova))
        main_terms_anova <- main_terms_anova[na.omit( match(unlist(renamed_list),names(main_terms_anova)))]
        main_info_new <- as.data.frame( do.call(rbind,sapply(1:length(main_terms_anova_list),function(zer){
          cbind("d"=zer, "l" = match(renamed_list[[zer]],
                                     names(main_terms_anova_list[[zer]])), "d_adj"=zer)  })))
        main_info_new <- na.omit( main_info_new )
        print("EXTREME WARNING: RE CHECK THESE ORDERS!")
        main_info_new$d_index <- 1:nrow(main_info_new)
        main_info <- main_info_new[,c("d","l","d_index","d_adj")]
        main_info_leftoutLdminus1 <- main_info[which(c(diff(main_info$d),1)==0),]
        main_info_leftoutLdminus1$d_index <- 1:nrow(main_info_leftoutLdminus1)

        # interaction info
        interaction_info_new <- as.data.frame( do.call(rbind,sapply(1:length(inter_terms_anova),function(zer){
          inter_zer <- inter_terms_anova[[zer]]
          inter_zer_names <- do.call(rbind,strsplit(names(inter_zer),split=":"))
          inter_zer_names_d <- apply(inter_zer_names,2,function(z){
            which(unlist(lapply(renamed_list,function(l_){ any(z  %in% l_ ) })))
          })
          tmp1 <- renamed_list[[ inter_zer_names_d[1] ]]
          names(tmp1) <- tmp1; tmp1[] <- 1:length(tmp1)
          tmp2 <- renamed_list[[ inter_zer_names_d[2] ]]
          names(tmp2) <- tmp2; tmp2[] <- 1:length(tmp2)

          main_info$d_adj <- main_info$d
          ret_ <- cbind("d"= (d_ <- inter_zer_names_d[1]),
                        "l" = (l_ <- tmp1[inter_zer_names[,1]]),
                        "dl_index" = sapply(l_,function(l__){which(main_info$d == d_ & main_info$l == l__)}),
                        "dl_index_adj" = sapply(l_,function(l__){which(main_info$d == d_ & main_info$l == l__)}),
                        "dp" = (dp_ <- inter_zer_names_d[2]),
                        "lp" = (lp_ <- tmp2[inter_zer_names[,2]]),
                        "dplp_index" = sapply(lp_,function(lp__){which(main_info$d==dp_ & main_info$l == lp__)}),
                        "dplp_index_adj" = sapply(lp_,function(lp__){which(main_info$d==dp_ & main_info$l == lp__)}),
                        "d_adj" = inter_zer_names_d[1],
                        "dp_adj" = inter_zer_names_d[2] )
          if(any(length(ret_[,"dplp_index"]))==0){browser()}
          return(list(ret_))
        })))
        interaction_info_new$inter_index <- 1:nrow(interaction_info_new)
        interaction_info_new <- as.data.frame(apply(interaction_info_new,2, f2n))
        interaction_info_new <- na.omit( interaction_info_new )
        interaction_info <- interaction_info_new[,tmp_<-c("d", "l", "dl_index",
                                                          "dp", "lp", "dplp_index",
                                                          "inter_index", "d_adj", "dp_adj")]
        vcov_OutcomeModel <- my_model$vcov
        browser()
        vcov_OutcomeModel <- rbind(0,cbind(0,vcov_OutcomeModel))
        print(" CHECK THIS ")
        vcov_OutcomeModel[1,1] <- se(tapply(full_dat$selected,
                                            full_dat$pair_id,function(x){mean(diff(x)<0)}))
        my_mean <- unlist(my_model$coefs)
        EST_INTERCEPT_tf <- tf$Variable(as.matrix(my_model$intercept),
                                        dtype = tf$float32,trainable = T)
        EST_COEFFICIENTS_tf <- tf$Variable(as.matrix(my_mean), dtype=tf$float32, trainable=T)

        ret_chunks <- c("vcov_OutcomeModel","my_mean","EST_INTERCEPT_tf","my_model","EST_COEFFICIENTS_tf", "main_terms_anova","inter_terms_anova")
        if( MaxMinIter < max(MaxMinIters) ){for(chunk_ in ret_chunks){
          print("HERE jack")
          eval(parse(text = sprintf("%s_dag = %s",chunk_,chunk_)))
        }}
      }
    }
  }
}
