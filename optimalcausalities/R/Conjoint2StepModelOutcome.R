#' generate_ModelOutcome
#'
#' Implements the organizational record linkage algorithms of Jerzak and Libgober (2021).
#'
#' @usage
#'
#' generate_ModelOutcome(x, Y, by ...)
#'
#' @param x,Y data frames to be merged
#'
#' @return `z` The merged data frame.
#' @export
#'
#' @details `LinkOrgs` automatically processes the name text for each dataset (specified by `by`, `by.x`, and/or `by.Y`. Users may specify the following options:
#'
#' - Set `DistanceMeasure` to control algorithm for computing pairwise string distances. Options include "`osa`", "`jaccard`", "`jw`". See `?stringdist::stringdist` for all options. (Default is "`jaccard`")
#'
#' @examples
#'
#' #Create synthetic data
#' x_orgnames <- c("apple","oracle","enron inc.","mcdonalds corporation")
#' y_orgnames <- c("apple corp","oracle inc","enron","mcdonalds co")
#' x <- data.frame("orgnames_x"=x_orgnames)
#' Y <- data.frame("orgnames_y"=y_orgnames)
#'
#' # Perform merge
#' linkedOrgs <- LinkOrgs(x = x,
#'                        Y = Y,
#'                        by.x = "orgnames_x",
#'                        by.Y = "orgnames_y",
#'                        MaxDist = 0.6)
#'
#' print( linkedOrgs )
#'
#' @export
#'
#' @md
#'

generate_ModelOutcome <- function(){
  # obtain main + interaction info
  {
    # main info
    main_info <- do.call(rbind,sapply(1:length(factor_levels),function(d_){
      list(data.frame("d" = d_, "l" = 1:max(1,factor_levels[d_] - 1 )))}))
    heldout_levels_list <- lapply(factor_levels,function(xer){xer})
    main_info <- cbind(main_info,"d_index"=1:nrow(main_info))

    # interaction info
    main_info_inter <- do.call(rbind,sapply(1:length(factor_levels),function(d_){
      list(data.frame("d" = d_, "l" = 1:max(1,factor_levels[d_]-1 )))}))
    main_info_all <- main_info_inter <- cbind(main_info_inter,"d_full_index"=1:nrow(main_info_inter))

    interaction_helper <- expand.grid(1:nrow(main_info_inter),1:nrow(main_info_inter))
    interaction_helper <- interaction_helper[which(main_info_inter[interaction_helper[,1],1] !=
                                                     main_info_inter[interaction_helper[,2],1]),]
    interaction_helper <- t(   apply(interaction_helper,1,sort) )
    interaction_helper <- interaction_helper[!duplicated(apply(interaction_helper,1,
                                                               function(zer){paste(zer,collapse="_")})),]
    interaction_info <- do.call(rbind, sapply(1:nrow(interaction_helper),function(zer){
      interaction_ <- unlist( c( interaction_helper[zer,] ) )
      comp1 <- main_info_inter[interaction_[[1]],]
      comp2 <- main_info_inter[interaction_[[2]],]
      l_lp_ <- data.frame("d" = comp1$d,
                          "l" = comp1$l,
                          "dl_index" = interaction_[[1]],
                          "dp" = comp2$d,
                          "lp" = comp2$l,
                          "dplp_index" = interaction_[[2]],
                          "inter_index" = zer)
      return( list(l_lp_) )
    }))

    # pre-processing step if seeking to incorporate sparsity
    main_info$d_adj <- main_info$d
    interaction_info$d_adj <- interaction_info$d
    interaction_info$dp_adj <- interaction_info$dp

    # regularization entry
    UsedRegularization <- F
    ok_ <- F;ok_counter <- 0; while(ok_ == F){
      print(sprintf("ok_counter = %s", ok_counter))
      ok_counter <- ok_counter + 1
      forceSparsity <- nrow(W) <= (nrow(interaction_info)+nrow(main_info) + 2)
      if(forceSparsity){
        print("WARNING! More regression parameters than observations, enforcing sparsity...")
        UseRegularization <- T
      }
      if(diff){
        DiffType <- "glm"
        #table(table(pair_id_)); length(unique(pair_id_))
        pair_mat <- do.call(rbind, tapply(1:nrow(full_dat_), pair_id_, c) )
        if(!is.null(competing_candidate_group_variable)){
           pair_mat <- do.call(rbind, tapply(1:nrow(full_dat_), pair_id_, function(zer){
              zer[ order( competing_candidate_group_variable[zer]) ] }) )
        }
        main_dat_use <- main_dat <- apply(main_info,1,function(row_){
          1*(W_[,row_[['d']]] == row_[['l']]) })
        if(ok_counter > 1){
          main_dat_use <- apply(main_info_PreRegularization,1,function(row_){
            1*(W_[,row_[['d']]] == row_[['l']]) })
        }
        interacted_dat <- NULL;if(nrow(interaction_info)>0){
          interacted_dat <- apply(interaction_info,1,function(row_){
            1*(main_dat_use[,row_[["dl_index"]]]) *
              1*(main_dat_use[,row_[["dplp_index"]]]) })
          rm( main_dat_use )
        }
        Y_glm <- Y_[pair_mat[,1]]
        main_dat <- main_dat[pair_mat[,1],] - main_dat[pair_mat[,2],]
        interacted_dat <- interacted_dat[pair_mat[,1],] - interacted_dat[pair_mat[,2],]
        varcov_cluster_variable_glm <- varcov_cluster_variable_[pair_mat[,1]]
        #-mean(varcov_cluster_variable_[pair_mat[,1]] == varcov_cluster_variable_[pair_mat[,2]])
        #table( full_dat_$Party.affiliation[pair_mat[,1]] )
        #table( full_dat_$Party.affiliation[pair_mat[,2]] )
        #table( full_dat_$R_Partisanship[pair_mat[,1]] )
        #table( full_dat_$R_Partisanship[pair_mat[,2]] )
      }

      if(diff == F){
        main_dat <- apply(main_info,1,function(row_){
          1*(W_[,row_[['d']]] == row_[['l']]) })
        interacted_dat <- apply(interaction_info,1,function(row_){
          1*(W_[,row_[['d']]] == row_[['l']]) *
            1*(W_[,row_[['dp']]] == row_[['lp']]) })
        Y_glm <- Y_
        varcov_cluster_variable_glm <- varcov_cluster_variable_
      }
      interacted_dat <- interacted_dat[,indicator_InteractionVariation <- apply(interacted_dat,2,sd)>0]
      interaction_info <- interaction_info[indicator_InteractionVariation,]
      interaction_info$inter_index <- 1:nrow(interaction_info)

      # get adj
      if(ok_counter == 1){
        interaction_info$dl_index_adj <- interaction_info$dl_index
        interaction_info$dplp_index_adj <- interaction_info$dplp_index
        regularization_adjust_hash <- main_info$d
        names(regularization_adjust_hash) <- main_info$d
        regularization_adjust_hash_PreRegularization <- regularization_adjust_hash
      }

      if( UseRegularization == F | ok_counter > 1 ){ ok_ <- T }
      if( UseRegularization == T & ok_counter == 1 ){
        # original keys
        UsedRegularization <- T
        main_info_PreRegularization <- main_info
        interaction_info_PreRegularization <- interaction_info
        {
          library(glinternet)
          InteractionPairs <- t(combn(1:nrow(main_info),m = 2))
          InteractionPairs <- InteractionPairs[main_info$d[ InteractionPairs[,1] ] != main_info$d[ InteractionPairs[,2] ],]
          #sum(duplicated(apply(InteractionPairs,1,function(zer){ paste(zer,collapse = "_") })))
          glinternet_results <- glinternet.cv(X = main_dat,
                                              Y = Y_glm, family = glm_family,
                                              numLevels = rep(1,times = ncol(main_dat)),
                                              interactionPairs = InteractionPairs,
                                              nFolds = 5 )
          keep_OnlyMain <- glinternet_results$activeSet[[1]]$cont
          keep_MainWithInter <- glinternet_results$activeSet[[1]]$contcont
          if(is.null(keep_MainWithInter)){
            glinternet_results <- glinternet(X = main_dat,
                                             Y = Y_glm, family = glm_family,
                                             numLevels = rep(1,times = ncol(main_dat)),
                                             interactionPairs = InteractionPairs,
                                             numToFind = 1L)
            keep_OnlyMain <- glinternet_results$activeSet[[length(glinternet_results$activeSet)]]$cont
            keep_MainWithInter <- glinternet_results$activeSet[[length(glinternet_results$activeSet)]]$contcont
          }
          AllMain <- sort(unique(c(keep_OnlyMain, c(keep_MainWithInter))))

          # main_info <- main_info_PreRegularization
          # interaction_info <- interaction_info_PreRegularization

          # adjust main
          main_info <- main_info[main_info$d %in% main_info$d[AllMain],]
          main_info$d_adj <- cumsum(!duplicated(main_info$d))
          regularization_adjust_hash <- c(main_info$d_adj)
          names(regularization_adjust_hash) <- main_info$d

          # adjust inter
          keep_inter_d <- cbind(main_info_PreRegularization$d[keep_MainWithInter[,1]],
                                main_info_PreRegularization$d[keep_MainWithInter[,2]])
          keep_inter_col <- apply(keep_inter_d,1,function(zer){ paste(sort(zer),collapse="_") })
          interaction_info_col <- apply(cbind(interaction_info$d,interaction_info$dp),1,function(zer){ paste(sort(zer),collapse="_") })
          interaction_info_keep_indices <- which( interaction_info_col %in% keep_inter_col )
          interaction_info <- interaction_info[interaction_info_keep_indices,]
          interaction_info$d_adj <- regularization_adjust_hash[as.character(interaction_info$d)]
          interaction_info$dp_adj <- regularization_adjust_hash[as.character(interaction_info$dp)]

          # get adjustments
          {
            dl_vec <- paste(interaction_info$d,interaction_info$l,sep ="_")
            dplp_vec <- paste(interaction_info$dp,interaction_info$lp,sep ="_")
            base_dl_vec <- paste(main_info$d,main_info$l,sep ="_")
            interaction_info$dl_index_adj <-  (1:length(base_dl_vec))[match(dl_vec,base_dl_vec)]
            interaction_info$dplp_index_adj <- (1:length(base_dl_vec))[match(dplp_vec,base_dl_vec)]
          }
        }
        UseRegularization <- F
      }
    }
  }

  # solve(t(cbind( main_dat, interacted_dat )) %*% cbind( main_dat, interacted_dat ))
  # solve(t(main_dat) %*% main_dat)
  # solve(t(interacted_dat) %*% interacted_dat)
  my_model <- glm(Y_glm ~ cbind( main_dat, interacted_dat ), family = glm_family)
  # summary(  my_model  )
  # sum(is.na(coef(my_model)))
  if(any(is.na(coef(my_model)))){
    browser()
    stop("WARNING: Some coefficients NA... This case hasn't been sufficiently tested!")
    which_na <- which( is.na(coef(my_model)[-1]) ) # minus 1 for intercept
    my_model <- try(glm(Y_glm ~ cbind( main_dat, interacted_dat)[,-which_na], family = glm_family),T)
    if(class(my_model) == "try-error"){browser()}
    Main_na <- which_na[which_na <= ncol(main_dat)]
    Inter_na <- which_na[which_na > ncol(main_dat)] - ncol(main_dat)

    # drop
    interaction_info <- interaction_info[-Inter_na,]
    interaction_info$inter_index <- 1:nrow(interaction_info) # check

    main_info <- main_info[-Main_na,]
    main_info$d_index <- 1:nrow(main_info) # check
  }
  if(!is.null(varcov_cluster_variable)){
    vcov_OutcomeModel <- sandwich::vcovCL(my_model, cluster = varcov_cluster_variable_glm, type = "HC1")
  }
  if(is.null(varcov_cluster_variable)){
    vcov_OutcomeModel <- vcov(my_model)
  }
  model_coef_vec <- coef(my_model)[-1]
  EST_INTERCEPT_tf <- tf$Variable(as.matrix(coef(my_model)[1]),
                                  dtype = tf$float32,trainable = T)

  # get EST COEFFICIENTS_tf
  {
    # main part
    my_mean_main_part <- matrix(0,nrow = nrow(main_info_PreRegularization), ncol = 1)
    my_mean_main_part[main_info$d_index] <- model_coef_vec[1:nrow(main_info)]

    # inter part
    my_mean_inter_part <- matrix(0,nrow = nrow(interaction_info_PreRegularization), ncol = 1)
    interaction_info$dl_dplp_comb <- paste(interaction_info$dl_index,
                                           interaction_info$dplp_index, sep = "_")
    interaction_info_PreRegularization$dl_dplp_comb <- paste(interaction_info_PreRegularization$dl_index,
                                                             interaction_info_PreRegularization$dplp_index, sep = "_")
    interaction_info$IntoPreRegIndex <- match(interaction_info$dl_dplp_comb,
                                              interaction_info_PreRegularization$dl_dplp_comb)
    my_mean_inter_part[interaction_info$IntoPreRegIndex] <- model_coef_vec[-c(1:nrow(main_info))]
    my_mean <- c(my_mean_main_part, my_mean_inter_part)
    EST_COEFFICIENTS_tf <- tf$Variable(as.matrix(my_mean), dtype = tf$float32, trainable = T)
  }

  # vcov adjust
  {
    vcov_OutcomeModel_full <- matrix(0, nrow = length(my_mean)+1, ncol = length(my_mean)+1)
    interaction_info$IntoPreRegIndex_vcov <- 1+interaction_info$IntoPreRegIndex+nrow(main_info_PreRegularization)
    vcov_OutcomeModel_full[c(1,main_info$d_index+1,interaction_info$IntoPreRegIndex_vcov),
                           c(1,main_info$d_index+1,interaction_info$IntoPreRegIndex_vcov)] <- vcov_OutcomeModel
  }

  # reset names
  {
    main_info <- main_info_PreRegularization
    interaction_info <- interaction_info_PreRegularization
    vcov_OutcomeModel <- vcov_OutcomeModel_full
    regularization_adjust_hash <- regularization_adjust_hash_PreRegularization
  }
}
