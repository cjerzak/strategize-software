generate_ModelOutcome <- function(){
  # obtain main + interaction info
  {
    # main_info + a_structure
    for(nrp in 1:2){
      main_info <- do.call(rbind,sapply(1:length(factor_levels),function(d_){
        list(data.frame("d" = d_, 
                        "l" = 1:max(1,factor_levels[d_] - 
                                      ifelse(nrp == 1,
                                             yes = 1,
                                             no  = holdout_indicator)) ))}))
      heldout_levels_list <- lapply(factor_levels,function(xer){xer})
      main_info <- cbind(main_info,"d_index"=1:nrow(main_info))
      if(nrp == 1){ a_structure <- main_info }
    }

    if(holdout_indicator == 0){
      a_structure_leftoutLdminus1 <- main_info[which(c(diff(main_info$d),1)==0),]
      a_structure_leftoutLdminus1$d_index <- 1:nrow(a_structure_leftoutLdminus1)
    }

    # interaction info
    main_info_inter <- do.call(rbind,sapply(1:length(factor_levels),function(d_){
      list(data.frame("d" = d_, 
                      "l" = 1:max(1,factor_levels[d_] - holdout_indicator )))}))
    main_info_all <- main_info_inter <- cbind(main_info_inter,"d_full_index"=1:nrow(main_info_inter))

    interaction_info <- data.frame(); if(nrow(main_info_inter) > 1){
      interaction_helper <- expand.grid(1:nrow(main_info_inter),1:nrow(main_info_inter))
      interaction_helper <- interaction_helper[which(main_info_inter[interaction_helper[,1],1] !=
                                                       main_info_inter[interaction_helper[,2],1]),]
      interaction_helper <- t(   apply(interaction_helper,1,sort) )
      interaction_helper <- interaction_helper[!duplicated(apply(interaction_helper,1,
                                                   function(zer){paste(zer,collapse="_")})),]
      interaction_info <- do.call(rbind, sapply(1:nrow(interaction_helper),function(zer){
        interaction_ <- unlist( c( interaction_helper[zer,] ) )
        l_lp_ <- data.frame("d" = (comp1 <- main_info_inter[interaction_[[1]],])$d,
                            "l" = comp1$l,
                            "dl_index" = interaction_[[1]],
                            "dp" = (comp2 <- main_info_inter[interaction_[[2]],])$d,
                            "lp" = comp2$l,
                            "dplp_index" = interaction_[[2]],
                            "inter_index" = zer)
        return( list(l_lp_) )
      }))
      
      # pre-processing step if seeking to incorporate sparsity
      main_info$d_adj <- main_info$d
      interaction_info$d_adj <- interaction_info$d
      interaction_info$dp_adj <- interaction_info$dp
    }
  }

  # regularization 
  {
  UsedRegularization <- FALSE
  ok_ <- F;ok_counter <- 0; while(ok_ == F){
      message(sprintf("ok_counter = %s", ok_counter))
      ok_counter <- ok_counter + 1
      interacted_dat <- data.frame(); 
      
      if( ( nrow(W) <= choose(ncol(W),2) ) ){
        message("WARNING! More regression parameters than observations, enforcing sparsity...")
        use_regularization <- T
      }
      if(diff == T){
        #table(table(pair_id_)); length(unique(pair_id_))
        pair_mat <- do.call(rbind, tapply(1:length(pair_id_), pair_id_, c) )
        
        if( !is.null(competing_group_variable_candidate_) ){
          # sort pair IDs by competing_group_variable_candidate_
           pair_mat <- do.call(rbind, tapply(1:length(pair_id_), pair_id_, function(zer){
              # pair_id_[zer]
              # competing_group_variable_candidate_[zer]
              zer[ order( competing_group_variable_candidate_[zer] ) ]
              # competing_group_variable_candidate_[zer][ order( competing_group_variable_candidate_[zer] ) ]
          }) )
           # competing_group_variable_candidate_[ pair_mat[,1] ]
           # plot(as.factor(competing_group_variable_candidate_[ pair_mat[,1] ]))
           # competing_group_variable_respondent_[ pair_mat[,1] ]
        }
        main_dat_use <- main_dat <- apply(main_info,1,function(row_){
                        1*(W_[,row_[['d']]] == row_[['l']] ) })
        if(ok_counter > 1){
          main_dat_use <- apply(main_info_PreRegularization,1,function(row_){
            1*(W_[,row_[['d']]] == row_[['l']]) })
        }
        if(nrow(interaction_info)>0){
            interacted_dat <- apply(interaction_info,1,function(row_){
              1*(main_dat_use[,row_[["dl_index"]]]) *
                1*(main_dat_use[,row_[["dplp_index"]]]) })
        }

        # table(Y_); table(Y_[pair_mat[,1]])+table(Y_[pair_mat[,2]]) # should match 
        # table(Y_[pair_mat[,1]]); table(Y_[pair_mat[,2]])
        Y_glm <- Y_[pair_mat[,1]]
        main_dat <- main_dat[pair_mat[,1],] - main_dat[pair_mat[,2],]
        if(nrow(interaction_info)>0){
          interacted_dat <- interacted_dat[pair_mat[,1],] - interacted_dat[pair_mat[,2],]
          if(length(interacted_dat) == length(Y_glm)){
            # deal with case where just one thing selected 
            interacted_dat <- as.matrix(interacted_dat)
          }
        }
        varcov_cluster_variable_glm <- varcov_cluster_variable_[pair_mat[,1]]
        #-mean(varcov_cluster_variable_[pair_mat[,1]] == varcov_cluster_variable_[pair_mat[,2]])
        #table( full_dat_$Party.affiliation[pair_mat[,1]] )
        #table( full_dat_$Party.affiliation[pair_mat[,2]] )
        #table( full_dat_$R_Partisanship[pair_mat[,1]] )
        #table( full_dat_$R_Partisanship[pair_mat[,2]] )
      }
      if(diff == F){
        main_dat_use <- main_dat <- apply(main_info,1,function(row_){
          1*(W_[,row_[['d']]] == row_[['l']]) })
        if(ok_counter > 1){
          main_dat_use <- apply(main_info_PreRegularization,1,function(row_){
            1*(W_[,row_[['d']]] == row_[['l']]) })
        }
        if(nrow(interaction_info)>0){
          interacted_dat <- apply(interaction_info,1,function(row_){
            1*(W_[,row_[['d']]] == row_[['l']]) *
              1*(W_[,row_[['dp']]] == row_[['lp']]) })
        }
        Y_glm <- Y_
        varcov_cluster_variable_glm <- varcov_cluster_variable_
      }
      
      if(nrow(interacted_dat)>0){ 
        interacted_dat <- try(as.matrix(interacted_dat[,indicator_InteractionVariation <- apply(as.matrix(interacted_dat), 2, sd)>0]), T)
        if('try-error' %in% class(interacted_dat)){
          stop("Error in interacted_dat <- try(interacted_dat[,indicator_InteractionVariation <- apply(interacted_dat, 2, sd)>0], T)")
        }
        interaction_info <- interaction_info[indicator_InteractionVariation,]
        interaction_info$inter_index <- 1:nrow(interaction_info)
      }

      # get adj
      if(ok_counter == 1){
        interaction_info$dl_index_adj <- interaction_info$dl_index
        interaction_info$dplp_index_adj <- interaction_info$dplp_index
        regularization_adjust_hash <- main_info$d
        names(regularization_adjust_hash) <- main_info$d
        regularization_adjust_hash_PreRegularization <- regularization_adjust_hash

        main_info_PreRegularization <- main_info
        interaction_info_PreRegularization <- interaction_info
      }
      if( use_regularization == FALSE | ok_counter > 1 ){ ok_ <- T }
      if( use_regularization == TRUE & ok_counter == 1 | K > 1 ){
        # original keys
        UsedRegularization <- TRUE
        if(K == 1){
            library(glinternet)
            InteractionPairs <- t(combn(1:nrow(main_info), m = 2))
            InteractionPairs <- InteractionPairs[main_info$d[ InteractionPairs[,1] ] != main_info$d[ InteractionPairs[,2] ],]
            
            message("Starting a glinternet fit...")
            glinternet_results <- glinternet.cv(X = main_dat,
                                                Y = Y_glm, family = glm_family,
                                                numLevels = rep(1,times = ncol(main_dat)),
                                                interactionPairs = InteractionPairs,
                                                nFolds = nFolds_glm )
            
            message("Done with glinternet fit...")
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

            # some debugging checks
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
        }
        if(K > 1){
            W_fh <- W
            colnames(W_fh) <- colnames(w_orig)
            design_fh <- as.data.frame(
              cbind("Yobs" = Y, 
                    "respondent_id" = respondent_id,
                    "respondent_task_id" = respondent_task_id,
                    "profile_order" = profile_order, W_fh, X))
            inter_fh <- t(combn(1:ncol(W_fh),m=2))
            OutcomeFormula_withInter <- as.formula( paste(
              # outcome
              "Yobs", "~",
              # main terms
              paste(colnames(W_fh),collapse = "+"), "+",
              # interactions
              paste(paste(colnames(W_fh)[inter_fh[,1]],
                          colnames(W_fh)[inter_fh[,2]],sep = ":"),collapse="+")) )
            OutcomeFormula_mainOnly <- as.formula( paste(
              # outcome
              "Yobs", "~",
              # main terms
              paste(colnames(W_fh),collapse = "+") ))

            ModeratorFormula <- as.formula(paste("~", 
                                                 paste(paste("`",colnames(X),"`",sep = ""), 
                                                       collapse = "+")))
            rm(W_fh); rm(inter_fh); #rm( interacted_dat ); rm(W_); rm(main_dat);rm(w_orig);rm(X)

            # check to ensure correct data setup (values should read 2)
            # table(table( paste(design_fh$respondent_task_id, design_fh$respondent_id,sep='_') ))
            # table(paste0(respondent_id,respondent_task_id))

            # devtools::install_github('mgoplerud/FactorHet'); install.packages("tgp"); install.packages("mclust")
            my_model <- FactorHet::FactorHet_mbo(
              formula = OutcomeFormula_mainOnly,
              group = as.formula("~ respondent_id"),
              task =  as.formula("~ respondent_task_id"),
              choice_order = as.formula("~ profile_order"),
              moderator = ModeratorFormula,
              design = design_fh,
              mbo_control = FactorHet::FactorHet_mbo_control(iters = 3),
              control = FactorHet::FactorHet_control(beta_method = "cpp"),
              K = K)
            my_mean_full <- my_model$parameters$beta
            my_mean_full <- rbind(my_mean_full, matrix(0,nrow = nrow( interaction_info),ncol=K))

            # define for forward compatability
            # k = 1 for now..
            EST_INTERCEPT_tf <- strenv$jnp$array(t( my_mean_full[1,1] ) )
            EST_COEFFICIENTS_tf <- strenv$jnp$array(as.matrix( my_mean_full[-1,1] ) )
            my_mean <- as.vector( my_mean_full[-1,1] )

            # more definitions
            point_est_predict_clust <- my_model$parameters$phi

            vcov_OutcomeModel <- as.matrix(  my_model$vcov$vcov )
            phi_drop <- which(  gsub(row.names(vcov_OutcomeModel),pattern="phi[1-9]_",replace="") %in%
                colnames( point_est_predict_clust ) )
            vcov_full <- matrix(0,nrow = length(my_mean)+1, ncol = length(my_mean)+1)# plus 1 for intercept
            vcov_OutcomeModel <- vcov_OutcomeModel[-phi_drop,-phi_drop]
            vcov_OutcomeModel_by_k <- sapply(1:K,function(k){
              k_indi <- grep(row.names(vcov_OutcomeModel),
                                pattern  = sprintf("beta%s_",k) )
              vcov_full[1:length(k_indi),1:length(k_indi)] <- vcov_OutcomeModel[k_indi,k_indi]
              return(  list( vcov_full ))
            })
            vcov_OutcomeModel <- vcov_OutcomeModel_by_k[[1]]

            # adjustements for competibility with k=1 case
            # do you need this?
            #AllMain <- 1:nrow( main_info  )
            #main_info$d_adj
            ok_ <- T
        }

        # perform get adjustments
        dl_vec <- paste(interaction_info$d,interaction_info$l,sep ="_")
        dplp_vec <- paste(interaction_info$dp,interaction_info$lp,sep ="_")
        base_dl_vec <- paste(main_info$d,main_info$l,sep ="_")
        interaction_info$dl_index_adj <-  (1:length(base_dl_vec))[match(dl_vec,base_dl_vec)]
        interaction_info$dplp_index_adj <- (1:length(base_dl_vec))[match(dplp_vec,base_dl_vec)]
        use_regularization <- FALSE
      }
    }
  }

  ###################################
  # re-run final outcomes to get covariance 
  ###################################
  if(K == 1){
  # fit outcome model, post regularization
  {
    # solve(t(cbind( main_dat, interacted_dat )) %*% cbind( main_dat, interacted_dat ))
    # solve(t(main_dat) %*% main_dat); solve(t(interacted_dat) %*% interacted_dat)
    #  + 0 + 1 to get the glm to return the var-covar for the intercept
    if(nrow(interacted_dat) == 0){ glm_input <- main_dat } 
    if(nrow(interacted_dat) > 0){ 
      glm_input <- cbind(main_dat, interacted_dat ) 
      if( ncol(glm_input) > 0.5*nrow(glm_input)){
        stop("Too many possible interactions given data size. Set use_regularization = TRUE")
      }
    } 
    my_model <- glm(Y_glm ~ glm_input, family = glm_family)
    if(any(is.na(coef(my_model)))){
      stop("Some coefficients NA... This case hasn't been sufficiently tested!")
      which_na <- which( is.na(coef(my_model)[-1]) ) # minus 1 for intercept
      my_model <- try(glm(Y_glm ~ cbind( main_dat, interacted_dat)[,-which_na], family = glm_family),T)
      Main_na <- which_na[which_na <= ncol(main_dat)]
      Inter_na <- which_na[which_na > ncol(main_dat)] - ncol(main_dat)
  
      # drop
      interaction_info <- interaction_info[-Inter_na,]
      interaction_info$inter_index <- 1:nrow(interaction_info) # check
  
      main_info <- main_info[-Main_na,]
      main_info$d_index <- 1:nrow(main_info) # check
    }
    if(!is.null(varcov_cluster_variable)){
      if(length(unique(varcov_cluster_variable))==1){ stop("Only 1 implied cluster in varcov_cluster_variable -- cannot compute cluster varcov")}
      vcov_OutcomeModel <- sandwich::vcovCL(my_model, cluster = varcov_cluster_variable_glm, type = "HC1")
    }
    if(is.null(varcov_cluster_variable)){
      vcov_OutcomeModel <- vcov(  my_model, complete = T)
    }
    model_coef_vec <- coef(my_model)[-1]
    EST_INTERCEPT_tf <- strenv$jnp$array(as.matrix(coef(my_model)[1]), dtype = strenv$dtj)

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
 
    EST_COEFFICIENTS_tf <- strenv$jnp$array(as.matrix(my_mean), dtype = strenv$dtj)
  }

  # vcov adjust - check this in regularization case!
  vcov_OutcomeModel_full <- matrix(0, nrow = length(my_mean)+1, ncol = length(my_mean)+1)
  interaction_info$IntoPreRegIndex_vcov <- 1+interaction_info$IntoPreRegIndex+nrow(main_info_PreRegularization)
  vcov_OutcomeModel_full[c(1,main_info$d_index+1,interaction_info$IntoPreRegIndex_vcov),
                           c(1,main_info$d_index+1,interaction_info$IntoPreRegIndex_vcov)] <- vcov_OutcomeModel

  # reset names
  main_info <- main_info_PreRegularization
  interaction_info <- interaction_info_PreRegularization
  vcov_OutcomeModel <- vcov_OutcomeModel_full
  regularization_adjust_hash <- regularization_adjust_hash_PreRegularization
  }
}
