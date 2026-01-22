generate_ModelOutcome <- function(){
  # Initialize vcov_OutcomeModel to ensure it's defined in all code paths
  # (particularly needed when K > 1 AND presaved_outcome_model == TRUE)
  vcov_OutcomeModel <- NULL
  vcov_OutcomeModel_by_k <- NULL
  neural_model_info <- NULL
  file_suffix <- if (!is.null(outcome_model_key)) {
    sprintf("%s_%s_%s", GroupsPool[GroupCounter], Round_, outcome_model_key)
  } else {
    sprintf("%s_%s", GroupsPool[GroupCounter], Round_)
  }
  if (isTRUE(adversarial) && adversarial_model_strategy == "two") {
    file_suffix <- sprintf("%s_two", file_suffix)
  }
  coef_cache_path <- if (isTRUE(adversarial) && adversarial_model_strategy == "two") {
    sprintf("./StrategizeInternals/coef_%s.rds", file_suffix)
  } else {
    NULL
  }

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
      a_structure_leftoutLdminus1 <- main_info[which(c(base::diff(main_info$d),1)==0),]
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
                                                       main_info_inter[interaction_helper[,2],1]), , drop = FALSE]
      if (nrow(interaction_helper) > 0) {
        interaction_helper <- t(   apply(interaction_helper,1,sort) )
        interaction_helper <- interaction_helper[!duplicated(apply(interaction_helper,1,
                                                     function(zer){paste(zer,collapse="_")})), , drop = FALSE]
        interaction_info <- do.call(rbind, sapply(seq_len(nrow(interaction_helper)),function(zer){
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
  }
  if (exists("force_no_interactions", inherits = TRUE) && isTRUE(force_no_interactions)) {
    interaction_info <- interaction_info[0, , drop = FALSE]
  }

  # regularization 
  {
  UsedRegularization <- FALSE
  if (exists("force_no_interactions", inherits = TRUE) && isTRUE(force_no_interactions)) {
    use_regularization <- FALSE
  }
  ok_ <- F;ok_counter <- 0; while(ok_ == F){
      message(sprintf("ok_counter = %s", ok_counter))
      ok_counter <- ok_counter + 1
      interacted_dat <- data.frame(); 
      force_no_intercept <- FALSE
      
      if( ( nrow(W) <= choose(ncol(W),2) ) ){
        message("WARNING! More regression parameters than observations, enforcing sparsity...")
        use_regularization <- T
      }
      
      if(diff == T){
        #table(table(pair_id_)); length(unique(pair_id_))
        pair_indices_list <- tapply(1:length(pair_id_), pair_id_, c)
        pair_sizes <- lengths(pair_indices_list)
        pair_size_ok <- all(pair_sizes == 2L)

        profile_order_present <- !is.null(profile_order_) && length(profile_order_) == length(Y_)

        row_key <- apply(W_, 1, function(row) {
          paste(ifelse(is.na(row), "NA", as.character(row)), collapse = "|")
        })
        row_hash <- vapply(row_key, function(key) {
          ints <- utf8ToInt(key)
          if (!length(ints)) {
            return(0)
          }
          sum(ints * seq_along(ints)) %% 2147483647
        }, numeric(1))

        pair_mat <- do.call(rbind, lapply(pair_indices_list, function(idx){
          order_by_profile <- profile_order_present &&
            length(idx) == 2L &&
            length(unique(profile_order_[idx])) == 2L &&
            !any(is.na(profile_order_[idx]))
          if (!is.null(competing_group_variable_candidate_)) {
            if (order_by_profile) {
              idx[order(competing_group_variable_candidate_[idx],
                        profile_order_[idx],
                        row_hash[idx],
                        idx)]
            } else {
              idx[order(competing_group_variable_candidate_[idx],
                        row_hash[idx],
                        idx)]
            }
          } else if (order_by_profile) {
            idx[order(profile_order_[idx],
                      row_hash[idx],
                      idx)]
          } else {
            idx[order(row_hash[idx], idx)]
          }
        }))
        Y_glm <- Y_[pair_mat[,1]]

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

        # Add stage (primary vs general) indicator and stage x factor interactions
        # when include_stage_interactions is TRUE and we have stage information
        stage_interaction_dat <- data.frame()
        n_stage_main_interactions <- 0L
        n_stage_inter_interactions <- 0L
        stage_keep_main <- NULL
        stage_keep_inter <- NULL
        if (exists("include_stage_interactions", inherits = TRUE) &&
            isTRUE(include_stage_interactions) &&
            isTRUE(adversarial) &&
            exists("adversarial_model_strategy", inherits = TRUE) &&
            adversarial_model_strategy == "two" &&
            exists("competing_group_competition_variable_candidate_", inherits = TRUE) &&
            !is.null(competing_group_competition_variable_candidate_)) {

          # Build stage indicator: 1 = "Different" (general), 0 = "Same" (primary)
          if (diff == TRUE) {
            # For diff case, stage is a property of the pair (both profiles share same stage)
            # Use pair_mat[,1] to get the stage for each pair
            stage_vec <- 1 * (competing_group_competition_variable_candidate_[pair_mat[,1]] == "Different")
          } else {
            # For non-diff case, stage is per observation
            stage_vec <- 1 * (competing_group_competition_variable_candidate_ == "Different")
          }

          # Check if stage has variation (needed for estimation)
          if (sd(stage_vec) > 0) {
            stage_factor_interactions <- sweep(main_dat, 1, stage_vec, `*`)
            stage_inter_sd <- apply(as.matrix(stage_factor_interactions), 2, sd)
            stage_keep_main <- stage_inter_sd > 0
            if (any(stage_keep_main)) {
              stage_factor_interactions <- as.matrix(stage_factor_interactions[, stage_keep_main, drop = FALSE])
              n_stage_main_interactions <- ncol(stage_factor_interactions)
            } else {
              stage_factor_interactions <- NULL
            }

            stage_interaction_interactions <- NULL
            if (ncol(interacted_dat) > 0) {
              stage_interaction_interactions <- sweep(interacted_dat, 1, stage_vec, `*`)
              stage_inter_sd_inter <- apply(as.matrix(stage_interaction_interactions), 2, sd)
              stage_keep_inter <- stage_inter_sd_inter > 0
              if (any(stage_keep_inter)) {
                stage_interaction_interactions <- as.matrix(
                  stage_interaction_interactions[, stage_keep_inter, drop = FALSE]
                )
                n_stage_inter_interactions <- ncol(stage_interaction_interactions)
              } else {
                stage_interaction_interactions <- NULL
              }
            }

            if ((n_stage_main_interactions + n_stage_inter_interactions) > 0) {
              stage_main <- matrix(stage_vec, ncol = 1)
              colnames(stage_main) <- "stage_general"

              stage_interaction_dat <- cbind(
                stage_main,
                if (!is.null(stage_factor_interactions)) stage_factor_interactions,
                if (!is.null(stage_interaction_interactions)) stage_interaction_interactions
              )

              message(sprintf(
                "Added stage indicator + %d stage x factor + %d stage x interaction terms",
                n_stage_main_interactions,
                n_stage_inter_interactions
              ))
            } else {
              message("Stage indicator has no varying interactions - skipping stage interactions")
            }
          } else {
            message("Stage indicator has no variation - skipping stage interactions")
          }
        }

        if (use_regularization) {
          main_sd <- apply(as.matrix(main_dat), 2, sd)
          if (length(main_sd) == 0 || any(!is.finite(main_sd)) || all(main_sd == 0)) {
            use_regularization <- FALSE
          }
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
        if(!presaved_outcome_model){
        if(K == 1){
            # glinternet is in Imports - use :: syntax
            InteractionPairs <- t(utils::combn(1:nrow(main_info), m = 2))
            InteractionPairs <- InteractionPairs[main_info$d[ InteractionPairs[,1] ] != main_info$d[ InteractionPairs[,2] ], , drop = FALSE]
            if (nrow(InteractionPairs) == 0) {
              InteractionPairs <- NULL
            }

            message("Starting a glinternet fit...")
            n_obs_glm <- length(Y_glm)
            nFolds_glm_use <- min(nFolds_glm, floor(n_obs_glm / 2))
            glinternet_results <- tryCatch({
              if (nFolds_glm_use < 2L) {
                glinternet::glinternet(X = main_dat,
                                       Y = Y_glm, family = glm_family,
                                       numLevels = rep(1,times = ncol(main_dat)),
                                       interactionPairs = InteractionPairs)
              } else {
                glinternet::glinternet.cv(X = main_dat,
                                          Y = Y_glm, family = glm_family,
                                          numLevels = rep(1,times = ncol(main_dat)),
                                          interactionPairs = InteractionPairs,
                                          nFolds = nFolds_glm_use )
              }
            }, error = function(e) NULL)

            if (is.null(glinternet_results)) {
              use_regularization <- FALSE
              interaction_info <- interaction_info[0, , drop = FALSE]
              interacted_dat <- interacted_dat[0, , drop = FALSE]
              force_no_interactions <- TRUE
              ok_ <- TRUE
              next
            }
            message("Done with glinternet fit...")
            keep_OnlyMain <- glinternet_results$activeSet[[1]]$cont
            keep_MainWithInter <- glinternet_results$activeSet[[1]]$contcont
            if(is.null(keep_MainWithInter)){
              glinternet_results <- glinternet::glinternet(X = main_dat,
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
            phi_drop <- which(  gsub(row.names(vcov_OutcomeModel),pattern="phi[1-9]_",replacement="") %in%
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

            factorhet_cache <- list(
              my_mean_full = my_mean_full,
              vcov_OutcomeModel_by_k = vcov_OutcomeModel_by_k,
              vcov_OutcomeModel = vcov_OutcomeModel,
              my_model = my_model,
              K = K
            )

            # adjustements for competibility with k=1 case
            # do you need this?
            #AllMain <- 1:nrow( main_info  )
            #main_info$d_adj
            ok_ <- T
        }
        }
        
        if(save_outcome_model){
          dir.create('./StrategizeInternals',showWarnings=FALSE)
          write.csv(main_info, file = sprintf("./StrategizeInternals/main_%s.csv", file_suffix))
          write.csv(interaction_info, file = sprintf("./StrategizeInternals/inter_%s.csv", file_suffix))
          if(K > 1 && !presaved_outcome_model){
            saveRDS(factorhet_cache, file = sprintf("./StrategizeInternals/factorhet_%s.rds", file_suffix))
          }
        }
        if(presaved_outcome_model){
          main_info <- read.csv(file = sprintf("./StrategizeInternals/main_%s.csv", file_suffix))
          interaction_info <- read.csv(file = sprintf("./StrategizeInternals/inter_%s.csv", file_suffix))
          if(K > 1){
            factorhet_cache <- readRDS(file = sprintf("./StrategizeInternals/factorhet_%s.rds", file_suffix))
            my_mean_full <- factorhet_cache$my_mean_full
            vcov_OutcomeModel_by_k <- factorhet_cache$vcov_OutcomeModel_by_k
            vcov_OutcomeModel <- factorhet_cache$vcov_OutcomeModel
            if(is.null(vcov_OutcomeModel) && !is.null(vcov_OutcomeModel_by_k)){
              vcov_OutcomeModel <- vcov_OutcomeModel_by_k[[1]]
            }
            my_model <- factorhet_cache$my_model
            if(is.null(my_model)){ my_model <- NULL }

            EST_INTERCEPT_tf <- strenv$jnp$array(t( my_mean_full[1,1] ) )
            EST_COEFFICIENTS_tf <- strenv$jnp$array(as.matrix( my_mean_full[-1,1] ) )
            my_mean <- as.vector( my_mean_full[-1,1] )
          }
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
    use_coeff_cache <- FALSE
    coeff_cache <- NULL
    if (isTRUE(presaved_outcome_model) && !is.null(coef_cache_path) &&
        file.exists(coef_cache_path)) {
      coeff_cache <- readRDS(coef_cache_path)
      if (!is.null(coeff_cache$coefficients_base) &&
          !is.null(coeff_cache$intercept_base) &&
          !is.null(coeff_cache$vcov_OutcomeModel)) {
        use_coeff_cache <- TRUE
      }
    }

    if (!use_coeff_cache) {
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

        # Add stage interactions to GLM input if available
        # These are: stage main effect + stage x factor interactions
        if (exists("stage_interaction_dat") && NROW(stage_interaction_dat) > 0) {
          glm_input <- cbind(glm_input, stage_interaction_dat)
          n_stage_cols <- ncol(stage_interaction_dat)
          message(sprintf("Including %d stage-related columns in GLM", n_stage_cols))
        } else {
          n_stage_cols <- 0L
        } 
        glm_formula <- if (force_no_intercept) {
          Y_glm ~ 0 + glm_input
        } else {
          Y_glm ~ glm_input
        }
        my_model <- glm(glm_formula, family = glm_family)
        coef_vec <- coef(my_model)
        coef_no_intercept <- if (force_no_intercept) coef_vec else coef_vec[-1]
        if(any(is.na(coef_no_intercept))){
          stop("Some coefficients NA... This case hasn't been sufficiently tested!")
          which_na <- which(is.na(coef_no_intercept))
          glm_refit <- if (force_no_intercept) {
            Y_glm ~ 0 + cbind( main_dat, interacted_dat)[,-which_na]
          } else {
            Y_glm ~ cbind( main_dat, interacted_dat)[,-which_na]
          }
          my_model <- try(glm(glm_refit, family = glm_family), T)
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
        if (force_no_intercept) {
          vcov_OutcomeModel <- rbind(c(0, rep(0, ncol(vcov_OutcomeModel))),
                                     cbind(rep(0, nrow(vcov_OutcomeModel)), vcov_OutcomeModel))
        }
        coef_vec <- coef(my_model)
        if (force_no_intercept) {
          model_coef_vec <- coef_vec
          EST_INTERCEPT_tf <- strenv$jnp$array(0, dtype = strenv$dtj)
        } else {
          model_coef_vec <- coef_vec[-1]
          EST_INTERCEPT_tf <- strenv$jnp$array(as.matrix(coef_vec[1]), dtype = strenv$dtj)
        }

        # main part
        my_mean_main_part <- matrix(0,nrow = nrow(main_info_PreRegularization), ncol = 1)
        my_mean_main_part[main_info$d_index] <- model_coef_vec[1:nrow(main_info)]

        # inter part (factor x factor interactions)
        n_factor_inter <- nrow(interaction_info)
        inter_coef_indices <- if (n_factor_inter > 0) {
          (nrow(main_info) + 1):(nrow(main_info) + n_factor_inter)
        } else {
          integer(0)
        }

        my_mean_inter_part <- matrix(0,nrow = nrow(interaction_info_PreRegularization), ncol = 1)
        interaction_info$dl_dplp_comb <- paste(interaction_info$dl_index,
                                               interaction_info$dplp_index, sep = "_")
        interaction_info_PreRegularization$dl_dplp_comb <- paste(interaction_info_PreRegularization$dl_index,
                                                                 interaction_info_PreRegularization$dplp_index, sep = "_")
        interaction_info$IntoPreRegIndex <- match(interaction_info$dl_dplp_comb,
                                                  interaction_info_PreRegularization$dl_dplp_comb)
        if (n_factor_inter > 0) {
          my_mean_inter_part[interaction_info$IntoPreRegIndex] <- model_coef_vec[inter_coef_indices]
        }
        my_mean <- c(my_mean_main_part, my_mean_inter_part)

        EST_COEFFICIENTS_tf <- strenv$jnp$array(as.matrix(my_mean), dtype = strenv$dtj)

        # Handle stage interaction coefficients for "two" strategy
        # Stage interactions are at the end of the coefficient vector
        stage_intercept_adjustment <- 0
        stage_coef_adjustment_main <- rep(0, nrow(main_info_PreRegularization))
        stage_coef_adjustment_inter <- rep(0, nrow(interaction_info_PreRegularization))
        if (exists("n_stage_cols") && n_stage_cols > 0) {
          # Stage coefficients start after factorxfactor interactions
          stage_coef_start <- nrow(main_info) + n_factor_inter + 1
          stage_coef_end <- length(model_coef_vec)

          if (stage_coef_end >= stage_coef_start) {
            stage_coefs <- model_coef_vec[stage_coef_start:stage_coef_end]

            # First stage coefficient is the stage main effect (intercept shift)
            stage_intercept_adjustment <- stage_coefs[1]

            coef_cursor <- 2
            if (exists("n_stage_main_interactions") && n_stage_main_interactions > 0) {
              stage_main_end <- coef_cursor + n_stage_main_interactions - 1
              stage_factor_coefs <- stage_coefs[coef_cursor:stage_main_end]
              coef_cursor <- stage_main_end + 1

              # Map stage x factor coefficients to main effect positions
              if (exists("stage_keep_main") && length(stage_factor_coefs) == sum(stage_keep_main)) {
                main_indices_with_stage <- which(stage_keep_main)
                stage_coef_adjustment_main[main_info$d_index[main_indices_with_stage]] <- stage_factor_coefs
              }
            }

            if (exists("n_stage_inter_interactions") && n_stage_inter_interactions > 0) {
              stage_inter_end <- coef_cursor + n_stage_inter_interactions - 1
              stage_inter_coefs <- stage_coefs[coef_cursor:stage_inter_end]

              if (exists("stage_keep_inter") && length(stage_inter_coefs) == sum(stage_keep_inter)) {
                inter_indices_with_stage <- which(stage_keep_inter)
                stage_coef_adjustment_inter[
                  interaction_info$IntoPreRegIndex[inter_indices_with_stage]
                ] <- stage_inter_coefs
              }
            }

            message(sprintf("Stage intercept adjustment: %.4f", stage_intercept_adjustment))
            message(sprintf(
              "Stage coefficient adjustments: %d main, %d interaction non-zero",
              sum(stage_coef_adjustment_main != 0),
              sum(stage_coef_adjustment_inter != 0)
            ))
          }
        }

        # Store both base and stage-adjusted coefficients for "two" strategy
        # Base (primary, s=0): EST_INTERCEPT_tf, EST_COEFFICIENTS_tf
        # General (s=1): adjusted intercept + adjusted coefficients
        EST_INTERCEPT_tf_general <- strenv$jnp$array(
          as.matrix(if (force_no_intercept) stage_intercept_adjustment else coef_vec[1] + stage_intercept_adjustment),
          dtype = strenv$dtj
        )

        my_mean_general <- my_mean
        my_mean_general[1:nrow(main_info_PreRegularization)] <- my_mean_main_part + stage_coef_adjustment_main
        if (nrow(interaction_info_PreRegularization) > 0) {
          inter_start <- nrow(main_info_PreRegularization) + 1
          inter_end <- nrow(main_info_PreRegularization) + nrow(interaction_info_PreRegularization)
          my_mean_general[inter_start:inter_end] <- my_mean_inter_part + stage_coef_adjustment_inter
        }
        EST_COEFFICIENTS_tf_general <- strenv$jnp$array(as.matrix(my_mean_general), dtype = strenv$dtj)
      }

      vcov_OutcomeModel_general <- vcov_OutcomeModel

      # Drop stage-related vcov rows/cols: downstream parameters only include base terms.
      if (exists("n_stage_cols") && n_stage_cols > 0) {
        n_base_params <- nrow(main_info) + n_factor_inter
        n_stage_params <- n_stage_cols
        p_base <- 1 + n_base_params
        p_full <- p_base + n_stage_params
        if (nrow(vcov_OutcomeModel) >= p_full) {
          vcov_full <- vcov_OutcomeModel[seq_len(p_full), seq_len(p_full), drop = FALSE]
          V_bb <- vcov_full[seq_len(p_base), seq_len(p_base), drop = FALSE]
          V_bs <- vcov_full[seq_len(p_base), (p_base + 1):p_full, drop = FALSE]
          V_ss <- vcov_full[(p_base + 1):p_full, (p_base + 1):p_full, drop = FALSE]

          A <- matrix(0, nrow = p_base, ncol = n_stage_params)
          A[1, 1] <- 1
          if (exists("stage_keep_main") &&
              length(stage_keep_main) > 0 &&
              exists("n_stage_main_interactions") &&
              n_stage_main_interactions > 0) {
            stage_main_indices <- seq_len(n_stage_main_interactions) + 1
            base_main_indices <- 1 + which(stage_keep_main)
            A[base_main_indices, stage_main_indices] <- 1
          }
          if (exists("stage_keep_inter") &&
              length(stage_keep_inter) > 0 &&
              exists("n_stage_inter_interactions") &&
              n_stage_inter_interactions > 0) {
            stage_inter_indices <- (2 + n_stage_main_interactions):
              (1 + n_stage_main_interactions + n_stage_inter_interactions)
            base_inter_indices <- 1 + nrow(main_info) + which(stage_keep_inter)
            A[base_inter_indices, stage_inter_indices] <- 1
          }

          V_sb <- t(V_bs)
          vcov_OutcomeModel_general <- V_bb +
            A %*% V_sb +
            V_bs %*% t(A) +
            A %*% V_ss %*% t(A)
        } else if (nrow(vcov_OutcomeModel) >= p_base) {
          vcov_OutcomeModel_general <- vcov_OutcomeModel[seq_len(p_base), seq_len(p_base), drop = FALSE]
        }

        keep_idx <- seq_len(p_base)
        vcov_OutcomeModel <- vcov_OutcomeModel[keep_idx, keep_idx, drop = FALSE]
      } else {
        vcov_OutcomeModel_general <- vcov_OutcomeModel
      }

      # vcov adjust - check this in regularization case!
      vcov_OutcomeModel_full <- matrix(0, nrow = length(my_mean)+1, ncol = length(my_mean)+1)
      interaction_info$IntoPreRegIndex_vcov <- 1+interaction_info$IntoPreRegIndex+nrow(main_info_PreRegularization)
      vcov_OutcomeModel_full[c(1,main_info$d_index+1,interaction_info$IntoPreRegIndex_vcov),
                               c(1,main_info$d_index+1,interaction_info$IntoPreRegIndex_vcov)] <- vcov_OutcomeModel
      vcov_OutcomeModel <- vcov_OutcomeModel_full
      vcov_OutcomeModel_general_full <- vcov_OutcomeModel_full
      vcov_OutcomeModel_general_full[c(1,main_info$d_index+1,interaction_info$IntoPreRegIndex_vcov),
                                      c(1,main_info$d_index+1,interaction_info$IntoPreRegIndex_vcov)] <-
        vcov_OutcomeModel_general
      vcov_OutcomeModel_general <- vcov_OutcomeModel_general_full
    } else {
      EST_INTERCEPT_tf <- strenv$jnp$array(as.matrix(coeff_cache$intercept_base), dtype = strenv$dtj)
      EST_COEFFICIENTS_tf <- strenv$jnp$array(as.matrix(coeff_cache$coefficients_base), dtype = strenv$dtj)
      my_mean <- as.vector(coeff_cache$coefficients_base)
      if (!is.null(coeff_cache$intercept_general) &&
          !is.null(coeff_cache$coefficients_general)) {
        EST_INTERCEPT_tf_general <- strenv$jnp$array(as.matrix(coeff_cache$intercept_general), dtype = strenv$dtj)
        EST_COEFFICIENTS_tf_general <- strenv$jnp$array(as.matrix(coeff_cache$coefficients_general), dtype = strenv$dtj)
      } else {
        EST_INTERCEPT_tf_general <- EST_INTERCEPT_tf
        EST_COEFFICIENTS_tf_general <- EST_COEFFICIENTS_tf
      }
      vcov_OutcomeModel <- as.matrix(coeff_cache$vcov_OutcomeModel)
      if (!is.null(coeff_cache$vcov_OutcomeModel_general)) {
        vcov_OutcomeModel_general <- as.matrix(coeff_cache$vcov_OutcomeModel_general)
      } else {
        vcov_OutcomeModel_general <- vcov_OutcomeModel
      }
      my_model <- NULL
    }

    # reset names
    main_info <- main_info_PreRegularization
    interaction_info <- interaction_info_PreRegularization
    regularization_adjust_hash <- regularization_adjust_hash_PreRegularization

    if (!use_coeff_cache && !is.null(coef_cache_path) && isTRUE(save_outcome_model)) {
      dir.create('./StrategizeInternals',showWarnings=FALSE)
      coeff_cache_out <- list(
        intercept_base = as.numeric(strenv$np$array(EST_INTERCEPT_tf)),
        coefficients_base = as.numeric(strenv$np$array(EST_COEFFICIENTS_tf)),
        intercept_general = if (exists("EST_INTERCEPT_tf_general", inherits = TRUE)) {
          as.numeric(strenv$np$array(EST_INTERCEPT_tf_general))
        } else {
          NULL
        },
        coefficients_general = if (exists("EST_COEFFICIENTS_tf_general", inherits = TRUE)) {
          as.numeric(strenv$np$array(EST_COEFFICIENTS_tf_general))
        } else {
          NULL
        },
        vcov_OutcomeModel = vcov_OutcomeModel,
        vcov_OutcomeModel_general = if (exists("vcov_OutcomeModel_general", inherits = TRUE)) {
          vcov_OutcomeModel_general
        } else {
          NULL
        }
      )
      saveRDS(coeff_cache_out, file = coef_cache_path)
    }
  }
}
