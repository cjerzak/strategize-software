getQPiStar_gd <-  function(REGRESSION_PARAMETERS_ast,
                          REGRESSION_PARAMETERS_dag,
                          REGRESSION_PARAMETERS_ast0,
                          REGRESSION_PARAMETERS_dag0,
                          P_VEC_FULL_ast,
                          P_VEC_FULL_dag,
                          SLATE_VEC_ast, 
                          SLATE_VEC_dag,
                          LAMBDA,
                          SEED,
                          functionList,
                          a_i_ast,  # initial value, ast 
                          a_i_dag,  # initial value, dag 
                          functionReturn = T,
                          gd_full_simplex, 
                          quiet = TRUE,
                          optimism = c("none", "ogda", "extragrad", "smp", "rain"),
                          optimism_coef = 1,
                          rain_lambda = 1,
                          rain_gamma = 0.01,
                          rain_L = NULL,
                          rain_eta = 0.001,
                          rain_variant = "alg10_staged",
                          rain_output = "last"
                          ){
  optimism <- match.arg(optimism)
  optimism_coef <- as.numeric(optimism_coef)
  if (optimism != "none" && use_optax) {
    stop("optimism/extra-gradient not supported with optax; set use_optax = FALSE.")
  }
  use_smp <- optimism == "smp"
  use_rain <- optimism == "rain"
  use_extragrad <- optimism %in% c("extragrad", "smp", "rain")
  use_joint_extragrad <- adversarial && use_extragrad && !use_optax
  # throw fxns into env 
  dQ_da_ast <- functionList[[1]]
  dQ_da_dag <- functionList[[2]]
  QFXN <- functionList[[3]]

  REGRESSION_PARAMETERS_ast <- gather_fxn(REGRESSION_PARAMETERS_ast)
  INTERCEPT_ast_ <- REGRESSION_PARAMETERS_ast[[1]]
  COEFFICIENTS_ast_ <- REGRESSION_PARAMETERS_ast[[2]]


  INTERCEPT_dag0_ <- INTERCEPT_ast0_ <- INTERCEPT_dag_ <- INTERCEPT_ast_
  COEFFICIENTS_dag0_ <- COEFFICIENTS_ast0_ <- COEFFICIENTS_dag_ <- COEFFICIENTS_ast_
  if( adversarial ){
    REGRESSION_PARAMETERS_dag <- gather_fxn(REGRESSION_PARAMETERS_dag)
    INTERCEPT_dag_ <- REGRESSION_PARAMETERS_dag[[1]]
    COEFFICIENTS_dag_ <- REGRESSION_PARAMETERS_dag[[2]]
  }
  if(adversarial && isTRUE(use_stage_models)){
    REGRESSION_PARAMETERS_ast0 <- gather_fxn(REGRESSION_PARAMETERS_ast0)
    INTERCEPT_ast0_ <- REGRESSION_PARAMETERS_ast0[[1]]
    COEFFICIENTS_ast0_ <- REGRESSION_PARAMETERS_ast0[[2]]

    REGRESSION_PARAMETERS_dag0 <- gather_fxn(REGRESSION_PARAMETERS_dag0)
    INTERCEPT_dag0_ <- REGRESSION_PARAMETERS_dag0[[1]]
    COEFFICIENTS_dag0_ <- REGRESSION_PARAMETERS_dag0[[2]]
  }

  # gradient descent iterations
  strenv$grad_mag_ast_vec <- strenv$grad_mag_dag_vec <- rep(NA, times = nSGD)
  strenv$loss_ast_vec <- strenv$loss_dag_vec <- rep(NA, times = nSGD)
  strenv$inv_learning_rate_ast_vec <- strenv$inv_learning_rate_dag_vec <- rep(NA, times = nSGD)
  grad_prev_ast <- grad_prev_dag <- NULL
  if (use_joint_extragrad) {
    strenv$extragrad_eval_points <- vector("list", length = nSGD)
  }
  if (use_rain) {
    strenv$rain_lambda_vec <- vector("list", length = nSGD)
    strenv$rain_lambda_s_vec <- vector("list", length = nSGD)
    strenv$rain_lambda_sum_vec <- vector("list", length = nSGD)
    strenv$rain_stage_idx <- rep(NA_integer_, times = nSGD)
    strenv$rain_anchor_bar_norm_ast <- vector("list", length = nSGD)
    if (adversarial) {
      strenv$rain_anchor_bar_norm_dag <- vector("list", length = nSGD)
    } else {
      strenv$rain_anchor_bar_norm_dag <- NULL
    }
  }
  if (use_smp) {
    smp_sum_weighted_ast <- strenv$jnp$multiply(a_i_ast, strenv$jnp$array(0., strenv$dtj))
    smp_sum_gamma_ast <- strenv$jnp$array(0., strenv$dtj)
    smp_gamma_ast_vec <- vector("list", length = nSGD)
    if (adversarial) {
      smp_sum_weighted_dag <- strenv$jnp$multiply(a_i_dag, strenv$jnp$array(0., strenv$dtj))
      smp_sum_gamma_dag <- strenv$jnp$array(0., strenv$dtj)
      smp_gamma_dag_vec <- vector("list", length = nSGD)
    }
  }
  goOn <- F; i<-0
  INIT_MIN_GRAD_ACCUM <- strenv$jnp$array(0.1)
  if (use_rain) {
    # Stage-wise RAIN (Algorithm 10): recursive anchoring with stage-constant SEG.
    rain_variant <- match.arg(rain_variant, c("alg10_staged", "alg9_single_loop"))
    rain_output <- match.arg(rain_output, c("uniform_half", "last"))
    if (!identical(rain_variant, "alg10_staged")) {
      stop("RAIN variant 'alg9_single_loop' is not implemented yet; use 'alg10_staged'.")
    }

    if (is.null(rain_gamma)) {
      rain_gamma <- 0
    }
    rain_gamma <- as.numeric(rain_gamma)
    if (length(rain_gamma) != 1 || !is.finite(rain_gamma) || rain_gamma < 0) {
      rain_gamma <- 0
    }

    if (is.null(rain_lambda)) {
      rain_lambda <- 0
    }
    rain_lambda <- as.numeric(rain_lambda)
    if (length(rain_lambda) != 1 || !is.finite(rain_lambda) || rain_lambda < 0) {
      rain_lambda <- 0
    }

    rain_L_val <- NULL
    if (!is.null(rain_L)) {
      rain_L_val <- as.numeric(rain_L)
      if (length(rain_L_val) != 1 || !is.finite(rain_L_val) || rain_L_val <= 0) {
        rain_L_val <- NULL
      }
    }

    if (is.null(rain_eta)) {
      if (!is.null(rain_L_val)) {
        rain_eta <- 1 / (8 * rain_L_val)
      } else {
        rain_eta <- as.numeric(learning_rate_max)
      }
    } else {
      rain_eta <- as.numeric(rain_eta)
    }
    if (length(rain_eta) != 1 || !is.finite(rain_eta) || rain_eta <= 0) {
      rain_eta <- 1e-8
    }

    # In the theory, eta = 1/(8L) and stages run for T_s = 16L/lambda_s iterations.
    if (!is.null(rain_L_val)) {
      L_est <- rain_L_val
    } else {
      L_est <- 1 / (8 * rain_eta)
    }
    if (!is.finite(L_est) || L_est <= 0) {
      L_est <- 1 / (8 * 1e-8)
    }

    eta_t <- strenv$jnp$array(rain_eta, strenv$dtj)
    inv_lr_val <- strenv$jnp$reciprocal(eta_t)

    Lambda_sum <- strenv$jnp$array(0., strenv$dtj)
    B_ast <- strenv$jnp$multiply(a_i_ast, strenv$jnp$array(0., strenv$dtj))
    if (adversarial) {
      B_dag <- strenv$jnp$multiply(a_i_dag, strenv$jnp$array(0., strenv$dtj))
    } else {
      B_dag <- NULL
    }

    rain_log_step <- function(step_idx) {
      if (step_idx < 5 || step_idx %in% unique(ceiling(c(0.25, 0.5, 0.75, 1) * nSGD))) {
        message(sprintf("SGD Iteration: %s of %s", step_idx, nSGD))
      }
    }

    stage_idx <- 0L
    stage_cap <- Inf
    if (!is.null(rain_L_val) && rain_gamma > 0 && rain_lambda > 0) {
      ratio_val <- rain_L_val / rain_lambda
      if (is.finite(ratio_val) && ratio_val > 0) {
        stage_cap <- ceiling(log(ratio_val) / log(1 + rain_gamma))
        if (!is.finite(stage_cap) || stage_cap < 1) {
          stage_cap <- 1
        }
      }
    }

    lambda_stage_num <- rain_gamma * rain_lambda
    if (!is.finite(lambda_stage_num) || lambda_stage_num < 0) {
      lambda_stage_num <- 0
    }

    use_uniform_half <- identical(rain_output, "uniform_half")

    while (i < nSGD && stage_idx < stage_cap) {
      z_s_ast_start <- a_i_ast
      if (adversarial) {
        z_s_dag_start <- a_i_dag
      }

      Lambda_sum_num <- as.numeric(strenv$np$array(Lambda_sum))
      has_anchors <- is.finite(Lambda_sum_num) && Lambda_sum_num > 0
      if (has_anchors) {
        zbar_ast <- strenv$jnp$divide(B_ast, Lambda_sum)
        if (adversarial) {
          zbar_dag <- strenv$jnp$divide(B_dag, Lambda_sum)
        }
      } else {
        zbar_ast <- NULL
        zbar_dag <- NULL
      }

      lambda_use_num <- lambda_stage_num
      if (!is.finite(lambda_use_num) || lambda_use_num < 0) {
        lambda_use_num <- 0
      }
      if (lambda_use_num > 0) {
        stage_len_num <- ceiling((16 * L_est) / lambda_use_num)
        if (!is.finite(stage_len_num) || stage_len_num < 1) {
          stage_len_num <- 1
        }
      } else {
        stage_len_num <- nSGD - i
      }
      stage_len <- as.integer(min(stage_len_num, nSGD - i))
      if (!is.finite(stage_len) || stage_len < 1) {
        stage_len <- as.integer(min(1, nSGD - i))
      }

      lambda_use_tf <- strenv$jnp$array(lambda_use_num, strenv$dtj)

      half_count <- 0L
      half_sample_ast <- NULL
      half_sample_dag <- NULL

      for (t in seq_len(stage_len)) {
        i <- i + 1
        rain_log_step(i)

        SEED <- strenv$jax$random$split(SEED)[[1L]]
        seed_base <- SEED

        if (adversarial) {
          base_grad_dag <- dQ_da_dag(
            a_i_ast, a_i_dag,
            INTERCEPT_ast_,  COEFFICIENTS_ast_,
            INTERCEPT_dag_,  COEFFICIENTS_dag_,
            INTERCEPT_ast0_, COEFFICIENTS_ast0_,
            INTERCEPT_dag0_, COEFFICIENTS_dag0_,
            P_VEC_FULL_ast, P_VEC_FULL_dag,
            SLATE_VEC_ast, SLATE_VEC_dag,
            LAMBDA,
            Q_SIGN_ <- strenv$jnp$array(-1.),
            seed_base
          )
          loss_dag_val <- base_grad_dag[[1]]
          grad_dag <- base_grad_dag[[2]]
          if (has_anchors) {
            reg_dag <- strenv$jnp$multiply(Lambda_sum, strenv$jnp$subtract(a_i_dag, zbar_dag))
            grad_dag <- strenv$jnp$subtract(grad_dag, reg_dag)
          }
        }

        base_grad_ast <- dQ_da_ast(
          a_i_ast, a_i_dag,
          INTERCEPT_ast_,  COEFFICIENTS_ast_,
          INTERCEPT_dag_,  COEFFICIENTS_dag_,
          INTERCEPT_ast0_, COEFFICIENTS_ast0_,
          INTERCEPT_dag0_, COEFFICIENTS_dag0_,
          P_VEC_FULL_ast, P_VEC_FULL_dag,
          SLATE_VEC_ast, SLATE_VEC_dag,
          LAMBDA,
          Q_SIGN_ <- strenv$jnp$array(1.),
          seed_base
        )
        loss_ast_val <- base_grad_ast[[1]]
        grad_ast <- base_grad_ast[[2]]
        if (has_anchors) {
          reg_ast <- strenv$jnp$multiply(Lambda_sum, strenv$jnp$subtract(a_i_ast, zbar_ast))
          grad_ast <- strenv$jnp$subtract(grad_ast, reg_ast)
        }

        if (adversarial) {
          a_pred_dag <- strenv$jnp$add(a_i_dag, strenv$jnp$multiply(eta_t, grad_dag))
        } else {
          a_pred_dag <- a_i_dag
        }
        a_pred_ast <- strenv$jnp$add(a_i_ast, strenv$jnp$multiply(eta_t, grad_ast))

        if (use_uniform_half) {
          half_count <- half_count + 1L
          if (stats::runif(1) <= 1 / half_count) {
            half_sample_ast <- a_pred_ast
            if (adversarial) {
              half_sample_dag <- a_pred_dag
            }
          }
        }

        SEED <- strenv$jax$random$split(SEED)[[1L]]
        seed_look <- SEED

        if (adversarial) {
          grad_pred_dag <- dQ_da_dag(
            a_pred_ast, a_pred_dag,
            INTERCEPT_ast_,  COEFFICIENTS_ast_,
            INTERCEPT_dag_,  COEFFICIENTS_dag_,
            INTERCEPT_ast0_, COEFFICIENTS_ast0_,
            INTERCEPT_dag0_, COEFFICIENTS_dag0_,
            P_VEC_FULL_ast, P_VEC_FULL_dag,
            SLATE_VEC_ast, SLATE_VEC_dag,
            LAMBDA,
            Q_SIGN_ <- strenv$jnp$array(-1.),
            seed_look
          )[[2]]
          if (has_anchors) {
            reg_pred_dag <- strenv$jnp$multiply(Lambda_sum, strenv$jnp$subtract(a_pred_dag, zbar_dag))
            grad_pred_dag <- strenv$jnp$subtract(grad_pred_dag, reg_pred_dag)
          }
        }

        grad_pred_ast <- dQ_da_ast(
          a_pred_ast, a_pred_dag,
          INTERCEPT_ast_,  COEFFICIENTS_ast_,
          INTERCEPT_dag_,  COEFFICIENTS_dag_,
          INTERCEPT_ast0_, COEFFICIENTS_ast0_,
          INTERCEPT_dag0_, COEFFICIENTS_dag0_,
          P_VEC_FULL_ast, P_VEC_FULL_dag,
          SLATE_VEC_ast, SLATE_VEC_dag,
          LAMBDA,
          Q_SIGN_ <- strenv$jnp$array(1.),
          seed_look
        )[[2]]
        if (has_anchors) {
          reg_pred_ast <- strenv$jnp$multiply(Lambda_sum, strenv$jnp$subtract(a_pred_ast, zbar_ast))
          grad_pred_ast <- strenv$jnp$subtract(grad_pred_ast, reg_pred_ast)
        }

        if (adversarial) {
          a_i_dag <- strenv$jnp$add(a_i_dag, strenv$jnp$multiply(eta_t, grad_pred_dag))
          strenv$grad_mag_dag_vec[i] <- list(strenv$jnp$linalg$norm(grad_pred_dag))
          strenv$loss_dag_vec[i] <- list(loss_dag_val)
        }
        a_i_ast <- strenv$jnp$add(a_i_ast, strenv$jnp$multiply(eta_t, grad_pred_ast))
        strenv$grad_mag_ast_vec[i] <- list(strenv$jnp$linalg$norm(grad_pred_ast))
        strenv$loss_ast_vec[i] <- list(loss_ast_val)

        strenv$inv_learning_rate_ast_vec[i] <- list(inv_lr_val)
        if (adversarial) {
          strenv$inv_learning_rate_dag_vec[i] <- list(inv_lr_val)
        }

        strenv$rain_lambda_vec[i] <- list(lambda_use_tf)
        strenv$rain_lambda_s_vec[i] <- list(lambda_use_tf)
        strenv$rain_lambda_sum_vec[i] <- list(Lambda_sum)
        strenv$rain_stage_idx[i] <- stage_idx
        if (has_anchors) {
          strenv$rain_anchor_bar_norm_ast[i] <- list(
            strenv$jnp$linalg$norm(strenv$jnp$subtract(a_i_ast, zbar_ast))
          )
          if (adversarial) {
            strenv$rain_anchor_bar_norm_dag[i] <- list(
              strenv$jnp$linalg$norm(strenv$jnp$subtract(a_i_dag, zbar_dag))
            )
          }
        } else {
          strenv$rain_anchor_bar_norm_ast[i] <- list(strenv$jnp$array(0., strenv$dtj))
          if (adversarial) {
            strenv$rain_anchor_bar_norm_dag[i] <- list(strenv$jnp$array(0., strenv$dtj))
          }
        }
      }

      if (use_uniform_half && half_count > 0L) {
        a_i_ast <- half_sample_ast
        if (adversarial && !is.null(half_sample_dag)) {
          a_i_dag <- half_sample_dag
        }
      }

      Lambda_sum <- strenv$jnp$add(Lambda_sum, lambda_use_tf)
      B_ast <- strenv$jnp$add(B_ast, strenv$jnp$multiply(lambda_use_tf, z_s_ast_start))
      if (adversarial) {
        B_dag <- strenv$jnp$add(B_dag, strenv$jnp$multiply(lambda_use_tf, z_s_dag_start))
      }
      lambda_stage_num <- lambda_stage_num * (1 + rain_gamma)
      stage_idx <- stage_idx + 1L
    }
  } else {
    while(goOn == F){
      if ((i <- i + 1) < 5 | i %in% unique(ceiling(c(0.25, 0.5, 0.75, 1) * nSGD))) { 
        message(sprintf("SGD Iteration: %s of %s", i, nSGD) ) 
      }

      # do dag updates ("min" step - only do in adversarial mode)
      if( i %% 1 == 0 & adversarial & !use_joint_extragrad ){
      # note: dQ_da_dag built off FullGetQStar_
      SEED   <- strenv$jax$random$split(SEED)[[1L]]
      grad_i_dag <- dQ_da_dag(  a_i_ast, a_i_dag,                    #1,2
                                INTERCEPT_ast_,  COEFFICIENTS_ast_,  #3,4
                                INTERCEPT_dag_,  COEFFICIENTS_dag_,  #5,6
                                INTERCEPT_ast0_, COEFFICIENTS_ast0_, #7,8
                                INTERCEPT_dag0_, COEFFICIENTS_dag0_, #9,10
                                P_VEC_FULL_ast, P_VEC_FULL_dag,      #11,12
                                SLATE_VEC_ast, SLATE_VEC_dag,        #13,14
                                LAMBDA,                              #15
                                Q_SIGN_ <- strenv$jnp$array(-1.),    #16
                                SEED                                 #17
                                )
      strenv$loss_dag_vec[i] <- list(grad_i_dag[[1]]); grad_i_dag <- grad_i_dag[[2]]
      if (use_rain) {
        grad_i_dag <- strenv$jnp$subtract(
          grad_i_dag,
          strenv$jnp$multiply(rain_lambda_t, strenv$jnp$subtract(a_i_dag, anchor_dag))
        )
      }

      # choose gradient to apply (OGDA / Extra-Gradient)
      grad_step_dag <- grad_i_dag
      if (!is.null(grad_prev_dag) && optimism == "ogda") {
        grad_step_dag <- strenv$jnp$add(grad_i_dag,
                                        strenv$jnp$multiply(strenv$jnp$array(optimism_coef, strenv$dtj),
                                                            strenv$jnp$subtract(grad_i_dag, grad_prev_dag)))
      }

      if(FALSE){ # sanity view of utilities (debug code) 
        TSAMP1 <- strenv$jax$vmap(function(s_){
          strenv$getMultinomialSamp(SLATE_VEC_ast, MNtemp, s_, strenv$ParameterizationType, strenv$d_locator_use)
        }, in_axes = list(0L))(strenv$jax$random$split(SEED, nMonte_Qglm))
        TSAMP2 <- strenv$jax$vmap(function(s_){
          strenv$getMultinomialSamp(SLATE_VEC_ast, MNtemp, s_, strenv$ParameterizationType, strenv$d_locator_use)
        }, in_axes = list(0L))(strenv$jax$random$split(SEED, nMonte_Qglm))
        ast0_ <- strenv$np$array(strenv$jax$vmap(function(t_,t__){ getQStar_single(t_, t__,
                                                                  INTERCEPT_ast0_, COEFFICIENTS_ast0_,
                                                                  INTERCEPT_ast0_, COEFFICIENTS_ast0_)}, in_axes = list(0L,0L))(TSAMP1,TSAMP2))[,1,1]
        ast_ <- strenv$np$array(strenv$jax$vmap(function(t_,t__){ getQStar_single(t_, t__,
                        INTERCEPT_ast_, COEFFICIENTS_ast_,
                        INTERCEPT_ast_, COEFFICIENTS_ast_)}, in_axes = list(0L,0L))(TSAMP1,TSAMP2))[,1,1]
        dag0_ <- strenv$np$array(strenv$jax$vmap(function(t_,t__){ getQStar_single(t_, t__,
                                                                   INTERCEPT_dag0_, COEFFICIENTS_dag0_,
                                                                   INTERCEPT_dag0_, COEFFICIENTS_dag0_)}, in_axes = list(0L,0L))(TSAMP1,TSAMP2))[,1,1]
        dag_ <- strenv$np$array(strenv$jax$vmap(function(t_,t__){ getQStar_single(t_, t__,
                                                                  INTERCEPT_dag_, COEFFICIENTS_dag_,
                                                                  INTERCEPT_dag_, COEFFICIENTS_dag_)}, in_axes = list(0L,0L))(TSAMP1,TSAMP2))[,1,1]
        cor(cbind(ast0_,ast_,dag0_,dag_))
        plot(ast_,dag_)
      }
      
      if(!use_optax){
        if(i == 1){
          inv_learning_rate_da_dag <- 
            strenv$jax$lax$stop_gradient(
                strenv$jnp$maximum(INIT_MIN_GRAD_ACCUM, 10*strenv$jnp$square(strenv$jnp$linalg$norm( grad_i_dag )))
            )
        }
        inv_learning_rate_da_dag <-  strenv$jax$lax$stop_gradient(GetInvLR(inv_learning_rate_da_dag, grad_i_dag))
        lr_dag_i <- strenv$jnp$sqrt(inv_learning_rate_da_dag)

        if (use_extragrad) {
          # look-ahead step, then recompute gradient
          a_pred_dag <- GetUpdatedParameters(a_vec = a_i_dag,
                                             grad_i = grad_i_dag,
                                             inv_learning_rate_i = lr_dag_i)
          if (use_smp) {
            gamma_dag <- strenv$jnp$reciprocal(lr_dag_i)
            smp_sum_weighted_dag <- strenv$jnp$add(smp_sum_weighted_dag,
                                                   strenv$jnp$multiply(gamma_dag, a_pred_dag))
            smp_sum_gamma_dag <- strenv$jnp$add(smp_sum_gamma_dag, gamma_dag)
            smp_gamma_dag_vec[[i]] <- gamma_dag
          }
          SEED   <- strenv$jax$random$split(SEED)[[1L]]
          grad_pred_dag <- dQ_da_dag(  a_i_ast, a_pred_dag,                 #1,2
                                       INTERCEPT_ast_,  COEFFICIENTS_ast_,  #3,4
                                       INTERCEPT_dag_,  COEFFICIENTS_dag_,  #5,6
                                       INTERCEPT_ast0_, COEFFICIENTS_ast0_, #7,8
                                       INTERCEPT_dag0_, COEFFICIENTS_dag0_, #9,10
                                       P_VEC_FULL_ast, P_VEC_FULL_dag,      #11,12
                                       SLATE_VEC_ast, SLATE_VEC_dag,        #13,14
                                       LAMBDA,                              #15
                                       Q_SIGN_ <- strenv$jnp$array(-1.),    #16
                                       SEED                                 #17
                                       )
          grad_step_dag <- grad_pred_dag[[2]]
          if (use_rain) {
            grad_step_dag <- strenv$jnp$subtract(
              grad_step_dag,
              strenv$jnp$multiply(rain_lambda_t, strenv$jnp$subtract(a_pred_dag, anchor_dag))
            )
          }
        }

        a_i_dag <- GetUpdatedParameters(a_vec = a_i_dag, 
                                        grad_i = grad_step_dag,
                                        inv_learning_rate_i = lr_dag_i)
      }
      if(use_optax){
        updates_and_opt_state_dag <- jit_update_dag( updates = grad_i_dag, 
                                                     state = opt_state_dag, 
                                                     params = a_i_dag )
        opt_state_dag <- updates_and_opt_state_dag[[2]]
        a_i_dag <- jit_apply_updates_dag(params = a_i_dag,  
                                         updates = updates_and_opt_state_dag[[1]])
      }

      strenv$grad_mag_dag_vec[i] <- list(strenv$jnp$linalg$norm( grad_step_dag ))
      grad_prev_dag <- grad_i_dag
      if(!use_optax){ strenv$inv_learning_rate_dag_vec[i] <- list( inv_learning_rate_da_dag ) }
    }

    # do updates ("max" step)
    if( (i %% 1 == 0 | (!adversarial)) && !use_joint_extragrad ){
      SEED   <- strenv$jax$random$split(SEED)[[1L]] 
      grad_i_ast <- dQ_da_ast( a_i_ast, a_i_dag,
                               INTERCEPT_ast_,  COEFFICIENTS_ast_,
                               INTERCEPT_dag_,  COEFFICIENTS_dag_,
                               INTERCEPT_ast0_, COEFFICIENTS_ast0_,
                               INTERCEPT_dag0_, COEFFICIENTS_dag0_,
                               P_VEC_FULL_ast, P_VEC_FULL_dag,
                               SLATE_VEC_ast, SLATE_VEC_dag,
                               LAMBDA, 
                               Q_SIGN_ <- strenv$jnp$array(1.),
                               SEED
                               )
      strenv$loss_ast_vec[i] <- list(grad_i_ast[[1]]); grad_i_ast <- grad_i_ast[[2]]
      if (use_rain) {
        grad_i_ast <- strenv$jnp$subtract(
          grad_i_ast,
          strenv$jnp$multiply(rain_lambda_t, strenv$jnp$subtract(a_i_ast, anchor_ast))
        )
      }
      grad_step_ast <- grad_i_ast
      if (!is.null(grad_prev_ast) && optimism == "ogda") {
        grad_step_ast <- strenv$jnp$add(grad_i_ast,
                                        strenv$jnp$multiply(strenv$jnp$array(optimism_coef, strenv$dtj),
                                                            strenv$jnp$subtract(grad_i_ast, grad_prev_ast)))
      }
      
      if(!use_optax){
        if(i==1){ 
          inv_learning_rate_da_ast <- strenv$jax$lax$stop_gradient(
                strenv$jnp$maximum(INIT_MIN_GRAD_ACCUM, 10*strenv$jnp$square(strenv$jnp$linalg$norm(grad_i_ast)))  
            )
        }
        inv_learning_rate_da_ast <-  strenv$jax$lax$stop_gradient( GetInvLR(inv_learning_rate_da_ast, grad_i_ast) )
        lr_ast_i <- strenv$jnp$sqrt(inv_learning_rate_da_ast)

        if (use_extragrad) {
          a_pred_ast <- GetUpdatedParameters(a_vec = a_i_ast, 
                                             grad_i = grad_i_ast,
                                             inv_learning_rate_i = lr_ast_i)
          if (use_smp) {
            gamma_ast <- strenv$jnp$reciprocal(lr_ast_i)
            smp_sum_weighted_ast <- strenv$jnp$add(smp_sum_weighted_ast,
                                                   strenv$jnp$multiply(gamma_ast, a_pred_ast))
            smp_sum_gamma_ast <- strenv$jnp$add(smp_sum_gamma_ast, gamma_ast)
            smp_gamma_ast_vec[[i]] <- gamma_ast
          }
          SEED   <- strenv$jax$random$split(SEED)[[1L]] 
          grad_pred_ast <- dQ_da_ast( a_pred_ast, a_i_dag,
                                      INTERCEPT_ast_,  COEFFICIENTS_ast_,
                                      INTERCEPT_dag_,  COEFFICIENTS_dag_,
                                      INTERCEPT_ast0_, COEFFICIENTS_ast0_,
                                      INTERCEPT_dag0_, COEFFICIENTS_dag0_,
                                      P_VEC_FULL_ast, P_VEC_FULL_dag,
                                      SLATE_VEC_ast, SLATE_VEC_dag,
                                      LAMBDA, 
                                      Q_SIGN_ <- strenv$jnp$array(1.),
                                      SEED
                                      )
          grad_step_ast <- grad_pred_ast[[2]]
          if (use_rain) {
            grad_step_ast <- strenv$jnp$subtract(
              grad_step_ast,
              strenv$jnp$multiply(rain_lambda_t, strenv$jnp$subtract(a_pred_ast, anchor_ast))
            )
          }
        }

        a_i_ast <- GetUpdatedParameters(a_vec = a_i_ast, 
                                        grad_i = grad_step_ast,
                                        inv_learning_rate_i = lr_ast_i)
      }

      if(use_optax){
        updates_and_opt_state_ast <- jit_update_ast( updates = grad_i_ast, 
                                                     state = opt_state_ast, 
                                                     params = a_i_ast)
        opt_state_ast <- updates_and_opt_state_ast[[2]]
        a_i_ast <- jit_apply_updates_ast(params = a_i_ast, 
                                         updates = updates_and_opt_state_ast[[1]])
      }

      strenv$grad_mag_ast_vec[i] <- list( strenv$jnp$linalg$norm( grad_step_ast ) )
      grad_prev_ast <- grad_i_ast
      if(!use_optax){ strenv$inv_learning_rate_ast_vec[i] <- list( inv_learning_rate_da_ast ) }
    }

    # Joint look-ahead (adversarial mode, non-optax)
    if (use_joint_extragrad) {
      start_ast <- a_i_ast
      start_dag <- a_i_dag

      SEED   <- strenv$jax$random$split(SEED)[[1L]]
      base_grad_dag <- dQ_da_dag(  a_i_ast, a_i_dag,                    #1,2
                                   INTERCEPT_ast_,  COEFFICIENTS_ast_,  #3,4
                                   INTERCEPT_dag_,  COEFFICIENTS_dag_,  #5,6
                                   INTERCEPT_ast0_, COEFFICIENTS_ast0_, #7,8
                                   INTERCEPT_dag0_, COEFFICIENTS_dag0_, #9,10
                                   P_VEC_FULL_ast, P_VEC_FULL_dag,      #11,12
                                   SLATE_VEC_ast, SLATE_VEC_dag,        #13,14
                                   LAMBDA,                              #15
                                   Q_SIGN_ <- strenv$jnp$array(-1.),    #16
                                   SEED                                 #17
      )
      strenv$loss_dag_vec[i] <- list(base_grad_dag[[1]]); base_grad_dag <- base_grad_dag[[2]]
      if (use_rain) {
        base_grad_dag <- strenv$jnp$subtract(
          base_grad_dag,
          strenv$jnp$multiply(rain_lambda_t, strenv$jnp$subtract(a_i_dag, anchor_dag))
        )
      }

      SEED   <- strenv$jax$random$split(SEED)[[1L]]
      base_grad_ast <- dQ_da_ast( a_i_ast, a_i_dag,
                                  INTERCEPT_ast_,  COEFFICIENTS_ast_,
                                  INTERCEPT_dag_,  COEFFICIENTS_dag_,
                                  INTERCEPT_ast0_, COEFFICIENTS_ast0_,
                                  INTERCEPT_dag0_, COEFFICIENTS_dag0_,
                                  P_VEC_FULL_ast, P_VEC_FULL_dag,
                                  SLATE_VEC_ast, SLATE_VEC_dag,
                                  LAMBDA, 
                                  Q_SIGN_ <- strenv$jnp$array(1.),
                                  SEED
      )
      strenv$loss_ast_vec[i] <- list(base_grad_ast[[1]]); base_grad_ast <- base_grad_ast[[2]]
      if (use_rain) {
        base_grad_ast <- strenv$jnp$subtract(
          base_grad_ast,
          strenv$jnp$multiply(rain_lambda_t, strenv$jnp$subtract(a_i_ast, anchor_ast))
        )
      }

      if(i == 1){
        inv_learning_rate_da_dag <- 
          strenv$jax$lax$stop_gradient(
            strenv$jnp$maximum(INIT_MIN_GRAD_ACCUM, 10*strenv$jnp$square(strenv$jnp$linalg$norm( base_grad_dag )))
          )
        inv_learning_rate_da_ast <- strenv$jax$lax$stop_gradient(
          strenv$jnp$maximum(INIT_MIN_GRAD_ACCUM, 10*strenv$jnp$square(strenv$jnp$linalg$norm(base_grad_ast)))  
        )
      }
      inv_learning_rate_da_dag <-  strenv$jax$lax$stop_gradient(GetInvLR(inv_learning_rate_da_dag, base_grad_dag))
      inv_learning_rate_da_ast <-  strenv$jax$lax$stop_gradient(GetInvLR(inv_learning_rate_da_ast, base_grad_ast))
      lr_dag_i <- strenv$jnp$sqrt(inv_learning_rate_da_dag)
      lr_ast_i <- strenv$jnp$sqrt(inv_learning_rate_da_ast)

      # look-ahead for both players
      a_pred_dag <- GetUpdatedParameters(a_vec = a_i_dag,
                                         grad_i = base_grad_dag,
                                         inv_learning_rate_i = lr_dag_i)
      a_pred_ast <- GetUpdatedParameters(a_vec = a_i_ast, 
                                         grad_i = base_grad_ast,
                                         inv_learning_rate_i = lr_ast_i)
      if (use_smp) {
        gamma_dag <- strenv$jnp$reciprocal(lr_dag_i)
        gamma_ast <- strenv$jnp$reciprocal(lr_ast_i)
        smp_sum_weighted_dag <- strenv$jnp$add(smp_sum_weighted_dag,
                                               strenv$jnp$multiply(gamma_dag, a_pred_dag))
        smp_sum_gamma_dag <- strenv$jnp$add(smp_sum_gamma_dag, gamma_dag)
        smp_gamma_dag_vec[[i]] <- gamma_dag
        smp_sum_weighted_ast <- strenv$jnp$add(smp_sum_weighted_ast,
                                               strenv$jnp$multiply(gamma_ast, a_pred_ast))
        smp_sum_gamma_ast <- strenv$jnp$add(smp_sum_gamma_ast, gamma_ast)
        smp_gamma_ast_vec[[i]] <- gamma_ast
      }

      SEED   <- strenv$jax$random$split(SEED)[[1L]]
      grad_pred_dag <- dQ_da_dag(  a_pred_ast, a_pred_dag,               #1,2
                                   INTERCEPT_ast_,  COEFFICIENTS_ast_,  #3,4
                                   INTERCEPT_dag_,  COEFFICIENTS_dag_,  #5,6
                                   INTERCEPT_ast0_, COEFFICIENTS_ast0_, #7,8
                                   INTERCEPT_dag0_, COEFFICIENTS_dag0_, #9,10
                                   P_VEC_FULL_ast, P_VEC_FULL_dag,      #11,12
                                   SLATE_VEC_ast, SLATE_VEC_dag,        #13,14
                                   LAMBDA,                              #15
                                   Q_SIGN_ <- strenv$jnp$array(-1.),    #16
                                   SEED                                 #17
      )[[2]]
      if (use_rain) {
        grad_pred_dag <- strenv$jnp$subtract(
          grad_pred_dag,
          strenv$jnp$multiply(rain_lambda_t, strenv$jnp$subtract(a_pred_dag, anchor_dag))
        )
      }

      SEED   <- strenv$jax$random$split(SEED)[[1L]] 
      grad_pred_ast <- dQ_da_ast( a_pred_ast, a_pred_dag,
                                  INTERCEPT_ast_,  COEFFICIENTS_ast_,
                                  INTERCEPT_dag_,  COEFFICIENTS_dag_,
                                  INTERCEPT_ast0_, COEFFICIENTS_ast0_,
                                  INTERCEPT_dag0_, COEFFICIENTS_dag0_,
                                  P_VEC_FULL_ast, P_VEC_FULL_dag,
                                  SLATE_VEC_ast, SLATE_VEC_dag,
                                  LAMBDA, 
                                  Q_SIGN_ <- strenv$jnp$array(1.),
                                  SEED
      )[[2]]
      if (use_rain) {
        grad_pred_ast <- strenv$jnp$subtract(
          grad_pred_ast,
          strenv$jnp$multiply(rain_lambda_t, strenv$jnp$subtract(a_pred_ast, anchor_ast))
        )
      }

      a_i_dag <- GetUpdatedParameters(a_vec = a_i_dag, 
                                      grad_i = grad_pred_dag,
                                      inv_learning_rate_i = lr_dag_i)
      a_i_ast <- GetUpdatedParameters(a_vec = a_i_ast, 
                                      grad_i = grad_pred_ast,
                                      inv_learning_rate_i = lr_ast_i)

      strenv$grad_mag_dag_vec[i] <- list(strenv$jnp$linalg$norm( grad_pred_dag ))
      strenv$grad_mag_ast_vec[i] <- list(strenv$jnp$linalg$norm( grad_pred_ast ))
      grad_prev_dag <- base_grad_dag
      grad_prev_ast <- base_grad_ast
      strenv$inv_learning_rate_dag_vec[i] <- list( inv_learning_rate_da_dag )
      strenv$inv_learning_rate_ast_vec[i] <- list( inv_learning_rate_da_ast )
      strenv$extragrad_eval_points[[i]] <- list(
        start = list(a_ast = start_ast, a_dag = start_dag),
        ast = list(a_pred_ast = a_pred_ast, a_pred_dag = a_pred_dag),
        dag = list(a_pred_ast = a_pred_ast, a_pred_dag = a_pred_dag)
      )
    }
      if(i >= nSGD){goOn <- TRUE}
      if (use_rain) {
        anchor_ast <- a_i_ast
        anchor_dag <- a_i_dag
      }
    }
  }

  if (use_smp) {
    a_i_ast <- strenv$jnp$divide(smp_sum_weighted_ast, smp_sum_gamma_ast)
    if (adversarial) {
      a_i_dag <- strenv$jnp$divide(smp_sum_weighted_dag, smp_sum_gamma_dag)
    }
    strenv$smp_sum_gamma_ast <- smp_sum_gamma_ast
    strenv$smp_sum_gamma_dag <- if (adversarial) smp_sum_gamma_dag else NULL
    strenv$smp_avg_ast <- a_i_ast
    strenv$smp_avg_dag <- if (adversarial) a_i_dag else NULL
    strenv$smp_gamma_ast_vec <- smp_gamma_ast_vec
    strenv$smp_gamma_dag_vec <- if (adversarial) smp_gamma_dag_vec else NULL
  }

  message("Saving output from gd [getQPiStar_gd]...")
  {
    pi_star_ast_full_simplex_ <- getPrettyPi( pi_star_ast_<-strenv$a2Simplex_diff_use(a_i_ast),
                                              strenv$ParameterizationType,
                                              strenv$d_locator_use,       
                                              strenv$main_comp_mat,   
                                              strenv$shadow_comp_mat)
    pi_star_dag_full_simplex_ <- getPrettyPi( pi_star_dag_<-strenv$a2Simplex_diff_use(a_i_dag),
                                              strenv$ParameterizationType,
                                              strenv$d_locator_use,       
                                              strenv$main_comp_mat,   
                                              strenv$shadow_comp_mat)

    if(glm_family=="gaussian"){ 
      pi_star_ast_f_all <- strenv$jnp$expand_dims(pi_star_ast_,0L)
      pi_star_dag_f_all <- strenv$jnp$expand_dims(pi_star_dag_,0L)
    }
    if(glm_family != "gaussian"){ 
      draw_ast <- draw_profile_samples(
        pi_star_ast_, nMonte_Qglm, SEED,
        MNtemp, strenv$ParameterizationType, strenv$d_locator_use,
        sampler = strenv$getMultinomialSamp
      )
      pi_star_ast_f_all <- draw_ast$samples
      SEED <- draw_ast$seed_next
      draw_dag <- draw_profile_samples(
        pi_star_dag_, nMonte_Qglm, SEED,
        MNtemp, strenv$ParameterizationType, strenv$d_locator_use,
        sampler = strenv$getMultinomialSamp
      )
      pi_star_dag_f_all <- draw_dag$samples
      SEED <- draw_dag$seed_next
    }
    
    if(!adversarial){ 
      q_star_f <- strenv$Vectorized_QMonteIter(
                                         pi_star_ast_f_all,  pi_star_dag_f_all,
                                         INTERCEPT_ast_, COEFFICIENTS_ast_,
                                         INTERCEPT_dag_, COEFFICIENTS_dag_)$mean(0L)
    }
    if(adversarial){ 
      n_q_samp <- as.integer(pi_star_ast_f_all$shape[[1L]])
      if (primary_pushforward == "multi") {
        samp_ast <- sample_pool_jax(
          pi_star_ast_, n_q_samp, primary_n_entrants, SEED,
          MNtemp, strenv$ParameterizationType, strenv$d_locator_use,
          sampler = strenv$getMultinomialSamp
        )
        TSAMP_ast_all <- samp_ast$samples
        SEED <- samp_ast$seed_next

        samp_dag <- sample_pool_jax(
          pi_star_dag_, n_q_samp, primary_n_entrants, SEED,
          MNtemp, strenv$ParameterizationType, strenv$d_locator_use,
          sampler = strenv$getMultinomialSamp
        )
        TSAMP_dag_all <- samp_dag$samples
        SEED <- samp_dag$seed_next

        samp_ast_field <- sample_pool_jax(
          SLATE_VEC_ast, n_q_samp, primary_n_field, SEED,
          MNtemp, strenv$ParameterizationType, strenv$d_locator_use,
          sampler = strenv$getMultinomialSamp
        )
        TSAMP_ast_PrimaryComp_all <- samp_ast_field$samples
        SEED <- samp_ast_field$seed_next

        samp_dag_field <- sample_pool_jax(
          SLATE_VEC_dag, n_q_samp, primary_n_field, SEED,
          MNtemp, strenv$ParameterizationType, strenv$d_locator_use,
          sampler = strenv$getMultinomialSamp
        )
        TSAMP_dag_PrimaryComp_all <- samp_dag_field$samples
        SEED <- samp_dag_field$seed_next
      } else {
        TSAMP_ast_all <- pi_star_ast_f_all
        TSAMP_dag_all <- pi_star_dag_f_all
        draw_ast_field <- draw_profile_samples(
          SLATE_VEC_ast, n_q_samp, SEED,
          MNtemp, strenv$ParameterizationType, strenv$d_locator_use,
          sampler = strenv$getMultinomialSamp
        )
        TSAMP_ast_PrimaryComp_all <- draw_ast_field$samples
        SEED <- draw_ast_field$seed_next
        
        draw_dag_field <- draw_profile_samples(
          SLATE_VEC_dag, n_q_samp, SEED,
          MNtemp, strenv$ParameterizationType, strenv$d_locator_use,
          sampler = strenv$getMultinomialSamp
        )
        TSAMP_dag_PrimaryComp_all <- draw_dag_field$samples
        SEED <- draw_dag_field$seed_next
      }

      Qres <- strenv$Vectorized_QMonteIter_MaxMin(
        TSAMP_ast_all, TSAMP_dag_all,
        TSAMP_ast_PrimaryComp_all, TSAMP_dag_PrimaryComp_all,
        a_i_ast, a_i_dag,
        INTERCEPT_ast_, COEFFICIENTS_ast_,
        INTERCEPT_dag_, COEFFICIENTS_dag_,
        INTERCEPT_ast0_, COEFFICIENTS_ast0_,
        INTERCEPT_dag0_, COEFFICIENTS_dag0_,
        P_VEC_FULL_ast, P_VEC_FULL_dag,
        LAMBDA, strenv$jnp$array(1.), 
        strenv$jax$random$split(SEED, n_q_samp)
      )
      q_star_val <- Qres$q_ast$mean(0L)
      q_star_dims <- if (length(pi_star_ast_full_simplex_$shape) >= 2L) {
        list(1L, 1L)
      } else {
        list(1L)
      }
      q_star_val <- strenv$jnp$reshape(q_star_val, q_star_dims)
      q_star_f <- strenv$jnp$concatenate(list(q_star_val, q_star_val, q_star_val), 0L)
    }

    # setup results for returning
    ret_array <- strenv$jnp$concatenate(ifelse(gd_full_simplex, 
                                               yes = list(list( q_star_f, 
                                                                pi_star_ast_full_simplex_, 
                                                                pi_star_dag_full_simplex_ )), 
                                               no  = list(list( q_star_f, 
                                                                pi_star_ast_, 
                                                                pi_star_dag_ )))[[1]])
    if( functionReturn ){ 
                            ret_array <- list(ret_array,
                                               list("dQ_da_ast" = dQ_da_ast, 
                                                    "dQ_da_dag" = dQ_da_dag,
                                                    "QFXN" = QFXN, 
                                                    "getMultinomialSamp" = strenv$getMultinomialSamp,
                                                    "a_i_ast" = a_i_ast, 
                                                    "a_i_dag" = a_i_dag
                                               ) )
    }
    return( ret_array )
  }
}
