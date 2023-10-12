get_se <- function(){

  # start here - get SEs from M-estimation for marginal mean computation
  # do variational inference
  if(useVariational == T){
    #as.matrix(getClassProb( X  ))

    # analyze uncertainty - cluster coefficients
    ClassProbsPosterior_draws <- replicate(100,as.matrix(ClassProbProj$kernel_prior$sample()))
    ClassProbProjCoefs_se <- apply(ClassProbsPosterior_draws,1:2,sd)
    ClassProbProjCoefs <- apply(ClassProbsPosterior_draws,1:2,mean)
    sort(ClassProbProjCoefs/ClassProbProjCoefs_se)
    row.names(ClassProbProjCoefs_se) <- row.names(ClassProbProjCoefs) <- colnames(X)
    colnames(ClassProbProjCoefs_se) <- colnames(ClassProbProjCoefs) <- paste("k",2:K,sep="")

    PiPosterior_draws <- replicate(100,list(getPiList()))
    hypotheticalProbList_full <- sapply(1:K,function(k_){
      tmp_ <- sapply(1:length(PiPosterior_draws),function(boo_){
        PiPosterior_draws[[boo_]][[k_]] })
      posterior_means_k <- apply(tmp_,1,function(zer){
        colMeans(do.call(rbind,zer)) })
      list(posterior_means_k) })
    seList_automatic <- sapply(1:K,function(k_){
      tmp_ <- sapply(1:length(PiPosterior_draws),function(boo_){
        PiPosterior_draws[[boo_]][[k_]] })
      posterior_means_k <- apply(tmp_,1,function(zer){
        apply(do.call(rbind,zer),2,sd) })
      list(posterior_means_k)
    })
  }

  # Pr(ClassProb) = X beta + intercept
  VarCov_ProbClust <- ClassProbsXobs <- ClassProbProjCoefs <- ClassProbProjCoefs_se <- NULL
  if((piSEtype  == "automatic" | piSEtype == "both") & useVariational==F){
    {
      ClassProbsXobs <- as.array( getClassProb( X) )
      tf_LAMBDA_selected <- jnp$array(LAMBDA_selected,jnp$float32)
      tf_FALSE <- jnp$array( returnWeightsFxn(F),jnp$bool); tf_TRUE <- jnp$array(  returnWeightsFxn(T),jnp$bool)
      tf_LAMBDA_selected <- jnp$array( returnWeightsFxn(LAMBDA_selected),jnp$float32)

      {
        tf_true <- jnp$array(T,jnp$bool); tf_false <- jnp$array(F,jnp$bool)
        nParam_tf <- sum(unlist( lapply(lapply(tv_trainWith,dim),prod)) )
        nParam_total <- tf$cast( nParam_tf + 1, jnp$int32)

        # unvectorized
        getPsiAndJacobian_unvectorized <- jax$jit(function(
    Y__, X__,
    factorMat__, logProb__){
          #Y__ <- jnp$expand_dims(Y__,0L);  X__ <- jnp$expand_dims(X__,0L)
          #factorMat__ <- jnp$expand_dims(factorMat__,0L)

          with(tf$GradientTape(persistent = (persistBool <- F), watch_accessed_variables = F) %as% tape_OUT, {
            tape_OUT$watch( c( Qhat_tf, tv_trainWith ) )
            with(tf$GradientTape(persistent = persistBool, watch_accessed_variables = F) %as% tape_IN, {
              tape_IN$watch(  tv_trainWith  )
              loss_i <-   getLoss_tf_unnormalized(Y_ = Y__,
                                                  X_ = X__,
                                                  factorMat_ = factorMat__,
                                                  logProb_ = logProb__,
                                                  REGULARIZATION_LAMBDA = tf_LAMBDA_selected)
              probRatio_i <-   getProbRatio_tf(Y_ = Y__,
                                               X_ = X__,
                                               factorMat_ = factorMat__,
                                               logProb_ = logProb__,
                                               REGULARIZATION_LAMBDA = tf_LAMBDA_selected)
              probRatio_i <- jnp$reshape(probRatio_i,list(1L,1L))
            })
            # obtain inner gradient information (Jacobian)
            my_grad_i <- tape_IN$gradient( loss_i, tv_trainWith )

            # reshape
            my_grad_i <- lapply(my_grad_i,function(zer){jnp$reshape(zer,list(as.integer(  prod(dim(zer))) ,1L))  })
            my_grad_i <- jnp$concatenate(my_grad_i,0L)

            # combine elements of psi
            my_psi_i <- jnp$concatenate(list(Y__*probRatio_i - Qhat_tf*probRatio_i,  my_grad_i),0L)
          })
          my_jacob_i <- tape_OUT$jacobian( my_psi_i, c(Qhat_tf, tv_trainWith ) )
          my_jacob_i <- lapply(my_jacob_i,function(zer){jnp$reshape(zer,list(jnp$shape(my_psi_i)[1] ,-1L))  })

          JacobianMat_i <- jnp$concatenate(my_jacob_i,1L)
          PsiWithJacobian_i <- jnp$concatenate(list(my_psi_i,JacobianMat_i),1L)
          #if(any(is.na(PsiWithJacobian_i))){browser()}
          return(  PsiWithJacobian_i )
        })

        var_names <- c("Qhat",unlist( lapply(tv_trainWith,function(zer){
          tmp_ <- gsub(as.character( zer$shape ),pattern="\\(",replace="")
          tmp_ <- gsub(gsub(tmp_,pattern="\\)",replace=""),pattern=" ",replace="")
          paste(zer$name, 1:prod(f2n(unlist(
            gsub(gsub(strsplit(tmp_,split=",")[[1]],pattern="TensorShape\\[",replace=""),pattern="\\]",replace="")
          ))),sep="_")
        }) ))

        getJacobianComponents <- function(){
          psi_list <- neg_jacobian_list <- list()
          counter_ji <- 0; for(ji in split2_indices){
            counter_ji <- counter_ji + 1
            if(counter_ji %% 100 == 0){print(sprintf("M Estimation Iteration %i of %i",counter_ji,length(split2_indices)))}

            # tf calc
            if(optimization_language == "tf"){
              PsiWithJacobian_ji <- as.matrix( jnp$array(
                getPsiAndJacobian_unvectorized(Y__ = tfConst(as.matrix(Y[ji])),
                                               X__ = tfConst(t(X[ji,])),
                                               factorMat__ = tfConst(t(FactorsMat_numeric_0Indexed[ji,]),jnp$int32),
                                               logProb__ = tfConst(as.matrix(log_PrW[ji]) ) ),jnp$float32))
              psi_ji <- PsiWithJacobian_ji[,1]; names(psi_ji) <- var_names
              jacobian_ji <- PsiWithJacobian_ji[,-1]; colnames(jacobian_ji) <- row.names(jacobian_ji) <- var_names
            }

            # jax calc
            if(optimization_language == "jax"){
              # convert
              i_ <- ji
              if(counter_ji == 1){
                for(fn_ in rev(c("getLoss_tf_unnormalized","getProbRatio_tf"))){
                  jax_fxn_raw <- tf2jax$convert(eval(parse(text = fn_)),
                                                Y_  = jnp$array(as.matrix(Y[i_]),jnp$float32),
                                                X_  = jnp$array(t(X[i_,]),jnp$float32),
                                                factorMat_  = jnp$array(t(FactorsMat_numeric_0Indexed[i_,]),jnp$int32),
                                                logProb_ = jnp$array(as.matrix(log_PrW[i_]),jnp$float32),
                                                REGULARIZATION_LAMBDA = jnp$array(returnWeightsFxn(LAMBDA_selected),jnp$float32))

                  eval(parse(text = sprintf("%s <- jax_fxn_raw",
                                            internal_jax_fxn_name <- sprintf("internal_%s",
                                                                             gsub(fn_,pattern="_tf",replace="_jax")))))

                  # select parameters + names
                  param_set <- jax_fxn_raw[[2]]
                  param_set_names <- names( param_set )

                  # convert fxn with eval+params output into eval only
                  def_sig <- gsub(jax_fxn_raw[[1]]$signature,pattern="\\<Signature ",replace="")
                  def_sig  <- gsub(gsub(def_sig,pattern ="\\(",replace=""),pattern="\\)",replace="")
                  input_sig  <- gsub(gsub(def_sig,pattern ="\\(",replace=""),pattern="\\)",replace="")
                  jax_fxn <- sprintf('function(params,%s){
                             out_ <- %s[[1]](params,%s)[[1]];
                             return( jnp$reshape(out_,list()) )
                             }', def_sig ,  internal_jax_fxn_name,  input_sig)
                  grad_jax_fxn <- jax$grad(jax_fxn <- eval(parse(text = jax_fxn)),argnums = 0L)

                  # test the function
                  jax_eval <- jax_fxn(
                    param_set,
                    Y_  = jnp$array(as.matrix(Y[i_])),
                    X_  = jnp$array(t(X[i_,])),
                    factorMat_  = jnp$array(t(FactorsMat_numeric_0Indexed[i_,])),
                    logProb_ = jnp$array(as.matrix(log_PrW[i_])),
                    REGULARIZATION_LAMBDA = jnp$array(returnWeightsFxn(LAMBDA_selected))
                  )

                  # do some renaming
                  eval(parse(text = sprintf("%s <- jax_fxn", gsub(fn_,pattern="_tf",replace="_jax"))))
                  eval(parse(text = sprintf("grad_%s <- grad_jax_fxn", gsub(fn_,pattern="_tf",replace="_jax"))))
                  #eval(parse(text = sprintf("param_set_%s <- param_set", gsub(fn_,pattern="_tf",replace="_jax"))))
                  #eval(parse(text = sprintf("param_set_names_%s <- param_set_names", gsub(fn_,pattern="_tf",replace="_jax"))))
                  rm(jax_fxn, grad_jax_fxn)
                } #end loop setting up dx/dp

                # get psi fxn
                psi_fxn_jax <- function(
    # d with respect to these
                  Qhat_tf, param_set,

                  # data inputs:
                  Y_, X_, factorMat_, logProb_, REGULARIZATION_LAMBDA){
                  probRatio_i_jax <- getProbRatio_jax(
                    param_set,
                    Y_  = Y_, X_  = X_,
                    factorMat_  = factorMat_, logProb_ = logProb_,
                    REGULARIZATION_LAMBDA = REGULARIZATION_LAMBDA)
                  probRatio_contrib_i <- jnp$subtract(jnp$multiply(jnp$array(Y_), probRatio_i_jax),
                                                      jnp$multiply(jnp$array(Qhat_tf),probRatio_i_jax))$flatten()
                  my_grad_i_jax <- grad_getLoss_jax_unnormalized(
                    param_set,
                    Y_  = Y_,
                    X_  = X_,
                    factorMat_  = factorMat_,
                    logProb_ = logProb_,
                    REGULARIZATION_LAMBDA = REGULARIZATION_LAMBDA)[param_set_names]
                  my_grad_i_jax <- unlist(lapply(my_grad_i_jax, function(zap){zap$flatten()}))
                  names(my_grad_i_jax) <- NULL

                  # set up hessian
                  my_psi_i_jax <- jnp$concatenate( c(probRatio_contrib_i,
                                                     my_grad_i_jax), 0L)

                  return( my_psi_i_jax )
                }
                jacobian_psi_fxn_jax <- jax$jacobian(psi_fxn_jax, argnums = 0L:1L)
                #jacobian_psi_fxn_jax <- jax$jacrev(psi_fxn_jax, argnums = 0L:1L)

                # compile
                psi_fxn_jax <- jax$jit(psi_fxn_jax)
                jacobian_psi_fxn_jax <- jax$jit(jacobian_psi_fxn_jax)
              }

              Y__i <- jnp$array(as.matrix(Y[i_]))
              X__i <- jnp$array(t(X[i_,]))
              factorMat__i <- jnp$array(t(FactorsMat_numeric_0Indexed[i_,]))
              logProb__i <- jnp$array(as.matrix(log_PrW[i_]))
              REGULARIZATION_LAMBDA__ <- jnp$array(returnWeightsFxn(LAMBDA_selected))
              Qhat__ <- jnp$array(as.matrix(  Qhat_tf) )
              psi_i <- psi_fxn_jax(
                Qhat_tf = Qhat__,
                param_set = param_set,
                Y_  = Y__i,
                X_  = X__i,
                factorMat_  = factorMat__i,
                logProb_ = logProb__i,
                REGULARIZATION_LAMBDA = REGULARIZATION_LAMBDA__)
              psi_i_jacobian <- jacobian_psi_fxn_jax(
                Qhat__,
                param_set,
                Y__i,
                X__i,
                factorMat__i,
                logProb__i,
                REGULARIZATION_LAMBDA__)
              psi_i_jacobian <- unlist( psi_i_jacobian )
              names(psi_i_jacobian)[1] <- c("Qhat")
              #cbind(param_set_names,unlist(lapply(tv_trainWith,function(zer){zer$name})))
              psi_i_jacobian <- psi_i_jacobian[c("Qhat",param_set_names)]
              psi_i_jacobian <- sapply(1:length(psi_i_jacobian), function(zer){
                psi_i_jacobian[[zer]]$reshape(nParam_total, -1L)$to_py() })
              if("list" %in% class(psi_i_jacobian)){
                psi_i_jacobian <- do.call(cbind, psi_i_jacobian)
              }
              psi_i_jacobian <- apply( psi_i_jacobian,2,f2n )

              psi_ji <- psi_i$to_py(); names(psi_ji) <- var_names
              jacobian_ji <- psi_i_jacobian; colnames(jacobian_ji) <- row.names(jacobian_ji) <- var_names
            }

            # checks
            if(T == F){
              cbind(psi_i$to_py()[1:10],psi_ji[1:10])
              plot(psi_i$to_py(),psi_ji); abline(a=0,b=1)
              plot(c(jacobian_ji),c(psi_i_jacobian)); abline(a=0,b=1)
              plot(c(jacobian_ji),c(t(psi_i_jacobian))); abline(a=0,b=1)
              plot(abs(c(jacobian_ji)-c(psi_i_jacobian)))
              abline(a=0, b = 1); abline(a=0, b = -1)
              image(abs(jacobian_ji-psi_i_jacobian))
              plot(c(jacobian_ji[,1]),c(psi_i_jacobian[,1]));abline(a=0,b=1)
              plot(c(jacobian_ji[,2]),c(psi_i_jacobian[,2]));abline(a=0,b=1)
              plot(c(jacobian_ji[,2])-c(psi_i_jacobian[,2]));abline(h=0)
            }

            # sum(is.na(jacobian_ji))
            # save m est results to list holders
            #if(is.na(sum(psi_ji))){browser()}
            #if(is.na(sum(jacobian_ji))){browser()}
            psi_list[[counter_ji]] <- psi_ji
            neg_jacobian_list[[counter_ji]] <-   -1 * jacobian_ji
          }
          return(list("psi_list" = psi_list,
                      "neg_jacobian_list" = neg_jacobian_list))
        }
        JacobianComponents <- getJacobianComponents()
        psi_list <- JacobianComponents$psi_list
        neg_jacobian_list <- JacobianComponents$neg_jacobian_list
        jacob_NAs <- unlist( lapply(neg_jacobian_list,function(l_){is.na(sum(l_))}) )
        # X[which( jacob_NAs ),]
        # FactorsMat_numeric_0Indexed[which( jacob_NAs ),]
        if(sum(jacob_NAs) > 0){ print("Warning: Jacobian contains NAs! Dropping...") }
        psi_list <- psi_list[!jacob_NAs]
        neg_jacobian_list <- neg_jacobian_list[!jacob_NAs]
        rm( JacobianComponents )

        # resources
        #https://www.tensorflow.org/guide/advanced_autodiff
        #https://tensorflow.google.cn/api_docs/python/tf/vectorized_map?hl=zh-cn
        # need to deal with k > 2 case?
      }

      # calculate variance-covariance matrix
      {
        l1_psi <- unlist(lapply(psi_list,function(zer){mean(abs(zer))}))
        l1_neg_jacobian <- unlist(lapply(neg_jacobian_list,function(zer){mean(abs(zer))}))
        l1_check_ref <- split2_indices[which.max(l1_neg_jacobian)]
        print( as.vector(W[l1_check_ref,]) )
        print( sapply(1:length(hypotheticalProbList[[1]]),function(zer){
          tmp_ <- hypotheticalProbList[[1]][[zer]]; tmp_[which.max(tmp_)] }) )
        sapply(1:length(hypotheticalProbList[[1]]),function(zer){
          tmp_ <- hypotheticalProbList[[1]][[zer]]; tmp_[which.min(tmp_)] })

        try(hist( log(l1_neg_jacobian ), main=sprintf("Max value is %.3f",max(l1_neg_jacobian))),T)

        # calculate variance covariance matrix using A and B
        {
          A_n <- 1/(n_m <- length(psi_list)) * Reduce("+", neg_jacobian_list)
          A_n_inv <- try(solve(A_n), T)
          if(all( "try-error" %in% class(A_n_inv))){warning("Singular A_n -- Forcing Correction"); A_n_inv <- solve(A_n+0.01*diag(nrow(A_n)))}
          B_n <- 1/n_m * Reduce("+", lapply(psi_list,function(psi_){ psi_ %*% t(psi_) }))
          VarCov_n_automatic <- 1 / n_m * ( A_n_inv %*% B_n %*% t(A_n_inv) )
        }
      }
      Mean_n_automatic <- c("Qhat"=Qhat, unlist(  lapply(tv_trainWith,function(zer){
        values_ <- as.numeric( jnp$reshape(zer,-1L) )
        if(prod(dim(zer))>0){
          names(values_) <- paste(zer$name,1:length(values_),sep="_")
        }
        return( values_ )
      }) ) )
      #(row.names(VarCov_n_automatic))==(names(Mean_n_automatic))
      colnames(VarCov_n_automatic) <- row.names( VarCov_n_automatic )  <- names(Mean_n_automatic)

      # obtain uncertainties
      {
        m_se_Q   = sqrt( VarCov_n_automatic[1,1] )
        seList_automatic <- getPiList(simplex = T, rename= T,return_SE = T,
                                      VarCov = VarCov_n_automatic)$FinalSEList

        # projection coefficients + SEs
        if(K > 1){
          BinaryCovariateIndicator <- apply(X_factorized, 2, function(zer){names(table(zer))})
          BinaryCovariateIndicator <- unlist( lapply(BinaryCovariateIndicator,function(zer){mean(zer[1:2] == c("0","1") & length(zer)==2)}))
          X_completion <- sapply(which(BinaryCovariateIndicator==0),function(zer){
            zap_ <- X_factorized[,zer]
            if(length(zap_) < 10){
              zap_completion <- eval(parse(text = sprintf("
            model.matrix(~0+factor(%s),data = as.data.frame(X_factorized))",
                                                          colnames(X_factorized)[zer])))
            }
            if(length(zap_) > 10){
              X_factorized_ <- as.data.frame( X_factorized )
              eval(parse(text = sprintf("
                      X_factorized_$%s_quant <- as.numeric( gtools::quantcut(zap_, q = 5) )
                                    ", colnames(X_factorized)[zer])))
              zap_completion <- eval(parse(text = sprintf("
            model.matrix(~0+factor(%s_quant),data = (X_factorized_))",
                                                          colnames(X_factorized_)[zer])))
            }
            return( list(zap_completion ))
          })
          X_completion <- do.call(cbind, X_completion)
          X_factorized_complete <- cbind(
            X_factorized[,which(BinaryCovariateIndicator==1)],
            X_completion)
          which_factor <- unlist( lapply(strsplit(colnames( X_factorized_complete ),split="factor\\("),function(xer){xer[2]}) )
          which_factor_names <- which_factor <- unlist(lapply(strsplit(which_factor,split="\\)"),function(zer){zer[1]}))
          which_factor <- cumsum(!duplicated(which_factor))
          which_factor_mat <- jnp$array( model.matrix(~0+factor(which_factor)),jnp$float32)
          # some tests
          {
            #\sum X * Pr(Clust|X) 1/n
            #1  = sum_x Pr(X_d=x | Clust)
            #Pr(X_d=x | Clust) =
            #\sum_x' Pr(X_d=x,X_d'=x'|Clust) =
            #\sum_x' Pr(Clust|X_d=x,X_d'=x') Pr(Clust) / Pr(X_d=x,X_d'=x') =
            #\sum_x' Pr(Clust|X_d=x,X_d'=x') Pr(Clust) / U =
            #1/3*1/3 / (0.5)

            # Pr(Clust|X_d=x,X_d'=x')
            # Pr(Clust) /
            #  sum_x Pr(Clust|X) Pr(Clust)
            #E[X|K=k]
          }
          with(tf$GradientTape(persistent = T) %as% tape, {
            PrClustGivenX <- getClassProb(jnp$array(X, jnp$float32))

            # for checking:
            #PrClustGivenX <- jnp$array(matrix(rep(c(1/2), times = length( PrClustGivenX )), ncol=2),jnp$float32)
            #PrClustGivenX <- jnp$array(matrix(rep(c(0.9,0.1), times = length( PrClustGivenX )/2), ncol=2,byrow=T),jnp$float32)
            if(T == F){
              # testing
              PrClustGivenX_orig <- PrClustGivenX <- jnp$array(
                cbind(1*X_factorized_complete[,3] == max(X_factorized_complete[,3]),
                      1*X_factorized_complete[,3] == min(X_factorized_complete[,3])),jnp$float32)
            }
            PrClust <- jnp$mean(PrClustGivenX,0L,keepdims=T)
            PrClustGivenX <- jnp$expand_dims(PrClustGivenX,2L)
            X_factorized_expand <- jnp$expand_dims(jnp$array(X_factorized_complete,jnp$float32),1L)

            #Pr(X=1 | Clust = k) = Pr(Clust = k | X = 1) Pr(X = 1) / Pr(Clust = k)
            TotalNumberWithXd <- jnp$sum(jnp$multiply(X_factorized_expand,jnp$expand_dims(jnp$transpose(which_factor_mat),0L)),0L:1L)
            # think of PrClustGivenXd as mean(as.matrix(PrClustGivenX_orig)[,2][X_factorized_complete[,3] == 1])
            PrClustGivenXd <- jnp$divide(jnp$sum(jnp$multiply(PrClustGivenX,X_factorized_expand),0L),TotalNumberWithXd)
            PrXd <- jnp$expand_dims(TotalNumberWithXd / nrow(X_factorized_complete),0L)
            PrXdGivenClust <- jnp$divide(jnp$multiply(PrClustGivenXd,PrXd),jnp$transpose(PrClust))

            #PrXdGivenClust <- jnp$sum(jnp$multiply(PrClustGivenX,X_factorized_expand),0L)
            #PrXd <- jnp$matmul(jnp$matmul(PrXdGivenClust,which_factor_mat),jnp$transpose(which_factor_mat))
            #PrXdGivenClust <- PrXdGivenClust / PrXd
            #orig_ <- as.matrix(PrXdGivenClust)
            # plot(c(as.matrix(PrXdGivenClust)),c(as.matrix(orig_)));abline(a=0,b=1)
          })
          #Pr(Xd|Clust) = Pr(Xd AND Clust 1) = Pr(Xd | Clust 1) Pr(Clust 1)
          #E[X|Clust]  = sum_x x Pr(X=x|Clust) = sum_x x (Pr(Clust | X) Pr(X) / Pr(Clust))
          # Pr(X | Clust) = Pr(Clust | X) Pr(X) / Pr(Clust)

          dPrXdGivenClust_dParam_orig <- dPrXdGivenClust_dParam <- tape$jacobian( PrXdGivenClust,
                                                                                  ClassProbProj$trainable_variables )
          dPrXdGivenClust_dParam <- lapply(dPrXdGivenClust_dParam,function(zer){
            if(length(dim(zer))==3){ zer <- jnp$expand_dims(zer,3L) }
            return( zer)  })
          dPrXdGivenClust_dParam <- lapply(dPrXdGivenClust_dParam,function(zer){
            jnp$reshape(zer,c(dim(zer)[1:2],-1L))})
          dPrXdGivenClust_dParam <- jnp$concatenate(dPrXdGivenClust_dParam,2L)
          PrXdGivenClust_mat <- np$array(PrXdGivenClust)
          row.names(PrXdGivenClust_mat) <- paste("k",1:K,sep="")
          colnames(PrXdGivenClust_mat) <- colnames(X_factorized_complete)
          PrXd_vec <- colMeans(X_factorized_complete)

          kernel_indices <- grep(row.names(VarCov_n_automatic), pattern="ClassProbProj/kernel")
          bias_indices <- grep(row.names(VarCov_n_automatic), pattern="ClassProbProj/bias")
          VarCov_n_ProbClustParam <- VarCov_n_automatic[c(kernel_indices,bias_indices),
                                                        c(kernel_indices,bias_indices)]
          VarCov_n_ProbClustParam <- jnp$array(VarCov_n_ProbClustParam,jnp$float32)
          VarCov_n_ProbClustParam <- jnp$expand_dims(VarCov_n_ProbClustParam,0L)
          PrXdGivenClust_se_tf <- jnp$matmul(
            jnp$matmul(dPrXdGivenClust_dParam, VarCov_n_ProbClustParam),
            jnp$transpose(dPrXdGivenClust_dParam,c(0L,2L,1L)))
          PrXdGivenClust_se_tf <- jnp$diag(PrXdGivenClust_se_tf)
          PrXdGivenClust_se <- np$array(PrXdGivenClust_se_tf)
          row.names(PrXdGivenClust_se) <- row.names(PrXdGivenClust_mat)
          colnames(PrXdGivenClust_se) <- colnames(PrXdGivenClust_mat)
        }
      }
      if(any(is.na(unlist(seList_automatic)))){browser()}

      for(k__ in 1:K){
        try(seList_automatic[[k__]] <- sapply(1:length(hypotheticalProbList[[k__]]),function(d_){
          names(seList_automatic[[k__]][[d_]]) <- names(  hypotheticalProbList[[k__]][[d_]] )
          list(seList_automatic[[k__]][[d_]]) }), T)
        try(  names(seList_automatic[[k__]]) <- names( hypotheticalProbList[[k__]]) , T)
      }

      names(hypotheticalProbList_full) <- paste("k",1:length(hypotheticalProbList_full),sep = "")
      names(seList_automatic) <- paste("k",1:length(seList_automatic),sep = "")

      seList <- seList_automatic
    }
  }
  #\ sum_i X_i Pr(Z_k = k)
  # \sum_i X_i

  if(( piSEtype  == "manual"  | piSEtype == "both") & useVariational==F ){

    EXPERIMENTAL_SCALING_FACTOR <- 1 #length(split2_indices)
    genPsi_vec <- function(           PI,
                                      Yobs_DATA ,
                                      pi_forYMean_outer,
                                      pi_forYVar_outer,
                                      pi_forQ_outer ,
                                      pi_forGrad_simplexListed_outer,
                                      log_pr_w_DATA,
                                      FactorsMat_DATA,
                                      GradComp_DATA,
                                      GradContribInfo_DATA,
                                      openBrowser__ = F,
                                      USE_PRECOMPUTED             ){

      # open browser
      if(openBrowser__ == T){ browser() }

      # arrange components of pi
      {
        pi_forQ <- PI[1]
        if(USE_PRECOMPUTED == T){
          pi_forGrad_simplexListed <- pi_forGrad_simplexListed_outer
        }
        if(USE_PRECOMPUTED == F){
          #pi_forGrad_simplexListed <- vec2list( PI[-c(1:3)] )
          pi_forGrad_simplexListed <- vec2list( PI[-c(1)] )
        }
      }

      # simple parameter components
      {
        # sum Y mean comp
        #Ymean_comp <- sum( Yobs_DATA  -   pi_forYMean )

        # sum Yvar comp
        #YVar_comp <- sum( (Yobs_DATA  -   pi_forYMean)^2 - pi_forYVar )

        # Q hat with 1 observation, known normalization
        Qhat__sumComp <- computeQ_conjoint_internal(FactorsMat_internal = FactorsMat_DATA,
                                                    Yobs_internal = Yobs_DATA,
                                                    knownNormalizationFactor = NULL, hajek = F,
                                                    log_pr_w_internal = log_pr_w_DATA,
                                                    assignmentProbList_internal = p_list,
                                                    hypotheticalProbList_internal = pi_forGrad_simplexListed)

        # Q comp
        prob_ratio <- exp(Qhat__sumComp$log_pr_w_new - Qhat__sumComp$log_PrW)
        if(forceHajek == F){ Q_comp <- sum(Yobs_DATA * prob_ratio  -   pi_forQ )} # without hajek
        if(forceHajek == T){ Q_comp <- sum(Yobs_DATA * prob_ratio  -   pi_forQ * prob_ratio) }# with hajek
      }

      # gradient comp
      if(USE_PRECOMPUTED == T){
        GradContribInfo <- GradContribInfo_DATA
        Grad_comp <- GradComp_DATA
      }
      if(USE_PRECOMPUTED == F){
        GradContribInfo <- GRADIENT_FXN_vectorized(FactorsMat_DATA = FactorsMat_DATA,
                                                   Yobs_DATA = Yobs_DATA,
                                                   log_pr_w_DATA = log_pr_w_DATA,
                                                   pi_forGrad_simplexListed = pi_forGrad_simplexListed,
                                                   SIGMA2_ = pi_forYVar,
                                                   n_in_sum_DATA = (n_in_sum_Mest <- length(Y)) )
        Grad_comp <- unlist(  GradContribInfo["full_grad",] )
      }

      psi_vec <- c( Q_comp, Grad_comp )
      return( list("psi_vec"=psi_vec,
                   "GradContribInfo"=GradContribInfo) )
    }

    getPsiInfo_unnormalized <- function(pi,Yobs_DATA,log_pr_w_DATA,
                                        FactorsMat_DATA,
                                        GradComp_DATA, GradContribInfo_DATA,
                                        use_precomputed = T){

      # arrange components of pi
      {
        #pi_forYMean_outer    <- pi[1]
        #pi_forYVar_outer     <- pi[2]
        pi_forQ_outer        <- pi[1]
        if(use_precomputed == F){
          #pi_forGrad_simplexListed_outer    <<- vec2list( pi[-c(1:3)] )
          pi_forGrad_simplexListed_outer    <<- vec2list( pi[-c(1)] )
        }
      }

      # obtain psi vector
      psi_vec       <-  genPsi_vec( PI = pi,
                                    Yobs_DATA = Yobs_DATA,
                                    pi_forYMean_outer = pi_forYMean_outer,
                                    pi_forYVar_outer = pi_forYVar_outer,
                                    pi_forQ_outer = pi_forQ_outer,
                                    pi_forGrad_simplexListed_outer = pi_forGrad_simplexListed_outer,
                                    log_pr_w_DATA = log_pr_w_DATA,
                                    GradComp_DATA = GradComp_DATA,
                                    GradContribInfo_DATA = GradContribInfo_DATA,
                                    FactorsMat_DATA = FactorsMat_DATA,
                                    USE_PRECOMPUTED = use_precomputed)
      GradContribInfo <-  psi_vec$GradContribInfo
      psi_vec         <-  psi_vec$psi_vec

      # Calculate A,B contributions to obtain Hessian
      Jacobian_mat <- JACOBIAN_MAT
      #Jacobian_mat[1,1] <- -1
      #Jacobian_mat[2,1:2] <- c(-2*(Yobs_DATA - pi_forYMean_outer), -1)
      if(forceHajek == F){ stop("Gradient not correct");Jacobian_mat[1,] <- c(-1, unlist(  GradContribInfo["Q_contrib_grad",] - 1 )) }
      if(forceHajek == T){
        # Component is: Y__*probRatio__ - Qhat_tf*probRatio__
        # Q_contrib_grad is defined as  Yobs_DATA * pr_ratio_grad
        #Jacobian_mat[1,] <- c( -GradContribInfo["pr_ratio",][[1]][1], unlist(  GradContribInfo["Q_contrib_grad",] ) - pi_forQ_outer * unlist(  GradContribInfo["pr_ratio_grad",] ) )
        Jacobian_mat[1,] <- c( -GradContribInfo["pr_ratio",][[1]][1],
                               unlist(  GradContribInfo["Q_contrib_grad",] ) -
                                 pi_forQ_outer * unlist(  GradContribInfo["pr_ratio_grad",] ) )
        #Jacobian_mat[3,] <- c(0, 0, -GradContribInfo["pr_ratio",][[1]][1],
        #unlist(  GradContribInfo["Q_contrib_grad",] ) - pi_forQ_outer * unlist(  GradContribInfo["pr_ratio_grad",] ) )
      }
      #Q_comp <- sum(Yobs_DATA * prob_ratio  -   pi_forQ * prob_ratio)
      #if(PenaltyType != "L2"){
      #  Jacobian_mat[-c(1),2] <-  - LAMBDA_/n_target * 1/n_in_sum_Mest * treatment_combs * unlist(GradContribInfo["maxProb_grad",])
      #}

      HESSIAN <- HESSIAN_FXN(GradContribInfo = GradContribInfo,
                             FactorsMat_DATA= as.vector(unlist(FactorsMat_DATA)),
                             Yobs_DATA = Yobs_DATA,
                             log_pr_w_DATA = log_pr_w_DATA,
                             pi_forYVar_outer = varHat,
                             pi_forGrad_simplexListed_outer = pi_forGrad_simplexListed_outer,
                             n_in_sum_DATA = n_in_sum_Mest)
      Jacobian_mat[-c(1),-c(1)] <-  HESSIAN$HESSIAN

      return( list("psi_vec"=psi_vec,
                   "Jacobian_mat"=Jacobian_mat) )
    }
    Qhat_hajek_split2 <- sum( Qhat_split2$Y * exp(Qhat_split2$log_pr_w_new-Qhat_split2$log_PrW) / sum( exp(Qhat_split2$log_pr_w_new-Qhat_split2$log_PrW) ))
    pi_unnormalized <- c( Qhat_hajek_split2, optim_max_hajek_full )

    # all gradient info - all observations -  later we process this down to split2
    JACOBIAN_MAT <- matrix(0,nrow = length(pi_unnormalized),ncol=length(pi_unnormalized))
    GradContribInfo_allObs <- GRADIENT_FXN_vectorized(FactorsMat_DATA = FactorsMat_numeric,
                                                      Yobs_DATA = Y,
                                                      log_pr_w_DATA = log_PrW,
                                                      pi_forGrad_simplexListed = vec2list(optim_max_hajek_full),
                                                      SIGMA2_ = varHat,
                                                      n_in_sum_DATA = length( Y ) )
    GradComp_allObs <- do.call(cbind, GradContribInfo_allObs["full_grad",])
    GradContribInfo_allObs <- apply(GradContribInfo_allObs,1,function(ze){
      unitwiseCompIndicator <-  length(unlist(ze)) %% length(Y) == 0
      if(!unitwiseCompIndicator){ return(  matrix(do.call(c,ze)) )  }
      if( unitwiseCompIndicator){ return( do.call(cbind,ze) )  }
    })

    pi_forGrad_simplexListed_outer <- vec2list(optim_max_hajek_full)
    if(length(pi_forGrad_simplexListed_outer) == 2){print("NEED TO HANDLE k = 2 case!");browser()}
    maxDim <- max(unlist( lapply(pi_forGrad_simplexListed_outer,length)))
    pi_forGrad_minus_mat_list <- sapply(1:nrow(expanded_kl),function(ja){
      pi_forGrad_minusk_tmp <- pi_forGrad_simplexListed_outer[-unique(expanded_kl[ja,c(1,3)])]
      list( pi_forGrad_minusk_mat <- t(do.call(rbind,lapply(pi_forGrad_minusk_tmp,function(ze){
        c(ze,rep(0,times=maxDim-length(ze)))           }))) ) })
    pi_forGrad_minusk_mat_vec_comp <- nrow(pi_forGrad_minus_mat_list[[1]])*(1:ncol(pi_forGrad_minus_mat_list[[1]])-1)
    pi_forGrad_minuskkt_mat_vec_comp <- nrow(pi_forGrad_minus_mat_list[[1]])*(1:min(unlist(lapply(pi_forGrad_minus_mat_list,ncol))) - 1)

    {
      DD_L2Pen <- Reduce("+",sapply(1:length(pi_forGrad_simplexListed_outer),function(K___){
        pi_k <- pi_forGrad_simplexListed_outer[[K___]]
        p_k <- p_list[[K___]]

        DD_ <- matrix(0,nrow = length(pi_k)-1,ncol=length(pi_k)-1)
        DD_[] <- apply(GRID_LIST[[K___]],1,function(ze){
          pi_kl <- pi_k[ze[1]]; pi_klt <- pi_k[ze[2]]
          p_kl <- p_k[ze[1]]; p_klt <- p_k[ze[2]];
          if(ze[1] == ze[2]){
            pi_kminusl <- pi_k[-ze[1]]
            p_kminusl <- p_k[-ze[1]]
            VAL <- 2*(pi_kl*(1-pi_kl)*(pi_kl-pi_kl^2) - pi_kl^2 * (1-pi_kl)*(pi_kl-p_kl) +
                        pi_kl*(1-pi_kl)*(pi_kl-p_kl) + pi_kl*(pi_kl^2-pi_kl)*(pi_kl-p_kl) -
                        sum( pi_kl*pi_kminusl*(pi_kminusl - p_kminusl) - 2*pi_kl^2*pi_kminusl*(pi_kminusl-p_kminusl) - pi_kl^2*pi_kminusl^2 ))
          }
          if(ze[1] != ze[2]){
            pi_kminusllt <- pi_k[-c(ze)]
            p_kminuslllt <- p_k[-c(ze)]
            VAL <- 2*( pi_kl^2*pi_klt*(pi_kl - p_kl) - pi_kl^2 * pi_klt * (1 - pi_kl) - pi_kl * pi_klt * (1-pi_kl)*(pi_kl-p_kl) -
                         (pi_kl*pi_klt*(pi_klt-pi_klt^2) - 2 *pi_kl*pi_klt^2 * (pi_klt - p_klt) + pi_kl*pi_klt*(pi_klt-p_klt) -
                            sum( pi_kl * pi_klt * pi_kminusllt^2 + 2 * pi_kl * pi_klt * pi_kminusllt * (pi_kminusllt-p_kminuslllt) )  ))
          }
          return( VAL ) })
        DD_full[(DD_full_indices[K___]+1):DD_full_indices[K___+1],
                (DD_full_indices[K___]+1):DD_full_indices[K___+1]]  <- DD_
        list(  DD_full )
      }) )
    }

    An_Bn <- sapply(split2_indices,function(iiii){
      GradContribInfo_iii <- lapply(GradContribInfo_allObs,function(comp_){
        tmp1_ <- nrow(comp_) == length(Y)
        tmp2_ <- ncol(comp_) == 1
        if(tmp1_){comp_ <- comp_[iiii,]}
        if(tmp2_){comp_ <- c(comp_)}
        list(comp_) })
      GradContribInfo_iii <- do.call(rbind,GradContribInfo_iii)
      row.names(GradContribInfo_iii) <- names(GradContribInfo_allObs)
      psiInfo_iii <- getPsiInfo_unnormalized(pi = pi_unnormalized,
                                             Yobs_DATA     = Y[iiii],
                                             log_pr_w_DATA = log_PrW[iiii],
                                             GradComp_DATA = GradComp_allObs[iiii,],
                                             GradContribInfo_DATA = GradContribInfo_iii,
                                             FactorsMat_DATA = as.data.frame( t(FactorsMat_numeric[iiii,])),
                                             use_precomputed = T)
      A_iii <-    -1 * psiInfo_iii$Jacobian_mat
      B_iii <- psiInfo_iii$psi_vec %*% t(psiInfo_iii$psi_vec)
      return(     list(A_iii, B_iii, psiInfo_iii$psi_vec)    )
    })

    A_n <- 1/ncol(An_Bn) * Reduce("+", An_Bn[1,])
    B_n <- 1/ncol(An_Bn) * Reduce("+", An_Bn[2,])
    #Jacobian_numerical <- numDeriv::hessian(function(th_){return( minThis_fxn(th_))}, optim_max_hajek_)
    #plot(c(A_n[-c(1:3),-c(1:3)]),c(Jacobian_numerical));abline(a=0,b=1)
    #plot(c(Reduce("+", An_Bn[2,])[-c(1:3),-c(1:3)]),c(Jacobian_numerical));abline(a=0,b=1)
    #sum_psi <- Reduce("+", An_Bn[3,]); plot( abs(sum_psi)+0.01,log = "y" );abline(h=0.01)
    A_n_inv <- try(solve( A_n ), T)
    if(any("try-error" %in% class(A_n_inv))){stop("A_n cannot be inverted!")}
    VarCov_n_manual <- 1 / ncol(  An_Bn  ) * (  A_n_inv %*% B_n %*% t( A_n_inv )  )

    # obtain variance or var/cov info
    m_se_Q   = sqrt( VarCov_n_manual[1,1] )
    m_mean_max = pi_unnormalized[-c(1)]
    m_cov_max  = VarCov_n_manual[-c(1),-c(1)]

    transformation_list <- zeros_vec;transformation_list[] <- "x"
    transformation_list[nonZero_indices] <- 1:length(m_mean_max)
    transformation_list <- strsplit(paste(transformation_list,collapse=" "),split= "x")[[1]][-1]
    transformation_list <- sapply(transformation_list,function(ze){
      zee <- strsplit(ze,split = ' ')[[1]]
      my_string <- sapply(zee,function(k__){
        if(k__ == ""){str_ <- sprintf("~exp(0)/(exp(0)+%s)", paste( paste(  paste("exp(x",zee[-1],sep=""),")",sep=''),collapse="+")) }
        if(k__ != ""){str_ <- sprintf("~exp(x%s)/(exp(0)+%s)", k__, paste( paste(  paste("exp(x",zee[-1],sep=""),")",sep=''),collapse="+")) }
        return( str_ )
      })
      return( my_string) })
    transformation_list <- sapply(sapply( unlist(transformation_list),function(ze){as.formula(ze)}),list)
    names(transformation_list) <- paste("x",1:length(transformation_list),sep="")
    optim_max_SEs_mEst_vec <- optim_max_SEs_mEst <- msm::deltamethod(transformation_list,
                                                                     mean = m_mean_max,
                                                                     cov  = m_cov_max,
                                                                     ses = T) # returns SE, NOT Var
    seList_manual <- vec2list_noTransform(optim_max_SEs_mEst)
    names(upperList) <- names(lowerList) <- names(seList_manual) <- names( p_list )
    seList <- seList_manual
  }

  # get upper and lower lists
  ensure0t1 <- function(ze){ ze[ze<0] <- 0; ze[ze>1] <- 1;ze}
  for(sign_ in c(-1,1)){
    boundsList <- sapply(1:length(hypotheticalProbList_full),function(k_){
      k_res <- sapply(1:length(hypotheticalProbList_full[[k_]]),function(d_){
        tmp_ <- hypotheticalProbList_full[[k_]][[d_]] + sign_*abs(qnorm((1-confLevel)/2))*seList[[k_]][[d_]]
        list( tmp_ )
      })
      names(k_res) <- names( hypotheticalProbList_full[[k_]] )
      return( list(k_res) )
    })
    if(sign_ == -1){lowerList <- boundsList}
    if(sign_ == 1){upperList <- boundsList}
  }
  names(upperList) <- names(lowerList) <-  names( hypotheticalProbList_full )
}
