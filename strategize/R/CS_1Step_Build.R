ml_build <- function(){

  zero_ep <- jnp$array(1e-5,jnp$float32)
  print("Begin building block...")

  for(d_ in 1:length(p_list)){
    eval(parse(text = sprintf("ilrBasis%s <- jnp$array(compositions::ilrBase(D = length(p_list[[d_]])),jnp$float32)",d_)))
    eval(parse(text = sprintf("ilrBasis%s_t <- jnp$transpose(ilrBasis%s)",d_,d_)))
  }
  ilrBasisClust <- jnp$array(compositions::ilrBase(D = K),jnp$float32)
  ilrBasisClust_t <- jnp$expand_dims(jnp$transpose(ilrBasisClust),0L)
  Zero1by1 <- jnp$array(as.matrix(0.),jnp$float32)

  # test
  #compositions::ilr(t(c(0.49,0.51))) - t(log(c(0.49,0.51))) %*% compositions::ilrBase(D=2)

 # setup main functions
  useHajekInOptimization_orig <- useHajekInOptimization
  if(penaltyType == "LogMaxProb"){
    RegularizationPiAction <- 'jnp$add(tf$reduce_max(( (  (log_pidk) ) )),
    tf$math$log(tf$squeeze(jnp$divide(jnp$take(ClassProbsMarginal, k_ - 1L, axis = 1L),DFactors)))'
  }
  if(penaltyType == "L2"){
    ## if equal weighting of the penalty by cluster
    if(K == 1){ RegularizationPiAction <- 'tf$reduce_sum(jnp$square( jnp$subtract( pd%s, jnp$exp(log_pidk) ) ))' }

    ## if weighting the penalty by the marginal probability for that factor
    # approach 1: (pi_z - p)^2 * pr(z)
    #if(K > 1){ RegularizationPiAction <- 'jnp$multiply(tf$reduce_sum(jnp$square( jnp$subtract( pd%s, jnp$exp(log_pidk) ) )),
               #tf$squeeze(jnp$take(ClassProbsMarginal, k_ - 1L, axis = 1L)))' }

    # approach 2: mean( ( (p_z-pi) * pr(z|x) )^2 )
    if(K > 1){ RegularizationPiAction <- 'tf$reduce_sum(tf$reduce_mean(jnp$square(jnp$multiply( jnp$subtract( pd%s, jnp$exp(log_pidk) ) ,
                                          jnp$expand_dims(jnp$take(ClassProbs, k_ - 1L, axis = 1L),0L) )),1L))' }

  }
  for(forMEst in c(T,F)){
    if(forMEst == T){subtypes <- "probRatio"; useHajekInOptimization <- F; return_text <- 'returnThis_ <- minThis_;' }
    if(forMEst == F){ subtypes <- c("probRatio","objToMin") }
  for(subtype in subtypes){
    if(forMEst == F){
      useHajekInOptimization <- useHajekInOptimization_orig;
      if(subtype == "probRatio"){ return_text <- 'returnThis_ <- probRatio;'}
      if(subtype == "objToMin"){ return_text <- 'returnThis_ <- minThis_;'}
    }
    if(useVariational == T){
      regularizationContrib <- 'jnp$multiply(REGULARIZATION_LAMBDA, RegularizationContribPi)'
    }
    if(useVariational == F & K > 1){
      regularizationContrib <- 'jnp$add(jnp$multiply(REGULARIZATION_LAMBDA, RegularizationContribPi),
                                                jnp$multiply(lambda_coef, tf$reduce_mean(jnp$square(ClassProbProj$kernel))) )'
    }
    if(useVariational == F & K == 1){
      regularizationContrib <- 'jnp$multiply(REGULARIZATION_LAMBDA, RegularizationContribPi)'
    }
    if(useHajekInOptimization == F){
      minThis_text <- sprintf('minThis_ <- jnp$add(jnp$negative(tf$reduce_mean( jnp$multiply(Y_,probRatio) )), %s);',regularizationContrib)
    }
    if(useHajekInOptimization == T){
      minThis_text <- sprintf('minThis_ <- jnp$add(jnp$negative(tf$reduce_sum( tf$truediv(jnp$multiply(Y_,probRatio),tf$reduce_sum(probRatio)))), %s );',regularizationContrib)
    }
    # define main function
    {
      base_tf_function <- eval(parse(text = paste('myLoss_or_returnWeights <- (function(Y_, X_,factorMat_,logProb_,REGULARIZATION_LAMBDA){
          if(K == 1){ logClassProbs <- jnp$zeros(tf$shape(Y_),jnp$float32)  }
          if(K >  1){
              logClassProbs <- tf$math$log( ClassProbs <- getClassProb(  X_ )  )
              ClassProbsMarginal <- tf$reduce_mean(ClassProbs, 0L, keepdims=T)
          }

          RegularizationContribPi <- jnp$zeros( list() )
          logPrWGivenAllClasses <- replicate(  K, list()   )

          # Approach:
          # Pr(W) = sum_k Pr(W, k) = sum_k Pr(W | K = k) Pr(K = k) = sum_k prod_d [Pr(W_d | K=k)] Pr(K=k)

          # iterate over clusters
          for(k_ in 1L:K){

            # log( prod_d [Pr(W_d|K=k)] Pr(K=k) ) =  [sum_d { log(Pr(W_d|K=k)) }] + log( Pr(K=k) )
            logPrW_given_k <- jnp$expand_dims(jnp$take(logClassProbs, indices = k_ - 1L, axis=1L),1L)

            # iterate over factors
            for(d__ in 1L:DFactors){
             if(useVariational == T){
              if(normType == "alr"){
                 log_pidk <- tf$math$log_softmax(tf$concat(list( Zero1by1,
                    eval(parse(text=sprintf("av%sk%s_()",d__, k_))) ),0L),axis = 0L)
              }
             }
             if(useVariational == F){
              if(normType == "ilr"){
                  log_pidk <- eval(parse(text=sprintf("av%sk%s",d__, k_)))
                  basis__t <- eval(parse(text=sprintf("ilrBasis%s_t",d__)))
                  log_pidk <- jnp$transpose(jnp$exp(tf$matmul(jnp$transpose(log_pidk), basis__t)))
                  pidk_denom <- jnp$add(zero_ep, jnp$sum(log_pidk,axis = 0L, keepdims=T))
                  log_pidk <- jnp$divide(log_pidk, pidk_denom)
                  log_pidk <- tf$math$log( log_pidk )
              }
              if(normType == "alr"){
                  log_pidk <- tf$math$log_softmax(tf$concat(list( Zero1by1,
                                         eval(parse(text=sprintf("av%sk%s",d__, k_))) ),0L),axis = 0L)
              }
             }
              received_wd <- jnp$take(factorMat_, indices = d__ - 1L, axis = 1L)
              log_PrWd_given_k <- jnp$take(log_pidk, indices = received_wd, axis = 0L)

              logPrW_given_k <- jnp$add(logPrW_given_k, log_PrWd_given_k)
              RegularizationContribPi <- RegularizationContribPi + eval(parse(text=sprintf(" ', RegularizationPiAction,' ",d__)))
            }
            logPrWGivenAllClasses[[k_]] <- logPrW_given_k
          }
          logPrW_new <- tf$reduce_logsumexp(tf$concat(logPrWGivenAllClasses, 1L),keepdims = T, axis =  1L)
          probRatio <- jnp$exp( jnp$subtract(logPrW_new,  logProb_ ));',
          minThis_text, return_text,'return(   returnThis_    )})', sep = "")))
      }
    #tf_function_use <- function(x){x};print("Using non-compiled fxns") # for debugging
    tf_function_use <- tf_function; print("Using compiled fxns") # for performance
    if(forMEst == T){
      getLoss_tf_unnormalized <- tf_function_use(   base_tf_function  )
    }
    if(forMEst == F){
      if(subtype == "probRatio"){
        getProbRatio_tf <- tf_function_use(   base_tf_function  )
      }
      if(subtype == "objToMin"){
        getLoss_tf <- tf_function_use(   base_tf_function  )
      }
    }
    rm( myLoss_or_returnWeights_tf_ )
  }
  }

## start  building model
{
  # define the class prob projection, the trainable vars, the baseline probs
  KFreeProbProj <- as.integer(K - 1L)
  b_proj <- 0.1
  if(KFreeProbProj > 0){
    value_seq <- sapply(b_seq <- 10^seq(-10,1,length.out=100),function(b_){
       value_ <- mean(replicate(25,{
        X_ <- X[sample(1:nrow(X),batch_size),]
        #p_ <- X_ %*% matrix(rnorm(ncol(X_)*KFreeProbProj,mean=0,sd=b_),ncol=KFreeProbProj)
        if(useVariational == T){  p_ <- X_ %*% matrix(rnorm(ncol(X_)*KFreeProbProj,mean=0,sd=b_),ncol=KFreeProbProj) }
        if(useVariational == F){  p_ <- X_ %*% matrix(runif(ncol(X_)*KFreeProbProj,-b_,b_),ncol=KFreeProbProj) }
        #if(normType == "alr"){ penalty_val <- max(apply(p_,1,function(zer){max( prop.table(exp(c(0,zer))) ) })) }
        if(normType == "alr"){
          penalty_val <- mean(apply((apply(p_,1,function(zer){ prop.table(exp(c(0,zer))) })),1,sd))
        }
        if(normType == "ilr"){ penalty_val <- max(apply(p_,1,function(zer){
            f2n(c(compositions::ilrInv(t(zer))))
            }))
        }
        penalty_val
      }))
  })
    #value_seq <- value_seq - 1 / K
    b_proj <- b_seq[which.min(abs(value_seq - CLUSTER_EPSILON))]
    print(sprintf("Uniform init bounds for kernel: %.5f",b_proj))
  }

  if(useVariational == T){
    ClassProbProj <- tfp$layers$DenseReparameterization(KFreeProbProj ,
                              #kernel_prior_fn = tfp$layers$default_mean_field_normal_fn(is_singular=F,
                              #loc_initializer = tf$initializers$random_normal(mean = 0., stddev=0.01),
                              #untransformed_scale_initializer = tf$initializers$random_normal(mean=b_proj,stddev=0.01)),
                              activation = "linear",name="ClassProbProj" )
  }
  if(useVariational == F){
    ClassProbProj <- tf$keras$layers$Dense(units = KFreeProbProj, activation = "linear",name="ClassProbProj", bias_initializer  = jnp$zeros,
                                      kernel_initializer = tf$keras$initializers$random_uniform(minval = -b_proj, maxval = b_proj),
                                      trainable = T)
  }
  if( normType == "ilr"){
    getClassProb <- tf_function( function(x){
      # k = 1 case
      if(KFreeProbProj == 0){ x <- tf$ones(list(tf$shape(x)[1],1L)) }

      # k > 1 case
      if(KFreeProbProj > 0){
        jnp$transpose(ilrBasisClust_t, list(0L,2L,1L))
        x <- ClassProbProj(x)
        #x <- jnp$exp(tf$squeeze(jnp$multiply(jnp$expand_dims(x,2L),ilrBasisClust_t),1L))

        # CHECK THE FOLLOWING LINE
        x <- jnp$exp(tf$squeeze(tf$matmul(jnp$transpose(ilrBasisClust_t, list(0L,2L,1L)),
                                        jnp$expand_dims(x,2L)),2L))
        x <- jnp$divide(x, tf$reduce_sum(x,axis = 1L,keepdims=T))
      }
      return(  x ) } )
  }
  if( normType == "alr"){
    getClassProb <- tf_function(function(x){
      return(  tf$keras$activations$softmax( tf$concat( list(jnp$zeros(c(nrow = nrow(x),1L),
                                                                      dtype=jnp$float32),ClassProbProj(x)),1L ) ,1L ) ) })
  }
  #ClassProbProj(jnp$array(X,jnp$float32))
  print(paste("Initial Class Probs: ", paste(round(colMeans(as.array(getClassProb(jnp$array(X,jnp$float32) ))),7L),collapse=','),sep=""))
  DFactors <- length( nLevels_vec <- apply(W,2,function(l){length(unique(l))}) )

  if(useVariational == T){
    KERNEL_LOC_INIT  <- as.matrix(ClassProbProj$kernel_posterior$trainable_variables[[1]])
    KERNEL_SCALE_INIT <- as.matrix(ClassProbProj$kernel_posterior$trainable_variables[[2]])
  }

  # initialize pd -  assignment probabilities for penalty and pr(w)
    for(d_ in 1:(DFactors <- length(nLevels_vec))){
        eval(parse(text=sprintf( "pd%s = jnp$array(as.matrix(p_list[[d_]]), dtype = jnp$float32,name='pd%s')",d_,d_)) )
    }

  # initialize av- the new probability generators
  initialize_avs <- function(b_SCALE=1){
    for(k_ in 1:K){ for(d_ in 1:DFactors){
      if(useVariational == T){
        sd_d <- f2n(attr(pi_init_list[d_,k_,1L][[1]],"random_"))[1]
        sd_d <- jnp$array(as.matrix(sd_d),dtype=jnp$float32)
        Zeros_init <- jnp$array(as.matrix(rep(0,times=((nrow(pi_init_list[d_,k_,1L][[1]]))))),dtype=jnp$float32)
        Raw_SD_init <- jnp$array(as.matrix(rep(as.numeric(tfp$math$softplus_inverse(sd_d)),times=((nrow(pi_init_list[d_,k_,1L][[1]]))))),dtype=jnp$float32)
        if(!sprintf("av%sk%s",d_,k_) %in% ls(envir = evaluation_environment)){
          eval(parse(text = sprintf( "av%sk%s_mean = tf$Variable(Zeros_init,trainable = T)",d_,k_)))
          eval(parse(text = sprintf( "av%sk%s_sd_raw = tf$Variable(Raw_SD_init,trainable = T)",d_,k_)))
          eval(parse(text = sprintf( "av%sk%s = tfd$Normal( loc = av%sk%s_mean,
                scale = tf$nn$softplus(av%sk%s_sd_raw), name = 'av%sk%s')",d_,k_,d_,k_,d_,k_,d_,k_)))
          eval(parse(text = sprintf( "av%sk%s_ = function(){
            tf$squeeze(tfd$Normal(loc = av%sk%s_mean, scale = tf$nn$softplus(av%sk%s_sd_raw))$sample(1L),0L)
                                  }",d_,k_,d_,k_,d_,k_)))

          # assign to correct environment
          eval(parse(text=sprintf("assign('av%sk%s_mean', av%sk%s_mean, envir = evaluation_environment)",d_,k_,d_,k_)))
          eval(parse(text=sprintf("assign('av%sk%s_sd_raw', av%sk%s_sd_raw, envir = evaluation_environment)",d_,k_,d_,k_)))
          eval(parse(text=sprintf("assign('av%sk%s', av%sk%s, envir = evaluation_environment)",d_,k_,d_,k_)))
          eval(parse(text=sprintf("assign('av%sk%s_', av%sk%s_, envir = evaluation_environment)",d_,k_,d_,k_)))
        }
        if(sprintf("av%sk%s",d_,k_) %in% ls(envir = evaluation_environment)){
          eval(parse(text = sprintf( "av%sk%s_mean$assign(Zeros_init)",d_,k_)))
          eval(parse(text = sprintf( "av%sk%s_sd_raw$assign(Raw_SD_init)",d_,k_)))
        }
      }
      if(useVariational == F){
          b_ <- f2n(attr(pi_init_list[d_,k_,1L][[1]],"random_"))[1]
          INIT_systemic <- as.matrix(f2n(attr(pi_init_list[d_,k_,1L][[1]],"sys_")))
          INIT_random <- tf$random_uniform_initializer(minval = -b_SCALE*b_, maxval = b_SCALE*b_)(tf$shape(pi_init_list[d_,k_,1L][[1]]))
          INIT_VALUE <- INIT_systemic + INIT_random
          if(!sprintf("av%sk%s",d_,k_) %in% ls(envir = evaluation_environment)){
            eval(parse(text=sprintf( "av%sk%s = tf$Variable( INIT_VALUE,
                       trainable = T, dtype = jnp$float32,name = 'av%sk%s')",d_,k_,d_,k_)) )
            eval(parse(text = sprintf( "av%sk%s_ = function(){ av%sk%s }",d_,k_,d_,k_)))

            # assign to correct environment
            eval(parse(text=sprintf("assign('av%sk%s',av%sk%s, envir = evaluation_environment)",d_,k_,d_,k_)))
            eval(parse(text=sprintf("assign('av%sk%s_',av%sk%s_, envir = evaluation_environment)",d_,k_,d_,k_)))
          }
          if(sprintf("av%sk%s",d_,k_) %in% ls(envir = evaluation_environment)){
            eval(parse(text=sprintf( "av%sk%s$assign( INIT_VALUE )",d_,k_)) )
          }
      }
  } }
  }

  # initialize environment
  environment(initialize_avs) <- evaluation_environment

  # initialize
  initialize_avs()

  getPiList <- function(simplex=T,rename=T,return_SE = F,VarCov = NULL){
    FinalSEList <- FinalProbList <- eval(parse(text=paste("list(",paste(rep("list(),",times=K-1),collapse=""), "list())",collapse="")))
    for(k_ in 1:K){
     for(d_ in 1:length(nLevels_vec)){
         if(simplex == F & normType == "ilr"){
           pidk <- as.numeric(eval(parse(text=sprintf("av%sk%s_()",d_,k_))) )
         }
         if(simplex == F & normType == "alr"){
           pidk <- as.numeric(tf$concat(list(as.matrix(0L),
                            eval(parse(text=sprintf("av%sk%s_()",d_,k_))) ),0L)); pidk[1] <- NA
         }
         if(simplex == T){
           with(tf$GradientTape(persistent = T) %as% tape, {
             if(normType == "ilr"){
               tmp_ <- eval(parse(text=sprintf("av%sk%s",d_, k_)))
               ilr_basis__t <- eval(parse(text=sprintf("ilrBasis%s_t",d_)))
               tmp_ <- jnp$transpose(jnp$exp(tf$matmul(jnp$transpose(tmp_),ilr_basis__t)))
               pidk <- (tmp_ / (zero_ep+tf$reduce_sum(tmp_,axis = 0L,keepdims=T)))
             }

             if(normType == "alr"){
               pidk <- keras$activations$softmax(tf$concat(list(as.matrix(0L),
                               eval(parse(text=sprintf("av%sk%s_()",d_,k_))) ),0L),0L)
             }
           })
            dpidk_davdk <- as.matrix(tf$squeeze(tf$squeeze(
                 tape$jacobian( pidk,
                    eval(parse(text=sprintf("av%sk%s",d_, k_)))),1L),2L))
           if(!is.null(VarCov) & return_SE == T){
              VarCov_subset <- grep(row.names(VarCov),
                              pattern = sprintf("av%sk%s",d_, k_))
              VarCov_subset <- as.matrix(VarCov[VarCov_subset,VarCov_subset])
              # CHECK ORDERING (lexiconographical problem)
             #if(ncol(VarCov_subset) >= 10){browser()}
             dpidk_davdk <- as.matrix(dpidk_davdk)
             VarCov_transformation <- (dpidk_davdk) %*% VarCov_subset %*% t(dpidk_davdk)
             pidk_se <- sqrt(diag(VarCov_transformation))
           }
           pidk <- as.numeric(pidk)
         }

        if(return_SE == T){
          FinalSEList[[k_]][[d_]] <- as.numeric(pidk_se)
        }
        FinalProbList[[k_]][[d_]] <- as.numeric(pidk)
        if(rename == F){
          names(FinalProbList[[k_]][[d_]]) <- paste(sprintf("av%sk%s:0_",d_,k_),1:(length(pidk)),sep="")
          names(FinalProbList[[k_]])[d_] <- paste("k",k_,sep="")

          if(return_SE == T){
            names(FinalSEList[[k_]][[d_]]) <- names(FinalProbList[[k_]][[d_]])
            names(FinalSEList[[k_]])[d_] <- names(FinalProbList[[k_]])[d_]
          }
        }
        if(rename == T){
          tmp_<-try(names(FinalProbList[[k_]][[d_]]) <- names(p_list[[d_]]),T)
          names(FinalProbList[[k_]])[d_] <- names(p_list)[d_]

          if(return_SE == T){
            names(FinalSEList[[k_]][[d_]]) <- names(FinalProbList[[k_]][[d_]])
            names(FinalSEList[[k_]])[d_] <- names(FinalProbList[[k_]])[d_]
          }
        }
      }
    }
    RET <- FinalProbList
    if(return_SE == T){RET <- list("FinalProbList" = FinalProbList, "FinalSEList" = FinalSEList ) }
    return( RET )
  }

  # initial forward pass (with small batch size for speed)
  returnWeightsFxn <- c; regLambdaFxn <- c
  tfConst <- function(x,ty=jnp$float32){  jnp$array(x,dtype=ty)  }

  # initial forward pass
  my_batch <- sample(1:length(Y), batch_size,replace=F)
  tmp_loss <- getProbRatio_tf(Y_ = tfConst(as.matrix(Y[my_batch])),
                         X_ = tfConst(X[my_batch,]),
                         factorMat_ = tfConst(FactorsMat_numeric_0Indexed[my_batch,],tf$int32),
                         logProb_ = tfConst(as.matrix(log_PrW[my_batch])),
                         REGULARIZATION_LAMBDA = tfConst(returnWeightsFxn(lambda_seq[1])))

  with(tf$GradientTape() %as% tape, {
      my_batch <- sample(1:length(Y), batch_size,replace=F)
      tmp_loss <- getLoss_tf(Y_ = tfConst(as.matrix(Y[my_batch])),
                                 X_ = tfConst(X[my_batch,]),
                                 factorMat_ = tfConst(FactorsMat_numeric_0Indexed[my_batch,],tf$int32),
                                 logProb_ = tfConst(as.matrix(log_PrW[my_batch])),
                                 REGULARIZATION_LAMBDA = tfConst(returnWeightsFxn(lambda_seq[1])))
  })

# trainable variables: the class prob projection variables + the a's
# possibility for bug here
tmp_loss_grad <- tape$gradient( tmp_loss, tape$watched_variables() )
tv_trainWith <- tape$watched_variables()[!(unlist(lapply(tmp_loss_grad,is.null))|unlist(lapply(tmp_loss_grad,function(er){ prod(dim(er))}))==0)]
rm(tmp_loss_grad)

  reinitialize_all <- function(b_SCALE = 1){
    # re-initialize avs
    initialize_avs(b_SCALE = b_SCALE)

    # re-initialize class proj variables
    if(useVariational == T){
      eval.parent(parse(text =
      'ClassProbProj$trainable_variables[[1]]$assign(KERNEL_LOC_INIT);
        ClassProbProj$trainable_variables[[2]]$assign(KERNEL_SCALE_INIT);
        ClassProbProj$trainable_variables[[3]]$assign(jnp$expand_dims(jnp$array(0.,jnp$float32),0L))'))
    }
    if(useVariational == F & K > 1){
     values_ <- unlist( lapply(strsplit(grep(names(ClassProbProj),pattern="initializer",value=T),
                         split="_"),function(zer){zer[1]}) )
     for(value_ in values_){
      eval.parent(parse(text = sprintf("
        ClassProbProj$%s$assign(ClassProbProj$%s_initializer(
                tf$shape(ClassProbProj$%s)))",value_,value_,value_)))
    }
    }
  }

  #optimization_language <- "tf"
  optimization_language <- "jax"; adaptiveMomentum <- F

  # define training function - using tf
  trainStep <-  (function(y_,x_,f_,lp_,lambda_,applyGradients = T){
    if(is.null(dim(f_))){ f_ <- t( f_ );  x_ <- t( x_ ) }
    with(tf$GradientTape(persistent = F, watch_accessed_variables = F) %as% tape, {
      tape$watch(  tv_trainWith  )
      myLoss_forGrad <- getLoss_tf(Y_ = tfConst(y_),
                                   X_ = tfConst(x_),
                                   factorMat_ = tfConst(f_,tf$int32),
                                   logProb_ = tfConst(lp_),
                                   REGULARIZATION_LAMBDA = tfConst(returnWeightsFxn(lambda_)))
    })
    my_grads <<- tape$gradient( myLoss_forGrad, tv_trainWith )

    # apply gradients
    if(applyGradients == T){ optimizer_tf$apply_gradients( rzip_tf(my_grads, tv_trainWith) )}

    # save information to global environment via <<-
    currentLossGlobal <<- as.numeric(myLoss_forGrad)
    L2_norm_squared <<- sum( unlist(lapply(my_grads, function(zer){
                        zer <- as.numeric(zer); if(length(zer) == 0){  zer <- 0 }; zer }))^2 )
  })

  # define optimizer
  if(sg_method == "cosine"){ optimizer_tf = tf$optimizers$Nadam(learning_rate = LEARNING_RATE_BASE, clipnorm = clipAT_factor) }
  if(sg_method %in% c("wngrad","adanorm")){ optimizer_tf = tf$optimizers$SGD(learning_rate = LEARNING_RATE_BASE,
                                    momentum = momentum, nesterov = F,  clipnorm = clipAT_factor) }
}

}
