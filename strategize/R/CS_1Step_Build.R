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
  # setup main functions
  useHajekInOptimization_orig <- useHajekInOptimization
  if(penaltyType == "LogMaxProb"){
    RegularizationPiAction <- 'jnp$add(jnp$max(( (  (log_pidk) ) )),
    jnp$log(jnp$squeeze(jnp$divide(jnp$take(ClassProbsMarginal, k_ - 1L, axis = 1L),DFactors)))'
  }
  if(penaltyType == "L2"){
    ## if equal weighting of the penalty by cluster
    if(K == 1){ RegularizationPiAction <- 'jnp$sum(jnp$square( jnp$subtract( pd%s, jnp$exp(log_pidk) ) ))' }

    ## if weighting the penalty by the marginal probability for that factor
    # approach 1: (pi_z - p)^2 * pr(z)
    #if(K > 1){ RegularizationPiAction <- 'jnp$multiply(jnp$sum(jnp$square( jnp$subtract( pd%s, jnp$exp(log_pidk) ) )),
               #jnp$squeeze(jnp$take(ClassProbsMarginal, k_ - 1L, axis = 1L)))' }

    # approach 2: mean( ( (p_z-pi) * pr(z|x) )^2 )
    if(K > 1){ RegularizationPiAction <- 'jnp$sum(jnp$mean(jnp$square(jnp$multiply( jnp$subtract( pd%s, jnp$exp(log_pidk) ) ,
                                          jnp$expand_dims(jnp$take(ClassProbs, k_ - 1L, axis = 1L),0L) )),1L))' }

  }
  for(forMEst in c(T,F)){
    if(forMEst == T){ subtypes <- "probRatio"; useHajekInOptimization <- F; return_text <- 'returnThis_ <- minThis_;' }
    if(forMEst == F){ subtypes <- c("probRatio","objToMin") }
  for(subtype in subtypes){
    if(forMEst == F){
      useHajekInOptimization <- useHajekInOptimization_orig;
      if(subtype == "probRatio"){ return_text <- 'returnThis_ <- probRatio;' }
      if(subtype == "objToMin"){ return_text <- 'returnThis_ <- minThis_;' }
    }
    if( K > 1 ){
      regularizationContrib <- 'jnp$add(jnp$multiply(REGULARIZATION_LAMBDA, RegularizationContribPi),
                                                jnp$multiply(lambda_coef, jnp$mean(jnp$square(ClassProbProj$kernel))) )'
    }
    if( K == 1 ){
      regularizationContrib <- 'jnp$multiply(REGULARIZATION_LAMBDA, RegularizationContribPi)'
    }
    if(useHajekInOptimization == F){
      minThis_text <- sprintf('minThis_ <- jnp$add(jnp$negative(jnp$mean( jnp$multiply(Y_,probRatio) )), %s);',regularizationContrib)
    }
    if(useHajekInOptimization == T){
      minThis_text <- sprintf('minThis_ <- jnp$add(jnp$negative(jnp$sum( jnp$divide(jnp$multiply(Y_,probRatio),jnp$sum(probRatio)))), %s );',regularizationContrib)
    }
    # define main function
    {
      base_tf_function <- eval(parse(text = paste('myLoss_or_returnWeights <- (function(
                                                ModelList_, Y_, X_,factorMat_, logProb_,REGULARIZATION_LAMBDA){
          if(K == 1){ logClassProbs <- jnp$zeros(jnp$shape(Y_),jnp$float32)  }
          if(K >  1){
              ProjectionList_ <- ModelList_[[2]]
              logClassProbs <- jnp$log( ClassProbs <- getClassProb(ProjectionList_,  X_ )  )
              ClassProbsMarginal <- jnp$mean(ClassProbs, 0L, keepdims=T)
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
              eval(parse(text=sprintf("av%sk%s <- ModelList_[[1]][[%s]][[%s]]", d__, k_,  k_, d__)))
              log_pidk <- jax$nn$log_softmax(jnp$concatenate(list( Zero1by1,
                                         eval(parse(text=sprintf("av%sk%s",d__, k_))) ),0L),axis = 0L)
              received_wd <- jnp$take(factorMat_, indices = d__ - 1L, axis = 1L)
              log_PrWd_given_k <- jnp$take(log_pidk, indices = received_wd, axis = 0L)

              logPrW_given_k <- jnp$add(logPrW_given_k, log_PrWd_given_k)
              RegularizationContribPi <- RegularizationContribPi + eval(parse(text=sprintf(" ', RegularizationPiAction,' ",d__)))
            }
            logPrWGivenAllClasses[[k_]] <- logPrW_given_k
          }
          logPrW_new <- jax$scipy$special$logsumexp(jnp$concatenate(logPrWGivenAllClasses, 1L),keepdims = T, axis =  1L)
          probRatio <- jnp$exp( jnp$subtract(logPrW_new,  logProb_ ));',
          minThis_text, return_text,'return(   returnThis_    )})', sep = "")))
      }
    if(forMEst == T){
      getLoss_tf_unnormalized <- jax$jit(   base_tf_function  )
    }
    if(forMEst == F){
      if(subtype == "probRatio"){
        getProbRatio_tf <- jax$jit(   base_tf_function  )
      }
      if(subtype == "objToMin"){
        getLoss_tf <- jax$jit(   base_tf_function  )
      }
    }
    rm( myLoss_or_returnWeights_tf_ )
  }
  }

## start  building model
{
  # define the class prob projection, the trainable vars, the baseline probs
  KFreeProbProj <- ai(K - 1L)
  b_proj <- 0.1
  ProjectionList <- list(jnp$array(0.))
  if(KFreeProbProj > 0){
    value_seq <- sapply(b_seq <- 10^seq(-10,1,length.out=100),function(b_){
       value_ <- mean(replicate(25,{
        X_ <- X[sample(1:nrow(X),batch_size),]
        #p_ <- X_ %*% matrix(rnorm(ncol(X_)*KFreeProbProj,mean=0,sd=b_),ncol=KFreeProbProj)
        p_ <- X_ %*% matrix(runif(ncol(X_)*KFreeProbProj,-b_,b_), ncol=KFreeProbProj)
        penalty_val <- mean(apply((apply(p_,1,function(zer){ prop.table(exp(c(0,zer))) })),1,sd))
        penalty_val
      }))
  })
    #value_seq <- value_seq - 1 / K
    b_proj <- b_seq[which.min(abs(value_seq - CLUSTER_EPSILON))]
    print(sprintf("Uniform init bounds for kernel: %.5f",b_proj))

    # projection + bias arrays
    ClassProbKernel <- jnp$array(matrix(KFreeProbProj*ncol(X), rnorm(KFreeProbProj*ncol(X),
                                                                     mean=0, sd = 1/sqrt(KFreeProbProj)),
                                       ncol = KFreeProbProj), jnp$float32)
    ClassProbBias <- jnp$array(t(rep(0,times = KFreeProbProj)), jnp$float32)

    # initialize
    {
      getClassProb <- jax$jit(function(projectionList, x){
        ClassProbKernel <- projectionList[[1]][[1]]
        ClassProbBias <- projectionList[[1]][[2]]
        x <- jnp$add( jnp$matmul(ClassProbKernel, x), ClassProbBias)
        return(  jax$nn$softmax( jnp$concatenate( list(jnp$zeros(c(nrow = jnp$shape(x)[[1]],1L),
                                                                        dtype=jnp$float32),x),1L ) ,1L ) ) })
    }
    projectionList <- list(list(ClassProbKernel,ClassProbBias))
    print(paste("Initial Class Probs: ", paste(round(colMeans(as.array(getClassProb(projList,
                                                                                    jnp$array(X,jnp$float32)))),7L),collapse=','),sep=""))
  }

  # initialize pd -  assignment probabilities for penalty and pr(w)
    DFactors <- length( nLevels_vec <- apply(W,2,function(l){length(unique(l))}) )
    for(d_ in 1:(DFactors <- length(nLevels_vec))){
        eval(parse(text=sprintf( "pd%s = jnp$array(as.matrix(p_list[[d_]]), dtype = jnp$float32)",d_,d_)) )
    }

  # initialize av- the new probability generators
  initialize_avs <- function(b_SCALE=1){
    AVSList <- replicate(K, list(replicate(DFactors,list())))
    for(k_ in 1:K){ for(d_ in 1:DFactors){
          b_ <- f2n(attr(pi_init_list[d_,k_,1L][[1]],"random_"))[1]
          INIT_systemic <- as.matrix(f2n(attr(pi_init_list[d_,k_,1L][[1]],"sys_")))
          INIT_random <- matrix(runif(length(pi_init_list[d_,k_,1L][[1]]),
                                min = -b_SCALE*b_, max = b_SCALE*b_),
                                ncol = dim(pi_init_list[d_,k_,1L][[1]])[2])
          INIT_VALUE <- INIT_systemic + INIT_random
          if(!sprintf("av%sk%s",d_,k_) %in% ls(envir = evaluation_environment)){
            eval(parse(text=sprintf( "AVSList[[k_]][[d_]] <-  av%sk%s <- jnp$array( INIT_VALUE, dtype = jnp$float32)",d_,k_)) )
            eval(parse(text = sprintf( "av%sk%s_ = function(){ av%sk%s }",d_,k_,d_,k_)))

            # assign to correct environment
            eval(parse(text=sprintf("assign('av%sk%s',av%sk%s, envir = evaluation_environment)",d_,k_,d_,k_)))
            eval(parse(text=sprintf("assign('av%sk%s_',av%sk%s_, envir = evaluation_environment)",d_,k_,d_,k_)))
          }
          #if(sprintf("av%sk%s",d_,k_) %in% ls(envir = evaluation_environment)){
            #eval(parse(text=sprintf( "av%sk%s$assign( INIT_VALUE )",d_,k_)) ) }
    } }
    eval(parse(text=sprintf("assign('AVSList',AVSList, envir = evaluation_environment)")))
  }

  # initialize environment
  environment(initialize_avs) <- evaluation_environment

  # initialize
  initialize_avs()

  getPiDk <- function(AiDk){
    pidk <- jax$nn$softmax(jnp$concatenate(list(jnp$array(as.matrix(0L)),
                                                AiDk ),0L),0L)
  }
  Jacobian_getPiDk <- jax$jit(jax$jacobian( getPiDk ))
  getPiList <- function(SimplexList, simplex=T,rename=T,return_SE = F,VarCov = NULL){
    # SimplexList is not on the simplex, but is projected to the simplex
    FinalSEList <- FinalProbList <- eval(parse(text=paste("list(",paste(rep("list(),",times=K-1),collapse=""), "list())",collapse="")))
    for(k_ in 1:K){
     for(d_ in 1:length(nLevels_vec)){
          pidk <- np$array( getPiDk( SimplexList[[k_]][[d_]] )  )
          #pidk[1] <- NA
         if(simplex == T){
           if(!is.null(VarCov) & return_SE == T){
              dpidk_davdk <- np$array(jnp$squeeze(jnp$squeeze(
                Jacobian_getPiDk( SimplexList[[k_]][[d_]] ),1L), 2L))

              VarCov_subset <- grep(row.names(VarCov),
                              pattern = sprintf("k%sav%sd",k_, d_) )
              VarCov_subset <- as.matrix(VarCov[VarCov_subset,VarCov_subset]) # CHECK ORDER IN Ld > 2 case!!
              # CHECK ORDERING (lexiconographical problem???)
              #if(ncol(VarCov_subset) >= 10){browser()}
              VarCov_transformation <- try((dpidk_davdk) %*% VarCov_subset %*% t(dpidk_davdk), T)
              pidk_se <- sqrt(diag(VarCov_transformation))
           }
           pidk <- np$array(pidk)
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
  tfConst <- function(x, ty=jnp$float32){  jnp$array(x,dtype=ty)  }

  # initial forward pass
  my_batch <- sample(1:length(Y), batch_size,replace=F)
  tmp_loss <- getProbRatio_tf(
                         ModelList_ = list(AVSList, ProjectionList),
                         Y_ = tfConst(as.matrix(Y[my_batch])),
                         X_ = tfConst(X[my_batch,]),
                         factorMat_ = tfConst(FactorsMat_numeric_0Indexed[my_batch,],jnp$int32),
                         logProb_ = tfConst(as.matrix(log_PrW[my_batch])),
                         REGULARIZATION_LAMBDA = tfConst(returnWeightsFxn(lambda_seq[1])))

  gradient_getLoss_tf <- jax$jit( jax$grad(getLoss_tf) )

  # test gradient
  ModelList_object <- list(AVSList, ProjectionList)
  gradient_init <- gradient_getLoss_tf(
              ModelList_object, # ModelList_ =
              tfConst(as.matrix(Y[my_batch])), # Y_ =
              tfConst(X[my_batch,]), # X_ =
              tfConst(FactorsMat_numeric_0Indexed[my_batch,],jnp$int32), # factorMat_ =
              tfConst(as.matrix(log_PrW[my_batch])), # logProb_ =
              tfConst(returnWeightsFxn(lambda_seq[1])) # REGULARIZATION_LAMBDA =
             )
}

}
