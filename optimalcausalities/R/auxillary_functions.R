toSimplex = function(x){
  x[x>22] <- 22; x[x< -22] <- -22
  sim_x = exp(x)/sum(exp(x))
  if(any(is.nan(sim_x))){browser()}
  return(sim_x)
}

PrWGivenPi_fxn = function(doc_indices,pi_mat,terms_posterior,log_=F,
                          doc_indices_u = NULL,d_indices_u = NULL){
  if(is.null(doc_indices_u)){
    doc_indices_u = unlist(doc_indices,recursive = F)
    d_indices_u = unlist(sapply(1:length(doc_indices),
                                function(se){list(rep(se,length(doc_indices[[se]])))}))
  }
  row.names(terms_posterior) <- colnames( terms_posterior ) <- row.names(pi_mat) <- colnames(pi_mat) <- NULL
  terms_posterior = as.matrix( terms_posterior )
  pr_w_position_given_theta = colSums(terms_posterior[,doc_indices_u] * pi_mat[,d_indices_u])
  if(!log_){
    pr_w_given_theta = c(tapply(1:length(d_indices_u),d_indices_u,function(indi_){ prod( pr_w_position_given_theta[indi_] )}))
  }
  if(log_){
    pr_w_given_theta = c(tapply(1:length(d_indices_u),d_indices_u,function(indi_){ sum(log( pr_w_position_given_theta[indi_] ))}))
  }
  return( pr_w_given_theta)
}
