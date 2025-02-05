{
# Nash via grid search
{
  rm(list=ls()); options(error = NULL)
  
  # Goal: Implement grid search to 
  # find a Nash equilibrium in under two-stage primary
  # where R and D compete, selecting candidate gender.
  
  # Define parameters
  p_RVoters <- 0.6  # Proportion of Republican voters
  p_DVoters <- 1 - p_RVoters  # Proportion of Democrat voters
  lambda <- 0.1  # Regularization parameter for L2 penalty
  
  # Voter preferences in the general election
  # R voters' probability of voting for R based on candidate genders
  pr_mm_RVoters <- 0.9  # R male vs D male
  pr_mf_RVoters <- 0.85 # R male vs D female
  pr_fm_RVoters <- 0.8  # R female vs D male
  pr_ff_RVoters <- 0.87 # R female vs D female
  
  # D voters' probability of voting for R based on candidate genders
  pr_mm_DVoters <- 0.10  # R male vs D male
  pr_mf_DVoters <- 0.05  # R male vs D female
  pr_fm_DVoters <- 0.15  # R female vs D male
  pr_ff_DVoters <- 0.10  # R female vs D female
  
  # R voters' probability of voting for R in primary based on gender 
  pr_m_RVoters <- 0.56
  pr_f_RVoters <- 1-pr_m_RVoters
  
  # D voters' probability of voting for D in primary based on gender 
  pd_m_DVoters <- 0.45
  pd_f_DVoters <- 1-pd_m_DVoters
  
  # Generate grid of possible strategies (π_R and π_D)
  pi_R_seq <- seq(0, 1, by = 0.01)
  pi_D_seq <- seq(0, 1, by = 0.01)
  
  # Function to compute expected vote share for R
  compute_vote_share_R <- function(pi_R, pi_D) {
    # note: mf refers to an R "m" and D "f"
    # Comb prob * E[Vote Share Among R] = Comb prob * [Pr(R)*Pr(R|GG)*P(G) + ...]
    term_mm <- pi_R * pi_D             * (p_RVoters * pr_mm_RVoters * pr_m_RVoters + p_DVoters * pr_mm_DVoters * pd_m_DVoters)
    term_mf <- pi_R * (1 - pi_D)       * (p_RVoters * pr_mf_RVoters * pr_m_RVoters + p_DVoters * pr_mf_DVoters * pd_f_DVoters)
    term_fm <- (1 - pi_R) * pi_D       * (p_RVoters * pr_fm_RVoters * pr_f_RVoters + p_DVoters * pr_fm_DVoters * pd_m_DVoters)
    term_ff <- (1 - pi_R) * (1 - pi_D) * (p_RVoters * pr_ff_RVoters * pr_f_RVoters + p_DVoters * pr_ff_DVoters * pd_f_DVoters)
    return( term_mm + term_mf + term_fm + term_ff ) 
  }
  
  # Utility functions incorporating L2 penalty
  utility_R <- function(pi_R, pi_D) {
    vote_share <- compute_vote_share_R(pi_R, pi_D)
    vote_share - lambda * (pi_R - 0.5)^2
  }
  
  utility_D <- function(pi_R, pi_D) {
    vote_share <- 1 - compute_vote_share_R(pi_R, pi_D)
    vote_share - lambda * (pi_D - 0.5)^2
  }
  
  # Compute best responses for each party
  best_response_D <- sapply(pi_R_seq, function(r) {
    utilities <- sapply(pi_D_seq, function(d) utility_D(r, d))
    pi_D_seq[which.max(utilities)]
  })
  
  best_response_R <- sapply(pi_D_seq, function(d) {
    utilities <- sapply(pi_R_seq, function(r) utility_R(r, d))
    pi_R_seq[which.max(utilities)]
  })
  
  # Identify Nash equilibria
  nash_equilibrium <- matrix(FALSE, nrow = length(pi_R_seq), ncol = length(pi_D_seq))
  for (i in seq_along(pi_R_seq)) {
    for (j in seq_along(pi_D_seq)) {
      current_r <- pi_R_seq[i]
      current_d <- pi_D_seq[j]
      if (abs(current_r - best_response_R[j]) < 1e-6 && 
          abs(current_d - best_response_D[i]) < 1e-6) {
        nash_equilibrium[i, j] <- TRUE
      }
    }
  }
  
  # Extract and print equilibria
  equilibria <- which(nash_equilibrium, arr.ind = TRUE)
  if (nrow(equilibria) > 0) {
    cat("Nash Equilibria Found:\n")
    for (k in 1:nrow(equilibria)) {
      pi_R <- pi_R_seq[equilibria[k, 1]]
      pi_D <- pi_D_seq[equilibria[k, 2]]
      cat(sprintf("π_R: %.2f, π_D: %.2f\n", pi_R, pi_D))
    }
  } else {
    cat("No Nash equilibrium found within the grid.\n")
  }
}

# Nash via estimation + iterative optimization
{
  #1. Respondent and candidate info
  # -----------------------------
  nObs <- 1000
  competing_group_variable_respondent <- sample(c("Democrat","Republican"), 
                                                size = nObs, replace = TRUE,
                                                prob = c(p_DVoters, p_RVoters))
  
  # Each respondent sees exactly 2 profiles => forced choice:
  competing_group_variable_respondent <- c(competing_group_variable_respondent,
                                           competing_group_variable_respondent)
  pair_id        <- respondent_id <- c(1:nObs, 1:nObs)
  respondent_id  <- respondent_id <- c(1:nObs, 1:nObs)
  profile_order  <- c(rep(1, nObs), rep(2, nObs))
  respondent_task_id <- rep(1, times = 2*nObs)
  
  # For simplicity, randomly assign candidate group. We then mark “Same” vs. “Different”
  competing_group_variable_candidate <- sample(c("Democrat","Republican"), 
                                               size = 2*nObs, replace = TRUE,
                                               prob = c(0.5,0.5))
  # E.g., "Same" means candidate’s party matches the respondent’s party
  competing_group_competition_variable_candidate <- ifelse(
    competing_group_variable_candidate[1:nObs] == competing_group_variable_candidate[-c(1:nObs)],
    yes = "Same", no = "Different"
  )
  # Expand for 2*nObs (long format)
  competing_group_competition_variable_candidate <-
    c(competing_group_competition_variable_candidate,
      competing_group_competition_variable_candidate)
  
  # -----------------------------
  # 2. Generate the design matrix X (binary factors) and outcomes
  # -----------------------------
  # We'll have kFactors columns, each in {0,1}, for 2*nObs profiles
  # THIS IS GENDER 
  X <- as.data.frame(matrix(rbinom(1 * nObs * 2, size = 1, prob = 0.5),
              nrow = 2*nObs))
  colnames(X) <- "female"
  
  # Suppose 'X' is a vector or matrix of genders for each profile row i=1,...,2*nObs.
  # We can make a vector X_other, which for row i returns the gender of the *other* row
  # in the same pair, by matching on 'pair_id' and ignoring the same row.
  
  X_other <- numeric(length = nrow(X))
  for (p in unique(pair_id)) {
    rows_in_pair <- which(pair_id == p)
    # exactly two rows per pair, e.g. i and j
    i <- rows_in_pair[1]
    j <- rows_in_pair[2]
    X_other[i] <- X[j,]
    X_other[j] <- X[i,]
  }
  
  Yobs <- rep(NA, times = length(competing_group_competition_variable_candidate))
  i_pool <- c()
  
  ConfirmSanity <- TRUE 
  
  # Republican primary probabilities if "Same":
  #   pr_m_RVoters (R voter chooses an R male), pr_f_RVoters (R voter chooses an R female)
  i_ <- which(
      competing_group_competition_variable_candidate == "Same" &
      competing_group_variable_respondent == "Republican" & 
      competing_group_variable_candidate == "Republican" &
      X$female == 0 # R is male 
  ); i_pool <- c(i_,i_pool)
  if(!ConfirmSanity){ Yobs[i_] <- rbinom(length(i_), size = 1, prob = pr_m_RVoters) }
  if(ConfirmSanity){ Yobs[i_] <- pr_m_RVoters }
  
  i_ <- which(
    competing_group_competition_variable_candidate == "Same" &
      competing_group_variable_respondent == "Republican" & 
      competing_group_variable_candidate == "Republican" &
      X$female == 1 # R is female 
  ); i_pool <- c(i_,i_pool)
  if(!ConfirmSanity){ Yobs[i_] <- rbinom(length(i_), size = 1, prob = pr_f_RVoters) }
  if(ConfirmSanity){ Yobs[i_] <- pr_f_RVoters }
  
  # Republican candidate in a "Different" matchup => the respondent is Democrat,
  # so we use the R-vs-D general-election probabilities for R's gender vs D's gender.
  i_ <- which(
    competing_group_competition_variable_candidate == "Different" &
      competing_group_variable_respondent == "Republican" & 
      competing_group_variable_candidate == "Republican" &
      X$female == 0 & # R is male
      X_other == 0 # D is male 
  ); i_pool <- c(i_,i_pool)
  if(!ConfirmSanity){  Yobs[i_] <- rbinom(length(i_), size = 1, prob = pr_mm_RVoters) }
  if(ConfirmSanity){ Yobs[i_] <- pr_mm_RVoters }
  
  i_ <- which(
    competing_group_competition_variable_candidate == "Different" &
      competing_group_variable_respondent == "Republican" & 
      competing_group_variable_candidate == "Republican" &
      X$female == 0 &    # R is male
      X_other == 1 # D is female
  ); i_pool <- c(i_,i_pool)
  if(!ConfirmSanity){ Yobs[i_] <- rbinom(length(i_), size = 1, prob = pr_mf_RVoters) }
  if(ConfirmSanity){ Yobs[i_] <- pr_mf_RVoters }
  
  i_ <- which(
    competing_group_competition_variable_candidate == "Different" &
      competing_group_variable_respondent == "Republican" & 
      competing_group_variable_candidate == "Republican" &
      X$female == 1 &    # R is female
      X_other == 0 # D is male 
  ); i_pool <- c(i_,i_pool)
  if(!ConfirmSanity){  Yobs[i_] <- rbinom(length(i_), size = 1, prob = pr_fm_RVoters) }
  if(ConfirmSanity){ Yobs[i_] <- pr_fm_RVoters }
  
  i_ <- which(
    competing_group_competition_variable_candidate == "Different" &
      competing_group_variable_respondent == "Republican" & 
      competing_group_variable_candidate == "Republican" &
      X$female == 1 &    # R is female
      X_other == 1 # D is female 
  ); i_pool <- c(i_,i_pool)
  if(!ConfirmSanity){ Yobs[i_] <- rbinom(length(i_), size = 1, prob = pr_ff_RVoters) }
  if(ConfirmSanity){ Yobs[i_] <- pr_ff_RVoters }
  
  # Democrat primary probabilities if "Same":
  #   pd_m_DVoters (D voter chooses a D male), pd_f_DVoters (D voter chooses a D female)
  i_ <- which(
    competing_group_competition_variable_candidate == "Same" &
      competing_group_variable_respondent == "Democrat" & 
      competing_group_variable_candidate == "Democrat" &
      X$female == 0 # D is male 
  ); i_pool <- c(i_,i_pool)
  if(!ConfirmSanity){ Yobs[i_] <- rbinom(length(i_), size = 1, prob = pd_m_DVoters) }
  if(ConfirmSanity){ Yobs[i_] <- pd_m_DVoters }
  
  i_ <- which(
    competing_group_competition_variable_candidate == "Same" &
      competing_group_variable_respondent == "Democrat" & 
      competing_group_variable_candidate == "Democrat" &
      X$female == 1 # # is female 
  ); i_pool <- c(i_, i_pool)
  if(!ConfirmSanity){ Yobs[i_] <- rbinom(length(i_), size = 1, prob = pd_f_DVoters) }
  if(ConfirmSanity){ Yobs[i_] <- pd_f_DVoters }
  
  # Democrat candidate in a "Different" matchup => the respondent is Republican,
  # so we use the R-vs-D general-election probabilities from DEMOCRAT's perspective
  # (pr_mm_DVoters, pr_mf_DVoters, pr_fm_DVoters, pr_ff_DVoters).
  # !!! NOTE: THINGS HERE ARE FLIPPED!! 
  i_ <- which(
    competing_group_competition_variable_candidate == "Different" &
      competing_group_variable_respondent == "Democrat" & 
      competing_group_variable_candidate == "Democrat" &
      X$female == 0 &      # D is male
      X_other == 0         # R male
  ); i_pool <- c(i_,i_pool)
  if(!ConfirmSanity){ Yobs[i_] <- rbinom(length(i_), size = 1, prob = pr_mm_DVoters) }
  if(ConfirmSanity){ Yobs[i_] <- pr_mm_DVoters }
  
  i_ <- which(
    competing_group_competition_variable_candidate == "Different" &
      competing_group_variable_respondent == "Democrat" & 
      competing_group_variable_candidate == "Democrat" &
      X$female == 0 &      # D is male
      X_other == 1         # R is female
  ); i_pool <- c(i_,i_pool)
  if(!ConfirmSanity){ Yobs[i_] <- rbinom(length(i_), size = 1, prob = pr_fm_DVoters) }
  if(ConfirmSanity){ Yobs[i_] <- pr_mf_DVoters }
  
  i_ <- which(
    competing_group_competition_variable_candidate == "Different" &
      competing_group_variable_respondent == "Democrat" & 
      competing_group_variable_candidate == "Democrat" &
      X$female == 1 &      # D is female
      X_other == 0         # R is male
  ); i_pool <- c(i_,i_pool)
  if(!ConfirmSanity){ Yobs[i_] <- rbinom(length(i_), size = 1, prob = pr_mf_DVoters) }
  if(ConfirmSanity){ Yobs[i_] <- pr_fm_DVoters}
  
  i_ <- which(
    competing_group_competition_variable_candidate == "Different" &
      competing_group_variable_respondent == "Democrat" & 
      competing_group_variable_candidate == "Democrat" &
      X$female == 1 &      # D is female
      X_other == 1         # R is female
  ); i_pool <- c(i_,i_pool)
  if(!ConfirmSanity){ Yobs[i_] <- rbinom(length(i_), size = 1, prob = pr_ff_DVoters) }
  if(ConfirmSanity){ Yobs[i_] <- pr_ff_DVoters }
  
  plot(Yobs)
  table(Yobs)
  table(table(i_pool))
  
  stop("XXX")
  # sanity checks - force forced choice
  sanity_ <- cbind(Yobs[1:nObs], Yobs[-c(1:nObs)])
  plot(sanity_[,1], sanity_[,2] ); abline(a=0,b=1)
  plot(sanity_[,1], 1-sanity_[,2] ); abline(a=0,b=1)
  plot(sanity_[,1] - sanity_[,2] )
  mean(sanity_[,1] == sanity_[,2] )
  Yobs[-c(1:nObs)] <- 1 - Yobs[1:nObs]

  # run X analysis 
  X_run <- cbind(X,
                 rbinom(nrow(X),size=1,prob=0.5),
                 rbinom(nrow(X),size=1,prob=0.5))
  colnames(X_run) <- c("female", "placeholder1",  "placeholder2")
  Qoptimized <-  {strategize::strategize(
      Y = Yobs,
      W = X_run,
      lambda = lambda,
      conda_env = "strategize",
      conda_env_required = TRUE,
      
      # Adversarial parameters
      competing_group_variable_respondent = competing_group_variable_respondent,
      competing_group_variable_candidate   = competing_group_variable_candidate,
      competing_group_competition_variable_candidate = competing_group_competition_variable_candidate,
      pair_id         = pair_id,
      respondent_id   = respondent_id,
      respondent_task_id = respondent_task_id,
      profile_order   = profile_order,
      diff           = TRUE,
      adversarial    = TRUE,
      
      # Main parameters
      compute_se       = FALSE,
      penalty_type     = "L2",
      use_regularization= FALSE,
      use_optax        = FALSE,
      nSGD             = 1000L,
      nMonte_adversarial = 34L, 
      nMonte_Qglm       = 100L, 
      optim_type       = "tryboth",
      force_gaussian   = FALSE,
      a_init_sd        = 0.001,
      conf_level       = 0.95
    )}
  print(Qoptimized$pi_star_point$k1$female)
  print(Qoptimized$pi_star_point$k2$female)
  cat(sprintf("π_R: %.2f, π_D: %.2f\n", pi_R, pi_D))
  
}

}
