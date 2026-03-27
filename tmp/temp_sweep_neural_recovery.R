extract_average_case_pi_hat <- function(res) {
  pi_obj <- res$pi_star_point
  if (is.list(pi_obj) && length(pi_obj) == 1L && is.list(pi_obj[[1L]])) {
    pi_obj <- pi_obj[[1L]]
  }
  vapply(pi_obj, function(prob_vec) {
    if (!is.null(names(prob_vec)) && "1" %in% names(prob_vec)) {
      return(as.numeric(prob_vec[["1"]]))
    }
    as.numeric(prob_vec[[2L]])
  }, numeric(1))
}

pkgload::load_all("strategize", quiet = TRUE)
source("strategize/tests/testthat/helper-strategize.R")
fixture <- generate_linear_average_case_fixture()

temps <- c(1.0, 0.5, 0.2, 0.1, 0.05)
seed <- 20260326L

for (temp in temps) {
  withr::with_seed(seed, {
    res <- strategize(
      Y = fixture$Y,
      W = fixture$W,
      lambda = fixture$lambda,
      outcome_model_type = "neural",
      diff = FALSE,
      adversarial = FALSE,
      compute_se = FALSE,
      penalty_type = "L2",
      use_regularization = FALSE,
      use_optax = FALSE,
      force_gaussian = FALSE,
      nSGD = 1000L,
      nMonte_Qglm = 1000L,
      a_init_sd = 0.001,
      optim_type = "gd",
      temperature = temp,
      neural_mcmc_control = list(
        subsample_method = "batch_vi",
        ModelDims = 64L,
        ModelDepth = 2L,
        qk_norm = FALSE,
        batch_size = 512L,
        optimizer = "adam",
        vi_guide = "auto_diagonal",
        svi_steps = 200L,
        svi_num_draws = 100L,
        uncertainty_scope = "output",
        eval_enabled = FALSE
      )
    )
    pi_hat <- extract_average_case_pi_hat(res)
    Q_hat <- if (!is.null(res$Q_point_mEst) && all(is.finite(res$Q_point_mEst))) {
      as.numeric(res$Q_point_mEst)
    } else {
      as.numeric(res$Q_point)
    }
    pi_rel_err <- mean(abs(pi_hat - fixture$pi_star_true) / pmax(abs(fixture$pi_star_true), 1e-8))
    Q_rel_err <- abs(Q_hat - fixture$trueQ) / pmax(abs(fixture$trueQ), 1e-8)
    cat(sprintf("temp=%.3f pi_rel_err=%.6f Q_rel_err=%.6f Q_hat=%.6f trueQ=%.6f\n",
                temp, pi_rel_err, Q_rel_err, Q_hat, fixture$trueQ))
  })
}
