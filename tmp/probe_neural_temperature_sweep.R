pkgload::load_all("/Users/cjerzak/Documents/strategize-software/strategize", quiet = TRUE)
source("/Users/cjerzak/Documents/strategize-software/strategize/tests/testthat/helper-strategize.R")

extract_average_case_pi_hat <- function(res) {
  pi_obj <- res[["pi_star_point"]]
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

fx <- generate_linear_average_case_fixture()
seed <- 20260326L
temperatures <- c(1.0, 0.5, 0.25, 0.1)

results <- lapply(temperatures, function(temp_val) {
  withr::local_seed(seed)
  res <- strategize(
    Y = fx[["Y"]],
    W = fx[["W"]],
    lambda = fx[["lambda"]],
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
    temperature = temp_val,
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
  q_hat <- if (!is.null(res$Q_point_mEst) && all(is.finite(res$Q_point_mEst))) {
    as.numeric(res$Q_point_mEst)
  } else {
    as.numeric(res$Q_point)
  }

  data.frame(
    temperature = temp_val,
    pi_rel_err = mean(abs(pi_hat - fx$pi_star_true) / pmax(abs(fx$pi_star_true), 1e-8)),
    Q_rel_err = abs(q_hat - fx$trueQ) / pmax(abs(fx$trueQ), 1e-8),
    Q_hat = q_hat,
    trueQ = fx$trueQ
  )
})

out <- do.call(rbind, results)
print(out, row.names = FALSE)
