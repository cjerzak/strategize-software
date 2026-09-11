policy_efficiency_fixture <- function(adversarial = FALSE) {
  d <- generate_test_data(n = 240L, n_factors = 2L, seed = 133)
  if (adversarial) d <- add_adversarial_structure(d, seed = 144)
  args <- c(d, list(p_list = generate_test_p_list(d$W), lambda = .1, nSGD = 6L,
    nMonte_Qglm = 5L, nMonte_adversarial = 3L, compute_se = FALSE,
    compute_hessian = FALSE, use_regularization = FALSE, force_gaussian = TRUE,
    diff = adversarial, adversarial = adversarial, rain_eta = .4,
    rain_lambda = 2, rain_gamma = 1))
  args
}

policy_efficiency_run <- function(args, control = list()) {
  withr::local_seed(991)
  withr::local_dir(withr::local_tempdir())
  do.call(strategize, c(args, list(policy_control = control)))
}

test_that("compiled policy trajectories match the R reference across update rules", {
  skip_if_no_jax()
  withr::local_envvar(STRATEGIZE_GLM_SKIP_EVAL = "1")
  for (adv in c(FALSE, TRUE)) {
    args <- policy_efficiency_fixture(adv)
    for (mode in c("none", "ogda", "extragrad", "smp", "rain")) {
      args$optimism <- mode
      a <- policy_efficiency_run(args, list(loop = "r"))
      b <- policy_efficiency_run(args, list(loop = "scan"))
      label <- paste(adv, mode)
      expect_equal(unlist(a$pi_star_point), unlist(b$pi_star_point), tolerance = 1e-5, info = label)
      expect_equal(a$Q_point, b$Q_point, tolerance = 1e-5, info = label)
      expect_equal(a$convergence_history$grad_ast, b$convergence_history$grad_ast,
                   tolerance = 1e-5, info = label)
      expect_equal(a$convergence_history$loss_ast, b$convergence_history$loss_ast,
                   tolerance = 1e-5, info = label)
      expect_null(b$vcov_outcome_model_concat)
      expect_null(b$jacobian_mat)
      expect_null(b$strenv$extragrad_eval_points)
    }
  }
})

test_that("Optax and RAIN reservoir outputs match without replaying the solve", {
  skip_if_no_jax()
  withr::local_envvar(STRATEGIZE_GLM_SKIP_EVAL = "1")
  args <- policy_efficiency_fixture(TRUE)
  for (mode in c("optax", "uniform_half")) {
    args$optimism <- if (mode == "optax") "none" else "rain"
    args$use_optax <- mode == "optax"
    args$rain_output <- if (mode == "uniform_half") "uniform_half" else "last"
    a <- policy_efficiency_run(args, list(loop = "r"))
    b <- policy_efficiency_run(args, list(loop = "scan", trace = TRUE))
    expect_equal(unlist(a$pi_star_point), unlist(b$pi_star_point), tolerance = 1e-5)
    expect_equal(a$Q_point, b$Q_point, tolerance = 1e-5)
    expect_length(b$strenv$extragrad_eval_points, args$nSGD)
    expect_equal(as.numeric(b$strenv$np$array(b$pi_star_red_ast)),
      as.numeric(b$strenv$np$array(b$strenv$a2Simplex_diff_use(b$a_i_ast))), tolerance = 1e-6)
  }
})

test_that("full-trace SEs agree across R and checkpointed scan implementations", {
  skip_if_no_jax()
  withr::local_envvar(STRATEGIZE_GLM_SKIP_EVAL = "1")
  args <- policy_efficiency_fixture(FALSE)
  args$compute_se <- TRUE; args$nSGD <- 4L
  args$se_method <- "full"
  a <- policy_efficiency_run(args, list(loop = "r", se_chunk_size = 2L))
  b <- policy_efficiency_run(args, list(loop = "scan", se_chunk_size = 3L))
  expect_equal(a$jacobian_mat, b$jacobian_mat, tolerance = 1e-5)
  expect_equal(a$Q_se, b$Q_se, tolerance = 1e-5)
  expect_equal(a$pi_star_se_vec, b$pi_star_se_vec, tolerance = 1e-5)
  expect_true(all(is.finite(b$jacobian_mat)))
  expect_s4_class(b$vcov_outcome_model_concat, "sparseMatrix")
})

test_that("blockwise covariance matches a dense reference", {
  withr::local_seed(11)
  J <- matrix(rnorm(35), 5, 7)
  blocks <- list(c(.3, .7), crossprod(matrix(rnorm(9), 3, 3)), c(.2, .1))
  dense <- as.matrix(Matrix::bdiag(lapply(blocks, function(x) {
    if (is.null(dim(x))) diag(x) else x
  })))
  expect_equal(strategize:::cs_policy_covariance(J, blocks), J %*% dense %*% t(J), tolerance = 1e-12)
})

test_that("CV reuses fold outcome fits and never optimizes evaluation policies", {
  skip_if_no_jax()
  withr::local_envvar(STRATEGIZE_GLM_SKIP_EVAL = "1")
  withr::local_options(strategize_test_policy_fits = 0L)
  withr::local_options(strategize_test_policy_solves = 0L)
  fit <- strategize:::generate_ModelOutcome
  body(fit) <- substitute({
    options(strategize_test_policy_fits = getOption("strategize_test_policy_fits") + 1L)
    BODY
  }, list(BODY = body(fit)))
  testthat::local_mocked_bindings(generate_ModelOutcome = fit, .package = "strategize")
  solver <- strategize:::getQPiStar_gd
  body(solver) <- substitute({
    options(strategize_test_policy_solves = getOption("strategize_test_policy_solves") + 1L)
    BODY
  }, list(BODY = body(solver)))
  testthat::local_mocked_bindings(getQPiStar_gd = solver, .package = "strategize")
  args <- policy_efficiency_fixture(FALSE)
  args$lambda <- NULL; args$lambda_seq <- c(.1, .3); args$folds <- 2L; args$nSGD <- 2L
  args$compute_hessian <- NULL; args$nMonte_Qglm <- NULL
  set.seed(127)
  cached <- do.call(cv_strategize, c(args, list(policy_control = list(reuse_outcomes = TRUE))))
  expect_identical(getOption("strategize_test_policy_fits"), 5L)
  expect_identical(getOption("strategize_test_policy_solves"), 5L)
  options(strategize_test_policy_fits = 0L)
  set.seed(127)
  fresh <- do.call(cv_strategize, c(args, list(policy_control = list(reuse_outcomes = FALSE))))
  expect_identical(getOption("strategize_test_policy_fits"), 9L)
  expect_equal(nrow(cached$CVInfo), 2L)
  expect_equal(cached$CVInfo, fresh$CVInfo, tolerance = 1e-6)
  expect_equal(cached$lambda, fresh$lambda)
})

test_that("nonlinear GLM policies preserve all primary estimators", {
  skip_if_no_jax()
  withr::local_envvar(STRATEGIZE_GLM_SKIP_EVAL = "1")
  args <- policy_efficiency_fixture(TRUE)
  args$force_gaussian <- FALSE; args$nSGD <- 3L
  module <- strategize:::cs_policy_module()
  original_pairs <- function(args) {
    testthat::local_mocked_bindings(cs_policy_module = function() list(
      GLMPairs = function(...) NULL,
      PolicyLoop = module$PolicyLoop,
      chunked_jacrev = module$chunked_jacrev
    ), .package = "strategize")
    policy_efficiency_run(args, list(loop = "r"))
  }
  for (method in c("mc", "linearized", "multi")) {
    args$primary_pushforward <- method
    args$primary_n_entrants <- args$primary_n_field <- if (method == "multi") 2L else 1L
    a <- policy_efficiency_run(args, list(loop = "r"))
    b <- policy_efficiency_run(args, list(loop = "scan"))
    original <- original_pairs(args)
    expect_equal(unlist(a$pi_star_point), unlist(b$pi_star_point), tolerance = 1e-5)
    expect_equal(a$Q_point, b$Q_point, tolerance = 1e-5)
    expect_true(is.finite(b$Q_point))
    expect_equal(unlist(b$pi_star_point), unlist(original$pi_star_point), tolerance = 1e-5)
    expect_equal(b$Q_point, original$Q_point, tolerance = 1e-5)
    expect_equal(b$convergence_history$grad_ast, original$convergence_history$grad_ast,
                 tolerance = 1e-5)
  }
})

test_that("one-step scans and nullable RAIN controls work", {
  skip_if_no_jax()
  withr::local_envvar(STRATEGIZE_GLM_SKIP_EVAL = "1")
  args <- policy_efficiency_fixture(FALSE)
  args$nSGD <- 1L; args$optimism <- "rain"
  args["rain_eta"] <- list(NULL)
  args["rain_gamma"] <- list(NULL)
  args["rain_lambda"] <- list(NULL)
  a <- policy_efficiency_run(args, list(loop = "r"))
  b <- policy_efficiency_run(args, list(loop = "scan"))
  expect_equal(a$Q_point, b$Q_point, tolerance = 1e-6)
  expect_equal(unlist(a$pi_star_point), unlist(b$pi_star_point), tolerance = 1e-6)
  expect_length(b$convergence_history$grad_ast, 1L)
})

test_that("screened GLM cache replays fitting and invalidates changed outcomes", {
  skip_if_no_jax()
  withr::local_envvar(STRATEGIZE_GLM_SKIP_EVAL = "1")
  withr::local_options(strategize_test_policy_fits = 0L)
  fit <- strategize:::generate_ModelOutcome
  body(fit) <- substitute({
    options(strategize_test_policy_fits = getOption("strategize_test_policy_fits") + 1L)
    BODY
  }, list(BODY = body(fit)))
  testthat::local_mocked_bindings(generate_ModelOutcome = fit, .package = "strategize")
  args <- policy_efficiency_fixture(FALSE)
  args$use_regularization <- TRUE; args$nSGD <- 2L
  control <- list(.fit_cache = new.env(parent = emptyenv()))
  first <- policy_efficiency_run(args, control)
  again <- policy_efficiency_run(args, control)
  expect_identical(getOption("strategize_test_policy_fits"), 1L)
  expect_equal(first$Q_point, again$Q_point, tolerance = 1e-6)
  expect_equal(first$pi_star_vec, again$pi_star_vec, tolerance = 1e-6)
  args$Y <- 1 - args$Y
  changed <- policy_efficiency_run(args, control)
  expect_identical(getOption("strategize_test_policy_fits"), 2L)
  fresh <- policy_efficiency_run(args)
  expect_equal(changed$Q_point, fresh$Q_point, tolerance = 1e-6)
  expect_equal(changed$pi_star_vec, fresh$pi_star_vec, tolerance = 1e-6)
})
