test_that("nonlinear report and CV integrate categorical profiles", {
  skip_if_not_installed("reticulate")
  if (!reticulate::py_module_available("jax")) skip("JAX runtime unavailable")
  strenv$jax <- reticulate::import("jax")
  strenv$jnp <- reticulate::import("jax.numpy")
  strenv$np <- reticulate::import("numpy")
  strenv$nUniqueFactors <- 1L
  strenv$nUniqueLevelsByFactors <- 2L
  env <- new.env(parent = asNamespace("strategize"))
  env$outcome_model_type <- "glm"
  env$main_indices_i0 <- strenv$jnp$array(0L)
  env$inter_indices_i0 <- NULL
  env$Q_DISAGGREGATE <- FALSE
  env$glm_outcome_transform <- strenv$jax$nn$sigmoid
  qfxn <- getQStar_diff_BASE
  environment(qfxn) <- env
  arr <- function(x) strenv$jnp$reshape(strenv$jnp$array(x), list(-1L, 1L))
  loc <- strenv$jnp$array(1L)
  evaluate <- function(pi, phase = "report", locator = loc, qfun = qfxn) {
    evaluate_average_case_q(pi, arr(.2), arr(0), arr(6), arr(0), arr(6),
      strenv$jax$random$PRNGKey(9L), phase, "glm", "binomial", 20000L, .5,
      "Implicit", locator, qfun, single_party = FALSE)
  }
  exact <- .5 + (.8 - .2) * (plogis(6) - .5)
  report <- evaluate(arr(.8))
  expect_true(report$spec$use_exact_support)
  expect_equal(as.numeric(strenv$np$array(report$q_max)), exact, tolerance = 1e-6)
  grad <- strenv$jax$grad(function(pi) evaluate(arr(pi))$q_max)(strenv$jnp$array(.8))
  expect_equal(as.numeric(strenv$np$array(grad)), plogis(6) - .5, tolerance = 1e-6)
  expect_identical(evaluate(arr(.8), "objective")$spec$profile_draw_mode, "relaxed")
  expect_identical(resolve_q_eval_spec("report", TRUE, "glm", "binomial", 100L)$profile_draw_mode, "hard")

  training <- list(pi_star_red_ast = arr(.8), pi_star_red_dag = arr(.2))
  model <- list(gather_fxn = function(x) list(arr(0), arr(6)),
    REGRESSION_PARAMETERS_ast = arr(c(0, 6)), REGRESSION_PARAMETERS_dag = arr(c(0, 6)),
    ParameterizationType = "Implicit", d_locator_use = loc, QFXN = qfxn)
  cv <- cs_cv_policy_value(training, model, "glm", TRUE, FALSE, FALSE)
  expect_equal(cv, exact, tolerance = 1e-6)
  plugin <- as.numeric(strenv$np$array(qfxn(arr(.8), arr(.2), arr(0), arr(6), arr(0), arr(6))))[[1L]]
  expect_gt(abs(plugin - cv), .1)
})

test_that("large-support report Monte Carlo retains discrete values and derivatives", {
  skip_if_not_installed("reticulate")
  if (!reticulate::py_module_available("jax")) skip("JAX runtime unavailable")
  strenv$jax <- reticulate::import("jax")
  strenv$jnp <- reticulate::import("jax.numpy")
  strenv$np <- reticulate::import("numpy")
  strenv$nUniqueFactors <- 6L
  strenv$nUniqueLevelsByFactors <- rep(2L, 6L)
  env <- new.env(parent = asNamespace("strategize"))
  env$outcome_model_type <- "glm"
  env$main_indices_i0 <- strenv$jnp$array(0:5)
  env$inter_indices_i0 <- NULL
  env$Q_DISAGGREGATE <- FALSE
  env$glm_outcome_transform <- strenv$jax$nn$sigmoid
  qfxn <- getQStar_diff_BASE
  environment(qfxn) <- env
  arr <- function(x) strenv$jnp$reshape(strenv$jnp$array(x), list(-1L, 1L))
  policy <- function(p) strenv$jnp$concatenate(list(
    strenv$jnp$reshape(p, list(1L, 1L)), arr(rep(.5, 5L))), axis = 0L)
  evaluate <- function(p) evaluate_average_case_q(
    policy(p), arr(c(.2, rep(.5, 5L))), arr(0), arr(c(6, rep(0, 5L))),
    arr(0), arr(c(6, rep(0, 5L))),
    strenv$jax$random$PRNGKey(17L), "report", "glm", "binomial", 50000L, .5,
    "Implicit", strenv$jnp$array(1:6), qfxn, single_party = FALSE)
  p <- strenv$jnp$array(.8)
  report <- evaluate(p)
  expect_false(report$spec$use_exact_support) # 64 x 64 profile pairs
  expect_identical(report$spec$profile_draw_mode, "hard")
  expect_equal(as.numeric(strenv$np$array(report$q_max)),
    .5 + (.8 - .2) * (plogis(6) - .5), tolerance = .006)
  grad <- strenv$jax$grad(function(p) evaluate(p)$q_max)(p)
  expect_equal(as.numeric(strenv$np$array(grad)), plogis(6) - .5, tolerance = .012)
})

test_that("adversarial hard-report gradients align with the Monte Carlo axis", {
  skip_if_not_installed("reticulate")
  if (!reticulate::py_module_available("jax")) skip("JAX runtime unavailable")
  strenv$jax <- reticulate::import("jax")
  strenv$jnp <- reticulate::import("jax.numpy")
  strenv$np <- reticulate::import("numpy")
  previous <- strenv$Vectorized_QMonteIter_MaxMin
  withr::defer(strenv$Vectorized_QMonteIter_MaxMin <- previous)
  # Preserve trailing output axes: accidental broadcasting would mix each
  # reward with every sample's score and erase the policy derivative.
  strenv$Vectorized_QMonteIter_MaxMin <- function(ast, dag, ...) {
    q <- strenv$jax$nn$sigmoid(6 * (strenv$jnp$take(ast, 0L, axis = 1L) -
                                    strenv$jnp$take(dag, 0L, axis = 1L)))
    list(q_ast = q, q_dag = 1 - q)
  }
  arr <- function(x) strenv$jnp$reshape(strenv$jnp$array(x), list(-1L, 1L))
  evaluate <- function(p) evaluate_adversarial_q(
    arr(p), arr(.2), arr(0), arr(0),
    arr(0), arr(6), arr(0), arr(6), arr(0), arr(6), arr(0), arr(6),
    arr(.5), arr(.5), arr(.5), arr(.5), arr(0), 1,
    strenv$jax$random$PRNGKey(29L), "report", "glm", "binomial",
    30000L, 100L, "mc", 1L, 1L, .5, "Implicit", strenv$jnp$array(1L))
  p <- strenv$jnp$array(.8)
  q <- evaluate(p)$q_max
  expect_equal(as.numeric(strenv$np$array(q)),
    .5 + (.8 - .2) * (plogis(6) - .5), tolerance = .007)
  grad <- strenv$jax$grad(function(p) evaluate(p)$q_max)(p)
  expect_equal(as.numeric(strenv$np$array(grad)), plogis(6) - .5, tolerance = .015)
})
