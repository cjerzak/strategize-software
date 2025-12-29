#' Validate Nash Equilibrium Quality
#'
#' Computes best-response error to verify that the optimized strategies
#' form a Nash equilibrium. At a true Nash equilibrium, neither player
#' can improve their payoff by unilaterally changing strategy.
#'
#' @importFrom graphics barplot text
#'
#' @param result Output from \code{\link{strategize}} with \code{adversarial = TRUE}
#' @param method Character string specifying the search method:
#'   \itemize{
#'     \item \code{"grid"}: Exhaustive grid search (more accurate, slower)
#'     \item \code{"gradient"}: Gradient ascent from current solution (faster)
#'   }
#'   Default is \code{"grid"}.
#' @param resolution Integer. Number of grid points per dimension for grid search.
#'   Default is 50.
#' @param tolerance Numeric. Maximum BR error to consider as equilibrium.
#'   Default is 0.01 (i.e., neither player can improve vote share by more than 1\%).
#' @param nMonte Integer. Number of Monte Carlo samples for Q evaluation.
#'   Default is 100.
#' @param plot Logical. Whether to generate visualization. Default is \code{TRUE}.
#' @param verbose Logical. Whether to print progress messages. Default is \code{TRUE}.
#'
#' @return A list containing:
#'   \describe{
#'     \item{br_error_ast}{Best-response error for AST player (how much AST
#'       could improve by switching to best response)}
#'     \item{br_error_dag}{Best-response error for DAG player}
#'     \item{is_equilibrium}{Logical. TRUE if both errors are below tolerance}
#'     \item{Q_current}{Current objective value at the solution}
#'     \item{Q_br_ast}{Objective value if AST switched to best response}
#'     \item{Q_br_dag}{Objective value if DAG switched to best response}
#'     \item{br_strategy_ast}{The best response strategy for AST (for comparison)}
#'     \item{br_strategy_dag}{The best response strategy for DAG}
#'     \item{plot}{ggplot object if plot=TRUE and ggplot2 available, else NULL}
#'   }
#'
#' @details
#'
#' \strong{What is a Nash Equilibrium?}
#'
#' In the adversarial setting, two parties (AST and DAG) simultaneously choose
#' probability distributions over candidate attributes (e.g., gender, age, policy
#' positions). Each party wants to maximize their expected vote share given the
#' opponent's strategy. A Nash equilibrium is a pair of strategies where:
#' \itemize{
#'   \item AST's strategy is optimal given DAG's strategy
#'   \item DAG's strategy is optimal given AST's strategy
#' }
#' At equilibrium, neither party can improve by unilaterally changing their strategy.
#'
#' \strong{What is Best-Response Error?}
#'
#' The best-response error measures how far a player's current strategy is from
#' being optimal. For player p, it is defined as:
#' \deqn{BR\_error_p = \max_{\pi_p} Q(\pi_p, \pi^*_{-p}) - Q(\pi^*_p, \pi^*_{-p})}
#'
#' In words: if we fix the opponent's strategy and search for the best possible
#' response, how much better could we do compared to our current strategy?
#'
#' \itemize{
#'   \item \strong{BR error = 0}: The player is already playing optimally (true equilibrium)
#'   \item \strong{BR error > 0}: The player could improve by switching strategies
#'   \item \strong{BR error = 0.05}: The player could gain 5 percentage points in vote share
#' }
#'
#' \strong{How Validation Works}
#'
#' This function performs the following steps:
#' \enumerate{
#'   \item Evaluates Q (expected vote share) at the current solution
#'   \item For AST: fixes DAG's strategy and searches for AST's best response
#'   \item For DAG: fixes AST's strategy and searches for DAG's best response
#'   \item Computes the improvement each player could achieve (BR error)
#'   \item If both BR errors are below tolerance, declares it a valid equilibrium
#' }
#'
#' \strong{Interpretation}
#'
#' \itemize{
#'   \item \code{is_equilibrium = TRUE}: The solution is a valid Nash equilibrium
#'     (within numerical tolerance). Both parties are playing optimally.
#'   \item \code{is_equilibrium = FALSE}: At least one party could improve by
#'     changing strategy. This may indicate insufficient SGD iterations, a local
#'     minimum, or numerical issues.
#' }
#'
#' \strong{Search Methods}
#'
#' \itemize{
#'   \item \code{"grid"}: Searches over a discretized grid of strategies around
#'     the current solution. More thorough but slower. Recommended for validation.
#'   \item \code{"gradient"}: Runs additional gradient ascent steps from the
#'     current solution. Faster but may miss improvements in other directions.
#' }
#'
#' @examples
#' \dontrun{
#' # Run adversarial strategize
#' result <- strategize(Y = y, W = w, adversarial = TRUE, nSGD = 500)
#'
#' # Validate equilibrium
#' validation <- validate_equilibrium(result)
#' print(validation$is_equilibrium)
#' print(validation$br_error_ast)
#' print(validation$br_error_dag)
#'
#' # If validation fails, try more SGD iterations
#' if (!validation$is_equilibrium) {
#'   result2 <- strategize(Y = y, W = w, adversarial = TRUE, nSGD = 2000)
#'   validation2 <- validate_equilibrium(result2)
#' }
#' }
#'
#' @export
validate_equilibrium <- function(result,
                                  method = c("grid", "gradient"),
                                  resolution = 50,
                                  tolerance = 0.01,
                                  nMonte = 100,
                                  plot = TRUE,
                                  verbose = TRUE) {

  method <- match.arg(method)

  # Validate input
  if (!isTRUE(result$convergence_history$adversarial)) {
    stop("validate_equilibrium() requires an adversarial strategize result. ",
         "Set adversarial = TRUE in strategize().")
  }

  if (is.null(result$FullGetQStar_)) {
    stop("Result does not contain Q function. ",
         "Make sure you are using a recent version of strategize().")
  }

  strenv <- result$strenv
  if (is.null(strenv) || is.null(strenv$jnp)) {
    stop("JAX environment not available in result. ",
         "Cannot evaluate Q function.")
  }

  if (verbose) message("Validating Nash equilibrium...")

  # Extract current solution
  a_i_ast_current <- result$a_i_ast
  a_i_dag_current <- result$a_i_dag

  # Get model parameters
  REGRESSION_PARAMS_ast <- result$REGRESSION_PARAMETERS_ast
  REGRESSION_PARAMS_dag <- result$REGRESSION_PARAMETERS_dag
  REGRESSION_PARAMS_ast0 <- result$REGRESSION_PARAMETERS_ast0
  REGRESSION_PARAMS_dag0 <- result$REGRESSION_PARAMETERS_dag0

  P_VEC_FULL_ast <- result$P_VEC_FULL_ast
  P_VEC_FULL_dag <- result$P_VEC_FULL_dag
  SLATE_VEC_ast <- result$SLATE_VEC_ast
  SLATE_VEC_dag <- result$SLATE_VEC_dag

  lambda <- result$lambda
  LAMBDA <- strenv$jnp$array(lambda)

  # Get the gather function and extract intercepts/coefficients
  gather_fxn <- result$gather_fxn
  if (is.null(gather_fxn)) {
    gather_fxn <- function(x) x
  }

  params_ast <- gather_fxn(REGRESSION_PARAMS_ast)
  params_dag <- gather_fxn(REGRESSION_PARAMS_dag)
  params_ast0 <- gather_fxn(REGRESSION_PARAMS_ast0)
  params_dag0 <- gather_fxn(REGRESSION_PARAMS_dag0)

  INTERCEPT_ast <- params_ast[[1]]
  COEFFICIENTS_ast <- params_ast[[2]]
  INTERCEPT_dag <- params_dag[[1]]
  COEFFICIENTS_dag <- params_dag[[2]]
  INTERCEPT_ast0 <- params_ast0[[1]]
  COEFFICIENTS_ast0 <- params_ast0[[2]]
  INTERCEPT_dag0 <- params_dag0[[1]]
  COEFFICIENTS_dag0 <- params_dag0[[2]]

  # Function to evaluate Q at given strategies
  eval_Q <- function(a_ast, a_dag, Q_SIGN = 1.0) {
    SEED <- strenv$jax$random$PRNGKey(as.integer(Sys.time()))

    Q_val <- result$FullGetQStar_(
      a_ast, a_dag,
      INTERCEPT_ast, COEFFICIENTS_ast,
      INTERCEPT_dag, COEFFICIENTS_dag,
      INTERCEPT_ast0, COEFFICIENTS_ast0,
      INTERCEPT_dag0, COEFFICIENTS_dag0,
      P_VEC_FULL_ast, P_VEC_FULL_dag,
      SLATE_VEC_ast, SLATE_VEC_dag,
      LAMBDA,
      strenv$jnp$array(Q_SIGN),
      SEED
    )

    # Convert JAX array to R numeric via numpy
    as.numeric(strenv$np$array(Q_val))
  }

  # Evaluate Q at current solution
  Q_current_ast <- eval_Q(a_i_ast_current, a_i_dag_current, Q_SIGN = 1.0)
  Q_current_dag <- eval_Q(a_i_ast_current, a_i_dag_current, Q_SIGN = -1.0)

  if (verbose) message(sprintf("  Current Q (AST perspective): %.4f", Q_current_ast))

  # Find best response for each player
  if (method == "grid") {
    br_result <- find_best_response_grid(
      result, a_i_ast_current, a_i_dag_current,
      eval_Q, resolution, verbose
    )
  } else {
    br_result <- find_best_response_gradient(
      result, a_i_ast_current, a_i_dag_current,
      eval_Q, verbose
    )
  }

  # Compute BR errors
  br_error_ast <- br_result$Q_br_ast - Q_current_ast
  br_error_dag <- br_result$Q_br_dag - Q_current_dag

  # Ensure errors are non-negative (numerical issues can cause small negatives)
  br_error_ast <- max(0, br_error_ast)
  br_error_dag <- max(0, br_error_dag)

  is_equilibrium <- (br_error_ast < tolerance) && (br_error_dag < tolerance)

  if (verbose) {
    message(sprintf("  BR Error (AST): %.4f %s",
                    br_error_ast,
                    ifelse(br_error_ast < tolerance, "[PASS]", "[FAIL]")))
    message(sprintf("  BR Error (DAG): %.4f %s",
                    br_error_dag,
                    ifelse(br_error_dag < tolerance, "[PASS]", "[FAIL]")))
    if (is_equilibrium) {
      message("  Result: Validated as Nash equilibrium")
    } else {
      message("  Result: NOT a Nash equilibrium (players can improve)")
    }
  }

  # Create visualization if requested
  plot_obj <- NULL
  if (plot) {
    plot_obj <- plot_equilibrium_validation(br_error_ast, br_error_dag, tolerance)
  }

  return(list(
    br_error_ast = br_error_ast,
    br_error_dag = br_error_dag,
    is_equilibrium = is_equilibrium,
    Q_current = Q_current_ast,
    Q_br_ast = br_result$Q_br_ast,
    Q_br_dag = br_result$Q_br_dag,
    br_strategy_ast = br_result$br_a_ast,
    br_strategy_dag = br_result$br_a_dag,
    tolerance = tolerance,
    method = method,
    plot = plot_obj
  ))
}


#' Internal: Grid search for best response
#' @keywords internal
#' @noRd
find_best_response_grid <- function(result, a_i_ast_current, a_i_dag_current,
                                    eval_Q, resolution, verbose) {

  strenv <- result$strenv

  # Get parameter dimensions
  n_params_ast <- as.integer(strenv$np$array(a_i_ast_current$shape[[1]]))
  n_params_dag <- as.integer(strenv$np$array(a_i_dag_current$shape[[1]]))

  if (verbose) {
    message(sprintf("  Grid search: %d params (AST), %d params (DAG), resolution=%d",
                    n_params_ast, n_params_dag, resolution))
  }

  # For high-dimensional problems, use random search instead of full grid
  if (n_params_ast > 5 || n_params_dag > 5) {
    if (verbose) message("  Using random search for high-dimensional problem...")
    return(find_best_response_random(result, a_i_ast_current, a_i_dag_current,
                                      eval_Q, resolution * 100, verbose))
  }

  # Generate grid points in the unconstrained parameter space
  # We search around the current solution
  current_ast <- as.numeric(strenv$np$array(a_i_ast_current))
  current_dag <- as.numeric(strenv$np$array(a_i_dag_current))

  # Search range: +/- 2 units in unconstrained space
  search_range <- 2.0

  # Find best response for AST (fixing DAG)
  best_Q_ast <- eval_Q(a_i_ast_current, a_i_dag_current, Q_SIGN = 1.0)
  best_a_ast <- a_i_ast_current

  grid_ast <- seq(-search_range, search_range, length.out = resolution)

  if (n_params_ast == 1) {
    for (offset in grid_ast) {
      a_test <- strenv$jnp$array(current_ast + offset)
      Q_test <- eval_Q(a_test, a_i_dag_current, Q_SIGN = 1.0)
      if (Q_test > best_Q_ast) {
        best_Q_ast <- Q_test
        best_a_ast <- a_test
      }
    }
  } else {
    # Multi-dimensional grid (simplified - random sampling)
    for (i in seq_len(resolution^2)) {
      offsets <- runif(n_params_ast, -search_range, search_range)
      a_test <- strenv$jnp$array(current_ast + offsets)
      Q_test <- eval_Q(a_test, a_i_dag_current, Q_SIGN = 1.0)
      if (Q_test > best_Q_ast) {
        best_Q_ast <- Q_test
        best_a_ast <- a_test
      }
    }
  }

  # Find best response for DAG (fixing AST)
  # Note: DAG minimizes, so we use Q_SIGN = -1 and look for max of that
  best_Q_dag <- eval_Q(a_i_ast_current, a_i_dag_current, Q_SIGN = -1.0)
  best_a_dag <- a_i_dag_current

  if (n_params_dag == 1) {
    for (offset in grid_ast) {
      a_test <- strenv$jnp$array(current_dag + offset)
      Q_test <- eval_Q(a_i_ast_current, a_test, Q_SIGN = -1.0)
      if (Q_test > best_Q_dag) {
        best_Q_dag <- Q_test
        best_a_dag <- a_test
      }
    }
  } else {
    for (i in seq_len(resolution^2)) {
      offsets <- runif(n_params_dag, -search_range, search_range)
      a_test <- strenv$jnp$array(current_dag + offsets)
      Q_test <- eval_Q(a_i_ast_current, a_test, Q_SIGN = -1.0)
      if (Q_test > best_Q_dag) {
        best_Q_dag <- Q_test
        best_a_dag <- a_test
      }
    }
  }

  return(list(
    Q_br_ast = best_Q_ast,
    Q_br_dag = best_Q_dag,
    br_a_ast = best_a_ast,
    br_a_dag = best_a_dag
  ))
}


#' Internal: Random search for best response (high-dimensional)
#' @keywords internal
#' @noRd
find_best_response_random <- function(result, a_i_ast_current, a_i_dag_current,
                                       eval_Q, n_samples, verbose) {

  strenv <- result$strenv

  current_ast <- as.numeric(strenv$np$array(a_i_ast_current))
  current_dag <- as.numeric(strenv$np$array(a_i_dag_current))

  n_params_ast <- length(current_ast)
  n_params_dag <- length(current_dag)

  search_range <- 2.0

  # Find best response for AST
  best_Q_ast <- eval_Q(a_i_ast_current, a_i_dag_current, Q_SIGN = 1.0)
  best_a_ast <- a_i_ast_current

  for (i in seq_len(n_samples)) {
    offsets <- rnorm(n_params_ast, mean = 0, sd = search_range / 2)
    a_test <- strenv$jnp$array(current_ast + offsets)
    Q_test <- eval_Q(a_test, a_i_dag_current, Q_SIGN = 1.0)
    if (Q_test > best_Q_ast) {
      best_Q_ast <- Q_test
      best_a_ast <- a_test
    }
  }

  # Find best response for DAG
  best_Q_dag <- eval_Q(a_i_ast_current, a_i_dag_current, Q_SIGN = -1.0)
  best_a_dag <- a_i_dag_current

  for (i in seq_len(n_samples)) {
    offsets <- rnorm(n_params_dag, mean = 0, sd = search_range / 2)
    a_test <- strenv$jnp$array(current_dag + offsets)
    Q_test <- eval_Q(a_i_ast_current, a_test, Q_SIGN = -1.0)
    if (Q_test > best_Q_dag) {
      best_Q_dag <- Q_test
      best_a_dag <- a_test
    }
  }

  return(list(
    Q_br_ast = best_Q_ast,
    Q_br_dag = best_Q_dag,
    br_a_ast = best_a_ast,
    br_a_dag = best_a_dag
  ))
}


#' Internal: Gradient-based best response search
#' @keywords internal
#' @noRd
find_best_response_gradient <- function(result, a_i_ast_current, a_i_dag_current,
                                         eval_Q, verbose) {

  strenv <- result$strenv

  # Use the existing gradient functions
  dQ_da_ast <- result$dQ_da_ast
  dQ_da_dag <- result$dQ_da_dag

  if (is.null(dQ_da_ast) || is.null(dQ_da_dag)) {
    stop("Gradient functions not available in result. ",
         "Cannot use gradient method.")
  }

  # Get parameters
  params_ast <- result$gather_fxn(result$REGRESSION_PARAMETERS_ast)
  params_dag <- result$gather_fxn(result$REGRESSION_PARAMETERS_dag)
  params_ast0 <- result$gather_fxn(result$REGRESSION_PARAMETERS_ast0)
  params_dag0 <- result$gather_fxn(result$REGRESSION_PARAMETERS_dag0)

  INTERCEPT_ast <- params_ast[[1]]
  COEFFICIENTS_ast <- params_ast[[2]]
  INTERCEPT_dag <- params_dag[[1]]
  COEFFICIENTS_dag <- params_dag[[2]]
  INTERCEPT_ast0 <- params_ast0[[1]]
  COEFFICIENTS_ast0 <- params_ast0[[2]]
  INTERCEPT_dag0 <- params_dag0[[1]]
  COEFFICIENTS_dag0 <- params_dag0[[2]]

  P_VEC_FULL_ast <- result$P_VEC_FULL_ast
  P_VEC_FULL_dag <- result$P_VEC_FULL_dag
  SLATE_VEC_ast <- result$SLATE_VEC_ast
  SLATE_VEC_dag <- result$SLATE_VEC_dag
  LAMBDA <- strenv$jnp$array(result$lambda)

  # Run additional gradient steps for AST
  a_ast <- a_i_ast_current
  n_steps <- 100
  lr <- 0.01

  for (i in seq_len(n_steps)) {
    SEED <- strenv$jax$random$PRNGKey(as.integer(i))
    grad_result <- dQ_da_ast(
      a_ast, a_i_dag_current,
      INTERCEPT_ast, COEFFICIENTS_ast,
      INTERCEPT_dag, COEFFICIENTS_dag,
      INTERCEPT_ast0, COEFFICIENTS_ast0,
      INTERCEPT_dag0, COEFFICIENTS_dag0,
      P_VEC_FULL_ast, P_VEC_FULL_dag,
      SLATE_VEC_ast, SLATE_VEC_dag,
      LAMBDA,
      strenv$jnp$array(1.0),
      SEED
    )
    grad <- grad_result[[2]]
    a_ast <- a_ast + lr * grad
  }

  Q_br_ast <- eval_Q(a_ast, a_i_dag_current, Q_SIGN = 1.0)

  # Run additional gradient steps for DAG
  a_dag <- a_i_dag_current

  for (i in seq_len(n_steps)) {
    SEED <- strenv$jax$random$PRNGKey(as.integer(i + 1000))
    grad_result <- dQ_da_dag(
      a_i_ast_current, a_dag,
      INTERCEPT_ast, COEFFICIENTS_ast,
      INTERCEPT_dag, COEFFICIENTS_dag,
      INTERCEPT_ast0, COEFFICIENTS_ast0,
      INTERCEPT_dag0, COEFFICIENTS_dag0,
      P_VEC_FULL_ast, P_VEC_FULL_dag,
      SLATE_VEC_ast, SLATE_VEC_dag,
      LAMBDA,
      strenv$jnp$array(-1.0),
      SEED
    )
    grad <- grad_result[[2]]
    a_dag <- a_dag + lr * grad
  }

  Q_br_dag <- eval_Q(a_i_ast_current, a_dag, Q_SIGN = -1.0)

  return(list(
    Q_br_ast = Q_br_ast,
    Q_br_dag = Q_br_dag,
    br_a_ast = a_ast,
    br_a_dag = a_dag
  ))
}


#' Internal: Plot equilibrium validation results
#' @keywords internal
#' @noRd
plot_equilibrium_validation <- function(br_error_ast, br_error_dag, tolerance) {

  # Base R bar plot
  old_par <- par(mar = c(5, 4, 4, 2))
  on.exit(par(old_par))

  errors <- c(AST = br_error_ast, DAG = br_error_dag)
  colors <- ifelse(errors < tolerance, "#4DAF4A", "#E41A1C")

  bp <- barplot(errors,
                main = "Best-Response Error by Player",
                ylab = "BR Error (improvement potential)",
                col = colors,
                ylim = c(0, max(errors, tolerance) * 1.2),
                border = "gray30")

  abline(h = tolerance, lty = 2, col = "gray50", lwd = 2)
  text(mean(bp), tolerance, sprintf("Tolerance = %.3f", tolerance),
       pos = 3, col = "gray40")

  # Add pass/fail labels
  text(bp, errors + max(errors) * 0.05,
       ifelse(errors < tolerance, "PASS", "FAIL"),
       col = colors, font = 2)

  invisible(NULL)
}


#' Internal: Evaluate Hessian at equilibrium
#' @keywords internal
#' @noRd
eval_hessian <- function(result, player = c("ast", "dag")) {
  player <- match.arg(player)
  strenv <- result$strenv

  # Get Q evaluation parameters (same as validate_equilibrium)
  gather_fxn <- result$gather_fxn
  if (is.null(gather_fxn)) {
    gather_fxn <- function(x) x
  }

  params_ast <- gather_fxn(result$REGRESSION_PARAMETERS_ast)
  params_dag <- gather_fxn(result$REGRESSION_PARAMETERS_dag)
  params_ast0 <- gather_fxn(result$REGRESSION_PARAMETERS_ast0)
  params_dag0 <- gather_fxn(result$REGRESSION_PARAMETERS_dag0)

  INTERCEPT_ast <- params_ast[[1]]
  COEFFICIENTS_ast <- params_ast[[2]]
  INTERCEPT_dag <- params_dag[[1]]
  COEFFICIENTS_dag <- params_dag[[2]]
  INTERCEPT_ast0 <- params_ast0[[1]]
  COEFFICIENTS_ast0 <- params_ast0[[2]]
  INTERCEPT_dag0 <- params_dag0[[1]]
  COEFFICIENTS_dag0 <- params_dag0[[2]]

  # Select Hessian function and sign
  hess_fn <- if (player == "ast") result$d2Q_da2_ast else result$d2Q_da2_dag
  Q_SIGN <- if (player == "ast") 1.0 else -1.0

  # Evaluate Hessian
  SEED <- strenv$jax$random$PRNGKey(42L)
  H <- hess_fn(
    result$a_i_ast, result$a_i_dag,
    INTERCEPT_ast, COEFFICIENTS_ast,
    INTERCEPT_dag, COEFFICIENTS_dag,
    INTERCEPT_ast0, COEFFICIENTS_ast0,
    INTERCEPT_dag0, COEFFICIENTS_dag0,
    result$P_VEC_FULL_ast, result$P_VEC_FULL_dag,
    result$SLATE_VEC_ast, result$SLATE_VEC_dag,
    strenv$jnp$array(result$lambda),
    strenv$jnp$array(Q_SIGN),
    SEED
  )

  return(H)
}


#' Check Hessian Geometry at Nash Equilibrium
#'
#' Computes Hessian eigenvalues to verify proper saddle point structure
#' at the Nash equilibrium. For a valid Nash equilibrium in a zero-sum game:
#' \itemize{
#'   \item AST (maximizer) Hessian should be negative semi-definite (all eigenvalues <= 0)
#'   \item DAG (minimizer) Hessian should be positive semi-definite (all eigenvalues >= 0)
#' }
#'
#' @param result Output from \code{\link{strategize}} with \code{adversarial = TRUE}
#' @param tol Numeric. Tolerance for near-zero eigenvalues (default 1e-6)
#' @param verbose Logical. Whether to print progress messages. Default is TRUE.
#'
#' @return A list of class \code{"hessian_analysis"} containing:
#'   \describe{
#'     \item{status}{Character: "PASS", "WARNING", or "FAIL"}
#'     \item{valid_saddle}{Logical. TRUE if both Hessians have correct definiteness}
#'     \item{eigenvalues_ast}{Eigenvalues of AST player's Hessian}
#'     \item{eigenvalues_dag}{Eigenvalues of DAG player's Hessian}
#'     \item{is_negative_semidefinite_ast}{Logical. TRUE if AST Hessian is negative semi-definite}
#'     \item{is_positive_semidefinite_dag}{Logical. TRUE if DAG Hessian is positive semi-definite}
#'     \item{condition_number_ast}{Condition number of AST Hessian (stability indicator)}
#'     \item{condition_number_dag}{Condition number of DAG Hessian}
#'     \item{flat_directions_ast}{Number of near-zero eigenvalues (weak identification)}
#'     \item{flat_directions_dag}{Number of near-zero eigenvalues}
#'     \item{interpretation}{Character string with human-readable interpretation}
#'   }
#'   Returns NULL with a warning if Hessian functions are not available.
#'
#' @details
#' \strong{Mathematical Foundation}
#'
#' At a Nash equilibrium in a zero-sum game, the Hessian matrices encode local
#' curvature information:
#' \itemize{
#'   \item \strong{AST (maximizes Q)}: The Hessian should be negative semi-definite,
#'     meaning all eigenvalues are <= 0. This confirms AST is at a local maximum.
#'   \item \strong{DAG (minimizes Q)}: The Hessian should be positive semi-definite,
#'     meaning all eigenvalues are >= 0. This confirms DAG is at a local minimum.
#' }
#'
#' The condition number (ratio of largest to smallest eigenvalue magnitude)
#' indicates equilibrium robustness. High condition numbers suggest the
#' equilibrium is sensitive to small perturbations.
#'
#' \strong{Interpretation}
#'
#' \itemize{
#'   \item \code{status = "PASS"}: Valid saddle point with no flat directions
#'   \item \code{status = "WARNING"}: Valid saddle point but has flat directions
#'     (weak identification). Consider increasing regularization.
#'   \item \code{status = "FAIL"}: Not a proper saddle point. The solution may
#'     not have converged to a true Nash equilibrium.
#' }
#'
#' @examples
#' \dontrun{
#' # Run adversarial strategize
#' result <- strategize(Y = y, W = w, adversarial = TRUE, nSGD = 500)
#'
#' # Check Hessian geometry
#' hess <- check_hessian_geometry(result)
#' print(hess)
#'
#' # Check if it's a valid saddle point
#' if (hess$valid_saddle) {
#'   message("Valid Nash equilibrium geometry!")
#' }
#' }
#'
#' @seealso \code{\link{validate_equilibrium}} for best-response validation
#'
#' @export
check_hessian_geometry <- function(result, tol = 1e-6, verbose = TRUE) {

  # Check if this is an adversarial result
  if (!isTRUE(result$convergence_history$adversarial)) {
    stop("Hessian analysis requires adversarial=TRUE result. ",
         "Set adversarial = TRUE in strategize().")
  }

  # Check if Hessian was computed
  if (!isTRUE(result$hessian_available)) {
    reason <- result$hessian_skipped_reason
    if (is.null(reason)) reason <- "unknown"
    warning(sprintf("Hessian not available (reason: %s). ", reason),
            "Re-run strategize() with compute_hessian=TRUE or adjust hessian_max_dim.")
    return(NULL)
  }

  strenv <- result$strenv
  if (is.null(strenv) || is.null(strenv$jnp)) {
    stop("JAX environment not available in result.")
  }

  if (verbose) message("Computing Hessian eigenvalue analysis...")

  # Compute Hessians
  H_ast <- eval_hessian(result, player = "ast")
  H_dag <- eval_hessian(result, player = "dag")

  # Convert to R matrices
  H_ast_r <- as.matrix(strenv$np$array(H_ast))
  H_dag_r <- as.matrix(strenv$np$array(H_dag))

  # Eigendecomposition (symmetric Hessian)
  eig_ast <- eigen(H_ast_r, symmetric = TRUE)
  eig_dag <- eigen(H_dag_r, symmetric = TRUE)

  # Check definiteness
  # AST maximizes: needs negative semi-definite (lambda <= 0)
  is_neg_def_ast <- all(eig_ast$values < -tol)
  is_neg_semidef_ast <- all(eig_ast$values <= tol)

  # DAG minimizes Q (equivalently maximizes -Q): needs positive semi-definite (lambda >= 0)
  is_pos_def_dag <- all(eig_dag$values > tol)
  is_pos_semidef_dag <- all(eig_dag$values >= -tol)

  # Condition numbers (stability indicator)
  nonzero_ast <- abs(eig_ast$values) > tol
  nonzero_dag <- abs(eig_dag$values) > tol

  cond_ast <- if (sum(nonzero_ast) >= 2) {
    max(abs(eig_ast$values[nonzero_ast])) / min(abs(eig_ast$values[nonzero_ast]))
  } else {
    NA_real_
  }

  cond_dag <- if (sum(nonzero_dag) >= 2) {
    max(abs(eig_dag$values[nonzero_dag])) / min(abs(eig_dag$values[nonzero_dag]))
  } else {
    NA_real_
  }

  # Count flat directions
  flat_ast <- sum(abs(eig_ast$values) < tol)
  flat_dag <- sum(abs(eig_dag$values) < tol)

  # Overall validity
  valid_saddle <- is_neg_semidef_ast && is_pos_semidef_dag

  # Status determination
  if (valid_saddle && flat_ast == 0 && flat_dag == 0) {
    status <- "PASS"
  } else if (valid_saddle) {
    status <- "WARNING"  # Valid but has flat directions
  } else {
    status <- "FAIL"
  }

  # Generate interpretation
  interpretation <- interpret_hessian_result(
    status, valid_saddle,
    is_neg_semidef_ast, is_pos_semidef_dag,
    flat_ast, flat_dag,
    cond_ast, cond_dag,
    eig_ast$values, eig_dag$values
  )

  if (verbose) {
    message(sprintf("  Status: %s", status))
    message(sprintf("  AST: %s (eigenvalues: [%.4f, %.4f])",
                    ifelse(is_neg_semidef_ast, "negative semi-def", "NOT negative semi-def"),
                    min(eig_ast$values), max(eig_ast$values)))
    message(sprintf("  DAG: %s (eigenvalues: [%.4f, %.4f])",
                    ifelse(is_pos_semidef_dag, "positive semi-def", "NOT positive semi-def"),
                    min(eig_dag$values), max(eig_dag$values)))
  }

  result_obj <- list(
    status = status,
    valid_saddle = valid_saddle,

    # AST (maximizer) analysis
    eigenvalues_ast = eig_ast$values,
    eigenvectors_ast = eig_ast$vectors,
    is_negative_definite_ast = is_neg_def_ast,
    is_negative_semidefinite_ast = is_neg_semidef_ast,
    condition_number_ast = cond_ast,
    flat_directions_ast = flat_ast,

    # DAG (minimizer) analysis
    eigenvalues_dag = eig_dag$values,
    eigenvectors_dag = eig_dag$vectors,
    is_positive_definite_dag = is_pos_def_dag,
    is_positive_semidefinite_dag = is_pos_semidef_dag,
    condition_number_dag = cond_dag,
    flat_directions_dag = flat_dag,

    # Interpretation
    interpretation = interpretation,
    tolerance = tol
  )

  class(result_obj) <- c("hessian_analysis", "list")
  return(result_obj)
}


#' Internal: Generate interpretation text for Hessian analysis
#' @keywords internal
#' @noRd
interpret_hessian_result <- function(status, valid_saddle,
                                      is_neg_semidef_ast, is_pos_semidef_dag,
                                      flat_ast, flat_dag,
                                      cond_ast, cond_dag,
                                      eigenvalues_ast, eigenvalues_dag) {
  lines <- character()

  if (status == "PASS") {
    lines <- c(lines,
      "The solution is a valid Nash equilibrium (proper saddle point).",
      "- AST is at a local maximum (Hessian negative definite)",
      "- DAG is at a local minimum (Hessian positive definite)",
      "Neither player can improve by small local deviations."
    )
  } else if (status == "WARNING") {
    lines <- c(lines,
      "The solution has correct saddle point structure but with flat directions.",
      sprintf("- AST has %d near-zero eigenvalue(s)", flat_ast),
      sprintf("- DAG has %d near-zero eigenvalue(s)", flat_dag),
      "This indicates weak identification on some parameters.",
      "Consider increasing regularization (lambda) to sharpen the equilibrium."
    )
  } else {
    lines <- c(lines, "The solution is NOT a valid saddle point!")
    if (!is_neg_semidef_ast) {
      pos_eigs <- sum(eigenvalues_ast > 0)
      lines <- c(lines,
        sprintf("- AST has %d positive eigenvalue(s): not at a local maximum", pos_eigs),
        "  AST could improve by moving in these directions."
      )
    }
    if (!is_pos_semidef_dag) {
      neg_eigs <- sum(eigenvalues_dag < 0)
      lines <- c(lines,
        sprintf("- DAG has %d negative eigenvalue(s): not at a local minimum", neg_eigs),
        "  DAG could improve by moving in these directions."
      )
    }
    lines <- c(lines,
      "",
      "Recommendations:",
      "- Increase nSGD iterations to allow better convergence",
      "- Check learning rate (may be too high causing oscillation)",
      "- Verify the problem is well-posed"
    )
  }

  # Add condition number warnings
  if (!is.na(cond_ast) && cond_ast > 100) {
    lines <- c(lines, "",
      sprintf("Warning: AST condition number is high (%.1f), equilibrium may be fragile.", cond_ast))
  }
  if (!is.na(cond_dag) && cond_dag > 100) {
    lines <- c(lines,
      sprintf("Warning: DAG condition number is high (%.1f), equilibrium may be fragile.", cond_dag))
  }

  paste(lines, collapse = "\n")
}


#' Print method for hessian_analysis objects
#'
#' @param x A hessian_analysis object from \code{\link{check_hessian_geometry}}
#' @param ... Additional arguments (ignored)
#'
#' @export
print.hessian_analysis <- function(x, ...) {
  cat("\n=== Hessian Geometry Analysis ===\n")
  cat(sprintf("Status: %s\n\n", x$status))

  cat("AST Player (Maximizer):\n")
  cat(sprintf("  Negative semi-definite: %s\n", x$is_negative_semidefinite_ast))
  cat(sprintf("  Eigenvalue range: [%.6f, %.6f]\n",
              min(x$eigenvalues_ast), max(x$eigenvalues_ast)))
  if (!is.na(x$condition_number_ast)) {
    cat(sprintf("  Condition number: %.2f\n", x$condition_number_ast))
  }
  cat(sprintf("  Flat directions: %d\n\n", x$flat_directions_ast))

  cat("DAG Player (Minimizer):\n")
  cat(sprintf("  Positive semi-definite: %s\n", x$is_positive_semidefinite_dag))
  cat(sprintf("  Eigenvalue range: [%.6f, %.6f]\n",
              min(x$eigenvalues_dag), max(x$eigenvalues_dag)))
  if (!is.na(x$condition_number_dag)) {
    cat(sprintf("  Condition number: %.2f\n", x$condition_number_dag))
  }
  cat(sprintf("  Flat directions: %d\n\n", x$flat_directions_dag))

  cat("Interpretation:\n")
  cat(x$interpretation, "\n")

  invisible(x)
}
