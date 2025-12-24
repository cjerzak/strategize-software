#' Plot Dimension-by-Dimension Best-Response Curves from Adversarial \code{strategize()} Output
#'
#' @description
#' \code{plot_best_response_curves} takes the result of an adversarial 
#' \code{\link{strategize}} run (i.e., with \code{adversarial = TRUE}) and produces 
#' dimension-specific best-response curves. Specifically, for a chosen factor dimension 
#' \eqn{d}, it plots:
#' \enumerate{
#'   \item The curve of \eqn{\pi_{\mathrm{dag}, d}^{*}} as a function of 
#'         \eqn{\pi_{\mathrm{ast}, d}}.
#'   \item The curve of \eqn{\pi_{\mathrm{ast}, d}^{*}} as a function of 
#'         \eqn{\pi_{\mathrm{dag}, d}}.
#' }
#' Potential intersection points in this 2D space can indicate approximate equilibria 
#' for dimension \eqn{d}, holding the other dimensions fixed at the solution found by 
#' \code{strategize}.
#'
#' This function is computationally intensive: for each of \code{nPoints_br} grid values
#' of \eqn{\pi_{\mathrm{ast}, d}}, it searches over possible \eqn{\pi_{\mathrm{dag}, d}} 
#' (and vice versa) to find each side's best response, re-running partial objective evaluations. 
#' Nonetheless, it provides a direct visualization of how each player (ast or dag) responds 
#' to changes in the other's distribution along a single factor dimension.
#'
#' @usage
#' plot_best_response_curves(
#'   res,
#'   d_ = 1,
#'   nPoints_br = 100L,
#'   nPoints_heat = 50L,
#'   title = NULL,
#'   col_ast = "blue",
#'   col_dag = "red",
#'   lwd_ast = 2,
#'   lwd_dag = 2,
#'   point_pch = 19,
#'   silent = FALSE
#' )
#'
#' @param res A list returned by \code{\link{strategize}}, which must include 
#'   adversarial references. Internally, \code{res} should contain items like 
#'   \code{res$a_i_ast}, \code{res$a_i_dag}, the JAX-based functions 
#'   (\code{dQ_da_ast}, \code{dQ_da_dag}, \code{QFXN}, \ldots), along with the 
#'   appropriate unconstrained parameter vectors.
#' @param d_ (Integer) The dimension of \eqn{\pi_{\mathrm{ast}}, \pi_{\mathrm{dag}}} to examine. 
#'   For example, if you have multiple factors (dimensions), each is indexed by a positive integer. 
#'   Defaults to 1.
#' @param nPoints_br (Integer) Number of equally spaced grid points in \eqn{[0,1]}
#'   to sample for \emph{the outer} loop. The code does an internal small search
#'   for best responses at each grid point. Larger \code{nPoints_br} => smoother curves
#'   but more computation. Defaults to \code{100L}.
#' @param nPoints_heat (Integer) Number of grid points for heat map calculations.
#'   Defaults to \code{50L}.
#' @param title (Character or \code{NULL}) Main plot title. If \code{NULL}, 
#'   an auto-generated title is used, e.g. "Best-Response Curves (Dimension d_=1)".
#' @param col_ast,col_dag (Character) Colors for ast’s and dag’s best-response curves, 
#'   respectively. Defaults to \code{"blue"} (ast) and \code{"red"} (dag).
#' @param lwd_ast,lwd_dag (Numeric) Line widths for ast and dag curves, respectively. 
#'   Default is 2.
#' @param point_pch (Numeric) Symbol for marking the approximate intersection 
#'   (if found) on the plot. Defaults to 19 (filled circle).
#' @param silent (Logical) If \code{TRUE}, suppresses printed messages during 
#'   the search for intersection. Defaults to \code{FALSE}.
#'
#' @return (Invisibly) A list containing:
#' \describe{
#'   \item{\code{grid_points}}{A numeric vector of the \code{nPoints_br} grid 
#'         values in \eqn{[0,1]} used for the outer loop.}
#'   \item{\code{br_dag_given_ast}}{A numeric vector of the same length, giving 
#'         the dag best-response \eqn{\pi_{\mathrm{dag}, d}} at each grid point 
#'         for \eqn{\pi_{\mathrm{ast}, d}}.}
#'   \item{\code{br_ast_given_dag}}{A numeric vector with the ast best-response 
#'         for each grid point \eqn{\pi_{\mathrm{dag}, d}}.}
#' }
#'
#' @details
#' \strong{Mechanics:}
#' For each \eqn{\pi_{\mathrm{ast}, d} \in \{0,\frac{1}{nPoints_br-1},\ldots,1\}}, 
#' the function temporarily fixes that dimension in the ast player’s unconstrained 
#' parameter vector. It then does an internal grid search over \eqn{\pi_{\mathrm{dag}, d}} 
#' to see which value yields the largest dag payoff (lowest ast payoff), consistent 
#' with \code{adversarial=TRUE}. The resulting curve is \eqn{\mathrm{BR}_{\mathrm{dag}}(\pi_{\mathrm{ast},d})}.
#'
#' Likewise, it holds \eqn{\pi_{\mathrm{dag}, d}} fixed and searches over 
#' \eqn{\pi_{\mathrm{ast}, d}} to get \eqn{\mathrm{BR}_{\mathrm{ast}}(\pi_{\mathrm{dag},d})}.
#'
#' The intersection in \eqn{\bigl(\pi_{\mathrm{ast}, d}, \pi_{\mathrm{dag}, d}\bigr)}-space 
#' (if one exists in the discretized grid) is a candidate local equilibrium for that factor 
#' dimension, \emph{given that the other factor dimensions remain at the solution from 
#' \code{\link{strategize}}.}
#'
#' \strong{Performance Caution:} This brute force line-search re-runs partial 
#' objective evaluations many times, which may be slow for large \code{nPoints_br} or 
#' complex outcome models. Consider using a smaller \code{nPoints_br} if performance 
#' is an issue, or focusing on only a handful of crucial dimensions \eqn{d}.
#'
#' @seealso
#' \code{\link{strategize}} for obtaining the result object \code{res} in adversarial mode.
#' See also \code{\link{cv_strategize}}, and if one-step M-estimation is desired, 
#' see \code{\link{strategize_onestep}}.
#'
#' @examples
#' \dontrun{
#' # After fitting an adversarial strategize model:
#' adv_res <- strategize(
#'   Y = Yobs,
#'   W = W,
#'   adversarial = TRUE,
#'   ...
#' )
#'
#' # Suppose dimension 1 is "Gender." Then to see each player's best response:
#' plot_best_response_curves(
#'   res    = adv_res,
#'   d_     = 1,
#'   nPoints_br= 41,         # can reduce or enlarge
#'   title  = "Gender Best-Response Curves",
#'   col_ast= "blue",
#'   col_dag= "red"
#' )
#'
#' # The intersection (if shown) approximates an equilibrium for dimension 1.
#' }
#'
#' @md
#' @export
plot_best_response_curves  <- function(
                               res,
                               d_ = 1,
                               nPoints_br = 100L,
                               nPoints_heat = 50L,
                               title = NULL,
                               col_ast = "blue",
                               col_dag = "red",
                               lwd_ast = 2,
                               lwd_dag = 2,
                               point_pch = 19,
                               silent = FALSE){
  # define evaluation environment 
  evaluation_environment <- environment()
  package_environment <- environment(res$QFXN)
  strenv <- res$strenv
  compile_fxn <- function(x, static_argnums=NULL){return(strenv$jax$jit(x, static_argnums=static_argnums))}
  if(!silent) {
    message(sprintf(
      "[plot_best_response_curves] Building best-response curves for dimension d_ = %d...",
      d_))
  }
  
  ########################################################################
  ## 1) Extract from res the references we need
  ########################################################################
  a_ast_current <- res$a_i_ast
  a_dag_current <- res$a_i_dag
  
  # JAX function references
  #dQ_da_ast <- res$dQ_da_ast
  #dQ_da_dag <- res$dQ_da_dag
  #QFXN      <- res$QFXN
  #getQPiStar_gd      <- res$getQPiStar_gd
  #getMultinomialSamp <- res$getMultinomialSamp
  #getPrettyPi_diff <- res$getPrettyPi_diff
  #getPrettyPi_diff_R <- res$getPrettyPi_diff_R
  #environment(getPrettyPi_diff) <- strenv
  
  # Pi vectors + slates
  P_VEC_FULL_ast <- res$P_VEC_FULL_ast
  P_VEC_FULL_dag <- res$P_VEC_FULL_dag
  SLATE_VEC_ast  <- res$SLATE_VEC_ast
  SLATE_VEC_dag  <- res$SLATE_VEC_dag
  AstProp <- res$AstProp
  DagProp <- res$DagProp
  adversarial <- TRUE 
  
  # Regression param arrays
  REGRESSION_PARAMS_ast  <- res$REGRESSION_PARAMETERS_ast
  REGRESSION_PARAMS_dag  <- res$REGRESSION_PARAMETERS_dag
  REGRESSION_PARAMS_ast0 <- res$REGRESSION_PARAMETERS_ast0
  REGRESSION_PARAMS_dag0 <- res$REGRESSION_PARAMETERS_dag0
  
  # gather_fxn, a2Simplex, lambda
  gather_fxn          <- res$gather_fxn
  LAMBDA_             <- strenv$jnp$array(res$lambda, strenv$dtj)
  penalty_type        <- res$penalty_type
  
  d_locator <- res$d_locator
  d_locator_use <- res$d_locator_use
  main_comp_mat <- res$main_comp_mat
  shadow_comp_mat <- res$shadow_comp_mat
  intersection_dist_threshold <- 0.001
  
  START_VAL_SEARCH <- 0; STOP_VAL_SEARCH <- 1
  
  ########################################################################
  ## 2) Helper: We need to "override" dimension d_ of the unconstrained 'a' 
  ##    to achieve a specific [0,1] probability in that dimension. 
  ##    We do a small numeric bisection for each point.
  ########################################################################
  override_a_value <- function(a_in, dimensionIndex, newPiValue) {
    dimensionIndex <- ai(dimensionIndex - 1L) # for zero indices 
    a0 <- -10
    b0 <- 10
    a_use <- a_in
    for (iter_ in 1:18) {
      m_ <- 0.5 * (a0 + b0)
      a_use <- a_use$at[[dimensionIndex]]$set(strenv$jnp$array(m_, strenv$dtj))  # update a_in
      piTrial <- strenv$a2Simplex_diff_use(a_use)
      piValTrial <- as.numeric(strenv$np$array(piTrial)[dimensionIndex+1])
      if (piValTrial < newPiValue) {
        a0 <- m_
      } else {
        b0 <- m_
      }
    }
    return(a_use)
  }
  
  # setup environments 
  # multiround material
  for(DisaggreateQ in ifelse(adversarial, 
                             yes = list(c(F,T)), 
                             no = list(F))[[1]]){
    # general specifications
    getQStar_diff_ <- paste(deparse(getQStar_diff_BASE),collapse="\n")
    getQStar_diff_ <- gsub(getQStar_diff_, pattern = "Q_DISAGGREGATE", replace = sprintf("T == %s", DisaggreateQ))
    getQStar_diff_ <- eval( parse( text = getQStar_diff_ ), envir = package_environment )
    
    # specifications for case (getQStar_diff_MultiGroup getQStar_diff_SingleGroup)
    eval(parse(text = sprintf("getQStar_diff_%sGroup <- compile_fxn( getQStar_diff_ )", 
                              ifelse(DisaggreateQ, yes = "Multi", no = "Single") )))
  }
  
  # environment management
  adversarial <- TRUE
  nMonte_adversarial <- 200L
  MNtemp <- res$temperature
  nMonte_Qglm <- 200L
  # d_locator_use already defined at line 186
  environment(adversarial) <- evaluation_environment
  environment(AstProp) <- environment(DagProp) <- evaluation_environment
  # rlang and reticulate are in Suggests/Imports - use :: syntax when needed
  environment(FullGetQStar_) <- evaluation_environment
  FullGetQStar_jit <- strenv$jax$jit(FullGetQStar_,
                                  static_argnames = c("ParameterizationType", "d_locator"))
  
  ########################################################################
  ## 3) We define two "best response" functions:
  ##    - best_response_dag_for_astVal: given x in [0..1] for pi_ast[d_], 
  ##      find argmax for dag. 
  ##    - best_response_ast_for_dagVal: similarly. 
  ##    Both do a small grid search. 
  ########################################################################
  # We'll do a "small search" in nPoints_br points for the best response.
  oneD_grid <- seq(0,1,length.out=nPoints_br)
  
  best_response_dag_for_astVal <- function(x_ast_dim_d, seed){
    # fix ast dimension d_ = x
    a_ast_FROZEN <- override_a_value(a_ast_current, d_, x_ast_dim_d)
    best_y    <- 0
    best_loss <- -Inf
    cnt__ <- 0
    val_seq <- rep(NA,times = nPoints_br)
    for(yCandidate in oneD_grid){
      cnt__ <- cnt__ + 1 
      a_dag_test <- override_a_value(a_dag_current, d_, yCandidate)

      res_val_ <- FullGetQStar_jit(
          a_i_ast            = a_ast_FROZEN,
          a_i_dag            = a_dag_test,
          INTERCEPT_ast_     = gather_fxn(REGRESSION_PARAMS_ast)[[1]],
          COEFFICIENTS_ast_  = gather_fxn(REGRESSION_PARAMS_ast)[[2]],
          INTERCEPT_dag_     = gather_fxn(REGRESSION_PARAMS_dag)[[1]],
          COEFFICIENTS_dag_  = gather_fxn(REGRESSION_PARAMS_dag)[[2]],
          INTERCEPT_ast0_    = gather_fxn(REGRESSION_PARAMS_ast0)[[1]],
          COEFFICIENTS_ast0_ = gather_fxn(REGRESSION_PARAMS_ast0)[[2]],
          INTERCEPT_dag0_    = gather_fxn(REGRESSION_PARAMS_dag0)[[1]],
          COEFFICIENTS_dag0_ = gather_fxn(REGRESSION_PARAMS_dag0)[[2]],
          P_VEC_FULL_ast_    = P_VEC_FULL_ast,
          P_VEC_FULL_dag_    = P_VEC_FULL_dag,
          SLATE_VEC_ast_     = SLATE_VEC_ast,
          SLATE_VEC_dag_     = SLATE_VEC_dag,
          LAMBDA_            = LAMBDA_,
          Q_SIGN             = strenv$jnp$array(-1), # Evaluate objective with Q_SIGN_=-1 for dag's side
          SEED_IN_LOOP       = strenv$jnp$array(ai(cnt__*seed)),
          
          ParameterizationType = strenv$ParameterizationType, # don't trace
          d_locator         = d_locator,  # don't trace 
          main_comp_mat     = main_comp_mat,
          shadow_comp_mat   = shadow_comp_mat 
      )
      
      # Convert the result to a numeric value (assuming the first element holds the objective value)
      val_seq[cnt__] <- val__ <- strenv$np$array(res_val_)[1]
      # Larger val__ => better for dag if Q_SIGN=-1 is used internally
      if(val__ > best_loss){
        best_loss <- val__
        best_y    <- yCandidate
      }
    }
    # plot(oneD_grid, val_seq); abline(v=best_y)
    return(best_y)
  }
  
  best_response_ast_for_dagVal <- function(y_dag_dim_d, seed){
    # fix dag dimension d_ = y
    a_dag_FROZEN  <- override_a_value(a_dag_current, d_, y_dag_dim_d)
    best_x     <- 0
    best_loss  <- -Inf
    cntr__ <- 0
    val_seq <- rep(NA,times = nPoints_br)
    for(xCandidate in oneD_grid){
      cntr__<- cntr__ +1 
      a_ast_test <- override_a_value(a_ast_current, d_, xCandidate)
      
      # Evaluate objective with Q_SIGN_=+1 for ast
      res_val_ <- FullGetQStar_jit(
        a_i_ast            = a_ast_test,
        a_i_dag            = a_dag_FROZEN,
        INTERCEPT_ast_     = gather_fxn(REGRESSION_PARAMS_ast)[[1]],
        COEFFICIENTS_ast_  = gather_fxn(REGRESSION_PARAMS_ast)[[2]],
        INTERCEPT_dag_     = gather_fxn(REGRESSION_PARAMS_dag)[[1]],
        COEFFICIENTS_dag_  = gather_fxn(REGRESSION_PARAMS_dag)[[2]],
        INTERCEPT_ast0_    = gather_fxn(REGRESSION_PARAMS_ast0)[[1]],
        COEFFICIENTS_ast0_ = gather_fxn(REGRESSION_PARAMS_ast0)[[2]],
        INTERCEPT_dag0_    = gather_fxn(REGRESSION_PARAMS_dag0)[[1]],
        COEFFICIENTS_dag0_ = gather_fxn(REGRESSION_PARAMS_dag0)[[2]],
        P_VEC_FULL_ast_    = P_VEC_FULL_ast,
        P_VEC_FULL_dag_    = P_VEC_FULL_dag,
        SLATE_VEC_ast_     = SLATE_VEC_ast,
        SLATE_VEC_dag_     = SLATE_VEC_dag,
        LAMBDA_            = LAMBDA_,
        Q_SIGN             = strenv$jnp$array(1.),
        SEED_IN_LOOP       = strenv$jnp$array(ai(cntr__*seed)),
        
        ParameterizationType = strenv$ParameterizationType,
        d_locator         = d_locator,  
        main_comp_mat     = main_comp_mat,
        shadow_comp_mat   = shadow_comp_mat 
      )
      val_seq[cntr__] <- val__ <- as.numeric( strenv$np$array( res_val_ ) )
      if(val__ > best_loss){
        best_loss <- val__
        best_x    <- xCandidate
      }
    }
    # plot(oneD_grid, val_seq)
    return(best_x)
  }
  
  ########################################################################
  # 4) Evaluate these curves on an outer grid
  ########################################################################
  grid_points <- seq(0,1,length.out=nPoints_br)
  br_ast_given_dag <- br_dag_given_ast <- numeric(nPoints_br)
  for(i_ in seq_along(grid_points)){
    print(i_)
    br_dag_given_ast[i_] <- best_response_dag_for_astVal( grid_points[i_], seed = i_ )
    br_ast_given_dag[i_] <- best_response_ast_for_dagVal( grid_points[i_], seed = i_ )
  }
  
  ########################################################################
  # 5) Plot 
  ########################################################################
  if(is.null(title)){
    mainTitle <- sprintf("Best-Response Curves (Dimension d_=%d)", d_)
  } else {
    mainTitle <- title
  }
  
  # Try to find approximate intersection
  best_pt <- c(NA, NA)
  distMin <- Inf
  for(i in seq_along(grid_points)){
    x_ <- grid_points[i]
    y_ <- br_dag_given_ast[i]
    # find index i2 s.t. grid_points[i2] is close to y_
    i2 <- which.min(abs(grid_points - y_))
    x2_ <- br_ast_given_dag[i2]
    dist_ <- sqrt( (x_ - x2_)^2 + (y_ - grid_points[i2])^2 )
    if(dist_ < distMin){
      distMin <- dist_
      best_pt <- c(x_, y_)
    }
  }
  if(distMin < 0.05){
    if(!silent){
      message(sprintf(
        "Closest intersection is (x=%.3f, y=%.3f) with dist=%.4f in dimension d_=%d.",
        best_pt[1], best_pt[2], distMin, d_))
    }
  }
  
  # HEATMAP 
  {
    xvals <- seq(START_VAL_SEARCH, STOP_VAL_SEARCH, length.out=nPoints_heat)
    yvals <- seq(START_VAL_SEARCH, STOP_VAL_SEARCH, length.out=nPoints_heat)
    
    yvals_mat <- xvals_mat <- matrix(NA_real_, nrow=nPoints_heat, ncol=nPoints_heat)
    z_ast <- matrix(NA_real_, nrow=nPoints_heat, ncol=nPoints_heat)
    z_dag <- matrix(NA_real_, nrow=nPoints_heat, ncol=nPoints_heat)
    
    cnt_ <- 0L
    for(ix in seq_along(xvals)){
      for(iy in seq_along(yvals)){
        cnt_ <- cnt_ + 1L
        if(iy %% 10 == 0){print(c(ix,iy))}
        # override dimension d_ for ast and dag
        a_ast_test <- override_a_value(a_ast_current, d_, xvals[ix])
        a_dag_test <- override_a_value(a_dag_current, d_, yvals[iy])
        
        # AST objective (Q_SIGN=+1 if zero-sum)
        val_ast_ <- FullGetQStar_jit(
          a_i_ast            = a_ast_test,
          a_i_dag            = a_dag_test,
          INTERCEPT_ast_     = gather_fxn(REGRESSION_PARAMS_ast)[[1]],
          COEFFICIENTS_ast_  = gather_fxn(REGRESSION_PARAMS_ast)[[2]],
          INTERCEPT_dag_     = gather_fxn(REGRESSION_PARAMS_dag)[[1]],
          COEFFICIENTS_dag_  = gather_fxn(REGRESSION_PARAMS_dag)[[2]],
          INTERCEPT_ast0_    = gather_fxn(REGRESSION_PARAMS_ast0)[[1]],
          COEFFICIENTS_ast0_ = gather_fxn(REGRESSION_PARAMS_ast0)[[2]],
          INTERCEPT_dag0_    = gather_fxn(REGRESSION_PARAMS_dag0)[[1]],
          COEFFICIENTS_dag0_ = gather_fxn(REGRESSION_PARAMS_dag0)[[2]],
          P_VEC_FULL_ast_    = P_VEC_FULL_ast,
          P_VEC_FULL_dag_    = P_VEC_FULL_dag,
          SLATE_VEC_ast_     = SLATE_VEC_ast,
          SLATE_VEC_dag_     = SLATE_VEC_dag,
          LAMBDA_            = LAMBDA_,
          Q_SIGN             = strenv$jnp$array(1.),
          SEED_IN_LOOP       = strenv$jnp$array(ai(cnt_)),
          ParameterizationType = strenv$ParameterizationType,
          d_locator         = d_locator,  
          main_comp_mat     = main_comp_mat,
          shadow_comp_mat   = shadow_comp_mat
        )
        z_ast[ix, iy] <- strenv$np$array(val_ast_)[1]
        
        # DAG objective (Q_SIGN=-1) if zero-sum 
        val_dag_ <- FullGetQStar_jit(
          a_i_ast            = a_ast_test,
          a_i_dag            = a_dag_test,
          INTERCEPT_ast_     = gather_fxn(REGRESSION_PARAMS_ast)[[1]],
          COEFFICIENTS_ast_  = gather_fxn(REGRESSION_PARAMS_ast)[[2]],
          INTERCEPT_dag_     = gather_fxn(REGRESSION_PARAMS_dag)[[1]],
          COEFFICIENTS_dag_  = gather_fxn(REGRESSION_PARAMS_dag)[[2]],
          INTERCEPT_ast0_    = gather_fxn(REGRESSION_PARAMS_ast0)[[1]],
          COEFFICIENTS_ast0_ = gather_fxn(REGRESSION_PARAMS_ast0)[[2]],
          INTERCEPT_dag0_    = gather_fxn(REGRESSION_PARAMS_dag0)[[1]],
          COEFFICIENTS_dag0_ = gather_fxn(REGRESSION_PARAMS_dag0)[[2]],
          P_VEC_FULL_ast_    = P_VEC_FULL_ast,
          P_VEC_FULL_dag_    = P_VEC_FULL_dag,
          SLATE_VEC_ast_     = SLATE_VEC_ast,
          SLATE_VEC_dag_     = SLATE_VEC_dag,
          LAMBDA_            = LAMBDA_,
          Q_SIGN             = strenv$jnp$array(-1),
          SEED_IN_LOOP       = strenv$jnp$array(ai(cnt_)),
          ParameterizationType = strenv$ParameterizationType,
          d_locator         = d_locator,  
          main_comp_mat     = main_comp_mat,
          shadow_comp_mat   = shadow_comp_mat
        )
        z_dag[ix, iy] <- strenv$np$array(val_dag_)[1]
        xvals_mat[ix,iy] <- xvals[ix] # for ast 
        yvals_mat[ix,iy] <- yvals[iy] # for dag 
      }
    }
  }
  
  # helper fxn
  {
    geom_smooth2 <- function(mapping = NULL,
                             data = NULL,
                             threshold = 3,  # Number of unique points required in x and y
                             method = "loess",
                             se = FALSE,
                             color = "blue",
                             linewidth = 1.2,
                             na.rm = FALSE,
                             show.legend = NA,
                             inherit.aes = TRUE,
                             span = 1.5, 
                             ...) {
      
      # Make sure both 'mapping' and 'data' are provided
      if (is.null(mapping) || is.null(data)) {
        stop("Please specify both 'mapping' and 'data' for geom_smooth2().",
             call. = FALSE)
      }
      
      # Extract x and y variable names from the mapping
      x_var <- rlang::get_expr(mapping$x)
      y_var <- rlang::get_expr(mapping$y)
      
      if (is.null(x_var) || is.null(y_var)) {
        stop("Mapping must define both 'x' and 'y' for geom_smooth2().",
             call. = FALSE)
      }
      
      # Convert x_var and y_var to strings so we can use them to index the data
      x_col <- rlang::as_label(x_var)
      y_col <- rlang::as_label(y_var)
      
      # Count number of unique values for x and y
      n_unique_x <- length(unique(data[[x_col]]))
      n_unique_y <- length(unique(data[[y_col]]))
      
      # Conditionally return geom_smooth or geom_line
      if (n_unique_x > threshold && n_unique_y > threshold) {
        ggplot2::geom_smooth(
          mapping = mapping,
          data = data,
          method = method,
          se = se,
          color = color,
          linewidth = linewidth,
          na.rm = na.rm,
          show.legend = show.legend,
          inherit.aes = inherit.aes,
          span = span, 
          ...
        )
      } else {
        ggplot2::geom_line(
          mapping = mapping,
          data = data,
          color = color,
          linewidth = linewidth,
          na.rm = na.rm,
          show.legend = show.legend,
          inherit.aes = inherit.aes,
          ...
        )
      }
    }
  }
  
  # HEATMAP - MORE PLOTTING 
  {
    #############################################################################
    # 4) Convert these utility matrices to data.frames for ggplot
    ##############################################################################
    # We'll rely on row ix varying fastest along x, col iy along y
    # Actually, it’s standard to do expand.grid(xvals, yvals).
    # Then fill row by row: z_ast[ix, iy] => df_ast$utility_ast[row].
    if (!requireNamespace("ggplot2", quietly = TRUE)) {
      stop("Package 'ggplot2' is required for plotting. Please install it.", call. = FALSE)
    }
    if (!requireNamespace("gridExtra", quietly = TRUE)) {
      stop("Package 'gridExtra' is required for plotting. Please install it.", call. = FALSE)
    }
    
    df_ast <- data.frame(
                    "pi_ast"=c(xvals_mat),
                    "pi_dag"=c(yvals_mat),
                    "utility_ast"=c(z_ast))
    df_dag <- data.frame(
                    "pi_ast"=c(xvals_mat),
                    "pi_dag"=c(yvals_mat),
                    "utility_dag"=c(z_dag))
    
    # Best-response lines data
    df_br_astGivenDag <- data.frame(
      pi_ast_br = br_ast_given_dag,
      pi_dag    = grid_points
    )
    df_br_dagGivenAst <- data.frame(
      pi_ast    = grid_points,
      pi_dag_br = br_dag_given_ast
    )
    
    # For labeling or highlighting intersection:
    intersection_point <- NULL
    showIntersection <- (!is.na(best_pt[1]) && distMin < intersection_dist_threshold)
    if(showIntersection){
      intersection_point <- data.frame(
        pi_ast = best_pt[1],
        pi_dag = best_pt[2]
      )
      if(!silent){
        message(sprintf(
          "Closest intersection ~ (%.3f, %.3f) in dimension d_=%d [dist=%.4f]",
          best_pt[1], best_pt[2], d_, distMin
        ))
      }
    } else {
      if(!silent){
        message("No intersection found under threshold; min distance was ", round(distMin,4))
      }
    }
    
    # Title logic
    if(is.null(title)){
      mainTitle <- sprintf("Best-Response Curves (Dimension d_=%d)", d_)
    } else {
      mainTitle <- title
    }
    
    ##############################################################################
    # 5) Build the two ggplot heatmaps side by side
    #    p_ast => ast’s utility, p_dag => dag’s utility
    #    Overlaid with both best-response curves
    ##############################################################################
    mid_ast <- mean(df_ast$utility_ast, na.rm = TRUE)
    mid_dag <- mean(df_dag$utility_dag, na.rm = TRUE)

    eq_pt <- data.frame(
      pi_ast = as.numeric(strenv$np$array(res$pi_star_ast_vec_jnp[d_-1L])),
      pi_dag = as.numeric(strenv$np$array(res$pi_star_dag_vec_jnp[d_-1L]))
    )
    

    model_br_ast <- lm(pi_ast_br~pi_dag+pi_dag^2, 
                       data = df_br_astGivenDag)
    df_br_astGivenDag$pi_ast_br_hat <- predict(model_br_ast,
                                            newx=data.frame("pi_dag"=seq(0,1,length.out=10)))
    
    p_ast <- ggplot2::ggplot(df_ast, ggplot2::aes(x = pi_ast, y = pi_dag, fill = utility_ast)) +
      ggplot2::geom_tile() +
      ggplot2::scale_fill_gradient2(
        midpoint = mid_ast,
        low = "white", mid = "skyblue", high = "blue"
      ) +
      # best-response lines
      geom_smooth2(
        data = df_br_astGivenDag,
        #aes(x = pi_ast_br, y = pi_dag),
        ggplot2::aes(x = pi_ast_br_hat, y = pi_dag), # to handle
        color = col_ast, linewidth = lwd_ast, inherit.aes = FALSE
      ) +
      geom_smooth2(
        data = df_br_dagGivenAst,
        ggplot2::aes(x = pi_ast, y = pi_dag_br),
        color = col_dag, linewidth = lwd_dag, inherit.aes = FALSE,
      ) +
      # intersection point if found
      {
        if(showIntersection) {
          ggplot2::annotate("point", x = intersection_point$pi_ast, y = intersection_point$pi_dag,
                   shape = point_pch, size = 3, color = "black")
        } else {
          NULL
        }
      } +
      # equilibrium point
      ggplot2::geom_point(
        data = eq_pt,
        ggplot2::aes(x = pi_ast, y = pi_dag),
        shape = 8, color = "green", size = 5, inherit.aes=FALSE
      ) +
      ggplot2::labs(
        title = mainTitle,
        subtitle = " -- ast's Utility",
        x = expression(pi[ast]^{(d)}),
        y = expression(pi[dag]^{(d)}),
        fill = "Utility(ast)"
      ) +
      ggplot2::theme_minimal()
    
    p_dag <- ggplot2::ggplot(df_dag, ggplot2::aes(x = pi_ast, y = pi_dag, fill = utility_dag+0.001)) +
      ggplot2::geom_tile() +
      ggplot2::scale_fill_gradient2(
        midpoint = mid_dag,
        low = "white", mid = "pink", high = "red"
      )  +
      # best response line
      geom_smooth2(
        data = df_br_astGivenDag,
        #aes(x = pi_ast_br, y = pi_dag),
        ggplot2::aes(x = pi_ast_br_hat, y = pi_dag),
        color = col_ast, linewidth = lwd_ast, inherit.aes = FALSE
      ) +
      geom_smooth2(
        data = df_br_dagGivenAst,
        ggplot2::aes(x = pi_ast, y = pi_dag_br),
        color = col_dag, linewidth = lwd_dag, inherit.aes = FALSE
      ) +
      {
        if(showIntersection) {
          ggplot2::annotate("point", x = intersection_point$pi_ast, y = intersection_point$pi_dag,
                   shape = point_pch, size = 3, color = "black")
        } else {
          NULL
        }
      } +
      # equilibrium point
      ggplot2::geom_point(
        data = eq_pt,
        ggplot2::aes(x = pi_ast, y = pi_dag),
        shape = 8, color = "green", size = 5, inherit.aes=FALSE
      ) +
      ggplot2::labs(
        title = mainTitle,
        subtitle = " -- dag's Utility",
        x = expression(pi[ast]^{(d)}),
        y = expression(pi[dag]^{(d)}),
        fill = "Utility(dag)"
      ) +
      ggplot2::theme_minimal()
    
    # Print side-by-side via grid.arrange (optional)
    p_grid <- gridExtra::grid.arrange(p_ast, p_dag, nrow = 1)
    
    # plot(df_br_astGivenDag$pi_ast_br, df_br_astGivenDag$pi_dag) # 
    # plot(df_br_dagGivenAst$pi_ast, df_br_dagGivenAst$pi_dag_br) # 
  }
  
  if(!silent) {
    message("Done plotting best-response curves for dimension d_=", d_)
  }
  
  invisible(list(grid_points = grid_points,
                 br_dag_given_ast = br_dag_given_ast,
                 br_ast_given_dag = br_ast_given_dag,
                 p_ast = p_ast, 
                 p_dag = p_dag, 
                 p_grid = p_grid
                 ))
}
