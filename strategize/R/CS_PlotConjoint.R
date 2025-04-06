#' Plot Estimated Probabilities for Hypothetical Scenarios
#'
#' This function creates a grid of base R plots to visualize and compare
#' probabilities (and optionally their confidence intervals) across multiple
#' hypothetical or assignment scenarios. By default, it arranges the plots
#' in a 3xN grid, labeling each row according to the factor or condition
#' being displayed.
#'
#' @param pi_star_list A list of numeric vectors, each corresponding to a set of
#'   hypothetical probabilities to be plotted. These are typically model-based or
#'   derived values.
#' @param pi_star_se_list A list of numeric vectors of the same structure as
#'   \code{pi_star_list}, containing standard errors for each probability.
#'   Used to plot confidence intervals.
#' @param p_list A list of numeric vectors of "assignment" or
#'   baseline probabilities to be overlaid as vertical ticks on each plot
#'   (depending on \code{ticks_type}).
#' @param col.main Character. Color for the main title in each subplot.
#'   Default is \code{"black"}.
#' @param zStar Numeric. Multiplier for the standard error bars (e.g., 1.96
#'   for approximately 95\% confidence intervals). Default is \code{1}.
#' @param xlim Numeric vector of length 2. The x-axis limits for all subplots.
#'   Defaults to \code{c(0, 1)} if not specified.
#' @param ticks_type Character. Controls the type of reference ticks added. 
#'     \code{"assignmentProbs"}: Vertical ticks drawn at the positions from \code{p_list} (default). 
#'     \code{"zero"}: Vertical ticks drawn at 0. 
#'     \code{"none"}: No vertical reference ticks.
#' @param col_vec Optional character vector of colors (one per set of probabilities
#'   in \code{pi_star_list}). If \code{NULL}, uses sequential indexing
#'   for color.
#' @param plot_names Logical. If \code{TRUE} (default), factor/condition labels
#'   will be placed along the y-axis.
#' @param plot_ci Logical. If \code{TRUE} (default), error bars will be drawn
#'   using \code{pi_star_se_list}.
#' @param widths_vec,heights_vec Currently unused. Reserved for future layout
#'   expansions.
#' @param main_title Character. An overall title for the plot. Default is
#'   an empty string.
#' @param margins_vec Currently unused. Reserved for future layout expansions.
#' @param add Logical. If \code{FALSE} (default), a new plot is created. If
#'   \code{TRUE}, points/error bars are added to an existing plot space.
#' @param pch Numeric or character. The plotting symbol. Default is \code{20}.
#' @param factor_name_transformer Function to transform factor names for display.
#'   Should accept and return a character vector. Default is identity function.
#' @param level_name_transformer Function to transform level names for display.
#'   Should accept and return a character vector. Default is identity function.
#'
#' @details
#' \code{strategize.plot} arranges multiple subplots (3 columns by default) in a grid
#' that depends on the number of elements in \code{p_list}. Each subplot
#' will show a factor level or condition on the y-axis, with probabilities along
#' the x-axis. If confidence intervals are provided (\code{pi_star_se_list}), horizontal
#' error bars around each probability point will be displayed. Additionally, vertical
#' reference ticks can be added, showing values from \code{p_list} or zero
#' depending on \code{ticks_type}.
#'
#' @return Invisibly returns \code{NULL}. This function is primarily called for its
#' side effect: producing a multi-panel base R plot.
#'
#' @examples
#' \donttest{
#' # Example usage (assuming appropriate data structures)
#' hypotheticalProbs <- list(est1 = c(0.2, 0.5), est2 = c(0.3, 0.6))
#' SEs <- list(est1 = c(0.05, 0.07), est2 = c(0.06, 0.08))
#' assignmentProbs <- list(factor1 = c(0.25, 0.55))
#' strategize.plot(
#'   pi_star_list = hypotheticalProbs,
#'   pi_star_se_list = SEs,
#'   p_list = assignmentProbs,
#'   col_vec = c("red", "blue"),
#'   main_title = "Example Plot"
#' )
#' }
#' 
#' @export
#' @md
strategize.plot <- function(pi_star_list=NULL, 
                            pi_star_se_list = NULL, 
                            p_list=NULL, 
                            col.main = "black", 
                            zStar = 1, 
                            xlim = NULL, 
                            ticks_type = "assignmentProbs",
                            col_vec = NULL, 
                            plot_names = TRUE, 
                            plot_ci = TRUE, 
                            widths_vec, heights_vec,
                            main_title = "", 
                            margins_vec = NULL, 
                            add = FALSE, 
                            pch = 20,
                            factor_name_transformer = function(x){x},
                            level_name_transformer = function(x){x},
                            open_browser = FALSE
                            ) {
  if(open_browser){browser()}
  p_list_ <- sapply(1:length(p_list), function(ze) {
    names(p_list[[ze]]) <- gsub(names(p_list[[ze]]), 
                                            pattern = names(p_list)[ze], 
                                            replace = "")
    names(p_list[[ze]]) <- gsub(names(p_list[[ze]]), 
                                            pattern = "\\.", 
                                            replace = "")
    p_list[[ze]]
  })
  
  ncols <- 3
  nrows <- ceiling(length(p_list_)/3)
  par(mfrow = c(nrows, ncols))
  
  for(d_ in 1:length(p_list_)) {
    ordering_d <- order(unlist(lapply(strsplit(names(pd_ <- p_list_[[d_]]), 
                                               split = ""), 
                                      function(zer) { zer[1] })),
                        decreasing = TRUE)
    
    # Apply factor name transformation
    prettyFactorNames <- factor_name_transformer(names(p_list)[d_])
    
    # Preliminary calculations
    zStar <- 1; yScale <- 0.2
    k_EST <- length(pi_star_list)
    p_d <- p_list[[d_]][ordering_d]
    ypos_grid <- expand.grid(k = 1:k_EST, l = 1:length(pd_))
    ypos_grid$ypos <- (1.5 * ypos_grid$l + (yScale) * (ypos_grid$k - 1)/1)
    
    # Start plot
    par(mar = c(3.5, 17, 2, 1))
    ylim <- c(0.8, max(ypos_grid$ypos) + 0.20)
    plot(pd_[ordering_d], 1:length(pd_),
         ylim = ylim,
         main = prettyFactorNames,
         cex.main = 2, cex = 0,
         xlim = if(is.null(xlim)) c(0, 1) else xlim,
         yaxt = "n", xlab = "", ylab = "")
    
    for(l_ in 1:length(pd_)) {
      shadowk_ <- 0
      for(k_ in k_EST:1) {
        shadowk_ <- shadowk_ + 1
        pi_kd <- pi_star_list[[k_]][[d_]][ordering_d]
        se_kd <- pi_star_se_list[[k_]][[d_]][ordering_d]
        
        col_ <- if(is.null(col_vec)) k_ else col_vec[k_]
        y_loc <- ypos_grid[ypos_grid$l == l_ & ypos_grid$k == shadowk_, "ypos"]
        
        if(ticks_type == "assignmentProbs") {
          points(p_d[l_], y_loc, pch = "|", col = "gray", cex = 1.5)
        }
        if(ticks_type == "zero") {
          points(0, y_loc, pch = "|", col = "gray", cex = 1.5)
        }
        
        if(plot_ci) {
          points(c(min(xlim[2], pi_kd[l_] + zStar * se_kd[l_]),
                   max(xlim[1], pi_kd[l_] - zStar * se_kd[l_])),
                 c(y_loc, y_loc), lwd = 2, type = "l", col = col_)
        }
        points(pi_kd[l_], y_loc, pch = pch, col = col_, cex = 1.5)
      }
    }
    
    # Apply level name transformation
    my_names <- level_name_transformer(names(p_list_[[d_]][ordering_d]))
    
    if(plot_names && !add) {
      axis(2,
           at = tapply(ypos_grid$ypos, ypos_grid$l, mean),
           labels = my_names, 
           tick = FALSE,
           col = "black",
           lwd = 1,
           las = 2,
           cex.axis = 1.25)
    }
  }
  
  invisible(NULL)
}