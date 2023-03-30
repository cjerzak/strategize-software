#' generate_ExactSol
#'
#' Implements the organizational record linkage algorithms of Jerzak and Libgober (2021).
#'
#' @usage
#'
#' generate_ExactSol(x, y, by ...)
#'
#' @param x,y data frames to be merged
#'
#' @return `z` The merged data frame.
#' @export
#'
#' @details `LinkOrgs` automatically processes the name text for each dataset (specified by `by`, `by.x`, and/or `by.y`. Users may specify the following options:
#'
#' - Set `DistanceMeasure` to control algorithm for computing pairwise string distances. Options include "`osa`", "`jaccard`", "`jw`". See `?stringdist::stringdist` for all options. (Default is "`jaccard`")
#'
#' @examples
#'
#' #Create synthetic data
#' x_orgnames <- c("apple","oracle","enron inc.","mcdonalds corporation")
#' y_orgnames <- c("apple corp","oracle inc","enron","mcdonalds co")
#' x <- data.frame("orgnames_x"=x_orgnames)
#' y <- data.frame("orgnames_y"=y_orgnames)
#'
#' # Perform merge
#' linkedOrgs <- LinkOrgs(x = x,
#'                        y = y,
#'                        by.x = "orgnames_x",
#'                        by.y = "orgnames_y",
#'                        MaxDist = 0.6)
#'
#' print( linkedOrgs )
#'
#' @export
#'
#' @md
#'

generate_ExactSol <- function(){

  # ParameterizationType == "Implicit" solution
  if(ParameterizationType == "Implicit"){
  Neg4lambda_diag <-tf$constant( rep(-4 * lambda,times=n_main_params))
  Neg4lambda_update <- tf$constant(as.matrix(-4*lambda),dttf)
  Neg2lambda_update <- tf$constant(as.matrix(-2*lambda),dttf)
  Const_4_lambda_pl <- tf$constant(as.matrix( 4*lambda*p_vec_use),dttf)
  Const_2_lambda_plprime <- tf$constant(as.matrix( 2*lambda*p_vec_sum_prime_use),dttf)

  generate_ExactSolImplicit <- function(){
    main_coef <- tf$gather(EST_COEFFICIENTS_tf,indices = main_indices_i0,axis=0L)
    inter_coef <- tf$gather(EST_COEFFICIENTS_tf,indices = inter_indices_i0,axis=0L)
    b_vec <- tf$negative( main_coef ) - Const_4_lambda_pl - Const_2_lambda_plprime

    C_mat <- sapply(1:n_main_params,function(main_comp){
      # initialize to 0
      row_ <- tf$zeros(list(n_main_params,1L))

      # update diagonal component
      row_ <- tf$tensor_scatter_nd_update(row_,
                                          updates = Neg4lambda_update,
                                          indices = n2int(as.matrix(ai(main_comp-1L))))

      # update off-diagonal component (same d)
      same_d_diff_l <- n2int(as.matrix(ai(setdiff(which(main_info$d_adj == main_info[main_comp,]$d_adj),main_comp)-1L)))
      row_ <- tf$tensor_scatter_nd_update(row_,
                                          updates = -2*lambda*tf$ones(list(nrow(same_d_diff_l),1L)),
                                          indices = same_d_diff_l)

      # update off-diagonal component (different d)
      #interaction_info_red <- interaction_info[interaction_info$dl_index %in% main_comp | interaction_info$dplp_index %in% main_comp,]
      interaction_info_red <- interaction_info[
        (ind1<-(interaction_info$d_adj %in% main_info[main_comp,]$d_adj &
                  interaction_info$l %in% main_info[main_comp,]$l)) |
          (ind2<-(interaction_info$dp_adj %in% main_info[main_comp,]$d_adj &
                    interaction_info$lp %in% main_info[main_comp,]$l ) ),]
      id_d <- apply(interaction_info_red[,c("d_adj","l")],1,function(zer){paste(zer,collapse="_")})
      id_dp <- apply(interaction_info_red[,c("dp_adj","lp")],1,function(zer){paste(zer,collapse="_")})
      id_ <- ifelse(!(interaction_info_red$d_adj %in% main_info[main_comp,]$d_adj),
                    yes = id_d, no = id_dp)
      id_main <- apply(main_info[,c("d_adj","l")],1,function(zer){paste(zer,collapse="_")})
      which_inter <- which(ind1|ind2)
      inter_into_main <- sapply(id_,function(zer){which(id_main %in% zer)})

      #interaction_info_ordering <- interaction_info_red$dl_index * (ind1) + interaction_info_red$dplp_index * (ind2)
      if(nrow(interaction_info_red)>0){
        inter_coef_ <- tf$gather(inter_coef,
                                 indices = n2int(ai(which_inter-1L)),
                                 axis = 0L)
        if(length(which_inter) == 1){ inter_coef_ <- tf$expand_dims(inter_coef_,0L) }
        row_ <- tf$tensor_scatter_nd_update(row_,
                                            updates = inter_coef_,
                                            indices = n2int(as.matrix(ai(inter_into_main-1L))))
      }
      return( row_ )
    })
    C_mat <- tf$concat(C_mat,1L)

    return(  pi_star <- tf$matmul(tf$linalg$inv(C_mat),b_vec)  )
  }
  }

  # ParameterizationType == "Full" solution
  if(ParameterizationType == "Full"){
    Const_2_lambda_pl <- tf$constant(as.matrix( 2*lambda*p_vec_use),dttf)
    Const_2_lambda_plprime <- tf$constant(as.matrix( 2*lambda*p_vec_sum_prime_use),dttf)
    Neg4lambda_update <- tf$constant(as.matrix(-4*lambda),dttf)
    Neg2lambda_update <- tf$constant(as.matrix(-2*lambda),dttf)
    getPiStar_exact <- function(){
      main_coef <- tf$gather(EST_COEFFICIENTS_tf,indices = main_indices_i0,axis=0L)
      inter_coef <- tf$gather(EST_COEFFICIENTS_tf,indices = inter_indices_i0,axis=0L)
      b_vec <- tf$negative( main_coef ) - Const_2_lambda_pl

      C_mat <- sapply(1:n_main_params,function(main_comp){
        # initialize to 0
        row_ <- tf$zeros(list(n_main_params,1L))

        # update diagonal component
        row_ <- tf$tensor_scatter_nd_update(row_,
                                            updates = Neg2lambda_update,
                                            indices = n2int(as.matrix(ai(main_comp-1L))))

        # update off-diagonal component (same d)
        same_d_diff_l <- n2int(as.matrix(ai(setdiff(which(main_info$d_adj == main_info[main_comp,]$d_adj),main_comp)-1L)))
        row_ <- tf$tensor_scatter_nd_update(row_,
                                            #updates = -2*lambda*tf$ones(list(nrow(same_d_diff_l),1L)),
                                            updates = tf$zeros(list(nrow(same_d_diff_l),1L)),
                                            indices = same_d_diff_l)

        # update off-diagonal component (different d)
        #interaction_info_red <- interaction_info[interaction_info$dl_index %in% main_comp | interaction_info$dplp_index %in% main_comp,]
        interaction_info_red <- interaction_info[
          (ind1<-(interaction_info$d_adj %in% main_info[main_comp,]$d_adj &
                    interaction_info$l %in% main_info[main_comp,]$l)) |
            (ind2<-(interaction_info$dp_adj %in% main_info[main_comp,]$d_adj &
                      interaction_info$lp %in% main_info[main_comp,]$l ) ),]
        id_d <- apply(interaction_info_red[,c("d_adj","l")],1,function(zer){paste(zer,collapse="_")})
        id_dp <- apply(interaction_info_red[,c("dp_adj","lp")],1,function(zer){paste(zer,collapse="_")})
        id_ <- ifelse(!(interaction_info_red$d_adj %in% main_info[main_comp,]$d_adj),
                      yes = id_d, no = id_dp)
        id_main <- apply(main_info[,c("d_adj","l")],1,function(zer){paste(zer,collapse="_")})
        which_inter <- which(ind1|ind2)
        inter_into_main <- sapply(id_,function(zer){which(id_main %in% zer)})

        if(nrow(interaction_info_red)>0){
          inter_coef_ <- tf$gather(inter_coef,
                                   indices = n2int(ai(which_inter-1L)),
                                   axis = 0L)
          if(length(which_inter) == 1){ inter_coef_ <- tf$expand_dims(inter_coef_,0L) }
          row_ <- tf$tensor_scatter_nd_update(row_,
                                              updates = inter_coef_,
                                              indices = n2int(as.matrix(ai(inter_into_main-1L))))
        }
        return( row_ )
      })
      C_mat <- tf$concat(C_mat,1L)

      return(  pi_star <- tf$matmul(tf$linalg$inv(C_mat),b_vec)  )
    }
  }
}
