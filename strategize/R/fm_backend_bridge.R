#' Backend bridge for preference foundation-model training
#'
#' @description
#' Returns the small set of internal neural/JAX helpers needed by external
#' foundation-model training packages. This is a developer bridge for
#' \pkg{preference.fm}; user workflows should call the public strategize APIs.
#'
#' @return A named list of helper functions and the shared JAX environment.
#' @export
strategize_fm_backend <- function() {
  list(
    `%||%` = `%||%`,
    initialize_jax = initialize_jax,
    strenv = strenv,
    cs_build_names_list = cs_build_names_list,
    cs2step_build_pair_mat = cs2step_build_pair_mat,
    cs2step_validate_pairwise_ids = cs2step_validate_pairwise_ids,
    cs_encode_W_indices = cs_encode_W_indices,
    cs2step_eval_outcome_model_neural = cs2step_eval_outcome_model_neural,
    cs2step_capture_text_embedding_metadata = cs2step_capture_text_embedding_metadata,
    cs2step_restore_text_embedding_metadata = cs2step_restore_text_embedding_metadata,
    cs2step_neural_pack_model_info = cs2step_neural_pack_model_info,
    cs2step_neural_to_r_array = cs2step_neural_to_r_array,
    cs2step_unpack_predictor = cs2step_unpack_predictor,
    neural_params_from_theta = neural_params_from_theta,
    neural_resolve_max_factor_tokens = neural_resolve_max_factor_tokens,
    neural_validate_factor_token_budget = neural_validate_factor_token_budget,
    neural_factor_order_from_names = neural_factor_order_from_names,
    neural_resolve_max_covariate_tokens = neural_resolve_max_covariate_tokens,
    neural_validate_covariate_token_budget = neural_validate_covariate_token_budget,
    neural_covariate_order_from_names = neural_covariate_order_from_names,
    neural_resolve_shared_projection_value_encoder = neural_resolve_shared_projection_value_encoder,
    neural_resolve_schema_dropout = neural_resolve_schema_dropout
  )
}
