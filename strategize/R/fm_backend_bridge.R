#' Backend bridge for preference foundation-model training
#'
#' @description
#' Returns the set of internal neural/JAX helpers needed by external
#' foundation-model training packages, plus an explicit API version and
#' capability list so consumers can verify compatibility instead of probing
#' the installed source. This is a developer bridge for \pkg{preference.fm};
#' user workflows should call the public strategize APIs.
#'
#' @details
#' \code{api_version} increments whenever the bridge contract changes
#' incompatibly. \code{capabilities} advertises backend properties consumers
#' previously had to infer by source-sniffing or \code{formals()} probing:
#' \describe{
#'   \item{\code{universal_likelihood_masked}}{The universal mixed-outcome
#'     likelihood evaluates each family on sanitized labels (no masked-mixture
#'     NaN contamination).}
#'   \item{\code{compact_training}}{\code{cs2step_eval_outcome_model_neural}
#'     accepts \code{W_idx_compact} / \code{X_compact} /
#'     \code{X_present_compact} as named formals.}
#'   \item{\code{checkpoint_restore_helpers}}{The Orbax checkpoint restore
#'     primitives are exported through this bridge.}
#'   \item{\code{strict_muon_optimizer_v1}}{Muon training fails on incompatible
#'     dependencies or guides and verifies the actual momentum partition.}
#'   \item{\code{training_row_validity_guard}}{Universal training validates
#'     outcome rows against their declared family at ingestion.}
#'   \item{\code{uncertainty_scope_none_v1}}{\code{uncertainty_scope = "none"}
#'     fits every site as a point estimate (pure maximum likelihood: no priors,
#'     no KL term, no variational guide); \code{theta_var} is zero.}
#'   \item{\code{residual_init_policy_v1}}{\code{neural_mcmc_control} accepts
#'     \code{residual_gate_init} (initial value of the trunk residual gates) and
#'     \code{output_proj_init = "depth_scaled"} (residual-writing projections
#'     start at \eqn{1/\sqrt{2 \cdot depth}} of their regular init sd).}
#'   \item{\code{svi_weight_decay_v1}}{\code{neural_mcmc_control} accepts
#'     \code{svi_weight_decay} (decoupled decay on Muon hidden matrices, applied
#'     as \eqn{lr(t) \cdot decay \cdot p}) and
#'     \code{svi_weight_decay_batch_scaling} (also apply the LR batch factor).}
#' }
#'
#' @return A named list of helper functions, the shared JAX environment,
#'   \code{api_version} (integer), and \code{capabilities} (character).
#' @export
strategize_fm_backend <- function() {
  list(
    api_version = 2L,
    capabilities = c(
      "universal_likelihood_masked",
      "compact_training",
      "checkpoint_restore_helpers",
      "training_row_validity_guard",
      "data_parallel_svi_v1", "full_svi_state_resume", "transformer_moe_v1", "mla_dsa_v1",
      "looped_transformer_v1", "looped_transformer_v2", "strict_muon_optimizer_v1",
      "level_aux_text_v1", "uncertainty_scope_none_v1", "residual_init_policy_v1",
      "svi_weight_decay_v1"
    ),
    `%||%` = `%||%`,
    initialize_jax = initialize_jax,
    data_parallel_enabled = strategize_dp_enabled,
    data_parallel_primary = strategize_dp_primary_rank,
    data_parallel_primary_call = strategize_dp_primary,
    data_parallel_shutdown = strategize_dp_shutdown,
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
    cs2step_neural_extract_internal = cs2step_neural_extract_internal,
    cs_foundation_unpack_group = cs_foundation_unpack_group,
    cs_foundation_is_checkpoint_dir = cs_foundation_is_checkpoint_dir,
    cs_foundation_build_abstract_tree = cs_foundation_build_abstract_tree,
    cs_foundation_orbax_load_tree = cs_foundation_orbax_load_tree,
    cs_foundation_py_get_item = cs_foundation_py_get_item,
    cs_foundation_restore_text_matrix = cs_foundation_restore_text_matrix,
    cs_foundation_restore_theta_mean = cs_foundation_restore_theta_mean,
    neural_params_from_theta = neural_params_from_theta,
    neural_resolve_transformer_moe = neural_resolve_transformer_moe,
    neural_resolve_latent_attention = neural_resolve_latent_attention,
    neural_resolve_transformer_loop = neural_resolve_transformer_loop,
    neural_recurrent_activation_diagnostics = neural_recurrent_activation_diagnostics,
    neural_recurrent_prediction_diagnostics = neural_recurrent_prediction_diagnostics,
    neural_moe_control_from_info = neural_moe_control_from_info,
    neural_unstack_standard_transformer_layers = neural_unstack_standard_transformer_layers,
    neural_resolve_max_factor_tokens = neural_resolve_max_factor_tokens,
    neural_resolve_text_embedding_normalization = neural_resolve_text_embedding_normalization,
    neural_validate_text_embedding_normalizer = neural_validate_text_embedding_normalizer,
    neural_resolve_struct_feature_encoding = neural_resolve_struct_feature_encoding,
    neural_resolve_level_text_encoding = neural_resolve_level_text_encoding,
    neural_level_distinctive_text = neural_level_distinctive_text,
    neural_struct_cardinality_feature = neural_struct_cardinality_feature,
    neural_struct_raw_value_feature = neural_struct_raw_value_feature,
    neural_validate_factor_token_budget = neural_validate_factor_token_budget,
    neural_factor_order_from_names = neural_factor_order_from_names,
    neural_resolve_max_covariate_tokens = neural_resolve_max_covariate_tokens,
    neural_validate_covariate_token_budget = neural_validate_covariate_token_budget,
    neural_covariate_order_from_names = neural_covariate_order_from_names,
    neural_resolve_shared_projection_value_encoder = neural_resolve_shared_projection_value_encoder,
    neural_resolve_schema_dropout = neural_resolve_schema_dropout
  )
}
