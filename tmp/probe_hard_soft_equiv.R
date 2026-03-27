library(pkgload)
pkgload::load_all('strategize', quiet = TRUE)
strategize:::initialize_jax()
strenv <- strategize:::strenv

model_dims <- 4L
ff_dim <- 4L
model_info <- list(
  n_factors = 1L,
  factor_index_list = list(strenv$jnp$array(as.integer(c(0L, 1L)))),
  factor_levels = as.integer(2L),
  implicit = FALSE,
  model_dims = model_dims,
  model_depth = 1L,
  n_heads = 1L,
  head_dim = model_dims,
  likelihood = 'bernoulli',
  resp_cov_mean = NULL,
  n_resp_covariates = 0L,
  cand_party_to_resp_idx = strenv$jnp$array(as.integer(0L)),
  n_candidate_tokens = 3L,
  cross_candidate_encoder_mode = 'attn',
  params = NULL
)
params <- list(
  E_choice = strenv$jnp$zeros(list(model_dims), dtype = strenv$dtj),
  E_feature_id = strenv$jnp$zeros(list(1L, model_dims), dtype = strenv$dtj),
  E_factor_1 = strenv$jnp$array(matrix(c(1,0,0,0, 0,1,0,0), nrow = 2, byrow = TRUE), dtype = strenv$dtj),
  E_party = strenv$jnp$zeros(list(1L, model_dims), dtype = strenv$dtj),
  E_rel = strenv$jnp$zeros(list(3L, model_dims), dtype = strenv$dtj),
  E_stage = strenv$jnp$zeros(list(1L, 2L, model_dims), dtype = strenv$dtj),
  E_resp_party = strenv$jnp$zeros(list(1L, model_dims), dtype = strenv$dtj),
  W_q_l1 = strenv$jnp$eye(model_dims, dtype = strenv$dtj),
  W_k_l1 = strenv$jnp$eye(model_dims, dtype = strenv$dtj),
  W_v_l1 = strenv$jnp$eye(model_dims, dtype = strenv$dtj),
  W_o_l1 = strenv$jnp$eye(model_dims, dtype = strenv$dtj),
  RMS_attn_l1 = strenv$jnp$ones(list(model_dims), dtype = strenv$dtj),
  RMS_ff_l1 = strenv$jnp$ones(list(model_dims), dtype = strenv$dtj),
  RMS_final = strenv$jnp$ones(list(model_dims), dtype = strenv$dtj),
  W_ff1_l1 = strenv$jnp$zeros(list(model_dims, ff_dim), dtype = strenv$dtj),
  W_ff2_l1 = strenv$jnp$zeros(list(ff_dim, model_dims), dtype = strenv$dtj),
  W_out = strenv$jnp$ones(list(model_dims, 1L), dtype = strenv$dtj),
  b_out = strenv$jnp$zeros(list(1L), dtype = strenv$dtj),
  alpha_cross = strenv$jnp$array(1.0, dtype = strenv$dtj),
  W_q_cross = strenv$jnp$eye(model_dims, dtype = strenv$dtj),
  W_k_cross = strenv$jnp$eye(model_dims, dtype = strenv$dtj),
  W_v_cross = strenv$jnp$eye(model_dims, dtype = strenv$dtj),
  W_o_cross = strenv$jnp$eye(model_dims, dtype = strenv$dtj),
  RMS_cross = strenv$jnp$ones(list(model_dims), dtype = strenv$dtj),
  RMS_q_cross = strenv$jnp$ones(list(model_dims), dtype = strenv$dtj),
  RMS_k_cross = strenv$jnp$ones(list(model_dims), dtype = strenv$dtj)
)
model_info$params <- params

pi_left <- strenv$jnp$array(c(1,0), dtype = strenv$dtj)
pi_right <- strenv$jnp$array(c(0,1), dtype = strenv$dtj)
soft <- strategize:::neural_predict_pair_soft(
  pi_left = pi_left,
  pi_right = pi_right,
  party_left_idx = 0L,
  party_right_idx = 0L,
  resp_party_idx = 0L,
  model_info = model_info,
  params = params,
  return_logits = FALSE
)
prep <- list(
  pairwise = TRUE,
  X_left = strenv$jnp$array(matrix(as.integer(0L), nrow = 1L, ncol = 1L))$astype(strenv$jnp$int32),
  X_right = strenv$jnp$array(matrix(as.integer(1L), nrow = 1L, ncol = 1L))$astype(strenv$jnp$int32),
  party_left = strenv$jnp$array(as.integer(0L))$astype(strenv$jnp$int32),
  party_right = strenv$jnp$array(as.integer(0L))$astype(strenv$jnp$int32),
  resp_party = strenv$jnp$array(as.integer(0L))$astype(strenv$jnp$int32),
  resp_cov = strenv$jnp$array(matrix(numeric(0), nrow = 1L, ncol = 0L))$astype(strenv$dtj)
)
hard <- strategize:::cs2step_neural_predict_pair_prepared(params, model_info, prep, return_logits = FALSE)
cat(sprintf('soft=%0.8f\nhard=%0.8f\ndiff=%0.8f\n',
            as.numeric(strenv$np$array(soft)),
            as.numeric(strenv$np$array(hard)),
            as.numeric(strenv$np$array(soft - hard))))
