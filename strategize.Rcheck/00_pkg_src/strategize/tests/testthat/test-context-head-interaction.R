# =============================================================================
# Pairwise Utility Consistency Tests
# =============================================================================

test_that("pairwise logits are utility differences and swap invariant", {
  skip_on_cran()
  skip_if_no_jax()

  strategize:::initialize_jax()
  strenv <- strategize:::strenv

  model_dims <- 4L
  ff_dim <- 4L

  model_info <- list(
    n_factors = 1L,
    factor_index_list = list(strenv$jnp$array(as.integer(c(0L, 1L)))),
    implicit = FALSE,
    model_dims = model_dims,
    model_depth = 1L,
    n_heads = 1L,
    head_dim = model_dims,
    likelihood = "bernoulli",
    resp_cov_mean = NULL,
    n_resp_covariates = 0L,
    resp_party_levels = c("A", "B"),
    cand_party_to_resp_idx = strenv$jnp$array(as.integer(c(0L))),
    params = NULL
  )

  params_base <- list(
    E_choice = strenv$jnp$ones(list(model_dims), dtype = strenv$dtj),
    E_feature_id = strenv$jnp$zeros(list(1L, model_dims), dtype = strenv$dtj),
    E_factor_1 = strenv$jnp$array(
      matrix(1, nrow = 2, ncol = model_dims),
      dtype = strenv$dtj
    ),
    E_party = strenv$jnp$ones(list(1L, model_dims), dtype = strenv$dtj),
    E_rel = strenv$jnp$ones(list(3L, model_dims), dtype = strenv$dtj),
    E_stage = strenv$jnp$zeros(list(2L, 2L, model_dims), dtype = strenv$dtj),
    E_resp_party = strenv$jnp$zeros(list(2L, model_dims), dtype = strenv$dtj),
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
    b_out = strenv$jnp$zeros(list(1L), dtype = strenv$dtj)
  )

  pi_left <- strenv$jnp$array(c(1, 0), dtype = strenv$dtj)
  pi_right <- strenv$jnp$array(c(0, 1), dtype = strenv$dtj)

  logits_lr <- strategize:::neural_predict_pair_soft(
    pi_left = pi_left,
    pi_right = pi_right,
    party_left_idx = 0L,
    party_right_idx = 0L,
    resp_party_idx = 1L,
    model_info = model_info,
    resp_cov_vec = NULL,
    params = params_base,
    return_logits = TRUE
  )

  u_left <- strategize:::neural_candidate_utility_soft(
    pi_left,
    party_idx = 0L,
    resp_party_idx = 1L,
    stage_idx = 1L,
    model_info = model_info,
    resp_cov_vec = NULL,
    params = params_base
  )
  u_right <- strategize:::neural_candidate_utility_soft(
    pi_right,
    party_idx = 0L,
    resp_party_idx = 1L,
    stage_idx = 1L,
    model_info = model_info,
    resp_cov_vec = NULL,
    params = params_base
  )
  expected_logits <- u_left - u_right

  expect_equal(
    as.numeric(strenv$np$array(logits_lr)),
    as.numeric(strenv$np$array(expected_logits)),
    tolerance = 1e-6
  )

  logits_rl <- strategize:::neural_predict_pair_soft(
    pi_left = pi_right,
    pi_right = pi_left,
    party_left_idx = 0L,
    party_right_idx = 0L,
    resp_party_idx = 1L,
    model_info = model_info,
    resp_cov_vec = NULL,
    params = params_base,
    return_logits = TRUE
  )

  expect_equal(
    as.numeric(strenv$np$array(logits_lr)),
    -as.numeric(strenv$np$array(logits_rl)),
    tolerance = 1e-6
  )
})

test_that("pairwise logits include cross-candidate interaction terms when enabled", {
  skip_on_cran()
  skip_if_no_jax()

  strategize:::initialize_jax()
  strenv <- strategize:::strenv

  model_dims <- 4L
  ff_dim <- 4L

  model_info <- list(
    n_factors = 1L,
    factor_index_list = list(strenv$jnp$array(as.integer(c(0L, 1L)))),
    implicit = FALSE,
    model_dims = model_dims,
    model_depth = 1L,
    n_heads = 1L,
    head_dim = model_dims,
    likelihood = "bernoulli",
    resp_cov_mean = NULL,
    n_resp_covariates = 0L,
    resp_party_levels = c("A", "B"),
    cand_party_to_resp_idx = strenv$jnp$array(as.integer(c(0L))),
    cross_candidate_encoder = TRUE,
    params = NULL
  )

  params_base <- list(
    E_choice = strenv$jnp$ones(list(model_dims), dtype = strenv$dtj),
    E_feature_id = strenv$jnp$zeros(list(1L, model_dims), dtype = strenv$dtj),
    E_factor_1 = strenv$jnp$array(
      matrix(1, nrow = 2, ncol = model_dims),
      dtype = strenv$dtj
    ),
    E_party = strenv$jnp$ones(list(1L, model_dims), dtype = strenv$dtj),
    E_rel = strenv$jnp$ones(list(3L, model_dims), dtype = strenv$dtj),
    E_stage = strenv$jnp$zeros(list(2L, 2L, model_dims), dtype = strenv$dtj),
    E_resp_party = strenv$jnp$zeros(list(2L, model_dims), dtype = strenv$dtj),
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
    M_cross = strenv$jnp$ones(list(model_dims, model_dims), dtype = strenv$dtj),
    W_cross_out = strenv$jnp$array(c(0.25), dtype = strenv$dtj)
  )

  pi_left <- strenv$jnp$array(c(1, 0), dtype = strenv$dtj)
  pi_right <- strenv$jnp$array(c(0, 1), dtype = strenv$dtj)

  phi_left <- strategize:::neural_encode_candidate_soft(
    pi_left,
    party_idx = 0L,
    resp_party_idx = 1L,
    stage_idx = 1L,
    model_info = model_info,
    resp_cov_vec = NULL,
    params = params_base,
    matchup_idx = NULL
  )
  phi_right <- strategize:::neural_encode_candidate_soft(
    pi_right,
    party_idx = 0L,
    resp_party_idx = 1L,
    stage_idx = 1L,
    model_info = model_info,
    resp_cov_vec = NULL,
    params = params_base,
    matchup_idx = NULL
  )

  u_left <- strenv$jnp$einsum("nm,mo->no", phi_left, params_base$W_out) + params_base$b_out
  u_right <- strenv$jnp$einsum("nm,mo->no", phi_right, params_base$W_out) + params_base$b_out
  base_logits <- u_left - u_right
  cross_term <- strenv$jnp$einsum("nm,mp,np->n", phi_left, params_base$M_cross, phi_right)
  cross_term <- strenv$jnp$reshape(cross_term, list(-1L, 1L))
  cross_out <- strenv$jnp$reshape(params_base$W_cross_out, list(1L, -1L))
  expected_logits <- base_logits + cross_term * cross_out

  logits_lr <- strategize:::neural_predict_pair_soft(
    pi_left = pi_left,
    pi_right = pi_right,
    party_left_idx = 0L,
    party_right_idx = 0L,
    resp_party_idx = 1L,
    model_info = model_info,
    resp_cov_vec = NULL,
    params = params_base,
    return_logits = TRUE
  )

  cross_numeric <- as.numeric(strenv$np$array(cross_term))
  expect_true(any(abs(cross_numeric) > 1e-8))
  expect_equal(
    as.numeric(strenv$np$array(logits_lr)),
    as.numeric(strenv$np$array(expected_logits)),
    tolerance = 1e-6
  )
})

test_that("pairwise logits ignore cross-candidate interaction terms when disabled", {
  skip_on_cran()
  skip_if_no_jax()

  strategize:::initialize_jax()
  strenv <- strategize:::strenv

  model_dims <- 4L
  ff_dim <- 4L

  model_info <- list(
    n_factors = 1L,
    factor_index_list = list(strenv$jnp$array(as.integer(c(0L, 1L)))),
    implicit = FALSE,
    model_dims = model_dims,
    model_depth = 1L,
    n_heads = 1L,
    head_dim = model_dims,
    likelihood = "bernoulli",
    resp_cov_mean = NULL,
    n_resp_covariates = 0L,
    resp_party_levels = c("A", "B"),
    cand_party_to_resp_idx = strenv$jnp$array(as.integer(c(0L))),
    cross_candidate_encoder = FALSE,
    params = NULL
  )

  params_base <- list(
    E_choice = strenv$jnp$ones(list(model_dims), dtype = strenv$dtj),
    E_feature_id = strenv$jnp$zeros(list(1L, model_dims), dtype = strenv$dtj),
    E_factor_1 = strenv$jnp$array(
      matrix(1, nrow = 2, ncol = model_dims),
      dtype = strenv$dtj
    ),
    E_party = strenv$jnp$ones(list(1L, model_dims), dtype = strenv$dtj),
    E_rel = strenv$jnp$ones(list(3L, model_dims), dtype = strenv$dtj),
    E_stage = strenv$jnp$zeros(list(2L, 2L, model_dims), dtype = strenv$dtj),
    E_resp_party = strenv$jnp$zeros(list(2L, model_dims), dtype = strenv$dtj),
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
    M_cross = strenv$jnp$ones(list(model_dims, model_dims), dtype = strenv$dtj),
    W_cross_out = strenv$jnp$array(c(0.25), dtype = strenv$dtj)
  )

  pi_left <- strenv$jnp$array(c(1, 0), dtype = strenv$dtj)
  pi_right <- strenv$jnp$array(c(0, 1), dtype = strenv$dtj)

  phi_left <- strategize:::neural_encode_candidate_soft(
    pi_left,
    party_idx = 0L,
    resp_party_idx = 1L,
    stage_idx = 1L,
    model_info = model_info,
    resp_cov_vec = NULL,
    params = params_base,
    matchup_idx = NULL
  )
  phi_right <- strategize:::neural_encode_candidate_soft(
    pi_right,
    party_idx = 0L,
    resp_party_idx = 1L,
    stage_idx = 1L,
    model_info = model_info,
    resp_cov_vec = NULL,
    params = params_base,
    matchup_idx = NULL
  )

  u_left <- strenv$jnp$einsum("nm,mo->no", phi_left, params_base$W_out) + params_base$b_out
  u_right <- strenv$jnp$einsum("nm,mo->no", phi_right, params_base$W_out) + params_base$b_out
  base_logits <- u_left - u_right
  cross_term <- strenv$jnp$einsum("nm,mp,np->n", phi_left, params_base$M_cross, phi_right)
  cross_term <- strenv$jnp$reshape(cross_term, list(-1L, 1L))
  cross_out <- strenv$jnp$reshape(params_base$W_cross_out, list(1L, -1L))
  expected_logits <- base_logits + cross_term * cross_out

  logits_lr <- strategize:::neural_predict_pair_soft(
    pi_left = pi_left,
    pi_right = pi_right,
    party_left_idx = 0L,
    party_right_idx = 0L,
    resp_party_idx = 1L,
    model_info = model_info,
    resp_cov_vec = NULL,
    params = params_base,
    return_logits = TRUE
  )

  cross_numeric <- as.numeric(strenv$np$array(cross_term))
  expect_true(any(abs(cross_numeric) > 1e-8))
  expect_equal(
    as.numeric(strenv$np$array(logits_lr)),
    as.numeric(strenv$np$array(base_logits)),
    tolerance = 1e-6
  )
  expect_true(
    any(abs(as.numeric(strenv$np$array(logits_lr)) -
              as.numeric(strenv$np$array(expected_logits))) > 1e-6)
  )
})

test_that("pairwise logits use full cross-candidate encoder when mode is full", {
  skip_on_cran()
  skip_if_no_jax()

  strategize:::initialize_jax()
  strenv <- strategize:::strenv

  model_dims <- 2L
  ff_dim <- 2L

  model_info <- list(
    n_factors = 1L,
    factor_index_list = list(strenv$jnp$array(as.integer(c(0L, 1L)))),
    implicit = FALSE,
    model_dims = model_dims,
    model_depth = 1L,
    n_heads = 1L,
    head_dim = model_dims,
    likelihood = "bernoulli",
    resp_cov_mean = NULL,
    n_resp_covariates = 0L,
    resp_party_levels = c("A", "B"),
    cand_party_to_resp_idx = strenv$jnp$array(as.integer(c(0L))),
    cross_candidate_encoder_mode = "full",
    params = NULL
  )

  params_base <- list(
    E_choice = strenv$jnp$array(c(1, 2), dtype = strenv$dtj),
    E_feature_id = strenv$jnp$zeros(list(1L, model_dims), dtype = strenv$dtj),
    E_factor_1 = strenv$jnp$zeros(list(2L, model_dims), dtype = strenv$dtj),
    E_party = strenv$jnp$zeros(list(1L, model_dims), dtype = strenv$dtj),
    E_rel = strenv$jnp$zeros(list(3L, model_dims), dtype = strenv$dtj),
    E_stage = strenv$jnp$zeros(list(2L, 2L, model_dims), dtype = strenv$dtj),
    E_resp_party = strenv$jnp$zeros(list(2L, model_dims), dtype = strenv$dtj),
    E_sep = strenv$jnp$zeros(list(model_dims), dtype = strenv$dtj),
    E_segment = strenv$jnp$zeros(list(2L, model_dims), dtype = strenv$dtj),
    W_q_l1 = strenv$jnp$zeros(list(model_dims, model_dims), dtype = strenv$dtj),
    W_k_l1 = strenv$jnp$zeros(list(model_dims, model_dims), dtype = strenv$dtj),
    W_v_l1 = strenv$jnp$zeros(list(model_dims, model_dims), dtype = strenv$dtj),
    W_o_l1 = strenv$jnp$zeros(list(model_dims, model_dims), dtype = strenv$dtj),
    RMS_attn_l1 = strenv$jnp$zeros(list(model_dims), dtype = strenv$dtj),
    RMS_ff_l1 = strenv$jnp$zeros(list(model_dims), dtype = strenv$dtj),
    W_ff1_l1 = strenv$jnp$zeros(list(model_dims, ff_dim), dtype = strenv$dtj),
    W_ff2_l1 = strenv$jnp$zeros(list(ff_dim, model_dims), dtype = strenv$dtj),
    W_out = strenv$jnp$ones(list(model_dims, 1L), dtype = strenv$dtj),
    b_out = strenv$jnp$zeros(list(1L), dtype = strenv$dtj),
    RMS_final = NULL
  )

  pi_left <- strenv$jnp$array(c(1, 0), dtype = strenv$dtj)
  pi_right <- strenv$jnp$array(c(0, 1), dtype = strenv$dtj)

  logits_full <- strategize:::neural_predict_pair_soft(
    pi_left = pi_left,
    pi_right = pi_right,
    party_left_idx = 0L,
    party_right_idx = 0L,
    resp_party_idx = 1L,
    model_info = model_info,
    resp_cov_vec = NULL,
    params = params_base,
    return_logits = TRUE
  )

  model_info_none <- model_info
  model_info_none$cross_candidate_encoder_mode <- "none"
  logits_none <- strategize:::neural_predict_pair_soft(
    pi_left = pi_left,
    pi_right = pi_right,
    party_left_idx = 0L,
    party_right_idx = 0L,
    resp_party_idx = 1L,
    model_info = model_info_none,
    resp_cov_vec = NULL,
    params = params_base,
    return_logits = TRUE
  )

  expected <- sum(as.numeric(strenv$np$array(params_base$E_choice)))

  expect_equal(
    as.numeric(strenv$np$array(logits_full)),
    expected,
    tolerance = 1e-6
  )
  expect_equal(
    as.numeric(strenv$np$array(logits_none)),
    0,
    tolerance = 1e-6
  )
  expect_true(
    abs(as.numeric(strenv$np$array(logits_full)) -
          as.numeric(strenv$np$array(logits_none))) > 1e-6
  )
})
