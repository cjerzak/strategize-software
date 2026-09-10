test_that("transformer MoE defaults resolve by inference method and depth", {
  resolve <- strategize:::neural_resolve_transformer_moe
  cfg <- resolve(model_dims = 16L, model_depth = 2L)
  expect_identical(cfg$transformer_ffn, "moe")
  expect_identical(cfg$transformer_moe$moe_d_ff, 16L)
  expect_identical(cfg$transformer_moe$first_k_dense, 1L)
  expect_identical(cfg$transformer_moe$n_routed_experts, 8L)
  expect_identical(resolve(model_depth = 1L)$transformer_moe$first_k_dense, 0L)
  expect_identical(resolve(use_svi = FALSE)$transformer_ffn, "swiglu")
  expect_null(resolve(list(transformer_ffn = "swiglu"))$transformer_moe)
  expect_error(resolve(list(transformer_ffn = "moe"), use_svi = FALSE), "requires SVI")
  expect_error(resolve(list(transformer_moe = list(n_experts_per_tok = 9))), "top-k")
  expect_error(resolve(list(transformer_moe = list(n_routed_experts = 2.5))), "integer")
  expect_error(resolve(list(transformer_moe = list(capacity_factor = NA_real_))), "finite")
  expect_error(resolve(list(transformer_moe = list(misspelled = 1))), "Unknown")
})

test_that("saved MoE models require complete matching architecture and routing state", {
  validate <- strategize:::neural_validate_saved_transformer_moe
  cfg <- strategize:::neural_resolve_transformer_moe(model_dims = 8L)$transformer_moe
  info <- list(model_dims = 8L, model_depth = 2L, transformer_ffn = "moe",
               transformer_moe = cfg, transformer_moe_router_bias = matrix(0, 1L, 8L),
               param_names = "W_moe_router_layers")
  expect_invisible(validate(info))
  expect_invisible(validate(list(model_dims = 8L, model_depth = 2L)))
  expect_error(validate(within(info, rm(transformer_ffn))), "explicit")
  expect_error(validate(within(info, rm(transformer_moe))), "require architecture")
  expect_error(validate(within(info, rm(transformer_moe_router_bias))), "frozen")
  expect_error(validate(modifyList(info, list(transformer_moe_router_bias = matrix(0, 2L, 8L)))), "match")
  expect_error(validate(modifyList(info, list(transformer_moe_router_bias = matrix(NA_real_, 1L, 8L)))), "finite")
})

test_that("mixed dense and MoE stacks preserve unrolled values and theta schema", {
  skip_on_cran()
  skip_if_no_jax()
  strategize:::initialize_jax()
  jnp <- strategize:::strenv$jnp
  cfg <- strategize:::neural_resolve_transformer_moe(
    list(transformer_moe = list(n_routed_experts = 4L)), 4L, 2L)$transformer_moe
  info <- strategize:::neural_make_transformer_model_info(2L, 4L, 1L, 4L,
                                                         attention_backend = "xla")
  info$transformer_ffn <- "moe"
  info$transformer_moe <- cfg
  info$transformer_moe_router_bias <- matrix(c(2, 1, -1, -2), 1L)
  params <- list(RMS_final = jnp$ones(4L))
  for (layer in 1:2) {
    for (name in c("q", "k", "v", "o")) params[[paste0("W_", name, "_l", layer)]] <- 0.2 * jnp$eye(4L)
    for (name in c("attn", "ff", "q", "k")) params[[paste0("RMS_", name, "_l", layer)]] <- jnp$ones(4L)
    for (name in c("attn", "ff")) params[[paste0("alpha_", name, "_l", layer)]] <- jnp$array(0.1)
  }
  params$W_ff1_l1 <- jnp$ones(c(4L, 30L)) * .1
  params$W_ff2_l1 <- jnp$ones(c(15L, 4L)) * .1
  params$W_moe_router_l2 <- jnp$eye(4L)
  params$W_moe_expert1_l2 <- jnp$ones(c(4L, 4L, 8L)) * .1
  params$W_moe_expert2_l2 <- jnp$ones(c(4L, 4L, 4L)) * .1
  params$W_moe_shared1_l2 <- jnp$ones(c(4L, 8L)) * .1
  params$W_moe_shared2_l2 <- jnp$ones(c(4L, 4L)) * .1
  stacked <- strategize:::neural_stack_standard_transformer_layers(params, 2L, TRUE)
  expect_true(strategize:::neural_has_stacked_standard_transformer(stacked))
  unstacked <- strategize:::neural_unstack_standard_transformer_layers(stacked, 2L)
  expect_setequal(names(unstacked), names(params))
  expect_false("W_ff1_l2" %in% names(unstacked))
  expect_false("W_moe_router_l1" %in% names(unstacked))
  x <- jnp$array(array(seq_len(24) / 24, c(2L, 3L, 4L)))
  mask <- jnp$array(matrix(c(1, 1, 1, 1, 0, 0), 2L))
  a <- strategize:::neural_run_transformer(x, info, params, mask)
  b <- strategize:::neural_run_transformer(x, info, stacked, mask)
  expect_equal(strategize:::cs2step_neural_to_r_array(a), strategize:::cs2step_neural_to_r_array(b), tolerance = 1e-5)
  schema <- strategize:::neural_build_param_schema(stacked, 1L, 2L)
  expect_true("W_moe_expert1_layers" %in% schema$param_names)
  expect_false("transformer_moe_router_bias" %in% schema$param_names)
  info <- modifyList(info, schema)
  theta <- strategize:::neural_flatten_params(stacked, schema)
  rebuilt <- strategize:::neural_params_from_theta(theta, info)
  expect_equal(strategize:::cs2step_neural_to_r_array(rebuilt$W_moe_router_layers),
               strategize:::cs2step_neural_to_r_array(stacked$W_moe_router_layers))
  expect_equal(strategize:::cs2step_neural_to_r_array(rebuilt$transformer_moe_router_bias), info$transformer_moe_router_bias)
  expect_error(strategize:::neural_params_from_theta(theta, within(info, rm(transformer_moe_router_bias))), "missing frozen")
})

test_that("Muon treats expert axes as independent matrices", {
  expect_true(strategize:::neural_muon_targets_matrix_weight("W_moe_expert1_l2", 3L))
  expect_true(strategize:::neural_muon_targets_matrix_weight("W_moe_shared2_l2_auto_loc", 2L))
  expect_false(strategize:::neural_muon_targets_matrix_weight("W_moe_router_l2", 2L))
  expect_false(strategize:::neural_muon_targets_matrix_weight("W_moe_expert1_l2_auto_scale", 3L))
  skip_on_cran()
  skip_if_no_jax()
  strategize:::initialize_jax()
  callable <- strategize:::neural_get_muon_dimension_numbers_callable(force_refresh = TRUE)
  skip_if(is.null(callable), "Optax lacks Muon dimension specifications")
  result <- callable(list(W_moe_expert1_l2 = strategize:::strenv$jnp$ones(c(4L, 4L, 8L))))
  expect_equal(reticulate::py_to_r(result$W_moe_expert1_l2$reduction_axis), 1L)
  expect_equal(reticulate::py_to_r(result$W_moe_expert1_l2$output_axis), 2L)
})
