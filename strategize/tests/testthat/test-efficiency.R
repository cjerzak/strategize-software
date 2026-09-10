test_that("single sequence packing preserves valid tokens, CLS, and attention gradients", {
  skip_on_cran()
  skip_if_no_jax()
  strategize:::initialize_jax()
  env <- strategize:::strenv
  jnp <- env$jnp
  info <- list(n_factors = 2L, default_factor_order = 0:1,
    has_covariate_fused_tokens = TRUE, n_resp_covariates = 2L,
    default_covariate_order = 0:1, has_candidate_group_context = FALSE,
    has_respondent_group_context = FALSE, has_relation_token_context = FALSE)
  choice <- jnp$array(array(c(.1, .2, .3, .4), c(1L, 1L, 4L)))
  cm <- jnp$ones(list(1L, 1L))
  context <- jnp$array(array(seq_len(32) / 50, c(1L, 8L, 4L)))
  candidate <- context * 2
  xm <- jnp$array(matrix(c(0, 1, 0, 0, 1, 0, 0, 0), 1L))
  am <- jnp$array(matrix(c(1, 0, 0, 1, 0, 0, 0, 0), 1L))
  pack <- function(x, a) {
    # Builders remove schema padding before assembling the sequence. Keep the
    # attention equivalence check sensitive to masks and nonadjacent tokens.
    ctx <- strategize:::neural_pack_token_block(x, xm, trim_tokens = 2L)
    cand <- strategize:::neural_pack_token_block(a, am, trim_tokens = 2L)
    strategize:::neural_pack_single_sequence(choice, cm,
      ctx$tokens, ctx$mask, cand$tokens, cand$mask, info)
  }
  packed <- pack(context, candidate)
  array_r <- function(x) as.array(env$np$array(x))
  expect_equal(as.integer(packed$tokens$shape[[2]]), 5L)
  expect_equal(array_r(packed$tokens)[, 1, ], array_r(choice)[, 1, ])
  expect_equal(array_r(packed$tokens)[, 2:3, ], array_r(context)[, c(2, 5), ])
  expect_equal(array_r(packed$tokens)[, 4:5, ], array_r(candidate)[, c(1, 4), ])
  expect_true(all(array_r(packed$mask) == 1))
  objective <- function(x, compact) {
    seq <- if (compact) pack(x, candidate) else list(
      tokens = jnp$concatenate(list(choice, x, candidate), axis = 1L),
      mask = jnp$concatenate(list(cm, xm, am), axis = 1L))
    qkv <- jnp$expand_dims(seq$tokens, axis = 2L)
    mask <- jnp$reshape(seq$mask > 0, list(1L, 1L, 1L, seq$mask$shape[[2]]))
    out <- env$jax$nn$dot_product_attention(qkv, qkv, qkv, mask = mask, implementation = "xla")
    jnp$sum(jnp$take(out, 0L, axis = 1L)^2)
  }
  dense <- function(x) objective(x, FALSE)
  compact <- function(x) objective(x, TRUE)
  expect_equal(array_r(dense(context)), array_r(compact(context)), tolerance = 1e-6)
  expect_equal(array_r(env$jax$grad(dense)(context)),
    array_r(env$jax$grad(compact)(context)), tolerance = 1e-6)
})

test_that("shared context preserves auxiliary-penalty weights and derivatives", {
  skip_on_cran()
  skip_if_no_jax()
  strategize:::initialize_jax()
  env <- strategize:::strenv
  old_active <- env$moe_aux_active
  old_pending <- env$moe_aux_pending
  on.exit({env$moe_aux_active <- old_active; env$moe_aux_pending <- old_pending}, add = TRUE)
  env$moe_aux_active <- TRUE
  # Model three distinct context calls plus one reuse. Without replaying the
  # saved term, the mean regularizer assigns the wrong weight to each context.
  testthat::local_mocked_bindings(add_context_tokens = function(..., params) {
    env$moe_aux_pending <- c(env$moe_aux_pending, list(env$jnp$sum(params$gate^2)))
    list(tokens = env$jnp$reshape(params$gate, list(1L, 1L, 2L)),
      mask = env$jnp$ones(list(1L, 1L)))
  }, .package = "strategize")
  loss <- function(w, reuse) {
    env$moe_aux_pending <- list()
    build <- function(gate) strategize:::neural_build_context_tokens_batch(list(), 0L,
      params = list(gate = gate), return_mask = TRUE)
    a <- build(w)
    build(w * 2)
    build(w * 3)
    if (reuse) {
      a <- strategize:::neural_take_token_info(a, env$jnp$arange(1L))
      strategize:::neural_reuse_context_info(a)
    } else build(w)
    Reduce(`+`, env$moe_aux_pending) / length(env$moe_aux_pending)
  }
  shared <- function(w) loss(w, TRUE)
  separate <- function(w) loss(w, FALSE)
  w <- env$jnp$array(c(.2, .4))
  numeric_r <- function(a) as.numeric(env$np$array(a))
  expect_equal(numeric_r(shared(w)), numeric_r(separate(w)), tolerance = 1e-6)
  expect_equal(numeric_r(env$jax$grad(shared)(w)), numeric_r(env$jax$grad(separate)(w)), tolerance = 1e-6)
  expect_length(env$moe_aux_pending, 4L)
})


test_that("schema trimming follows actual lookup orders and retains explicit overrides", {
  skip_on_cran()
  skip_if_no_jax()
  strategize:::initialize_jax()
  env <- strategize:::strenv
  jnp <- env$jnp
  tokens <- jnp$array(array(seq_len(32), c(1L, 8L, 4L)))
  mask <- jnp$array(matrix(c(1, 1, 1, 0, 0, 0, 0, 0), 1L))
  orders <- list(0:1, 2:4)
  trim <- function(idx = NULL, explicit = NULL) strategize:::neural_trim_schema_tokens(
    tokens, mask, order_list = orders, default_order = 0:6,
    experiment_idx = idx, explicit_order = explicit)
  expect_equal(as.integer(trim(0L)$tokens$shape[[2]]), 3L)
  expect_equal(as.integer(trim()$tokens$shape[[2]]), 7L)
  expect_equal(as.integer(trim(0L, jnp$array(matrix(c(0:6, 1L), 1L)))$tokens$shape[[2]]), 8L)
  expect_equal(as.array(env$np$array(trim(0L)$mask)), matrix(1, 1L, 3L))

  # Runtime metadata intentionally lacks saved-bundle has_* fields. Packing
  # must retain every covariate and auxiliary token supplied by the builder.
  info <- strategize:::neural_make_runtime_token_model_info(model_dims = 4L,
    covariate_names = letters[1:7], default_covariate_order = 0:6)
  choice <- jnp$zeros(list(1L, 1L, 4L))
  cm <- jnp$ones(list(1L, 1L))
  context <- trim(0L)
  candidate <- trim(0L)
  simple <- strategize:::neural_pack_single_sequence(choice, cm,
    context$tokens, context$mask, candidate$tokens, candidate$mask, info)
  mixed <- strategize:::neural_pack_candidate_sequence(choice, cm,
    context$tokens, context$mask, candidate$tokens, candidate$mask, info)
  cross <- strategize:::neural_pack_full_cross_sequence(choice, cm, choice, cm,
    candidate$tokens, candidate$mask, candidate$tokens, candidate$mask, info,
    ctx_tokens = context$tokens, ctx_mask = context$mask)
  expect_equal(as.integer(simple$tokens$shape[[2]]), 7L)
  expect_equal(as.integer(mixed$tokens$shape[[2]]), 7L)
  expect_equal(as.integer(cross$tokens$shape[[2]]), 12L)
  expect_equal(as.numeric(env$np$array(jnp$sum(simple$mask))), 7)
  expect_equal(as.numeric(env$np$array(jnp$sum(mixed$mask))), 7)
  expect_equal(as.numeric(env$np$array(jnp$sum(cross$mask))), 12)
  # A wider explicit order remains valid even when its last token is far past
  # the training lookup bound. Sequence packing must not cut it off again.
  wide_mask <- jnp$array(matrix(c(1, 1, 0, 0, 0, 0, 0, 1), 1L))
  wide <- strategize:::neural_pack_candidate_sequence(choice, cm,
    context$tokens, context$mask, tokens, wide_mask, info)
  expect_equal(as.numeric(env$np$array(jnp$sum(wide$mask))), 7)
})
