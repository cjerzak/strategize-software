# =============================================================================
# Tests for REINFORCE policy probability helpers
# =============================================================================

ensure_test_jax <- function() {
  if (!"jnp" %in% ls(envir = strategize:::strenv)) {
    strategize:::initialize_jax(conda_env = "strategize_env", conda_env_required = TRUE)
  }
}

test_that("policy support weights match grouped multinomial probabilities", {
  skip_on_cran()
  skip_if_no_jax()
  ensure_test_jax()

  full_locator <- strategize:::strenv$jnp$array(c(1L, 1L, 2L, 2L), dtype = strategize:::strenv$jnp$int32)
  full_pi <- strategize:::strenv$jnp$array(c(0.2, 0.8, 0.7, 0.3), dtype = strategize:::strenv$dtj)
  full_profiles <- strategize:::strenv$jnp$array(
    rbind(
      c(1, 0, 1, 0),
      c(0, 1, 0, 1),
      c(1, 0, 0, 1)
    ),
    dtype = strategize:::strenv$dtj
  )

  full_weights <- strategize:::compute_policy_support_weights(
    pi_vec = full_pi,
    profiles = full_profiles,
    ParameterizationType = "Full",
    d_locator_use = full_locator
  )

  expect_equal(
    as.numeric(strategize:::strenv$np$array(full_weights)),
    c(0.2 * 0.7, 0.8 * 0.3, 0.2 * 0.3),
    tolerance = 1e-6
  )

  implicit_locator <- strategize:::strenv$jnp$array(c(1L, 1L, 2L), dtype = strategize:::strenv$jnp$int32)
  implicit_pi <- strategize:::strenv$jnp$array(c(0.2, 0.3, 0.4), dtype = strategize:::strenv$dtj)
  implicit_profiles <- strategize:::strenv$jnp$array(
    rbind(
      c(1, 0, 1),
      c(0, 1, 0),
      c(0, 0, 1),
      c(0, 0, 0)
    ),
    dtype = strategize:::strenv$dtj
  )

  implicit_weights <- strategize:::compute_policy_support_weights(
    pi_vec = implicit_pi,
    profiles = implicit_profiles,
    ParameterizationType = "Implicit",
    d_locator_use = implicit_locator
  )

  expect_equal(
    as.numeric(strategize:::strenv$np$array(implicit_weights)),
    c(
      0.2 * 0.4,
      0.3 * (1 - 0.4),
      (1 - 0.2 - 0.3) * 0.4,
      (1 - 0.2 - 0.3) * (1 - 0.4)
    ),
    tolerance = 1e-6
  )
})

test_that("policy sample log-prob helper handles rank-4 pooled batches", {
  skip_on_cran()
  skip_if_no_jax()
  ensure_test_jax()

  locator <- strategize:::strenv$jnp$array(c(1L, 1L, 2L, 2L), dtype = strategize:::strenv$jnp$int32)
  pi_vec <- strategize:::strenv$jnp$array(c(0.2, 0.8, 0.7, 0.3), dtype = strategize:::strenv$dtj)
  pooled_profiles <- array(0, dim = c(2L, 2L, 1L, 4L))
  pooled_profiles[1L, 1L, 1L, ] <- c(1, 0, 1, 0)
  pooled_profiles[1L, 2L, 1L, ] <- c(0, 1, 0, 1)
  pooled_profiles[2L, 1L, 1L, ] <- c(1, 0, 0, 1)
  pooled_profiles[2L, 2L, 1L, ] <- c(0, 1, 1, 0)
  pooled_profiles <- strategize:::strenv$jnp$array(pooled_profiles, dtype = strategize:::strenv$dtj)

  log_probs <- strategize:::compute_policy_sample_log_probs(
    pi_vec = pi_vec,
    profiles = pooled_profiles,
    ParameterizationType = "Full",
    d_locator_use = locator
  )

  expected <- c(
    log(0.2 * 0.7) + log(0.8 * 0.3),
    log(0.2 * 0.3) + log(0.8 * 0.7)
  )
  expect_equal(
    as.numeric(strategize:::strenv$np$array(log_probs)),
    expected,
    tolerance = 1e-6
  )
})
