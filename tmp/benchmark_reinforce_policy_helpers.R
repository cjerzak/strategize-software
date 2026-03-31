suppressPackageStartupMessages({
  library(reticulate)
})

if (!requireNamespace("devtools", quietly = TRUE)) {
  stop("devtools is required to run this benchmark.")
}

devtools::load_all("strategize", quiet = TRUE)

if (!"jnp" %in% ls(envir = strategize:::strenv)) {
  strategize:::initialize_jax(conda_env = "strategize_env", conda_env_required = TRUE)
}

jnp <- strategize:::strenv$jnp
np <- strategize:::strenv$np
dtype <- strategize:::strenv$dtj

block_ready <- function(x) {
  if (reticulate::py_has_attr(x, "block_until_ready")) {
    x$block_until_ready()
  } else {
    x
  }
}

old_group_prob_matrix <- function(pi_vec, flat_profiles, ParameterizationType, index_spec) {
  grouped_probs <- lapply(index_spec$index_list, function(group_idx) {
    group_idx_vec <- as.integer(unlist(group_idx, use.names = FALSE))
    if (length(group_idx_vec) < 1L) {
      return(jnp$ones(list(strategize:::ai(flat_profiles$shape[[1L]])), dtype = pi_vec$dtype))
    }
    group_idx_jax <- jnp$array(group_idx_vec, dtype = jnp$int32)
    profile_group <- jnp$take(flat_profiles, group_idx_jax, axis = 1L)
    pi_group <- jnp$take(pi_vec, group_idx_jax, axis = 0L)
    active_prob <- jnp$sum(profile_group * pi_group, axis = 1L)
    if (!identical(ParameterizationType, "Implicit")) {
      return(active_prob)
    }
    holdout_prob <- jnp$array(1, dtype = pi_vec$dtype) - jnp$sum(pi_group)
    group_sum <- jnp$sum(profile_group, axis = 1L)
    jnp$where(
      jnp$equal(group_sum, jnp$array(0, dtype = group_sum$dtype)),
      holdout_prob,
      active_prob
    )
  })
  jnp$stack(grouped_probs, axis = 1L)
}

old_compute_policy_sample_log_probs <- function(pi_vec,
                                                profiles,
                                                ParameterizationType,
                                                index_spec) {
  profile_rank <- length(profiles$shape)
  n_outer <- strategize:::ai(profiles$shape[[1L]])
  if (profile_rank %in% c(2L, 3L)) {
    n_inner <- 1L
    param_dims <- vapply(seq.int(2L, profile_rank), function(idx) {
      strategize:::ai(profiles$shape[[idx]])
    }, integer(1))
  } else {
    n_inner <- strategize:::ai(profiles$shape[[2L]])
    param_dims <- vapply(seq.int(3L, profile_rank), function(idx) {
      strategize:::ai(profiles$shape[[idx]])
    }, integer(1))
  }
  n_params <- as.integer(prod(param_dims))
  flat_profiles <- jnp$reshape(profiles, list(-1L, n_params))
  grouped_probs <- old_group_prob_matrix(pi_vec, flat_profiles, ParameterizationType, index_spec)
  log_probs <- jnp$sum(jnp$log(jnp$clip(grouped_probs, 1e-8, 1)), axis = 1L)
  if (n_inner == 1L) {
    return(jnp$reshape(log_probs, list(n_outer)))
  }
  reshaped <- jnp$reshape(log_probs, list(n_outer, n_inner))
  jnp$sum(reshaped, axis = 1L)
}

set.seed(20260331)
group_count <- 6L
levels_per_group <- 3L
d_locator <- rep(seq_len(group_count), each = levels_per_group)
d_locator_jax <- jnp$array(as.integer(d_locator), dtype = jnp$int32)
pi_raw <- matrix(runif(group_count * levels_per_group), nrow = group_count, ncol = levels_per_group)
pi_mat <- pi_raw / rowSums(pi_raw)
pi_vec <- jnp$array(as.numeric(t(pi_mat)), dtype = dtype)
index_spec <- strategize:::resolve_multinomial_group_index_spec(d_locator_jax, "Full")

n_outer <- 1024L
n_inner <- 4L
n_params <- length(d_locator)
draws <- array(0, dim = c(n_outer, n_inner, 1L, n_params))
for (outer_i in seq_len(n_outer)) {
  for (inner_i in seq_len(n_inner)) {
    offset <- 0L
    for (group_i in seq_len(group_count)) {
      chosen <- sample.int(levels_per_group, size = 1L, prob = pi_mat[group_i, ])
      draws[outer_i, inner_i, 1L, offset + chosen] <- 1
      offset <- offset + levels_per_group
    }
  }
}
profiles <- jnp$array(draws, dtype = dtype)
support_profiles <- jnp$array(matrix(draws[1L, , 1L, ], nrow = n_inner, byrow = TRUE), dtype = dtype)

run_old_log_probs <- function() {
  out <- old_compute_policy_sample_log_probs(pi_vec, profiles, "Full", index_spec)
  invisible(block_ready(out))
}

run_new_log_probs <- function() {
  out <- strategize:::compute_policy_sample_log_probs(
    pi_vec = pi_vec,
    profiles = profiles,
    ParameterizationType = "Full",
    index_spec = index_spec
  )
  invisible(block_ready(out))
}

run_old_support_weights <- function() {
  flat_profiles <- jnp$reshape(support_profiles, list(strategize:::ai(support_profiles$shape[[1L]]), -1L))
  out <- jnp$prod(old_group_prob_matrix(pi_vec, flat_profiles, "Full", index_spec), axis = 1L)
  invisible(block_ready(out))
}

run_new_support_weights <- function() {
  out <- strategize:::compute_policy_support_weights(
    pi_vec = pi_vec,
    profiles = support_profiles,
    ParameterizationType = "Full",
    d_locator_use = d_locator_jax
  )
  invisible(block_ready(out))
}

# Warmup before timing steady-state execution.
run_old_log_probs()
run_new_log_probs()
run_old_support_weights()
run_new_support_weights()

old_log_time <- system.time(replicate(10L, run_old_log_probs()))[["elapsed"]]
new_log_time <- system.time(replicate(10L, run_new_log_probs()))[["elapsed"]]
old_weight_time <- system.time(replicate(20L, run_old_support_weights()))[["elapsed"]]
new_weight_time <- system.time(replicate(20L, run_new_support_weights()))[["elapsed"]]

results <- data.frame(
  benchmark = c(
    "old_log_probs",
    "new_log_probs",
    "old_support_weights",
    "new_support_weights"
  ),
  elapsed_seconds = c(old_log_time, new_log_time, old_weight_time, new_weight_time),
  stringsAsFactors = FALSE
)
results$speedup_vs_old <- c(
  1,
  old_log_time / new_log_time,
  1,
  old_weight_time / new_weight_time
)

print(results)
write.csv(results, "Tmp/benchmark_reinforce_policy_helpers_results.csv", row.names = FALSE)
