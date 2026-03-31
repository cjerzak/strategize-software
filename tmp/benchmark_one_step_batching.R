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

jax <- strategize:::strenv$jax
jnp <- strategize:::strenv$jnp
np <- strategize:::strenv$np
optax <- strategize:::strenv$optax
dtype <- strategize:::strenv$float32

block_ready <- function(x) {
  if (reticulate::py_has_attr(x, "block_until_ready")) {
    x$block_until_ready()
  } else {
    x
  }
}

set.seed(20260331)
n_obs <- 4096L
n_features <- 24L
batch_size <- 256L
n_steps <- 75L

X_host <- matrix(rnorm(n_obs * n_features), nrow = n_obs, ncol = n_features)
y_host <- drop(X_host[, 1] * 0.5 - X_host[, 2] * 0.25 + rnorm(n_obs, sd = 0.1))
batch_index_pool <- replicate(
  n_steps,
  sample.int(n_obs, size = batch_size, replace = FALSE),
  simplify = FALSE
)

params0 <- list(
  w = jnp$array(matrix(rnorm(n_features), ncol = 1L), dtype = dtype),
  b = jnp$array(0, dtype = dtype)
)
optimizer <- optax$adabelief(learning_rate = 0.01)
opt_state0 <- optimizer$init(params0)

loss_fn <- function(params, X_batch, y_batch) {
  preds <- jnp$squeeze(jnp$add(jnp$matmul(X_batch, params$w), params$b), axis = 1L)
  jnp$mean(jnp$square(preds - y_batch))
}

value_and_grad_fn <- jax$value_and_grad(loss_fn, argnums = 0L)

train_step_host <- jax$jit(function(params, opt_state, X_batch, y_batch) {
  step_eval <- value_and_grad_fn(params, X_batch, y_batch)
  grad_set <- step_eval[[2L]]
  updates_and_state <- optimizer$update(grad_set, opt_state, params)
  params_next <- optax$apply_updates(params, updates_and_state[[1L]])
  list(params_next, updates_and_state[[2L]], step_eval[[1L]])
})

X_device <- jnp$array(X_host, dtype = dtype)
y_device <- jnp$array(y_host, dtype = dtype)
train_step_device <- jax$jit(function(params, opt_state, batch_idx) {
  X_batch <- jnp$take(X_device, batch_idx, axis = 0L)
  y_batch <- jnp$take(y_device, batch_idx, axis = 0L)
  step_eval <- value_and_grad_fn(params, X_batch, y_batch)
  grad_set <- step_eval[[2L]]
  updates_and_state <- optimizer$update(grad_set, opt_state, params)
  params_next <- optax$apply_updates(params, updates_and_state[[1L]])
  list(params_next, updates_and_state[[2L]], step_eval[[1L]])
})

run_host_materialized <- function() {
  params <- params0
  opt_state <- opt_state0
  for (batch_idx in batch_index_pool) {
    X_batch <- jnp$array(X_host[batch_idx, , drop = FALSE], dtype = dtype)
    y_batch <- jnp$array(y_host[batch_idx], dtype = dtype)
    step_eval <- train_step_host(params, opt_state, X_batch, y_batch)
    params <- step_eval[[1L]]
    opt_state <- step_eval[[2L]]
    block_ready(step_eval[[3L]])
  }
  invisible(NULL)
}

run_device_indexed <- function() {
  params <- params0
  opt_state <- opt_state0
  for (batch_idx in batch_index_pool) {
    batch_idx_device <- jnp$array(as.integer(batch_idx - 1L), dtype = jnp$int32)
    step_eval <- train_step_device(params, opt_state, batch_idx_device)
    params <- step_eval[[1L]]
    opt_state <- step_eval[[2L]]
    block_ready(step_eval[[3L]])
  }
  invisible(NULL)
}

# Warmup before timing steady-state execution.
run_host_materialized()
run_device_indexed()

host_time <- system.time(run_host_materialized())[["elapsed"]]
device_time <- system.time(run_device_indexed())[["elapsed"]]

results <- data.frame(
  variant = c("host_materialized_batches", "device_indexed_batches"),
  elapsed_seconds = c(host_time, device_time),
  step_seconds = c(host_time, device_time) / n_steps,
  stringsAsFactors = FALSE
)
results$speedup_vs_host <- results$step_seconds[1L] / results$step_seconds

print(results)
write.csv(results, "Tmp/benchmark_one_step_batching_results.csv", row.names = FALSE)
