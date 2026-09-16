test_that("readiness checks do not enumerate host NumPy arrays or scalars", {
  skip_if_not_installed("reticulate")
  if (!reticulate::py_module_available("numpy")) skip("NumPy unavailable")
  np <- reticulate::import("numpy", convert = FALSE)
  # A large broadcast view exercises checkpoint scale without allocating its
  # logical payload. Intercept enumeration so a regression also fails quickly.
  host <- np$broadcast_to(np$array(1), reticulate::tuple(100000000L))
  scalar <- np$float64(1)
  enumerations <- 0L
  block <- strategize:::strategize_jax_block_until_ready
  probe_env <- new.env(parent = environment(block))
  probe_env$as.list <- function(x, ...) {
    enumerations <<- enumerations + 1L
    stop("Host arrays must not be enumerated to check readiness")
  }
  environment(block) <- probe_env

  expect_identical(block(host), host)
  expect_identical(block(list(array = host, nested = list(scalar))),
                   list(array = host, nested = list(scalar)))
  expect_identical(enumerations, 0L)
})

test_that("readiness checks still block nested JAX computations", {
  skip_if_no_jax()
  py <- reticulate::py_run_string(paste(
    "import jax",
    "import jax.numpy as jnp",
    "from unittest.mock import patch",
    "array = jax.jit(lambda a: a @ a)(jnp.ones((128, 128)))",
    "array_class = type(array)",
    "original_block = array_class.block_until_ready",
    "calls = []",
    "def counted_block(value, original=original_block, counter=calls):",
    "    counter.append(True)",
    "    return original(value)",
    "patcher = patch.object(array_class, 'block_until_ready', counted_block)",
    "patcher.__enter__()",
    sep = "\n"
  ), local = TRUE, convert = FALSE)
  withr::defer(py$patcher$`__exit__`(NULL, NULL, NULL))

  arr <- py$array
  strategize:::strategize_jax_block_until_ready(list(arr, inner = list(arr)))

  expect_length(reticulate::py_to_r(py$calls), 2L)
  expect_true(reticulate::py_to_r(arr$is_ready()))
  expect_equal(as.matrix(reticulate::py_to_r(
    reticulate::import("numpy", convert = FALSE)$asarray(arr)
  )), matrix(128, 128, 128))
})

test_that("phase-boundary collection releases dead arrays and preserves live state and JITs", {
  skip_if_no_jax()
  py <- reticulate::py_run_string(paste(
    "import jax, jax.numpy as jnp, weakref",
    "traces = []",
    "def update(x, counter=traces):",
    "    counter.append(True)",
    "    return x + 1",
    "compiled_update = jax.jit(update)",
    "live_state = compiled_update(jnp.arange(128.))",
    sep = "\n"
  ), local = TRUE, convert = FALSE)
  temporary_array <- reticulate::import("jax.numpy", convert = FALSE)$ones(
    reticulate::tuple(256L, 256L))
  weak <- reticulate::import("weakref", convert = FALSE)$ref(temporary_array)
  expect_false(inherits(weak(), "python.builtin.NoneType"))
  rm(temporary_array)

  strategize:::strategize_jax_collect_garbage()
  expect_true(inherits(weak(), "python.builtin.NoneType"))
  updated <- py$compiled_update(py$live_state)
  expect_equal(as.numeric(reticulate::py_to_r(
    reticulate::import("numpy", convert = FALSE)$asarray(updated)
  )), seq_len(128) + 1)
  expect_length(reticulate::py_to_r(py$traces), 1L)
})
