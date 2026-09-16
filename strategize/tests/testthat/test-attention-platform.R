test_that("ROCm's generic gpu backend cannot select cuDNN attention", {
  device <- list(client = list(platform_version = "PJRT C API\nrocm 71526333"))
  testthat::local_mocked_bindings(
    strenv = list(jax = list(default_backend = function() "gpu", devices = function() list(device))),
    neural_attention_has_dpa = function() TRUE,
    .package = "strategize"
  )
  expect_false(strategize:::neural_attention_cuda_available())
  resolved <- strategize:::neural_attention_resolve_backend(list(attention_backend = "auto"))
  expect_identical(resolved$backend, "xla")
  expect_false(resolved$cuda_available)
  expect_identical(resolved$fallback_reason, "cuda_unavailable")
  expect_error(
    strategize:::neural_attention_resolve_backend(list(attention_backend = "cudnn")),
    "requires a CUDA-backed JAX device", fixed = TRUE
  )
})

test_that("CUDA client metadata retains automatic cuDNN attention", {
  device <- list(client = list(platform_version = "PJRT C API\ncuda 13000"))
  testthat::local_mocked_bindings(
    strenv = list(jax = list(default_backend = function() "gpu", devices = function() list(device))),
    neural_attention_has_dpa = function() TRUE,
    .package = "strategize"
  )
  expect_true(strategize:::neural_attention_cuda_available())
  resolved <- strategize:::neural_attention_resolve_backend(list(attention_backend = "auto"))
  expect_identical(resolved$backend, "cudnn")
  expect_true(resolved$cuda_available)
})

test_that("legacy device names remain usable without client metadata", {
  device <- "CudaDevice(id=0)"
  testthat::local_mocked_bindings(
    strenv = list(jax = list(devices = function() list(device))),
    .package = "strategize"
  )
  expect_true(strategize:::neural_attention_cuda_available())
  device <- "RocmDevice(id=0)"
  expect_false(strategize:::neural_attention_cuda_available())
  device <- "CpuDevice(id=0)"
  expect_false(strategize:::neural_attention_cuda_available())
})

test_that("unavailable devices do not claim CUDA support", {
  testthat::local_mocked_bindings(
    strenv = list(jax = list(devices = function() stop("backend unavailable"))),
    .package = "strategize"
  )
  expect_false(strategize:::neural_attention_cuda_available())
})
