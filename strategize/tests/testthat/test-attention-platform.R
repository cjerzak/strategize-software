mock_attention_platform <- function(platform_version, pallas_status = "", env = parent.frame()) {
  device <- list(client = list(platform_version = platform_version))
  testthat::local_mocked_bindings(
    strenv = list(jax = list(default_backend = function() "gpu", devices = function() list(device))),
    neural_attention_has_dpa = function() TRUE,
    neural_attention_pallas_status = function(head_dim) pallas_status,
    .package = "strategize", .env = env
  )
}

resolve_attention <- function(head_dim, backend = "auto", dtype = "auto") {
  strategize:::neural_attention_resolve_backend(
    list(attention_backend = backend, attention_dtype = dtype, head_dim = head_dim)
  )
}

test_that("ROCm selects FP32 flash attention for narrow heads and never cuDNN", {
  mock_attention_platform("PJRT C API\nrocm 71526333")
  expect_false(strategize:::neural_attention_cuda_available())
  expect_identical(strategize:::neural_attention_gpu_platform(), "rocm")
  expect_identical(resolve_attention(15L)$backend, "pallas")
  wide <- resolve_attention(60L)
  expect_identical(wide$backend, "xla")
  expect_identical(wide$fallback_reason, "rocm_pallas_head_dim")
  expect_identical(resolve_attention(60L, backend = "pallas")$backend, "pallas")
  expect_error(resolve_attention(16L, backend = "cudnn"), "cuda_unavailable", fixed = TRUE)
})

test_that("CUDA prefers FP32 Pallas and uses cuDNN only for supported half-precision heads", {
  mock_attention_platform("PJRT C API\ncuda 13000")
  expect_true(strategize:::neural_attention_cuda_available())
  resolved <- resolve_attention(15L)
  expect_identical(resolved$backend, "pallas")
  expect_true(resolved$cuda_available)
  expect_identical(resolve_attention(16L, dtype = "bf16")$backend, "cudnn")
  # cuDNN rejects head widths that are not multiples of 8 (the 120/8 production shape).
  expect_identical(resolve_attention(15L, dtype = "bf16")$backend, "pallas")
  expect_error(resolve_attention(15L, backend = "cudnn"), "cudnn_head_dim", fixed = TRUE)
  expect_error(resolve_attention(16L, backend = "cudnn", dtype = "float32"),
               "cudnn_requires_fp16_or_bf16", fixed = TRUE)
})

test_that("flash attention falls back to XLA with a recorded reason", {
  mock_attention_platform("PJRT C API\ncuda 13000", pallas_status = "pallas_probe_failed: no Triton")
  failed <- resolve_attention(16L)
  expect_identical(failed$backend, "xla")
  expect_identical(failed$fallback_reason, "pallas_probe_failed: no Triton")
  expect_error(resolve_attention(16L, backend = "pallas"), "no Triton", fixed = TRUE)
  expect_identical(resolve_attention(16L, backend = "xla")$backend, "xla")
})

test_that("CPU-only JAX resolves to XLA", {
  testthat::local_mocked_bindings(
    strenv = list(jax = list(devices = function() list("CpuDevice(id=0)"))),
    neural_attention_has_dpa = function() TRUE,
    .package = "strategize"
  )
  resolved <- resolve_attention(16L)
  expect_identical(resolved$backend, "xla")
  expect_identical(resolved$fallback_reason, "gpu_unavailable")
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
