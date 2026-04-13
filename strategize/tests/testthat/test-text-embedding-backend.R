test_that("text embedding selector chooses MLX on Apple Silicon in auto mode", {
  host <- list(
    os = "Darwin",
    machine = "arm64",
    conda = "/usr/bin/conda",
    conda_env = "strategize_env",
    conda_registered = TRUE,
    python = "/tmp/python",
    python_exists = TRUE,
    core_modules_ready = TRUE,
    core_module_status = setNames(rep(TRUE, 5L), strategize:::cs2step_backend_core_modules()),
    core_module_details = setNames(rep("", 5L), strategize:::cs2step_backend_core_modules()),
    mlx_host_capable = TRUE,
    rocm_tools = list(rocminfo = FALSE, hipcc = FALSE, rocm_smi = FALSE, rocm_root = FALSE),
    rocm_runtime = list(validated = FALSE, cuda_available = FALSE, hip_version = "", device_name = "")
  )
  candidates <- strategize:::cs2step_resolve_text_embedding_candidates(host)

  testthat::local_mocked_bindings(
    cs2step_evaluate_text_embedding_candidate = function(candidate, host) {
      if (identical(candidate$backend, "mlx")) {
        candidate$status <- "ready"
      } else {
        candidate$status <- "needs_install"
      }
      candidate$issues <- character(0)
      candidate
    },
    .package = "strategize"
  )

  inspected <- strategize:::cs2step_select_text_embedding_candidate(
    candidates = candidates,
    host = host,
    runtime = "auto",
    family = "qwen3",
    profile = "portable"
  )

  expect_equal(inspected$selected$backend, "mlx")
  expect_equal(inspected$selected$device, "metal")
})

test_that("text embedding selector prefers ROCm on validated Linux hosts", {
  host <- list(
    os = "Linux",
    machine = "x86_64",
    conda = "/usr/bin/conda",
    conda_env = "strategize_env",
    conda_registered = TRUE,
    python = "/tmp/python",
    python_exists = TRUE,
    core_modules_ready = TRUE,
    core_module_status = setNames(rep(TRUE, 5L), strategize:::cs2step_backend_core_modules()),
    core_module_details = setNames(rep("", 5L), strategize:::cs2step_backend_core_modules()),
    mlx_host_capable = FALSE,
    rocm_tools = list(rocminfo = TRUE, hipcc = TRUE, rocm_smi = TRUE, rocm_root = TRUE),
    rocm_runtime = list(validated = TRUE, cuda_available = TRUE, hip_version = "6.3", device_name = "AMD GPU")
  )
  candidates <- strategize:::cs2step_resolve_text_embedding_candidates(host)

  testthat::local_mocked_bindings(
    cs2step_evaluate_text_embedding_candidate = function(candidate, host) {
      candidate$status <- "ready"
      candidate$issues <- character(0)
      candidate
    },
    .package = "strategize"
  )

  inspected <- strategize:::cs2step_select_text_embedding_candidate(
    candidates = candidates,
    host = host,
    runtime = "auto",
    family = "qwen3",
    profile = "portable"
  )

  expect_equal(inspected$selected$backend, "sentence_transformers")
  expect_equal(inspected$selected$device, "rocm")
})

test_that("text embedding selector falls back to CPU when ROCm is not validated", {
  host <- list(
    os = "Linux",
    machine = "x86_64",
    conda = "/usr/bin/conda",
    conda_env = "strategize_env",
    conda_registered = TRUE,
    python = "/tmp/python",
    python_exists = TRUE,
    core_modules_ready = TRUE,
    core_module_status = setNames(rep(TRUE, 5L), strategize:::cs2step_backend_core_modules()),
    core_module_details = setNames(rep("", 5L), strategize:::cs2step_backend_core_modules()),
    mlx_host_capable = FALSE,
    rocm_tools = list(rocminfo = TRUE, hipcc = TRUE, rocm_smi = TRUE, rocm_root = TRUE),
    rocm_runtime = list(validated = FALSE, cuda_available = FALSE, hip_version = "", device_name = "")
  )
  candidates <- strategize:::cs2step_resolve_text_embedding_candidates(host)

  testthat::local_mocked_bindings(
    cs2step_evaluate_text_embedding_candidate = function(candidate, host) {
      if (identical(candidate$device, "cpu")) {
        candidate$status <- "needs_install"
      } else {
        candidate$status <- "unavailable"
      }
      candidate$issues <- character(0)
      candidate
    },
    .package = "strategize"
  )

  inspected <- strategize:::cs2step_select_text_embedding_candidate(
    candidates = candidates,
    host = host,
    runtime = "auto",
    family = "qwen3",
    profile = "portable"
  )

  expect_equal(inspected$selected$device, "cpu")
  expect_true(any(grepl("falling back to CPU", inspected$issues, fixed = TRUE)))
})

test_that("canonical text embedding width truncates larger matrices", {
  spec <- list(
    family = "qwen3",
    profile = "portable",
    runtime = "auto",
    backend = "mlx",
    label = "mlx",
    model_id = "mlx-community/Qwen3-Embedding-8B-mxfp8",
    conda_env = "strategize_env",
    conda = "/usr/bin/conda",
    canonical_dim = 1024L,
    raw_dim = 4096L
  )
  emb <- matrix(seq_len(2L * 4096L), nrow = 2L, ncol = 4096L)
  out <- strategize:::cs2step_text_embedding_canonicalize_matrix(emb, spec)

  expect_equal(dim(out), c(2L, 1024L))
  expect_equal(out[, 1], emb[, 1])
  expect_equal(out[, 1024], emb[, 1024])
})
