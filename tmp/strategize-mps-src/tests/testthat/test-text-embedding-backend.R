make_text_embedding_host <- function(...) {
  core_modules <- strategize:::cs2step_backend_core_modules()
  defaults <- list(
    os = "Linux",
    machine = "x86_64",
    conda = "/usr/bin/conda",
    conda_env = "strategize_env",
    conda_registered = TRUE,
    python = tempfile("python"),
    python_exists = TRUE,
    core_modules_ready = TRUE,
    core_module_status = setNames(rep(TRUE, length(core_modules)), core_modules),
    core_module_details = setNames(rep("", length(core_modules)), core_modules),
    mlx_host_capable = FALSE,
    nvidia_tools = list(nvidia_smi = FALSE, nvcc = FALSE),
    nvidia_driver = list(available = FALSE, driver_version = "", driver_major = NA_integer_, device_name = ""),
    cuda_runtime = list(validated = FALSE, cuda_available = FALSE, cuda_version = "", hip_version = "", device_name = ""),
    rocm_tools = list(rocminfo = FALSE, hipcc = FALSE, rocm_smi = FALSE, rocm_root = FALSE),
    rocm_runtime = list(validated = FALSE, cuda_available = FALSE, hip_version = "", device_name = "")
  )
  modifyList(defaults, list(...))
}

test_that("text embedding selector chooses MLX on Apple Silicon in auto mode", {
  host <- make_text_embedding_host(
    os = "Darwin",
    machine = "arm64",
    mlx_host_capable = TRUE
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

test_that("CUDA candidate is installable on supported Nvidia hosts even before validation", {
  host <- make_text_embedding_host(
    nvidia_tools = list(nvidia_smi = TRUE, nvcc = TRUE),
    nvidia_driver = list(
      available = TRUE,
      driver_version = "580.12",
      driver_major = 580L,
      device_name = "NVIDIA RTX"
    )
  )
  candidate <- strategize:::cs2step_resolve_text_embedding_candidates(host)$cuda

  testthat::local_mocked_bindings(
    cs2step_python_module_probe = function(python, modules) {
      list(
        ok = setNames(rep(TRUE, length(modules)), modules),
        details = setNames(rep("", length(modules)), modules),
        status = 0L
      )
    },
    .package = "strategize"
  )

  evaluated <- strategize:::cs2step_evaluate_text_embedding_candidate(candidate, host)

  expect_equal(evaluated$status, "needs_install")
  expect_true(isTRUE(evaluated$installable))
  expect_true(any(grepl("CUDA validation did not succeed", evaluated$issues, fixed = TRUE)))
})

test_that("text embedding selector prefers CUDA on validated Nvidia hosts", {
  host <- make_text_embedding_host(
    nvidia_tools = list(nvidia_smi = TRUE, nvcc = TRUE),
    nvidia_driver = list(
      available = TRUE,
      driver_version = "580.12",
      driver_major = 580L,
      device_name = "NVIDIA RTX"
    ),
    cuda_runtime = list(
      validated = TRUE,
      cuda_available = TRUE,
      cuda_version = "13.0",
      hip_version = "",
      device_name = "NVIDIA RTX"
    )
  )
  candidates <- strategize:::cs2step_resolve_text_embedding_candidates(host)

  testthat::local_mocked_bindings(
    cs2step_evaluate_text_embedding_candidate = function(candidate, host) {
      if (identical(candidate$device, "cuda")) {
        candidate$status <- "ready"
      } else if (identical(candidate$device, "cpu")) {
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

  expect_equal(inspected$selected$backend, "sentence_transformers")
  expect_equal(inspected$selected$device, "cuda")
})

test_that("text embedding selector prefers ROCm when CUDA is unavailable", {
  host <- make_text_embedding_host(
    rocm_tools = list(rocminfo = TRUE, hipcc = TRUE, rocm_smi = TRUE, rocm_root = TRUE),
    rocm_runtime = list(validated = TRUE, cuda_available = TRUE, hip_version = "6.3", device_name = "AMD GPU")
  )
  candidates <- strategize:::cs2step_resolve_text_embedding_candidates(host)

  testthat::local_mocked_bindings(
    cs2step_evaluate_text_embedding_candidate = function(candidate, host) {
      if (identical(candidate$device, "rocm")) {
        candidate$status <- "ready"
      } else if (identical(candidate$device, "cpu")) {
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

  expect_equal(inspected$selected$backend, "sentence_transformers")
  expect_equal(inspected$selected$device, "rocm")
})

test_that("text embedding selector falls back to CPU when GPU runtimes are unavailable", {
  host <- make_text_embedding_host(
    nvidia_tools = list(nvidia_smi = TRUE, nvcc = TRUE),
    nvidia_driver = list(
      available = TRUE,
      driver_version = "510.12",
      driver_major = 510L,
      device_name = "Old NVIDIA"
    ),
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
  expect_true(any(grepl("falling back", inspected$issues, fixed = TRUE)))
})

test_that("CUDA sentence-transformers install uses the CUDA 13 wheel index for new drivers", {
  calls <- list()
  spec <- list(
    family = "qwen3",
    profile = "portable",
    runtime = "cuda",
    backend = "sentence_transformers",
    label = "sentence_transformers_cuda",
    device = "cuda",
    model_id = "Qwen/Qwen3-Embedding-0.6B",
    conda_env = "strategize_env",
    conda = "/usr/bin/conda",
    canonical_dim = 1024L,
    raw_dim = 1024L
  )

  testthat::local_mocked_bindings(
    cs2step_probe_nvidia_driver = function() {
      list(available = TRUE, driver_version = "580.12", driver_major = 580L, device_name = "NVIDIA RTX")
    },
    cs2step_pip_install_in_conda = function(conda, conda_env, packages, index_url = NULL, force_reinstall = FALSE) {
      calls <<- c(calls, list(list(packages = packages, index_url = index_url, force_reinstall = force_reinstall)))
      invisible(TRUE)
    },
    .package = "strategize"
  )

  strategize:::cs2step_install_cuda_sentence_transformers(spec)

  expect_equal(calls[[1]]$packages, "torch")
  expect_equal(calls[[1]]$index_url, "https://download.pytorch.org/whl/cu130")
  expect_true(isTRUE(calls[[1]]$force_reinstall))
  expect_equal(calls[[2]]$packages, c("sentence-transformers", "transformers"))
})

test_that("CUDA sentence-transformers install uses the CUDA 12 wheel index for mid-range drivers", {
  calls <- list()
  spec <- list(
    family = "qwen3",
    profile = "portable",
    runtime = "cuda",
    backend = "sentence_transformers",
    label = "sentence_transformers_cuda",
    device = "cuda",
    model_id = "Qwen/Qwen3-Embedding-0.6B",
    conda_env = "strategize_env",
    conda = "/usr/bin/conda",
    canonical_dim = 1024L,
    raw_dim = 1024L
  )

  testthat::local_mocked_bindings(
    cs2step_probe_nvidia_driver = function() {
      list(available = TRUE, driver_version = "530.40", driver_major = 530L, device_name = "NVIDIA RTX")
    },
    cs2step_pip_install_in_conda = function(conda, conda_env, packages, index_url = NULL, force_reinstall = FALSE) {
      calls <<- c(calls, list(list(packages = packages, index_url = index_url, force_reinstall = force_reinstall)))
      invisible(TRUE)
    },
    .package = "strategize"
  )

  strategize:::cs2step_install_cuda_sentence_transformers(spec)

  expect_equal(calls[[1]]$packages, "torch")
  expect_equal(calls[[1]]$index_url, "https://download.pytorch.org/whl/cu128")
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
