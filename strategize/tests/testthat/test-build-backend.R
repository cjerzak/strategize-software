test_that("build_backend is idempotent when the core env is already healthy", {
  skip_on_cran()
  skip_if_not_installed("withr")

  py_path <- file.path(tempdir(), "env", "bin", "python")
  dir.create(dirname(py_path), recursive = TRUE, showWarnings = FALSE)
  file.create(py_path)

  testthat::local_mocked_bindings(
    cs2step_resolve_conda_binary = function(conda = "auto") "/usr/bin/conda",
    cs2step_backend_env_state = function(conda_env, conda) {
      list(
        conda = "/usr/bin/conda",
        conda_env = conda_env,
        registered = TRUE,
        python = py_path,
        python_exists = TRUE,
        core_module_status = setNames(rep(TRUE, 5L), strategize:::cs2step_backend_core_modules()),
        core_module_details = setNames(rep("", 5L), strategize:::cs2step_backend_core_modules()),
        core_modules_ready = TRUE
      )
    },
    .package = "strategize"
  )

  testthat::local_mocked_bindings(
    conda_create = function(...) stop("conda_create should not run"),
    py_install = function(...) stop("py_install should not run"),
    .package = "reticulate"
  )

  withr::local_envvar(HOME = tempdir())

  expect_invisible(build_backend(conda_env = "test_env", conda = "auto"))
})

test_that("build_backend installs CPU JAX when nvidia-smi is unavailable", {
  skip_on_cran()
  skip_if_not_installed("withr")

  install_calls <- list()
  py_path <- file.path(tempdir(), "env", "bin", "python")
  registered <- FALSE
  installed <- FALSE

  states <- function(conda_env, conda) {
    if (registered) {
      dir.create(dirname(py_path), recursive = TRUE, showWarnings = FALSE)
      if (!file.exists(py_path)) {
        file.create(py_path)
      }
    }
    ready <- registered && installed
    list(
      conda = "/usr/bin/conda",
      conda_env = conda_env,
      registered = registered,
      python = if (registered) py_path else "",
      python_exists = registered,
      core_module_status = setNames(rep(ready, 5L), strategize:::cs2step_backend_core_modules()),
      core_module_details = setNames(rep("", 5L), strategize:::cs2step_backend_core_modules()),
      core_modules_ready = ready
    )
  }

  testthat::local_mocked_bindings(
    cs2step_resolve_conda_binary = function(conda = "auto") "/usr/bin/conda",
    cs2step_backend_env_state = states,
    cs2step_probe_nvidia_driver = function() {
      list(available = FALSE, driver_version = "unknown", driver_major = NA_integer_, device_name = "")
    },
    .package = "strategize"
  )

  testthat::local_mocked_bindings(
    conda_create = function(envname, conda, python_version) {
      registered <<- TRUE
      TRUE
    },
    py_install = function(packages, envname, conda, pip, ...) {
      install_calls <<- c(install_calls, list(list(packages = packages, conda = conda)))
      installed <<- TRUE
      TRUE
    },
    .package = "reticulate"
  )

  testthat::local_mocked_bindings(Sys.info = function() c(sysname = "Linux"), .package = "base")

  withr::local_envvar(HOME = tempdir())

  build_backend(conda_env = "test_env", conda = "auto")

  installed <- unlist(lapply(install_calls, `[[`, "packages"))
  expect_true("jax" %in% installed)
  expect_true(all(c("numpy", "equinox", "numpyro", "optax") %in% installed))
  expect_false("mlx-embeddings" %in% installed)
})

test_that("build_backend falls back to PATH conda and selects CUDA 13", {
  skip_on_cran()
  skip_if_not_installed("withr")

  install_calls <- list()
  py_path <- file.path(tempdir(), "env", "bin", "python")
  registered <- FALSE
  installed <- FALSE

  states <- function(conda_env, conda) {
    if (identical(conda, "auto")) {
      return(list(
        conda = "auto",
        conda_env = conda_env,
        registered = FALSE,
        python = "",
        python_exists = FALSE,
        core_module_status = setNames(rep(FALSE, 5L), strategize:::cs2step_backend_core_modules()),
        core_module_details = setNames(rep("", 5L), strategize:::cs2step_backend_core_modules()),
        core_modules_ready = FALSE
      ))
    }
    if (registered) {
      dir.create(dirname(py_path), recursive = TRUE, showWarnings = FALSE)
      if (!file.exists(py_path)) {
        file.create(py_path)
      }
    }
    ready <- registered && installed
    list(
      conda = "/usr/bin/conda",
      conda_env = conda_env,
      registered = registered,
      python = if (registered) py_path else "",
      python_exists = registered,
      core_module_status = setNames(rep(ready, 5L), strategize:::cs2step_backend_core_modules()),
      core_module_details = setNames(rep("", 5L), strategize:::cs2step_backend_core_modules()),
      core_modules_ready = ready
    )
  }

  testthat::local_mocked_bindings(
    cs2step_resolve_conda_binary = function(conda = "auto") "auto",
    cs2step_backend_env_state = states,
    cs2step_probe_nvidia_driver = function() {
      list(available = TRUE, driver_version = "580.12", driver_major = 580L, device_name = "NVIDIA GPU")
    },
    .package = "strategize"
  )

  testthat::local_mocked_bindings(
    conda_create = function(envname, conda, python_version) {
      if (identical(conda, "auto")) {
        stop("auto conda failed")
      }
      registered <<- TRUE
      TRUE
    },
    py_install = function(packages, envname, conda, pip, ...) {
      install_calls <<- c(install_calls, list(list(packages = packages, conda = conda)))
      installed <<- TRUE
      TRUE
    },
    .package = "reticulate"
  )

  testthat::local_mocked_bindings(
    Sys.info = function() c(sysname = "Linux"),
    Sys.which = function(x) "/usr/bin/conda",
    .package = "base"
  )

  withr::local_envvar(HOME = tempdir())

  build_backend(conda_env = "test_env", conda = "auto")

  installed <- unlist(lapply(install_calls, `[[`, "packages"))
  condas <- unlist(lapply(install_calls, `[[`, "conda"))
  expect_true("jax[cuda13]" %in% installed)
  expect_true(any(condas == "/usr/bin/conda"))
})

test_that("build_backend selects CUDA 12 for mid-range drivers", {
  skip_on_cran()
  skip_if_not_installed("withr")

  install_calls <- list()
  py_path <- file.path(tempdir(), "env", "bin", "python")
  registered <- FALSE
  installed <- FALSE

  states <- function(conda_env, conda) {
    if (registered) {
      dir.create(dirname(py_path), recursive = TRUE, showWarnings = FALSE)
      if (!file.exists(py_path)) {
        file.create(py_path)
      }
    }
    ready <- registered && installed
    list(
      conda = "/usr/bin/conda",
      conda_env = conda_env,
      registered = registered,
      python = if (registered) py_path else "",
      python_exists = registered,
      core_module_status = setNames(rep(ready, 5L), strategize:::cs2step_backend_core_modules()),
      core_module_details = setNames(rep("", 5L), strategize:::cs2step_backend_core_modules()),
      core_modules_ready = ready
    )
  }

  testthat::local_mocked_bindings(
    cs2step_resolve_conda_binary = function(conda = "auto") "/usr/bin/conda",
    cs2step_backend_env_state = states,
    cs2step_probe_nvidia_driver = function() {
      list(available = TRUE, driver_version = "530.40", driver_major = 530L, device_name = "NVIDIA GPU")
    },
    .package = "strategize"
  )

  testthat::local_mocked_bindings(
    conda_create = function(envname, conda, python_version) {
      registered <<- TRUE
      TRUE
    },
    py_install = function(packages, envname, conda, pip, ...) {
      install_calls <<- c(install_calls, list(list(packages = packages, conda = conda)))
      installed <<- TRUE
      TRUE
    },
    .package = "reticulate"
  )

  testthat::local_mocked_bindings(Sys.info = function() c(sysname = "Linux"), .package = "base")

  withr::local_envvar(HOME = tempdir())

  build_backend(conda_env = "test_env", conda = "auto")

  installed <- unlist(lapply(install_calls, `[[`, "packages"))
  expect_true("jax[cuda12]" %in% installed)
  expect_false("jax[cuda13]" %in% installed)
})
