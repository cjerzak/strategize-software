test_that("build_backend installs CPU JAX when nvidia-smi is unavailable", {
  skip_on_cran()
  skip_if_not_installed("withr")

  install_calls <- list()

  mock_py_install <- function(packages, envname, conda, pip, ...) {
    install_calls <<- c(install_calls, list(list(packages = packages, conda = conda)))
    TRUE
  }

  testthat::local_mocked_bindings(
    conda_create = function(envname, conda, python_version) TRUE,
    conda_list = function(conda = NULL) data.frame(name = "test_env"),
    py_install = mock_py_install,
    .package = "reticulate"
  )

  testthat::local_mocked_bindings(
    system = function(...) stop("nvidia-smi not available"),
    Sys.info = function() c(sysname = "Linux"),
    Sys.which = function(x) "",
    .package = "base"
  )

  withr::local_envvar(HOME = tempdir())

  build_backend(conda_env = "test_env", conda = "auto")

  installed <- unlist(lapply(install_calls, `[[`, "packages"))
  expect_true("jax" %in% installed)
  expect_true(all(c("numpy", "equinox", "numpyro", "optax") %in% installed))
})

test_that("build_backend falls back to PATH conda and selects CUDA 13", {
  skip_on_cran()
  skip_if_not_installed("withr")

  install_calls <- list()

  mock_py_install <- function(packages, envname, conda, pip, ...) {
    install_calls <<- c(install_calls, list(list(packages = packages, conda = conda)))
    TRUE
  }

  mock_conda_create <- function(envname, conda, python_version) {
    if (identical(conda, "auto")) {
      stop("auto conda failed")
    }
    TRUE
  }

  mock_conda_list <- function(conda = NULL) {
    if (identical(conda, "/usr/bin/conda")) {
      return(data.frame(name = "test_env"))
    }
    data.frame(name = character())
  }

  testthat::local_mocked_bindings(
    conda_create = mock_conda_create,
    conda_list = mock_conda_list,
    py_install = mock_py_install,
    .package = "reticulate"
  )

  testthat::local_mocked_bindings(
    system = function(...) "580.12",
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

  mock_py_install <- function(packages, envname, conda, pip, ...) {
    install_calls <<- c(install_calls, list(list(packages = packages, conda = conda)))
    TRUE
  }

  testthat::local_mocked_bindings(
    conda_create = function(envname, conda, python_version) TRUE,
    conda_list = function(conda = NULL) data.frame(name = "test_env"),
    py_install = mock_py_install,
    .package = "reticulate"
  )

  testthat::local_mocked_bindings(
    system = function(...) "530.40",
    Sys.info = function() c(sysname = "Linux"),
    Sys.which = function(x) "",
    .package = "base"
  )

  withr::local_envvar(HOME = tempdir())

  build_backend(conda_env = "test_env", conda = "auto")

  installed <- unlist(lapply(install_calls, `[[`, "packages"))
  expect_true("jax[cuda12]" %in% installed)
  expect_false("jax[cuda13]" %in% installed)
})
