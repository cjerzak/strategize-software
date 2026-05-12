#' Build the computational environment for `strategize`
#'
#' Creates the conda environment used by the package's JAX-backed optimization
#' workflow, neural APIs, and foundation checkpoint loading. Users may also
#' create and manage a compatible environment themselves.
#'
#' @param conda_env Name of the conda environment in which to place the backends.
#'   Defaults to \code{"strategize_env"}.
#' @param conda The path to a conda executable. Using \code{"auto"} allows reticulate
#'   to attempt to automatically find an appropriate conda binary. Defaults to \code{"auto"}.
#'   If creation fails and a conda binary is found on PATH, the function retries with it.
#'
#' @return Invisibly returns \code{NULL}. This function is called for its side effects
#'   of creating and configuring a conda environment for \code{strategize}.
#'   This function requires an Internet connection.
#'   You can find a list of conda Python paths via: \code{Sys.which("python")}
#'
#' @examples
#' \dontrun{
#' # Create a conda environment named "strategize_env"
#' # and install the required Python packages (jax, numpy, orbax-checkpoint, etc.)
#' build_backend(conda_env = "strategize_env", conda = "auto")
#'
#' # If you want to specify a particular conda path:
#' # build_backend(conda_env = "strategize_env", conda = "/usr/local/bin/conda")
#' }
#'
#' @export
#' @md

build_backend <- function(conda_env = "strategize_env", conda = "auto") {
  env_registered <- function(conda_bin) {
    state <- cs2step_backend_env_state(conda_env = conda_env, conda = conda_bin)
    isTRUE(state$registered)
  }

  create_env <- function(conda_bin) {
    reticulate::conda_create(envname = conda_env, conda = conda_bin, python_version = "3.12")
    invisible(TRUE)
  }

  remove_env <- function(conda_bin) {
    tryCatch(
      reticulate::conda_remove(envname = conda_env, conda = conda_bin),
      error = function(e) {
        if (nzchar(conda_bin %||% "")) {
          suppressWarnings(system2(
            conda_bin,
            c("env", "remove", "-n", conda_env, "-y"),
            stdout = TRUE,
            stderr = TRUE
          ))
        }
        invisible(FALSE)
      }
    )
    invisible(TRUE)
  }

  conda_bin <- cs2step_resolve_conda_binary(conda) %||% conda
  state <- cs2step_backend_env_state(conda_env = conda_env, conda = conda_bin)
  if (isTRUE(state$core_modules_ready)) {
    try({
      actdir <- file.path(dirname(dirname(state$python)), "etc", "conda", "activate.d")
      dir.create(actdir, recursive = TRUE, showWarnings = FALSE)
      writeLines("unset LD_LIBRARY_PATH", file.path(actdir, "00-unset-ld.sh"))
    }, silent = TRUE)
    message(sprintf("Environment '%s' is ready.", conda_env))
    return(invisible(NULL))
  }

  if (isTRUE(state$registered) && !isTRUE(state$python_exists)) {
    message(sprintf(
      "Conda environment '%s' is registered but its Python interpreter is missing; recreating it.",
      conda_env
    ))
    remove_env(conda_bin)
    state <- cs2step_backend_env_state(conda_env = conda_env, conda = conda_bin)
  }

  ok <- TRUE
  if (!isTRUE(state$registered)) {
    ok <- tryCatch({
      create_env(conda_bin)
      TRUE
    }, error = function(e) {
      message(sprintf("conda_create failed using '%s': %s", conda_bin, e$message))
      FALSE
    })
  }

  if (!ok || !env_registered(conda_bin)) {
    conda_fallback <- Sys.which("conda")
    if (nzchar(conda_fallback) && conda_fallback != conda_bin) {
      message(sprintf("Retrying conda_create with: %s", conda_fallback))
      tryCatch({
        create_env(conda_fallback)
      }, error = function(e) {
        message(sprintf("conda_create failed using '%s': %s", conda_fallback, e$message))
      })
      if (env_registered(conda_fallback)) {
        conda_bin <- conda_fallback
      }
    }
  }

  state <- cs2step_backend_env_state(conda_env = conda_env, conda = conda_bin)
  if (!isTRUE(state$registered)) {
    stop(
      sprintf("Failed to create conda environment '%s'.\n", conda_env),
      "Try passing conda = '/path/to/conda' or setting RETICULATE_CONDA.\n",
      call. = FALSE
    )
  }
  
  os <- Sys.info()[["sysname"]]
  msg <- function(...) message(sprintf(...))
  
  pip_install <- function(pkgs, ...) {
    reticulate::py_install(packages = pkgs, envname = conda_env, conda = conda_bin, pip = TRUE, ...)
    TRUE
  }
  
  # --- (A) Choose CUDA 13 vs 12 *by driver version* and install JAX FIRST ---
  install_jax_gpu <- function() {
    if (!identical(os, "Linux")){
      return(pip_install("jax")) 
    }
    
    driver <- cs2step_probe_nvidia_driver()
    drv <- driver$driver_version %||% "unknown"
    drv_major <- driver$driver_major %||% NA_integer_
    
    # Prefer CUDA 13 if the driver is new enough; otherwise CUDA 12; else CPU fallback
    if (!is.na(drv_major) && drv_major >= 580) {
      msg("Driver %s detected (>=580): installing JAX CUDA 13 wheels.", drv)
      tryCatch(pip_install('jax[cuda13]'), error = function(e) {
        msg("CUDA 13 wheels failed (%s); falling back to CUDA 12.", e$message)
        pip_install('jax[cuda12]')
      })
    } else if (!is.na(drv_major) && drv_major >= 525) {
      msg("Driver %s detected (>=525,<580): installing JAX CUDA 12 wheels.", drv)
      pip_install('jax[cuda12]')
    } else {
      msg("Driver %s too old for CUDA wheels; installing CPU-only JAX.", drv)
      pip_install('jax')
    }
  }
  
  missing_core <- names(state$core_module_status)[!state$core_module_status]
  if (length(missing_core) > 0L || !isTRUE(state$python_exists)) {
    # Install JAX first so later dependencies do not pin a CPU-only variant.
    install_jax_gpu()
    pip_specs <- cs2step_backend_core_pip_packages()
    other_pkgs <- unname(pip_specs[setdiff(names(pip_specs), "jax")])
    pip_install(other_pkgs)
  }

  state <- cs2step_backend_env_state(conda_env = conda_env, conda = conda_bin)
  if (!isTRUE(state$core_modules_ready)) {
    missing_now <- names(state$core_module_status)[!state$core_module_status]
    stop(
      sprintf(
        "Conda environment '%s' is still missing required backend modules: %s",
        conda_env,
        paste(missing_now, collapse = ", ")
      ),
      call. = FALSE
    )
  }

  # (Optional) neutralize LD_LIBRARY_PATH inside this env to prevent overrides
  try({
    actdir <- file.path(dirname(dirname(state$python)), "etc", "conda", "activate.d")
    dir.create(actdir, recursive = TRUE, showWarnings = FALSE)
    writeLines("unset LD_LIBRARY_PATH", file.path(actdir, "00-unset-ld.sh"))
  }, silent = TRUE)
  
  msg("Environment '%s' is ready.", conda_env)
}
