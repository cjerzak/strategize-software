#' Build the environment for `strategize`. Creates a conda environment in which
#' 'JAX' and 'np' are installed. Users may also create such an environment
#' themselves.
#'
#' @param conda_env (default = `"strategize"`) Name of the conda environment in which to place the backends.
#' @param conda (default = `auto`) The path to a conda executable. Using `"auto"` allows reticulate to attempt to automatically find an appropriate conda binary.

#' @return Invisibly returns NULL; this function is used for its side effects 
#' of creating and configuring a conda environment for `strategize`. 
#' This function requires an Internet connection.
#' You can find out a list of conda Python paths via: `Sys.which("python")`
#'
#' @examples
#' \dontrun{
#' # Create a conda environment named "strategize"
#' # and install the required Python packages (jax, numpy, etc.)
#' build_backend(conda_env = "strategize", conda = "auto")
#'
#' # If you want to specify a particular conda path:
#' # build_backend(conda_env = "strategize", conda = "/usr/local/bin/conda")
#' }
#'
#' @export
#' @md

build_backend <- function(conda_env = "strategize_env", conda = "auto") {
  reticulate::conda_create(envname = conda_env, conda = conda, python_version = "3.12")
  
  os <- Sys.info()[["sysname"]]
  msg <- function(...) message(sprintf(...))
  
  pip_install <- function(pkgs, ...) {
    reticulate::py_install(packages = pkgs, envname = conda_env, conda = conda, pip = TRUE, ...)
    TRUE
  }
  
  # --- (A) Choose CUDA 13 vs 12 *by driver version* and install JAX FIRST ---
  install_jax_gpu <- function() {
    if (!identical(os, "Linux")){
      return(pip_install("jax")) 
    }
    
    # Read driver version as integer major (e.g., 580)
    drv <- try(suppressWarnings(system("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1", intern=TRUE)), TRUE)
    drv_major <- suppressWarnings(as.integer(sub("^([0-9]+).*", "\\1", drv[1])))
    
    # Prefer CUDA 13 if the driver is new enough; otherwise CUDA 12; else CPU fallback
    if (!is.na(drv_major) && drv_major >= 580) {
      msg("Driver %s detected (>=580): installing JAX CUDA 13 wheels.", drv[1])
      tryCatch(pip_install('jax[cuda13]'), error = function(e) {
        msg("CUDA 13 wheels failed (%s); falling back to CUDA 12.", e$message)
        pip_install('jax[cuda12]')
      })
    } else if (!is.na(drv_major) && drv_major >= 525) {
      msg("Driver %s detected (>=525,<580): installing JAX CUDA 12 wheels.", drv[1])
      pip_install('jax[cuda12]')
    } else {
      msg("Driver %s too old for CUDA wheels; installing CPU-only JAX.", drv[1])
      pip_install('jax')
    }
  }
  
  # (Optional) neutralize LD_LIBRARY_PATH inside this env to prevent overrides
  try({
    actdir <- file.path(Sys.getenv("HOME"), "miniconda3/envs", conda_env, "etc", "conda", "activate.d")
    dir.create(actdir, recursive = TRUE, showWarnings = FALSE)
    writeLines("unset LD_LIBRARY_PATH", file.path(actdir, "00-unset-ld.sh"))
  }, silent = TRUE)
  
  # Install JAX first (so later deps don't pull a CPU variant)
  install_jax_gpu()
  
  # --- (B) Now install the rest (pip’s default upgrade strategy is "only-if-needed") ---
  other_pkgs <- c("numpy", "equinox", "numpyro", "optax")
  pip_install(other_pkgs)
  
  msg("Environment '%s' is ready.", conda_env)
}


