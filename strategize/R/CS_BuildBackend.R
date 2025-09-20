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
  # Create the conda environment with Python 3.12
  reticulate::conda_create(envname = conda_env, conda = conda, python_version = "3.12")
  
  os <- Sys.info()[["sysname"]]
  msg <- function(...) message(sprintf(...))
  
  # Helper: install packages with pip into our env; return TRUE on success, FALSE on error
  pip_install <- function(pkgs) {
    tryCatch({
      reticulate::py_install(
        packages = pkgs,
        envname  = conda_env,
        conda    = conda,
        pip      = TRUE
      )
      TRUE
    }, error = function(e) {
      msg("  -> Install failed for: %s\n     %s", paste(pkgs, collapse = ", "), e$message)
      FALSE
    })
  }
  
  # --- Ecosystem packages ---
  other_pkgs <- c(
    "numpy",
    "equinox",
    "numpyro",
    "optax"
  )
  invisible(pip_install(other_pkgs))
  
  # --- Install JAX (GPU/METAL first, else CPU fallback) ---
  jax_installed <- FALSE
  
  # --- Install JAX (GPU first, else CPU fallback) ---
  jax_installed <- FALSE
  
  if (identical(os, "Linux")) {
    msg("Detected Linux: checking NVIDIA driver for CUDA support...")
    
    # Run nvidia-smi and capture output
    nvidia_output <- suppressWarnings(system("nvidia-smi", intern = TRUE))
    
    if (length(nvidia_output) == 0 || attr(nvidia_output, "status") != 0) {
      msg("No NVIDIA GPU or nvidia-smi unavailable; falling back to CPU-only JAX.")
      jax_installed <- pip_install("jax")
    } else {
      # Parse the CUDA Version line (e.g., "| NVIDIA-SMI ... CUDA Version: 12.6 |")
      cuda_line <- nvidia_output[grep("CUDA Version", nvidia_output)]
      if (length(cuda_line) > 0) {
        cuda_ver_str <- sub(".*CUDA Version: ([0-9.]+).*", "\\1", cuda_line)
        cuda_major <- as.numeric(sub("([0-9]+)\\..*", "\\1", cuda_ver_str))
        
        if (is.na(cuda_major)) {
          msg("Could not parse CUDA version from nvidia-smi; falling back to CPU-only JAX.")
          jax_installed <- pip_install("jax")
        } else if (cuda_major >= 13) {
          msg("CUDA driver supports >= 13.x; installing JAX with CUDA 13 support.")
          jax_installed <- pip_install("jax[cuda13]")
        } else if (cuda_major >= 12) {
          msg("CUDA driver supports 12.x; installing JAX with CUDA 12 support.")
          jax_installed <- pip_install("jax[cuda12]")
        } else {
          msg("CUDA driver supports < 12.x; falling back to CPU-only JAX.")
          jax_installed <- pip_install("jax")
        }
      } else {
        msg("Could not find CUDA version in nvidia-smi output; falling back to CPU-only JAX.")
        jax_installed <- pip_install("jax")
      }
    }
  } 
  if (identical(os, "Darwin")) {
    msg("Detected macOS: attempting Metal-enabled JAX via jax-metal (pinned to JAX 0.5.0).")
    # As noted, jax-metal currently works with JAX==0.5.0; pin both jax and jaxlib.
    #jax_installed <- pip_install(c("jax==0.5.0", "jaxlib==0.5.0", "jax-metal"))
    jax_installed <- FALSE
    if (!jax_installed) {
      msg("  Metal install failed; falling back to CPU-only JAX.")
      jax_installed <- pip_install("jax")
    }
  }
  if (!(os %in% c("Darwin","Linux")) ){ 
    msg("Non-Linux/macOS detected (%s); installing CPU-only JAX.", os)
    jax_installed <- pip_install("jax")
  }
  
  if (!jax_installed) stop("Failed to install JAX in any configuration.", call. = FALSE)

  msg("Environment '%s' is ready.", conda_env)
}

