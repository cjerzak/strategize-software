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
  
  if (identical(os, "Linux")) {
    msg("Detected Linux: attempting CUDA-enabled JAX.")
    # Try CUDA 13 first (newer), then CUDA 12; finally fall back to CPU JAX
    jax_installed <- pip_install("jax[cuda13]")
    if (!jax_installed) {
      msg("  CUDA 13 install failed; trying CUDA 12 wheels …")
      jax_installed <- pip_install("jax[cuda12]")
    }
    if (!jax_installed) {
      msg("  CUDA wheels unavailable; falling back to CPU-only JAX.")
      jax_installed <- pip_install("jax")
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


build_backend_OLD <- function(conda_env = "strategize_env", conda = "auto"){
  # Create a new conda environment
  reticulate::conda_create(envname = conda_env,
                           conda = conda,
                           python_version = "3.12")
  
  # Install Python packages within the environment
  Packages2Install <- c("numpy",
                        #"tensorflow",
                        #"tensorflow_probability",  depreciated 
                        "jax",
                        "jaxlib",
                        "equinox", 
                        "numpyro", 
                        "optax")
  reticulate::py_install(Packages2Install, conda = conda, pip = TRUE, envname = conda_env)
}
# build_backend("strategize")
