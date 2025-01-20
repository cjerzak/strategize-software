#' A function to build the environment for strategize Builds a conda environment in which 'JAX' and 'np' are installed. Users can also create a conda environment where 'JAX' and 'np' are installed themselves. 
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

build_backend <- function(conda_env = "strategize", conda = "auto"){
  # Create a new conda environment
  reticulate::conda_create(envname = conda_env,
                           conda = conda,
                           python_version = "3.11")
  
  # Install Python packages within the environment
  Packages2Install <- c("numpy",
                        "tensorflow",
                        "tensorflow_probability",
                        "jax",
                        "jaxlib",
                        "equinox", 
                        "optax")

  for(pack_ in Packages2Install){
      try_ <- try(reticulate::py_install(pack_, conda = conda, pip = TRUE, envname = conda_env), TRUE)
  }
}
# build_backend("strategize")
