initialize_jax <- function(conda_env = "strategize_env", 
                           conda_env_required = TRUE) {
  library(reticulate)
  # Load reticulate (Declared in Imports: in DESCRIPTION)
  reticulate::use_condaenv(condaenv = conda_env, required = conda_env_required)
  
  # Import Python packages once, storing them in strenv
  strenv$jax <- reticulate::import("jax")
  strenv$jnp <- reticulate::import("jax.numpy")
  strenv$np  <- reticulate::import("numpy")
  strenv$py_gc  <- reticulate::import("gc")
  strenv$numpyro  <- reticulate::import("numpyro")
  strenv$optax  <- reticulate::import("optax")
  
  # setup numerical precisions
  strenv$jaxFloatType <- strenv$jnp$float32
  #strenv$dtj <- strenv$jnp$float64; strenv$jax$config$update("jax_enable_x64", TRUE) # use float64
  strenv$dtj <- strenv$jnp$float32; strenv$jax$config$update("jax_enable_x64", FALSE) # use float32
}
strenv <- new.env( parent = emptyenv() )

