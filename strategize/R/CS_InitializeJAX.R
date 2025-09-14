initialize_jax <- function(conda_env = "strategize_env", 
                           conda_env_required = TRUE) {
  library(reticulate)
  # Load reticulate (Declared in Imports: in DESCRIPTION)
  browser()
  reticulate::use_condaenv(condaenv = conda_env, required = conda_env_required)
  
  # Import Python packages once, storing them in strenv
  strenv$jax <- reticulate::import("jax")
  strenv$jnp <- reticulate::import("jax.numpy")
  strenv$np  <- reticulate::import("numpy")
  # strenv$oryx  <- reticulate::import("tensorflow_probability.substrates.jax") # depreciated 
  strenv$py_gc  <- reticulate::import("gc")
  strenv$optax  <- reticulate::import("optax")
  
  # Disable 64-bit computations
  strenv$jax$config$update("jax_enable_x64", FALSE)
  strenv$jaxFloatType <- strenv$jnp$float32
  
  # setup numerical precision for delta method
  strenv$dtj <- strenv$jnp$float64; strenv$jax$config$update("jax_enable_x64", T) # use float64
  #strenv$dtj <- strenv$jnp$float32; strenv$jax$config$update("jax_enable_x64", F) # use float32
}
strenv <- new.env( parent = emptyenv() )

