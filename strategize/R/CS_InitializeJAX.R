initialize_jax <- function(conda_env = "strategize", 
                           conda_env_required = TRUE) {
  # Load reticulate (Declared in Imports: in DESCRIPTION)
  reticulate::use_condaenv(condaenv = conda_env, required = conda_env_required)
  
  # Import Python packages once, storing them in strenv
  if (!exists("jax", envir = strenv, inherits = FALSE)) {
    strenv$jax <- reticulate::import("jax")
    strenv$jnp <- reticulate::import("jax.numpy")
    strenv$np  <- reticulate::import("numpy")
    strenv$oryx  <- reticulate::import("tensorflow_probability.substrates.jax")
    strenv$py_gc  <- reticulate::import("py_gc")
    strenv$optax  <- reticulate::import("optax")
  }
  
  # Disable 64-bit computations
  strenv$jax$config$update("jax_enable_x64", FALSE)
  strenv$jaxFloatType <- strenv$jnp$float32
  
  # Setup core JAX functions and store them in strenv
  {
    strenv$InsertOnes <- strenv$jax$jit( function(treat_indices_, zeros_){
      zeros_ <- zeros_$at[treat_indices_]$add(1L)
      return(  zeros_ )
    } )
  }
}
