% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/CS_BuildBackend.R
\name{build_backend}
\alias{build_backend}
\title{Build the environment for strategize}
\usage{
build_backend(conda_env = "strategize", conda = "auto")
}
\arguments{
\item{conda_env}{(default = \code{"strategize"}) Name of the conda environment in which to place the backends.}

\item{conda}{(default = \code{auto}) The path to a conda executable. Using \code{"auto"} allows reticulate to attempt to automatically find an appropriate conda binary.}
}
\value{
Invisibly returns NULL; this function is used for its side effects
of creating and configuring a conda environment for \code{strategize}.
This function requires an Internet connection.
You can find out a list of conda Python paths via: \code{Sys.which("python")}
}
\description{
Build the environment for \code{strategize}. Creates a conda environment where 'JAX' and 'np' are installed. Users may also create such an environment themselves.
}
\examples{
\dontrun{
# Create a conda environment named "strategize"
# and install the required Python packages (jax, numpy, etc.)
build_backend(conda_env = "strategize", conda = "auto")

# If you want to specify a particular conda path:
# build_backend(conda_env = "strategize", conda = "/usr/local/bin/conda")
}

}
