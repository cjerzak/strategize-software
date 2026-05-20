\name{build_text_embedding_backend}
\alias{build_text_embedding_backend}
\alias{inspect_text_embedding_backend}
\title{Inspect or build a host-aware text embedding backend}
\usage{
build_text_embedding_backend(
  conda_env = "strategize_env",
  family = "qwen3",
  runtime = "auto",
  profile = "portable",
  cache_dir = NULL,
  required = TRUE,
  conda = "auto"
)

inspect_text_embedding_backend(
  conda_env = "strategize_env",
  family = "qwen3",
  runtime = "auto",
  profile = "portable",
  conda = "auto"
)
}
\arguments{
\item{conda_env}{Conda env name used for the Python runtime.}

\item{family}{Embedding-family selector. Currently only \code{"qwen3"} is supported.}

\item{runtime}{Runtime preference. Use \code{"auto"} to resolve per host, or choose one of \code{"mlx"}, \code{"cuda"}, \code{"rocm"}, or \code{"cpu"} explicitly.}

\item{profile}{Embedding profile. Currently only \code{"portable"} is supported.}

\item{cache_dir}{Optional directory for cached text embeddings.}

\item{required}{Logical; if \code{TRUE}, ensure the selected runtime is usable immediately. If \code{FALSE}, return a lazy backend that validates on first use.}

\item{conda}{Conda binary to use. Defaults to \code{"auto"}.}
}
\value{
\code{inspect_text_embedding_backend()} returns a serializable list describing the host, Python env health, candidate runtimes, and the selected backend.

\code{build_text_embedding_backend()} returns a list with \code{$fn}, \code{$spec}, and \code{$runtime}.
}
\description{
Resolve a text embedding backend that is compatible with the current machine while preserving a portable Qwen-family embedding surface across Mac and Linux hosts.
}
\details{
The current implementation keeps the Qwen embedding family but selects a different runtime per host:
\itemize{
\item Apple Silicon defaults to MLX with \code{mlx-community/Qwen3-Embedding-8B-mxfp8}.
\item Linux prefers a CUDA-capable \code{sentence-transformers} path on validated Nvidia hosts.
\item Linux prefers a ROCm-capable \code{sentence-transformers} path when the Python runtime validates it.
\item Other hosts fall back to a CPU \code{sentence-transformers} path.
}

All runtimes are normalized to a canonical 1024-dimensional embedding width so saved artifacts remain portable across hosts.
}
\examples{
\dontrun{
backend <- build_text_embedding_backend(conda_env = "strategize_env")

study_a$experiment_description <- "Source study A"
study_b$experiment_description <- "Source study B"

foundation_fit <- preference.fm::fit_conjoint_foundation_model(
  experiments = list(study_a, study_b),
  foundation_control = list(
    text_embedding_fn = backend$fn
  )
)

inspect_text_embedding_backend(conda_env = "strategize_env")
}
}
