#' Inspect a host-aware text-embedding backend
#'
#' @param conda_env Conda env name used for the Python runtime.
#' @param family Embedding-family selector. Currently only \code{"qwen3"} is supported.
#' @param runtime Runtime preference. Use \code{"auto"} to resolve per host, or
#'   choose one of \code{"mlx"}, \code{"cuda"}, \code{"rocm"}, or \code{"cpu"} explicitly.
#' @param profile Embedding profile. Currently only \code{"portable"} is supported.
#' @param conda Conda binary to use. Defaults to \code{"auto"}.
#'
#' @return A serializable list describing the host, Python env health, candidate
#'   text-embedding runtimes, and the selected backend.
#' @export
inspect_text_embedding_backend <- function(conda_env = "strategize_env",
                                           family = "qwen3",
                                           runtime = "auto",
                                           profile = "portable",
                                           conda = "auto") {
  family <- cs2step_text_embedding_family(family)
  runtime <- cs2step_text_embedding_runtime(runtime)
  profile <- cs2step_text_embedding_profile(profile)
  host <- cs2step_collect_text_embedding_host_info(conda_env = conda_env, conda = conda)
  inspected <- cs2step_inspect_text_embedding_candidates(
    host = host,
    family = family,
    runtime = runtime,
    profile = profile
  )
  structure(inspected, class = "strategize_text_embedding_backend")
}

#' Build a host-aware text-embedding backend
#'
#' @param conda_env Conda env name used for the Python runtime.
#' @param family Embedding-family selector. Currently only \code{"qwen3"} is supported.
#' @param runtime Runtime preference. Use \code{"auto"} to resolve per host, or
#'   choose one of \code{"mlx"}, \code{"cuda"}, \code{"rocm"}, or \code{"cpu"} explicitly.
#' @param profile Embedding profile. Currently only \code{"portable"} is supported.
#' @param cache_dir Optional directory for cached text embeddings.
#' @param required Logical; if \code{TRUE}, ensure the selected runtime is usable
#'   immediately. If \code{FALSE}, return a lazy backend that validates on first use.
#' @param conda Conda binary to use. Defaults to \code{"auto"}.
#'
#' @return A list with \code{$fn}, \code{$spec}, and \code{$runtime}.
#' @export
build_text_embedding_backend <- function(conda_env = "strategize_env",
                                         family = "qwen3",
                                         runtime = "auto",
                                         profile = "portable",
                                         cache_dir = NULL,
                                         required = TRUE,
                                         conda = "auto") {
  family <- cs2step_text_embedding_family(family)
  runtime <- cs2step_text_embedding_runtime(runtime)
  profile <- cs2step_text_embedding_profile(profile)

  env_state <- cs2step_backend_env_state(conda_env = conda_env, conda = conda)
  if (isTRUE(required) && !isTRUE(env_state$core_modules_ready)) {
    build_backend(conda_env = conda_env, conda = conda)
  }

  inspected <- inspect_text_embedding_backend(
    conda_env = conda_env,
    family = family,
    runtime = runtime,
    profile = profile,
    conda = conda
  )
  spec <- inspected$selected

  if (is.null(spec) || identical(spec$status, "unavailable")) {
    stop(
      sprintf(
        "No compatible text embedding backend is available for family '%s' and runtime '%s'.\n%s",
        family,
        runtime,
        paste(unique(inspected$issues %||% character(0)), collapse = "\n")
      ),
      call. = FALSE
    )
  }

  if (isTRUE(required)) {
    cs2step_ensure_text_embedding_runtime(spec)
    inspected <- inspect_text_embedding_backend(
      conda_env = conda_env,
      family = family,
      runtime = runtime,
      profile = profile,
      conda = conda
    )
    spec <- inspected$selected
    if (is.null(spec) || !identical(spec$status, "ready")) {
      label_use <- if (is.null(spec)) "unknown" else spec$label %||% "unknown"
      stop(
        sprintf(
          "Selected text embedding backend '%s' is not ready after installation.\n%s",
          label_use,
          paste(unique(inspected$issues %||% character(0)), collapse = "\n")
        ),
        call. = FALSE
      )
    }
  }

  fn <- cs2step_build_text_embedding_fn(spec = spec, cache_dir = cache_dir)
  list(fn = fn, spec = spec, runtime = inspected)
}

cs2step_text_embedding_family <- function(family) {
  family <- tolower(as.character(family %||% "qwen3"))
  if (!identical(family, "qwen3")) {
    stop("Only family = 'qwen3' is currently supported.", call. = FALSE)
  }
  family
}

cs2step_text_embedding_runtime <- function(runtime) {
  runtime <- tolower(as.character(runtime %||% "auto"))
  valid <- c("auto", "mlx", "cuda", "rocm", "cpu")
  if (!runtime %in% valid) {
    stop(
      sprintf("Unsupported text embedding runtime '%s'. Valid values: %s.",
              runtime, paste(valid, collapse = ", ")),
      call. = FALSE
    )
  }
  runtime
}

cs2step_text_embedding_profile <- function(profile) {
  profile <- tolower(as.character(profile %||% "portable"))
  if (!identical(profile, "portable")) {
    stop("Only profile = 'portable' is currently supported.", call. = FALSE)
  }
  profile
}

cs2step_text_embedding_backend_version <- function() 1L

cs2step_resolve_conda_binary <- function(conda = "auto") {
  if (!cs2step_has_reticulate()) {
    return(NULL)
  }
  conda_use <- conda %||% "auto"
  conda_bin <- tryCatch(reticulate::conda_binary(conda = conda_use), error = function(e) NULL)
  if (!nzchar(conda_bin %||% "")) {
    env_bin <- Sys.getenv("RETICULATE_CONDA", unset = "")
    if (nzchar(env_bin)) {
      conda_bin <- path.expand(env_bin)
    }
  }
  if (!nzchar(conda_bin %||% "")) {
    path_bin <- Sys.which("conda")
    if (nzchar(path_bin)) {
      conda_bin <- path_bin
    }
  }
  if (!nzchar(conda_bin %||% "")) {
    return(NULL)
  }
  path.expand(conda_bin)
}

cs2step_python_probe <- function(python, code, env = character()) {
  python <- path.expand(as.character(python %||% ""))
  if (!nzchar(python) || !file.exists(python)) {
    return(list(status = 127L, output = "python interpreter not found"))
  }
  script <- tempfile(pattern = "strategize-python-probe-", fileext = ".py")
  on.exit(unlink(script), add = TRUE)
  writeLines(code, script, useBytes = TRUE)
  output <- tryCatch(
    suppressWarnings(system2(
      python,
      script,
      stdout = TRUE,
      stderr = TRUE,
      env = as.character(env %||% character())
    )),
    error = function(e) structure(conditionMessage(e), status = 127L)
  )
  status <- attr(output, "status")
  if (is.null(status)) {
    status <- 0L
  }
  list(status = as.integer(status), output = as.character(output))
}

cs2step_python_version_major_minor <- function(python) {
  code <- paste(
    "import sys",
    "print('PYTHON_VERSION::{}.{}'.format(sys.version_info.major, sys.version_info.minor))",
    sep = "\n"
  )
  probe <- cs2step_python_probe(python, code)
  line <- grep("^PYTHON_VERSION::", probe$output, value = TRUE)
  version <- if (length(line) > 0L) sub("^PYTHON_VERSION::", "", line[[1]]) else ""
  list(
    ok = identical(probe$status, 0L) && nzchar(version),
    major_minor = version,
    status = probe$status,
    output = probe$output
  )
}

cs2step_python_distribution_probe <- function(python, distributions) {
  distributions <- unique(as.character(distributions %||% character(0)))
  if (!length(distributions)) {
    return(list(ok = logical(0), version = character(0), details = character(0), status = 0L))
  }
  quoted <- paste(sprintf("'%s'", gsub("'", "\\\\'", distributions, fixed = TRUE)), collapse = ", ")
  code <- paste(
    "from importlib import metadata",
    sprintf("dists = [%s]", quoted),
    "for name in dists:",
    "    try:",
    "        print('OK::' + name + '::' + metadata.version(name))",
    "    except Exception as exc:",
    "        print('FAIL::' + name + '::' + exc.__class__.__name__ + '::' + str(exc))",
    sep = "\n"
  )
  probe <- cs2step_python_probe(python, code)
  ok <- setNames(rep(FALSE, length(distributions)), distributions)
  version <- setNames(rep("", length(distributions)), distributions)
  details <- setNames(rep("", length(distributions)), distributions)
  for (line in probe$output) {
    if (!nzchar(line)) {
      next
    }
    if (startsWith(line, "OK::")) {
      parts <- strsplit(line, "::", fixed = TRUE)[[1]]
      if (length(parts) >= 3L && parts[[2]] %in% distributions) {
        ok[[parts[[2]]]] <- TRUE
        version[[parts[[2]]]] <- paste(parts[-(1:2)], collapse = "::")
      }
    } else if (startsWith(line, "FAIL::")) {
      parts <- strsplit(line, "::", fixed = TRUE)[[1]]
      if (length(parts) >= 2L && parts[[2]] %in% distributions) {
        details[[parts[[2]]]] <- paste(parts[-(1:2)], collapse = "::")
      }
    }
  }
  list(ok = ok, version = version, details = details, status = probe$status)
}

cs2step_python_jax_backend_probe <- function(python, platform = NULL) {
  code <- paste(
    "import jax",
    "print('JAX_DEFAULT_BACKEND::' + str(jax.default_backend()))",
    sep = "\n"
  )
  platform <- as.character(platform %||% "")
  env <- if (nzchar(platform)) sprintf("JAX_PLATFORMS=%s", platform) else character()
  probe <- cs2step_python_probe(python, code, env = env)
  line <- grep("^JAX_DEFAULT_BACKEND::", probe$output, value = TRUE)
  backend <- if (length(line) > 0L) sub("^JAX_DEFAULT_BACKEND::", "", line[[1]]) else ""
  ok <- identical(probe$status, 0L) && nzchar(backend)
  if (nzchar(platform)) {
    ok <- ok && identical(backend, platform)
  }
  list(ok = ok, backend = backend, status = probe$status, output = probe$output)
}

cs2step_python_module_probe <- function(python, modules) {
  modules <- unique(as.character(modules %||% character(0)))
  if (!length(modules)) {
    return(list(ok = logical(0), details = character(0), status = 0L))
  }
  quoted <- paste(sprintf("'%s'", gsub("'", "\\\\'", modules, fixed = TRUE)), collapse = ", ")
  code <- paste(
    "import importlib",
    sprintf("mods = [%s]", quoted),
    "for name in mods:",
    "    try:",
    "        importlib.import_module(name)",
    "        print('OK::' + name)",
    "    except Exception as exc:",
    "        print('FAIL::' + name + '::' + exc.__class__.__name__ + '::' + str(exc))",
    sep = "\n"
  )
  probe <- cs2step_python_probe(python, code)
  ok <- setNames(rep(FALSE, length(modules)), modules)
  details <- setNames(rep("", length(modules)), modules)
  for (line in probe$output) {
    if (!nzchar(line)) {
      next
    }
    if (startsWith(line, "OK::")) {
      mod <- sub("^OK::", "", line)
      if (mod %in% modules) {
        ok[[mod]] <- TRUE
      }
    } else if (startsWith(line, "FAIL::")) {
      parts <- strsplit(line, "::", fixed = TRUE)[[1]]
      if (length(parts) >= 2L && parts[[2]] %in% modules) {
        details[[parts[[2]]]] <- paste(parts[-(1:2)], collapse = "::")
      }
    }
  }
  list(ok = ok, details = details, status = probe$status)
}

cs2step_backend_core_modules <- function() {
  modules <- c("jax", "numpyro", "optax", "equinox", "numpy", "orbax.checkpoint")
  if (isTRUE(cs2step_backend_host_info()$is_macos)) {
    modules <- c(modules, "mlx_embeddings", "mlx.core")
  }
  modules
}

cs2step_backend_core_pip_packages <- function() {
  packages <- c(
    jax = "jax",
    numpyro = "numpyro",
    optax = "optax",
    equinox = "equinox",
    numpy = "numpy",
    "orbax.checkpoint" = "orbax-checkpoint"
  )
  if (isTRUE(cs2step_backend_host_info()$is_macos)) {
    packages <- c(packages, mlx_embeddings = "mlx-embeddings")
  }
  packages
}

cs2step_backend_env_state <- function(conda_env = "strategize_env", conda = "auto") {
  conda_bin <- cs2step_resolve_conda_binary(conda)
  envs <- tryCatch(
    reticulate::conda_list(conda = conda_bin %||% conda),
    error = function(e) NULL
  )
  registered <- FALSE
  python <- ""
  if (!is.null(envs) && nrow(envs) > 0L && "name" %in% names(envs)) {
    idx <- match(conda_env, as.character(envs$name))
    if (!is.na(idx)) {
      registered <- TRUE
      if ("python" %in% names(envs)) {
        python <- as.character(envs$python[[idx]] %||% "")
      }
    }
  }
  python <- path.expand(python)
  python_exists <- nzchar(python) && file.exists(python)
  module_probe <- if (python_exists) {
    cs2step_python_module_probe(python, cs2step_backend_core_modules())
  } else {
    list(
      ok = setNames(rep(FALSE, length(cs2step_backend_core_modules())), cs2step_backend_core_modules()),
      details = setNames(rep("", length(cs2step_backend_core_modules())), cs2step_backend_core_modules()),
      status = 127L
    )
  }
  list(
    conda = conda_bin,
    conda_env = conda_env,
    registered = registered,
    python = python,
    python_exists = python_exists,
    core_module_status = module_probe$ok,
    core_module_details = module_probe$details,
    core_modules_ready = python_exists && all(module_probe$ok)
  )
}

cs2step_backend_host_info <- function() {
  info <- Sys.info()
  os <- as.character(unname(info["sysname"]))
  if (!nzchar(os) || is.na(os)) {
    os <- .Platform$OS.type
  }
  machine <- as.character(unname(info["machine"]))
  if (!nzchar(machine) || is.na(machine)) {
    machine <- R.version$arch %||% ""
  }
  list(
    os = os,
    machine = machine,
    is_macos = identical(os, "Darwin"),
    is_arm64 = grepl("arm64|aarch64", machine, ignore.case = TRUE)
  )
}

cs2step_backend_mps_compatibility <- function(state) {
  python <- path.expand(as.character(state$python %||% ""))
  python_exists <- isTRUE(state$python_exists) && nzchar(python) && file.exists(python)
  if (!python_exists) {
    return(list(
      compatible = FALSE,
      python_major_minor = "",
      python_313 = FALSE,
      jax_mps_installed = FALSE,
      jax_mps_version = "",
      jax_backend = "",
      jax_backend_mps = FALSE,
      details = "python interpreter not found"
    ))
  }

  python_version <- cs2step_python_version_major_minor(python)
  python_313 <- identical(python_version$major_minor, "3.13")
  dist_probe <- cs2step_python_distribution_probe(python, "jax-mps")
  jax_mps_installed <- isTRUE(dist_probe$ok[["jax-mps"]])
  jax_backend_probe <- if (python_313 && jax_mps_installed) {
    cs2step_python_jax_backend_probe(python, platform = "mps")
  } else {
    list(ok = FALSE, backend = "", status = NA_integer_, output = character())
  }

  list(
    compatible = python_313 && jax_mps_installed && isTRUE(jax_backend_probe$ok),
    python_major_minor = python_version$major_minor,
    python_313 = python_313,
    jax_mps_installed = jax_mps_installed,
    jax_mps_version = dist_probe$version[["jax-mps"]] %||% "",
    jax_backend = jax_backend_probe$backend,
    jax_backend_mps = isTRUE(jax_backend_probe$ok),
    details = paste(c(python_version$output, dist_probe$details, jax_backend_probe$output), collapse = "\n")
  )
}

cs2step_describe_mps_compatibility <- function(compatibility) {
  issues <- character()
  if (!isTRUE(compatibility$python_313)) {
    found <- compatibility$python_major_minor %||% ""
    if (!nzchar(found)) {
      found <- "unknown"
    }
    issues <- c(issues, sprintf("Python 3.13 required, found %s", found))
  }
  if (!isTRUE(compatibility$jax_mps_installed)) {
    issues <- c(issues, "Python distribution 'jax-mps' is not installed")
  }
  if (!isTRUE(compatibility$jax_backend_mps)) {
    backend <- compatibility$jax_backend %||% ""
    if (!nzchar(backend)) {
      backend <- "unavailable"
    }
    issues <- c(issues, sprintf("JAX default backend under JAX_PLATFORMS=mps is %s", backend))
  }
  if (!length(issues)) {
    return("compatible")
  }
  paste(issues, collapse = "; ")
}

cs2step_command_probe <- function(command, args = character()) {
  output <- tryCatch(
    suppressWarnings(system2(command, args = args, stdout = TRUE, stderr = TRUE)),
    error = function(e) structure(conditionMessage(e), status = 127L)
  )
  status <- attr(output, "status")
  if (is.null(status)) {
    status <- 0L
  }
  list(status = as.integer(status), output = as.character(output))
}

cs2step_run_command_available <- function(cmd) {
  nzchar(Sys.which(cmd))
}

cs2step_collect_nvidia_tools <- function() {
  list(
    nvidia_smi = cs2step_run_command_available("nvidia-smi"),
    nvcc = cs2step_run_command_available("nvcc")
  )
}

cs2step_probe_nvidia_driver <- function() {
  if (!isTRUE(cs2step_run_command_available("nvidia-smi"))) {
    return(list(available = FALSE, driver_version = "", driver_major = NA_integer_, device_name = ""))
  }
  probe <- cs2step_command_probe(
    "nvidia-smi",
    c("--query-gpu=driver_version,name", "--format=csv,noheader")
  )
  line <- probe$output[[1]] %||% ""
  if (!nzchar(line)) {
    return(list(available = FALSE, driver_version = "", driver_major = NA_integer_, device_name = ""))
  }
  parts <- strsplit(line, ",", fixed = TRUE)[[1]]
  driver_version <- trimws(parts[[1]] %||% "")
  driver_major <- suppressWarnings(as.integer(sub("^([0-9]+).*", "\\1", driver_version)))
  device_name <- if (length(parts) >= 2L) trimws(paste(parts[-1], collapse = ",")) else ""
  list(
    available = identical(probe$status, 0L) && nzchar(driver_version),
    driver_version = driver_version,
    driver_major = driver_major,
    device_name = device_name
  )
}

cs2step_cuda_torch_wheel <- function(driver_major) {
  if (is.na(driver_major)) {
    return(NULL)
  }
  if (driver_major >= 580L) {
    return(list(label = "cu130", index_url = "https://download.pytorch.org/whl/cu130"))
  }
  if (driver_major >= 525L) {
    return(list(label = "cu128", index_url = "https://download.pytorch.org/whl/cu128"))
  }
  NULL
}

cs2step_collect_rocm_tools <- function() {
  list(
    rocminfo = cs2step_run_command_available("rocminfo"),
    hipcc = cs2step_run_command_available("hipcc"),
    rocm_smi = cs2step_run_command_available("rocm-smi"),
    rocm_root = dir.exists("/opt/rocm")
  )
}

cs2step_probe_rocm_runtime <- function(python) {
  if (!nzchar(python %||% "") || !file.exists(python)) {
    return(list(validated = FALSE, cuda_available = FALSE, hip_version = "", device_name = ""))
  }
  code <- paste(
    "try:",
    "    import torch",
    "    hip = getattr(getattr(torch, 'version', None), 'hip', None)",
    "    avail = bool(torch.cuda.is_available())",
    "    print('TORCH_OK')",
    "    print('CUDA_AVAILABLE::' + ('1' if avail else '0'))",
    "    print('HIP_VERSION::' + (str(hip) if hip is not None else ''))",
    "    if avail:",
    "        try:",
    "            print('DEVICE_NAME::' + str(torch.cuda.get_device_name(0)))",
    "        except Exception as exc:",
    "            print('DEVICE_NAME::' + exc.__class__.__name__)",
    "except Exception as exc:",
    "    print('TORCH_FAIL::' + exc.__class__.__name__ + '::' + str(exc))",
    sep = "\n"
  )
  probe <- cs2step_python_probe(python, code)
  lines <- probe$output %||% character(0)
  torch_ok <- any(startsWith(lines, "TORCH_OK"))
  cuda_available <- any(lines == "CUDA_AVAILABLE::1")
  hip_line <- lines[startsWith(lines, "HIP_VERSION::")]
  device_line <- lines[startsWith(lines, "DEVICE_NAME::")]
  hip_version <- if (length(hip_line)) sub("^HIP_VERSION::", "", hip_line[[1]]) else ""
  device_name <- if (length(device_line)) sub("^DEVICE_NAME::", "", device_line[[1]]) else ""
  list(
    validated = isTRUE(torch_ok) && isTRUE(cuda_available) && nzchar(hip_version),
    cuda_available = cuda_available,
    hip_version = hip_version,
    device_name = device_name
  )
}

cs2step_probe_cuda_runtime <- function(python) {
  if (!nzchar(python %||% "") || !file.exists(python)) {
    return(list(validated = FALSE, cuda_available = FALSE, cuda_version = "", hip_version = "", device_name = ""))
  }
  code <- paste(
    "try:",
    "    import torch",
    "    ver = getattr(torch, 'version', None)",
    "    cuda = getattr(ver, 'cuda', None)",
    "    hip = getattr(ver, 'hip', None)",
    "    avail = bool(torch.cuda.is_available())",
    "    print('TORCH_OK')",
    "    print('CUDA_AVAILABLE::' + ('1' if avail else '0'))",
    "    print('CUDA_VERSION::' + (str(cuda) if cuda is not None else ''))",
    "    print('HIP_VERSION::' + (str(hip) if hip is not None else ''))",
    "    if avail:",
    "        try:",
    "            print('DEVICE_NAME::' + str(torch.cuda.get_device_name(0)))",
    "        except Exception as exc:",
    "            print('DEVICE_NAME::' + exc.__class__.__name__)",
    "except Exception as exc:",
    "    print('TORCH_FAIL::' + exc.__class__.__name__ + '::' + str(exc))",
    sep = "\n"
  )
  probe <- cs2step_python_probe(python, code)
  lines <- probe$output %||% character(0)
  torch_ok <- any(startsWith(lines, "TORCH_OK"))
  cuda_available <- any(lines == "CUDA_AVAILABLE::1")
  cuda_line <- lines[startsWith(lines, "CUDA_VERSION::")]
  hip_line <- lines[startsWith(lines, "HIP_VERSION::")]
  device_line <- lines[startsWith(lines, "DEVICE_NAME::")]
  cuda_version <- if (length(cuda_line)) sub("^CUDA_VERSION::", "", cuda_line[[1]]) else ""
  hip_version <- if (length(hip_line)) sub("^HIP_VERSION::", "", hip_line[[1]]) else ""
  device_name <- if (length(device_line)) sub("^DEVICE_NAME::", "", device_line[[1]]) else ""
  list(
    validated = isTRUE(torch_ok) && isTRUE(cuda_available) && nzchar(cuda_version) && !nzchar(hip_version),
    cuda_available = cuda_available,
    cuda_version = cuda_version,
    hip_version = hip_version,
    device_name = device_name
  )
}

cs2step_collect_text_embedding_host_info <- function(conda_env = "strategize_env", conda = "auto") {
  env_state <- cs2step_backend_env_state(conda_env = conda_env, conda = conda)
  os_name <- as.character(Sys.info()[["sysname"]] %||% .Platform$OS.type)
  machine <- as.character(Sys.info()[["machine"]] %||% R.version$arch)
  nvidia_tools <- cs2step_collect_nvidia_tools()
  nvidia_driver <- cs2step_probe_nvidia_driver()
  cuda_runtime <- cs2step_probe_cuda_runtime(env_state$python)
  rocm_tools <- cs2step_collect_rocm_tools()
  rocm_runtime <- cs2step_probe_rocm_runtime(env_state$python)
  list(
    os = os_name,
    machine = machine,
    conda = env_state$conda,
    conda_env = conda_env,
    conda_registered = env_state$registered,
    python = env_state$python,
    python_exists = env_state$python_exists,
    core_modules_ready = env_state$core_modules_ready,
    core_module_status = env_state$core_module_status,
    core_module_details = env_state$core_module_details,
    mlx_host_capable = identical(os_name, "Darwin") &&
      grepl("arm|aarch64", machine, ignore.case = TRUE),
    nvidia_tools = nvidia_tools,
    nvidia_driver = nvidia_driver,
    cuda_runtime = cuda_runtime,
    rocm_tools = rocm_tools,
    rocm_runtime = rocm_runtime
  )
}

cs2step_text_embedding_candidate <- function(label,
                                             backend,
                                             device,
                                             model_id,
                                             conda_env,
                                             conda,
                                             canonical_dim = 1024L,
                                             raw_dim = canonical_dim,
                                             required_modules = character(0),
                                             host_constraints = list()) {
  list(
    version = cs2step_text_embedding_backend_version(),
    label = label,
    family = "qwen3",
    profile = "portable",
    backend = backend,
    device = device,
    model_id = model_id,
    conda_env = conda_env,
    conda = conda,
    canonical_dim = as.integer(canonical_dim),
    raw_dim = as.integer(raw_dim),
    required_modules = as.character(required_modules),
    cache_key_version = 1L,
    host_constraints = host_constraints
  )
}

cs2step_evaluate_text_embedding_candidate <- function(candidate, host) {
  issues <- character(0)
  status <- "ready"
  installable <- FALSE

  if (!isTRUE(host$conda_registered)) {
    status <- "unavailable"
    issues <- c(issues, sprintf("Conda env '%s' is not registered.", candidate$conda_env))
  }
  if (!isTRUE(host$python_exists)) {
    status <- "unavailable"
    issues <- c(issues, sprintf("Python interpreter for env '%s' is missing.", candidate$conda_env))
  }
  if (!isTRUE(host$core_modules_ready)) {
    status <- "unavailable"
    issues <- c(
      issues,
      sprintf("Core strategize backend modules are not ready in '%s'. Run build_backend() first.", candidate$conda_env)
    )
  }

  if (identical(candidate$backend, "mlx")) {
    if (!isTRUE(host$mlx_host_capable)) {
      status <- "unavailable"
      issues <- c(issues, "MLX is only supported on Apple Silicon hosts.")
    } else {
      installable <- TRUE
    }
  } else if (identical(candidate$backend, "sentence_transformers")) {
    installable <- identical(candidate$device, "cpu")
    if (identical(candidate$device, "cuda")) {
      if (!identical(host$os, "Linux")) {
        status <- "unavailable"
        issues <- c(issues, "CUDA text embeddings are currently supported only on Linux hosts.")
      } else if (!isTRUE(any(unlist(host$nvidia_tools)))) {
        status <- "unavailable"
        issues <- c(issues, "Nvidia tooling is not present on this host.")
      } else if (!isTRUE(host$nvidia_driver$available)) {
        status <- "unavailable"
        issues <- c(issues, "Nvidia tooling is present but the GPU driver could not be resolved.")
      } else if (is.null(cs2step_cuda_torch_wheel(host$nvidia_driver$driver_major))) {
        status <- "unavailable"
        issues <- c(
          issues,
          sprintf(
            "Nvidia driver '%s' is too old for the supported CUDA torch wheels.",
            host$nvidia_driver$driver_version %||% "unknown"
          )
        )
      } else {
        installable <- TRUE
      }
    } else if (identical(candidate$device, "rocm")) {
      if (!isTRUE(any(unlist(host$rocm_tools)))) {
        status <- "unavailable"
        issues <- c(issues, "ROCm tooling is not present on this host.")
      } else if (!isTRUE(host$rocm_runtime$validated)) {
        status <- "unavailable"
        issues <- c(issues, "ROCm tooling is present but Python torch ROCm validation did not succeed.")
      }
    }
  }

  module_probe <- if (identical(status, "unavailable")) {
    list(ok = setNames(rep(FALSE, length(candidate$required_modules)), candidate$required_modules))
  } else {
    cs2step_python_module_probe(host$python, candidate$required_modules)
  }

  if (length(candidate$required_modules)) {
    missing <- names(module_probe$ok)[!module_probe$ok]
    if (length(missing)) {
      if (installable) {
        status <- "needs_install"
      } else {
        status <- "unavailable"
      }
      issues <- c(
        issues,
        sprintf(
          "%s runtime is missing Python modules: %s.",
          candidate$label,
          paste(missing, collapse = ", ")
        )
      )
    }
  }

  if (identical(candidate$device, "cuda") &&
      !identical(status, "unavailable") &&
      !isTRUE(host$cuda_runtime$validated)) {
    status <- "needs_install"
    issues <- c(issues, "Nvidia tooling is present but Python torch CUDA validation did not succeed.")
  }

  candidate$status <- status
  candidate$installable <- installable
  candidate$issues <- unique(issues)
  candidate$module_status <- module_probe$ok %||% logical(0)
  candidate
}

cs2step_resolve_text_embedding_candidates <- function(host) {
  list(
    mlx = cs2step_text_embedding_candidate(
      label = "mlx",
      backend = "mlx",
      device = "metal",
      model_id = "mlx-community/Qwen3-Embedding-8B-mxfp8",
      conda_env = host$conda_env,
      conda = host$conda,
      canonical_dim = 1024L,
      raw_dim = 4096L,
      required_modules = c("mlx_embeddings", "mlx.core", "numpy"),
      host_constraints = list(os = "Darwin", machine = "arm64")
    ),
    cuda = cs2step_text_embedding_candidate(
      label = "sentence_transformers_cuda",
      backend = "sentence_transformers",
      device = "cuda",
      model_id = "Qwen/Qwen3-Embedding-0.6B",
      conda_env = host$conda_env,
      conda = host$conda,
      canonical_dim = 1024L,
      raw_dim = 1024L,
      required_modules = c("sentence_transformers", "transformers", "torch", "numpy")
    ),
    rocm = cs2step_text_embedding_candidate(
      label = "sentence_transformers_rocm",
      backend = "sentence_transformers",
      device = "rocm",
      model_id = "Qwen/Qwen3-Embedding-0.6B",
      conda_env = host$conda_env,
      conda = host$conda,
      canonical_dim = 1024L,
      raw_dim = 1024L,
      required_modules = c("sentence_transformers", "transformers", "torch", "numpy")
    ),
    cpu = cs2step_text_embedding_candidate(
      label = "sentence_transformers_cpu",
      backend = "sentence_transformers",
      device = "cpu",
      model_id = "Qwen/Qwen3-Embedding-0.6B",
      conda_env = host$conda_env,
      conda = host$conda,
      canonical_dim = 1024L,
      raw_dim = 1024L,
      required_modules = c("sentence_transformers", "transformers", "torch", "numpy")
    )
  )
}

cs2step_select_text_embedding_candidate <- function(candidates,
                                                    host,
                                                    runtime = "auto",
                                                    family = "qwen3",
                                                    profile = "portable") {
  family <- cs2step_text_embedding_family(family)
  profile <- cs2step_text_embedding_profile(profile)
  runtime <- cs2step_text_embedding_runtime(runtime)
  candidates <- lapply(candidates, cs2step_evaluate_text_embedding_candidate, host = host)

  selected <- NULL
  issues <- character(0)
  if (identical(runtime, "auto")) {
    if (isTRUE(host$mlx_host_capable)) {
      selected <- candidates$mlx
    } else if (identical(host$os, "Linux") &&
               candidates$cuda$status %in% c("ready", "needs_install")) {
      selected <- candidates$cuda
    } else if (identical(host$os, "Linux") &&
               isTRUE(any(unlist(host$rocm_tools))) &&
               isTRUE(host$rocm_runtime$validated)) {
      selected <- candidates$rocm
    } else {
      selected <- candidates$cpu
      if (identical(host$os, "Linux") && isTRUE(any(unlist(host$nvidia_tools)))) {
        issues <- c(
          issues,
          "Nvidia tooling is present but CUDA text embeddings are not usable; falling back to the next runtime."
        )
      }
      if (identical(host$os, "Linux") && isTRUE(any(unlist(host$rocm_tools))) &&
          !isTRUE(host$rocm_runtime$validated)) {
        issues <- c(
          issues,
          "ROCm tooling is present but not validated in Python; falling back to CPU text embeddings."
        )
      }
    }
  } else if (identical(runtime, "mlx")) {
    selected <- candidates$mlx
  } else if (identical(runtime, "cuda")) {
    selected <- candidates$cuda
  } else if (identical(runtime, "rocm")) {
    selected <- candidates$rocm
  } else {
    selected <- candidates$cpu
  }

  issues <- unique(c(
    issues,
    unlist(lapply(candidates, `[[`, "issues"), use.names = FALSE)
  ))

  list(
    family = family,
    profile = profile,
    runtime = runtime,
    host = host,
    candidates = candidates,
    selected = selected,
    issues = issues
  )
}

cs2step_inspect_text_embedding_candidates <- function(host,
                                                      family = "qwen3",
                                                      runtime = "auto",
                                                      profile = "portable") {
  candidates <- cs2step_resolve_text_embedding_candidates(host)
  cs2step_select_text_embedding_candidate(
    candidates = candidates,
    host = host,
    runtime = runtime,
    family = family,
    profile = profile
  )
}

cs2step_conda_run <- function(conda, conda_env, args) {
  conda_bin <- cs2step_resolve_conda_binary(conda)
  if (!nzchar(conda_bin %||% "")) {
    stop("Unable to resolve a conda binary for text embedding runtime installation.", call. = FALSE)
  }
  probe <- cs2step_command_probe(
    conda_bin,
    c("run", "-n", conda_env, args)
  )
  if (!identical(probe$status, 0L)) {
    stop(
      sprintf(
        "Command failed while preparing text embedding runtime in '%s':\n%s",
        conda_env,
        paste(probe$output, collapse = "\n")
      ),
      call. = FALSE
    )
  }
  invisible(probe$output)
}

cs2step_pip_install_in_conda <- function(conda,
                                         conda_env,
                                         packages,
                                         index_url = NULL,
                                         force_reinstall = FALSE) {
  args <- c("python", "-m", "pip", "install", "--upgrade")
  if (isTRUE(force_reinstall)) {
    args <- c(args, "--force-reinstall")
  }
  if (!is.null(index_url) && nzchar(index_url)) {
    args <- c(args, "--index-url", index_url)
  }
  args <- c(args, as.character(packages))
  cs2step_conda_run(conda = conda, conda_env = conda_env, args = args)
}

cs2step_install_cuda_sentence_transformers <- function(spec) {
  driver <- cs2step_probe_nvidia_driver()
  wheel <- cs2step_cuda_torch_wheel(driver$driver_major)
  if (is.null(wheel)) {
    stop(
      sprintf(
        "No supported CUDA torch wheel is available for Nvidia driver '%s'.",
        driver$driver_version %||% "unknown"
      ),
      call. = FALSE
    )
  }
  cs2step_pip_install_in_conda(
    conda = spec$conda,
    conda_env = spec$conda_env,
    packages = "torch",
    index_url = wheel$index_url,
    force_reinstall = TRUE
  )
  cs2step_pip_install_in_conda(
    conda = spec$conda,
    conda_env = spec$conda_env,
    packages = c("sentence-transformers", "transformers")
  )
}

cs2step_ensure_text_embedding_runtime <- function(spec) {
  spec <- cs2step_normalize_text_embedding_spec(spec)
  env_state <- cs2step_backend_env_state(conda_env = spec$conda_env, conda = spec$conda)
  if (!isTRUE(env_state$core_modules_ready)) {
    build_backend(conda_env = spec$conda_env, conda = spec$conda)
  }

  if (identical(spec$backend, "mlx")) {
    reticulate::py_install(
      packages = c("mlx", "mlx-embeddings"),
      envname = spec$conda_env,
      conda = spec$conda,
      pip = TRUE
    )
  } else if (identical(spec$backend, "sentence_transformers")) {
    if (identical(spec$device, "cuda")) {
      cs2step_install_cuda_sentence_transformers(spec)
    } else if (identical(spec$device, "cpu")) {
      reticulate::py_install(
        packages = c("torch", "sentence-transformers", "transformers"),
        envname = spec$conda_env,
        conda = spec$conda,
        pip = TRUE
      )
    } else {
      reticulate::py_install(
        packages = c("sentence-transformers", "transformers"),
        envname = spec$conda_env,
        conda = spec$conda,
        pip = TRUE
      )
    }
  }
  invisible(TRUE)
}

cs2step_normalize_text_embedding_spec <- function(spec) {
  if (!is.list(spec) || is.null(spec$backend) || is.null(spec$model_id)) {
    stop("Invalid text embedding backend specification.", call. = FALSE)
  }
  spec$family <- cs2step_text_embedding_family(spec$family %||% "qwen3")
  spec$profile <- cs2step_text_embedding_profile(spec$profile %||% "portable")
  spec$runtime <- cs2step_text_embedding_runtime(spec$runtime %||% "auto")
  spec$canonical_dim <- as.integer(spec$canonical_dim %||% 1024L)
  spec$raw_dim <- as.integer(spec$raw_dim %||% spec$canonical_dim)
  spec$conda_env <- as.character(spec$conda_env %||% "strategize_env")
  spec$conda <- cs2step_resolve_conda_binary(spec$conda %||% "auto")
  spec$cache_key_version <- as.integer(spec$cache_key_version %||% 1L)
  spec
}

cs2step_text_embedding_cache_file <- function(spec, cache_dir = NULL) {
  spec <- cs2step_normalize_text_embedding_spec(spec)
  root <- if (!is.null(cache_dir) && nzchar(cache_dir)) {
    path.expand(as.character(cache_dir))
  } else {
    file.path(tools::R_user_dir("strategize", which = "cache"), "text-embeddings")
  }
  dir.create(root, recursive = TRUE, showWarnings = FALSE)
  file.path(
    root,
    sprintf(
      "%s.rds",
      digest::digest(list(
        family = spec$family,
        backend = spec$backend,
        device = spec$device,
        model = spec$model_id,
        dim = spec$canonical_dim,
        version = spec$cache_key_version
      ))
    )
  )
}

cs2step_text_embedding_canonicalize_matrix <- function(emb, spec) {
  emb <- as.matrix(emb)
  storage.mode(emb) <- "double"
  target_dim <- as.integer(spec$canonical_dim %||% 1024L)
  if (ncol(emb) < target_dim) {
    stop(
      sprintf(
        "Text embedding backend '%s' returned %d columns, below the required canonical width %d.",
        spec$label %||% spec$backend,
        ncol(emb),
        target_dim
      ),
      call. = FALSE
    )
  }
  if (ncol(emb) > target_dim) {
    emb <- emb[, seq_len(target_dim), drop = FALSE]
  }
  emb
}

cs2step_build_text_embedding_fn <- function(spec, cache_dir = NULL) {
  spec <- cs2step_normalize_text_embedding_spec(spec)
  cache_file <- function() cs2step_text_embedding_cache_file(spec, cache_dir = cache_dir)
  state <- new.env(parent = emptyenv())
  state$loaded <- FALSE
  state$cache <- NULL

  load_cache <- function() {
    if (!is.null(state$cache)) {
      return(invisible(TRUE))
    }
    cache_path <- cache_file()
    state$cache <- if (file.exists(cache_path)) readRDS(cache_path) else list()
    invisible(TRUE)
  }

  save_cache <- function() {
    cache_path <- cache_file()
    saveRDS(state$cache, cache_path)
    invisible(TRUE)
  }

  load_model <- function() {
    if (isTRUE(state$loaded)) {
      return(invisible(TRUE))
    }
    cs2step_ensure_text_embedding_runtime(spec)
    reticulate::use_condaenv(spec$conda_env, required = TRUE)

    if (identical(spec$backend, "mlx")) {
      state$mlx_embeddings <- reticulate::import("mlx_embeddings", delay_load = FALSE)
      state$mx <- reticulate::import("mlx.core", delay_load = FALSE)
      state$np <- reticulate::import("numpy", delay_load = FALSE)
      model_bundle <- state$mlx_embeddings$load(spec$model_id)
      state$model <- model_bundle[[1]]
      state$processor <- model_bundle[[2]]
    } else {
      state$sentence_transformers <- reticulate::import("sentence_transformers", delay_load = FALSE)
      device_use <- if (spec$device %in% c("cuda", "rocm")) "cuda" else "cpu"
      state$model <- state$sentence_transformers$SentenceTransformer(
        spec$model_id,
        device = device_use
      )
    }
    state$loaded <- TRUE
    invisible(TRUE)
  }

  encode_mlx <- function(texts) {
    output <- state$mlx_embeddings$generate(
      state$model,
      state$processor,
      texts = as.list(texts)
    )
    emb_obj <- tryCatch(reticulate::py_get_attr(output, "text_embeds"), error = function(e) output)
    state$mx$eval(emb_obj)
    emb <- reticulate::py_to_r(state$np$array(emb_obj))
    cs2step_text_embedding_canonicalize_matrix(emb, spec)
  }

  encode_sentence_transformers <- function(texts) {
    emb <- reticulate::py_to_r(
      state$model$encode(
        texts,
        show_progress_bar = FALSE,
        convert_to_numpy = TRUE
      )
    )
    cs2step_text_embedding_canonicalize_matrix(emb, spec)
  }

  fn <- function(texts) {
    load_cache()
    texts <- trimws(as.character(texts))
    texts[is.na(texts) | !nzchar(texts)] <- "(missing)"
    missing <- setdiff(unique(texts), names(state$cache))
    if (length(missing)) {
      load_model()
      emb <- if (identical(spec$backend, "mlx")) {
        encode_mlx(missing)
      } else {
        encode_sentence_transformers(missing)
      }
      if (nrow(emb) != length(missing)) {
        stop("Text embedding backend returned a row count that does not match the input text count.", call. = FALSE)
      }
      for (i in seq_along(missing)) {
        state$cache[[missing[[i]]]] <- unname(emb[i, ])
      }
      save_cache()
    }
    out <- do.call(rbind, lapply(texts, function(text) state$cache[[text]]))
    rownames(out) <- NULL
    storage.mode(out) <- "double"
    out
  }
  attr(fn, "text_embedding_backend") <- spec
  class(fn) <- c("strategize_text_embedding_fn", class(fn))
  fn
}

cs2step_capture_text_embedding_metadata <- function(text_embedding_fn = NULL,
                                                    text_embedding_backend = NULL) {
  backend <- text_embedding_backend %||%
    if (is.function(text_embedding_fn)) attr(text_embedding_fn, "text_embedding_backend") else NULL
  list(
    text_embedding_fn = text_embedding_fn,
    text_embedding_backend = backend
  )
}

cs2step_restore_text_embedding_metadata <- function(metadata,
                                                    conda_env = NULL,
                                                    required = FALSE) {
  metadata <- metadata %||% list()
  backend <- metadata$text_embedding_backend %||%
    if (is.function(metadata$text_embedding_fn)) {
      attr(metadata$text_embedding_fn, "text_embedding_backend")
    } else {
      NULL
    }
  if (is.null(backend)) {
    return(metadata)
  }
  backend <- cs2step_normalize_text_embedding_spec(backend)
  if (!is.null(conda_env) && nzchar(conda_env)) {
    backend$conda_env <- conda_env
  }
  metadata$text_embedding_backend <- backend
  if (!is.function(metadata$text_embedding_fn)) {
    metadata$text_embedding_fn <- cs2step_build_text_embedding_fn(
      spec = backend,
      cache_dir = NULL
    )
  } else if (is.null(attr(metadata$text_embedding_fn, "text_embedding_backend"))) {
    attr(metadata$text_embedding_fn, "text_embedding_backend") <- backend
  }
  metadata
}
