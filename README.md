# `strategize`: An R package for discovering optimal treatment strategies in high-dimensional data

[<img src="https://img.shields.io/badge/Demo-View%20Demo-blue" alt="Demo Button">](https://connorjerzak.com/wp-content/uploads/2025/02/MainVignette.html)

Software for implementing optimal stochastic intervention analysis. Current implementation handles conjoint data from experiments. Future work may also include text, network, and time series data, with observational designs potentially supported.

# Installation

The most recent version of `strategize` can be installed directly from the repository using the `devtools` package:
```r
devtools::install_github("cjerzak/strategize-software/strategize")
```

The package can then be loaded into your R session:
```r
library(strategize)
```
Package functions can also be accessed as `strategize::function_name`.

## Python Backend Setup

The package uses JAX for automatic differentiation. Set up the Python backend with:
```r
strategize::build_backend(conda_env = "strategize_env")
```
This creates a conda environment with JAX, numpy, optax, equinox, and numpyro. On Linux with NVIDIA GPUs, it auto-detects the driver version and installs appropriate CUDA wheels.

# Tutorial

Below is a minimal example demonstrating how to discover an optimal set of
factor‐level probabilities for a simple conjoint design.  Because the package
does not ship with data, we begin by simulating a small dataset.

```r
set.seed(123)

# Example data with two factors and a binary outcome
n <- 200
W <- data.frame(
  sex   = sample(c("Male", "Female"), n, replace = TRUE),
  party = sample(c("A", "B"),       n, replace = TRUE)
)
Y <- rbinom(n, 1, 0.5)

# Original (uniform) assignment probabilities for each factor
p_list <- list(
  sex   = c(Male = 0.5, Female = 0.5),
  party = c(A = 0.5,   B = 0.5)
)

# Search for a probability distribution that maximizes the expected outcome
library(strategize)
fit <- cv_strategize(
  Y = Y,
  W = W,
  p_list = p_list,
  lambda = 0.1,
  nSGD = 100,
  adversarial = FALSE
)

# Optimized factor-level probabilities and predicted outcome
fit$pi_star_point
fit$Q_point_mEst
```

This script simulates a two-factor forced-choice design, fits the model, and
returns `pi_star_point`, the recommended distribution over factor levels, along
with the expected outcome `Q_point_mEst` under that distribution.

# Key Features

- **Adversarial Mode**: Set `adversarial = TRUE` to find Nash equilibrium strategies in two-player zero-sum games (e.g., competing candidates)
- **Optimistic Updates**: Toggle `optimism = "ogda"`, `optimism = "smp"`, or `optimism = "none"`; default is `optimism = "extragrad"` (extra-gradient look-ahead) for more stable min-max training
- **Cross-Validation**: Use `cv_strategize()` to automatically select the regularization parameter lambda
- **One-Step Estimation**: Use `strategize_onestep()` for simultaneous outcome modeling and distribution optimization
- **Helper Functions**: Use `create_p_list()` to easily create probability lists from your data, and `strategize_preset()` for quick analysis with sensible defaults
- **Validation & Diagnostics**: Use `validate_equilibrium()` to verify Nash equilibrium quality, `plot_convergence()` to visualize optimization, `plot_quadrant_breakdown()` for scenario analysis, and `summarize_adversarial()` for comprehensive summaries

# Documentation

- [Main Vignette](https://connorjerzak.com/wp-content/uploads/2025/02/MainVignette.html) - Comprehensive tutorial and methodology overview
- QuickStart vignette (`vignettes/QuickStart.Rmd`) - Getting started guide
- Troubleshooting vignette (`vignettes/Troubleshooting.Rmd`) - Common issues and solutions

# License

GPL-3.

## References

Jerzak, Connor T., Priyanshi Chandra, and Rishi Hazra. 2025. "Selecting Optimal Candidate Profiles in Adversarial Environments Using Conjoint Analysis and Machine Learning." *arXiv preprint* arXiv:2504.19043. https://arxiv.org/abs/2504.19043.

```bibtex
@misc{jerzak2025selectingoptimalcandidateprofiles,
      title={Selecting Optimal Candidate Profiles in Adversarial Environments Using Conjoint Analysis and Machine Learning},
      author={Connor T. Jerzak and Priyanshi Chandra and Rishi Hazra},
      year={2025},
      eprint={2504.19043},
      archivePrefix={arXiv},
      primaryClass={stat.ME},
      url={https://arxiv.org/abs/2504.19043}, 
}
```
