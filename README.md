# `strategize`

[<img src="https://img.shields.io/badge/Demo-View%20Demo-blue" alt="Demo Button">](https://connorjerzak.com/wp-content/uploads/2025/02/MainVignette.html)

`strategize` is an R package for learning optimal or adversarial probability distributions over factor levels in conjoint and related factorial experiments. The main workflow fits an outcome model, shifts the design distribution toward better-performing profiles, and returns both the learned distribution and the implied outcome under that distribution.

Today the package supports:

- single-study optimization with `strategize()` and `cv_strategize()`
- adversarial optimization for two-player settings
- prediction-only outcome modeling with `strategic_prediction()`
- cached predictor and neural bundle I/O with `save_strategic_predictor()`, `load_strategic_predictor()`, `save_neural_outcome_bundle()`, and `load_neural_outcome_bundle()`
- foundation-model runtime support: load trained `preference.fm` checkpoints, adapt them to target studies, extract embeddings, and run predictors

# Installation

Install the package directly from GitHub:

```r
remotes::install_github("cjerzak/strategize-software/strategize")
library(strategize)
```

## Backend Setup

The optimization workflow in `strategize()` and `cv_strategize()`, along with the neural prediction and foundation-model APIs, uses the package's JAX-backed computational environment. After installing the R package, run this once:

```r
strategize::build_backend(conda_env = "strategize_env")
```

This creates a conda environment with JAX, numpy, optax, equinox, numpyro, Orbax checkpoint support, and the other Python-side dependencies used by the package.

# Minimal Example

The example below uses a simple pairwise conjoint setup with two factors and a binary forced-choice outcome.

```r
set.seed(123)

n_pairs <- 200

W_left <- data.frame(
  Gender = sample(c("Male", "Female"), n_pairs, replace = TRUE),
  Message = sample(c("Jobs", "Taxes"), n_pairs, replace = TRUE)
)

W_right <- data.frame(
  Gender = sample(c("Male", "Female"), n_pairs, replace = TRUE),
  Message = sample(c("Jobs", "Taxes"), n_pairs, replace = TRUE)
)

score_left <- 0.15 * (W_left$Gender == "Female") + 0.20 * (W_left$Message == "Jobs")
score_right <- 0.15 * (W_right$Gender == "Female") + 0.20 * (W_right$Message == "Jobs")

Y_left <- as.numeric(score_left > score_right)
Y <- c(Y_left, 1 - Y_left)

W <- rbind(W_left, W_right)
pair_id <- c(seq_len(n_pairs), seq_len(n_pairs))
profile_order <- c(rep(1L, n_pairs), rep(2L, n_pairs))

p_list <- create_p_list(W, uniform = TRUE)

fit <- cv_strategize(
  Y = Y,
  W = W,
  p_list = p_list,
  lambda_seq = c(0.01, 0.1, 0.5),
  pair_id = pair_id,
  profile_order = profile_order,
  diff = TRUE,
  nSGD = 100,
  compute_se = TRUE
)

fit$pi_star_point
fit$Q_point
fit$Q_se
```

The return object retains backward-compatible aliases for older code, but the current documentation uses `Q_point` and `Q_se` as the primary public result fields.

For supported pairwise forced-choice binomial GLM analyses, set `crossfit_q = TRUE` to add an out-of-fold policy-value diagnostic for the final learned policy. This covers non-adversarial average-case runs and the two-party adversarial `adversarial_model_strategy = "four"` case:

```r
fit_cf <- strategize(
  Y = Y,
  W = W,
  p_list = p_list,
  lambda = fit$lambda,
  pair_id = pair_id,
  profile_order = profile_order,
  diff = TRUE,
  crossfit_q = TRUE,
  crossfit_q_control = list(folds = 3)
)

fit_cf$Q_gain_crossfit
fit_cf$Q_crossfit_info$summary
```

For adversarial runs, `crossfit_q_control = list(perspective_group = "...")` selects the candidate group whose win probability is reported. If omitted, the first sorted group label is used. Adversarial cross-fit weights use the independent product assignment implied by `p_list`.

# Other Workflows

## Prediction-only API

Use `strategic_prediction()` when you want the fitted outcome model without the intervention optimization step:

```r
predictor <- strategic_prediction(
  Y = Y,
  W = W,
  model = "glm",
  mode = "pairwise",
  pair_id = pair_id,
  profile_order = profile_order
)

predict(predictor, newdata = list(W = W, pair_id = pair_id, profile_order = profile_order))
```

Predictors can be cached and reused with `save_strategic_predictor()` and `load_strategic_predictor()`.

## Foundation-model workflow

Pooled foundation-model training now lives in the separate `preference.fm` package. Use `preference.fm::fit_conjoint_foundation_model()` and `preference.fm::save_conjoint_foundation_bundle()` to train and write checkpoint directories. Use `strategize::load_conjoint_foundation_bundle()`, `adapt_conjoint_foundation_model()`, and `extract_embeddings()` to consume trained artifacts.

# Documentation

- [Quick Start](strategize/vignettes/QuickStart.Rmd)
- [Main Vignette](strategize/vignettes/MainVignette.Rmd)
- [Foundation Models](strategize/vignettes/FoundationModels.Rmd)
- [Troubleshooting](strategize/vignettes/Troubleshooting.Rmd)
- [Reference Manual](strategize.pdf)

# License

GPL-3.

## References

Jerzak, Connor T., Priyanshi Chandra, and Rishi Hazra. 2026. "MiniMax Learning of Interpretable Factored Stochastic Policies from Conjoint Data, with Uncertainty Quantification." *International Conference on Machine Learning (ICML)*. [[PDF]](https://arxiv.org/abs/2504.19043) [[bib]](https://connorjerzak.com/wp-content/uploads/2025/04/Strat.txt)

```bibtex
@inproceedings{jerzak2026minimax,
  author    = {Jerzak, Connor T. and Chandra, Priyanshi and Hazra, Rishi},
  title     = {MiniMax Learning of Interpretable Factored Stochastic Policies from Conjoint Data, with Uncertainty Quantification},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```
