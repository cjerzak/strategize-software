# `strategize`: An R package for discovering optimal treatment strategies in high-dimensional data

[<img src="https://img.shields.io/badge/Demo-View%20Demo-blue" alt="Demo Button">](https://connorjerzak.com/wp-content/uploads/2025/02/MainVignette.html)

Software for implementing optimal stochastic intervention analysis. Current implementation handles conjoint data from experiments. Future work may also include text, network, and time series data, with observational designs potentially supported.

# Installation

The most recent version of `strategize` can be installed directly from the repository using the `devtools` package
```
devtools::install_github("cjerzak/strategize-software/strategize")
```

The package can then be loaded into your R session like so:
```
library(strategize)
```
Package functions can also be accessed as `strategize::function_name`.

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
fit$PiStar_point
fit$Q_point_mEst
```

This script simulates a two-factor forced-choice design, fits the model, and
returns `PiStar_point`, the recommended distribution over factor levels, along
with the expected outcome `Q_point_mEst` under that distribution.

# License

GPL-3.

## References

TBD.

