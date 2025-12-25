# =============================================================================
# Tests for Visualization Functions
# =============================================================================
# These tests cover strategize.plot() and plot_best_response_curves()
# Note: These tests primarily check that functions run without error,
# as visual output is difficult to test automatically.
# =============================================================================

# =============================================================================
# strategize.plot() Tests
# =============================================================================

test_that("strategize.plot runs without error on mock data", {
  skip_on_cran()

  # Create mock data (no JAX needed)
  pi_star_list <- list(k1 = list(
    Gender = c(Male = 0.4, Female = 0.6),
    Age = c(Young = 0.3, Middle = 0.3, Old = 0.4)
  ))
  pi_star_se_list <- list(k1 = list(
    Gender = c(Male = 0.05, Female = 0.05),
    Age = c(Young = 0.04, Middle = 0.04, Old = 0.04)
  ))
  p_list <- list(
    Gender = c(Male = 0.5, Female = 0.5),
    Age = c(Young = 0.33, Middle = 0.33, Old = 0.34)
  )

  # Should run without error (may produce warnings about graphics device)
  expect_error(
    suppressWarnings({
      strategize.plot(
        pi_star_list = pi_star_list,
        pi_star_se_list = pi_star_se_list,
        p_list = p_list
      )
    }),
    NA
  )
})

test_that("strategize.plot handles various ticks_type options", {
  skip_on_cran()

  pi_star_list <- list(k1 = list(
    Factor1 = c(A = 0.6, B = 0.4)
  ))
  pi_star_se_list <- list(k1 = list(
    Factor1 = c(A = 0.05, B = 0.05)
  ))
  p_list <- list(Factor1 = c(A = 0.5, B = 0.5))

  for (ticks in c("assignmentProbs", "zero", "none")) {
    expect_error(
      suppressWarnings({
        strategize.plot(
          pi_star_list = pi_star_list,
          pi_star_se_list = pi_star_se_list,
          p_list = p_list,
          ticks_type = ticks
        )
      }),
      NA
    )
  }
})

test_that("strategize.plot handles multiple factors", {
  skip_on_cran()

  # Create mock data with 4 factors
  pi_star_list <- list(k1 = list(
    Factor1 = c(A = 0.6, B = 0.4),
    Factor2 = c(X = 0.3, Y = 0.7),
    Factor3 = c(P = 0.5, Q = 0.5),
    Factor4 = c(M = 0.4, N = 0.6)
  ))
  pi_star_se_list <- list(k1 = list(
    Factor1 = c(A = 0.05, B = 0.05),
    Factor2 = c(X = 0.04, Y = 0.04),
    Factor3 = c(P = 0.03, Q = 0.03),
    Factor4 = c(M = 0.05, N = 0.05)
  ))
  p_list <- list(
    Factor1 = c(A = 0.5, B = 0.5),
    Factor2 = c(X = 0.5, Y = 0.5),
    Factor3 = c(P = 0.5, Q = 0.5),
    Factor4 = c(M = 0.5, N = 0.5)
  )

  expect_error(
    suppressWarnings({
      strategize.plot(
        pi_star_list = pi_star_list,
        pi_star_se_list = pi_star_se_list,
        p_list = p_list
      )
    }),
    NA
  )
})

test_that("strategize.plot handles multi-level factors", {
  skip_on_cran()

  # Factor with 5 levels
  pi_star_list <- list(k1 = list(
    Region = c(North = 0.15, South = 0.25, East = 0.20, West = 0.25, Central = 0.15)
  ))
  pi_star_se_list <- list(k1 = list(
    Region = c(North = 0.03, South = 0.04, East = 0.03, West = 0.04, Central = 0.03)
  ))
  p_list <- list(
    Region = c(North = 0.2, South = 0.2, East = 0.2, West = 0.2, Central = 0.2)
  )

  expect_error(
    suppressWarnings({
      strategize.plot(
        pi_star_list = pi_star_list,
        pi_star_se_list = pi_star_se_list,
        p_list = p_list
      )
    }),
    NA
  )
})

test_that("strategize.plot handles plot_ci = FALSE", {
  skip_on_cran()

  pi_star_list <- list(k1 = list(
    Gender = c(Male = 0.4, Female = 0.6)
  ))
  p_list <- list(Gender = c(Male = 0.5, Female = 0.5))

  expect_error(
    suppressWarnings({
      strategize.plot(
        pi_star_list = pi_star_list,
        pi_star_se_list = NULL,
        p_list = p_list,
        plot_ci = FALSE
      )
    }),
    NA
  )
})

test_that("strategize.plot handles custom colors", {
  skip_on_cran()

  pi_star_list <- list(k1 = list(
    Gender = c(Male = 0.4, Female = 0.6)
  ))
  pi_star_se_list <- list(k1 = list(
    Gender = c(Male = 0.05, Female = 0.05)
  ))
  p_list <- list(Gender = c(Male = 0.5, Female = 0.5))

  expect_error(
    suppressWarnings({
      strategize.plot(
        pi_star_list = pi_star_list,
        pi_star_se_list = pi_star_se_list,
        p_list = p_list,
        col_vec = c("darkblue")
      )
    }),
    NA
  )
})

test_that("strategize.plot handles multiple clusters", {
  skip_on_cran()

  # Two clusters
  pi_star_list <- list(
    k1 = list(Gender = c(Male = 0.4, Female = 0.6)),
    k2 = list(Gender = c(Male = 0.7, Female = 0.3))
  )
  pi_star_se_list <- list(
    k1 = list(Gender = c(Male = 0.05, Female = 0.05)),
    k2 = list(Gender = c(Male = 0.04, Female = 0.04))
  )
  p_list <- list(Gender = c(Male = 0.5, Female = 0.5))

  expect_error(
    suppressWarnings({
      strategize.plot(
        pi_star_list = pi_star_list,
        pi_star_se_list = pi_star_se_list,
        p_list = p_list,
        col_vec = c("blue", "red")
      )
    }),
    NA
  )
})

# =============================================================================
# plot_best_response_curves() Tests
# =============================================================================
# These tests require a full adversarial strategize result, so they
# are marked to skip unless JAX is available.

test_that("plot_best_response_curves requires adversarial result", {
  skip_on_cran()
  skip_if_no_jax()
  skip_if_slow()

  # This test would require running a full adversarial strategize,
 # which is slow. Skip for normal testing.

  # Mock test: at minimum, function should exist and be callable
  expect_true(is.function(plot_best_response_curves))
})

# =============================================================================
# Name Transformer Tests
# =============================================================================

test_that("strategize.plot applies factor_name_transformer", {
  skip_on_cran()

  pi_star_list <- list(k1 = list(
    gender = c(male = 0.4, female = 0.6)
  ))
  pi_star_se_list <- list(k1 = list(
    gender = c(male = 0.05, female = 0.05)
  ))
  p_list <- list(gender = c(male = 0.5, female = 0.5))

  # Capitalize factor names
  expect_error(
    suppressWarnings({
      strategize.plot(
        pi_star_list = pi_star_list,
        pi_star_se_list = pi_star_se_list,
        p_list = p_list,
        factor_name_transformer = function(x) toupper(x)
      )
    }),
    NA
  )
})

test_that("strategize.plot applies level_name_transformer", {
  skip_on_cran()

  pi_star_list <- list(k1 = list(
    Gender = c(m = 0.4, f = 0.6)
  ))
  pi_star_se_list <- list(k1 = list(
    Gender = c(m = 0.05, f = 0.05)
  ))
  p_list <- list(Gender = c(m = 0.5, f = 0.5))

  # Expand level names
  label_map <- c(m = "Male", f = "Female")
  expect_error(
    suppressWarnings({
      strategize.plot(
        pi_star_list = pi_star_list,
        pi_star_se_list = pi_star_se_list,
        p_list = p_list,
        level_name_transformer = function(x) {
          ifelse(x %in% names(label_map), label_map[x], x)
        }
      )
    }),
    NA
  )
})
