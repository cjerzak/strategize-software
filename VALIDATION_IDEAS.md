# Adversarial Validation Improvement Ideas

**Generated:** December 28, 2025
**Models Consulted:** GPT 5.2 (Thinking High), Gemini 3 Pro Preview
**Consensus Confidence:** High (8-9/10)

## Executive Summary

Both models strongly agree: the current validation suite is functional but heuristic. Moving to a rigorous standard requires three core improvements:

1. **Replace grid search** with intelligent optimization (Bayesian Optimization or multi-start local+global)
2. **Add bootstrap confidence intervals** for Q*, exploitability, and equilibrium positions
3. **Implement Hessian eigenvalue analysis** for geometric verification of saddle point structure

---

## Current Limitations (Identified)

| Limitation | Impact | Priority |
|------------|--------|----------|
| Grid search falls back to random for D>5 | High - mathematically futile in high dimensions | Critical |
| No confidence intervals on metrics | High - users can't assess statistical significance | Critical |
| Hardcoded parameters (±2.0 range, 50 pts, 0.01 tol) | Medium - inappropriate for many problems | High |
| No multi-start validation | High - local optima go undetected | Critical |
| No robustness/stability analysis | Medium - fragile equilibria appear valid | High |
| Quadrant breakdown crude fallback | Low - undermines trust when sampling unavailable | Medium |

---

## Tier 1: High-Impact Improvements

### 1. Intelligent Best-Response Search

**Problem:** Grid search is O(n^D) and random search provides no guarantees.

**Solution A - Multi-Start Local+Global (GPT 5.2):**
```r
validate_equilibrium(..., method = c("multistart", "global", "grid"))
```
- Sample n_starts initial points via Latin hypercube (`lhs` package)
- Run L-BFGS-B from each start (use gradients if available)
- Fall back to BOBYQA/CMA-ES if gradients unavailable
- Report spread across starts (indicates nonconvexity)

**Solution B - Bayesian Optimization (Gemini 3 Pro):**
```r
validate_with_bayes <- function(fixed_player_vec, model, bounds) {
  scoring_fn <- function(candidate_vec) {
    val <- predict_payoff(model, fixed_player_vec, candidate_vec)
    return(list(Score = val, Pred = 0))
  }

  res <- ParBayesianOptimization::bayesianOptimization(
    FUN = scoring_fn,
    bounds = bounds,
    init_points = 10,
    n_iter = 20,
    acq = "ei"  # Expected Improvement
  )
  return(res$Best_Par)
}
```

**Recommendation:** Implement both. Use multi-start as default (faster), BO as `method="bayesian"` option for thorough validation.

**R Dependencies:** `lhs`, `nloptr`, `ParBayesianOptimization` or `DiceOptim`

---

### 2. Bootstrap Confidence Intervals

**Problem:** Point estimates don't indicate if equilibrium is statistically distinguishable from noise.

**Solution:**
```r
bootstrap_equilibrium <- function(result, n_boot = 100, parallel = TRUE) {
  # Cluster bootstrap: resample respondents with replacement
  # Re-run adversarial optimization for each bootstrap sample
  # Return percentile or BCa intervals for:
  #   - Q* (equilibrium vote share)
  #   - Exploitability (per player and max)
  #   - Equilibrium positions (a_ast, a_dag)
}
```

**Key Outputs:**
- 95% CI for Q*
- "Probability that exploitability ≤ ε"
- CI overlap with null strategy (origin) indicates weak result

**Computational Note:** Expensive - requires parallelization via `future.apply` or `parallel`. Consider influence-function/sandwich approximation as fast alternative.

---

### 3. Hessian Eigenvalue Analysis (Geometric Verification)

**Problem:** Gradient near zero is necessary but insufficient - could be saddle point, local optimum, or flat spot.

**Solution:**
```r
check_curvature <- function(equilibrium_point, model) {
  h_mat <- numDeriv::hessian(func = objective_fn, x = equilibrium_point)

  # Player A (Maximizer) block must be Negative Definite
  eigen_A <- eigen(h_mat[player_A_indices, player_A_indices])$values
  is_local_max <- all(eigen_A < 0)

  # Player B (Minimizer) block must be Positive Definite
  eigen_B <- eigen(h_mat[player_B_indices, player_B_indices])$values
  is_local_min <- all(eigen_B > 0)

  condition_number <- max(abs(eigen_A)) / min(abs(eigen_A))

  return(list(
    valid_saddle = is_local_max && is_local_min,
    eigenvalues_A = eigen_A,
    eigenvalues_B = eigen_B,
    condition_number = condition_number,
    flat_directions = sum(abs(c(eigen_A, eigen_B)) < 1e-6)
  ))
}
```

**Interpretation:**
- `valid_saddle = FALSE` → not a proper Nash equilibrium
- High condition number → poorly identified, sensitive
- Flat directions → weak identification on some parameters

**R Dependencies:** `numDeriv`

---

## Tier 2: Medium-Impact Improvements

### 4. Composite Health Check Function

**Unified wrapper that runs all validation checks:**
```r
check_nash_health <- function(result, level = c("quick", "standard", "strict")) {
  # Level determines thoroughness vs runtime tradeoff
  checks <- list(
    gradient_norm = check_gradients(result),
    curvature = check_curvature(result),
    multistart = run_multistart_validation(result, n_starts = switch(level, quick=3, standard=10, strict=25)),
    bootstrap = if(level == "strict") bootstrap_equilibrium(result, n_boot=100) else NULL
  )

  status <- determine_status(checks)  # "PASS" / "WARNING" / "FAIL"

  return(structure(checks, class = "nash_health", status = status))
}
```

**Output:** S3 object with print method showing green/yellow/red status.

---

### 5. Perturbation Stress Tests

**Test robustness to small deviations:**
```r
stress_test <- function(result, radii = c(0.1, 0.25, 0.5), n_perturb = 20) {
  # For each radius:
  #   1. Perturb equilibrium by random vectors of that magnitude
  #   2. Re-evaluate Q at perturbed points
  #   3. Check if any perturbation improves objective

  # Compute "Lipschitz-like" sensitivity: ΔQ / ||Δs||
  # Return stability_score = 1 / variance(equilibrium under perturbation)
}
```

**Also consider:** Perturb conjoint coefficients by their SEs to test model sensitivity.

---

### 6. Adaptive Tolerance & Auto-Configuration

**Replace hardcoded parameters:**
```r
validate_equilibrium(..., control = list(
  bounds = NULL,        # Auto-scale from observed strategy magnitudes
  n_starts = 10,
  tol_abs = 0.005,
  tol_rel = 0.01,
  max_evals = 1000,
  seed = NULL
))
```

**Auto-scaling logic:**
- Bounds: Estimate from distribution of a_ast, a_dag across convergence history
- Tolerance: Scale by |Q*| for relative tolerance

---

## Tier 3: Enhancement Ideas

### 7. Regret Decomposition by Dimension

**When BR error is high, show *why*:**
```r
decompose_regret <- function(result) {
  # For each dimension/feature group:
  #   1. Optimize that dimension alone (hold others fixed)
  #   2. Record potential improvement
  # Return ranked table of dimensions by regret contribution
}
```

**Use case:** Guides users to constrain strategy space or increase regularization on problematic dimensions.

---

### 8. Improved Quadrant Breakdown

**Add variance reporting:**
```r
plot_quadrant_breakdown(..., show_ci = TRUE, n_boot = 100)
# Report mean ± SE for each quadrant weight
# Show CI bars on contribution chart
```

**Fix fallback:** When sampling unavailable, use delta method with model coefficient covariance instead of crude 25% uniform.

---

### 9. Convergence Certification Heuristics

**Automatic warnings:**
```r
certify_convergence <- function(result) {
  warnings <- c()

  if (detect_oscillation(result$convergence_history))
    warnings <- c(warnings, "Potential cycling detected")

  if (gradient_small_but_improvement_found(result))
    warnings <- c(warnings, "Stationary but not optimal")

  if (learning_rate_collapsed(result))
    warnings <- c(warnings, "Learning rate collapsed early")

  return(list(certified = length(warnings) == 0, warnings = warnings))
}
```

---

## Implementation Roadmap

### Phase 1 (Core - 1-2 weeks effort)
1. `check_nash_health()` wrapper with Pass/Warning/Fail
2. Multi-start best-response search
3. `control = list(...)` parameter interface

### Phase 2 (Statistical Rigor - 2-3 weeks effort)
4. Hessian eigenvalue analysis
5. Bootstrap confidence intervals (with parallelization)
6. Perturbation stress tests

### Phase 3 (Polish - 1 week effort)
7. Validation tiers (quick/standard/strict)
8. Improved quadrant breakdown with CIs
9. Regret decomposition diagnostics

---

## Dependencies to Add

```r
# DESCRIPTION Suggests:
Suggests:
    lhs,                    # Latin hypercube sampling
    nloptr,                 # Global optimization
    ParBayesianOptimization,# Bayesian optimization
    numDeriv,               # Hessian computation
    future.apply            # Parallel bootstrap
```

---

## Key Metrics Summary

| Metric | Current | Proposed | Source |
|--------|---------|----------|--------|
| BR Error | Point estimate | Bounds + CI | Both |
| Q* | Point estimate | Bootstrap CI | Both |
| Convergence | Gradient plot | Certified status | GPT 5.2 |
| Geometry | None | Hessian eigenvalues | Gemini 3 Pro |
| Robustness | None | Perturbation score | Both |
| Dimensions | None | Regret decomposition | GPT 5.2 |

---

## References

- GPT 5.2 recommended `optim()`, `nloptr`, `lhs` for multi-start optimization
- Gemini 3 Pro recommended `ParBayesianOptimization`, `numDeriv` for BO and Hessian
- Both emphasized parallelization via `future`/`parallel` for bootstrap
- Both recommended S3 validation report object for consistent UX
