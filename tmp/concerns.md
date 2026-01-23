# Validated concerns

1) Approximate SEs for neural models (delta method + diagonal variance only)
- The neural outcome model builds `vcov_OutcomeModel` as a vector of per-parameter variances (no covariance), and downstream SEs are computed via a Jacobian-based delta method. For the transformer (highly nonlinear), this is an approximation that can understate uncertainty; it is not documented in the code path that produces `q_star_se`/`pi_star_se`.
- References: `strategize/R/CS_2Step_ModelOutcome_neural.R:1848`, `strategize/R/CS_2Step_ModelOutcome_neural.R:1875`, `strategize/R/CS_2Step_Master.R:1859`, `strategize/R/CS_2Step_Master.R:1868`.

2) Stage indicator encoding differs between neural vs GLM stage logic
- Neural models set `stage_idx = 1` when `party_left == party_right` (same-party), while the GLM stage indicator uses `1 = "Different" (general)`. This mismatch is internally consistent for the neural path but can confuse interpretation or comparisons across model types unless explicitly documented.
- References: `strategize/R/CS_2Step_ModelOutcome_neural.R:674`, `strategize/R/CS_2Step_Optimize_GetQ.R:370`, `strategize/R/CS_2Step_ModelOutcome.R:209`.

3) No explicit length/shape validation in `neural_params_from_theta`
- `neural_params_from_theta()` reshapes slices of `theta_vec` based on offsets/sizes but does not verify that the vector length matches expected totals. If an incorrect `theta_vec` is paired with a `model_info` object, parameters can be silently mis-shaped or error deep in JAX, making debugging harder.
- Reference: `strategize/R/CS_2Step_Optimize_GetQ.R:72`.
