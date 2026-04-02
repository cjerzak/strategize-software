# Residual Attention JAX Refactor Plan

Date: 2026-04-01

## Goal

Refactor the full residual-attention forward path so that:

1. prediction and validation run inside one stable outer `jax.jit`,
2. posterior/draw prediction uses `vmap` instead of an R loop,
3. the `full_attn` history path stops rebuilding stacked history from an R list on every residual read,
4. training, validation, and serving reuse the same forward implementation instead of carrying near-duplicate eager code.

## Current Problems

### 1. Prediction is eager and fragmented

The serve/eval path calls `neural_run_transformer()` directly from R without a persistent `jax.jit` boundary.

Affected surfaces:

- `strategize/R/prediction_api.R`
  - `cs2step_neural_predict_pair_prepared()`
  - `cs2step_neural_predict_single_prepared()`
  - `cs2step_neural_predict_internal()`
- `strategize/R/two_step_model_outcome_neural.R`
  - `TransformerPredict_pair()`
  - `TransformerPredict_single()`
  - `svi_validation_predict()`

### 2. Draw prediction uses a host loop

`cs2step_neural_predict_internal()` loops over draws in R, rebuilds params draw-by-draw, and reruns the full transformer once per draw.

This should become a batched `vmap` prediction path over `theta_draws`.

### 3. `full_attn` history is repeatedly restacked

`neural_run_transformer()` accumulates `layer_outputs` as an R list and each call to `neural_full_attn_residual()` restacks the entire history.

This is both memory-heavy and a poor fit for a large fused compiled step.

### 4. Validation falls back to host-heavy eager execution

SVI early-stopping validation rebuilds the forward path eagerly and converts predictions back to R on each check.

This adds extra device dispatch and host synchronization in the middle of optimization.

## Target Architecture

### A. One canonical pure forward core

Create one pairwise core and one single-observation core that:

- accept only JAX arrays plus static model metadata,
- return logits or typed prediction outputs,
- contain the full token-build -> transformer -> head path,
- are reused by training-adjacent validation, fitted predictor functions, and public prediction API helpers.

Recommended functions:

- `neural_predict_pair_core_prepared()`
- `neural_predict_single_core_prepared()`
- `neural_predict_from_theta_prepared()`

These should live next to the transformer implementation in `strategize/R/two_step_model_outcome_neural.R`, not as duplicated wrappers spread across files.

### B. Cached outer compiled entrypoints

Add a small JIT cache keyed by:

- mode: `pairwise` or `single`
- output mode: `response` or `logits`
- likelihood
- residual mode
- cross-encoder mode
- model depth / dimensions / head count

Do not create fresh jitted closures inside loops or per request. Compile once per stable signature and reuse.

Recommended environment:

- `cs2step_neural_jit_cache <- new.env(parent = emptyenv())`

Recommended constructors:

- `cs2step_neural_get_predict_jit(model_info, pairwise, return_logits)`
- `cs2step_neural_get_predict_from_theta_jit(model_info, pairwise, return_logits)`

### C. Batched draw path

Replace the R draw loop with:

1. build `theta_draws` as one JAX array of shape `[n_draws, n_params]`,
2. `vmap` a pure `predict_from_theta(theta_i, prep)` function,
3. convert the final stacked result back to R once.

This preserves exact semantics while removing the independent host loop.

### D. Tensor-based residual history

Refactor `full_attn` history handling so the residual source history is a JAX tensor, not an R list that gets restacked at each query.

Preferred progression:

1. Phase 1: replace list-plus-stack with explicit tensor history plus append helper.
2. Phase 2: if memory still dominates, switch to fixed-capacity history storage using `lax.dynamic_update_slice` or indexed `.at[...]` writes.

The phase-1 version is the lowest-risk change and should happen first.

## Implementation Plan

### Phase 1. Consolidate the forward graph

Status: complete on 2026-04-01.

Files:

- `strategize/R/two_step_model_outcome_neural.R`
- `strategize/R/prediction_api.R`

Actions:

1. Extract duplicated pairwise forward logic from:
   - `cs2step_neural_predict_pair_prepared()`
   - `TransformerPredict_pair()`
   - `svi_validation_predict()`

2. Extract duplicated single forward logic from:
   - `cs2step_neural_predict_single_prepared()`
   - `TransformerPredict_single()`
   - `svi_validation_predict()`

3. Move those into shared pure helpers in `two_step_model_outcome_neural.R`.

4. Make `prediction_api.R` call the shared helpers instead of carrying its own copy of the transformer/token path.

Acceptance criteria:

- no duplicated serve/eval transformer path remains across those three call sites,
- pairwise and single prediction still match current outputs exactly in existing tests.

### Phase 2. Add outer `jax.jit` for prepared prediction

Status: complete on 2026-04-01.

Files:

- `strategize/R/two_step_model_outcome_neural.R`
- `strategize/R/prediction_api.R`

Actions:

1. Build a cached jitted wrapper around the prepared pairwise core.
2. Build a cached jitted wrapper around the prepared single core.
3. Use separate compiled functions for logits vs response output instead of a dynamic `return_logits` branch if that improves cache stability.
4. Route:
   - `cs2step_neural_predict_prepared()`
   - `TransformerPredict_pair()`
   - `TransformerPredict_single()`
   through those cached compiled functions.

Notes:

- `model_depth`, `residual_mode`, likelihood, and cross-encoder mode should stay outside the dynamic array arguments and participate in cache selection.
- Avoid compiling inside per-request code after the first cache fill.

Acceptance criteria:

- first call compiles, subsequent same-shape calls reuse the compiled path,
- prediction outputs are numerically unchanged,
- no fresh `jit` construction happens inside draw loops or validation loops.

### Phase 3. Replace R draw loop with `vmap`

Status: complete on 2026-04-01.

Files:

- `strategize/R/prediction_api.R`
- `strategize/R/two_step_model_outcome_neural.R`

Actions:

1. Add a pure helper:
   - `neural_predict_from_theta_prepared(theta_vec, model_info, prep, return_logits)`

2. Internally:
   - unpack params with `neural_params_from_theta()`,
   - call the shared prepared forward core.

3. Add:
   - `vmap_predict_from_theta = jax.vmap(...)`
   - outer `jax.jit(vmap_predict_from_theta)`

4. In `cs2step_neural_predict_internal()`, replace:
   - the R `for (i in seq_len(n_draws))` loop
   with one batched call on `theta_draws`.

Notes:

- If `n_draws` varies frequently and recompilation becomes annoying, bucket draw counts or keep the batched path for interval requests only.
- Convert the final batched output back to R once per request.

Acceptance criteria:

- same predictive intervals as current code up to floating-point tolerance,
- draw prediction no longer runs one transformer call per draw from R.

### Phase 4. Refactor residual history storage

Status: complete on 2026-04-01 for the phase-1 tensor-history refactor.

Files:

- `strategize/R/two_step_model_outcome_neural.R`

Actions:

1. Replace `layer_outputs <- list(tokens)` plus repeated `stack()` in `neural_full_attn_residual()` with a tensor history representation.
2. Introduce helpers:
   - `neural_init_residual_history(tokens)`
   - `neural_append_residual_history(history, x)`
   - `neural_full_attn_residual_from_history(history, pseudo_query, model_dims, n_used = NULL)`

3. Keep the external semantics of `full_attn` unchanged.

Preferred shape:

- history tensor: `[depth_state, batch, token, model_dim]`

Phase-1 storage strategy:

- append with explicit tensor ops,
- pass the tensor directly to residual attention,
- stop accepting an R list in the internal hot path.

Phase-2 optional optimization:

- preallocate max size `1 + 2 * model_depth`,
- update by index,
- pass only active prefix to the residual read helper.

Acceptance criteria:

- identical outputs for `standard` and `full_attn`,
- no `jnp$stack(layer_outputs, ...)` on the hot residual path.

### Phase 5. Move validation prediction onto the shared compiled path

Status: complete on 2026-04-01.

Files:

- `strategize/R/two_step_model_outcome_neural.R`

Actions:

1. Make `svi_validation_predict()` call the shared prepared prediction helper instead of locally rebuilding the transformer path.
2. Keep predictions on device until the final metric computation boundary whenever possible.
3. Convert to R only once per validation check, or move the metric computation to JAX if the current R metric path becomes the next bottleneck.

Acceptance criteria:

- no duplicate eager validation forward path remains,
- early-stopping checks reuse the same compiled serve/eval forward graph.

## Code-Level Notes

### `neural_params_from_theta()`

Keep the schema static and reuse it. The current implementation is acceptable as a traced unpack step as long as:

- the schema is built once,
- `theta_vec` shape stays stable,
- the function is called inside one compiled outer boundary for batched prediction.

Do not rebuild parameter schema inside the draw loop.

### `return_details`

Split compiled entrypoints by output type instead of treating `return_details` as a dynamic flag in the hottest serve path.

Suggested compiled variants:

- logits only
- response only
- details only when needed for special internals

### Prediction cache

Cache params separately from compiled functions.

Current param cache is useful:

- `cs2step_neural_param_cache`

Add a second cache for compiled callables rather than mixing the two concerns.

## Validation Plan

Status: partially complete on 2026-04-01.
Completed:
- parse checks for the modified R files
- `test-neural.R` file run under `testthat` with existing CRAN guards
- direct `full_attn` bundle interval validation through the public `predict_pair(..., interval = "ci")` API, confirming deterministic repeated output and ordered intervals on the new vmapped draw path

### Correctness tests

Files:

- `strategize/tests/testthat/test-neural.R`
- `strategize/tests/testthat/test-prediction-api.R`

Add tests for:

1. `full_attn` pairwise prediction parity:
   - old eager path vs new jitted prepared path

2. `full_attn` single prediction parity:
   - old eager path vs new jitted prepared path

3. draw interval parity:
   - old per-draw loop vs new vmapped draw path

4. cross-attention and cross-term variants:
   - pairwise `term`
   - pairwise `attn`
   - pairwise `full`

5. schema reuse:
   - repeated calls do not mutate outputs or fail after cache reuse

### Performance checks

Add a temporary benchmark script under `tmp/` that:

1. warms up once,
2. times repeated prediction calls only after warmup,
3. uses `.block_until_ready()` before stopping the timer,
4. compares:
   - current eager prepared prediction
   - jitted prepared prediction
   - vmapped draw prediction

Suggested benchmark cases:

- single prediction, batch size 32
- pairwise prediction, batch size 32
- pairwise prediction with `full_attn`
- interval prediction with 200 draws

### Memory checks

After the history refactor, inspect:

- batch-size sensitivity of `full_attn`,
- latency increase with model depth,
- whether the history tensor version reduces repeated allocations relative to list-plus-stack.

If needed, add a compile-time analysis follow-up using lowered/compiled objects.

## Rollout Order

1. Consolidate duplicated forward code.
2. Add cached jitted prepared prediction.
3. Convert draw path to `vmap`.
4. Refactor residual history representation.
5. Repoint SVI validation to the shared compiled path.
6. Benchmark and tune only after correctness is locked.

This order keeps risk low:

- Phase 1 reduces duplication before changing performance behavior.
- Phase 2 gives the largest practical latency win.
- Phase 3 removes the biggest remaining host loop.
- Phase 4 improves the `full_attn` hotspot without mixing it into the initial API refactor.

## Non-Goals

This plan does not currently require:

- `pmap`
- `shard_map`
- explicit sharding APIs
- host offloading
- custom partitioning

Those are follow-on options only if single-host compiled prediction remains insufficient after the above work.

## Definition of Done

The refactor is complete when:

1. the residual-attention prediction path runs through one cached outer `jax.jit`,
2. draw prediction uses one vmapped compiled call instead of an R loop,
3. `full_attn` no longer rebuilds residual history from an R list on every residual read,
4. training-adjacent validation and public prediction reuse the same forward core,
5. existing tests pass and new parity tests cover the refactor,
6. benchmarks show warm prediction latency improvement for `full_attn` prediction and draw intervals.
