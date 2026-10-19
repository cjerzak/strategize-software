# Development: October 19, 2026 exact grouped MoE GEMM and one level-text gather

- Training traces on GPU compute routed experts with an exact grouped GEMM
  written in Pallas Triton (`strategize_moe_gmm.py`). XLA GPU (JAX 0.11.1)
  lowered `ragged_dot` as a dense product over all 8 experts, with
  expert-sized mask and transpose temporaries. The new layout is expert-major,
  and each expert starts on a 64-row tile, so every tile multiplies by one
  expert's weights. A `custom_vjp` runs the same kernel for dX and a grouped
  `x^T dy` kernel for dW.
  - Work is now proportional to the routed assignments: one MoE layer's
    forward+backward at batch 16 fell from 16-21 ms to 3.0 ms.
  - `STRATEGIZE_MOE_EXPERT_KERNEL=auto` now means `pallas` on GPU. TPUs keep
    `grouped`, and `grouped` or `blocks` still override it.
  - Shapes the kernel does not tile (width or expert hidden size not a
    multiple of 64) fall back to `grouped`.
  - JAX's Pallas Triton backend looks GPUs up in a fixed name table. Unlisted
    CUDA GPUs, such as the GeForce RTX 5080 Laptop (sm_120), are registered
    by compute capability through a private JAX module (`register_device`).
- The hard factor-fused token path projects every factor's level text with
  one concatenated table per weight and gathers rows once
  (`neural_gather_level_tokens`). The per-factor loop's backward wrote one
  weight-sized cotangent per factor slot (719 in the foundation pool) and
  summed them.
  - Values and gradients agree with the loop to FP32 rounding (at most
    5e-7 absolute and 2e-7 relative on GPU).
  - `STRATEGIZE_LEVEL_TEXT_GATHER=0` restores the loop. The soft path is
    unchanged.
- Two opt-in reduced-precision switches (both default to float32):
  - `STRATEGIZE_MUON_NS_DTYPE=bfloat16` runs Muon's Newton-Schulz
    orthogonalization in bf16: 74 to 24 ms per step, update cosine 0.9993. It
    changes the optimizer trajectory and needs a longer A/B before adoption.
  - `STRATEGIZE_DP_REDUCE_DTYPE=bfloat16` all-reduces data-parallel gradients
    in bf16 (the loss stays f32).
- Measured on the 576-wide, 8-layer looped MoE foundation model (RTX 4090
  Laptop, batch 16, FP32 at highest precision, 48 updates):

  | Configuration | s per 8 updates | Peak memory | Compile |
  | --- | ---: | ---: | ---: |
  | Previous implementation | 7.2 | 9.76 GB | 336 s |
  | Pallas experts | 4.05 | 9.76 GB | 316 s |
  | Pallas experts + level-text gather | 3.42 | 5.39 GB | 233 s |

  - Validation NLL agreed within GPU run-to-run noise (0.7518-0.7524).
  - Two-GPU data parallel over Thunderbolt (global batch 32) fell from 11.25
    to 6.1 s per 8 updates (5.3 s with the bf16 reduce).
  - A single 4090 at batch 32 is faster still: 4.5 s per 8 updates and 7.7 GB
    peak, because the f32 all-reduce of the 352 MB of gradients takes about
    270 ms over the link.
- `strategize_moe_gmm.py` joins the runtime identity recorded in full-state
  checkpoints, so checkpoints written before this change cannot be resumed
  with it.

# Development: October 18, 2026 grouped MoE experts and checkpointed recurrences

- Training traces on GPU/TPU compute routed experts with grouped
  `jax.lax.ragged_dot` matmuls (`strategize_moe.grouped_experts`). The block
  loop's backward zero-filled, added and copied full-size expert-weight and token
  cotangents for every 64-assignment block, about half of all GPU time per
  update. The un-permute scatter declares `unique_indices`.
- Prediction, inference and other vmapped callers keep the block loop and are
  unchanged. So do vectorized particles, forward-mode SVI updates and CPU, where
  ragged_dot lowers to one dense product per expert.
  `STRATEGIZE_MOE_EXPERT_KERNEL` (`auto`, `grouped` or `blocks`) overrides the
  training choice.
- `run_layers` checkpoints each recurrence and saves only its layer boundaries
  and MoE routing decisions. The sampled-depth `cond` had made reverse mode
  stack a copy of the loop-invariant core weights per iteration. Gradients,
  rematerialization work and the routing `all_gather` count are unchanged.
- Measured on the 576-wide, 8-layer looped MoE foundation model (RTX 4090
  Laptop, batch 16, FP32), against the previous implementation:
  - grouped experts: updates 27% faster;
  - recurrence checkpoint: peak memory 12.85 to 9.78 GB;
  - both: updates 32% faster (1.43 to 0.97 s);
  - batch 32 now fits at the default memory fraction (peak 9.55 GB; it
    previously failed to allocate 11 GiB);
  - validation NLL agreed within GPU run-to-run noise.
- On XLA GPU (JAX 0.11.1) ragged_dot still costs E times the useful expert
  FLOPs, so a true grouped-GEMM kernel would be faster again.
- `strategize_moe.py` and `strategize_transformer.py` are part of the runtime
  identity recorded in full-state checkpoints, so checkpoints written before
  this change cannot be resumed with it.

# Development: October 17, 2026 decoupled weight decay for Muon

- `neural_mcmc_control$svi_weight_decay` (default 0) applies decoupled weight
  decay to Muon's hidden matrices. Optax applies it as `lr(t) * decay * p`
  per update; the auxiliary Adam branch is never decayed. With
  `svi_weight_decay_batch_scaling = TRUE` (the default) the decay also takes
  the LR batch factor from `svi_lr_batch_scaling`, so peak `lr * decay` grows
  in proportion to the batch and decay per epoch matches the reference batch.
  The rule follows sailfast-software. Other optimizers record the decay as not
  applied. `optimizer_diagnostics$weight_decay` reports base, factor,
  effective value and status, and the bridge advertises `svi_weight_decay_v1`.

# Development: October 15, 2026 linear-cost MoE expert backward

- The routed-expert blocks in `strategize_moe.selected_experts` return their
  rows and are scattered once after the loop. The loop used to carry the whole
  `(assignments, d)` output, so each backward iteration touched every
  assignment: quadratic in the routing-group size (4x per doubling on GPU at
  width 576). Results are bit-identical. On the 88.9M production model at batch
  32, expert forward+backward is 1.4-1.8x faster, single-GPU training 11-13%
  faster, and the scanned update compiles in about half the time.
- `strategize_moe.py` is part of the runtime identity recorded in full-state
  checkpoints, so checkpoints written before this change cannot be resumed
  with it.

# Development: October 14, 2026 canonical id maps match by name

- `cs_foundation_normalize_factor_ids()` and `cs_foundation_normalize_level_ids()`
  match named `canonical_factor_id` / `canonical_level_id` maps by name. A bare
  `as.character()` had dropped the names, so a map listed in a different order
  than the experiment's factors or levels was applied by position (the
  postbrexit crosswalk swapped two factor ids and, in cj2, two level ids). Only
  unnamed maps are positional. preference.fm carries the same fix.

# Development: October 11, 2026 maximum-likelihood fits and residual init policy

- `neural_mcmc_control$uncertainty_scope = "none"` fits every site as a point
  estimate: no priors or hyperpriors, no KL term and no variational guide, so
  the objective is the per-observation-normalized log-likelihood plus the
  existing auxiliary factors. `theta_var` is zero; interval draws and
  `compute_se = TRUE` stop with an explicit error. The default remains `"all"`.
- `neural_mcmc_control$residual_gate_init` sets the initial value of the trunk
  residual gates (`alpha_attn_l*`, `alpha_ff_l*`; still trained), and
  `output_proj_init = "depth_scaled"` multiplies the init sd of the projections
  that write into the residual stream (`W_o`, `W_ff2`, MoE `expert2`/`shared2`)
  by `1/sqrt(2 * effective depth)`. Both default to the previous behaviour. In
  the v2r3 foundation fit the 0.035-0.05 gates left the 14-block trunk
  functionally inert (removing it changed held-out AUC by 0.002).
- `strategize_fm_backend()` advertises `uncertainty_scope_none_v1` and
  `residual_init_policy_v1`.

# Development: September 12, 2026 text-embedding input normalization

- Schema text embeddings are centered with the mean of the training schema
  text and re-normalized before the `W_*_name_text`, `W_experiment_text`, and
  covariate-value projections (`text_embedding_normalization`, default on;
  optional `remove_top_pcs`). Sentence encoders leave one shared direction in
  every vector (about 60% of each unit vector for harrier), and uncentered
  projections spent 70-89% of every factor/level token on that constant in the
  576x8 Muon MoE foundation fit while the 192x8 Adam fit learned to suppress
  it. The fitted normalizer is stored in `neural_model_info` and applied
  identically during training validation, prediction, and adaptation; saved
  models without one remain uncentered.
- Structural features `raw_value_log1p_signed` and `cardinality_log` use the
  bounded encoding `tanh(signed_log1p(x) / 4)` (`struct_feature_encoding =
  "bounded_v2"`, default); saved models record their encoding and the
  prediction-time default builders honor it (`"legacy_v1"` for older models).
- `neural_model_info$text_pathway_diagnostics` reports, per text pathway, the
  constant share of token energy, the gain applied to the raw shared
  direction, and the separability of projected rows.

# Development: September 9, 2026 correctness repairs

- Binomial policy reports and CV integrate categorical profiles, with exact
  enumeration on small supports and hard Monte Carlo otherwise. The GLM
  optimizer's relaxation is confined to its optimization objective. Discrete
  reporting retains likelihood-ratio derivatives for uncertainty calculations.
- Pairwise interaction screening, including nested predictive evaluation, now
  regularizes the actual differences of within-profile products with glmnet.
  Refits retain all main effects and all levels of selected factor pairs.
- Average-case K=1 crossfit can select penalties using design overlap constraints,
  retain candidate diagnostics and respondent evaluation contributions, and
  evaluate marginal probability recipes from the same training-fold models.
  Evaluation Monte Carlo draws are independent across respondent clusters.
- Results record a correctness contract so downstream reporting can reject
  pre-repair fitted artifacts. Existing results require a new fit.

# strategize 0.0.2

## Statistical fixes

* **Position-intercept fix for K=1 pairwise binomial GLM (behavior change).**
  The differenced forced-choice outcome GLM is fit on the "profile shown in
  position 1" orientation, so its intercept is a display-position effect, not
  an attribute effect. Two consequences are fixed:

  - The model is now fit **without an intercept by default** for K=1,
    `diff = TRUE`, non-adversarial, binomial fits — matching the
    position-marginalized estimand (an antisymmetric pair model on the logit
    scale). `Q_reference_in_sample` is now 0.5 by construction for these fits
    (previously `plogis(b0)`), and `Q_point*`, `Q_gain*`, `Q_se`,
    `outcome_model_view$baseline`, and penalized/gradient-descent optimized
    policies no longer absorb the position offset. The legacy behavior is
    available via `options(strategize.glm_position_intercept = TRUE)`.

  - Cross-fitting (`crossfit_q = TRUE`) evaluated each held-out pair in both
    display orientations but applied `+intercept` to both, systematically
    mispredicting the swapped half; hypothetical policy-vs-opponent model
    values also carried the `+intercept`. `cs_crossfit_q_fold_eval` now scores
    the fit-canonical orientation once and uses the forced-choice complement
    for the swapped duplicates, and `cs_crossfit_q_policy_model_mu`
    marginalizes over display position (exact for binomial and gaussian
    families; identical to previous behavior when the fitted intercept is 0).

* Degenerate all-columns-aliased GLM refits under the intercept-free regime now
  fit the empty model (`Y ~ 0`, every prediction 0.5) rather than re-estimating
  the position effect via `Y ~ 1`.

* Fixed a latent JAX shape bug where the intercept-free path stored a rank-0
  intercept array, which broke `jnp$concatenate` with the (n, 1) coefficient
  array.

## Known residuals (documented, unchanged)

* Gaussian `diff = TRUE` fits (ratings and forced-choice LPM) retain an
  intercept that may absorb position effects in-sample; their cross-fit
  evaluation is position-correct via the complement/marginalization fix.
* Adversarial same-party Round-0 models orient pairs by profile order, so
  their intercepts are position effects; cross-party adversarial intercepts
  are substantive group effects and are unaffected by design.
