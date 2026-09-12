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
