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
