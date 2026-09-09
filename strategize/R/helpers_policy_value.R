# CV must integrate predictions over actual profiles. Applying QFXN to mean
# features is invalid for nonlinear links. The caller supplies policies and
# logits learned without the evaluation sample's outcomes.
cs_cv_policy_value <- function(training, evaluation, outcome_model_type, diff,
                                adversarial, force_gaussian, n_draws = 1000L,
                                temperature = 0.5, primary_pushforward = "mc",
                                primary_n_entrants = 1L, primary_n_field = 1L,
                                seed = 123L) {
  gather <- evaluation$gather_fxn
  ast <- gather(evaluation$REGRESSION_PARAMETERS_ast)
  dag <- gather(evaluation$REGRESSION_PARAMETERS_dag)
  common <- list(
    pi_star_ast = training$pi_star_red_ast,
    pi_star_dag = training$pi_star_red_dag,
    INTERCEPT_ast_ = ast[[1L]], COEFFICIENTS_ast_ = ast[[2L]],
    INTERCEPT_dag_ = dag[[1L]], COEFFICIENTS_dag_ = dag[[2L]],
    seed_in = strenv$jax$random$PRNGKey(as.integer(seed)), phase = "report",
    outcome_model_type = outcome_model_type,
    glm_family = if (!is.null(evaluation$glm_family)) evaluation$glm_family else
      if (isTRUE(force_gaussian)) "gaussian" else "binomial",
    nMonte_Qglm = as.integer(n_draws), temperature = temperature,
    ParameterizationType = evaluation$ParameterizationType,
    d_locator_use = evaluation$d_locator_use
  )
  if (!isTRUE(adversarial)) {
    value <- do.call(evaluate_average_case_q, c(common, list(
      q_fxn = evaluation$QFXN, single_party = !isTRUE(diff))))$q_max
  } else {
    ast0 <- gather(evaluation$REGRESSION_PARAMETERS_ast0)
    dag0 <- gather(evaluation$REGRESSION_PARAMETERS_dag0)
    value <- do.call(evaluate_adversarial_q, c(common, list(
      a_i_ast = training$a_i_ast, a_i_dag = training$a_i_dag,
      INTERCEPT_ast0_ = ast0[[1L]], COEFFICIENTS_ast0_ = ast0[[2L]],
      INTERCEPT_dag0_ = dag0[[1L]], COEFFICIENTS_dag0_ = dag0[[2L]],
      P_VEC_FULL_ast_ = evaluation$P_VEC_FULL_ast,
      P_VEC_FULL_dag_ = evaluation$P_VEC_FULL_dag,
      SLATE_VEC_ast_ = evaluation$SLATE_VEC_ast,
      SLATE_VEC_dag_ = evaluation$SLATE_VEC_dag,
      LAMBDA_ = strenv$jnp$array(evaluation$lambda),
      Q_SIGN = strenv$jnp$array(1.), nMonte_adversarial = as.integer(n_draws),
      primary_pushforward = primary_pushforward,
      primary_n_entrants = primary_n_entrants,
      primary_n_field = primary_n_field)))$q_ast
  }
  as.numeric(strenv$np$array(value))[[1L]]
}
