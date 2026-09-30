"""
Extra per-frame/per-query fields to append to a RECAP/LeRobot dataset.

Existing Destiny000621/RECAP fields are preserved:
  intervention, reward, reward_label, value_label, adv_ind

CU-RECAP adds:
  query_requested       bool
  query_propensity      float in (0,1]
  query_reason          none | utility | safety | human
  correction_id         integer, -1 when not corrected
  correction_cost_s     human takeover duration in seconds
  utility_target        one-step LoRA probe gain (queried states only)
  utility_fo            first-order gradient-alignment estimate
  utility_grad_cosine   diagnostic
  failure_label         future failure / intervention-risk target
  policy_uncertainty    optional baseline score
  acquisition_score     score used online
  acquisition_dual      current Lagrange multiplier
"""

CU_RECAP_COLUMNS = (
    "query_requested",
    "query_propensity",
    "query_reason",
    "correction_id",
    "correction_cost_s",
    "utility_target",
    "utility_fo",
    "utility_grad_cosine",
    "failure_label",
    "policy_uncertainty",
    "acquisition_score",
    "acquisition_dual",
)
