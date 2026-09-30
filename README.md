# CU-RECAP v0.2

Research scaffold for **Correction-Utility-Aware RECAP**.

## Core hypothesis

The state most likely to fail is not necessarily the state most worth correcting.

We distinguish:

- **execution risk**: probability that the current rollout fails;
- **correction utility**: expected downstream policy improvement if a human correction
  is obtained at this state.

## Utility definition

For a correction chunk c and held-out hard-state probe buffer H:

    theta_c = theta - eta * grad L_c(theta)

    U(c) = L_H(theta) - L_H(theta_c)

A first-order approximation is:

    U_FO(c) = eta * g_H^T g_c

Note the sign: positive gradient alignment predicts positive held-out loss reduction.

The high-quality post-hoc target is computed with a temporary **one-step LoRA update**.
The base policy is restored immediately afterwards.

## Why propensity logging is mandatory

Correction utility can only be observed after a state is queried. Once the online critic
starts choosing queries, the observed utility dataset becomes selection-biased.

CU-RECAP therefore uses a stochastic acquisition policy, logs the query probability
(propensity), and trains the utility head with clipped inverse-propensity weighting.

This is a central part of the method rather than an implementation detail.

## Online acquisition

The ensemble critic predicts:

- expected correction utility;
- uncertainty in that utility;
- execution failure risk;
- expected correction duration.

Learning queries use:

    utility_UCB - lambda * predicted_cost

where lambda is updated online to satisfy a human-control-time budget.

Safety interventions are a separate deterministic override and are never justified by
learning utility alone.

## Integration with Destiny000621/RECAP

Keep all existing columns:

- intervention
- reward
- reward_label
- value_label
- adv_ind

Append the fields in `cu_recap/data/schema.py`.

The RECAP policy/value training loop can remain unchanged initially. CU-RECAP changes
**which human corrections are collected**, not how RECAP consumes the final dataset.

## Repository contents

- `paper/PAPER_DRAFT.md` — working ICRA/IROS-level paper draft
- `docs/RESEARCH_PLAN.md` — overall research thesis, scope, and go/no-go milestones
- `EXPERIMENT_PROTOCOL.md` — budget-controlled experiment protocol
- `cu_recap/utility/` — correction-utility estimators
- `cu_recap/critic/` — pre-query utility critic and selective-feedback losses
- `cu_recap/acquisition/` — budgeted stochastic query policy
- `cu_recap/data/` — extra RECAP/LeRobot metadata schema

## Recommended implementation order

1. Integrate query metadata into real/sim rollout collection.
2. Build a fixed hard-state probe buffer.
3. Validate `U_FO` against exact one-step LoRA probe gain.
4. Train utility critic offline from an initial exploration round.
5. Turn on stochastic utility querying + propensity logging.
6. Compare against random, uncertainty and failure-risk querying under identical
   human-control-second budgets.
7. Only after the above is stable, add iterative RECAP policy updates.
