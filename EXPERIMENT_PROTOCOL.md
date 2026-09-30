# CU-RECAP experiment protocol v0.2

## Claims that must be tested

H1. Execution risk and correction utility are not equivalent.
H2. Post-hoc one-step LoRA probe gain predicts actual downstream learning benefit.
H3. A pre-query critic can predict correction utility with useful ranking accuracy.
H4. Utility-aware querying improves policy performance per unit of human correction time.
H5. The gain survives distribution shift / recovery conditions.

## Study 1 — Is correction utility a real quantity?

Collect a broad exploration dataset using random + risk-based human interventions.

For each correction c:
1. compute first-order gradient utility U_FO;
2. compute exact temporary one-step LoRA probe gain U_1step;
3. for a stratified subset, perform a fuller micro-update and evaluate U_oracle.

Report:
- Spearman rank correlation: U_FO vs U_1step
- Spearman: U_1step vs U_oracle
- top-k precision for identifying the most useful corrections
- scatter plots with confidence intervals

If these correlations are weak, the paper's central method must be revised before
running expensive real-robot acquisition experiments.

## Study 2 — Risk != utility

For every queried state, store both:
- predicted/realized failure risk
- post-hoc correction utility

Analyze four quadrants:
- high risk / high utility
- high risk / low utility
- low risk / high utility
- low risk / low utility

Report rank correlation and examples. The paper needs convincing high-risk/low-utility
and moderate-risk/high-utility cases, not just aggregate numbers.

## Study 3 — Budgeted acquisition

Methods:
B0 Random querying
B1 Policy uncertainty
B2 Failure-risk querying
B3 Novelty / embedding distance
B4 Utility critic without IPW
B5 CU-RECAP (utility + uncertainty + IPW + cost constraint)

All methods receive the same:
- initial policy checkpoint
- rollout opportunities
- task distribution
- human-control-seconds budget

Primary curve:
  task success vs cumulative human-control seconds

Primary summary:
  area under the success-vs-human-budget curve

Secondary:
- interventions per successful policy improvement
- unique failure-mode coverage
- correction duration
- final success after fixed budget

## Study 4 — RECAP integration

After each matched-budget collection round:
- run identical RECAP value training
- identical N=50 advantage relabel
- identical policy fine-tuning schedule

The ONLY experimental difference is the acquisition policy.

This isolation is important: otherwise reviewers cannot tell whether the gain came from
better correction selection or different post-training.

## Real robot tasks

Tianyi 2.0 + binary open/close hand abstraction.

R1 Pick-and-place with pose randomization
R2 Cluttered target retrieval
R3 Precision placement
R4 Recovery under controlled object displacement / grasp perturbation

Use paired initial-condition sets across methods.

## Statistics

Do not rely on one final success-rate number.

- report 95% bootstrap confidence intervals;
- use paired initial conditions where possible;
- report per-task as well as aggregate results;
- use multiple independent training seeds in simulation;
- for real robot, pre-register a fixed set of evaluation configurations;
- analyze success with a mixed-effects logistic model or paired binary test where
  appropriate.

A 10 percentage-point success difference can require hundreds of independent trials
for conventional 80% power, so treat 50 real-robot episodes as a coarse estimate, not
strong evidence for small improvements. Aim for large effect sizes or pooled,
pre-specified hierarchical analyses.
