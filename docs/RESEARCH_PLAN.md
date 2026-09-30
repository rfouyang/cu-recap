# CU-RECAP Research Plan

## One-sentence thesis

**The state most likely to fail is not necessarily the state most worth correcting.**

CU-RECAP asks which human correction is expected to improve the future VLA policy the most under a limited human-control-time budget.

## What is new

The paper is not about adding critic heads, plugging Jev/Laya into RECAP, or reranking action chunks. The core method is:

1. define a measurable post-hoc correction utility;
2. learn to predict that utility before asking for feedback;
3. account for selective-feedback bias through stochastic queries + logged propensity;
4. allocate a fixed human-control-time budget using utility and uncertainty;
5. feed the selected corrections into an otherwise unchanged RECAP pipeline.

## Key distinction

### Execution risk

`P(failure | current state)`

Answers: **Will the robot probably fail?**

### Correction utility

`E[downstream policy improvement | obtain a human correction here]`

Answers: **Is this state worth spending human correction effort on?**

## Core training loop

```text
VLA rollout
    |
    v
pre-query Utility Critic
    |-- failure risk --------> safety override (independent)
    |
    |-- utility mean/std
    |-- predicted human cost
    v
budgeted stochastic acquisition
    |
    | query probability p_t is logged
    v
human correction (if queried)
    |
    v
post-hoc utility labeling
    |-- one-step LoRA probe gain (target)
    |-- first-order gradient alignment (proxy/diagnostic)
    v
IPW utility-critic training
    |
    v
RECAP dataset + value model + N-step advantage + policy update
    |
    +------------------------ next iteration
```

## Why a hard-state probe buffer is needed

A correction should be valued according to whether it helps the policy's current weaknesses, not just whether the correction itself has low imitation loss.

Maintain a held-out buffer containing previous failures, recovery states, near-intervention states, controlled perturbations, and optionally OOD configurations.

## Why selection bias matters

Once the critic chooses where to query, labels are observed only on selected states. Therefore learning queries are stochastic, the exact query probability is recorded, and utility regression uses clipped inverse-propensity weights.

## Scope decisions

### Keep

- RECAP as downstream post-training infrastructure;
- fast temporal multimodal utility critic;
- one-step LoRA utility target;
- first-order gradient proxy;
- stochastic propensity-logged acquisition;
- human-time cost constraint;
- Tianyi 2.0 real-robot validation;
- Inspire hand treated as binary open/close.

### Do not put in the core paper yet

- action-chunk reranking;
- world-model rollouts;
- full dexterous-hand learning;
- direct Jev/Laya dependency;
- low-level learned safety.

## Go / no-go milestones

### M1 — Utility proxy validation

Show useful rank correlation `U_FO -> U_1step -> U_oracle`.

### M2 — Risk != utility

Demonstrate meaningful high-risk/low-utility and moderate-risk/high-utility examples.

### M3 — Offline utility prediction

Pre-query critic should rank high-utility corrections better than risk, uncertainty, and novelty baselines.

### M4 — Budgeted acquisition

Under identical human-control seconds, utility-aware querying should produce clearly better learning curves.

### M5 — End-to-end RECAP

After identical RECAP training, the advantage should persist in simulation and real-robot evaluation.

## Intended headline figures

1. **Task success vs cumulative human correction seconds**.
2. **Failure risk vs realized correction utility**.
