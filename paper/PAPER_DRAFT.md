# Learning When Corrections Are Worth It
## Correction-Utility-Aware Post-Training for Vision-Language-Action Policies

> Working draft v0.2 — targeted at ICRA/IROS-level method and real-robot evaluation. Results are intentionally left as TBD until experiments are run.

## Abstract

Real-world post-training of vision-language-action (VLA) policies increasingly relies on autonomous robot experience and human corrections. Methods such as RECAP demonstrate that intervention trajectories provide valuable supervision for improving deployed policies. However, human correction remains a scarce resource: existing systems primarily determine when to intervene based on execution failure, uncertainty, or human judgment, without estimating whether obtaining a correction at a particular state will substantially improve the future policy.

We introduce **CU-RECAP**, a correction-utility-aware framework for budgeted real-world VLA post-training. The key idea is to distinguish **execution risk** from **learning utility**. A state can be likely to fail yet provide little new training information, while a less obviously erroneous state can expose an underrepresented policy weakness and yield substantially greater improvement after correction.

We define the correction utility of a state as the expected improvement in policy performance resulting from incorporating a human corrective transition at that state. Since this quantity cannot be observed before requesting feedback, CU-RECAP constructs post-hoc utility targets from previous correction rounds using parameter-efficient local policy updates and a held-out hard-state probe buffer. A lightweight temporal multimodal critic then predicts correction utility online. Because utility labels are only observed for queried states, we use stochastic query policies with logged propensities and inverse-propensity-weighted training to reduce selection bias.

Given a fixed human-intervention budget, the critic allocates corrections to states with high expected utility per unit human cost rather than merely high failure probability. Safety intervention remains an independent hard constraint. Collected corrections are subsequently incorporated into standard RECAP advantage-conditioned post-training.

We evaluate CU-RECAP in simulation and on a Tianyi 2.0 humanoid manipulation platform, with the dexterous hand abstracted as a binary open/close gripper. Experiments compare standard RECAP, random querying, uncertainty-based querying, failure-risk querying, novelty-based querying, and correction-utility-aware querying under matched robot interaction and human-control-time budgets. The primary metric is downstream task success as a function of cumulative human correction time.

**Hypothesis:** optimizing where human corrections teach the policy the most, rather than where the robot is simply most likely to fail, improves the human efficiency of real-world VLA post-training.

## 1. Introduction

Vision-language-action models provide a promising route toward general-purpose robot manipulation, but deployment inevitably exposes failure states that are rare or absent in demonstration datasets. Collecting more demonstrations is expensive and often poorly targeted to the actual weaknesses of a deployed policy.

RECAP-style post-training addresses part of this problem by combining autonomous experience, human corrections, value estimation, and advantage-conditioned policy updates. Yet it leaves an important resource-allocation question largely open: **given a limited human correction budget, where should the system spend it?**

Most practical intervention systems rely on predicted failure, policy uncertainty, or direct human judgment. These signals estimate whether the robot may need help, but they do not estimate whether the resulting corrective trajectory will materially improve the future policy.

We argue that these are different quantities. Let

\[
R(x)=P(\text{failure}\mid x)
\]

be execution risk at state \(x\), while

\[
U(x)=\mathbb{E}[J(\theta\oplus c)-J(\theta)\mid x]
\]

is the expected learning utility of obtaining correction \(c\) at that state. In general,

\[
R(x)\neq U(x).
\]

A frequent failure mode can have high risk but low marginal learning value because the dataset already contains many similar corrections. Conversely, a moderately risky state may reveal a novel or broadly consequential weakness and therefore have high correction utility.

CU-RECAP turns this observation into a budgeted selective-feedback problem. It estimates post-hoc correction utility from earlier feedback, learns a pre-query critic, and uses stochastic cost-aware acquisition to decide where to request future corrections. The resulting data are then consumed by an otherwise unchanged RECAP post-training pipeline.

### Contributions

1. **Correction utility formulation.** We separate execution risk from the expected downstream learning value of a corrective trajectory.
2. **Post-hoc utility supervision.** We estimate per-correction learning value using temporary parameter-efficient updates and a held-out hard-state probe buffer, with a first-order gradient-alignment proxy for analysis and efficient labeling.
3. **Selective-feedback-aware utility learning.** We explicitly address query-induced selection bias through stochastic acquisition, propensity logging, and inverse-propensity-weighted utility regression.
4. **Budgeted acquisition.** We formulate online correction requests as a utility-under-cost problem, while keeping safety intervention separate from learning-driven intervention.
5. **Budget-controlled real-robot evaluation.** We evaluate performance improvement per minute of human corrective control, not only final success rate.

## 2. Background

### 2.1 RECAP-style post-training

Given a value function \(V(o_t)\), RECAP computes an N-step advantage

\[
A_t=\sum_{k=0}^{N-1}r_{t+k}+V(o_{t+N})-V(o_t),
\]

then maps it to an improvement condition such as positive/negative. The VLA is trained as

\[
\pi_\theta(a_t\mid o_t,l,c_t),
\]

and is conditioned on positive improvement at inference time. Human interventions provide especially valuable data because they occur at states where the current policy demonstrates concrete failure modes.

CU-RECAP does not change this downstream training rule. It changes **which human corrections are collected under a limited feedback budget**.

### 2.2 Active feedback and intervention learning

Prior robot learning systems have queried humans using uncertainty, risk, preference, information gain, or direct takeover triggers. Recent VLA critics also estimate task progress, done signals, or failure likelihood. CU-RECAP focuses on a narrower objective: predicting the **marginal policy-improvement value** of a corrective trajectory before deciding whether to request it.

## 3. Problem Formulation

Let \(x_t=(o_t,s_t,l,h_t)\) denote visual observation, robot state, language task, and short history. At a query opportunity, the system chooses \(q_t\in\{0,1\}\), where \(q_t=1\) requests human corrective control. If queried, the human supplies a corrective trajectory \(c_t\) with cost \(C(c_t)\), measured in human-control seconds.

Define downstream policy quality \(J(\theta)\). The ideal correction utility is

\[
U(x_t)=\mathbb{E}_{c_t}[J(\theta\oplus c_t)-J(\theta)\mid x_t].
\]

Given a human budget \(B\), the acquisition problem is

\[
\max_{q_{1:T}}\mathbb{E}\left[\sum_t q_t U(x_t)\right]
\]

subject to

\[
\mathbb{E}\left[\sum_t q_t C(c_t)\right]\le B.
\]

Safety intervention is modeled separately and can override this learning objective.

## 4. Post-Hoc Correction Utility

### 4.1 Hard-state probe buffer

We maintain a held-out hard-state buffer \(H\) containing difficult states such as previous failures, recovery states, near-intervention states, and controlled perturbations. The buffer is not used as the correction being scored.

For correction \(c\), let

\[
\theta_c=\theta-\eta\nabla_\theta L_c(\theta)
\]

where the update is restricted to a parameter-efficient subset such as LoRA adapters. Define the one-step probe-gain target

\[
U_{1\mathrm{step}}(c)=L_H(\theta)-L_H(\theta_c).
\]

Positive values mean the correction improves performance on hard states after one local update.

### 4.2 First-order proxy

Let \(g_c=\nabla L_c\) and \(g_H=\nabla L_H\). By first-order expansion,

\[
L_H(\theta-\eta g_c)\approx L_H(\theta)-\eta g_H^\top g_c.
\]

Therefore the predicted loss reduction is

\[
U_{FO}(c)=\eta g_H^\top g_c.
\]

The sign matters: positive gradient alignment predicts positive held-out improvement.

We use \(U_{1\mathrm{step}}\) as the primary post-hoc training target and \(U_{FO}\) as a cheaper proxy and diagnostic. On a stratified subset we additionally compute a more expensive micro-finetuning oracle \(U_{oracle}\) to validate that these local targets correlate with actual downstream improvement.

## 5. Utility Critic

A lightweight temporal multimodal critic receives only pre-query information:

\[
C_\psi(x_t)\rightarrow\{\mu_U,\sigma_U,R,\hat C\}.
\]

The outputs are expected correction utility, uncertainty, failure risk, and expected human correction cost. A short temporal history is used because failure and recoverability are often ambiguous from a single frame.

## 6. Selective Feedback and Propensity Correction

Correction utility is observed only when a state is queried. Once the current critic influences query decisions, the utility dataset becomes selection-biased.

We therefore make learning queries stochastic and log their propensity \(p_t=P(q_t=1\mid x_t)\). For queried samples, utility regression uses clipped inverse-propensity weighting:

\[
\mathcal L_U=
\frac{\sum_t \frac{q_t}{p_t}\ell(\hat U_t,U_t)}
{\sum_t \frac{q_t}{p_t}}.
\]

This prevents the utility model from naively training only on states selected by its own previous deterministic policy.

## 7. Budgeted Acquisition

For an ensemble utility prediction, define

\[
UCB(x)=\mu_U(x)+\beta\sigma_U(x).
\]

We use a Lagrangian cost score

\[
S(x)=UCB(x)-\lambda\hat C(x),
\]

and stochastic query probability

\[
p(q=1\mid x)=\sigma(S(x)/\tau).
\]

The dual variable \(\lambda\) is updated online based on budget consumption.

### Safety separation

If estimated execution risk crosses a hard safety threshold, safety logic may request intervention or stop the robot independently of correction utility. Thus a state may be unsafe but uninformative, or safe enough to continue but highly informative to correct.

## 8. Integration with RECAP

CU-RECAP modifies the data-acquisition stage only:

1. initialize from a task-adapted VLA;
2. run autonomous rollouts;
3. estimate utility/risk/cost at query opportunities;
4. request learning corrections according to the budgeted stochastic policy;
5. compute post-hoc utility labels for queried corrections;
6. append corrections to the standard RECAP dataset;
7. train the RECAP value model, compute N-step advantages, and fine-tune the advantage-conditioned policy;
8. repeat.

The RECAP training schedule is kept identical across acquisition baselines so improvements can be attributed to correction selection.

## 9. Experimental Design

### 9.1 Study 1: Does the utility estimator measure real learning value?

For each collected correction compute \(U_{FO}\), \(U_{1step}\), and for a stratified subset \(U_{oracle}\). Report Spearman correlation, top-k precision, and calibration/ranking plots.

**Go/no-go criterion:** if \(U_{1step}\) does not rank \(U_{oracle}\) meaningfully better than uncertainty/risk baselines, revise the utility definition before expensive real-robot experiments.

### 9.2 Study 2: Risk is not utility

Plot realized correction utility against failure risk. Analyze high-risk/high-utility, high-risk/low-utility, moderate-or-low-risk/high-utility, and low-risk/low-utility cases.

### 9.3 Study 3: Budgeted acquisition

Compare under identical human-control-second budgets: random querying, policy uncertainty, failure-risk querying, novelty/embedding distance, utility critic without propensity correction, and CU-RECAP.

Primary plot: **Task Success vs. Cumulative Human Correction Seconds**. Primary summary metric: area under this curve.

### 9.4 Study 4: End-to-end RECAP

After each collection round, all methods use identical value training, N=50 relabeling, policy optimization, LoRA rank, epochs, and evaluation protocol. The only experimental variable is correction acquisition.

## 10. Platforms and Tasks

### Simulation

Use a standard manipulation benchmark such as LIBERO or RoboCasa to obtain multiple seeds, larger sample sizes, and controlled perturbations.

### Real robot

Platform: **Beijing Humanoid Tianyi 2.0**. The Inspire dexterous hand is intentionally abstracted as a binary gripper \(g\in\{OPEN,CLOSE\}\).

Proposed task families:

1. randomized pick-and-place;
2. cluttered target retrieval;
3. precision placement;
4. controlled recovery from object displacement / near-miss grasp / post-grasp perturbation.

## 11. Metrics and Statistics

Report success rate with 95% confidence intervals, area under the success-vs-human-time curve, improvement per human correction minute, correction duration/count, failure-mode coverage/diversity, utility-estimator rank correlations, utility-critic ranking/calibration metrics, and risk-utility correlation.

Use paired initial conditions when possible, multiple seeds in simulation, and pre-specified real-robot evaluation configurations.

## 12. Key Ablations

- one-step utility vs first-order proxy;
- with vs without propensity correction;
- deterministic vs stochastic querying;
- utility only vs utility + uncertainty UCB;
- utility per query vs utility per predicted human second;
- fixed vs refreshed hard-state probe buffer;
- risk only vs utility only vs safety-risk + utility acquisition;
- short temporal history vs single-frame critic.

## 13. Limitations

Correction utility is policy- and dataset-dependent. The one-step target is only a local approximation to long-horizon retraining benefit. Probe-buffer construction can bias what the method considers useful. Propensity correction requires non-degenerate, correctly logged query probabilities. Human correction quality varies across operators. The current study does not address full dexterous manipulation.

## 14. Expected Failure Modes of the Method

CU-RECAP should not be considered successful merely because its critic predicts its own utility labels. Key failure modes include weak correlation between the utility proxy and real downstream improvement, risk and utility collapsing to the same signal, query collapse onto one failure cluster, unstable IPW variance, expensive utility labeling, and gains disappearing after controlled RECAP post-training.

## References / Closest Work to Track

- Physical Intelligence, π*0.6 / RECAP: VLA learning from experience and corrections.
- OpenPI, open-source π-series VLA implementation.
- Recent VLA critic / human-in-the-loop systems such as VLAC.
- Temporal/world-aware critics such as WCM.
- Value-guided action selection systems such as VGAS.
- Intervention/preference learning approaches such as PACT.
- Active robot feedback / information-gain query selection literature.
- Jev/Laya-style typed System-1 decision models as architectural inspiration, not core novelty.
