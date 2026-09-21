# Active plan — task-agnostic counterfactual safety monitor (revised 2026-09-21)

**This is the plan being executed.** Every other `PLAN*.md` is historical: `PLAN_WORLDMODEL.md`
records W0-W6, `PLAN_JENGA.md` and `PLAN_DINOWM.md` the earlier Jenga and toy work, `PLAN.md` the
original FTLE plan. Results go in `NOTES.md` (append-only); the orientation document is `HANDOFF.md`.

Notes marked **[repo note]** are additions from the state of the code and results, where the plan
touches something already measured. Everything else is the plan as set on 2026-09-21.

---

## Project goal

Build a **task-agnostic runtime safety monitor** that detects when realistic execution
uncertainty can push a robot across a consequential physical regime boundary.

The monitor must not know: what task is being performed; what constitutes failure; whether the scene
contains Jenga blocks, insertion holes, stacked objects; task-specific object identities; or
task-specific unsafe-state labels.

The monitoring principle:

```
observation + proprioception + intended action
  -> perturb the intended action with realistic execution uncertainty
  -> predict multiple counterfactual futures
  -> measure whether those futures stay close or split into qualitatively different outcomes
  -> flag the action if the counterfactual response is anomalously large

  du_i  ~ p_exec(du)
  tau_i = WM(o_{t-k:t}, s_t, u_t + du_i)
  S_t   = Spread{ tau_1, ..., tau_N }
```

The monitor does not need to know what physical event caused the spread. A fork may be stable vs
topple, stick vs slip, stays on table vs falls, insertion vs jam, grasp held vs object escapes,
stable stack vs collapse. **Failure labels are used only for evaluation.**

## Central research hypothesis

Standard world-model quality is not sufficient for safety monitoring. A model can have low one-step
error, low rollout error and visually plausible futures while still failing to preserve the local
action-conditioned branching structure of the physical system. The property the monitor needs is

```
high sensitivity near consequential action boundaries  +  low sensitivity away from them
```

and it must be measured with counterfactual interventions, not ordinary prediction error.

**[repo note]** This project already has direct evidence for the hypothesis. The W5 line had lower
rollout error than W6 and was far worse at forks; within W6, adding a rollout stage cut rollout
error and halved fork recall, driving fork separation to ~0.05 (NOTES.md, W6). Across 11 models,
training fit correlated with fork recall at Spearman -0.20.

---

## Two tracks — keep them separate

### Track A — privileged / state-space mechanism study

Purpose: understand what makes counterfactual fork monitoring succeed or fail under ideal
perception.

```
simulator state -> GNN dynamics -> perturbed-action rollouts -> counterfactual spread
```

**This is NOT the proposed deployable monitor.** It is an oracle / diagnostic system, and
Jenga-specific state features are allowed here because their purpose is causal diagnosis.

Current best dynamics: GNN, one-step training, 0.5x input noise, no rollout curriculum, no MoE
(`eval/jenga_w6_simple.py`). About 24 of 84 forks at 1x stay blind even with perfect state, and those
blind forks are now the main dynamics research target.

### Track B — universal deployable monitor

```
generic visual observations + robot proprioception + intended action
  -> generic action-conditioned world model -> counterfactual future representations
  -> task-independent safety score
```

It must NOT require conversion into Jenga block poses, object-specific contact flags, simulator
state, or task-specific physical variables. The explicit-state vision pipeline (V1) remains an
important diagnostic baseline but is not the final universal architecture.

### Terminology

* **V1 — vision as state estimator.** `image -> frozen DINO -> estimated object state -> state GNN`.
  A Track-A diagnostic: it routes vision through the Jenga-specific 61-dim state. Any readout that
  predicts block centres, corners or gaps and feeds the state GNN is still V1.
* **V2 — direct latent world model.** `video -> DINO or V-JEPA tokens -> action-conditioned latent
  dynamics -> predicted future tokens -> score`. The first Track-B system. No state decoder.

Proprioception is available directly in both and is never reconstructed from vision.

---

## Where things stand (frozen numbers)

Batch 3: 214 held-out states, 84 topple forks at 1x, 82 at 2x. Recall on forks, false positives on
quiet states.

| system | 1x recall | FP | blind forks (1x) | 2x recall |
|---|---|---|---|---|
| real simulator endings, DINO-scored (ceiling) | 88% [78-95] | 3% | -- | 99% |
| **privileged GNN, one-step + 0.5 noise**, 10 seeds | 64% mean (37-80) | 4% | -- | 96% |
| same, seed 1 (the frozen dynamics used below) | 67% | 4% | **24/84 (29%)** | 99% |
| **V1: DINO + proprioception -> state -> frozen GNN** | 67% [57-75] | 4% | 25/84 (30%) | 83% |
| V1 before the probe fix (4.44 mm state error) | 36% | 2% | 38/84 (45%) | 54% |

Privileged GNN at MATCHED false-positive rate (10 seeds pooled): AUC 0.942; recall 37 / 62 / 67 / 79%
at 1 / 3 / 5 / 10% FPR. Median fork/quiet contrast 124.

Established findings this plan builds on (details in NOTES.md):

* The monitor fails from a **quiet tail**, not from weak forks: no fork scores below a typical
  quiet state; overlap is 2-4%; a few quiet states whose rollout runs away set the threshold.
  Input noise works mainly by suppressing that tail. The noise curve is an inverted U, best at 0.5.
* DINO preserves fork ORDERING on real endings (AUC 0.980 vs 0.993 privileged) while compressing
  contrast about 7x.
* V1 recall tracks millimetre-scale joint pose precision: 4.44 mm -> 36%, 2.65 mm -> 67%. Velocity is
  irrelevant; position and orientation matter only together; proprioception alone is worth +16
  points and must not come from vision.
* A global-PCA readout saturates at ~3.6 mm relative pose error at both 14x14 and 28x28 patch grids.

---

## Phase 1 — Freeze the Jenga mechanism benchmark

Stop changing the evaluation protocol. Freeze: the held-out configurations; the fork and quiet
states; the realistic execution-perturbation distribution; the 1x and 2x scales; the calibration
procedure; the FPR operating points; the simulator oracle endings; the privileged-state GNN baseline;
and the corrected DINO + proprioception state-estimation baseline.

Primary metrics, in this order:

1. recall at fixed FPR: 1%, 3%, 5%, 10%
2. AUC
3. blind-fork count
4. quiet spread p50 / p95 / p99
5. fork spread p50 and its distribution
6. fork/quiet contrast
7. ordinary rollout error, as a secondary diagnostic only

**Never choose a model primarily from one-step or rollout MSE.**

**[repo note] What "frozen" points to.** States `results/jenga/holdout3_stage2_shared.json`
(dev: `holdout_stage2_shared.json`); snippets `holdout_stage0_cache.npz` (identical to batch 3's);
oracle endings `holdout3_stage0_cache.npz`; scoring `eval/jenga_w5_gate3.py` (pre-declared
operating point) + `eval/jenga_w6_report.py` (separation, contrast) + `eval/jenga_w6_curves.py`
(matched FPR, AUC, distributions, attribution); privileged baseline `results/jenga/w6_gnn_n5_s*.pt`;
V1 baseline `eval/jenga_v1_gate.py` with `results/jenga/v1_probe_px196_c4096.pt`. What is still
missing is a single command that emits every Phase-1 metric for any model, plus a frozen manifest
(file hashes) so a later change cannot slip in unnoticed. Build that first.

---

## Phase 2 — Understand the privileged blind forks

Before changing architecture, analyse the ~24/84 1x forks missed even with perfect simulator state.
For each blind fork record: state; nominal action; perturbation direction; real counterfactual
trajectories; predicted counterfactual trajectories; when the real trajectories first diverge; real
and predicted divergence magnitude over time; contact creation / breaking; stick / slip changes;
support changes; tipping onset; time-to-fork; branch persistence.

Cluster missed forks by physical mechanism. Questions:

1. Does the model ignore the action perturbation completely?
2. Does it respond correctly at first and then reconverge?
3. Does it miss specific contact transitions?
4. Are blind forks systematically later-horizon events?
5. Are they associated with small initial geometric margins?
6. Are certain perturbation directions disproportionately missed?

**Deliverable:** `blind_fork_analysis.md` with representative examples and a taxonomy.

**Decision gate:** no new architecture until there is evidence for what the current dynamics cannot
represent.

**[repo note] Two things the analysis needs.**
* *Separate systematic blind forks from seed luck.* 24/84 is seed 1 alone, and seed spread is large
  (37-80% recall). Score all 10 `w6_gnn_n5` seeds and split forks into blind-in-most-seeds (a
  property of the dynamics) and blind-in-few (optimisation noise). Only the first group is evidence
  about what the model cannot represent.
* *Full counterfactual trajectories.* The Stage-0 cache stores poses only at hold steps 5/10/20/29/30,
  so "when do real trajectories first diverge" needs a re-simulation of the blind states with
  every-step state and contact recording (`jenga_state_data.execute_recording` already does this).
  The existing per-probe anatomy (`eval/jenga_w6_diagnose.py`) found false negatives sit at ~0.8 mm
  predicted spread against thresholds of 1.5-3.4 mm -- genuinely near-zero response, not near-misses.

---

## Phase 3 — Counterfactual dynamics training (privileged state)

Privileged state throughout, so perception cannot confound the result. Keep the winning recipe
(one-step + 0.5 input noise). **Do NOT reintroduce the long rollout curriculum.**

| arm | objective |
|---|---|
| D0 | one-step + 0.5 noise (baseline) |
| D1 | + the previous branch-distance loss, as an ablation. Matches branch MAGNITUDE: `D_pred(a, a+d) ~ D_real(a, a+d)` |
| D2 | + CoCo-inspired intervention consistency [R1] |

**D2.** For the same physical state evaluate `u`, `u + d`, `u - d`, and where meaningful `u_noop`.

```
real intervention response:       Delta_real(t) = Phi(tau_{u+d}(t))  - Phi(tau_u(t))
predicted intervention response:  Delta_pred(t) = Phi(tau^_{u+d}(t)) - Phi(tau^_u(t))
```

Train so the predicted response has the correct magnitude, onset time, persistence, and
direction/structure where appropriate. Explicitly penalise drift when the real intervention makes
essentially no difference. The difference from D1: D2 asks *does the model respond appropriately to
an action intervention?*, not only *are the final pairwise distances right?*

**Primary success criterion:** fewer privileged blind forks **without raising quiet p99.** Do not
optimise for toppling specifically; the loss must work for arbitrary physical interventions.

**[repo note] A design tension to resolve up front.** Onset and persistence are properties of a
trajectory, so D1 and D2 need rollouts -- but a full 38-step rollout stage is exactly what halved
recall. The only same-state action pairs in the data are at the chunk start; after that the two
branches' states differ. So the first D2 variant should use a SHORT unroll (a few steps,
pushforward-style: gradient only through the last steps) from the shared chunk-start state, with
quiet p99 as the guardrail, and lengthen only if quiet p99 holds. Also note the existing D1 loss is
already applied over the whole rollout with a late-step ramp (`branch_terms` in
`eval/jenga_w6_simple.py`), not at the endpoint only. D1 and D2 each need >= 10 seeds.

---

## Phase 4 — The first universal visual world model

After Phase 3, build the first model that does NOT route vision through the 61-dim Jenga state.
This is the move from mechanism study to universal-monitor research.

```
z_t           = E_phi(o_{t-k:t})
z^_{t+1:t+H}  = F_theta(z_t, s_t, u_{t:t+H})        s_t: robot proprioception
```

Run the same action perturbations `u + du_i` and measure divergence directly in the learned
representation. No Jenga state decoder, no topple classifier, no object-specific handcrafted
features.

## Phase 5 — Visual representation baselines

* **V2-A — DINO latent world model.** DINO has already shown it retains the needed geometry given a
  good readout. Train action-conditioned latent dynamics directly on DINO tokens. Proprioception
  stays direct.
* **V2-B — V-JEPA 2 / V-JEPA 2-AC style [R2].** Visual representation + proprioception + actions ->
  latent future prediction; conceptually the closest published baseline to the target
  architecture. Not a full reproduction. The test: *does a modern latent action-conditioned visual
  WM preserve local counterfactual branching?* Same fork benchmark.

**[repo note]** The earlier DINO-WM predictor collapsed branches (Stage 3: separation d' 0.60
predicted vs 4.85 real), but it was trained with objectives this project has since shown destroy
fork structure. V2-A should reuse the Track-A lessons: one-step + input noise, no rollout
curriculum, graded on the frozen benchmark. The DINO-WM wrapper resizes every input to 196 px
before the encoder; call DINOv2 directly (`eval/jenga_v1_encode.py`) if resolution matters.

## Phase 6 — Generic spatial / object representation

Only if global DINO / V-JEPA latent prediction underperforms. **Avoid a Jenga-specific object
detector.** Use generic spatial tokens: dense patch tokens, unsupervised object slots, region tokens,
point-based geometric tokens, or generic segmentation proposals, with dynamics acting directly on
them: `{z_t^1..z_t^K} -> {z^_{t+1}^1..z^_{t+1}^K}`. No token is hardcoded as "Jenga block 1".

Compare a **generic GNN** (tokens as nodes, generic relational edges) against an **object
Transformer** (same tokens, same data, comparable parameters; self-attention in place of message
passing). Judge only on the counterfactual safety benchmark, not latent prediction error.

**[repo note] This reverses the previous next step.** The last revision proposed a block-centric
spatial readout (centres, corners, gaps) to close the V1 2x gap (83% vs 99%). That readout is
Jenga-specific and is now deprioritised. The 2x visual gap is deferred under "what not to do";
the cheap ground-truth-substitution diagnosis (`jenga_v1_gate.py --true-groups`) can still be run
if it becomes relevant, but no Jenga detector gets built.

## Phase 7 — A task-independent safety score

Raw latent distances will not be comparable across tasks. Standardise against the model's own
nominal quiet behaviour, for example

```
S_t = ( D_cf(t) - median(D_quiet) ) / ( MAD(D_quiet) + eps )
or
S_t = -log P_{D ~ p_quiet}( D >= D_cf )
```

The score must mean the same thing across tasks and scene types. Calibration data may contain normal
interaction trajectories; it must not require failure examples. Test (1) per-environment quiet
calibration and (2) one shared threshold across tasks. The second is the stronger result.

**[repo note]** The quiet distribution is heavy-tailed here (p99/median up to ~190 for the best
privileged model), so a median/MAD score will be dominated by that tail. Report the score's
behaviour on the quiet tail explicitly, not only its median.

## Phase 8 — Cross-task validation

Jenga alone cannot establish universality. Add at least two physically different tasks:

1. **Jenga / stacking** — stable vs topple / collapse.
2. **Pushing** — stick vs slip, or stays on the table vs falls over the edge.
3. **Insertion** — successful insertion vs jam / contact-mode transition.

(Alternative: grasp manipulation where small perturbations let the object escape.)

Same monitor algorithm for every task: estimate that task's realistic execution-error distribution;
fork simulator state; sample counterfactual actions; obtain real futures; identify physical fork
states for evaluation only; run world-model counterfactual rollouts; compute the generic score. No
per-environment failure classifier, no task-specific monitor features.

**[repo note]** The Jenga pipeline's execution-error model comes from lag-model tracking residuals
(`action_uncertainty.tracking_arrays`); each new task needs its own measured equivalent, and the
Stage-0 answer-key definition (mixed = >= 2 probes on each side) should carry over unchanged.

## Phase 9 — Cross-task world-model training

* **Setting A — task-specific dynamics, universal monitor.** A separate WM per environment, identical
  monitoring algorithm and scoring. Establishes whether the safety principle generalises.
* **Setting B — one shared multi-task WM** across Jenga, pushing, insertion, stacking, from generic
  observations + proprioception + actions, monitored without task identity. The stronger result.

Setting B is not required to demonstrate initial universality, but it substantially strengthens the
claim.

## Phase 10 — Contact sensing extension

After the RGB + proprioception monitor, add wrist force/torque as a controlled optional modality
[R3]: `[Fx, Fy, Fz, tx, ty, tz]_{t-k:t}`, simulated with realistic noise, bias, drift and
bandwidth/filtering. Compare vision + proprioception against vision + proprioception + F/T.
Main question: **does F/T reduce blind forks specifically around contact transitions?** Not mandatory
unless it gives clear cross-task benefit.

## Phase 11 — Epistemic uncertainty baseline

Distinguish counterfactual sensitivity from epistemic uncertainty. Build a simple ensemble WM and
compute `U_epi = Var{ tau^(m) }_{m=1..M}`. Look especially for states with **low U_epi but high
S_cf**: the model knows the state well, yet the planned action sits near a genuine physical
bifurcation. That is a fundamentally different safety signal from OOD detection.

**[repo note]** The 10-seed privileged grid is already a 10-member ensemble, so a first version of
this comparison needs no new training.

## Phase 12 — Twin-Rollout-style evaluation

The simulator-fork evaluation is strongly aligned with Twin Rollouts [R4]: paired trajectories share
the prefix and the exogenous randomness, and differ only in the intervened action. Wherever a WM is
stochastic, paired counterfactual rollouts must use common random numbers, `WM(z, u+d; xi)` vs
`WM(z, u-d; xi)` with the same `xi`, or stochasticity masquerades as action sensitivity.

**[repo note]** `state_dynamics.rollout` already supports common random numbers (`noise=`) but
nothing uses it. Every current model is deterministic, so this becomes binding in Phase 4-5 if a
stochastic WM is used.

## Phase 13 — Paper positioning

Motivation from [R5]: likelihood is not risk, prediction is not intervention, likely futures are not
necessarily the safety-relevant ones. The empirical contribution must be concrete: *we evaluate
whether learned world models preserve action-conditioned physical branches closely enough to
support runtime safety monitoring.*

---

## Proposed final architecture

```
camera / video        -> generic spatial / video representation
robot joint encoders  -> proprioception
controller            -> intended action
execution model       -> realistic perturbation distribution

{ o_{t-k:t}, s_t, u_t + du_i }  -> shared world model -> counterfactual future representations
S_t = NormalizedCounterfactualSpread(tau^_1 .. tau^_N)
S_t > gamma  =>  safety intervention
```

The monitor is never told whether the relevant event is a topple, slip, jam, collision, fall or grasp
loss.

## What NOT to do next

Do not spend substantial time on: improving Jenga from 67% to 70% before testing another task;
another Jenga-specific object detector; more Jenga-specific state variables; reviving MoE without
new evidence; scaling the GNN to reduce rollout MSE; training a topple classifier; tuning thresholds
with failure labels; assuming visually impressive generative WMs preserve safety-relevant action
sensitivity.

---

## Immediate execution order

**Now**
1. Freeze the corrected Jenga baseline (Phase 1).
2. Analyse the ~24 privileged blind forks (Phase 2).
3. Implement CoCo-inspired intervention-consistency training on privileged state (Phase 3).
4. Determine whether blind forks can be recovered without raising quiet p99.

**Immediately after**
5. Build a direct DINO latent action-conditioned WM baseline (V2-A).
6. Compare the state-GNN oracle pipeline against the direct visual-latent WM on the same benchmark.
7. Add a V-JEPA 2-AC-style visual-latent baseline (V2-B).

**Then stop optimising Jenga**
8. Add pushing near a physical regime boundary.
9. Add insertion / jamming or another distinct contact-rich task.
10. Apply the same counterfactual monitor unchanged.
11. Introduce the normalised task-independent fork score.
12. Measure whether one calibration rule works across all tasks.

**Then strengthen the system**
13. Test generic object / spatial tokens if latent visual WMs need better physical structure.
14. Add optional F/T history.
15. Compare counterfactual sensitivity against epistemic uncertainty.
16. Train a shared multi-task WM if per-task models validate the monitor principle.

## Decision gates

| gate | question | if yes | if no |
|---|---|---|---|
| 1 | Does CoCo-inspired training improve privileged blind-fork recall without raising quiet false positives? | use it for later dynamics models | keep one-step + noise |
| 2 | Does direct visual-latent prediction approach the explicit-state V1 pipeline? | candidate universal architecture | test structured spatial / object tokens |
| 3 | Does the identical monitor detect forks on at least two non-Jenga mechanisms? | meaningful evidence the principle is universal | the monitor is not yet universal |
| 4 | Can one normalised score / calibration operate across those tasks? | strong universal-monitor result | separate "universal algorithm" from "environment-specific calibration" |
| 5 | Does one shared multi-task WM preserve useful counterfactual branching? | strongest result | the monitor can still be universal with per-environment WMs; phrase the claim accordingly |

## Intended paper claim

Do not claim "we build a Jenga safety monitor", "we detect discontinuities", or primarily "we
introduce a better world-model architecture". Target:

> We investigate whether action-conditioned world models can act as task-agnostic runtime safety
> monitors by identifying proximity to consequential physical regime boundaries under realistic
> execution uncertainty. We introduce a counterfactual evaluation protocol, characterize when
> conventional world-model training preserves or destroys safety-relevant action sensitivity, and
> evaluate the same monitoring principle across distinct contact-rich manipulation tasks without
> using task-specific failure labels.

Central experimental thesis:

```
good prediction  =/=>  faithful counterfactual action sensitivity
faithful counterfactual action sensitivity  can provide a generic runtime safety signal
```

---

## References to read

Supplied with this plan. **[repo note]** The 2026 entries postdate what I can verify; their titles,
identifiers and claimed contents must be checked against the papers before they are cited or before
a method is built on them. V-JEPA 2 [R2] is a known 2025 paper.

* **[R1] CoCo.** Yuhong Shi et al., *Overcoming Statistical Bias in Action-Controllable World
  Models*, 2026, arXiv:2608.04653. Counterfactual training; action responsiveness; preventing
  no-action drift; same-state / multiple-action evaluation.
* **[R2] V-JEPA 2 / V-JEPA 2-AC.** Mido Assran et al., *V-JEPA 2: Self-Supervised Video Models
  Enable Understanding, Prediction and Planning*, 2025, arXiv:2506.09985. Generic video
  representation; action-conditioned latent prediction; direct use of proprioception; latent
  planning.
* **[R3] ContactWorld.** Zhiyuan Zhang et al., *ContactWorld: What Representations Matter in
  Vision-Tactile World Models for Contact-Rich Manipulation*, 2026, arXiv:2606.13877. 12 contact-rich
  tasks; spatially structured, temporally continuous, multimodal representations; candidate tasks.
* **[R4] Twin Rollouts.** Yu Ma, Hongli Shi, Xinran Xu, *Twin Rollouts: Noise-Coupled
  Counterfactual Branching in Interactive Video World Models*, 2026, arXiv:2608.08982.
  Counterfactual branching methodology; shared-noise paired rollouts; simulator-fork ground truth.
  Experiments are stated as forthcoming: methodological related work, not a performance baseline.
* **[R5] Risk-informed world models.** Kailang Ma et al., *Rethinking World Models for
  Safety-Critical Embodied Systems*, 2026, arXiv:2609.03774. Safety motivation; prediction vs
  intervention; consequential futures. A perspective article, not a robotics baseline.

Carried from the previous revision as an optional dense-feature baseline for Phase 6: Mur-Labadia
et al., *V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning*, 2026,
arXiv:2603.14482 (also unverified).

## Standing rules

* Grade on the frozen batch-3 benchmark; report recall at MATCHED FPR alongside the pre-declared
  Gate 3 point.
* >= 10 seeds per arm. Three-seed estimates were off by ~10 points three separate times.
* Never select an arm on validation: it comes from the training episode pool and understated the
  rollout-curriculum damage by an order of magnitude.
* Judge readouts and representations by fork recall, blind forks and quiet-tail statistics, never
  by reconstruction RMSE alone. Velocity was the worst-reconstructed group and the least relevant.
* End-effector pose comes from proprioception, never from vision.
* Before adding any feature, weighting or rule, state whether it is universal (Track B), oracle-only
  (Track A) or grading-only.
