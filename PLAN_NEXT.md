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
(`eval/jenga_w6_simple.py`). On seed 1, 24 of 84 forks at 1x stay blind even with perfect state. Forks
that are blind across most seeds, not one seed's misses, are the main dynamics research target
(Phase 2 establishes which they are).

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

**One canonical evaluation command.** For any checkpoint it emits:

* AUC;
* recall at 1%, 3%, 5% and 10% FPR;
* 1x and 2x recall;
* a blind-fork indicator for every individual fork;
* the fork spread distribution;
* quiet spread p50 / p95 / p99;
* fork/quiet contrast;
* rollout error;
* the raw monitor score for every state;
* the threshold used;
* the seed;
* a model / config hash.

**A manifest** with checksums of: the fork-state cache; the quiet-state cache; the perturbation set;
the evaluation configuration; and simulator / version information where practical. The goal is that
no later experiment can silently change the benchmark: the evaluator verifies the manifest before
scoring and refuses to run on a mismatch.

Metric priority when comparing models: recall at fixed FPR, then AUC, blind forks, quiet spread,
fork spread, contrast. Rollout error is a secondary diagnostic only. **Never choose a model primarily
from one-step or rollout MSE.**

**[repo note] What gets frozen.** States `results/jenga/holdout3_stage2_shared.json` (dev:
`holdout_stage2_shared.json`); perturbations = the 64 snippets in `holdout_stage0_cache.npz`
(byte-identical to batch 3's); oracle endings `holdout3_stage0_cache.npz`; the scoring logic now
split across `eval/jenga_w5_gate3.py` (pre-declared operating point), `eval/jenga_w6_report.py`
(separation, contrast) and `eval/jenga_w6_curves.py` (matched FPR, AUC, distributions); privileged
baseline `results/jenga/w6_gnn_n5_s*.pt`; V1 baseline `eval/jenga_v1_gate.py` with
`results/jenga/v1_probe_px196_c4096.pt`. The canonical command consolidates these three scorers so
every number comes from one pass over one set of per-state scores.

**Status (2026-09-21): DONE.** `eval/jenga_bench.py` (`freeze` / `eval` / `verify`). Frozen cache
`results/jenga/bench/jenga_bench.npz` (32 MB, gitignored), manifest
`results/jenga/bench/manifest.json` (committed; bench sha256 `8bf358f2b6ab56f5...`). Evaluation
needs no simulator (~2.7 min per checkpoint, was ~25), reproduces the earlier privileged results
bit-for-bit (all 642 test state-scales), and is bit-deterministic run to run for both the
privileged and the V1 path. Outputs go to `results/jenga/bench_eval/<checkpoint>.json`.

```
python eval/jenga_bench.py verify
python eval/jenga_bench.py eval --model results/jenga/w6_gnn_n5_s1.pt
python eval/jenga_bench.py eval --model results/jenga/w6_gnn_n5_s1.pt \
    --probe results/jenga/v1_probe_px196_c4096.pt            # V1: DINO + proprioception
```

## Phase 2 — Blind forks, defined across seeds

**Do not analyse the 24/84 misses of one seed as if they were inherent dynamics failures.** Seed
spread is large (37-80% recall across the 10 `gnn_n5` seeds).

Run the frozen Phase-1 evaluator on all 10 `gnn_n5` seeds. For every physical fork state compute

```
q_i = (# seeds that miss fork i) / 10
```

and report the complete distribution of `q_i`. Provisional classes, for analysis:

| class | rule | reading |
|---|---|---|
| robust blind | `q_i >= 0.8` | evidence of a systematic model limitation |
| seed-sensitive | `0.3 <= q_i < 0.8` | depends on optimisation, not on what the model can represent |
| usually detected | `q_i < 0.3` | |

Keep the raw frequencies in every report, so the cutoffs can change later without re-running the
evaluation. **Only the robust-blind group is initially interpreted as a systematic limitation.**

For the robust-blind subset only, re-simulate from the exact fork states and log at EVERY simulation
step: the full simulator state; contacts; the action; the real pairwise counterfactual spread; the
predicted spread; contact changes; pose changes. The current oracle cache is insufficient, since it
stores only hold steps 5/10/20/29/30 (`jenga_state_data.execute_recording` already records every
step).

Determine:

* the first real divergence time and the first predicted divergence time;
* whether the model never responds;
* whether it responds at first and then reconverges;
* whether divergence corresponds to contact creation / breaking, sliding, support loss, tipping,
  or another event;
* whether failures cluster by physical mechanism.

**Deliverable:** `blind_fork_analysis.md`, with the `q_i` distribution, representative examples and
a mechanism taxonomy.

**Decision gate:** no new architecture until there is evidence for what the current dynamics cannot
represent.

**Status (2026-09-21): steps 2-4 DONE** (`eval/jenga_blind_freq.py`,
`results/jenga/blind_freq_gnn_n5.json`). At 1x: 11 forks robust blind under both the pre-declared and
a matched-5%-FPR operating point (15 / 13 under either alone); at 2x, none. Seed 1's 28 misses split
12 robust / 13 seed-sensitive / 3 usually detected. The robust set shows near-zero predicted
response (0.3-0.7 mm vs 8-28 mm real), mostly on narrow branches (1-8 of 64 probes topple), and
clusters into 7 situations. Details in NOTES.md.

**[repo note]** The existing per-probe anatomy (`eval/jenga_w6_diagnose.py`) already found that
seed 1's false negatives sit at ~0.8 mm predicted spread against thresholds of 1.5-3.4 mm, i.e.
near-zero response rather than near-misses. The cross-seed `q_i` will show whether that holds for the
robust set.

**Status (2026-09-21): steps 5-6 DONE -- see [`blind_fork_analysis.md`](blind_fork_analysis.md).**
The systematic blind forks are upright neighbours pushed past their tipping angle (14/17; 0/39
pre-leaning forks are robust blind). The model responds to the push but under-delivers rotation (~11
vs ~16 deg), falls short of the tipping angle and reconverges; handed the true state after the push, it
completes the topple. The failing transition is 0.6% of training rollouts. This revises Phase 3 below.

## Phase 3 — Counterfactual training without reviving long-rollout training

Privileged state throughout, so perception cannot confound the result. **Do not add the old rollout
curriculum.** It has already been shown to damage fork sensitivity: a rollout stage halved recall
and drove fork separation to ~0.05 (NOTES.md, W6).

Keep `L_base = L_one-step` with 0.5x input-noise training, and compare:

| arm | objective |
|---|---|
| D0 | the current one-step + noise baseline |
| D1 | + the current branch-preservation loss. It already compares counterfactual separation THROUGHOUT the rollout (with a late-step ramp, `branch_terms` in `eval/jenga_w6_simple.py`); it is not endpoint-only |
| D2 | + a CoCo-inspired intervention-consistency loss [R1] |

**D2.** Use [R1] for the principle that a world model can predict plausible futures while being
insufficiently responsive to interventions. Adapt it to simulator counterfactual supervision rather
than copying its image-specific objective.

Start with a SHORT unroll, about 2-5 steps, from the same ground-truth chunk-start state. For
`u`, `u + d` and `u - d`, compare the real intervention effect with the predicted one over that
horizon:

```
Delta_real(t) = Phi(tau_{u+d}(t))  - Phi(tau_u(t))
Delta_pred(t) = Phi(tau^_{u+d}(t)) - Phi(tau^_u(t))
```

The objective must explicitly distinguish:

1. the intervention causes almost no real change -> the prediction should stay invariant;
2. the intervention causes real divergence -> the prediction should respond;
3. divergence onset;
4. persistence over the short horizon.

Do not merely match pairwise distance magnitude: D1 already tests that idea.

**Guardrail:** quiet p99 must not deteriorate materially.

**Primary success criterion:** fewer ROBUST blind forks (Phase 2), with quiet p99 and fixed-FPR
performance at least as good as D0.

Increase the unroll horizon only if the short-unroll experiment gives evidence that it helps.

**Revision from the Phase 2 taxonomy (2026-09-21).** The failure sits in the contact window (control
steps ~4-13), which a 2-5 step unroll from the chunk start does not reach. So:

* anchor the short unrolls at TRUE states inside the contact window, not only at the chunk start. That
  needs new same-state / different-action branches simulated from those states (the only same-state
  pairs in the current data are at the chunk start);
* `Phi` includes each object's orientation and angular velocity: before the tipping angle the fork
  signal is almost entirely rotation;
* match each branch's response to its real counterpart as well as the difference between branches: the
  error is a ~30% under-rotation shared by both branches, which difference-matching alone can leave in
  place;
* persistence past the tipping angle does not need supervising (the handoff shows it is modelled
  correctly);
* add an arm **D0 + contact-window data** (same one-step loss, new branches as ordinary data) to separate
  a coverage problem from an objective problem. Arms: D0, D0 + data, D1, D2 (+ data).

Primary target: the 45 upright-start forks (mean miss rate 0.52). D1 and
D2 each need >= 10 seeds, scored with the frozen Phase-1 command.

**[repo note]** The chunk start is the only point in the data where two probes share an exact state,
which is why D2 anchors there. D1 as currently implemented trains through a full 38-step rollout
stage; for a like-for-like comparison with D2 it should also be run on the same short horizon.

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

**The Jenga-specific 2x object readout is dropped from the immediate roadmap.** Do not build the
block-centric pose head proposed in the previous revision: the target is a universal monitor, and at
1x vision + proprioception already matches privileged-state dynamics. The 2x visual gap (83% vs 99%)
stays documented as an open diagnostic result, to revisit only if it becomes relevant to the generic
visual-world-model stage. The cheap ground-truth-substitution diagnosis
(`jenga_v1_gate.py --true-groups`) remains available if it does.

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

## Phase 11 — Counterfactual sensitivity vs model disagreement

Distinguish counterfactual sensitivity from epistemic uncertainty. **Exploit the existing models
first:** the ten independently trained `gnn_n5` seeds already give a cheap first
ensemble-disagreement baseline, so no new training is needed initially.

For every test state compute:

* counterfactual action sensitivity (the monitor score);
* across-model prediction disagreement for the nominal action.

Ask whether robust forks can occur with LOW ensemble disagreement. If yes, that is strong evidence
that counterfactual sensitivity is different from ordinary epistemic / model uncertainty: the model
knows the state well, yet the planned action sits near a genuine physical bifurcation.

**Do not call the ensemble disagreement calibrated epistemic uncertainty** unless calibration is
separately demonstrated. A trained ensemble (`U_epi = Var{tau^(m)}`) can follow later if this first
pass is inconclusive.

## Phase 12 — Shared-noise counterfactual evaluation

The rollout infrastructure already supports common / shared random numbers
(`state_dynamics.rollout(..., sample=True, noise=...)`), though every current model is
deterministic and nothing uses it yet. When stochastic world models are introduced, paired
counterfactual predictions must use

```
WM(z, u + d; xi)   and   WM(z, u - d; xi)   with the same xi
```

so that model stochasticity is never mistaken for action-induced branching. The simulator-fork
evaluation is strongly aligned with this methodology [R4]; that manuscript provides the framework
and states that its experiments are forthcoming.

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

1. Finish the Phase 1 benchmark freeze and checksum manifest.
2. Evaluate all 10 `gnn_n5` seeds with the frozen command.
3. Produce the per-fork miss frequency `q_i` across seeds.
4. Identify the robust-blind forks.
5. Re-simulate only those forks with dense state / contact logging.
6. Finish the physical-mechanism taxonomy (`blind_fork_analysis.md`).
7. Only then implement the short-unroll D2 counterfactual objective.

**No new architecture before steps 1-6 are complete.**

**Status (2026-09-21): steps 1-6 DONE; step 7 IN PROGRESS.** Contact-window branches are generated
(`eval/jenga_cw_data.py`: 24k contact + 7k no-contact points from training episodes only; adds 3.2x
the original count of upright-block rotation transitions). Arms: D0 (the existing `w6_gnn_n5` seeds,
reproduced bit-for-bit by the default trainer path), D0+CW, D1+CW, D2+CW, with D1 and D2 on the same
short 6-step unrolls so they differ only in the objective. First two paired D0+CW seeds improve
strongly (e.g. seed 2: 40 -> 7 blind forks); arm-level robust-blind counts are pending. New launches
are paused for a speed audit: only a batched branch loss is bit-identical to the current code;
torch.compile's cudagraphs backend gives wrong gradients and inductor diverges under Adam, so
neither may be mixed into an arm. Unfinished D1/D2 seeds restart on optimised code only if a
bit-identical configuration reaches >= 1.5x seeds/hour (`eval/jenga_w6_speed.py`).

Later, in order: D0 / D1 / D2 comparison (Gate 1); the existing-ensemble disagreement analysis
(Phase 11); a direct DINO latent world model (V2-A) against the state-GNN oracle pipeline; a V-JEPA
2-AC-style baseline (V2-B); then stop optimising Jenga -- pushing, insertion, the normalised
task-independent score and a shared calibration rule; then generic spatial tokens, optional F/T and a
shared multi-task world model.

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

## References

Verified by the user on 2026-09-21.

* **[R1] CoCo.** Yuhong Shi, Zhenhao Chu, Jie Wei, Jun Hao, Jianyi Liu, Jingwen Fu. *Overcoming
  Statistical Bias in Action-Controllable World Models.* arXiv:2608.04653, 2026. Use for: action
  responsiveness, zero-action drift, counterfactual consistency.
* **[R2] V-JEPA 2.** Mido Assran et al. *V-JEPA 2: Self-Supervised Video Models Enable
  Understanding, Prediction and Planning.* arXiv:2506.09985, 2025. Use for: generic visual latent
  world models, robot proprioception, action-conditioned latent planning.
* **[R3] ContactWorld.** Zhiyuan Zhang et al. *ContactWorld: What Matters in Vision-Tactile World
  Models for Contact-Rich Manipulation.* arXiv:2606.13877, 2026. Use for: cross-task contact-rich
  evaluation, spatial representations, later tactile / F/T experiments.
* **[R4] Twin Rollouts.** Yu Ma, Hongli Shi, Xinran Xu. *Twin Rollouts: Noise-Coupled
  Counterfactual Branching in Interactive Video World Models.* arXiv:2608.08982, 2026. Use for:
  same-prefix / shared-noise counterfactual evaluation. Experiments stated as forthcoming.
* **[R5] Risk-informed world models.** Kailang Ma, Heye Huang, Inhi Kim, Kitae Jang. *Rethinking
  World Models for Safety-Critical Embodied Systems.* arXiv:2609.03774, 2026. Use ONLY for research
  motivation and positioning: a perspective paper, not an implemented robotics baseline.

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
