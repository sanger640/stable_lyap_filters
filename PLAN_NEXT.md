# Active plan — task-agnostic counterfactual safety monitor (revised 2026-09-28)

**This is the plan being executed.** Every other `PLAN*.md` is historical: `PLAN_WORLDMODEL.md`
records W0-W6, `PLAN_JENGA.md` and `PLAN_DINOWM.md` the earlier Jenga and toy work, `PLAN.md` the
original FTLE plan. Results go in `NOTES.md` (append-only); the orientation document is `HANDOFF.md`.

Notes marked **[repo note]** are additions from the state of the code and results, where the plan
touches something already measured. Everything else is the plan as set on 2026-09-21.

---

## Completed gate: D4 replication and DEV (2026-09-27)

**Completed.** The five-seed TRAIN replication passed, so the matched DEV comparison was frozen
and run. D4 then failed DEV: median topple alarms did not exceed matched continuation (8/23 versus
8/23) and median physical-v0 agreement fell from 77.1% to 75.7%; quiet alarms were 3/89 versus
2/89. TEST was not opened. The steps below are retained as the executed protocol record.

D4 now exists end to end: `eval/jenga_d4_neighborhood_data.py` creates complete 64-probe
TRAIN-only neighborhoods, `src/neighborhood_topology_loss.py` supplies full pairwise topology and
nearest-action commitment losses, and `eval/jenga_d4_train.py` runs the frozen-D2/matched/topology/
full-D4 comparison. The seed-1 pilot preserves frozen candidate recall while matched continuation
forgets it, and improves continuous structural errors, but validation contains only seven states.

Next, in order:

1. Expand the same label-free, episode-balanced TRAIN neighborhood cache so validation contains at
   least 30 states and meaningful candidate/quiet counts. Do not change the frozen losses.
2. Repeat matched continuation and full D4 for fixed seeds 1-5. Require the topology/commitment
   advantage and candidate-recall protection to hold across seeds without unacceptable response
   degradation. Report distributions, not the best seed.
3. If that gate passes, freeze one matched DEV protocol comparing original D2, matched continuation
   and full D4 under unchanged Monitor v0. Only then use task labels for grading.
4. Keep TEST closed. If D4 fails the replicated TRAIN gate, try one explicitly multimodal
   trajectory model with the same universal supervision; do not resume pair-level routers.

Panda transfer and visual-state learning remain downstream. First establish that the learned
state-space dynamics can preserve task-independent counterfactual topology at all.

## Completed gate: D5 multimodal trajectory model

**Completed and failed at the TRAIN pilot.** D5 uses three coherent complete-trajectory modes and
does not collapse, but commitment error is 2.67% worse than deterministic D4, so the frozen gate
for replication fails. No additional seeds or DEV run are permitted under this protocol. The steps
below remain as the executed design record.

The deterministic StepGraphNet can reduce average pairwise geometry error yet still average or
misplace narrow alternatives, so further loss tuning on the same single-future architecture is not
justified. The next experiment is one explicit multimodal conditional trajectory model:

1. Keep the 256-state TRAIN corpus, 64 probes, D4 topology/commitment/boundary losses, episode split,
   Monitor v0 and all task-label exclusions unchanged.
2. Replace only the deterministic transition output with a small conditional mixture over complete
   trajectory responses (shared GNN encoder, 3 trajectory modes, action-conditioned mixing weights).
   Train by proper mixture likelihood plus the same relational D4 losses on sampled/soft-assigned
   mode trajectories. Do not route using physical futures or task labels.
3. Compare frozen D2, matched deterministic D4 and matched multimodal D4 on TRAIN first. Require
   lower topology/commitment error, retention of candidate recall, noncollapsed mode occupancy and
   no worse quiet alarm rate. Use fixed seeds and report all seeds.
4. Only a passed TRAIN architecture gate may receive a new prospectively frozen DEV comparison.
   TEST remains closed. If this one multimodal model also fails, stop privileged Jenga architecture
   search and reassess the monitor/model interface before Panda or vision transfer.

## Active next decision: reassess the monitor/world-model interface

The current evidence rules out two simple explanations: D4 shows that average relational geometry
can improve without DEV monitor improvement, and D5 shows that explicit finite multimodality alone
does not improve local commitment. Do not add another loss or Jenga-specific state feature. Before
new training, perform a read-only interface analysis with the existing physical and learned caches:

1. Measure how monitor decisions change when predictions are periodically re-anchored to observed
   state after 1, 2, 4 or 8 steps, versus the current 38-step open-loop rollout. This separates
   transition-model error from the requirement to hallucinate an entire autonomous future.
2. Decompose disagreement by decision stage (candidate partition, boundary, commitment,
   persistence) and re-observation interval, using physical Monitor v0 as the structural reference
   and no task labels for model selection.
3. Compare two deployable decompositions conceptually and computationally: short-horizon
   re-observation with repeated monitoring, versus a learned direct local-boundary/response model.
   The latter must still predict task-independent counterfactual structure, not unsafe labels.
4. Freeze a new architecture protocol only after this analysis identifies which interface removes
   the dominant error. Keep TEST closed and defer Panda/vision transfer until the state-space
   interface itself is defensible.

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

**Frozen-v0 learned-rollout update (2026-09-25).** The previously trained D2 GNN has now been tested
with the final calibration-free Regime Monitor v0, rather than an older spread or action-branch
formulation. Across all ten D2 seeds, unchanged v0 obtains 31.1% mean TEST topple-fork recall
(8.3-48.8%) and 1.95% quiet alarms (0-5.3%). Physical v0 on the same TEST states is 65.5% / 3.5%.
The largest attrition is adaptive local boundary refinement: D2 averages 63.3/84 initial topple
candidates but only 32.5 refined boundaries, compared with 80 and 78 physically. Commitment and
persistence independently remain present in 52.4 and 60.7 states, but the same-pair conjunction
leaves 26.1 alarms. Only three topples alarm in >=8/10 seeds and 35 are robustly missed. Therefore
Step 8 is complete and fails: D2 does not preserve the local scaling geometry required by v0.
Result: `results/jenga/regime_monitor_v0/d2/aggregate.json`; evaluator:
`eval/jenga_regime_monitor_v0_d2.py`. Do not propose training the already-existing state GNN again.
Use TRAIN/DEV to diagnose and improve midpoint/action-width consistency, with v0 and TEST untouched.

**Failure localization and D3 status (2026-09-25).** Five bisections reduce action width by 32x.
Physical topple-pair gaps stay essentially constant (median final/initial early/full ratio
0.997/0.997), whereas D2 topple gaps fall to 0.074/0.097; D2 quiet gaps fall to 0.0326/0.0329,
almost exactly the smooth-response prediction 1/32. Among 780 seed-state copies of physical refined
topple boundaries, 177 are missed initially, 294 more collapse during refinement, 62 fail the final
conjunction and 247 alarm. Both D1 and D2 were trained on finite branches over six steps, with no
nested same-direction action widths or gap-scaling loss.

The nested-width D3 objective, full TRAIN dataset and matched DEV pilot are complete. The fixed
256-state dataset contains 182 contact and 74 quiet groups; 60 are smooth in all phases, 167 retain
a substantial response in at least one phase and 100 retain one into the late hold. It therefore
passes the intended coverage audit without using task or failure labels.

The prospective seed-1 comparison continued the same D2 checkpoint for three epochs in both arms.
The D2-only control gets 10/23 DEV topple alarms and 1/89 quiet alarms. D2+D3 gets only 3/23 and
3/89. D3's 21/23 coarse topple candidates collapse to three refined boundaries; its median DEV
topple early/full final-to-initial ratios are 0.040/0.042 versus the control's 0.082/0.085. Although
D3 reduces TRAIN global scale loss from 0.825 initially to 0.680 (control 0.807), the gain is
concentrated on quiet curves and does not transfer to true DEV plateaus. The pilot fails: do not
expand seeds or touch TEST. A hard-contact feedback diagnostic gets 11/23 topples but 13/89 quiet
alarms, establishing that discrete dynamics restore sensitivity without selectivity. The next
model experiment should learn a selective hybrid/contact mode from the same task-agnostic physical
scale curves, with matched soft/hard controls. Canonical result: `results/jenga/d3_pilot_summary.json`.

**Switching-edge implementation (2026-09-26).** The proposed architecture now exists as
`StepSwitchingEdgeGNN`. Unlike the old block-level MoE, it chooses a latent mode for every directed
interaction edge, uses mode-specific message experts, and carries that mode into the next predicted
step. The gate has a shared action-conditioned context pass and reads the previous mode plus an
explicit first-step flag. Straight-through Gumbel choices train the discrete path; evaluation is
deterministic argmax. Universal balance, entropy and persistence terms are integrated into CW/D3
and optional rollout training. `jenga_w5_eval.load_model` and therefore the frozen-v0 learned-
trajectory evaluator load `kind=edge_switch` checkpoints. Unit tests and a real one-batch D2+D3
smoke test pass. This is implementation validation only.

The next prospective experiment is a four-arm DEV comparison at matched seeds and optimization:
continuous soft-contact D2, naïve hard-contact D2, switching edge without D3, and switching edge
with D3. Record mode occupancy, entropy and switch rate alongside scale curves and frozen-v0 stages.
Require the learned switcher to improve the DEV recall/quiet tradeoff over both controls before
expanding seeds or touching TEST.

**Four-arm result (2026-09-26): feasibility gate failed.** The protocol was frozen before training
at `results/jenga/switching_edge_four_arm_protocol.json` (SHA-256
`f3bbc20700f459368c761c00bb525393ac789336a05f11acffdd5e676743dc3c`). At the fixed epoch-3
endpoint, frozen-v0 DEV results are:

| arm | topple alarms | quiet alarms | initial -> refined topple boundaries |
|---|---:|---:|---:|
| soft D2 | 10/23 | 1/89 | 19 -> 11 |
| hard-contact D2 | 11/23 | 9/89 | 21 -> 14 |
| switching D2 | 4/23 | 1/89 | 17 -> 6 |
| switching D2+D3 | 4/23 | 2/89 | 18 -> 5 |

Switching D2 has high normalized entropy (0.988): its soft occupancy appears balanced, but tiny
probability differences make deterministic argmax select one mode on 99.994% of edge-steps.
Switching+D3 collapses both soft and hard occupancy to mode 1 with entropy approximately zero.
Neither arm learns selective regimes; topple final/initial early/full gaps are 0.043/0.046 and
0.047/0.049, still near smooth contraction and below soft D2's 0.082/0.085. Do not add seeds or run
TEST. The next justified change is an identifiable routing mechanism that balances actual hard
assignments (e.g. Sinkhorn/optimal-transport or expert-choice routing) while permitting temporal
persistence; simply retuning the current soft balance/entropy weights after DEV is not justified.
Canonical summary: `results/jenga/switching_edge_four_arm_summary.json`.

**D1 control (2026-09-25).** The exact frozen-v0 evaluator has also completed on all ten existing
D1 checkpoints. D1 reaches 44.6% mean topple recall (29.8-60.7%) with 4.78% quiet alarms
(1.8-8.8%), compared with D2 at 31.1% / 1.95% and physical futures at 65.5% / 3.5%. It retains more
topple boundaries through refinement (46.1/84 versus D2's 32.5/84), yielding 37.5 rather than 26.1
final alarms, 13 rather than 3 robust alarms, and 26 rather than 35 robust blind topples. Its median
topple final/initial gap ratio is also materially higher: 0.263 early and 0.727 full, versus D2's
0.074/0.097, though still below the physical 0.997/0.997 plateau. D1 quiet ratios remain correctly
smooth at 0.0341/0.0341, but its quiet alarm rate and added-alarm rate (13.45%) exceed D2's 1.95%
and 8.45%. This supports boundary-scale supervision: finite pair-distance preservation helps, but
does not reliably distinguish physical persistent branches from model-created ones. Result:
`results/jenga/regime_monitor_v0/d1/aggregate.json`, SHA-256
`7011941ed32424210d97e9a1c03ec42580e45dac7f2358b8197993034fc9589a`.

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

The runtime monitor must be calibration-free. Raw latent spread and any percentile of nominal quiet
behaviour are diagnostics only; no quiet dataset, failure dataset or environment-specific distance
threshold may decide an alarm. Use the within-probe geometry instead:

1. generate 64 realistic execution-noise counterfactuals;
2. at an early and late future time, fit one versus two groups along PC1;
3. require two groups to win BIC, Ashman's D > 2, and at least two probes per side;
4. require the same binary partition at both times (at most one probe changes side).

BIC supplies its own complexity penalty, Ashman's D is dimensionless, and the minority/persistence
rules follow from the fixed probe design. The implementation is `src/counterfactual_monitor.py`.
It exposes no `fit` or `calibrate` method. The same rule and constants must transfer unchanged to
every task and representation. A continuous spread/AUC curve may still diagnose a world model, but
must never be reported as the monitor's deployed alarm.

**Status (2026-09-22): implemented, performance gate FAILS on Jenga.** On the frozen 1x benchmark,
the original persistent-PC1 rule over 10 D2 seeds has 50.4% mean fork recall (31.0-81.0%), 7.8%
quiet alarms (2.7-14.2%), 16 robust blind forks and no quiet state that alarms in >=8 seeds. D0 gets
53.0% recall, 11.9% quiet alarms and 18 robust blind forks. Thus D2 is retained because it suppresses
spurious quiet branching, but its calibrated 84%-recall spread result does not transfer to discrete
mode detection. A stricter pre-existing multi-mode/coarse-persistence diagnostic gets 37.1% recall
and 5.1% quiet alarms. Do not tune these constants on Jenga. Results:
`results/jenga/calibration_free_d2.json` and `calibration_free_d0.json`.

**Action-conditioned smooth-versus-branch follow-up (2026-09-22).** The PC1 mixture test can call a
strongly curved but continuous response a mode, or miss a branch not aligned with terminal PC1. The
new frozen alternative (`src/action_branch_monitor.py`) instead models the mapping from each H=8
execution perturbation to its whole future. One RBF-GP surface is the continuous null; two surfaces
sharing a candidate action partition are the branch model. An alarm requires the branch model's
log evidence to exceed the smooth model after both split-search and action-graph description-length
charges at hold 10 and hold 30. There is no learned threshold, quiet calibration or failure label.

Run the physical control first. With identical 64 probes, block-pose coordinates and snapshots at
hold 5/10/20/29/30, simulator trajectories reach **80/84 = 95.2% fork recall** with **18/113 =
15.9% quiet alarms**. Replacing only the trajectory source with 10 independently trained D2 models
gives **75.4% mean recall** (61.9-89.3%) and **33.0% quiet alarms** (27.4-42.5%), three forks missed
in >=8 seeds and twelve quiet states alarming in >=8 seeds. This is a clean dynamics-fidelity gap:
D2 removes about 19.9 recall points while adding 17.1 quiet-alarm points. Results:
`results/jenga/action_branch_ground_truth.json`, `action_branch_d2.json`, and per-seed files under
`results/jenga/bench_eval/`.

Do not tune the evidence sign or graph charge on Jenga to reduce the 15.9% physical quiet rate. The
next method step is a pre-declared structural refinement—derive candidate action regions from local
neighbourhood continuity rather than outcome-PC ordering—and test it simulator-first. The next
model step is to inspect matched false branches/missed branches against D2 trajectory error and
contact transitions. Only a rule that passes the simulator control unchanged should be carried to
D2 and then to pushing/insertion.

**Held-out-probe refinement fails on development (2026-09-22).** Before changing the candidate
family, `cross_validated_smooth_vs_branch_alarm` tested the simpler overfitting hypothesis. In eight
folds it hides eight trajectories, discovers/fits the branch on the other 56, assigns hidden probes
from action neighbours only, and compares unseen-trajectory prediction error. Synthetic smooth,
branch and transient controls pass. On ground-truth DEV, however, fork recall falls from 23/23 to
14/23 (60.9%) while quiet alarms barely change from 27/89 to 26/89 (29.2%). It retains 19 old quiet
alarms, removes eight, and creates seven. Thus the false alarms are not mainly splits that fail to
generalise; narrow true branches are harder to reconstruct when eight probes are hidden. The rule
fails its development gate, is not promoted, and TEST remains unrun. Result:
`results/jenga/action_branch_cv_ground_truth_dev.json`. Next inspect the physical trajectory/contact
meaning of DEV quiet alarms before proposing another structural statistic.

**Adaptive boundary refinement implemented (2026-09-22): partial improvement, gate still fails.**
For each in-sample alarm, `eval/jenga_action_boundary_refine.py` takes the three nearest action pairs
on opposite response sides and simulates five successive midpoint bisections. At each bisection the
bracket width halves. A generic local-continuity null fits trajectory gap as linear + quadratic in
bracket width and therefore goes to zero; the branch model adds a nonnegative intercept. BIC must
favour the intercept for at least two of three brackets at both hold 10 and hold 30. No topple,
contact, object identity, quiet example or absolute distance enters the alarm.

On ground-truth DEV it retains **23/23 topple forks** while reducing quiet alarms from **27/89
(30.3%) to 21/89 (23.6%)**. All 12 initially detected nudge forks, all 7 small splits and all 3 weak
states also survive. Twenty-two of 23 topple forks support a plateau on all three early brackets and
all 23 do so on all three late brackets. The refinement therefore removes six smooth/curved quiet
responses without sacrificing the target forks, but most quiet alarms also contain a reproducible
finite local response gap. This misses the <=5% quiet gate, so the rule is not promoted and TEST
remains untouched. Do not tune pair count, bisection depth, BIC sign or vote count on Jenga. Result:
`results/jenga/action_boundary_refine_dev.json`.

The next universal question is now semantic rather than merely geometric: are the 21 retained quiet
cases genuine generic physical regime changes (contact mode, support, slip, object rearrangement),
or artifacts of the trajectory representation? Characterise them with generic state/contact-change
descriptors for diagnosis only; do not add Jenga failure logic to the alarm.

**Generic physical-regime audit completed (2026-09-22).** `eval/jenga_generic_regime_audit.py`
records every control step over the existing H=8 + hold-30 horizon for the final 32x-narrow brackets:
anonymous pose, velocity and binary contact edges. Cohorts are 23 forks, 21 retained quiet alarms,
6 refinement-rejected quiet alarms and 21 state/action-matched never-alarmed quiet controls. Outcome
classes form report cohorts only.

Retained quiet alarms are not numerical nulls: versus matched controls, median contact-edge
difference is 0.0088 vs 0 (Mann-Whitney p=0.00064), final pose gap 0.0105 vs ~0 (p=3.3e-6), and peak
velocity gap 0.279 vs ~0 (p=8.5e-6). But forks are a categorically stronger regime: persistent
contact branching on >=2/3 brackets is 91.3% vs 28.6%; late contact branching 87.0% vs 4.8%; final
pose gap 1.67 vs 0.0105; peak velocity gap 8.49 vs 0.279. Every fork-vs-retained continuous contrast
is p<=5.2e-6. Retained and refinement-rejected quiet groups are not clearly different except peak
velocity (p=0.042, uncorrected).

Therefore BIC branch *existence* is working: it finds real local sensitivity. It does not answer
whether that branch is consequential. A post-hoc persistent-contact gate would happen to retain
21/23 DEV forks and 6/89 quiet states, but adopting it would both tune on Jenga and exclude universal
non-contact consequences, so it is diagnostic only. The next method must encode task-independent
effect significance—persistent corroboration across generic dynamics channels relative to the
intervention—not a named contact or Jenga magnitude threshold. Result:
`results/jenga/generic_regime_audit_dev.json`. TEST remains untouched.

**Shared-versus-branch dynamics test completed (2026-09-22): specificity passes, recall collapses.**
The predeclared follow-up uses only the three final 32x brackets that survive boundary refinement.
It densely records anonymous pose and velocity over H=8 + hold 30, subtracts each endpoint's pose at
H=8 so a static displacement cannot itself trigger the rule, and fits transitions rather than
endpoints. The null is one pooled nonlinear GP transition law for both bracket sides. The alternative
removes cross-branch covariance, representing two laws, and pays a BIC-style one-regime-variable
description charge. At least two of three brackets must favour the alternative independently over
hold 1-10 and the disjoint hold 11-30 interval. Contact channels and outcome labels are absent.

Ground-truth DEV result: **4/23 topple forks (17.4%)** and **1/89 quiet alarms (1.1%)**. The frozen
gate (>=21/23 forks and <=4/89 quiet) fails. Loosening only the vote does not rescue the hypothesis:
requiring any persistent bracket gives 8/23 and 6/89; any early branch gives 17/23 and 8/89; any
late branch gives 16/23 and 11/89. No threshold-free structural variant of these same evidence signs
meets the target. TEST is untouched and D2 must not be run.

Interpretation: this is not merely a scaling bug. A deterministic Markov simulator has one physical
transition law everywhere; a fall and a settle can be different trajectories under that same law.
Hybrid contacts may make a local two-law approximation occasionally useful, explaining the very
high specificity, but “consequential branch” is not equivalent to “different governing dynamics.”
The experiment therefore falsifies this formulation while preserving the broader objective. Next
test a universal trajectory-level consequence criterion based on persistent amplification and
irreversibility of nearby counterfactual separation, normalized by their action-space separation
and by their own pre-effect variation. Predeclare it on synthetic mechanisms (stable offset,
decaying perturbation, sustained divergence, stick/slip, free-flight/contact) before DEV; do not use
Jenga labels or contact identity in the alarm. Result:
`results/jenga/shared_dynamics_ground_truth_dev.json`.

**Boundary + amplification + persistence completed (2026-09-22): post-action growth is not
necessary.** `src/consequence_monitor.py` was frozen first against synthetic mechanisms. It rejects
smooth shrinking responses, static offsets and recoverable wobble, while accepting sustained
divergence, stick/slip and supported/free-flight trajectories. The Jenga evaluator then replays all
three final 32x brackets from every original DEV action-branch candidate. For each bracket it uses:

* the existing early+late BIC plateau evidence for local boundary scaling;
* an internally normalized anonymous pose/velocity difference-of-displacement curve, with each
  endpoint's H=8 pose as its own origin;
* BIC evidence for a sustained nonnegative growth component after H=8;
* BIC evidence for a nonzero late asymptote rather than decay to zero.

The same pair must pass all three tests, and at least two of three pairs must agree. No distance,
growth-ratio or safe-population threshold is fitted.

Ground-truth DEV result: **3/23 topple forks (13.0%)**, **2/89 quiet alarms (2.2%)**, 9/17 nudge
forks, 1/8 small splits and 0/3 weak cases. The >=21/23, <=4/89 gate fails, so TEST and D2 remain
untouched. The stage decomposition is decisive:

| requirement on >=2/3 refined pairs | topple forks | initial-alarm quiet |
|---|---:|---:|
| local boundary | 23/23 | 21/27 |
| post-H=8 amplification | **3/23** | **2/27** |
| persistence | 23/23 | 26/27 |
| same pairs pass all three | 3/23 | 2/27 |

Thus amplification supplied specificity but imposed the wrong timing assumption. The target topple
branches are usually created during the H=8 action and then persist or settle; continued growth
after the differing action ends is not necessary. Nudge forks are more likely to continue moving
through the hold, explaining their 9/17 alarm rate. Persistence alone is nearly universal among the
already-refined branches and cannot define consequence.

Next formulation: retain local action-width scaling, but measure amplification from action
separation to physical trajectory separation over the **entire H=8 + hold** interval. Separately
test late commitment/reconvergence. The structural alternatives should distinguish (a) proportional
smooth response, (b) finite but passive static displacement, and (c) a finite branch that becomes
committed at any point during the action or immediate hold. Freeze synthetic cases before DEV and
do not select a time window from these labels. Result:
`results/jenga/consequence_ground_truth_dev.json`.

**Whole-trajectory action-to-consequence commitment completed (2026-09-22): large recovery, gate
still fails.** The follow-up fixes the prior timing error without changing labels or fitting a score
cutoff. Its passive null explains the full separation curve using cumulative refined-action exposure,
a nonlinear exposure term, a passive static component and post-H relaxation. The alternative adds
one nonnegative commitment event whose onset is searched over H=8 and the first five hold steps;
abrupt and two fixed gradual rise shapes are considered, with the complete candidate search charged
in description length. Late non-reconvergence is still tested independently. The same >=2/3 pairs
must pass boundary scaling, commitment and persistence.

Synthetic passive displacement and recoverable response are rejected; during-action commitment,
stick/slip and free-flight are accepted before Jenga scoring. Ground-truth DEV gives:

| rule | topple-fork recall | quiet alarms |
|---|---:|---:|
| post-H amplification | 3/23 = 13.0% | 2/89 = 2.2% |
| whole-trajectory commitment | **18/23 = 78.3%** | **7/89 = 7.9%** |
| required gate | >=21/23 = 91.3% | <=4/89 = 4.5% |

Among topple forks, boundary and persistence majorities remain 23/23; commitment is 18/23 and fully
determines the five misses. Among 27 original quiet candidates, boundary passes 21, persistence 26,
commitment 8 and the full same-pair conjunction 7. The recovered 15 topples confirms that commitment
must be allowed during the action. However, the detector also identifies seven reproducible benign
commitments, so time coverage alone does not solve generic consequence.

Do not tune the evidence margin, onset range, vote, or rise constants against these labels. TEST and
D2 remain untouched. The next structural experiment should replay **every bisection level densely**,
not only the final 32x bracket, and ask whether commitment onset and normalized response shape
converge consistently as action width shrinks. That tests whether the fitted event is a stable local
boundary property rather than a one-resolution approximation error. If true and benign branches
remain indistinguishable after that control, the project must explicitly confront the information
limit: task-independent structure alone may not define which real regime changes are safety-
consequential without a representation of generic physical severity. Result:
`results/jenga/whole_trajectory_consequence_dev.json`.

**Dense multi-resolution commitment convergence completed (2026-09-22): rejected.** The evaluator
reconstructs and densely replays both endpoints at all six widths for every candidate pair: original,
2x, 4x, 8x, 16x and 32x refinement. The frozen rule requires:

* the original early+late boundary-scaling evidence;
* positive whole-trajectory commitment evidence at each of the final three widths;
* fitted commitment onset spanning at most one recorded control step across those widths;
* BIC support that successive normalized-curve changes vanish with bracket width rather than
  approach a nonzero shape-change plateau;
* final-width late persistence;
* all clauses on the same >=2/3 pairs.

Synthetic stable-limit boundaries pass and alternating mechanisms fail. Ground-truth DEV gives
**2/23 topple forks (8.7%)** and **1/89 quiet alarms (1.1%)**, far below the recall gate. Diagnostic
stage majorities among 23 forks / 27 initial quiet candidates are:

| structural clause on >=2/3 pairs | topple forks | initial quiet candidates |
|---|---:|---:|
| boundary | 23/23 | 21/27 |
| stable commitment over final three widths | 13/23 | 1/27 |
| shape convergence | 6/23 | 1/27 |
| final persistence | 23/23 | 27/27 |
| complete same-pair conjunction | 2/23 | 1/27 |

At pair level, only 46/69 topple brackets retain positive commitment at all final three widths and
51/69 have one-step onset agreement. The dominant rejection is trajectory-shape convergence. This
is not evidence that the other 21 topple forks are unreal: all already pass local finite-gap and
persistence tests and are graded by independent physical outcomes. Rather, a hybrid boundary's
finite-resolution trajectory timing and shape can vary strongly as endpoints approach it; five
bisections do not guarantee the asymptotic limit assumed by the test.

Do not weaken the final-level count, onset tolerance or convergence BIC after seeing DEV. The
experiment successfully removes six of seven quiet alarms from the one-resolution rule but also 16
of 18 detected topples, so it is not a usable monitor. The central unresolved issue is now semantic,
not another continuity test: the seven quiet alarms are real committed physical regime changes but
not Jenga topples. There are two defensible research directions:

1. Define a **generic physical-severity representation** (e.g. normalized displacement, support,
   kinetic release or recoverability) and freeze it across Jenga, pushing and insertion; this is no
   longer purely scale-free branch detection, but can remain failure-label-free and task-agnostic.
2. Define every real committed regime change as a valid monitor positive, report the Jenga "quiet"
   cases as benign interventions rather than statistical false positives, and evaluate usefulness
   through downstream intervention cost rather than topple specificity.

Do not transfer to TEST/D2 until one interpretation is chosen prospectively. Result:
`results/jenga/multiresolution_consequence_dev.json`.

**Local corrective recoverability completed (2026-09-22): not equivalent to severity.** This is the
prospective test of direction 1 above, not a post-hoc Jenga rule. At each final 32x bracket endpoint,
the simulator applies the same seven corrections: neutral plus/minus the three PCA axes of the
frozen execution-error snippets at one standard deviation. Their magnitudes are 2.46, 0.68 and
0.39 mm. Each correction is held for five control steps, followed by ten steps at the shared nominal
target. The two resulting seven-point anonymous pose/velocity reachable sets are compared.

The overlap decision is internally scaled: minimum cross-set distance is divided by the geometric
mean of each set's median within-set nearest-neighbour spacing. Ratio <=1 means overlap at the
library's own resolution; ratio >1 means locally unrecoverable. A state alarms when the same >=2/3
pairs have the original local boundary and non-overlapping reachable sets. No outcome label,
distance unit, object role, quiet calibration or learned cutoff enters.

Ground-truth DEV result: **15/23 topple forks (65.2%)** and **12/89 quiet alarms (13.5%)**. Among 27
initial quiet candidates, 12 are locally unrecoverable. The correction test therefore performs worse
than the one-resolution commitment rule on both target metrics. Combining their frozen decisions
does not rescue the gate:

| combination | topple forks | quiet alarms |
|---|---:|---:|
| commitment only | 18/23 | 7/89 |
| recoverability only | 15/23 | 12/89 |
| commitment AND recoverability | 12/23 | 5/89 |
| commitment OR recoverability | 21/23 | 14/89 |

Interpretation: local recoverability and safety severity are different. A benign static displacement
or contact-mode change can be impossible to undo with a bounded millimetre-scale EE correction,
while two branches involving a topple can still have overlapping sampled reachable sets after the
robot moves. The enormous overlap ratios in some quiet cases are caused by nearly immobile reachable
sets, not a numerical alarm threshold. Increasing correction radius after seeing these labels would
introduce a new environment-specific control-authority parameter and can itself cause new regimes;
do not tune it on DEV.

This closes the simplest calibration-free recoverability formulation. To pursue severity, define a
dimensionless **cost of consequence** prospectively—e.g. corrective control effort normalized by
available authority, affected-mass/scene-scale motion, or loss of support/control—and freeze it
across at least Jenga, pushing and insertion before scoring. Otherwise adopt direction 2: call all
real committed regime changes intervention-worthy and evaluate task completion/intervention cost.
TEST and D2 remain untouched. Result: `results/jenga/recoverability_ground_truth_dev.json`.

## Plan decision — separate detection from intervention (2026-09-22)

Adopt the two-layer architecture. Stop asking one calibration-free statistic to both discover
unknown physical regimes and decide their task cost.

### Layer 1 — universal regime-sensitivity monitor

The monitor's claim is deliberately limited:

> Under realistic execution variation, nearby actions can commit the scene to distinct persistent
> physical futures.

Freeze the one-resolution whole-trajectory commitment rule as **Regime Monitor v0**
(`whole_trajectory_consequence_alarm`). It is currently the strongest prospectively frozen
ground-truth rule: 18/23 DEV topple forks and 7/89 quiet alarms. Those seven quiet alarms must be
reported as **benign committed branches / intervention candidates**, not statistical false alarms,
unless downstream evaluation shows that intervention has no value. Keep topple labels as one
grading slice, not the definition of the monitor.

Regime Monitor v0 remains calibration-free in the intended sense: no safe reference set, failure
labels, fitted percentile, named object role or environment-specific distance threshold. Each task
may estimate its own realistic execution-error distribution; that is part of the plant/controller,
not alarm calibration. Freeze the following before any new score:

* 64 execution probes, H=8 and the same immediate hold protocol;
* action-conditioned candidate discovery and three five-bisection brackets;
* whole-trajectory passive-response versus commitment comparison;
* description-length search charge, late persistence and >=2/3 same-pair vote;
* internal representation scaling and the alarm evidence sign.

Do not add multi-resolution convergence, local recoverability, contact gates or a positive BIC
margin. They are documented negative results. Do not call v0 a failure predictor or claim that every
alarm is dangerous.

### Layer 2 — consequence-aware intervention policy

Layer 2 consumes the structural alarm plus ordinary task/controller context and chooses among
`continue`, `slow`, `reobserve`, `replan`, or `stop`. It is allowed to express costs because action
selection always contains deployment preferences. Keep it separate from the universal detector so
that task cost is not disguised as physical branch evidence.

Evaluate policies by outcomes, not by relabeling Layer-1 alarms:

* failures prevented and residual failure rate;
* successful executions interrupted;
* task completion rate and extra control steps/time;
* alarm rate and intervention rate separately;
* success after reobservation/replanning;
* Pareto curves over failure cost versus intervention cost.

Start with low-cost responses (`reobserve`, then `replan`) rather than treating every Layer-1 alarm
as an emergency stop. Compare at minimum: no monitor, intervene on every v0 alarm, and an oracle
outcome-labelled upper bound used for grading only.

A learned or analytic generic consequence-cost model is optional and comes **after** cross-task
Layer-1 evaluation. If pursued, its inputs must be prospective dimensionless quantities such as
scene-scale-normalized affected motion, motion propagation, or corrective effort divided by
available control authority. Freeze one formulation across all tasks; never tune it on the seven
Jenga benign branches.

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

The frozen algorithm for this phase is Regime Monitor v0 above. Cross-task evaluation has two
separate scoreboards:

1. **Detector fidelity:** recall on independently verified physical regime forks, committed benign
   branch rate, candidate reach/censoring, and agreement with ground-truth counterfactual geometry.
2. **Intervention utility:** failures, completion, delay and intervention cost when the alarm drives
   a fixed low-cost response policy.

Do not collapse these into one Jenga-style false-positive number. A benign committed branch is a
valid Layer-1 detection and may still be an unnecessary Layer-2 intervention.

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
                                      |
                                      v
                         Regime Monitor v0
          "nearby execution errors commit to distinct futures"
                                      |
                                      v
                  consequence / intervention policy
             continue | slow | reobserve | replan | stop
```

Layer 1 is never told whether the event is a topple, slip, jam, collision, fall or grasp loss.
Layer 2 may use task cost and control authority, but its decisions and metrics must not be presented
as evidence that Layer 1 discovered a universal notion of danger.

## What NOT to do next

Do not spend substantial time on: improving Jenga from 67% to 70% before testing another task;
another Jenga-specific object detector; more Jenga-specific state variables; reviving MoE without
new evidence; scaling the GNN to reduce rollout MSE; training a topple classifier; tuning thresholds
with failure labels; assuming visually impressive generative WMs preserve safety-relevant action
sensitivity.

Also stop adding Jenga-only structural clauses to remove the seven benign committed branches. Do
not tune correction radius, overlap ratio, convergence depth, onset tolerance or BIC margin. Do not
use TEST to choose between monitor formulations.

---

## Immediate execution order

1. Tag and checksum **Regime Monitor v0** and its exact ground-truth DEV protocol. Preserve all
   rejected formulations as ablations; make no further Jenga-driven changes.
2. Run v0 once on the untouched Jenga TEST split. Report topple-fork recall, benign committed-branch
   rate, candidate reach and complete per-state evidence. Do not use TEST to revise v0.
3. Build a minimal intervention wrapper for the simulator: on alarm, first reobserve; if the alarm
   persists, replan or choose the most branch-robust available action. Measure completion, failures,
   interventions and delay against no-monitor and grading-oracle baselines.
4. Establish a ground-truth **pushing** benchmark with measured/simulated realistic execution error,
   including stick/slip and edge/fall branches plus benign contact branches.
5. Establish a ground-truth **insertion** benchmark with free motion/contact/jam branches and benign
   contact-mode alternatives.
6. Run the unchanged v0 detector and the same intervention protocol on Jenga, pushing and insertion.
   This is the universality test; task labels grade results but never enter Layer 1.
7. Apply v0 unchanged to representations of rendered real futures. The ground-truth-to-rendered gap
   tests representation fidelity without world-model error.
8. Apply v0 to D2/world-model futures. The rendered-to-predicted gap tests counterfactual dynamics
   fidelity. Only then consider direct visual-latent or multi-task world-model changes.
9. If benign interventions remain operationally expensive across tasks, predeclare one dimensionless
   consequence-cost model and evaluate it as Layer 2—not as a modification of v0.

Steps 1-2 are complete. Step 3 is the immediate handoff; steps 3-6 establish whether the method is
useful and universal, and steps 7-8 establish whether visual world models preserve the required
signal.

**Step 1 complete (2026-09-22).** Regime Monitor v0 is frozen in
`results/jenga/regime_monitor_v0/manifest.json`, protocol SHA-256
`4e1a20ca3c64bf18733315fe31df5850529027f8f48cda2c5f547ce43ecdc317`. The manifest binds the
monitor/configuration source, DEV boundary evidence and result, benchmark/cache inputs and simulator
archive. Any byte change fails closed. Verify before every future v0 evaluation with:

```bash
python eval/jenga_regime_monitor_v0_freeze.py verify
```

Do not overwrite this manifest to accommodate code changes; create a new named protocol version.

**Step 2 complete (2026-09-23).** The verifier-gated evaluator
`eval/jenga_regime_monitor_v0_test.py` consumed the previously untouched TEST split exactly once and
wrote the immutable result `results/jenga/regime_monitor_v0/test_ground_truth.json` (SHA-256
`9a7b4004647f09a165bd52d2ea959dc0ef81e93cadd760eb256327917cadf0f5`). The evaluator has no
`--force` or partial-run option, refuses to overwrite the canonical result, verifies the frozen v0
manifest and benchmark inputs, and checks every TEST action chunk against the frozen benchmark
before simulation. Labels are attached only after alarms are computed.

| TEST class | states | initial candidates | refined boundaries | v0 alarms |
|---|---:|---:|---:|---:|
| topple fork | 84 | 80 (95.2%) | 78 (92.9%) | **55 (65.5%)** |
| quiet | 113 | 18 (15.9%) | 16 (14.2%) | **4 (3.5%)** |
| nudge fork | 10 | 10 (100%) | 7 (70.0%) | 3 (30.0%) |
| small split | 7 | 6 (85.7%) | 6 (85.7%) | 4 (57.1%) |

The topple-fork Wilson 95% interval is 54.8-74.8% and the episode-clustered interval is
54.2-76.9%. The quiet committed-branch intervals are 1.4-8.7% and 0.8-7.3%. Relative to DEV,
topple recall falls 12.8 percentage points (78.3% to 65.5%) while the quiet committed-branch rate
falls 4.3 points (7.9% to 3.5%). Candidate discovery and boundary refinement generalize: only six
topples are lost before the final monitor stage. The main recall bottleneck is the frozen
commitment/persistence conjunction, which removes another 23. This TEST evidence must not be used
to revise v0; any redesign requires a new benchmark cycle.

Gate 1 therefore supports v0 as a useful but moderate-recall **Layer-1 structural signal**, not as
a standalone emergency-stop safety filter. Its four quiet alarms remain benign committed physical
branches under the adopted semantics, not automatically false positives. At runtime the raw burden
would be 66/214 alarmed states and 37/53 episodes with at least one alarm, which is too expensive
for an unconditional stop. This motivated the now-complete step 3 experiment below: keep v0 frozen
and test a low-cost reobserve-then-robust-action wrapper against no-monitor and grading-oracle
baselines.

**Step 3 complete (2026-09-23).** The intervention protocol was prospectively frozen before its
canonical run (`results/jenga/intervention_v0/manifest.json`, protocol SHA-256
`4ea6cc2a413ff2f46661869f16dc9d496b0ae9094c35b99d1e16e490d10caf40`). It replays all 100
recorded action trajectories with fresh reset seeds (`2000 + episode_id`) under four matched arms:
no monitor, one-step reobservation, the label-blind wrapper, and a privileged local oracle. On a
persistent v0 alarm, the wrapper compares fixed 100/75/50/25/0% action-displacement candidates and
chooses the fewest consequential v0 pairs, breaking ties toward the intended action. No Jenga
geometry, outcome label or fitted threshold enters this policy.

| arm | neighbor failures | pick successes | safe completions | episodes reobserved | modified chunks |
|---|---:|---:|---:|---:|---:|
| no monitor | 34 | 80 | 51 | 0 | 0 |
| reobserve only | 31 | 79 | 53 | 97 | 0 |
| **v0 wrapper** | **25** | **79** | **59** | **97** | **173** |
| local grading oracle | 20 | 79 | 63 | 97 | 17 |

The wrapper prevents ten baseline failures and creates one, for a net reduction of 9 percentage
points (paired episode bootstrap 95% CI -15 to -3 points; exact McNemar p=0.0117). Safe completion
rises by 8 points (CI +1 to +15; p=0.0386), while pick success changes by -1 point (CI -6 to +4).
Against reobservation alone, robust action selection removes six additional failures and creates
none (difference -6 points, CI -11 to -2; p=0.0313). It captures 64% of the oracle's net failure
reduction and 67% of its safe-completion gain.

The operational caveat is breadth, not per-event delay. Across 2,049 decisions the wrapper invokes
275 one-step reobservations (13.4% of decisions), of which 94 clear immediately; 97/100 episodes see
at least one. It modifies 173 persistent-alarm chunks. The oracle modifies only 17, showing that the
label-blind robustness rank remains conservative even though its episode-level utility is positive.
Do not tune these scales or add a Jenga consequence gate from this result. Gate 3 passes for
ground-truth Jenga with a conservative-burden qualification. The next step is step 4: establish the
prospective ground-truth pushing benchmark, then carry the **unchanged** v0 detector and intervention
policy to pushing and insertion to test universality.

**Timing follow-up complete (2026-09-23).** The immutable policy decisions were replayed without
rerunning the monitor. Among episodes where both the baseline and treatment pick successfully, mean
added time to the first pick criterion is 0.249 s for reobserve only, 0.230 s for the v0 wrapper and
0.252 s for the physical oracle; all have median 0.2 s. Mean full-episode execution time is 16.974 s
without monitoring, 17.244 s for reobserve and 17.249 s for wrapper/oracle. Thus the direct control
delay is small: roughly 0.27-0.275 s per episode. Action scaling can affect which episodes succeed,
so paired common-success timing is the primary comparison rather than unpaired successful-arm means.
These are **physical control-time** results only. They exclude monitor computation: the exhaustive
ground-truth reference takes about 18 s even for a quiet decision in an isolated smoke test and much
longer when refinement/candidate search fires. It is not a real-time implementation; runtime
reduction remains a later requirement after cross-task validity.

The replay also exposed one 45-degree neighbor-grade mismatch in each monitored arm. The original
Jenga simulator snapshot uses `mjSTATE_FULLPHYSICS`, which omits solver warm-start; interleaved
counterfactual rollouts can therefore leave tiny hidden numerical state behind after restoration.
The mismatch rate is 1/100 per monitored arm and does not erase the intervention effect, but future
cross-task benchmarks must snapshot `mjSTATE_INTEGRATION`. Result:
`results/jenga/intervention_v0/time_to_success.json`.

**Step 4 environment-design substep complete (2026-09-23); benchmark freeze still pending.** New
MuJoCo prototypes in `src/systems/contact_benchmarks.py` provide (a) planar pushing into a goal with
stick/slide, glancing and support/edge-fall regimes, and (b) tight peg insertion with free motion,
rim contact, insertion and jam regimes. They command tool pose directly to isolate contact mechanics
before adding arm/perception confounds, and use integration-state snapshots. Paired success/failure
videos and a design document are in `results/contact_benchmarks/previews/` and
`CONTACT_BENCHMARKS.md`. Next define prospective execution-error distributions, sample mechanism
coverage, and freeze DEV/TEST splits before running unchanged v0; do not treat the previews as a
completed benchmark evaluation.

**Steps 4-6 U1 result (2026-09-23): cross-task detector gates FAIL; intervention utility
inconclusive.** A 480-state pushing/insertion panel was graded for coverage and frozen before v0
scoring (protocol `3f2386af...`). On untouched TEST, v0 alarms on 0/12 pushing and 0/11 insertion
consequential outcome forks, versus 3/108 and 0/109 unanimous quiet states. Candidate discovery is
high-recall but nonspecific; the frozen commitment/persistence conjunction removes every true
cross-task fork. The natural 100-episode-per-task intervention follow-up has zero baseline failures
in both tasks, so unchanged outcomes cannot establish utility. Preserve these negative results.

Next, do **not** tune v0 on U1. Define a new protocol version with: (a) a label-only natural-episode
coverage gate requiring both failures and successes after an explicit terminal settle; and (b) a
new task-general Layer-1 hypothesis motivated by U1: directly test whether nearby actions predict
different future transition laws/contact graphs, rather than requiring Jenga-like autonomous
amplification after action exposure. Develop that formulation on new cross-task DEV mechanisms,
freeze it, and reserve new TEST episodes. Only after it passes ground truth should rendered and
world-model transfer resume.

**Panda block-push environment built (2026-09-24).** The abstract blue-pusher prototype is replaced
for future work by a Panda operational-space task with one upright Jenga block and a planar goal.
Success requires reaching the goal without ever crossing 45° tilt. Full visual checks cover upright
success, upright overshoot, path-level toppling and contact loss. This is only
environment validation: no monitor, dataset expansion or intervention evaluation has been run.
Use a small DEV mechanism panel next to develop the ground-truth transition-law monitor, then move
to rendered/world-model representations; do not expand task benchmarking first.

**Panda upright-push DEV panel and v0 diagnostic complete (2026-09-24).** A label-only coverage
scan selected and froze 56 states before monitor scoring: 20 mixed upright/topple forks, 15 stable
centered pushes, 10 unanimous-topple states, five contact-loss controls and six upright overshoots.
Unchanged v0 alarms on 19/20 mixed forks, 0/15 stable centered states and 0/6 overshoots. It alarms
on 5/5 contact-loss controls, correctly exposing a distinct physical regime without requiring a
topple, and on 1/10 unanimous-topple states whose failing futures still branch. All 20 true forks
pass commitment and persistence; the sole miss fails boundary refinement. Therefore do not replace
v0 yet from this DEV evidence. Next run the same frozen state/action futures through rendered visual
features and then the world model to localize representation loss. Reserve a fresh pushing TEST set.

**Rendered-future transfer steps 1-2 complete (2026-09-24): representation Gate 4 fails for the
baseline full-frame adapter.** All 56 x 64 frozen Panda probes were rendered at hold 5/10/20/29/30;
visual-selected bisections and final endpoint trajectories were also rendered exactly and densely.
The adapter is DINOv2-S/14 -> fixed 4x4 spatial pooling -> seeded 45-D random projection, with no
labels, fitted threshold or calibration states; frozen v0 is unchanged. Final decisions agree with
physical v0 on 33/56 states (58.9%). It recovers 8/25 physical alarms overall and 6/19 mixed
upright/topple alarms, with six visual-only alarms. All 56 states become visual candidates and 54
survive visual refinement/persistence, but only 16 have commitment majorities. Thus nuisance visual
branching is over-sensitive upstream while task-relevant action-to-consequence geometry is lost in
the generic 45-D adapter. Do not run predicted-world-model futures yet.

Next execute the already predeclared representation ladder on the same DEV futures, changing only
the visual readout: (1) DINO patch features with fixed background suppression, then (2) a
simulator-segmented block-region pool as a diagnostic upper bound. Compare each directly with
frozen physical-v0 stage decisions, not topple labels. If a representation passes on DEV, freeze
camera/preprocessing/readout and run one fresh Panda TEST panel. Only then evaluate world-model
predictions to isolate dynamics error from representation error. Result:
`results/panda_block_push/visual_v0/visual_v0_diagnostic.json` (SHA `8a84de40...`).

**PCA45 readout follow-up complete (2026-09-24): useful but Gate 4 still fails.** Replacing only
the seeded random projection with per-state, unlabeled PCA45 raises final physical/visual agreement
from 33/56 to 37/56, physical-alarm recall from 8/25 to 11/25, and mixed-fork recall from 6/19 to
10/19. Mixed-fork commitment majorities rise from 8/20 to 16/20, so random projection was a real
source of information loss. PCA captures a median 93.4% of pooled DINO variance. Nevertheless, all
56 states remain initial candidates, visual-only alarms remain 5, and all five physical contact-loss
alarms are missed. Therefore PCA alone is not an adequate visual representation. Keep PCA45 as the
preferred full-frame baseline and continue with fixed background suppression, followed by the
segmented block-region diagnostic upper bound. Result SHA `f4c6905e...`.

**Representation ladder complete (2026-09-24): signal is visible, but the current visual
formulation is not specific.** Motion-weighted DINO+PCA45 worsens agreement to 30/56, recovers
12/25 physical alarms and creates 13 visual-only alarms. Simulator-segmented block DINO+PCA45
recovers 21/25 physical alarms (16/19 mixed forks and 4/5 contact-loss), but creates 21 visual-only
alarms, including 10/15 stable-centered pushes and 5/6 overshoots. Full-frame PCA45 remains the
best agreement/specificity baseline at 37/56 and five visual-only alarms. Every readout still makes
all 56 states initial candidates.

Interpretation: removing background is insufficient; explicit object isolation recovers the signal
but does not separate smooth consequential motion from physical regime branching. The current
45-D compatibility adapter is also structurally mismatched: v0's consequence stage treats fixed
channel slices as position/rotation/linear/angular families, whereas PCA axes are exchangeable.
Next build a visual-native consequence formulation using tracked object geometry or patch
correspondence and one internally normalized representation-distance curve. Keep object masks as an
oracle diagnostic, not runtime input. Do not evaluate or retrain the world model until true rendered
features pass this gate. Results: motion SHA `2328bb22...`, block SHA `a09f7c09...`.

**Proper visual-native consequence port complete (2026-09-24).** The invalid semantic slicing of
PCA axes is replaced by one RMS-normalized Euclidean separation curve over the complete visual
latent. Boundary scaling, action exposure, commitment/persistence BIC, zero decisions and majority
logic remain unchanged. On segmented-block PCA45 this recovers 24/25 physical alarms: 18/19 mixed
forks and 5/5 contact-loss regimes. This is strong evidence that rendered visual futures contain
the signal and that the visual-native consequence stage preserves it. Specificity still fails:
18 visual-only alarms remain (8 stable-centered, five physically quiet unanimous-topple, five
overshoot), and all 56 states are initial visual candidates. Full-frame PCA is more specific but
low-recall (10/25 physical alarms, five extras).

The next experiment is precisely localized: fix upstream visual response geometry, not the
consequence model. Test tracked block geometry or patch correspondence so ordinary translation and
view-dependent DINO changes form a smooth action-response surface. Keep the visual-native
consequence stage frozen during that comparison. Do not run predicted futures yet. Result SHA
`b5afd81f...`.

**DINO patch-correspondence geometry tested (2026-09-24): negative.** A predeclared universal arm
tracks all 256 start-frame DINO patches with top-4 soft cosine matches and supplies only normalized
x/y position and finite-difference velocity to PCA45 plus the visual-native monitor. It uses no
mask, identity, labels or confidence threshold. It agrees with physical v0 on 33/56 states,
recovers 10/25 physical alarms (8/19 mixed forks), creates eight visual-only alarms, and still makes
all 56 states initial candidates. Coarse 16x16 semantic patches are not stable geometric tracks;
identity swaps and sub-patch aliasing preserve the nonsmooth-response problem. Do not tune matcher
constants on this panel. Next predeclare a higher-resolution temporally constrained point-tracking
or optical-flow baseline, or move to a dynamics-trained equivariant representation if that generic
geometry baseline also fails. Result SHA `7c0a15cf...`.

**Generic optical-flow baseline complete (2026-09-25): high recall, rejected specificity.** The
protocol was fixed before scoring: up to 128 generic Shi--Tomasi corners, frame-to-frame pyramidal
Lucas--Kanade flow, a one-pixel forward/backward check, permanent visibility loss, normalized
x/y/vx/vy/visibility, per-state unlabeled PCA45, and the unchanged visual-native monitor. It uses
no masks, identities, simulator state, labels or outcome calibration. It recovers 24/25 physical
alarms (18/19 mixed forks and 5/5 contact-loss), but emits 54/56 alarms and 30 visual-only alarms:
15/15 safe-centered, 9/9 physically quiet unanimous-topple, and 6/6 overshoot. All 56 states remain
initial candidates. Median retained points are 46--51 by cohort and final visibility is high, so
simple track loss does not explain the result. Full-image 2-D motion is too entangled with benign
robot/contact/perspective motion for this response geometry. Do not tune tracker constants on DEV.

**Next:** stop hand-designing on this panel. Specify a task-agnostic dynamics representation using
unlabeled video/action data and generic temporal correspondence, object-centric or 3-D equivariant
structure, velocity preservation, and local action-smoothness training. No topple, success, object
identity or physical-v0 decisions may enter training. Freeze the representation and its monitor
interface before opening a new Panda TEST split; require a recall/specificity Pareto improvement
there before evaluating world-model-predicted futures. Optical-flow result SHA `6742d28c...`.

**Label-free slot representation implemented; pre-monitor gate correctly blocks it (2026-09-25).**
The new training set is separate from monitor DEV: 80 paired generic interactions, 2,720 frames,
three cuboids, visual/camera/action variation, and no outcomes, simulator state, contacts, identities
or monitor decisions. Frozen DINO patch descriptors feed action-conditioned anonymous slots.
Training protocols are hashed before each run and checkpoints are selected only by unlabeled
validation objective. All large reproducible caches/checkpoints are ignored; manifests and reports
are retained in `results/panda_block_push/slot_dynamics/`.

The first appearance model leaves all eight masks uniform. Motion-conditioned v1 becomes sharp by
collapsing every patch into one slot. Balanced v2 prevents monopoly and gets 8.0 effective slots,
0.125 maximum mass and 0.337 motion-map correlation, but temporal geometry moves only 0.000240 RMS,
below the prospectively required 0.002. Thus its balanced masks are still nearly static rather than
entity trajectories. The v2 quality report returns `passed: false`; monitor DEV and fresh TEST were
not opened.

**Revised next representation:** train persistent motion-entity slots with a generic external
correspondence target on the same unlabeled data. Use dense optical flow and, if available, generic
monocular depth/scene flow to supervise per-slot trajectory, visibility and 3-D/SE(3)-like motion;
retain action-conditioned one-step and neighboring-action separation losses. Predeclare and require
the same non-collapse gates plus forward/backward track consistency. This is not a Panda-specific
mask: flow/depth targets are generic visual geometry. Only after that gate passes should the frozen
monitor be evaluated on true visual futures, followed by a newly frozen Panda TEST and finally
world-model futures.

**Status (2026-09-22): steps 1-7 DONE.** Contact-window branches (`eval/jenga_cw_data.py`: 24k contact
+ 7k no-contact points from training episodes only; 3.2x the original count of upright-block rotation
transitions). Arms, 10 paired seeds each, one code version per arm: D0 (the existing `w6_gnn_n5`
seeds), D0+CW, D1+CW, D2+CW, with D1 and D2 on the same short 6-step unrolls so they differ only in the
objective (`eval/jenga_step7_compare.py` -> `results/jenga/step7_compare.json`).

**Answer: the objective, not coverage.** At matched 5% FPR the upright miss rate is 0.48 (D0) -> 0.40
(D0+CW) -> 0.26 (D1+CW) / 0.25 (D2+CW), and robust-blind forks 11 -> 9 -> 3 / 3; other forks barely
move. D0+CW vs D0 is not significant on any metric paired. D2+CW beats D0+CW paired on AUC (9/10
seeds, p = 0.004) and on recall at 3/5/10% FPR (p = 0.008-0.018), with quiet p99 3.2 mm; D1+CW only on
AUC (p = 0.027); D2 vs D1 is not significant. Open issue: the pre-declared dev-set threshold realises
only 1.3-1.6% test FPR for the quieter D1/D2 models, and at that stricter point the CW arms' upright
miss rate is the same (0.43) -- the gains show at matched FPR. Full tables in NOTES.md.

Speed audit (`eval/jenga_w6_speed.py`): torch.compile is unusable (cudagraphs: wrong gradients;
inductor: diverges under Adam). The batched branch loss is bit-identical and 1.15-1.20x faster; it
was not switched in mid-grid, and future D1-style runs use it (`--fast-branch`). 5 concurrent
processes give ~2x throughput with identical numbers.

The older Phase 1-7 training program above is complete historical context. Do not resume Jenga-only
monitor invention. Existing-ensemble disagreement, direct DINO-latent prediction, V-JEPA-style
dynamics, optional F/T and a shared multi-task model are deferred until the ground-truth cross-task
detector and intervention experiments establish that the signal is useful. Do not reintroduce
quiet-score calibration or a shared learned threshold.

## Decision gates

| gate | question | if yes | if no |
|---|---|---|---|
| 1 | Does frozen v0 retain useful physical-branch recall on untouched Jenga TEST? | proceed unchanged | report DEV overfit; redesign only with a new benchmark cycle |
| 2 | Does unchanged v0 detect distinct mechanisms in pushing and insertion? | claim universal regime sensitivity | restrict the claim to demonstrated mechanisms |
| 3 | Does a fixed low-cost alarm response reduce failures at acceptable intervention cost? | claim runtime utility | detector may remain diagnostic only |
| 4 | Do generic rendered representations preserve ground-truth v0 decisions? | proceed to learned prediction | improve representation before dynamics |
| 5 | Do world-model futures preserve rendered-future v0 decisions? | candidate deployable monitor | world-model counterfactual fidelity is inadequate |
| 6 | Does one prospective Layer-2 consequence cost improve the intervention Pareto frontier unchanged across tasks? | add optional consequence policy | retain simple conservative responses |
| 7 | Does one shared multi-task WM preserve useful counterfactual branching? | strongest result | use per-environment WMs with one Layer-1 algorithm |

## Intended paper claim

Do not claim "we build a Jenga safety monitor", "we detect discontinuities", or primarily "we
introduce a better world-model architecture". Target:

> We investigate whether action-conditioned world models can provide a task-agnostic runtime signal
> of proximity to execution-sensitive physical regime boundaries. We separate universal structural
> detection from consequence-aware intervention, introduce a counterfactual evaluation protocol,
> characterize when world models preserve or destroy action sensitivity, and evaluate one frozen
> detector and intervention framework across distinct contact-rich manipulation tasks without using
> task-specific failure labels in the detector.

Central experimental thesis:

```
good prediction  =/=>  faithful counterfactual action sensitivity
faithful counterfactual action sensitivity  can provide a generic runtime safety signal
generic regime sensitivity  =/=>  universal task consequence
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

## 2026-09-25 fresh physical Panda TEST update

A new 56-state panel was selected from continuous random candidates using physical coverage only
and frozen under protocol SHA `bfc1490690f1...` before unchanged ground-truth v0 ran once. It gets
14/20 mixed topple forks (70%), 0/15 stable-centered alarms, 3/5 contact-loss alarms, 0/6 overshoot
alarms and 1/10 unanimous-topple alarms. Fork attrition is 20 -> 18 initial candidates -> 18 refined
boundaries -> 14 commitment majorities -> 14 final alarms. This replaces the optimistic 19/20 DEV
number as the best pushing generalization estimate. Result SHA `dd406d053e17...`.

Twenty exact visual audits are in `results/ground_truth_monitor_alarm_videos/`: ten immutable Jenga
TEST alarms and ten fresh Panda TEST alarms. Each is a selected refined A/B pair, not a nominal
warning overlay. Next ground-truth work should evaluate monitor-guided intervention on this frozen
pushing distribution. Visual representation work remains paused unless explicitly resumed.

## 2026-09-26 oracle-routing upper bound and next gate

Before building another discrete gate, a TRAIN-only necessary-condition test asked whether three
reusable future regimes exist at all. The protocol was frozen at
`results/jenga/oracle_routing_upper_bound_protocol.json` (SHA-256
`2418bca3cf0d8ddef64e0213d4a5087cafd1965b50b3c5600a6fb75d879520cc`). For each nested-width D3
group, the target is the 15-dimensional residual between physical and frozen-D2 log-gap curves:
five action-width levels by action, early-hold and late-hold phases. Three correction prototypes are
fit on 226 TRAIN groups using capacity-balanced hard assignment. Evaluation uses 30 held-out groups
from 21 disjoint TRAIN episodes; the oracle sees the true future residual and chooses its nearest
prototype. It uses no topple, failure, object identity, task-success, DEV or TEST labels.

The preregistered gate passes. Held-out global Huber loss falls 0.8810 -> 0.1499 (**83.0%**) and
adjacent-level local loss falls 0.1476 -> 0.0511 (**65.4%**), above the required 25% and 15%.
All modes transfer, with occupancy 26.7%, 43.3% and 30.0%, above the 5% floor. This establishes that
the missing scale geometry can be represented by a small reusable regime code; the previous
switcher failed to infer that code, rather than because no such code existed.

The result is only an architectural upper bound. The oracle's descriptive persistent recall rises
from 5/24 to 22/24, but smooth recall falls from 3/4 to 0/4. Mean curve fidelity therefore does not
by itself guarantee a selective monitor. The next experiment remains TRAIN-only:

1. Freeze these prototype definitions and train a causal router from information available before
   the counterfactual future (current state/contact graph and proposed action context).
2. Use balanced hard/optimal-transport assignment so executed modes, not average soft
   probabilities, receive capacity.
3. Evaluate on episode-disjoint TRAIN validation: assignment skill or prototype regret, global and
   local curve loss, persistent recall, and smooth specificity. Compare with a majority router and
   the continuous D2 baseline.
4. Do not touch DEV unless the causal router recovers useful oracle gain while preserving smooth
   cases. Do not touch TEST until a later multi-seed prospective DEV gate passes.

Canonical result: `results/jenga/oracle_routing_upper_bound_summary.json`; SHA-256
`1533473b307cb219e0a8670a3d4a8072a0d76d864d4befa6ca6c4aa69eac6981`.

## 2026-09-26 causal router result and revised next experiment

The causal-router protocol was frozen before training at
`results/jenga/causal_router_protocol.json` (SHA-256
`0228e9d15cc316781e36236054dfaad28af7d13cb7f0621984890eff438c695f`). To prevent physical-future
leakage, its 117 inputs contain only the current privileged state, the original eight-step nominal
action and sampled xyz execution-error direction reconstructed from the initial pair. It never reads
physical trajectories, response curves, midpoint choices or deeper bisection endpoints. Ten fixed
MLP seeds train for 400 epochs with no held-out epoch/model selection.

On 30 episode-disjoint held-out TRAIN groups, median results are 56.7% oracle-mode accuracy,
68.8%/70.8% of the oracle's global/local loss gain, and 75.0% persistent recall versus 20.8% for
uncorrected D2. All modes remain active. However, median smooth-case alarm FPR is 75% (range
50-100%; n=4 smooth), above the prospectively fixed <=50% veto, so the overall gate fails. DEV and
TEST remain untouched.

The future-informed oracle itself has 91.7% persistent recall and 0% alarm FPR. All four smooth
groups are correctly assigned to weak correction mode 0; causal seeds usually send them to modes
1/2. This localizes the failure to causal regime identification and smooth abstention. Note that
the earlier `smooth_recall=0/4` statistic meant the oracle predictions entered the unlabelled
0.1-0.25 ratio band; it did not mean they crossed the >0.25 persistent/alarm threshold.

Next remain on TRAIN and revise routing, not the dynamics or monitor:

1. Add an explicit identity/no-correction option so uncertainty need not force a branch-restoring
   correction.
2. Train the router against physical scale-curve reconstruction cost (including the identity
   option), rather than treating nearest-centroid identity as the sole objective.
3. Evaluate selective routing/abstention with the same episode split and frozen persistent/smooth
   thresholds. Require recovery of useful oracle gain, >=50% persistent recall and <=50% smooth
   alarm FPR before considering DEV.
4. Keep task/failure/topple labels out of training and keep TEST closed.

Canonical result: `results/jenga/causal_router_summary.json`; SHA-256
`ff68e2a62458926c6620e84550df460ed4c7a3760019b8f492f675f976eb6332`.

## 2026-09-27 selective cost router: TRAIN gate passed

The follow-up was frozen before fitting at
`results/jenga/selective_cost_router_protocol.json` (SHA-256
`d05dfc5073420e456f02b836dd303f6d2ec46cf9370c17810b3958667fd10cb6`). It preserves the same 117
causal inputs and three residual prototypes but adds option 0: identity/no correction. Rather than
copy a nearest-prototype class, a 117-64-32 MLP predicts all four task-label-free physical curve
reconstruction costs. The cost equally weights fit-normalized global and adjacent-scale Huber
errors; execution uses hard argmin. Ten fixed seeds train for 400 epochs without held-out selection.

All prospectively fixed TRAIN gates pass on the same 30 episode-disjoint held-out groups:

| metric | median | seed range | gate |
|---|---:|---:|---:|
| cost-oracle option accuracy | 60.0% | 56.7-66.7% | >=50% pass |
| global oracle-gain fraction | 74.9% | 72.4-81.0% | >=25% pass |
| local oracle-gain fraction | 73.8% | 70.7-81.6% | >=25% pass |
| persistent recall | 81.3% | 75.0-83.3% | >=50% pass |
| smooth alarm FPR | **37.5%** | 25-50% | <=50% pass |
| identity/weak selection on smooth | 62.5% | 50-75% | >=50% pass |

Median global/local loss is 0.3210/0.07386, versus D2 0.8810/0.14765 and future cost oracle
0.1330/0.04764. The constant fit-best option obtains 95.8% persistent recall but 100% smooth FPR;
the selective router's benefit is conditional abstention, not merely choosing the average best
correction. The oracle itself obtains 91.7% persistent recall and 0% smooth FPR.

This passes the necessary TRAIN gate but has only four smooth held-out groups. Next preregister one
matched DEV experiment before evaluating anything: compare unchanged D2, the prior forced router,
the new selective router, constant-option control and future-informed cost oracle on the existing
DEV scale groups and frozen-v0 stages. Fix model seeds, aggregation and recall/FPR criteria first.
Do not use DEV for epoch/seed selection, and do not touch TEST.

Canonical result: `results/jenga/selective_cost_router_summary.json`; SHA-256
`a9772b73bf2255d37fbbfc4cb0c497bdbab487403db49d6efa0d51c5bd66f0b9`.

## 2026-09-27 selective-router DEV: failed; stop post-hoc curve routing

The DEV protocol was frozen before generating original-D2 routed results at
`results/jenga/selective_router_dev_protocol.json` (SHA-256
`aa1472e51f76f1bd7fd4bd5a45e3b072a0db3419e56bfc35dc6234fbea712a17`). Routing is per refined
pair. Candidate discovery, pair selection, midpoint path, endpoint trajectories, commitment,
persistence and every v0 constant remain frozen. A selected option changes only levels 1-5 of the
early/full gap curve; boundary BIC and the final conjunction are recomputed. Ten fixed TRAIN router
seeds are used without DEV selection.

| arm | topple alarms | quiet alarms | decision |
|---|---:|---:|---|
| D2 identity | 8/23 | 2/89 | baseline |
| constant fit-best correction | 15/23 | 8/89 | nonselective |
| forced three-mode router, median (range) | 15 (14-16) | 4 (3-5) | quiet gate fails |
| selective four-option router, median (range) | **15.5 (14-16)** | **5 (3-5)** | gate fails |
| physical-future curve-cost oracle | 9/23 | 4/89 | descriptive upper bound fails |

The preregistered selective gate was >=10 topple alarms, <=3 quiet alarms, no lower topple median
and no higher quiet median than the forced router. It fails the two quiet conditions. The identity
rescore exactly reproduces the cached D2 baseline, validating the intervention point. Constant and
learned corrections turn many boundaries persistent: selective seeds produce 16-20 topple and
23-29 quiet boundary candidates, versus identity's 9 and 5. Downstream commitment suppresses most,
but not enough quiet cases.

The decisive result is the physical-future cost oracle. It replays all 207 D2-selected nested pairs
in MuJoCo, projects them through each candidate's frozen D2 projector, and chooses identity or one
of the three corrections by true task-label-free early/full scale-curve reconstruction cost. It
uses identity/weak/medium/strong options 75/39/31/62 times and still gets only 9 topples and four
quiet alarms. Therefore better causal classification among these four options cannot meet the
end-to-end gate. The learned router's higher recall reflects stronger correction, not oracle
fidelity. Do not tune thresholds, costs or router weights on DEV, and do not touch TEST.

Next perform one diagnostic trajectory-level oracle ablation on DEV, with no learning or tuning:

1. Keep D2 candidate discovery and pair paths fixed.
2. Compare D2 throughout; physical boundary-scale trajectories with D2 consequence trajectories;
   D2 boundary scales with physical dense consequence trajectories; and physical trajectories for
   both stages.
3. Use the unchanged v0 formulas and report all class/stage counts. This identifies whether the
   next model must repair local scale topology, post-action commitment/persistence, or both.
4. Only then choose between a trajectory-residual expert model, a genuinely stochastic/multimodal
   world model, or abandoning D2 candidate paths. No further prototype-router variants are justified.

Canonical result: `results/jenga/selective_router_dev_summary.json`; SHA-256
`1b00df73098d6a55a5ad45227d4e6fe7add8c2b35fd10115ec81768c022f096f`.

## 2026-09-27 trajectory-level 2x2 oracle ablation: D2 pair selection is wrong

The diagnostic protocol was frozen before dense physical replay at
`results/jenga/trajectory_level_ablation_protocol.json` (SHA-256
`450869610a5cf14bd257a8152bad8e7826aa38f6e1037e981f9662ddef66ff8b`). D2 initial candidates,
three pair identities, five midpoint-side choices and all endpoint actions remain fixed. Every one
of the 207 nested pairs is replayed physically at all six levels and 38 control steps. Boundary
scales and dense consequence trajectories are then independently sourced from D2 or physics, with
the unchanged v0 formulas.

| boundary source | consequence source | topple alarms | quiet alarms |
|---|---|---:|---:|
| D2 | D2 | 8/23 | 2/89 |
| physical | D2 | 9/23 | 1/89 |
| D2 | physical | **0/23** | 1/89 |
| physical | physical | **0/23** | 0/89 |

The D2/D2 arm reproduces every cached baseline decision exactly. Physical boundary geometry alone
is slightly better but nowhere near the reference gate. Physical dense consequences eliminate all
topple alarms because none of the 20 initial topple candidates has a majority of physically
committed outcomes on D2's chosen endpoint pairs. This is not a short-tail problem: 54/60 physical
topple-state pairs are persistent after the action, but only 8/60 show physical commitment.

Pair-level agreement explains the result:

| topple-state pair property (60 pairs) | D2 positive | physical positive | both | agreement |
|---|---:|---:|---:|---:|
| persistent local boundary | 31 | 31 | 15 | 46.7% |
| committed consequence | 49 | 8 | 6 | **25.0%** |
| late persistence | 60 | 54 | 54 | 90.0% |

Across all 207 pairs, boundary agreement is 48.3%, commitment agreement 52.7%, and persistence
agreement 72.5%. D2 hallucinates commitment on 88 pairs overall and on 43/60 topple-state pairs.
Therefore D2's response partition and nearest cross-branch pair selection do not identify the
physical committed fork. Once the wrong pairs are chosen, no gap rescaling, tail correction or
perfect future replay can turn them into the intended physical boundary.

Next run one pair-selection crossover diagnostic, still on DEV and without tuning:

1. Take the physically selected partition/pairs and exact nested paths from the existing physical
   refinement result.
2. Evaluate those paths with physical trajectories and with D2 trajectories using unchanged v0.
3. Compare against the now-measured D2-selected/physical and D2-selected/D2 arms.
4. If physical selection restores the physical ceiling while D2 on those same paths remains weak,
   the next model needs both neighbourhood-topology and trajectory-dynamics supervision. If D2
   succeeds on physical-selected paths, prioritize full-neighbourhood response topology and pair
   selection rather than a new dynamics family.

Do not train another correction router, tune DEV, or touch TEST. The likely training target after
this crossover is a task-label-free set-level loss matching physical pairwise trajectory geometry
and branch partitions over the full probe neighbourhood, not only losses on preselected pairs.

Canonical result: `results/jenga/trajectory_level_ablation_summary.json`; SHA-256
`a796be647234cd1a94f94412cc28105ffaea0caca9f38bfaebf44a96495c39fd`.

## 2026-09-27 pair-selection crossover: topology and dynamics both fail

The crossover protocol was frozen before running D2 on physical-selected paths at
`results/jenga/pair_selection_crossover_protocol.json` (SHA-256
`2ac21243b944e15af458eef01d4d335286e38fdec7826c2d715263f4391e33fc`). The new arm reuses the
physical action-response candidate set, three cross-branch pair identities and five midpoint-side
choices from `action_boundary_refine_dev.json`. It fits each projector on the 64 original D2 probe
trajectories, predicts all physical-selected nested endpoints with frozen D2 seed 1, and applies
unchanged v0 boundary, commitment and persistence formulas.

| selected paths | trajectory source | topple alarms | quiet alarms |
|---|---|---:|---:|
| D2 | D2 | 8/23 | 2/89 |
| D2 | physical | 0/23 | 0/89 |
| physical | **D2** | **4/23** | **5/89** |
| physical | physical | 18/23 | 7/89 |

The preregistered diagnostic rule required D2 on physical paths to recover at least 75% of the
physical reference's topple alarms: >=14/23, with no more than seven quiet alarms. It gets four and
five, so it is not near the physical reference. The reverse crossover already showed that physical
trajectories cannot rescue D2-selected paths. Together these establish independent pair-selection
and trajectory-prediction failures.

Pair evidence on the 69 physically selected topple-state pairs is:

| property | D2 positive | physical positive | shared | agreement |
|---|---:|---:|---:|---:|
| persistent local boundary | 29 | 68 | 28 | **40.6%** |
| committed consequence | 43 | 51 | 33 | **59.4%** |
| late persistence | 68 | 67 | 66 | 95.7% |

D2 misses 40 physical boundary pairs and 18 physical commitment pairs even after selection is
provided. Across all 216 physical-selected pairs, boundary/commitment/persistence agreement is
40.3%/55.6%/92.1%. Late settling is again not the problem. The physical reference itself gives
seven quiet structural alarms, so world-model development should first target faithful agreement
with that task-label-free structural signal, not optimize quiet/topple labels beyond its ceiling.

Next model experiment:

1. Build TRAIN supervision over complete counterfactual probe neighbourhoods, not preselected
   pairs: physical pairwise trajectory-distance matrices, response partitions and nearest
   cross-branch relationships at early and full horizons.
2. Add dense final-pair consequence supervision: task-agnostic whole-trajectory separation curves,
   commitment evidence and late persistence on physically selected nested paths.
3. Train these jointly with ordinary state dynamics. Use a discrete/stochastic trajectory-level
   mode only if needed, with balanced hard assignment; do not return to unidentifiable edge gates
   or post-hoc curve prototypes.
4. First gate on episode-disjoint TRAIN: recover physical partition/pair overlap and physical-v0
   decisions. Then run one prospective DEV comparison against D2. Keep task labels grading-only and
   TEST closed.

Canonical result: `results/jenga/pair_selection_crossover_summary.json`; SHA-256
`f3d8fa7c38d027700d4c64a2aaf09a1727b8ba9770aa48d4a598f352ba3eb630`.
