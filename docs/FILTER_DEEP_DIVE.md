# Deep dive: why the filter's fix adds little over the trigger, and the rebuild (Jenga)

User mandate 2026-10-22 night: keep improving the monitor and filter on Jenga until it converges. Constraints:
- the filter is an optimisation (CBF / HJ-inspired) that edits the policy's actions, with no hand-made strategies;
- the world model is off-the-shelf and reliable, working in latent space;
- the monitor uses sensitivity of the dynamics to actions;
- both monitor and filter stay task-agnostic.

## 1. Evidence so far (2026-10-22; all exploratory unless marked)

### 1a. The check spots risky states, but ranks candidate chunks badly
- **Relative-pass test (pre-registered, docs/JENGA_RESIDUAL.md): H1 FAIL.** At 403 decisions where the policy's plan was
  forked and holding was forked too, real one-chunk neighbour risk was:
  - policy 5.1%, hold 6.8%, G-CEM with the absolute rule 7.7%.
- Split by G-CEM's outcome:

  | G-CEM outcome | n | policy | G-CEM |
  |---|---|---|---|
  | certified, chunk essentially unchanged | 84 | 0.1% | 0.1% |
  | certified and actually changed (median 5.5 mm) | 136 | 4.3% | 5.7% |
  | no candidate certified (fell back to best) | 168 | 8.5% | 13.8% |

- **The check is a good detector of risky STATES** (these 403 have 5.1% risk vs 0.3% for decisions that pass outright),
  **but a poor ranker of nearby CHUNKS**: a modified chunk the check certifies is, on average, riskier than the
  policy's own.
- This matches benchmark v2 (dev), real one-chunk risk:
  - policy 2.83%;
  - free mm-offset optimiser 5.73% (CEM) / 7.17% (gradient), worse than doing nothing;
  - G-CEM 0.67%;
  - plain hold 0.48%.
  
  An optimiser with more freedom exploits the model's errors more.
- Within-state ranking test on benchmark v2 (all chunks with real risk): running (eval/rank_check.py).

### 1b. The latent discards most of what a chunk changes
results/jenga/deep_dive/pca_check.json, pca_curve.json:
- The 8-channel PCA of DINOv2 ViT-S tokens (fitted for appearance) keeps:
  - 69% of the static token energy;
  - 28% of the energy of the CHANGE a chunk causes;
  - 24% of that change outside the robot's own region (106k patches, 150 dev decisions).
- PCA fitted on the change itself keeps 38% / 52% / 63% / 75% / 87% / 96% at 8 / 16 / 32 / 64 / 128 / 256 channels.
- Every world model trained since early October lives in this 8-channel space:
  - conv FiLM;
  - ViT predictor (DINO-WM design, vit3);
  - more fork data.

  On the fork-selection set H, all of them plateau at about 70% caught / 10% false (phase1c_selection.json:
  E 69 / 8, conv3 71 / 12, vit3 68 / 12; Phase 1b: more fork data stopped helping).
  
  A full-token model (DINO-WM option 2, docs/PLAN_BRANCHING_WM.md) was planned and never run.

### 1c. World-model size and data
- model_OIL: 2.27M parameters (6 FiLM conv blocks, width 128).
- Trained on PCA8 latents:
  - the counterfactual set (cf_v1: 1700 states x 4 perturbations x 3 plans: the policy's plan + 2 random nudges);
  - plus 200 states of the optimiser's own proposals (oil_v1).
- Earlier ladder: more, and more counterfactual, data monotonically helped the optimiser. b50 real risk:
  threeblock_v3 24.8% -> closed-loop 15.1% -> CF 12.4% -> OIL 11.8%.

### 1d. Off-the-shelf models tried before
- V-JEPA 2-AC zero-shot (ViT-g + AC predictor, DROID-trained) on H: caught 2.6% of forks vs ours 38%.
- Its encoder on REAL futures caught 79%, so its predictor erases the branching.
- V-JEPA 2's encoder localises blocks ~3x worse than DINOv2 (11.8 vs 4.1 mm).
- The DINO-WM checkpoint bundled with the project (data/jenga/world_model.pt) has:
  - a ViT predictor (6 layers, dim 404, 20M parameters);
  - full DINOv2 ViT-S tokens at 224 px (16x16 patches), 3-frame history, proprio + action inputs.
  
  We use its encoder; its predictor was trained on the early single-block Jenga data.

### 1e. Noise model: consistent in simulation
The truth rollouts and the world model's futures use the same execution-noise snippets (subtracted from the commanded
targets), so the speed-independence of the noise is not a sim mismatch. It only matters on a real robot.

### 1f. Decomposition (running): is it the dynamics, the representation or the criterion?
eval/real_fork.py runs the certificate's exact criterion on REAL futures from the simulator:
- same perturbations, scales and split test;
- the monitor's close camera at the same two times;
- encoded (a) as PCA8, (b) as full DINOv2 tokens.

Compared with the world model's scores (rank_check) against real risk, on the same chunks:
- real-PCA8 ranks well, model ranks badly -> the dynamics model is the bottleneck;
- real-full ranks better than real-PCA8 -> the representation is too;
- even real-full ranks badly -> the criterion (fork + disturbance) is wrong for ranking chunks.

### 1g. Decomposition results (2026-10-22 night; results/jenga/deep_dive/)
Within-state concordance: the share of chunk pairs from the same state, differing by >= 2 of 32 futures in real risk,
that a readout orders correctly (ties excluded). Every chunk is scored as chunk + hold tail, the actions the truth
executes.

| set | readout | world model (model_OIL) | real futures, PCA8 | real futures, full tokens |
|---|---|---|---|---|
| dev (150 decisions, 1350 chunks; real: 10 decisions) | h | 0.72 (190/74) | 0.71 | 0.14 (2/12, few pairs) |
| dev | o | 0.80 (476/116) | 0.73 | 0.59 |
| relpass, same chunks (38+ decisions) | h | 0.83 (72/15) | 0.70 (52/22) | 0.68 (58/27) |
| relpass, same chunks | o | 0.67 (82/40) | 0.66 (81/41) | 0.65 (79/43) |

- relpass, all 403 decisions, world model: h concordance 0.80; pooled AUROC 0.76 (h), 0.70 (o). Certified
  (absolute) chunks carry real risk 2.8%, uncertified ones 11.4%.
- Reading 1: the world model is NOT worse than real futures at ranking chunks under this criterion. The criterion
  itself (fork + disturbance from about 32 futures) is a noisy, moderate ranker even on exact futures (0.65-0.75).
- Reading 2 (dev, by arm): the free-offset optimisers' chunks (s2_cem / s2e_cem / s2e_grad) have real risk 5.7-7.6%,
  yet the world model certifies 87-90% of them (hold: 0.5% risk, 92% certified). The model is fooled by large
  off-distribution edits; G's coarse family stays nearer the data.
- Reading 3 (dev, 10 decisions with both): real futures fork much more than imagined ones (hold h 1.0 real vs 1.585
  imagined). The learned model under-imagines branching, the "erased branching" seen before.
- Reading 4 (relpass): of G-CEM's 136 certified-and-changed edits, 124 also certify as chunk + hold (real risk 4.6%,
  about the policy's); the 12 certified only through the 15-step plan carry 16.9%. Choosing among {policy, edit,
  hold} by the chunk + hold certificate gives 5.95% vs policy 5.06%. At these hard states, a ranking accuracy of
  0.7-0.8 is not enough to find a reliably safer chunk.

**Where this leaves the diagnosis.**
1. Not the representation: full tokens don't rank better than PCA8 on real futures.
2. Not mainly a dynamics bug: the model ranks about as well as real futures.
3. The limits are:
   - (a) the criterion's statistical resolution: a fork test on about 32 futures is a coarse risk estimate;
   - (b) model exploitation by unconstrained edits, which needs a trust region / data support;
   - (c) under-imagined branching, which makes the model optimistic.

**Implications for the rebuild.**
- Use continuous readouts with a margin rather than a binary certificate.
- Stay inside the edit support the model was trained on.
- Correct the optimism: calibrate thresholds on the model's own scale, and/or train the model on more branching
  (counterfactual) data.
- The full-token DINO-WM is still worth testing as the off-the-shelf model (step 2). The gate is improvement on
  these ranking numbers, not on token fidelity.

## 2. Plan (each step gated offline on benchmark v2 real risk before any closed loop)
1. Decomposition (1f) + ranking test -> decide where to spend.
2. World model: DINO-WM recipe (off-the-shelf architecture and code, vendored), full DINOv2 tokens, proprio + actions.
   - Initialised from the bundled checkpoint.
   - Fine-tuned on counterfactual Jenga data: the policy's plan + nudged plans, paired perturbations, close camera.
   - Gate: within-state ranking vs real risk better than model_OIL, and fork detection on H no worse.
3. Monitor: sensitivity of the outcome to action perturbations only (no object-pose sampler, so it is task-agnostic).
   - Continuous score: spread / bimodality of the imagined end states under execution-noise draws.
   - Checked on real futures first (oracle) and then with the new world model.
4. Filter: optimisation over the policy's own action chunk, CBF-style.
   - Objective: minimise the distance to the policy's chunk.
   - Constraints: barrier from the continuous sensitivity score, plus the disturbance readout.
   - Robust to model error: pessimistic over an ensemble, and a trust region around the policy's actions.
   - No hold, primitive or backup heuristics.
   - Gates: benchmark v2 dev, then holdout, then closed loop vs G-CEM.

## 3. O-CEM: optimise the continuous disturbance, not "cheapest fork-free" (spec fixed 2026-10-23 ~00:30, before any O-CEM run; eval/opt4.py)
Motivation: section 1g (the min-o selection result came from relpass, which is therefore not used to test O-CEM).
- **Trigger:** unchanged (fork test; benchmark sets contain only triggered decisions).
- **Fix:** CEM over formulation G's box (edits of the policy's own chunk; stopping is inside the family, no hold fallback).
  - Objective: imagined outside-hand disturbance o of the chunk + hold tail (model_OIL search, 32 futures, common
    draws) + LAMBDA * cost (opt2 cost).
  - Final choice among the 4 finals + the unmodified chunk: pessimistic o, the max over OIL + ENS0-3 on 64 fresh futures,
    + LAMBDA * cost.
- **LAMBDA:** chosen on dev only, from {0, 0.001, 0.003, 0.01}, by the lowest dev neighbour risk (applied at the final
  choice; the search uses 0.003).
- **Test sets:**
  - holdout (150; G-CEM 2.25%, hold 1.77%, policy 5.75%);
  - relpass2: a FRESH hard set, every relative-pass decision of head-to-head 2's G-CEM arm (episodes 17000-17299,
    167 decisions, never used offline). G-CEM did nothing there, so its reference is the policy's chunk.
- **H1:** holdout: O-CEM non-inferior to G-CEM, one-sided 95% upper bound of (O-CEM - G-CEM) neighbour risk <= +0.25 pp.
- **H2:** relpass2: O-CEM lower risk than the policy chunk, paired bootstrap one-sided p < 0.05.
- If H1 and H2 pass: pre-register a closed-loop test, O-CEM vs G-CEM, with the trigger extended to relative passes.
  Otherwise report and revise.
- **Descriptive:** risk at every LAMBDA, unchanged share, hold.

### 3a. O-CEM results (benchmark v2 real one-chunk neighbour risk; results/program_b/bench2/report_opt4_*_o1.json)
| set | policy | hold | G-CEM | O-CEM (LAMBDA 0.003) | test |
|---|---|---|---|---|---|
| dev (150) | 2.83% | 0.48% | 0.67% | 0.88% | LAMBDA choice (0 / 0.001 / 0.003 / 0.01: 0.90 / 0.90 / 0.88 / 0.94%) |
| holdout (150) | 5.75% | 1.77% | 2.25% | 1.50% | H1 PASS: O-CEM - G-CEM -0.75 pp, upper -0.10 pp (bound +0.25) |
| relpass2 (167, fresh) | 5.07% | 6.61% | (did nothing = policy) | 3.71% | H2 PASS: -1.37 pp vs policy, p = 0.001, 30 better / 11 worse |

- Reading: minimising the continuous disturbance (a barrier VALUE, not a pass/fail certificate) is the better fix. It
  matches or beats G-CEM where G works and, unlike G-CEM or hold, also lowers risk at states where every option forks.
- Next (pre-registered, results/jenga/ocem_cl_protocol.json): closed loop, 300 fresh episodes (19000-19299).
  Arms: policy, G-CEM, O-CEM with the absolute trigger (cbfoa), and O-CEM with G's trigger (cbfo).

### 3b. O-CEM closed loop (pre-registered; results/jenga/ocl_result.json): PRIMARY FAIL
300 fresh episodes, topples / v4 success:
- policy 55 / 214;
- G-CEM 22 / 245;
- O-CEM with the absolute trigger 16 / 240 (vs G-CEM: 10 vs 16 discordant, p = 0.16; v4 -1.7 pp, lower -6.0);
- O-CEM with G's trigger 17 / 238.

Reading: the offline per-decision gains (holdout -0.75 pp, hard states -1.4 pp) translate into a handful of topples per 300
episodes, below what this test resolves, at a small success cost (the disturbance objective prefers slower / backed-off
edits). The trigger extension to relative passes contributed nothing measurable.

### 4. Full-token DINO-WM gate (step 2): FAIL
The DINO-WM ViT predictor (bundled checkpoint, widened action encoder) was fine-tuned 30k steps on 1000 + 100 states of
full-token counterfactual data (results/program_b/tok_v1, generated on Rorqual). Val one-step MSE 0.197 vs copy-last 0.339.
Within-state ranking on the same chunks as section 1g (token WM vs model_OIL):
- dev: h 0.55 vs 0.72, o 0.78 vs 0.80;
- relpass: h 0.57 vs 0.80, o 0.63 vs 0.70;
- the action-sensitivity gain g: 0.67 (dev), 0.75 (relpass).

Its imagined futures fork more often (24% vs 14% of chunks on dev), consistent with the full-token split test being
noisier. Together with 1g (full real tokens don't rank better than PCA8), the representation and model size are not the
bottleneck on Jenga.

## 5. Where the deep dive converged (2026-10-23)
- **Monitor:** the fork test (sensitivity of imagined outcomes to action perturbations) is validated as a TRIGGER:
  - fork-timed holds 8 topples vs random holds at the same rate 34 (p = 2e-5);
  - it beats LSF-as-published (19 vs 55).
- **Fix:** CBF-style optimisation of the policy's own chunk over a small, data-supported family. G-CEM (certificate) and
  O-CEM (continuous disturbance barrier) are equivalent in closed loop: 22 vs 16 topples, n.s. O-CEM is better offline
  at hard states.
- **Bottleneck:** the residual is the criterion's resolution at states where every option forks: within-state
  ranking about 0.7-0.8 even on exact futures. It is not the world model's capacity or representation (the full-token
  DINO-WM and real tokens rank no better).

### 5a. Resolution check (exploratory; eval/res_check.py, results/jenga/deep_dive/res_{dev,relpass}.json)
Same chunks as section 1g, model_OIL. Each decision is scored once with 256 futures; readouts on nested subsets.
Within-state concordance:

| readout | dev, 32 / 64 / 128 / 256 futures | relpass, 32 / 64 / 128 / 256 futures |
|---|---|---|
| o max over futures (current barrier) | 0.785 / 0.787 / 0.797 / 0.809 | 0.671 / 0.684 / 0.711 / 0.737 |
| o mean | 0.74 / 0.74 / 0.74 / 0.75 | 0.60 / 0.59 / 0.58 / 0.58 |
| o 90th percentile | 0.76 / 0.76 / 0.76 / 0.77 | 0.61 / 0.61 / 0.61 / 0.60 |
| h (fork scale) | 0.61 / 0.68 / 0.69 / 0.63 | 0.72 / 0.76 / 0.75 / 0.75 |

- The exceedance share P(o > O_MAX) is tied on almost all pairs, so it is uninformative at this threshold.
- Reading: the ranking information sits in the extreme TAIL of the imagined disturbance (max > q90 > mean), and only
  the tail readout gains from more futures (+5 pp on relpass from 64 to 256).
- O-CEM's final pick (max over 5 models x 64 futures) already approximates a 320-future tail.

### 5b. Where the residual topples sit (closed-loop test, exploratory)
The trigger fired within the 5 chunks before the topple in:
- G-CEM: 19 of 22 topples;
- O-CEM + absolute trigger: 16 of 16;
- O-CEM + G's trigger: 15 of 17.

Many alarms come only 0-1 chunks before the topple starts. The residual is mostly warning LEAD TIME, not detection and
not fix quality. Next check: a larger barrier margin (fork at 3x / 4x the errors) on the policy-alone episodes
(eval/margin_check.py).

### 5c. Barrier margin (exploratory; eval/margin_check.py, results/jenga/deep_dive/margin_check.json)
Policy-alone episodes of the O-CEM test: 55 topple episodes (last 10 chunks) and every 4th safe episode (61). At every
decision, the policy's plan was tested for forks at 0.5 / 1 / 2 / 3 / 4x the errors; the alarm is fork at <= m, or o > O_MAX.

| alarm | topples warned >= 3 chunks ahead | median lead | alarm rate, safe episodes |
|---|---|---|---|
| fork at <= 2x (current, absolute) | 54% | 3 | 8.3% |
| fork at <= 3x | 72% | 4 | 16.5% |
| fork at <= 4x | 89% | 6 | 22.7% |

A larger margin buys lead time at the cost of more alarms, the classic CBF margin trade-off. Last time the margin was
raised (no fork at 1x -> at 2x), topples fell 73 -> 30 and successes rose. The pre-registered 3x test is
results/jenga/margin3_protocol.json (300 fresh episodes 19500-19799).

### 5d. 3x margin closed loop (pre-registered; results/jenga/m3_result.json): PRIMARY PASS, adopted
300 fresh episodes (19500-19799), Rorqual, all arms of an episode in the same job.

| arm | neighbour topples | v4 | v4b | interventions / episode |
|---|---|---|---|---|
| policy alone | 60 | 213 | 174 | - |
| G-CEM, no fork up to 2x | 21 | 235 | 195 | 3.9 |
| **G-CEM, no fork up to 3x** | **7** | **239** | **193** | 8.8 |
| O-CEM, 3x trigger | 15 | 231 | 200 | 8.3 |

cbfg3m vs cbfg: 5 vs 19 discordant topple episodes, McNemar one-sided p = 0.0033; v4 +1.33 pp (one-sided lower -3.33 pp,
bound -5) -> both primary criteria pass. Topples -67% at no success cost, as the margin table (5c) predicted: the residual
was late warnings. Price: twice the interventions (16.8% vs 8.2% of decisions) and 72 vs 54 s per intervention. Same
trade-off as the last margin step (1x -> 2x: topples 73 -> 30). The trend rule (test running on Fir + Nibi, with a 3x arm
on fresh episodes as replication) is the candidate for the same lead at a lower alarm rate.

### 5e. Red-block falls (exploratory; eval/redfall_check.py, results/jenga/deep_dive/redfall_check.json)
With topples at 7 / 300, red falls (target knocked over or dropped) are the dominant failure: 3x filter 22 before + 46
after extraction; no filter reduces them (policy 15 + 39). The fork test does see them, phase-matched: before a fall with
the gripper closed the plan forks at <= 2x in 36% (before extraction) / 21% (after) of decisions vs 6% of safe closed
decisions. The disturbance readouts do not (AUC 0.35-0.47 within phase). So the trigger fires; the fix does not help.
Per decision on benchmark v2, hold has the lowest one-chunk red risk (4.8% / 4.1% dev / holdout) and G-CEM / O-CEM
(7.0 / 6.9%, 6.2 / 5.7%) are near the policy (7.6 / 7.4%). The fix's certificate checks forks and the outside-hand barrier;
the carried block is inside the hand mask and the 4 arm knobs keep the grasp as is. Candidate (in scope): the gripper in
the formulation (G-grip, running) and a hand-region term in the certificate when the gripper is closed; both need an
offline gate on the red-risk column first.

### 5f. Why the fix misses red falls, and GO-CEM (exploratory analysis; gate pre-registered in results/jenga/go_protocol.json)
Benchmark v2 dev + holdout, one-chunk red-fall risk by gripper phase (truth cache):

| decisions | policy | G-CEM | hold (gripper held) |
|---|---|---|---|
| gripper closed, policy opens it in the chunk (dev 8 / holdout 10) | 46% / 28% | 48% / 22% | 25% / 8% |
| gripper closed, no gripper event (carrying; 27 / 35) | 6.7% / 6.2% | 4.7% / 5.2% | 0.5% / 0.8% |
| gripper open (115 / 105) | 5.2% / 5.8% | 4.6% / 5.0% | 4.3% / 4.8% |

eval/red_readout.py (80 closed-gripper decisions, candidates policy / hold / G's chunk / O-CEM's 5 finals): the imagined
disturbance ranks red risk within a decision (concordance o 0.74, o_hand 0.74, fork distance 0.75, hand spread 0.65), and
the argmin-o candidate has red risk 6.2% vs G-CEM 11.5% (hold 4.1%, oracle 3.6%) with lower neighbour risk; adding the
change cost at LAMBDA 0.003 keeps most of it (8.3% / 5.9% vs 14.6% / 9.0%) and holds in 17% / 16%. So the barrier
already carries the information; what misses it is the OBJECTIVE: G-CEM takes the cheapest certified chunk and never
trades change for a calmer imagined outcome.
GO-CEM = G-CEM's certificate as the constraint, min (o + LAMBDA_O * change) as the objective, hold (gripper held) a
candidate under the same certificate (eval/opt3.py OPT3_OBJ=o). One rule in every phase, no phase switch. Gate on a
fresh set (m3int: the 3x filter's own interventions in the margin test, 298 decisions) vs the chunk it executed.
**GO-CEM offline gate: FAIL** (results/jenga/go_offline_result.json). On the fresh set (298 interventions of the 3x
filter in the margin test) GO's red-fall risk is 3.59% vs 3.67% for the executed G-CEM-3x chunk (upper bound +0.07 pp, needed
< 0); neighbour risk equal (0.37 vs 0.40%). Only 36 of the 298 interventions have the gripper closed, and there even hold
gives 9.0% vs the policy's 11.0%: the closed-gripper gap of benchmark v2 dev/holdout (built from the selection filter's
interventions) is not where the adopted filter acts. So red falls are not mainly a fix problem at intervened decisions;
next question is whether they happen at decisions the trigger lets through.
**Where red falls happen under the adopted filter** (exploratory; replay of the 68 red-fall episodes of the 3x arm in the
margin test, results/jenga/deep_dive/fall_timing_m3.json). Before extraction (22): the filter intervened within the 3
chunks before the fall in 82%. After extraction (46): only 35%, gripper closed at the fall chunk in 70% and opening in the
next chunk in 61%, i.e. mostly at the release/placement, with no intervention. The trigger lets these through; they look
like the pillar task's release failures (decided at a ~1 px offset, no fork spread). A fix-side change (GO) cannot reach
them; a release-aware trigger would need a measurable signal there, which the phase-matched check (5e) finds weak.
**Is the trigger blind at release because of the hold reference?** (exploratory; eval/hold_grip_check.py,
results/jenga/deep_dive/hold_grip.log) The trigger's "do nothing" plan sends gripper command 0.0, which the world model's
base data (0/1 gripper column) reads as "open", so with the gripper closed h_hold could imagine a drop and mask a risky
release. Closed-loop logs (3x arm, margin test): in the 3 chunks before after-extraction falls with the gripper closed the
plan forks below 4x in only 9% of decisions (safe closed decisions 11%), and the relative rule suppressed just 2%. Offline
(80 closed-gripper benchmark v2 decisions): hold with 0.0 vs the gripper's own command forks below 4x in 27.5% vs 23.7%;
the trigger rate is 45% either way (7 decisions flip each way). So no: the hold encoding is not the cause; the plan's
own fork test does not see these releases (a release that drops the block in every future is not a fork, and the outside-
hand barrier masks the carried block). Not pursued further. Where the trigger does fire with the gripper closed, the
policy's red risk is 20% vs 7.8% where it does not, so it still ranks risk there.
**Settling readout** (exploratory; eval/settle_check.py, results/jenga/deep_dive/settle_check.json). Idea: a failure that
happens in every imagined future has no fork spread, but an unstable scene keeps moving after the robot stops. Readout:
latent change over the imagined hold tail (HOLD10 -> HOLD30, robot still), max / mean over 64 futures, whole frame, hand
region, outside. Benchmark v2 dev + holdout, 300 decisions x up to 8 candidates with cached truth.

| phase (decisions) | red fall AUC, settle / mean / hand | policy chunk, any red fall | neighbour AUC, settle_mean |
|---|---|---|---|
| all (300) | 0.25 / 0.33 / 0.27 | 0.33 / 0.34 / 0.36 | 0.75 |
| gripper closed (80) | 0.55 / 0.57 / 0.59 | 0.68 / 0.60 / 0.70 | 0.97 |
| release in the chunk (18) | 0.29 / 0.33 / 0.37 | 0.20 / 0.25 / 0.40 | 0.88 |
| gripper open (220) | 0.13 / 0.20 / 0.14 | 0.29 / 0.29 / 0.34 | 0.67 |

The world model imagines red-fall chunks as CALMER after the stop (AUC < 0.5), as on the pillar task (doomed releases
imagined calmer; ~1 px deciding offset). For neighbour topples the settling readout is informative (0.75 / 0.97), but the
fork trigger already covers those. Reading: red falls (now the dominant residual failure) are below what model_OIL's
latent resolves; neither the fork spread nor imagined motion sees them. This is a scope boundary of the monitor with this
world model, not a filter-design gap; not pursued further within the current scope (WM fixed).

### 5g. The 3x filter's success cost is time-outs, not harm (exploratory; margin-test audit + logs)
Clean non-extraction (no topple, no red fall, block never extracted in 60 chunks): policy 12, G-CEM 24, G-CEM-3x 32,
O-CEM-3x 32 of 300. Of the 3x arm's 32, the paired policy episode succeeded in 18, toppled in 10, stalled in 4: about 18
successes become time-outs while 53 topples are prevented. Stalled episodes get 13 interventions vs 8 in successful ones;
the block barely moves (max lift 3.7 mm, pull 4.7 mm vs 44 / 60 mm). Not a gripper desync (the gripper stays on the
policy's clock while the arm is slowed): grasps reopened within 2 chunks are as common when the closing chunk was a fix
as when it was not (52% vs 52%; policy alone 48%). Per-fix progress is the same in stalled and successful episodes
(median 0.55 vs 0.56). Reading: G's four knobs (slow, pause, scale down, back off) can only remove motion along the plan,
so every certified fix trades progress for safety by construction; time-outs grow with the intervention load. Levers:
fewer unneeded interventions (trend rule, running) and a cheaper filter (torch.compile, results/jenga/compile_result.json:
decisions 100% identical on benchmark v2 dev, 2x faster per scoring call locally, 3.7x on the cluster's MIG slices).

### 5h. Trend (decrease-condition) CBF in closed loop (pre-registered; results/jenga/trend_protocol.json): PRIMARY FAIL; 3x margin replicates in direction

300 fresh episodes 19800-20099, all four arms per episode in one job (Fir + Nibi; amendments 1-3 are infrastructure only).

| arm | topples | red falls before / after extraction | all failures | time-outs (clean non-extraction) | v4 | v4b | interventions / ep | s / intervention |
|---|---|---|---|---|---|---|---|---|
| policy | 57 | 15 / 33 | 105 | 9 | 219 | 186 | - | - |
| G-CEM (cbfg) | 23 | 22 / 35 | 80 | 19 | 236 | 201 | 3.95 | 54 |
| G-CEM-3x (cbfg3m, adopted) | 15 | 16 / 32 | 63 | 35 | 234 | 202 | 8.59 | 72 |
| G-CEM-trend (cbfgtr) | 16 | 23 / 27 | 66 | 31 | 230 | 203 | 8.26 | 89 |

- PRIMARY (cbfgtr vs cbfg): topples 13 vs 20 discordant, p = 0.148; v4 -2.0 pp, lower bound -6.67 pp. Both fail; not adopted.
- The trend rule gives no saving over the 3x margin: same intervention load (8.3 vs 8.6 per episode), slower per intervention, and the same topples (13 vs 14 discordant). Its intended benefit (same lead time with fewer interventions, so fewer time-outs) does not appear in closed loop.
- 3x replication (secondary): topples 15 vs 23 (11 vs 19 discordant, p = 0.10). Same direction as the margin test (5 vs 19), smaller. Pooled over the two tests (post hoc) 16 vs 38, p = 0.002. All failures 63 vs 80 (38 vs 55, p = 0.048, exploratory endpoint). Time-outs again rise with the intervention load (19 -> 35), as in 5g.
- Read: the 3x margin stays the default. The next lever is the time-out cost, not more warning.

### 5i. G-grip (gripper-timing knob) in closed loop (pre-registered; results/jenga/gg_protocol.json): PRIMARY FAIL

300 fresh episodes 20100-20399 on Fir, all arms of an episode in one job.

| arm | topples | red falls before / after extraction | all failures | time-outs | v4 | v4b | interventions / ep |
|---|---|---|---|---|---|---|---|
| policy | 63 | 17 / 30 | 110 | 6 | 214 | 184 | - |
| G-CEM-3x (cbfg3m) | 10 | 17 / 34 | 61 | 34 | 239 | 205 | 8.31 |
| G-CEM-3x + gripper knob (cbfgg3m) | 8 | 18 / 26 | 52 | 39 | 235 | 209 | 7.91 |

- PRIMARY: failures 26 vs 35 discordant, p = 0.153; v4 -1.33 pp, lower bound -5.67 pp. Both fail, so the knob is not kept in the closed-loop method.
- The direction matches the offline result: fewer after-extraction red falls (26 vs 34) and fewer failures overall, but the effect is too small to detect at n = 300. The knob shifts the gripper in 28% of the interventions it optimises.
- Third look at the adopted 3x filter vs the policy: topples 63 -> 10, all failures 110 -> 61 (28 vs 77 discordant). The time-out cost recurs (6 -> 34).

### 5j. How the 3x filter times out: a stall below the execution noise (exploratory; tr + gg 3x arms, 69 time-outs)

- Time-out episodes are not refused all the time: the filter intervenes in about 13 of 60 chunks. The policy never issues a close at all (gripper open throughout, max lift 0-1.6 mm). In the paired policy-alone episode, the same policy closes at about chunk 25 and lifts (45 of the 69 paired episodes succeed alone, 16 fail, 8 time out).
- Example ep 20100 (gg): alone, the hand creeps into the grasp pose (x 488-492 mm, z 471 mm) and closes. Filtered, the hand hovers at x 465-480, z 480-495 for 45 chunks. Near the grasp pose the policy's own chunks move about 3 mm per chunk. The filter slows or shrinks exactly those chunks (back-off median 3.7 mm, not larger than in successful episodes, 2.8 mm). Under 2x execution noise the net progress is about zero, so the policy never gets deep enough to decide to close.
- Real one-chunk risk of the chunks the filter changes (m3int, one intervention per episode): time-out episodes 0.49% neighbour (88% exactly 0) vs 1.51% in successful episodes and 2.29% in failed ones. So the stall happens on low one-chunk risk, at the 3x margin. The margin's benefit is lead time (5c), which one-chunk risk does not show.
- Read: a CBF liveness failure where the policy's goal pose sits inside the 3x barrier's unsafe set. G's knobs only remove motion, so no knob setting reaches the pose. The levers that stay inside the scope are (i) fewer false triggers near the pose (trigger precision: seed-pool check, results/jenga/deep_dive/seed_pool_check.json) and (ii) accepting the trade: the 3x filter converts failures into safe stops at equal success (tr: 63 vs 80 failures, v4 234 vs 236; gg: 61 failures, v4 239).

### 5k. Is the fork trigger noise-limited? No (exploratory; eval/retest_check.py, eval/seed_pool_check.py)

- Benchmark v2 (300 enriched decisions, 4 seeds x 64 futures + 1 x 256): trigger retest kappa 0.54 (the 3x trigger fires on 84% of these states). Averaging h over 4 seeds ranks real one-chunk risk better than one seed (AUC 0.764 vs 0.719) and better than one test with 256 futures (0.728).
- Policy-alone episodes of the O-CEM test (3366 decisions, 55 topples, seeds 0-3): trigger retest agreement 94.7%, kappa 0.80 on ordinary states. At a matched alarm rate on safe episodes, topples warned >= 3 chunks ahead: single seed 73.6% at 12.0%; pooled mean h 67.3% at 11.8%; 2-of-4 vote 80.0% at 12.75%; 1-of-4 85.5% at 16.5%. The trend rule (5h) had 87% at 12.7% on this set and still did not beat G-CEM in closed loop.
- Read: more futures or seeds do not buy earlier warnings at the same alarm rate. Trigger precision near the grasp pose (5j) is not a sampling-noise problem. The held-out seed runs on Fir were cancelled; there is nothing to confirm.

### 5l. Baseline monitor: ensemble disagreement (pre-registered; results/jenga/disagree_protocol.json): fork test wins, PASS

Epistemic uncertainty = across-model variance of the imagined HOLD30 latent (model_OIL + 4 ens_v1 models, 16 perturbed futures at scale 1), thresholded at the fork test's alarm rate on safe episodes.

| set | fork 3x: warned >= 3 ahead | disagreement (best variant) | alarm rate |
|---|---|---|---|
| dev (paired_record_ocl, 55 topples) | 70.9% | 38-44% | 12.0% |
| held-out (paired_record_m3, 60 topples) | 76.7% | 38.3% (primary d_all_plan; others 38-43%) | 12.1% |

Held-out primary: +38.3 pp, lower bound +28.3 pp, 23 vs 0 discordant. Disagreement does fire before topples (57-59 of 60 within 8 chunks), but late, once the scene is already off the training distribution. The fork test sees the branching earlier. This is the epistemic-vs-branching comparison asked for in docs/LITERATURE_REVIEW.md 10.2 item 4.

### 5m. Is the fork needed? Relative SPREAD of the same futures warns earlier (pre-registered held-out PASS; results/jenga/spread_protocol.json)

Scalar baseline from docs/LITERATURE_REVIEW.md 10.2 item 2, computed on the same 64 imagined futures as the fork test: spread = variance along the principal axis (or total variance) of the HOLD10 / HOLD30 latents, for the plan and for hold, at scales 0.5-4. Relative rule: log spread_plan - log spread_hold > tau.

| set, alarm rate | fork test | relative spread (x3, HOLD30; primary) | total variance (x4, HOLD30) |
|---|---|---|---|
| dev, 6.6% (fork 2x) | 47.3% | 65.5% | 69.1% |
| dev, 12.0% (fork 3x) | 70.9% | 89.1% | 89.1% |
| held-out, 7.2% | 45.0% | 63.3% (+18.3, lower +5.0) | 78.3% |
| held-out, 12.1% | 76.7% | **85.0% (+8.3, lower +1.7; 6 vs 1)** | 91.7% |
| held-out, 16.6% | 83.3% | 86.7% | 95.0% |

(Share of topples warned >= 3 chunks ahead.) The bimodality machinery is not what makes the monitor work. What carries the early warning is how much more the plan spreads the imagined outcomes than stopping does, a continuous sensitivity. The absolute spread (no hold reference) is not better than the fork. Consequences:
- A continuous barrier for the CBF: h = -log spread, condition h(plan) >= h(hold) - TAU_S. In words: a plan may raise the spread of imagined outcomes at most e^TAU_S-fold over stopping. TAU_S = 1.0 (label-free, matched to the fork trigger's rate).
- Cheaper: 2 scales instead of 4.
- Closed-loop test pre-registered: results/jenga/scl_protocol.json (600 fresh episodes, cbfgs vs cbfg3m).
- Red falls (exploratory, eval/spread_check.py on deep_dive/redfall_rows decisions): relative spread does not see them either. AUC 1-4 decisions before the fall vs reference decisions: 0.58 before extraction, 0.47 after (tvar 0.56 / 0.38). The world-model resolution limit of 5e/5f stands.
- Spread barrier as a certificate, offline (exploratory; eval/cert_ablation.py, m3int, the 3x filter's 268 interventions): G-CEM with the spread barrier (g3s) vs the fork certificate (g3re). Real one-chunk neighbour risk 0.76 vs 0.40% (+0.36 pp, 90% CI +0.02 to +0.77), red +0.20 pp, progress kept median 0.42 vs 0.56, hold-equivalent 30 vs 8%, 15 vs 70 s. A better alarm is not automatically a better certificate: the relative-spread threshold rejects many of the small corrections the fork certificate accepts (more holds), and among accepted actions it is less selective about real risk. The closed loop (scl) is the pre-registered decision.
- Grasp-approach alarms (exploratory; eval/approach_alarm.py, policy-alone episodes without a topple, the 4 decisions before the policy first closes the gripper): the deployed spread rule fires there as often as the fork 3x trigger (held-out 42.5 vs 44.6%, dev 42.3 vs 47.6%; earlier decisions 15.9 vs 9.2%, 13.8 vs 10.4%). The approach is where both monitors see the most sensitivity, so the spread trigger alone is not expected to cut the 3x filter's time-outs (5j). Liveness has to come from the fix, not the trigger.
- G-steer (exploratory, offline; eval/opt3.py OPT3_STEER=1): G's four knobs plus a 3-D offset of the chunk (|d| <= 6 mm per axis, ramped in like the back-off; cost = the plan change itself). The aim is a fix that can reshape the approach instead of only removing motion (5j). Offline on m3int with the OIL certificate vs g3oil: real risk, progress along the plan's own direction, time-out-episode subset (results/jenga/deep_dive/cert_ablation.json).
  - Result (cert_ablation.json): neighbour risk 0.36 vs 0.40% (-0.03 pp, 90% CI -0.19 to +0.08), red +0.09 pp (CI -0.06 to +0.27); projected progress median 0.50 vs 0.38, time-out-episode decisions +0.30 mean (0.51 vs 0.28 median); holds 1.5 vs 3.7%. The theta offsets are used (|d| often 1-6 mm, mixed directions). Closed loop pre-registered: results/jenga/stl_protocol.json.

### 5n. Spread barrier in closed loop (pre-registered; results/jenga/scl_protocol.json): PRIMARY FAIL

600 fresh episodes 20400-20999 on Fir, all arms of an episode in one job.

| arm | topples | red falls before / after | failures | time-outs | v4 | v4b | interventions / ep | s / intervention |
|---|---|---|---|---|---|---|---|---|
| policy | 106 | 49 / 71 | 208 | 23 | 435 | 369 | - | - |
| G-CEM-3x (cbfg3m, adopted) | 22 | 39 / 68 | 122 | 73 | 470 | 405 | 8.61 | 33 |
| G-CEM spread barrier (cbfgs) | 31 | 50 / 74 | 142 | 112 | 418 | 346 | 12.22 | 26 |

- PRIMARY: topples 28 vs 19 discordant (p = 0.93); v4 -8.7 pp, lower bound -12.3. Both fail.
- Why: the relative spread is a better ALARM (5m) but a worse BARRIER. It fires on 23% of decisions (vs 17%), and as a certificate it holds more and approves riskier moves (offline: 30 vs 8% holds, neighbour risk 0.76 vs 0.40%). More interventions mean more time-outs (5g) and no topple gain.
- The adopted 3x filter, fourth closed-loop test and the largest: topples 106 -> 22 (p = 2e-16), success v4 +5.8 pp (95% CI +1.0 to +10.5), the first significant success gain at 600 episodes. Time-outs 23 -> 73 remain its cost (5j); G-steer (stl) targets them.
- Residual failures of the adopted filter in this test (exploratory; eval/residual_check.py -> deep_dive/residual_scl_cbfg3m.json):
  - Topples (22): in 16 the paired policy-alone episode did not topple (the trajectories diverge after the filter acts; the filter prevents 100, these 16 are its residual on episodes the policy got through). The filter acted within the 8 chunks before 19 of the 22, mostly early (lead >= 3 chunks in 17), so these are not missed warnings: the fixes did not hold. 7 toppling chunks were themselves filtered. At 8 of the 15 unfiltered toppling chunks the logged h was 2 (no split up to 3x; the cap cannot tell a 4x split from none; checked below).
  - Wider-margin check (eval/residual_split.py -> deep_dive/residual_split_scl_cbfg3m.json; 7 scales up to 8x): the 38 unfiltered h = 2 decisions in the 3 chunks before a residual topple vs 2424 unfiltered h = 2 decisions in safe episodes (every 10th). The plan forks where holding does not (the trigger a wider margin would add) at 4x in 23.7% vs 9.3%, at 6x 31.6% vs 15.7%, at 8x 21.1% vs 17.6%. So a 4x margin would reach about a quarter of the residual pre-topple moves, at roughly 2.5x the safe rate, but it would also fire on about 9% more safe decisions. The 6x and 8x margins add mostly noise. Reading: 4x is the only wider margin worth testing, and only if steer makes interventions cheap enough (stl).
  - Time-outs (73): 43 of them succeed with the policy alone; median 12 interventions, 0 holds, fix progress 0.52 per intervention. The stall is many slowed fixes, not holds (as 5j). G-steer (stl, 5o) targeted this; it raised per-fix progress but did not reduce time-outs.


### 5o. G-steer in closed loop (pre-registered; results/jenga/stl_protocol.json): PRIMARY FAIL (neutral)

| Arm (600 eps 21000-21599) | topples | time-outs | failures | v4 | v4b | interventions / ep |
|---|---|---|---|---|---|---|
| policy | 126 | 19 | 211 | 424 | 370 | - |
| 3x filter (cbfg3m) | 20 | 77 | 119 | 475 | 404 | 8.31 |
| 3x filter + steer (cbfgst3m) | 17 | 73 | 116 | 479 | 411 | 8.36 |

- PRIMARY_1 (fewer time-outs): 46 vs 42 discordant, p = 0.38 -> FAIL. PRIMARY_2 (topples non-inferior): -0.5 pp, upper bound +1.0 -> pass.
- The knob does its job (exploratory, run logs): progress per intervention median 0.70 vs 0.55; holds 101 vs 218 and best-effort fallbacks 60 vs 123.
- Why time-outs did not fall: in a time-out the filter acts on only 16-18% of the last 15 chunks (both arms), and in 67 of 73 the red block is never lifted (> 5 mm). The first intervention comes at median chunk 12, and median 11 interventions follow. After that the policy runs mostly unfiltered and still does not finish. So the stall is the policy failing to recover from states the fixes put it in, not slow fixes. That revises 5j / 5n ("many slowed fixes"). Making each fix faster cannot help. Fewer or smaller departures from the policy's own path might, or a policy that recovers better (out of scope for the filter).
- Fifth look at the adopted 3x filter: topples 126 -> 20 (p = 7e-23), v4 +8.5 pp.
- Consequence for the 4x margin (5n): it adds interventions, and interventions are what cause time-outs, so it is not pursued.

## 6. Pipeline (queued, not started)

### 6a. Label-free value function learned in simulation (from SimDist, docs/LITERATURE_REVIEW.md; queued 2026-10-24 at the user's request)
- Target: for a decision state, the exact simulator's own outcome sensitivity, computed with privileged state but without any failure definition: re-simulate the policy's next moves under N small action / perception perturbations and measure how much the final object states split or spread (the oracle version of the fork / relative-spread test, e.g. relative to holding still). Label-free and task-agnostic; privileged only in sim.
- Model: a value head on the PREDICTED latent trajectory (transformer over the imagined latents, as SimDist's sequence-to-sequence heads), trained by regression on that oracle target. Deployed frozen; the real robot only fine-tunes the dynamics (6b).
- Data: broad and perturbed, not only logged episodes (SimDist's ablation: 10% data or expert-only data collapses performance). Policy rollouts + contiguous action-noise windows + earlier policy checkpoints, many thousands of states; oracle targets on the cluster CPUs (bench2.truth-style pool, from a file, never stdin).
- Why it may succeed where HJ idea 3 and HJ v2 stage B failed: those were trained on ~1200 logged episodes with topple-derived targets; this uses far more, more diverse states and a dense, label-free oracle target.
- Use: as a long-horizon barrier term in the CBF / HJ filter (the fixes that "do not hold", section 5n residuals) and as an alarm, gated offline on benchmark v2 real risk and on held-out early warning before any closed loop.
- Prerequisite: 6b (sim-to-sim adaptation) result; cost estimate and a pre-registered gate before building.

### 6b. Dynamics-only adaptation (running): results/jenga/s2s_protocol.json
