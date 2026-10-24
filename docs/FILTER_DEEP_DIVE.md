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
