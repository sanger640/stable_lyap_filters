# Benchmark due diligence (before building anything on a new benchmark)

Rule (2026-10-19, after SafeLIBERO): first measure (1) how often real outcomes FORK under small perturbations and
(2) whether our fork test detects those real forks; report to the user before building a pipeline.

## SafeLIBERO (closed: negative)
Real contact forks 3.8% of untouched-obstacle decision points (~21% of contact moments); world model caught 0 of 13
real forks. See docs/FINDINGS.md.

## OopsieBench (RoboCasa pick_egg) — check 1 PASSES (fork prevalence)
Code: eval/oopsie_fork_check.py (copy of ~/wksp/oopsie/dd/fork_check.py); results/benchmark_dd/fork_*.json.
Recorded teleop demos, decision points every 20 steps; restore full state (sim state, solver warm start, gripper's
internal command, damage-evaluator velocities), reset egg health, replay the next 20 recorded actions (1 s) 9 times:
exact + 8 with smoothed Gaussian arm noise (0.05 xyz, 0.017 rot). Damage = egg health drop >= 10.
| split | decision points | never | always | any split | forks 2-6/8 | fork share of damage moments | exact vs recorded |
|---|---|---|---|---|---|---|---|
| without live feedback | 222 | 176 | 23 | 23 | 15 | 50% | 97% agree |
| with live feedback | 376 | 366 | 2 | 8 | 4 | 80% | 100% agree |
Noise sensitivity (without-feedback split): fork share of damage moments 28% at 0.02 (11 split / 28 always), 50% at
0.05, 82% at 0.10 (80 split / 18 always). Forks exist at every level; their share grows with the assumed uncertainty,
which is what the barrier h measures (the smallest uncertainty scale at which futures split).
Caveats: human teleop, not a policy. Check 2 (does our fork test
detect these forks?) needs a world model on this environment; not done.

## Physion — check 1 PASSES for Dominoes (fork prevalence)
Code: ~/wksp/physion/dd (tdw_physics noisy Dominoes controller via dd/noisy_run.py; scene RNG fixed per trial,
noise RNG varied). Perception-scale noise on all dynamic objects: position sd 5 mm (x, z), yaw von Mises kappa 1000
(~1.8 deg). 8 scenes x 16 noise draws per config; outcome = Physion label (target ever contacts zone).
v1 used a config whose outcome is always positive (8/8 originals positive) -> 0 forks, invalid sample (my selection).
v2, the four mixed-outcome configs that make up most of the Dominoes test set:
| config | contacts of 16, per scene | forks (2-14) |
|---|---|---|
| 1mid_J025R45_boxroom | 7 16 12 16 11 12 16 0 | 4/8 |
| 2mid_J020R15_d3chairs_o1plants_tdwroom | 16 16 0 12 0 1 15 0 | 1/8 |
| 2mid_J025R30_tdwroom | 0 15 3 0 0 7 15 7 | 3/8 |
| SJ020_d3chairs_o1plants_tdwroom | 2 16 16 0 0 7 12 1 | 3/8 |
Total 11 of 32 scenes (34%) fork under 5 mm noise (plus 4 near-forks at 1 or 15 of 16). Other scenarios not yet checked.

## Physion — check 2 (does our fork test detect real forks?) — spec fixed 2026-10-19 before any data
- Scenario: Dominoes, the four mixed-outcome configs. Training trials from new seeds (no overlap with test scenes).
- World model: same recipe as Jenga (frozen DINOv2-392 -> channel PCA 8 fitted on training frames; deterministic
  residual FiLM transition via eval/fast_train.py), passive (no actions), one model step = 10 frames, from frame 0.
- Perception uncertainty (the monitor's only uncertainty): object-free Gauss-TE latent field fitted on
  z(noisy re-render, frame 0) - z(clean, frame 0) of training scenes, noise = the same 5 mm / ~2 deg as check 1.
- Monitor on each test scene: clean frame 0 -> 64 Gauss-TE starts at s in (0.5, 1, 2, 3) -> roll out to the end ->
  two-mode split test on the final latents (as Jenga); h = log2 of the smallest splitting s, capped at log2 3.
- Test scenes: the 32 labelled in check 1 + 32 more labelled the same way (new seeds), real fork = 2..14 of 16.
- Gate: AUROC(-h, real fork) >= 0.7 AND alarm rate (h < cap) on real forks >= 2x that on non-forks.
- Amendment (22:30, before any check-2 data was encoded or any model trained): 3,000 training steps instead of 7,500
  (smoke test: ~1.5-3 s/step with 20 model steps; 3,000 x 128 samples = ~350 passes over ~1,080 trials). Labels are
  read from a JSON computed in the TDW env (h5py).

### Physion check 2 RESULT (2026-10-20 01:30): GATE FAILS
64 test scenes (17 real forks): AUROC(-h) 0.50; alarm rate on forks 12% vs non-forks 13%. Per config: 0.50 / 0.33 /
0.67 / 0.43 (16 scenes each). results/benchmark_dd/physion_check2_result.json.
Diagnosis (labels used for diagnosis only):
- Imagined spread is NOT the problem here (unlike SafeLIBERO): end-latent spread across noisy versions of the same
  40 scenes, real 8.7 vs imagined 8.4 (median).
- The world model does not simulate the chain reaction: a linear readout of the outcome (target reaches zone) gets
  AUROC 0.84 on REAL end latents of held-out trials but 0.59 on the model's IMAGINED end latents from frame 0
  (start frame alone: 0.51). From frame 100 the imagined end gets 0.72 vs 0.66 for reading the frame at that time;
  from frame 150, 0.81 vs 0.79 -> the model adds little beyond extrapolating what is visible.
- End-latent MSE vs a "nothing moves" baseline: 0.038 vs 0.081 (held-out), 0.044 vs 0.095 (test).
Conclusion: the fork test fails because the world model (1,080 trials, random camera per trial, 200-frame chains) cannot
predict Dominoes outcomes; forks exist (check 1), the monitor's simulator is the gap.

### Physion check 2b (EXPLORATORY, after the failure; set 2026-10-20 01:50 before data): is it the amount of data?
Same everything (configs, camera randomisation, encoder, Gauss-TE, model, 3,000 steps, eval, gate) except 5x the
training trials (+4,800 trials, seeds 20001+, so ~5,900 for training). Primary diagnostic: outcome readout on imagined
end latents (0.59 now; real-latent ceiling 0.84); then the same fork gate. Not a confirmatory test.

### Physion check 2b RESULT (exploratory, 2026-10-20 08:02): more data barely helps
~5,900 training trials (5x), everything else identical. Fork test AUROC 0.46 (gate 0.7), alarms 12% forks vs 19%
non-forks. Outcome readout on imagined end latents: from frame 0 0.65 (was 0.59), from 100 0.70, from 150 0.85;
ceiling on real end latents 0.91. results/benchmark_dd/physion_check2b_*.json.
Conclusion: our world-model recipe (DINOv2 latents + small FiLM transition, random camera per trial) cannot simulate
multi-step Dominoes chains from one frame; data volume is not the main limit. The fork phenomenon exists (check 1);
the learned simulator is the gap. Physion parked pending a much stronger world model; OopsieBench (short-horizon,
gripper-local, Jenga-like) recommended next.
- 2026-10-20: raw 2b frames (~/wksp/physion/dd/train2, 101 GB) deleted with the user's OK; latents
  (results/physion_check2b/z_train.npz) and all results kept; regenerable from seeds 20001+100*chunk.

## OopsieBench check 2a — visibility (spec fixed 2026-10-20 before any images were rendered)
Question: at real egg-damage forks, do the REAL futures' camera images split in our latent space? (If not, no
image-based fork monitor can work and we stop before training a world model.)
- Same decision points and noisy replays as check 1 (both splits, 5% arm noise, 9 replays: exact + 8 noisy), now also
  rendering the fixed robot camera robot0_agentview_right (256 px) at the end of each 1 s window.
- Encode with frozen DINOv2-392 + channel PCA 8 (fitted on these renders' exact-replay frames).
- Score per decision point: the two-mode split test (Jenga's, eval/cem_filter.split_test_gpu) on the 8 noisy end
  latents -> split yes/no and separation (continuous).
- Label: real damage fork (2..6 of 8 noisy replays drop egg health >= 10).
- Gate: AUROC(separation, real fork) >= 0.7 over all decision points, AND split rate on real forks >= 2x that on
  non-forks. Also reported: damage readout from real end latents (linear, held-out demos).
- Pre-run facts (2026-10-20, before rendering): in the RoboCasa backend damage is NOT visualised (colouring exists only
  in OmniGibson); a crushed egg looks unchanged. All 107 damage events (> 5 health) in the 60 demos happen with the egg
  at the gripper (none from drops); 24 coincide with the gripper closing. Prediction recorded now: 2a likely FAILS
  (force-only damage, no visible consequence). Secondary view added before rendering: robot0_eye_in_hand (wrist
  camera, egg large in view), reported alongside; the gate stays on agentview_right.

## SafeManip (RoboCasa365, pi0.5 multitask checkpoint) — check 1 (fork prevalence), spec fixed 2026-10-20 before running
- Policy: pi0.5 `pi05_pretrain_human300/multitask_learning/75000` (robocasa/robocasa365_checkpoints), served with
  SafeManip's serve_policy_wrapper; openpi's RoboCasa loop (3 cameras + state, replan every 5 steps). Smoke: 1/1
  success on PickPlaceCounterToCabinet.
- Tasks: the atomic pick-and-place tasks carrying SafeManip P2 (release stability) and P3 (grasp stability):
  PickPlaceCounterToCabinet, PickPlaceCounterToStove, PickPlaceDrawerToCounter, PickPlaceSinkToCounter,
  PickPlaceToasterToCounter; 4 episodes each (seed 7, pretrain split).
- Branching: every 3rd decision (15 env steps), save the full state (sim state, solver warm start, gripper internal
  command, env timestep); 9 replays of the policy's chunk (first 20 steps) + 10 hold steps: exact + 8 with smoothed
  Gaussian arm noise (0.05 xyz, 0.017 rot, as OopsieBench); restore and continue the nominal episode.
- Outcome per replay (physical stand-ins for P3 / P2, from simulator state; labels for this check only):
  dropped = task object grasped at the branch start, not grasped at the end, and its height fell > 2 cm;
  fell = any task/distractor object's tilt changed > 45 deg, or a non-grasped object fell > 5 cm.
  bad = dropped or fell.
- Exactness check: the exact replay's first 5 steps must match the nominal continuation (object poses).
- Pass (fork-type failures exist in useful numbers): >= 15 decision points with 2-6 of 8 noisy replays bad, AND forks
  >= 25% of decision points where any replay is bad.

### OopsieBench check 2a RESULT (2026-10-20): FAILS, as predicted
598 decision points, 19 real damage forks. agentview_right (gate): AUROC(split separation, fork) 0.55, split rate
37% forks vs 33% non-forks -> fails. Wrist camera (secondary): 0.71, 58% vs 42% (fails the 2x rule). Damage readout
from real end latents 0.82 / 0.83 (correlates such as hand pose), but the futures split visually about as often without
a damage fork (arm noise alone moves the hand visibly). Egg-crush damage is force-based and not a clean visual fork:
OopsieBench pick_egg closed for the image-based fork monitor. results/benchmark_dd/oopsie_check2a_result.json.
- **Amendment 1 (2026-10-20 ~11:15, after 2 of 5 tasks, before any fork result was interpreted):** run stopped as
  INVALID. (a) The "dropped" stand-in counted deliberate placements (all 9 "bad" points on PickPlaceCounterToStove were
  the placement moment in successful episodes). (b) Replays within a branch were not independent: the composite
  controllers' internal goals (arm OSC, mobile base, torso) carried over from the previous replay (exact-replay deviation
  up to 234 mm while holding). Fixes: snapshot/restore every numeric attribute of the composite controller, its part
  controllers and the gripper with the sim state; labels per step: slip_drop = object leaves the gripper while the
  policy's gripper command stays closed (P3); unstable_release = after a commanded release the object tilts > 45 deg or
  still moves > 5 cm/s at the end (P2); knocked = a non-target object tilts > 45 deg or falls > 5 cm. Same tasks, gate.
  Invalid run kept at ~/wksp/safemanip/dd_check1_invalid_labels (not used).

### SafeManip check 1 RESULT, first 20 episodes (2026-10-20 10:53): fork share passes, count fails
Exact replay 0.0 mm at all 640 decision points. pi0.5 success 15/20. Bad moments 8: forks (2-6/8) 5, edge (1 or 7) 1,
always (8/8) 2 -> 75% of bad moments split (gate 25%: passes) but 5 forks < 15 (gate: fails). Mostly unstable releases
(7 of 8), 1 slip-drop, 0 knock-overs; 7/20 episodes had a risky moment, 4/20 an actual event.
results/benchmark_dd/safemanip_check1/.
- **Extension (declared 10:58 before running):** 40 more episodes (seeds 8 and 9, 4 per task each), same tasks,
  definitions and gate, pooled with the first 20.

### SafeManip check 1 RESULT, pooled 60 episodes (2026-10-20 11:29): FAILS on count
2,056 decision points, exact replay 0.0 mm everywhere; pi0.5 success 38/60. Bad moments 24 (slip 11, unstable release
13, knocked 1): forks (2-6/8) 8, edge 6, always 10 -> forks are 33% of bad moments (gate 25%: passes) but 8 < 15
(gate: fails). The physics is fork-type when failures happen, but hazards are rare (~1 in 85 decisions) because pi0.5
is competent on these pick-and-place tasks; a closed-loop test would need hundreds of episodes per arm.
results/benchmark_dd/safemanip_check1{,_ext}/.

## SafeManip check 1c — harder (composite, multi-object) tasks (user decision; spec fixed 2026-10-20 11:40 before running)
Same pi0.5 checkpoint, branching, restore, labels and gate (>= 15 forks with 2-6/8 bad AND forks >= 25% of bad
moments) as check 1. Tasks (P2 + P3 apply): StackBowlsCabinet, LoadDishwasher, SetUpCuttingStation,
StoreLeftoversInBowl, PrepareCoffee, DeliverStraw (all seen in the checkpoint's pretraining) and RecycleBottlesByType
(unseen; bottles). 4 episodes each (seeds 7 and 8, 2 episodes each).
- 1c amendment (11:50, before any 1c result): composite tasks have no single "obj"; labels generalised to all task
  objects (slip: any object leaving the closed gripper and falling > 2 cm; unstable: any object held before a commanded
  release tips > 45 deg or still moves; knocked: any object not held in the window tips/falls). Same thresholds.

### SafeManip check 1c RESULT (2026-10-20 14:10): PASSES
28 episodes (pi0.5 success 4/28), 4,160 decision points, exact replay 0.0 mm everywhere. Bad moments 100 (knocked 62,
unstable release 22, slip 17): forks (2-6/8) 32 (gate >= 15 ok), 32% of bad moments (56% with edge; gate 25% ok),
always 44. Per task forks/bad: StackBowlsCabinet 13/23, RecycleBottlesByType 9/30, LoadDishwasher 4/10,
StoreLeftoversInBowl 4/25, PrepareCoffee 1/8, SetUpCuttingStation 1/4, DeliverStraw 0/0.
Caveat: low task success on composite tasks, so part of the hazards come from clumsy behaviour.
results/benchmark_dd/safemanip_check1c/.

## SafeManip check 2a — visibility (spec fixed 2026-10-20 14:15 before running)
Fresh rollouts (pi0.5 is stochastic; 1c states were not saved) of the four highest-hazard tasks: StackBowlsCabinet,
RecycleBottlesByType, LoadDishwasher, StoreLeftoversInBowl; 4 episodes each (seeds 11, 12; 2 each). Same branching,
restore and labels as 1c, plus the end frame of every replay from robot0_agentview_left (pi0.5's main camera; gate)
and robot0_eye_in_hand (secondary), 256 px. Frames are kept for all decision points with any bad replay and a random
20% of the others. Score and gate as OopsieBench 2a: DINOv2-392 + channel PCA 8 (fitted on exact-replay frames),
two-mode split test on the 8 noisy end latents; AUROC(separation, fork 2-6/8) >= 0.7 AND split rate on forks >= 2x
non-forks.

### SafeManip check 2a RESULT (2026-10-20 16:10): FAILS (wrist camera close)
553 saved decision points (78 with any bad replay, 15 real forks). agentview_left (gate): AUROC 0.62, split rate 47% on
forks vs 35% on non-forks (1.3x) -> fails. eye_in_hand (secondary): AUROC 0.75, 67% vs 37% (1.8x; fails the 2x rule).
As on OopsieBench, about a third of non-fork moments split visually: the whole-image split test reacts to the arm's own
trajectory variation under noise. Candidate fix (exploratory, not yet run): mask the robot's own pixels (proprioceptive
robot segmentation) before the split test. results/benchmark_dd/safemanip_check2a_result.json.

## SafeManip check 2a-masked (exploratory after the 2a failure; agreed with the user; spec fixed 2026-10-20 16:10)
Hypothesis: the whole-image split test is swamped by the robot's own trajectory variation; masking the robot's pixels
should remove false splits while keeping object outcomes (tips, knocks) visible.
- Fresh rollouts, same tasks/branching/labels as 2a (seeds 31, 32; 2 episodes per task each).
- For every replay end frame, three views at 256 px, each with the simulator's segmentation: (1) agentview_left (policy
  camera; GATED), (2) a dedicated fixed monitor camera, one rule for all tasks: placed above the robot (base xy +
  25% towards the gripper's start position, 0.6 m above it), aimed at the gripper's start position (exploratory),
  (3) eye_in_hand (comparison).
- Robot mask = pixels whose geom belongs to the robot (robot0_ / gripper0_ / mobilebase0_ bodies); latent patches
  (28x28) with > 30% robot pixels in any of the 8 replays are excluded before the split test ("ideal mask", an upper
  bound; a proprioceptive mask follows only if this passes).
- Score and gate as 2a: AUROC(separation, fork 2-6/8) >= 0.7 AND split rate on forks >= 2x non-forks, gated on
  view (1). Unmasked scores reported alongside.

### SafeManip check 2a-masked RESULT (2026-10-20 18:10): FAILS on every view
586 decision points with frames (65 with any bad replay, 20 real forks). AUROC(split separation, fork), unmasked ->
robot-masked: agentview_left (gate) 0.53 -> 0.60, split rate 50% vs 41% (1.2x); dedicated monitor camera 0.57 -> 0.68,
55% vs 34% (1.6x); wrist 0.72 -> 0.72 (1.2x). Masking helps a little, but 34-42% of non-fork moments still split, so the
arm was not the main source of false splits; likely candidates: held objects moving with the noisy hand (visible, not
part of the robot mask) and the injected execution noise (5% of the action range, cm-scale here vs Jenga's mm-scale
measured tracking error). Per the agreed timebox this closes the existing-benchmark search.
results/benchmark_dd/safemanip_check2a_masked_result.json.

## SafeManip check 2b-powered (spec fixed 2026-10-20 18:25, before running; agreed with the user after an audit)
Audit finding: checks 2a / 2a-masked used only 8 real futures per decision point; the two-mode split test is weak at
n = 8 (non-fork split rates of 34-42% on every benchmark and view are consistent with that), whereas the Jenga monitor
was validated with 64 futures. So the 2a negatives are under-powered rather than demonstrated.
Re-test: 32 noisy replays (+ the exact one) per decision point; tasks StackBowlsCabinet and RecycleBottlesByType (most
fork-rich in 1c), 4 episodes each (seeds 41, 410, 42, 420; 1 episode each, run in parallel - amended at launch, same count), branching every 3rd decision; same restore, labels
and robot masks. Gate on the ROBOT-MASKED MONITOR CAMERA (best view in 2a-masked; these are fresh episodes, so this
confirms that choice): AUROC(split separation, fork) >= 0.7 AND split rate on forks >= 2x non-forks, with fork = 2..30
of the 32 noisy replays bad (bad share strictly between ~5% and ~95%). agentview_left and wrist reported alongside.

### SafeManip check 2b-powered RESULT (2026-10-20 22:52): pre-registered gate (masked monitor camera) FAILS; two other views pass
279 decision points (44 with any bad replay, 29 real forks = 2..30 of 32 replays bad). With 32 replays the false-split
rate on non-forks fell from 34-42% (n = 8) to 10-21%, confirming the audit (n = 8 was under-powered).
| view (robot-masked) | AUROC (sep vs fork) | split rate forks / non-forks | gate (>= 0.7 and >= 2x) |
|---|---|---|---|
| monitor camera (pre-registered gate) | 0.65 | 34% / 14% (2.4x) | FAIL (AUROC) |
| agentview_left | 0.72 | 55% / 17% (3.2x) | would pass |
| eye_in_hand (wrist) | 0.83 (unmasked 0.83) | 52% / 20% (2.5x) | would pass |
Per protocol the result is a FAIL for the gated view; switching the view after seeing results is not allowed. The gated
view was chosen from the n = 8 run, which this audit showed was too noisy to rank views. The wrist view was the best in
both independent sets (0.72 at n = 8, 0.83 here) and is one of the policy's own cameras.
Next (spec below, fixed before running): a confirmation run on fresh episodes with the wrist view as the gate.

## SafeManip check 2c-confirm (spec fixed 2026-10-20 22:55, before running)
Fresh episodes: seeds 51, 510, 52, 520 (one episode each), tasks StackBowlsCabinet and RecycleBottlesByType (8 episodes,
as 2b), 32 noisy replays + the exact one, branching as in 2b. GATE: robot-masked eye_in_hand (wrist) view, AUROC(split
separation, fork) >= 0.7 AND split rate on forks >= 2x non-forks. Other views reported, not gated. Pass -> SafeManip is
the second benchmark: next is check 2b-WM (does the world model's imagined fork test detect these forks). Fail -> build
our own task.

### SafeManip check 2c-confirm RESULT (2026-10-21 01:10): PASSES
Fresh episodes (seeds 51, 510, 52, 520), 279 decision points, 43 with any bad replay, 28 real forks. GATE robot-masked
wrist view: AUROC 0.742, split rate 54% on forks vs 20% on non-forks (2.6x) -> PASS. Also: masked monitor camera 0.740
(46% vs 12%, 3.8x; would pass), masked agentview_left 0.687 (46% vs 17%). Unmasked wrist 0.746. Across the two
32-replay sets (2b, 2c; 57 forks) the wrist view gives 0.83 and 0.74.
=> SafeManip is the second benchmark: real forks are common in its multi-object tasks AND they are visible as a split of
real futures in the latent of the policy's own wrist camera. Next (per spec): check 2b-WM - does our world model's
IMAGINED fork test detect these forks (needs a SafeManip world model on wrist latents + proprioception).
results/benchmark_dd/safemanip_check2c_confirm_result.json.

## SafeManip check 2b-WM: does the IMAGINED fork test detect real forks? (spec fixed 2026-10-21 ~09:20, before any data)
User decision (2026-10-21): try BOTH views and see which is best: the fixed monitor camera and pi0.5's wrist camera,
each with robot pixels masked.
- Recipe: as Jenga / the SafeLIBERO world model (libero/wm_libero.py: frozen DINOv2-392 tokens -> channel PCA 8 ->
  28x28x8 latent; deterministic residual FiLM transition, eval/fast_train.py unchanged; proprioception as input; robot-
  pixel patches (> 30% robot) zeroed in every latent, the same mask rule as checks 2a-2c). One model per view, same data,
  same hyper-parameters. 1 model step = 2 control steps (as Jenga); horizon = the policy's chunk + hold (as the checks).
- Training data (label-free): pi0.5 episodes on StackBowlsCabinet + RecycleBottlesByType with branched noisy futures
  from regular decision points (exactly the 2b/2c branching, 32 noisy + exact replay, 5% arm noise), now saving the start
  frame, proprioception and executed actions of every replay at every model step. Seeds 61-...; amount set by a small
  run first (target ~ the Jenga closedloop data scale), all on Fir.
- Evaluation set (held out, never used for training or tuning): fresh episodes, seeds 71, 710, 72, 720 (8 episodes as in
  2b/2c), same branching; per-replay outcomes (slip / unstable release / knocked) saved. Fork = 2..30 of 32 noisy replays
  bad (as 2b/2c).
- Imagined fork test: as Jenga (fork score h from the two-mode split test on imagined futures under the Gauss-TE
  perception-noise model fitted on this data, hold-10 / hold-30 endings, scales 0.5-3).
- GATE per view: AUROC(h vs real fork) >= 0.7 AND imagined split rate on forks >= 2x non-forks. Both views reported.
  "Best view" = the one passing with the higher AUROC; it is used for the closed-loop SafeManip filter test (spec to be
  written after this result). If neither passes: diagnose as for Physion (outcome readout from imagined vs real endings).
- **Amendment 1 (2026-10-21 ~09:50, before any 2b-WM data):** (a) imagined futures use the SAME execution-noise model
  that generated the real forks (smoothed Gaussian on the 6 arm dims, 5% x scale s, s in {0.5, 1, 2, 3}; as
  libero/wm_libero.score_plans), not Gauss-TE; (b) robot pixels: the world model's TARGET latents have robot patches
  (> 30% robot pixels in that frame) zeroed, so it learns to predict the masked scene from proprioception + actions; the
  imagined split test then uses all patches (no future mask needed at test time). On a real robot the masks come from the
  kinematic model + camera calibration; here from MuJoCo segmentation (same information). (c) Endings: end of the chunk
  (20 control steps = model step 10) and end of the hold (30 = model step 15), as hold-10 / hold-30 on Jenga.
  (d) Data: train mode saves exact + 4 noisy replays per point (rendering cost); eval mode exact + 32.
- Data batch 1 (2026-10-21 09:52, Fir job 66554611, 2 x 3g.40gb slices, 8 clients each): held-out eval set complete
  (seeds 71, 710, 72, 720 x 2 tasks, exact + 32) + training seeds 61-64 x 2 tasks (exact + 4). Fir smoke: ~32 s per
  training point, ~70 s per eval point (OSMesa). Training data will be scaled up after batch 1 is checked.
- Data batch 2 (2026-10-21 ~10:05, Fir job 66559649, 3 slices x 8 clients): training seeds 65-69 and 81-87 x 2 tasks
  (24 episodes; no overlap with the eval seeds 71, 710, 72, 720). Training data total: 32 episodes.
- Note: RecycleBottlesByType seed 67 starts in a layout that already meets the task's success check, so its episode
  ends after step 1 (1 training point). A property of that scene, kept as is. Measured on Fir: ~19 s per training point.

### SafeManip check 2b-WM RESULT (2026-10-21 14:48): GATE FAILS on both views
Held-out eval set: 8 episodes (seeds 71, 710, 72, 720 x 2 tasks), 1,193 decision points, 16 real forks (2..30 of 32
replays bad; 1.3%, vs 29 / 279 in 2b-powered). Models: 7,500 steps on 4,903 training points (32 episodes); final loss
monitor 0.016, wrist 0.078.

| view | AUROC(-h, fork) | imagined split @ s=1, forks / non-forks | gate |
|---|---|---|---|
| fixed monitor camera (masked) | 0.45 | 0% / 5% | FAIL |
| wrist (masked) | 0.50 | 0% / 1% | FAIL |

Diagnosis (eval/safemanip_wm_diag.py, imagining with the real replays' own executed actions; results/safemanip/wm_*/diag_result.json):
- **The model does not generalise to new kitchens.** End-of-hold latent error vs the copy baseline (start latent as
  the ending): training points 0.020 vs 0.078 (monitor; 75% better), held-out points 0.052 vs 0.069 (25% better).
  Wrist: 0.096 vs 0.347 on training, 0.211 vs 0.339 held out. Every seed is a new kitchen layout; 32 episodes is too few.
- **Imagined spread is half the real spread** (monitor 0.020 vs 0.039, wrist 0.075 vs 0.180). It follows how much the
  arm moves (correlation 0.83 / 0.85 with the real spread), but not forks (AUROC 0.48 / 0.51).
- **Per replay, the model does not tell bad endings from good ones:** distance from the exact replay's ending vs replay
  bad, AUROC 0.53 (monitor) / 0.52 (wrist) imagined, 0.61 / 0.61 real.
- **Even REAL endings are a weak fork signal on this set:** split test on the 32 real endings, forks vs non-forks 69% vs
  38% (monitor, 1.8x) and 38% vs 17% (wrist, 2.2x); real spread AUROC 0.64 / 0.62. So a perfect world model in this
  latent would still be at or below the gate here (2c: 0.74 with a different, smaller eval set).
Conclusion: two problems stacked. (1) world-model generalisation (data-limited, as Physion); (2) at SafeManip's hazard
rate (1 in ~75 decisions) and with the robot masked, the forks are only weakly visible in the frozen DINOv2 latent at
the endings. Scaling data could fix (1) but not (2).

## Next second benchmark (user decision 2026-10-21 ~16:00): real robot + a standard sim task with frequent, visible forks, in parallel
Lessons from SafeManip / Physion / OopsieBench: hazards must be FREQUENT (SafeManip: 1 in ~75 decisions), VISIBLE in
the scene (not slips inside the gripper), and in a FIXED scene a world model can learn from tens of episodes (SafeManip:
a new kitchen per seed). Real-robot plan: docs/REAL_ROBOT_PLAN.md.

## robomimic ToolHang — check 1 + 2 (fork prevalence + visibility), spec fixed 2026-10-21 ~16:00, before any branching
Why ToolHang: the standard robomimic benchmark's hardest task (insert a frame into a stand, hang a tool on it). Its
failures are physical and visible (frame tips or falls out of the stand, tool slides off and falls, stand knocked over),
the scene is fixed (only object poses are randomised), and a fixed third-person camera sees everything.
- Policy: the robomimic v0.1 zoo checkpoints (BC-RNN image, robosuite offline_study) do NOT run on robosuite 1.5
  (ported config / weights / controller frame: 0/10 success; open-loop demo replay also drifts). So we train the policy
  ourselves with robomimic's own Diffusion Policy template (eval/../robomimic_bench/configs_toolhang_dp.json: low-dim
  obs, official ToolHang PH v1.5 dataset (200 demos), DDIM 10 steps, obs horizon 2, action chunk 8, 600 epochs).
  Checkpoint rule (fixed now): the LAST saved checkpoint whose training-rollout success (20 episodes) is in [50%, 90%];
  if none is below 90%, the earliest checkpoint >= 50%. Re-measured on 50 fresh episodes before branching.
- Branching (as SafeManip 2b-powered): every 2nd decision (16 env steps); save full sim state; 33 replays of the
  policy's chunk (8 steps) + 20 hold steps (gripper command kept): exact + 32 with the same smoothed Gaussian arm noise
  (0.05 xyz, 0.017 rot); restore and continue. 8 episodes (seeds 1-8).
- Outcome per replay (simulator state, labels for the checks only; same rules as SafeManip):
  dropped = object grasped at branch start, not grasped at end, height fell > 2 cm;
  fell = any object's (stand, frame, tool) tilt changed > 45 deg, or a non-grasped object fell > 5 cm. bad = either.
- Check 1 PASS: >= 15 decision points with 2..30 of 32 noisy replays bad, AND forks >= 25% of points where any replay is
  bad, AND forks >= 5% of all decision points (new: the SafeManip lesson).
- Check 2 (same run, gate view fixed now): robot-masked AGENTVIEW camera (fixed, third person; patches > 30% robot
  pixels dropped), DINOv2-392 features of the 32 real endings, two-mode split test: AUROC(split separation, fork) >= 0.7
  AND split rate on forks >= 2x non-forks. Not gated, reported: sideview (masked), wrist (masked).
- Pass both -> world-model check (2b-WM recipe) on ToolHang; fail -> stop ToolHang.
- Note (2026-10-21 18:50, fork-check run already in progress, gate unaffected): in ToolHang robosuite places `sideview`
  at the same orientation as `agentview`, 5 cm lower ((0.484, 0.251, 1.214) vs 1.264 m). So the non-gated "sideview"
  result is not a second viewpoint; it is reported as a near-duplicate of the gate view. (Dashboard videos use
  `frontview` as the second panel.)

## robomimic ToolHang RESULT (2026-10-21 19:05; results/benchmark_dd/toolhang_check1_result.json): check 1 FAILS
Checkpoint by the rule: epoch 600 (20-episode rollout 65%; re-measured 31/50 = 62% on fresh seeds 101-150). Seeds 1-8
(5 successes, 3 timeouts), 289 decision points branched (exact + 32 noisy replays of the chunk + 20 hold steps).
Forks (2..30 of 32 bad): **1** (bar: >= 15). Points with any bad replay: 7, and 6 of them are all-or-nearly-all bad
(32, 32, 32, 32, 32, 31 of 32; the one fork is 19/32) - deterministic outcomes of the chunk, not splits. Labels do see
ToolHang's motion (the tool swinging after release at seed 1 t528 is flagged in all 32), so the null is not a label
blind spot; the 3 failed episodes are timeouts where noise does not change the outcome. Exact-replay restore deviates
<= 0.6 mm (26 of 289 points above 1e-6 m), negligible against the noise. Check 2 is moot (no forks to score).
Verdict: ToolHang is not a fork benchmark for this policy. Per the plan agreed 2026-10-21, sim-benchmark hunting stops
here: Jenga is the sim evidence, the real robot is the second domain; ToolHang is reported as a scope limit
(failures are precision/timeout, not noise-sensitive).
(Worker note: seeds were run by 3 parallel workers; seed 6 ran twice with identical results, confirming determinism.)

## ManiSkill3 quick fork check: StackPyramid + LiftPegUpright (spec fixed 2026-10-21 19:20, before any branching)
User (19:15): two ManiSkill3 tasks, StackPyramid-v1 and LiftPegUpright-v1 (PickClutterYCB has no demos/solver); first
find out FAST whether they fork, before training any policy, then videos + the world-model camera.
- Policy for this check = the shipped expert demos (StackPyramid: motion planner, pd_joint_pos; LiftPegUpright: PPO,
  pd_ee_delta_pose); branch along the demo trajectory (restored from its saved env_states) at every 8th step: exact + 32
  noisy replays of the next 8 demo actions + 20 hold steps, on the GPU simulator (33 parallel envs). Demos 0-15
  (StackPyramid) / 0-39 (LiftPegUpright).
- Noise: 3-tap smoothed Gaussian, ~2.5 mm per step as before: joint targets sigma 0.004 rad (StackPyramid);
  normalised ee-delta actions sigma 0.025 xyz / 0.0083 rot (LiftPegUpright; 1 unit = 0.1 m).
- Bad replay: any object's final pose differs from the exact replay's by > 2 cm or > 20 deg (task-agnostic "outcome
  changed"); the env's success flag at the end is also recorded. Fork: 2..30 of 32 bad.
- Worth-it bar (same as check 1): >= 15 forks per task AND forks >= 25% of points with any bad replay AND >= 5% of all
  points. A task that passes goes on to videos + camera choice + its own policy; one that fails is dropped.
### RESULT (2026-10-21 19:40; results/benchmark_dd/maniskill/; 0.65-2.2 s per decision point on the GPU sim)
| task | points | forks (2..30 of 32 bad; bar 15) | any bad | env success flag split (2..30 of 32) | verdict |
|---|---|---|---|---|---|
| LiftPegUpright (PPO demos 0-39) | 192 | **73** (in 30 of 40 demos) | 86 | **64** | **PASS** |
| StackPyramid (motion-planner demos 0-15) | 373 | 10 | 18 | 0 | FAIL (dropped) |
LiftPegUpright forks are real outcome splits, not pose jitter: at 64 points the task's own success flag splits under
2.5 mm noise (the peg either ends upright or falls back). Caveat: this is the expert (PPO) policy; the deployed
policy (a chunked Diffusion Policy) must be re-checked once trained. Next: videos + world-model camera choice.
### Correction + strict count (2026-10-21 19:55, after viewing the fork clip; not part of the pre-registered label)
The pre-registered label (> 2 cm / > 20 deg from the exact replay) and the env success flag (within 4.6 deg of vertical
AND 5 mm of table height) both also count smooth tilt SPREAD (e.g. all replays between 1 and 10 deg). Strict physical
count from the per-replay final peg tilt (re-run, same seeds, identical 73 loose forks): two-mode moments (>= 2 replays
standing < 20 deg AND >= 2 flat > 60 deg) = **20** of 192 (10.4%); moments where any replay falls flat = 31, so forks
are 65% of them; all-or-nearly-all flat = 1. Still passes all three bars. We quote the strict 20, not the loose 73.
World-model camera (chosen from 6 candidates, frames inspected): fixed side camera at (0, -0.5, 0.15) m looking at
(0, 0, 0.05), fov 1.0 rad: peg tilt visible in profile, robot only in the top of the frame. ManiSkill's default
base_camera looks down (tilt foreshortened, gripper occludes the peg). Videos: eval/maniskill_video.py.

## LiftPegUpright policies (spec fixed 2026-10-21 19:40, before training; user: "do diff policy then but i want to also test on ppo")
- Main policy: ManiSkill's state-based Diffusion Policy baseline (examples/baselines/diffusion_policy/train.py, unchanged
  defaults: obs horizon 2, action chunk 8, prediction 16), trained on 100 of the PPO demos (state obs, pd_ee_delta_pose,
  physx_cuda; replayed from the shipped demos, 927 kept), 30k iterations, eval every 5k on 100 GPU episodes,
  max_episode_steps 100. Checkpoint rule (as ToolHang): the LAST checkpoint with eval success in [50%, 90%]; if none is
  below 90%, the earliest >= 50%; if none reaches 50%, train on more demos (all 927) and repeat once.
- Second policy: the shipped PPO checkpoint (ppo_pd_ee_delta_pose_ckpt.pt), per-step (not chunked). How the filter
  wraps a per-step policy is decided before any PPO filter episode runs.
- Next: fork check on the DP's OWN closed-loop rollouts (same branching as the quick check, strict stand-vs-flat count,
  same three bars) before any world-model work.
- Amendment (2026-10-21 20:05, before any DP checkpoint was evaluated): episodes are the task's default 50 steps and
  success = peg upright AT THE END (ManiSkill success_at_end); the demos run to 50 steps and keep the peg upright, and
  the quick-check forks were in that post-pivot "leave it standing" phase. (A first PPO run at 100 steps went out of
  distribution after step 50 and knocked every peg over; discarded.) The DP checkpoint rule uses success_at_end from
  eval/maniskill_policies.py rollout() on 100 episodes (seed 0).
- PPO (deterministic mean), 50 steps: success_at_end 38% (200 episodes), upright at some step 100%.
  **PPO closed-loop fork check PASSES** (fc_policy/ppo_result.json; 32 episodes, 224 points): 24 strict forks (bar 15),
  78 points with any flat replay (forks 31%, bar 25%), 10.7% of points (bar 5%); 42 points all/nearly all flat
  (deterministic falls also common).
- DP on 100 PPO demos FAILED (training's own eval and our rollout both 0-1% at 5k/10k/15k; RL demos are jerky and DP
  executes 8-step chunks open-loop). Per the rule: retraining on all 927 PPO demos (dp_liftpeg_all, 50-step episodes).
  Exploratory in parallel (decided 20:15, before any result): motion-planner demos (ManiSkill's own DP baselines use
  these; the LiftPegUpright solver grasps, lifts, rotates the peg upright in the air, lowers and OPENS the gripper -
  the release is the stand-or-fall moment). 1000 demos, converted to state obs + pd_ee_delta_pose, DP same defaults,
  episodes 250 steps. (mplib segfaulted under numpy 2.2; numpy pinned < 2 in this venv.) If both DPs qualify, the
  one with the higher success_at_end in [50%, 90%] is the main policy; reported either way.
- DP checkpoints (success_at_end / once, 100 episodes): PPO-demo DP on all 927 demos, 50 steps: <= 5% at every
  checkpoint (fails, as with 100 demos). Planner-demo DP, 250 steps: 10k 57%, 15k 69%, 20k 73%, **25k 79%** -> by the
  rule the main DP is dp_liftpeg_mp 25000.pt.
- **Planner-demo DP closed-loop fork check FAILS** (results/benchmark_dd/maniskill/dp_policy_fork_result.json; 32 episodes,
  1024 points): 1 strict fork. Its 6 failures of 32 are deterministic (never lifts; rotates to ~45 deg and stalls);
  when it succeeds it sets the peg fully upright (0 deg) and releases, which is robust to noise. PPO instead holds the
  peg balanced against the table, which forks (24). Fork prevalence on this task is a property of the policy's style.
  Videos: LiftPeg_DP_rollouts.mp4, LiftPeg_PPO_rollouts.mp4. (The planner DP lifts the peg ~30 cm, partly out of the
  side camera's frame; the camera would need widening for that policy.)
- Exploratory next (decided 20:40, before training): DP on the 927 PPO demos with SHORTER chunks (act_horizon 2 and 4;
  ManiSkill's own RL-demo DP baseline uses 1), hoping for a chunked policy with PPO's fork-prone style; same checkpoint
  rule (50-step episodes) and the same closed-loop fork check.
- Short-chunk DP on the 927 PPO demos: act_horizon 2: <= 3% success_at_end at every checkpoint; act_horizon 4: <= 11%.
  No checkpoint qualifies -> DP cannot imitate PPO on this task (chunks 2, 4, 8 all fail).
- PPO made chunked by action repeat (exploratory): success_at_end 38% / 11.5% / 1% / 0% for repeat 1 / 2 / 4 / 8 (200
  episodes) -> PPO needs per-step control; it cannot be given a chunk without breaking it.
- PPO branched on a SINGLE action + 20 hold steps (what a per-step filter could see; --branch-h 1): 2 forks in 224 points
  (bar 15; results/benchmark_dd/maniskill/ppo_bh1_policy_fork_result.json). PPO's 24 forks need ~8 steps of accumulated
  execution error, which a per-step policy never commits to at decision time.
**LiftPegUpright summary (2026-10-21 21:30):** forks exist (expert demos: 20 strict; PPO closed loop: 24), but no
policy we have is both chunked AND fork-prone: the planner-demo DP (79%) is chunked but robust (1 fork); PPO forks but
is per-step (its forks are invisible to a single-action check, and chunking it destroys it); DP cannot learn from PPO's
demos. Stopped here pending the user's decision.

## Re-check after the user challenged the "no forks" results (2026-10-21 22:00)
1. Noise level - NOT the error. Jenga closed-loop tests (scale 2): median gripper-target offset at chunk end 3.2 mm
   (scale 3: 4.8 mm). ManiSkill checks realised a 3.7 mm median gripper deviation at chunk end (1x); ToolHang and
   SafeManip use the same normalised per-step noise. Forks do grow steeply with noise (planner-demo DP: 1 / 29-33 / 55
   forks at 1x / 2x / 3x, 1024 points; 2x = 7.4 mm, beyond anything Jenga was judged at; DP sampling is unseeded, so
   counts vary by a few between runs).
2. A REAL blind spot: the look-ahead. Every check so far judged the outcome 20 hold steps after the chunk (robot
   stopped). New mode (--to-end): after the noisy chunk the policy runs CLOSED LOOP to the episode's end in all 33 copies
   (copies of one moment share the policy's sampling noise, so policy randomness cannot fake a split). PPO, 1x noise:
   final success splits at 106 / 224 points; strict two-mode (>= 2 copies end standing < 20 deg AND >= 2 flat > 60 deg)
   at **56 / 224** (vs 24 with the 28-step look-ahead), including 19 of 32 episodes at the very FIRST chunk. ToolHang,
   SafeManip, StackPyramid and SafeLIBERO were all judged with the short look-ahead (and ToolHang/SafeManip with
   damage-only labels: slip / tip / knock, not "the step succeeds vs fails"), so their "few forks" verdicts may be
   understated and should be re-checked with run-to-end labels.
   Caveat: a split that appears only at the episode's end may not be visible to our filter, which imagines the chunk +
   a short hold in the world model; check 2 (world-model visibility) decides that.
3. Run-to-end on the chunked planner-demo DP, 1x: final success splits at 365 / 1024; strict two-mode at **194 / 1024
   (19%), in all 32 episodes** (vs 1 with the 28-step look-ahead). **Zero-noise CONTROL (8 episodes, 256 points): 27
   strict two-mode (10.5%)** - identical copies diverge (0.006 mm after the chunk, GPU float differences between parallel
   envs) and the closed-loop task amplifies it over ~200 steps. So run-to-end over-counts: the outcome over long horizons
   is chaotic; only the excess over the control (~9% of points) is attributable to noise-sized errors, and a split that
   only appears ~200 steps later is not something our filter (chunk + short imagined hold) can see or prevent. The
   short look-ahead remains the measure aligned with the filter; by it these tasks have few forks at Jenga-level noise.
   PPO run-to-end (56 / 224 strict) has no zero-noise control yet.

## KinDER + MimicGen StackThree quick checks (user 2026-10-21 ~22:30: "go ahead with kinder and stack three, quick checks first")
- KinDER (github.com/Princeton-Robot-Planning-and-Learning/kindergarden; MuJoCo dynamic3d, TidyBot++ mobile base + Kinova
  arm): BalanceBeam3D ships NO demos, no scripted (ds) policy and no planner model -> no policy to branch on; a quick
  check would first need a scripted balancing policy or teleop demos. Dynamic 3D tasks with demos (Shelf3D 9,
  SweepIntoDrawer3D, ConstrainedCupboard3D) are not fragile-physics tasks. Parked.
- MimicGen StackThree_D1 (spec fixed 22:50, before running; eval/mimicgen_fork_check.py): 1000 shipped demos (robosuite
  1.4.1 + mimicgen, OSC_POSE 20 Hz). Branch along demos 0-31 at every 8th step: exact + 32 noisy replays of the next 8
  demo actions (ToolHang noise) + 20 hold. Strict label vs the exact replay's end: BAD = a cube ends > 2 cm lower or
  > 45 deg more tilted; GOOD = every cube within 1 cm. Fork = >= 2 BAD and >= 2 GOOD. Same three bars as before.
  Zero-noise control (--scale 0, 8 demos) must give 0 forks. Also run at 2x as information (not gated).
- **StackThree RESULT (demos 0-31, 1006 points; results in ~/wksp/mimicgen_bench/fc_quick):** 1x (realised ~2.3 mm, below
  Jenga's 3.2): 38 forks, 45 points with any bad (forks 84%), 0 all-bad, 3.8% of points -> misses only the 5% bar.
  2x (~4.6 mm): 55 forks / 74 any bad / 0 all-bad / 5.5% -> passes all bars. Zero-noise control (2 demos, 67 points): 0
  (after the gripper-ramp fix). Profile = Jenga's: when a cube falls it is almost always noise-decided. 1.4x (Jenga-
  matched) running. Policy: robomimic Diffusion Policy (low-dim incl. object poses, 1000 demos, 600 epochs, same
  config as ToolHang), checkpoint rule as ToolHang (horizon 400; success on 50 fresh episodes); its own closed-loop
  fork check (eval/stack_three_policy_check.py, 1.4x) decides.
- **StackThree with OUR low-dim Diffusion Policy (epoch 600; 20-episode success 85%; object poses as input) FAILS its
  closed-loop fork check at matched noise (1.4x, eef 3.2 mm): 1 fork / 145 points** (8 seeds; 6 successes, 2 timeouts;
  ~/wksp/mimicgen_bench/policy_fc). Same pattern as LiftPeg: a policy trained on 1000 demos with exact object poses is
  precise, and its failures are timeouts. The demo forks (48) come from MimicGen's own generation noise, which the
  policy averages away.
- Exploratory next (decided 00:55, before training): an IMAGE-based Diffusion Policy on StackThree (agentview + wrist
  84 px, ResNet18 + spatial softmax + crop; gripper proprioception only, no object poses), like our Jenga policy, which
  sees through cameras. MimicGen reports image DP at ~38% on this task, i.e. much less precise. Same checkpoint rule and
  closed-loop fork check (the check script must render camera observations for this policy).

### StackThree image Diffusion Policy (robomimic, agentview + wrist cameras; 2026-10-22, exploratory, training ongoing)
Success on 20 episodes: epoch 100 1/10, epoch 200 6/20 (training is ~0.9 epochs/min; 600 planned). Closed-loop fork
check on the epoch-200 checkpoint, seeds 1-4, 1.4x (realised gripper deviation 3.16 mm, Jenga-matched): 94 points,
**3 forks**, 5 with any bad. Even a less precise image policy rarely forks on StackThree; low-dim DP gave 1 / 145, and
its re-measured success is 37/50 (74%). A later checkpoint will be checked, but the trend does not favour StackThree.

## MANIGUARD (arXiv 2608.17386, Aug 2026) desk check (2026-10-26 15:29 EDT; reading only, nothing run)
- Isaac Sim 4.5 / OmniGibson (BEHAVIOR-1K), Franka joint-space control at 20 Hz (physics 120 Hz), 8-action chunks; RTX GPU required
  (no Alliance H100 runs); ~3.8 GPU-min per rollout on an RTX 4090 (1,326 h / 21,000 rollouts). Code Apache-2.0
  (github.com/NU-IDEAS-Lab/ManiGuard), 8,000 demos + evaluated SFT checkpoints on Hugging Face.
- Clutter family (55 tasks: 29 dry "knock over or drop nothing", 26 liquid "carry without spilling"; intact = not dropped and
  tilt <= 45 deg, liquid containers <= 15 deg, spill = > 15% particle loss): **zero violations for every policy in every run**
  (paper: "no policy ever violates on Clutter"; pi0-SFT 81% / pi0.5-SFT 80% safe success = success). Its violations are in the
  procedural families (lid / jar / dusty: order-of-operations specs), which are not physical side effects.
- Verdict: not a fit as released (nothing for a side-effect filter to prevent). Could be made one by adding execution noise
  (our Jenga stress recipe), at the cost of an Isaac Sim stack, RTX-only compute and ~4 min per rollout. Not pursued now.
- **Correction (2026-10-26 15:32 EDT, user pointed to the Stack examples): the STACK family is a fit.** Spec G(intact(stack) and
  intact(target)) = "unstack without toppling anything", upright threshold 30 deg. Engaged violation rate (engaged rollouts
  that topple / drop something), ID / OOD: pi0.5-SFT 30 / 37%, pi0-SFT 48 / 57%, GR00T-SFT 62 / 66%, zero-shot 58-78%;
  safe success only 20 / 13% (pi0.5-SFT). 28 Stack tasks. Topples are common, as in Jenga; unknown: whether they are
  noise-sensitive (forks) and whether a latent world model sees them (our checks 1 and 2). Practical: RTX-only (local 3090,
  shared with the Jenga runs), Isaac Sim stack, ~4 min per rollout, physics determinism under state restore unknown.
