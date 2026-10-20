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
