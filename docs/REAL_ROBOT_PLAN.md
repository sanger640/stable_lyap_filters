# Real-robot benchmark plan (Franka Panda) — draft 2026-10-21, for the user to confirm

Goal: the same filter (fork test trigger + G-CEM on the policy's own chunk, ensemble certificate, hold fallback) on a
real Panda, with a policy that is not ours to modify, and a fragile task whose failures are visible. Target: RSS 2027
Stage 1 (Dec 4) -> about 6 weeks.

## Task (mirrors the sim Jenga task, so the method transfers without redesign)
Three upright wooden (or foam-core) blocks on the table; the policy extracts the middle one without toppling its
neighbours. Block spacing and placement randomised by hand from a printed template (fixed set of ~20 layouts, half dev,
half test). Optional second task once the first works: place a block on a loosely stacked pile.

## Policy (frozen, treated as a black box)
Diffusion Policy (action chunks of 8-16 steps), trained from ~100-150 SpaceMouse teleop demos on the user's existing
DP stack (panda_express). Target 60-85% success so failures exist. The filter only edits the chunk the policy outputs.

## Sensing
- One fixed third-person RGB camera (the monitor's view), calibrated to the robot base; robot pixels masked from the
  Panda kinematic model + calibration (the same rule as sim: patches > 30% robot pixels).
- The policy may use its own cameras (wrist + third person); the monitor uses only the fixed camera + proprioception.

## World-model data (label-free)
No branching on hardware (cannot reset physics). Instead: policy rollouts with injected execution noise (the same
smoothed Gaussian as sim, 3 scales), ~300 episodes total (~6-8 robot hours), frames at 10 Hz. Train the 2b-WM recipe
(DINOv2 -> PCA 8 -> FiLM transition, proprio + actions as input).

## Offline check before closed loop (pre-registered, as in sim)
From the logged rollouts: does the imagined fork score h predict the topples that later happen (AUROC >= 0.7)? Only then
run closed loop.

## Closed-loop evaluation
~60 episodes policy alone vs ~60 filtered, alternating, same 10 test layouts each, outcomes (extracted; neighbour
toppled; knocked before extraction; v4 and v4b as in sim) scored from video by a script plus a blind human check.

## Timeline (approx.)
| week | work |
|---|---|
| 1 (Oct 26) | camera mount + calibration, blocks, teleop demos |
| 2 | train DP, measure success; logging pipeline (frames, proprio, actions, timestamps) |
| 3 | noisy rollouts for WM data |
| 4 | WM training + offline check |
| 5 | closed-loop eval |
| 6 | analysis, video, writing |

## Open questions for the user
1. Robot control stack (franka_ros / libfranka / deoxys / polymetis)? Control rate and how chunks are executed?
2. Cameras available (RealSense / ZED / other) and can one be fixed rigidly?
3. Lab time per week on the Panda?
4. Use the existing panda_express DP code, or train a fresh DP?
