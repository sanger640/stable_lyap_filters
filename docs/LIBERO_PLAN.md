# SafeLIBERO transfer plan (draft 2026-10-19, before any SafeLIBERO result)

Goal: test whether the Jenga method transfers to a second, standard benchmark with no task-specific engineering.
The method is the fork-monitor trigger, formulation G (speed / pause / amplitude / back-off of the ONE policy chunk,
G-CEM), the ensemble certifier and the hold fallback.

What transfers unchanged: the trigger rule, the G transformations, G-CEM settings, certificate, cost and fallback.
What cannot transfer: the world model. It was trained on Jenga scenes. A LIBERO world model is trained the same way:
latent dynamics on unlabelled rollouts, with no failure labels, no obstacle annotations and no task-specific code.
Collision labels are used only to score, never to train or tune.

## Setup (done 2026-10-19)
- Code: `~/wksp/safelibero/vlsa-aegis` (THU-RCSCT/vlsa-aegis, SafeLIBERO + AEGIS), kept outside this repo.
- Client env (py3.8, robosuite 1.4.1, mujoco 3.2.3): `vlsa-aegis/main/.venv`; LIBERO config in
  `~/wksp/safelibero/libero_config` (LIBERO_CONFIG_PATH), so no global ~/.libero is touched.
- Policy server: pi0.5-LIBERO (openpi, JAX) in `vlsa-aegis/openpi/.venv`; checkpoint
  `vlsa-aegis/checkpoints/pi05_libero` (12.4 GB, gs://openpi-assets). Same base policy as AEGIS, so the comparison is
  like-for-like (changed from the earlier OpenVLA-OFT idea for that reason).
- Evaluation client: `libero/policy_eval.py`, AEGIS's loop without their safety layer. Their collision metric is
  obstacle summed |displacement| > 1 mm. We also log max displacement and tilt change, to separate nudged from knocked over.

## Benchmark facts
4 suites (spatial, object, goal, long) x 4 tasks x 2 levels (I: obstacle next to the target; II: obstacle in the path)
= 32 scenarios x 50 fixed initial states = 1,600 episodes. Max steps 300 (550 for long). Control at 20 Hz.
Translation + rotation + gripper actions; pi0.5 replans every 5 steps.

## Stages (each gated; stop and report on a failed gate)
- **L0 Baseline.** pi0.5 alone on all 1,600 episodes (full action space). Reported by AEGIS (arXiv 2512.11891 v2,
  Table I, full action space; CAR = % collision-free episodes, TSR = % success within the limit, collisions do not end
  the episode):

  | suite | pi0.5 CAR | pi0.5 TSR | AEGIS CAR | AEGIS TSR | OpenVLA-OFT CAR / TSR |
  |---|---|---|---|---|---|
  | spatial | 14.0 | 59.3 | 68.0 | 68.5 | 8.3 / 36.5 |
  | goal | 20.0 | 60.0 | 76.5 | 82.8 | 19.5 / 24.3 |
  | object | 18.0 | 57.5 | 71.3 | 72.5 | 11.5 / 28.8 |
  | long | 16.5 | 54.3 | 59.8 | 46.3 | 6.0 / 15.3 |
  | average | 17.1 | 57.8 | 68.9 | 67.5 | 5.7 / 26.2 |

  Gate: our pi0.5 CAR and TSR within 7 pp of theirs per suite (400 eps per suite, SE ~2.5 pp) and within 4 pp on
  average. Otherwise debug the setup before anything else.
- **L1 Split (pre-registered here).** Initial states 0-24 of every scenario = development (world-model data, threshold
  calibration, any tuning); 25-49 = test, untouched until L4.
- **L2 Unlabelled data + world model.** On dev states only: pi0.5 rollouts, plus rollouts where random G
  transformations (and execution noise) are applied to random chunks, so that the model sees the consequences of
  slowing, pausing, shrinking and backing off. Same architecture and training recipe as the Jenga model (DINOv2
  latents, model_OIL recipe, ENS0-3 ensemble). No labels.
- **L3 Offline check (dev states).** Does the fork-monitor score predict obstacle displacement in held-out dev rollouts?
  Gate: AUROC >= 0.7 for collision within the next chunk window. The labels are used for evaluation only, and the
  threshold is calibrated on the unlabelled intervention rate as in Jenga.
- **L4 Closed loop on test states (25-49, 800 episodes).** Arms: policy | trigger + hold | trigger + G-CEM. Paired by
  initial state. Metrics: collision rate (their metric), task success, safe success (success and no collision),
  obstacle knocked over (tilt change >= 45 deg). Primary: G-CEM vs policy on safe success, one-sided 95%. Secondary:
  G-CEM vs hold on success (does optimisation keep progress better than holding?), vs AEGIS's reported numbers.
  Exact thresholds are to be fixed before L4 starts.

## Compute
- One RTX 3090: pi0.5 server (JAX) and the world model / DINOv2 encoder share it. Never run alongside the Jenga
  DINOv2 worker pools. Cluster (Rorqual slices) for the 1,600-episode baselines once the local smoke run works.

## Open decisions for the user
- AEGIS itself needs a ZhipuAI API key (VLM obstacle detection) and GroundingDINO. Plan: compare to their reported
  numbers; rerun AEGIS only if the user provides a key.
- Whether the success criterion here is LIBERO's own `done` (default) or something stricter.
