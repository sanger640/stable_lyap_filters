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
  **Result (2026-10-19): gate passed** — CAR / TSR spatial 14.2 / 62.2, goal 20.5 / 62.8, object 17.5 / 56.0,
  long 15.0 / 56.2, average 16.8 / 59.3 (results/libero/l0_result.json).
- **L1 Split (pre-registered here).** Initial states 0-24 of every scenario = development (world-model data, threshold
  calibration, any tuning); 25-49 = test, untouched until L4.
- **L2 Unlabelled data + world model** (spec fixed 2026-10-19 before collection; code `libero/collect_l2.py`,
  `libero/g_transform.py`). Dev states 0-24 of all 32 scenarios = 800 pi0.5 episodes. Per episode up to 4 random
  decision points; at each, 8 futures branched from the exact simulator state (see the data correction below):
  f0 the policy's 10-step chunk as-is, f1-f2 with execution noise, f3-f7 a uniformly random G transformation
  (v, k, s, b in [0,1] x [0,8] x [0,1] x [0,30 mm]) plus execution noise; each followed by 10 hold steps. Execution
  noise = smoothed Gaussian on delta actions (sigma 0.15 xyz, 0.05 rot, 3-step box filter). G on LIBERO deltas acts on
  the cumulative plan exactly as Jenga's G acts on targets (rotation re-timed only, gripper on the policy's clock).
  Frames: agentview 392 px every 2 env steps (one model step = 0.1 s), JPEG q95. Object poses are stored as
  `evalonly_*` for L3 scoring and never read by training.
  World model: same recipe as Jenga (frozen DINOv2 392 -> 28 x 28 patch tokens -> channel PCA 8 fitted on LIBERO dev
  nominal frames; deterministic residual FiLM transition, self-rollout batches; then a 4-member ensemble for the
  certifier). Action input per model step = the 2 env delta actions (14-d). Train on scenarios' dev states with
  tasks split 90/10 by episode for validation loss only.
- **L3 Offline check (dev states).** Does the fork-monitor score predict obstacle displacement in held-out dev rollouts?
  Gate: AUROC >= 0.7 for collision within the next chunk window. The labels are used for evaluation only, and the
  threshold is calibrated on the unlabelled intervention rate as in Jenga.
- **L2/L3 implementation details (fixed 2026-10-19, before any encoded latent or L3 number was looked at;
  code `libero/wm_libero.py`, `libero/camera_dump.py`).**
  - PCA: channel PCA 8 on 48,000 DINOv2-392 tokens from 4 nominal frames each of 120 training dev files.
  - Validation / L3 episodes: dev states with ep % 10 == 0 (0, 10, 20; 96 of 800 episodes). Training uses the rest.
  - Model: eval/fast_train.py unchanged (batch 128, 7,500 steps), 10 model steps (chunk end = step 4, hold end =
    step 9). Starts = the true start latent (LIBERO has no perception-error data, so no start sampler; uncertainty
    comes from execution noise, exactly the data's noise model, scaled by s). Batch mix: 25% label-free rare-mode
    branch futures (phase1e detect_modes), 25% other branch futures, 50% nominal-trajectory windows (every 5 model
    steps). Ensemble: 4 members fine-tuned 3,000 steps at 0.3x LR, seeds 0-3.
  - Monitor scores per decision (latent + proprioception only): 32 imagined futures of the chunk + 10 hold steps
    under execution noise x s, s in (0.5, 1, 2, 3); h = log2 of the smallest s at which the hold-10 and hold-30
    endings split (two-mode test, as Jenga), capped at log2 3; o = max over futures (s = 1) of the mean squared latent
    change outside the hand mask (projected hand path discs + base-to-hand arm corridor, from the camera matrix and
    the end-effector position); u = ensemble disagreement outside the hand. Ensemble versions follow the
    certificate (every member must pass): h_ens = min over the 4 members, o_ens = max over members (set 2026-10-19
    while writing the shared scorer, before any trained model existed). L3 and L4 use the same scorer
    (`wm_libero.score_plans`).
  - **Data correction (2026-10-19, found while building the arm replay, before any model was trained):** the
    flattened MuJoCo state misses two pieces of hidden state, robosuite's ramped gripper command
    (`gripper.current_action`, +-0.01 per step) and the solver warm start. In files collected before the fix
    (no `gripper_state_restored` key; about 520 of 800), future f0 starts exactly from the policy's state, but f1-f7
    and the nominal steps after a branch start from the hidden state the previous future left behind (gripper
    command ahead by up to 20 steps per earlier future; warm start differs). Every recorded future is still real
    physics from the recorded start frame, so the files are kept for training (it acts as extra execution noise),
    and L3 labels use f0 only, which is exact. Files from the fix on save and restore both. The earlier note
    "restore + replay is bit-identical" held for the simulator state only.
    Final composition: the 600 non-long episodes are all pre-fix (their runs had started before it); the 200 long
    episodes crashed in the first run (branch steps counted toward robosuite's horizon -> "executing action in
    terminated episode") and were re-collected with the full fix (gripper command, warm start, step counter / done
    flag; futures run with ignore_done).
  - Arm links for the mask (libero/arm_links.py) come from replaying each episode, which is approximate for the
    same reason (and because hidden gripper state carries over between episodes in one env); the replayed arm is
    shifted so its end effector matches the recorded one at every frame.
  - L3 labels (evaluation only): real future f0 (the chunk as-is + hold); collision = active-obstacle summed
    |displacement| > 1 mm within the window; damage = >= 50 mm or tilt change >= 45 deg.
  - **L3 primary**: AUROC of o (ensemble) for collision in f0, on branches where the obstacle had not yet moved
    (> 1 mm) before the branch point (the question the filter faces: will this chunk cause the first contact?).
    Gate >= 0.7. Secondary: damage label; all branches; h, u; o on the real encoded futures (ceiling without the
    world model). If o fails and h passes (or the reverse) that is reported as is, with no re-gating.
- **L4 Closed loop on test states (25-49, 800 episodes).** Arms: policy | trigger + hold | trigger + G-CEM. Paired by
  initial state. Metrics: collision rate (their metric), task success, safe success (success and no collision),
  obstacle knocked over (tilt change >= 45 deg). Primary: G-CEM vs policy on safe success, one-sided 95%. Secondary:
  G-CEM vs hold on success (does optimisation keep progress better than holding?), vs AEGIS's reported numbers.
  Exact thresholds are to be fixed before L4 starts.
- **Safety metrics, both reported (amendment 2026-10-19, agreed with the user before any filter run on LIBERO).**
  The benchmark's collision rule (obstacle summed |displacement| > 1 mm at any time) counts a brush the same as
  knocking the obstacle over. Like v4 / v4b on Jenga, both are reported everywhere:
  - *Strict (benchmark)*: collision = summed |displacement| > 1 mm. CAR and strict safe success (success and no
    collision). Used for every comparison with AEGIS and other published numbers.
  - *Damage*: obstacle knocked over (tilt change >= 45 deg) or moved >= 50 mm (summed |displacement|, the same L1 measure as the 1 mm rule; this is what
    `obstacle_max_disp_mm` logs).
    Damage-free rate and damage safe success (success and no damage).
  - Primary L4 test: G-CEM vs policy on damage safe success; strict safe success is the co-reported secondary with
    the same test. The filter is not designed for millimetre clearance, so a strict-metric gain is not expected to
    match AEGIS, and that is stated in advance.
  - L0 baseline at amendment time (1,556 / 1,600 episodes, test and dev states pooled): CAR 14-21 % per suite;
    damage-free 20-47 %; strict safe success 14-19 %; damage safe success 18-42 %.

- **Amendment (user decision 2026-10-19, before any model was trained): no ensemble for now.** The certificate is
  the base world model alone (64 fresh futures), as Jenga's single-model certificate; the 4-member ensemble is added
  only once the filter shows promise. O_MAX (label-free 99th percentile of o for the policy's own chunks) is computed
  on a random 600 training-dev decisions (time). L3 primary becomes AUROC of o from the base model (same population, label and
  gate); the ensemble scores (h_ens, o_ens, u) are not computed.
- **Amendment (user decision 2026-10-19 ~12:00, while the world model was still training, before any L3 score):
  L3 primary = AUROC of the FORK score (-h, base model) for collision in f0 on the undisturbed-obstacle branches,
  gate >= 0.7. The fork monitor is the method's trigger; o (disturbance away from the robot) is the secondary
  backstop and is reported (AUROC, and how often each check fires) but is not gated. The closed-loop trigger is
  unchanged (either check fails -> act), as on Jenga. h takes 4 values, so ties are ranked by average rank.
- **L4 scale (decided with the user 2026-10-19, before L3 and before any closed-loop result): start small.**
  Measured cost on the shared RTX 3090: ~2 s per decision for the trigger, ~43 s per G-CEM intervention.
  - L4a dev pilot (only if L3 passes): G-CEM on 32 scenarios x 2 dev states (5, 15), to measure trigger rate and
    time per episode and to catch bugs. Not a test; no claims.
  - L4b small test: test states 25-34 of all 32 scenarios (320 episodes per arm). Arms: G-CEM and hold (both run
    now); policy arm = the L0 episodes on the same initial states (same evaluation loop; pi0.5 sampling is random
    either way, so pairing is by initial state). Primary: G-CEM vs policy on damage safe success, one-sided 95%
    lower bound > 0; co-reported: strict safe success, success, collision, damage; G-CEM vs hold.
  - Scale to all 800 test episodes (cluster) only if L4b is promising but not conclusive; states 35-49 stay untouched
    until then.
  - Speed-ups allowed before L4a only (torch.compile, idle GPU); settings stay fixed from L4a on.

## Compute
- One RTX 3090: pi0.5 server (JAX) and the world model / DINOv2 encoder share it. Never run alongside the Jenga
  DINOv2 worker pools. Cluster (Rorqual slices) for the 1,600-episode baselines once the local smoke run works.

## Open decisions for the user
- AEGIS itself needs a ZhipuAI API key (VLM obstacle detection) and GroundingDINO. Plan: compare to their reported
  numbers; rerun AEGIS only if the user provides a key.
- Whether the success criterion here is LIBERO's own `done` (default) or something stricter.
