# Latent safety filter (LSF) baseline on Jenga — protocol (fixed 2026-10-20 before any LSF code ran)

Baseline: Nakamura, Peters & Bajcsy, "Generalizing Safety Beyond Collision-Avoidance via Latent-Space Reachability
Analysis" (RSS 2025, arXiv 2502.00935), with the design choices of the follow-up "Designing Latent Safety Filters using
Pre-Trained Vision Models" (arXiv 2509.14758) where the original is silent. Goal: a fair head-to-head against our
label-free G-CEM filter on the same world model, policy, episodes and seeds.

## The method (as published)
1. Failure classifier l(z): MLP on world-model latents, hinge loss with margin delta:
   L = mean_safe ReLU(delta - l(z+)) + mean_fail ReLU(delta + l(z-)), tanh on the output; failure set {z : l(z) < 0}.
   Labels from privileged simulator state in simulation (their block task: block fallen).
2. Safety value V(z): discounted latent reachability backup learned in imagination,
   V(z) = (1 - g) l(z) + g min{ l(z), max_a V(f(z, a)) }, g = 0.9999; continuous actions -> DDPG (critic = V/Q, actor =
   safety policy maximising V of the next latent); imagination starts from encoded dataset frames; 4 x 512 ReLU MLPs,
   AdamW, actor lr 1e-4, critic lr 1e-3, batch 512.
3. Least-restrictive runtime filter: imagine the task action one step ahead; execute it if V(next) > eps, else the safety
   actor's action. Published eps: 0.4 (simulation), 0.3 (hardware).

## Adaptation to Jenga (stated, not hidden)
- World model: model_OIL (the one G-CEM uses in Stage 4/4b; deterministic, DINOv2-392 latents 28x28x8). One model step
  = 2 control steps; action input = the 2 absolute targets + gripper (8-d), as in all our filters.
- Latent for the heads: the 28x28x8 latent (a small conv trunk -> MLP, since our latent is a patch grid, not a CLS
  vector) plus proprioception (current target xyz + gripper), which the follow-up paper found necessary.
- Labels (privileged, simulator): failure = a neighbour block toppled (the same topple rule as all Jenga results,
  eval/fresh_fragility_panel.topple_count), sticky from the step it happens; computed per imagined step from the
  stored traces of results/program_b/{closedloop_v1, cf_v1, cf_v2, oil_v1} training splits (validation splits held out).
- Safety actor action: a bounded change of the target per model step (|delta xyz| <= 1 cm per control step, the policy's
  own range), gripper held; DDPG with target networks (tau 0.005) and a replay buffer of imagined transitions; imagined
  rollouts of 15 model steps (the look-ahead G-CEM uses) from dataset latents.
- Action chunks (not covered by the paper): at each decision, imagine the executed part of the policy's chunk (8
  control steps = 4 model steps); if V > eps at every imagined step, execute it; else the safety actor drives the next
  8 control steps, re-encoding the real frame every model step.
- eps tuned on dev episodes only (grid {0.1, 0.2, 0.3, 0.4, 0.5}); selection rule fixed now: the eps with the fewest
  neighbour topples among those whose v4 success is not more than 5 pp below the policy alone; ties -> larger eps.
- Head-to-head (pre-registered separately before it runs): fresh episodes, arms policy | G-CEM | LSF, same seeds,
  metrics as Stage 4 (topples, v4, v4b), exact McNemar.

## What differs between the two methods (for the paper)
LSF needs failure labels, a trained classifier, value function and safety policy; G-CEM needs none of these (no labels,
no extra training beyond the world model). Same world model, so differences come from the safety logic, not the model.

## Clarifications (fixed 2026-10-20 ~17:00, after the data stage, before any training)
- Data stage result: train 231,748 labelled latents (93,484 failed), 5,969 imagination resets; val 32,796 (13,606
  failed), 729 resets (validation episodes of the same sources). Failure share is 40% (not 25%) because many files have
  few safe latents to sample; the hinge loss averages safe and failed terms separately, so balance does not matter.
- Classifier margin DELTA = 0.25 on the tanh output (the paper does not report its value). Model selection: the epoch
  with the lowest validation hinge loss.
- Runtime with chunks: our harness re-decides every 8 control steps (0.8 s), as for every other filter. If the policy's
  segment fails the check, the safety actor plans the 8-step segment in imagination (4 model steps) from the current
  real latent and that segment is executed; the next decision re-encodes the real frame. (The paper re-plans every
  control step at 15 Hz; per-step re-encoding is not possible in this harness without changing the comparison.)
- Compute (2026-10-20 ~17:40, before training): training runs on Nibi (H100 slice) where cf_v1 / cf_v2 already live.
  The data are built in two parts with the same stage and rules: cf_v1 + cf_v2 on Nibi (tag cf), closedloop_v1 +
  oil_v1 locally (tag clo: train 59,618 latents / 25,013 failed / 2,569 resets; val 5,251 / 2,206 / 329) and
  uploaded; merged with eval/lsf.py merge. Exact counts differ slightly from the earlier single local build (separate
  random streams per part); the merged summary files record the final numbers.

## Training run 1 (Nibi job 24697324, 2026-10-20 17:44-17:56) and sanity check
Classifier: validation accuracy ~98.7% (best val hinge 0.018, epoch 3). Value (DDPG, 300 iterations): COLLAPSED -
on 729 held-out decision points V = Q(z, g, pi) is about -0.47 everywhere (5-95% range -0.50 to -0.43; 100% below 0),
AUROC(-V, any real topple) 0.45 (no signal), while the classifier at the current state alone gives 0.75. Likely cause:
with g = 0.9999 the value approaches the worst classifier score along imagined rollouts, and 15-step imagined rollouts
under the untrained actor drift to latents the classifier calls failed. Not usable; next: measure l along imagined
rollouts (hold still / dataset policy actions / actor) to locate the drift, then fix training (no test data used).
results/jenga/lsf_value_diag.json.
- Drift check (same day, validation states only): the classifier calls 2-3% of non-toppling imagined futures failed
  under hold-still, dataset policy actions and the actor alike - the same as on real frames - so imagination is not the
  problem. Diagnosis: with a fixed g = 0.9999 a safe state's value can rise only (1-g) l = 0.0001 per backup while
  the min propagates pessimism at once, so the critic drifts down from its initialisation. Fix (run 2): anneal g from
  0.9 to 0.9999 over the first 80% of training (as in the reachability-RL toolbox the paper builds on, to my
  understanding), 600 iterations; nothing else changed. Selection of the run uses the same validation sanity check.
- Run 2 (gamma annealed 0.9 -> 0.9999, 600 iterations) also collapsed: critic ~-0.47 for every input although the
  classifier scores safe resets ~0.8 and failed latents ~-0.33. The log shows the value rising until the buffer fills,
  then sinking steadily: the min-backup with function approximation spreads the ~3% failed states' low values to all
  similar latents, and with g near 1 nothing pulls it back.
- **Tuning rule for g (fixed before running, validation data only, never test episodes):** fixed g in {0.9, 0.95,
  0.99} (effective horizons ~1, 2, 10 s at 0.1 s per model step), 600 iterations each; choose the g with the highest
  AUROC of -V for "any real future topples" on the 729 held-out validation decision points (eval/lsf_diag.py).
  If none exceeds the classifier-only score at the current state (0.75), report that the learned value adds nothing.
- Reference code checked (2026-10-20 ~18:35, before any of the gamma runs finished): the DINO-WM-based reference
  implementation (github.com/tabz23/Latent-Safety-Filters, train_HJ_mani_configs.yaml) uses gamma = 0.98, batch 128,
  actor lr 1e-4, critic lr 1e-3, tau 0.005; the Bajcsy group's UNISafe (github.com/CMU-IntentLab/UNISafe) anneals gamma
  0.85 -> 0.9999 in steps every 4,000 updates, uses twin critics (SAC), and truncates imagined episodes (target = l at
  the final step). Added: LSF_TRUNC=1 (final imagined step terminal) and gamma = 0.98. Runs: fixed gamma 0.9 / 0.95 /
  0.99 without truncation (running) + 0.9 / 0.95 / 0.98 / 0.99 with truncation. Selection rule unchanged: highest
  validation AUROC across all seven; the chosen setting then goes to the dev eps sweep.

## Value selection (Nibi job 24703953, 2026-10-20 19:33; rule above)
AUROC(-V, any real topple) on the 729 held-out validation decision points (classifier at the current state: 0.755):
| run | AUROC | share V < 0 |
|---|---|---|
| g 0.9999 (run 1) | 0.45 | 100% |
| g annealed 0.9 -> 0.9999 (run 2) | 0.39 | 100% |
| g 0.9 | 0.782 | 4% |
| **g 0.95 (selected)** | **0.785** | 16% |
| g 0.99 | 0.781 | 67% |
| g 0.9 / 0.95 / 0.98 / 0.99 + truncation | 0.72 / 0.66 / 0.74 / 0.71 | 2 / 2 / 2 / 47% |
Selected: fixed g = 0.95, no truncation (results/program_b/lsf_v1/value_g0.95.pt). The fixed-g runs are within noise of
each other; all three beat the classifier alone, so the learned value adds a little foresight. Next: dev eps sweep.
results/jenga/lsf_value_diag_value*.json.

## Dev eps sweep spec (fixed 2026-10-20 ~19:45, before any closed-loop LSF episode ran)
Episodes 15000-15099 (100 fresh dev episodes; no earlier run used 15000+), Stage 4 paired setup (stress layouts,
settled start, execution noise x2, policy seeded per episode), arms record (policy alone) and lsfe1..lsfe5 (eps 0.1 -
0.5), world model model_OIL, value_g0.95.pt. Packed on Nibi (paired_group.sbatch, 8 workers per GPU). Selection rule
as above (fewest topples with v4 no more than 5 pp below the policy; ties -> larger eps); topples by the same replay
outcome audit as Stage 4. Smoke first: episodes 15900-15901, record + lsfe3 (not part of any analysis).
- **Smoke result and grid amendment (2026-10-20 19:50, before any dev episode ran).** Smoke (15900-15901, eps 0.3): the
  LSF intervened on 60/60 chunks in both episodes (V of the policy's segment 0.05-0.28, never above 0.3); its safety
  actor then kept moving the target away (up to ~1.7 m from the policy's plan; least-restrictive switching never hands
  back while V < eps), so the block was never lifted. With g = 0.95 the validation V quantiles are 5/25/50/75/95% =
  -0.20 / 0.09 / 0.15 / 0.24 / 0.43 (83% below 0.3), so the published eps (0.3-0.4, on that paper's own value scale)
  means "always intervene" here. Amended grid, covering our value's range: eps in {0.00, 0.05, 0.10, 0.15, 0.20, 0.30}
  (arms lsfe000 ... lsfe030); selection rule unchanged. Behaviour of the safety actor recorded as is (method as published).
- Scheduling note (21:05, nothing had run): the packed whole-GPU job (24705559) was held behind a node in maintenance
  (est. start 02:37); cancelled and resubmitted unchanged in content as a 100-task slice array (paired_array.sbatch,
  1g.10gb, job 24709409; same episodes, arms, code).
- Scheduling note 2 (22:10): Nibi ran dev tasks 0-69; the never-started tasks 70-99 were cancelled there and run on Fir
  (job 66192391; same container, environment and code; Fir mirrors Nibi's user-site packages). Reproduction check:
  episode 15000 (record + lsfe010) re-run on Fir (tag _firrepro) must match Nibi's logged outcome; if not, tasks 70-99
  are re-run on Nibi instead.
