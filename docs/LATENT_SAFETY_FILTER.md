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
