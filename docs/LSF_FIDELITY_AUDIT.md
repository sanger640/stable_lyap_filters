# Latent safety filter baseline: fidelity audit against the paper and the authors' code (2026-10-27)

User request (2026-10-27): make sure the labelled latent safety filter (LSF) is implemented faithfully to the paper
and its code. Sources read:
- Paper: Nakamura, Peters & Bajcsy, "Generalizing Safety Beyond Collision-Avoidance via Latent-Space Reachability
  Analysis", RSS 2025, arXiv 2502.00935 (html version): Eq. 3-9, Sec. III-V, Tables V-VI, Appendix A.
- Code: github.com/CMU-IntentLab/latent-safety: `main` (Dreamer RSSM: PyHJ/reach_rl_gym_envs/franka-wm.py,
  scripts/run_training_ddpg-wm.py, PyHJ/policy/modelfree/BasePolicy_Annealing_Avoid.py, configs.yaml), `dino` (DINO-WM:
  franka-DINOwm.py, scripts/run_training_ddpg-dinowm.py, dino_wm/train_dino_classifier.py, ddpg_avoid_classical_dinowm.py),
  `isaaclab` (runtime filter: source/standalone/workflows/dreamerv3-torch/dreamer_rollout.py + configs.yaml).
  Read-only copies, cloned 2026-10-27, not executed.

The paper's simulated manipulation task is ours: "grasp and lift a green block placed between two red blocks without
toppling the red blocks" (Sec. IV-B). Failure labels: red blocks toppled, from privileged simulator state - same as ours.

## Comparison

| Element | Paper | Authors' code | Our LSF stage 2 (lsfa030, used in h2h3 / h2h4) | Paper-faithful re-implementation (lsfp, this audit) |
|---|---|---|---|---|
| World model | RSSM (sim), DINO-WM (hardware) | same | our model_OIL (DINOv2 patch latent, deterministic), shared by every method | same as ours (deterministic, like DINO-WM) |
| Classifier loss | hinge, delta > 0 (value not given), tanh on output | hinge, delta 0.75 on the raw output; margin = tanh(2 x) (DINO) / tanh(x) (RSSM) | hinge, delta 0.25 on the tanh output | ours unchanged (paper-consistent; delta unspecified) |
| Classifier input | latent | latent (+ state tokens in DINO-WM) | latent + proprio (follow-up paper) | same |
| Bellman target | (1 - g) l + g min(l, max V(z')) | g min(r, Q'(s', pi'(s'))) + (1 - g) r with r = margin of the NEXT latent; no terminal | same form, margin of the CURRENT latent + an uncertainty margin (UNISafe) | code's form: r = margin of the next latent, no terminal |
| Discount | 0.9999 (Table VI), no annealing | dino: 0 for a warm-up epoch, then 0.95 | annealed 0.85 -> ~0.93 at the end (UNISafe schedule), 15-step episodes truncated as terminal | `paper`: 0.9999 constant; `code`: 0 warm-up then 0.95 |
| RL algorithm | DDPG (manipulation, hardware) | DDPG (Tianshou / PyHJ) | SAC, twin critics, entropy, learned alpha (UNISafe, same group's follow-up) | DDPG |
| Networks | actor / critic 4 x 512 ReLU, AdamW, lr 1e-4 / 1e-3, batch 512 | same; critic weight decay 1e-3; actor loss + small action-magnitude penalties (DINO) | conv trunk + 3 x 512, Adam 1e-4, batch 512 | conv trunk + 4 x 512, AdamW 1e-4 / 1e-3 (critic wd 1e-3), batch 512; no action penalties |
| Exploration | not stated | Gaussian sigma 0.1 on the normalised action | SAC's own entropy | Gaussian sigma 0.1 on the normalised action |
| Training volume | 50 epochs | 16 epochs x 40k steps x 0.125 = ~80k updates, replay 40k | 60k updates, replay 400k | 76,250 updates, replay 40k, 1 update per 8 transitions |
| Imagination | resets = encoded random data observations; horizon not stated (25 in Dubins) | resets from data; 10-step episodes (DINO Franka) | resets from training data; 15-step episodes | resets from training data; 10-step episodes |
| Stochasticity / pessimism | expectation over the stochastic model; worst-of-N only as an optional footnote; none for deterministic DINO-WM | RSSM: worst of 5 samples per step; DINO: none | worst of 8 transitions under our measured noise model + perception noise at resets | none (deterministic) |
| Uncertainty margin | none | none | ensemble disagreement margin (UNISafe) | none |
| Runtime check | V at the ONE-step imagined next latent > eps | same | min over 8 noisy imagined next latents > eps | one imagined next latent from the real encoded frame |
| Fallback action | safety policy at the CURRENT latent | code queries the actor at the imagined next latent | safety actor at the current latent | current latent (paper) |
| eps | 0.4 (sim), 0.3 (hardware) | buffer_ddpg 0.8, buffer_sac 0.1 (isaaclab config) | 0.30, selected on dev from {0.1, 0.3, 0.5, 0.6, 0.65, 0.7} | grid {0.1, 0.2, 0.3, 0.4, 0.5}, LSF dev rule |
| Filter rate | every control step, 15 Hz | same | every model step (0.2 s = 2 control steps at 10 Hz; the finest our model allows) | same as ours |

## Verdict

- Our LSF stage 2 is a STRENGTHENED variant, not the method as published: SAC with an uncertainty margin and gamma
  annealing (from the same group's follow-up, UNISafe), plus pessimism over our noise model in training AND at runtime.
  Every addition makes it more robust or more conservative; none was taken away. It should be named
  "LSF (UNISafe-style training, noise-aware)" in the paper, not "LSF as published".
- Its runtime check (worst of 8 noisy next states) is more conservative than the paper's single imagined next state.
  Its eps was tuned on dev (0.30 = the paper's hardware value).
- The classifier is paper-consistent; its margin scale (delta 0.25 on tanh) differs from the code's (0.75 raw).
- Our earlier claim that "the published DDPG collapses in our setting" was not a test of the published DDPG: that
  run used 30% exploration noise, ~1 update per 60 transitions, ~19k updates, annealed gamma and current-state
  margins. The faithful DDPG has not been run until this audit.
- The switching rate is 5 Hz instead of 15 Hz (harness limit: one world-model step = 2 control steps).

## Plan (pre-registered in results/jenga/lsf_pub_protocol.json)

1. Train the paper-faithful DDPG value twice: `paper` (gamma 0.9999) and `code` (warm-up 0, then 0.95).
2. Same offline gate as the 2026-10-21 re-audit: actor saturated < 20%, imagined V after its own plan >= hold, AUROC
   (-V vs any toppling future) >= 0.70.
3. For each that passes: dev sweep eps {0.1, 0.2, 0.3, 0.4, 0.5} (paper's runtime rule, every model step) on episodes
   15000-15099 at 2x, LSF dev rule. Reported next to lsfa030 on the same episodes (9 / 76 / 59).
4. The decisive head-to-head then includes the stronger of {paper-faithful, strengthened} as the labelled baseline,
   with the other reported, and the naming above.

## Addendum (2026-10-27): our labelled baseline vs UNISafe's code

UNISafe (Seo, Nakamura & Bajcsy, CoRL 2025, arXiv 2505.00779; github.com/CMU-IntentLab/UNISafe,
reachability/env/takeoff_wm_failure.py, reachability/config.yaml) folds uncertainty into the reward exactly as ours does:
reward = min(processed (ood_threshold - ensemble disagreement of the current state-action), 1.5 tanh(0.5 - p_fail)),
SAC in imagination; ood_threshold = 4.0 in its config. (The paper describes reachability on an augmented state; the code
implements it as this min.) Differences to ours: threshold (theirs a fixed calibrated value; ours the 95th percentile of
disagreement under the policy's own actions), failure margin (probability classifier vs the LSF hinge classifier), and our
noise-aware pessimism (worst of 8 noisy transitions in training and at runtime), which UNISafe does not have.

So: value_unisafe.pt (2026-10-21; SAC + uncertainty margin, no noise pessimism) is UNISafe as implemented in its code, up
to the threshold calibration; with the paper's runtime rule (lsfs, one imagined next latent, every model step) its best
dev setting was 13 / 63 / 26 (topples / v4 / v4b; policy 27 / 66 / 58). value_unisafe_alea.pt (lsfa030, used in h2h3 /
h2h4) = UNISafe + noise-aware pessimism: 9 / 76 / 59. Naming in the paper: "LSF (as published)" (pending, lsf_pub),
"UNISafe (re-implementation)" = value_unisafe.pt, "UNISafe + noise-aware pessimism" = lsfa030 (the strongest; carried to
the decisive test; the others reported from dev).

## Result (2026-10-27 19:37 EDT): the faithful DDPG fails the pre-registered gate in both versions

| | actor saturated | V after actor / hold / policy (median) | AUROC | gate |
|---|---|---|---|---|
| paper (gamma 0.9999) | 100% | -2.75 / -2.74 / -2.74 (diverged out of the margin's range) | 0.72 | FAIL |
| code (gamma 0 warm-up -> 0.95) | 100% (bang-bang) | 0.87 / 0.84 / 0.84 | 0.75 | FAIL (saturation only) |

With gamma 0.9999 and no warm-up the critic never anchors to the margin and drifts; the authors' own schedule fixes the
critic (it ranks toppling futures, and the actor beats holding still) but its actor is bang-bang. Per protocol no dev
sweep was run.
