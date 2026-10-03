# Plan: general visual world models with branching fidelity (after the CoRL 2026 workshop draft)

Goal (user, 2026-10-03): IPE-like safety monitoring for ARBITRARY images, so the rollout engine must be a
general learned visual world model. Question: how do we make such models branch where physics branches?

## What the evidence says (all in NOTES.md)

| Finding | Source |
|---|---|
| Stochastic token WM: right amount of spread, isotropic, ~equal on forks and non-forks (kappa 0.19) | step3_diagnostics, fork_label_reanalysis |
| Deterministic token WM: spread aligned with the branch axis but too small (kappa 0.52) | same |
| Uncertainty in the inputs (G or exact rendered perturbed starts) + deterministic dynamics: no gain (0.29 / 0.19) | step3b P1 |
| Step-vs-ramp: models misplace the fork threshold (2+ push std), side agreement 67-70% vs 62% always-stays; transition 3x wider | step_vs_ramp |
| Data scaling fixes response size, not sharpness (W2a); coverage alone fails (Step 7, D12) | older state-model work |
| The compressed latent keeps forks (kappa 0.93), so representation is not the main bottleneck | gate D0 |

Main bottleneck: the learned dynamics do not know WHERE the fork is. Making jumps (discontinuous
architectures) is secondary.

## Levers (each with its hypothesis)

1. **Fork-rich paired counterfactual data**: many starts, each with dozens of slightly perturbed scenes and
   actions, generated in simulation. Hypothesis: models never see the contrast in logged data.
2. **Set-level objective**: match the K sampled futures to the K real futures as sets (energy score / MMD in
   latent space). Hypothesis: per-future squared error cannot reward correct placement or grouping.
3. **Whole-clip generator** (APPROVED by the user 2026-10-03 as an exception to "step-wise integrated
   dynamics only"): flow/diffusion over the whole 19-step latent trajectory conditioned on the start and the
   action chunk. Hypothesis: one global sample per future can commit to an outcome; per-step noise diffuses.
4. **Action-sensitivity loss** (D2-style intervention consistency on visual latents): predicted differences
   between nearby actions must match real differences. Hypothesis: under-amplification.
5. **Discrete event latents** (lower priority given step-vs-ramp and W2a).
6. **Scale**: fine-tune a pretrained WM (V-JEPA 2-AC) with the winning recipe.

## Phases

**Phase 1 (our three-block task; one frozen protocol; held-out pre-test P1 before any fresh panel).**
Factorial over the most likely levers, same latent (channel-PCA-8), same budget:
- (a) data: current 1,500 states x 8 futures vs fork-enriched 10x fork states x 32 futures;
- (b) objective: per-future flow matching vs set-level energy score;
- (c) generator: step-wise vs whole-clip;
- (d) + action-sensitivity loss on the best cell.
Selection on branching fidelity only (P1 kappa vs real futures, side agreement, threshold shift), never on
prediction error. Panel 6 (fresh) only for the selected model.

**Phase 2 (generality).** Procedurally varied scenes (rows, stacks, leaning objects, varied shapes and
masses) with physics ground-truth forks; train with the Phase-1 recipe; test the fork probe on unseen scene
TYPES. Also a Route-A baseline (foundation-model reconstruction with sampled uncertainty + physics engine).

**Phase 3 (large models).** Fine-tune and benchmark pretrained visual WMs (V-JEPA 2-AC; Cosmos-class if
compute allows) with the recipe; report branching fidelity next to average error.

## Standing rules
Frozen protocols before data; fresh panels for confirmation; select on branching fidelity; report the
always-stays / always-alarm baselines next to every agreement number.
