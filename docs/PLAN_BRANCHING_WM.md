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
| Label-free rebalancing (US: oversample each state's own rare outcome mode) matches topple-label rebalancing; passes the gate on fresh panels 7 (82/9) and 6 (85/11) | phase1e_result, phase1e_panel{7,6}_result |
| Object-free start sampler (Gauss: latent noise along fitted directions) replaces the 3-block pose operator G with no loss (0.709 vs 0.672); learned G* does not (0.584) | phase2_p0_result |
| Phase 2 (rows/stacks, H2): the persistent split test is task-tuned: on REAL futures it catches only 19/78 forks (hold-10 and hold-30 groups differ in multi-factor scenes); hold-30-only catches 69/78 (30% false). US2 also underfits (loss 2x US, spread 0.3x real even from exact starts) | phase2_h2_result, amendment_4 |
| DO NOT RETRY AS IS: selecting training states by the image split test (Phase 1f) fails (score 0.546 vs 0.672; contact forks 17/40 vs 28/40): 83% of image splits involve no topple (target, sliding, arm), so harm forks become rare in the data. A label-free screen would need to ignore the target/arm or target harm-like splits; separate protocol if ever | phase1f_result |

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

## Queued (agreed with the user 2026-10-06)

Status: Phase 1 cell E 69% / 8% on H; Phase 1b plateau (a1 72% / 15%, a2 68% / 13%; more fork data stopped
helping); Phase 1c running (conv3 = E + 3-frame history; vit3 = DINO-WM ViT predictor design, from scratch,
3-frame history). All image-latent models so far get executed actions only, NO proprioception (the arm is
visible only in the image latent; the original DINO-WM checkpoint took a 10-D proprio input).

1. **Phase 1d: proprioception input.** Add the robot's proprioceptive state (joint / end-effector position
   and velocity at each step) as an explicit input next to the actions, on the best Phase-1c arm. Allowed
   under the universal-monitor rule. Rationale: arm velocity at contact may decide a topple.
2. **Pretrained V-JEPA 2-AC fine-tuning** (user request: "lets also do a vjepa 2 ac (already pretrained)
   fine tuning on our data including procep"). The real pretrained weights (V-JEPA 2 ViT-g encoder + AC
   predictor), fine-tuned in the initial-condition recipe (perturbed starts, fork-rich data, unrolled
   rollout loss) with actions AND proprioception mapped to its action/state tokens. Needs: re-rendering
   the training futures and H at its resolution, encoding with ViT-g, its own perturbed starts (perception
   error drawn in its token space), and the Phase-1 evaluation on H. Known risk: V-JEPA 2's encoder
   localises blocks ~3x worse than DINOv2 (11.8 vs 4.1 mm, NOTES 2026-10-01). Own frozen protocol; run
   after Phase 1c; estimated 2-3 days of compute.
3. **DINO-WM option 2** (fine-tune the user's own checkpoint, default camera, full tokens, with proprio):
   needs the user's explicit approval after the vit3 result. Shares the re-rendering step with item 2.

## Later roadmap (agreed with the user 2026-10-07; NOT the current research goal)

Gate: start only after DETECTION is good (a learned monitor that passes the fork gate on a fresh panel).

1. **Safety filter (detect + repair, task-preserving).** Leave the policy alone unless the probe alarms; then search
   candidate variants of the intended chunk (small shifts of the approach path, slower execution, grasp offset or
   approach angle), run the probe on each, and pick the closest candidate whose 64 futures are **unimodal** (user's
   choice of criterion; goal-free, no failure labels). Fallbacks: slow down, retract, **look again** (a second view
   shrinks perception uncertainty, which can dissolve the fork). Optionally guide a diffusion policy's sampling with
   the fork score. Related: latent safety filters (Nakamura et al. 2025; Seo et al. 2025), which need failure labels.
2. **Real-world data for the monitor's world model (unsupervised, safe, generalising).** Safe envelope first
   (compliant control, speed/force limits, stop on force spikes, light/foam objects, fenced workspace); self-resetting
   play with repeated execution of the same chunk from re-made scenes (real outcome distributions, no labels);
   fork-seeking exploration where the model is unsure whether futures split; randomised objects, layouts, lighting
   and camera; pretrain on existing robot data (DROID, Open-X) and simulation/real-to-sim, fine-tune on real data.

## Standing rules
Frozen protocols before data; fresh panels for confirmation; select on branching fidelity; report the
always-stays / always-alarm baselines next to every agreement number.
