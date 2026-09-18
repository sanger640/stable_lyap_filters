# PLAN — make the world model reproduce outcome discontinuities

**Mode:** phase by phase, same rule as the other plans. Do not start a phase until the previous
one's gate is met and logged in `NOTES.md`. **If a phase fails twice, stop and report.**

Background: [`JENGA_EXPERIMENT.md`](JENGA_EXPERIMENT.md) (what works and what does not),
[`HANDOFF.md`](HANDOFF.md) (state and commands), `NOTES.md` (the append-only log).

---

## The problem, stated precisely

The fork monitor works on real endings (88% recall at 1% false alarms with a shared hold, 75% at
3% with the arm moving, on held-out episodes) and produces **no signal at all** on world-model
predicted endings. The cause is measured, not assumed:

| | spread of the 64 predicted endings | gap between toppling and stable | gap/spread |
|---|---|---|---|
| real endings | 54.4 | 138.9 | 3.17 |
| shipped checkpoint | 47.0 | 22.5 | 0.48 |
| rollout fine-tuned | 0.6 | 0.5 | 0.84 |

Rollout fine-tuning cut validation rollout MSE 10x (2.095 -> 0.207) and moved nothing that matters.

The action-response measurement says why. Pushing the whole chunk sideways by a fixed offset and
comparing the settled ending:

| offset | real block shift | shipped predicted change | fine-tuned predicted change |
|---|---|---|---|
| 1.6 mm | 1.2 mm | 17.9 | 0.53 |
| 5 mm | **52 mm (a topple)** | 32.8 | 1.67 |
| 20 mm | 55 mm | 90.0 | 5.59 |
| 50 mm | 87 mm | 98.2 | 9.78 |

**The models are not action-blind; they are action-smooth.** Reality jumps 40x between 1.6 mm and
5 mm. The models ramp by less than 2x and 3x. They interpolate across the outcome boundary.

**Why no amount of training fixes this in the current architecture.** A deterministic network from
actions to endings is a continuous function. Reproducing a jump requires either an enormous local
slope -- unstable, and it smears anyway because neighbouring training states disagree about where
the boundary is -- or **discrete structure that switches**. Three mechanisms can produce a
discontinuity: a discrete latent whose argmax flips, a latent variable plus sampling, or a
structured state where contact logic creates the jump. Continuous regression is the one option
that cannot.

**Do not** grade any phase below by rollout MSE. It fell 10x while the monitor stayed dead.

---

## W0 — the scoreboard (gate for everything else)

`eval/jenga_w0_response_curves.py`. A frozen benchmark of response curves: ~40 states spanning
fork and quiet, several push directions, signed offsets from -50 to +50 mm. For each state,
direction and offset, record the simulator's settled block positions and each model's predicted
ending latent.

Two label-free metrics, both scale-free so models with different latent scales compare:

* **spread** -- median pairwise distance between endings across the offset grid, reported relative
  to the same quantity on real endings;
* **jump ratio** -- largest slope between adjacent offsets divided by the median slope. Reality
  gives large values at a fork; a smooth ramp gives ~1.

**Acceptance:** the benchmark runs, is committed with its numbers, and reproduces the shipped and
fine-tuned baselines above. This is the scoreboard for W1-W3. No model changes in this phase.

## W1 — data and loss (DONE, FAILED)

`eval/jenga_gtf_data.py --probes-per-chunk 8 --chunk-stride 2` (453 states x K=8 perturbations
from the same snapshot) and `eval/jenga_w1_train.py` (paired MSE + energy score + difference
matching + temporal difference, steps weighted by true scene change). Every loss term improved on
held-out states; the response curves got worse (fork jump ratio 1.5, fork/quiet spread 0.96).

Kept for W2 and beyond: **the counterfactual dataset and the branch-preservation losses carry
forward unchanged.** Dropped: paired MSE at weight 1.0, which is the mean-seeking term.

**Conclusion: loss alone is insufficient.** A deterministic continuous map from actions to endings
cannot produce a jump, so W2 changes the output space and the loss together.

## W2 — discrete output space (current phase)

Three mechanisms can produce a discontinuity: a discrete latent whose argmax flips, a latent
variable plus sampling, or structured state where contact logic creates the jump. W2 takes the
first, in the smallest form that tests it.

**W2a — discrete ending head.** Quantise the settled endings into a codebook learned unsupervised
on training endings only (PCA then k-means; the count is not supplied by hand). Predict a
*distribution over codes* for the ending at hold steps 10 and 30, conditioned on the three real
history latents and the whole action window. Cross-entropy against the true code, plus the W1
branch terms over the distribution's mean embedding. The action window is rescaled relative to the
nominal chunk and by the execution-error size, so a 1.6 mm difference arrives as an O(1) input
instead of 0.05.

This is a temporally abstract (jumpy) world model: it predicts the ending the monitor reads, not
every intermediate frame. Justified because the monitor only ever reads endings, and it isolates
the question -- can a discrete output over actions reproduce the jump? -- from rollout drift.

**W2b — switching dynamics, only if W2a jumps.** Put the same discrete bottleneck back into the
autoregressive predictor, or an action-conditioned mixture-of-experts gate (K=2-4) over local
dynamics. Control: a stochastic single-expert model at matched compute. Adopt only if it beats
that control.

**Gate (W2a and W2b):** on the W0 benchmark, fork-state jump ratio at least 3x the same model's
quiet-state jump ratio (reality: 14.9 vs 3.6 = 4.1x), and fork/quiet spread ratio above 1.5
(reality 1.81). Then Stage 3 rerun unchanged, graded against the real-ending numbers on the same
states.

## W2 — outcome (2026-09-18): jumps, but never where the boundary is

W2a's discrete head produces real discontinuities (4-5 code switches per response curve, within-
curve jump ratio ~2.4 against W1's 1.5), but the fork/quiet jump contrast stays at ~1.0 against
reality's 4.14 at 453, 881 and 8,810 states. Doubling the data moved the SPREAD contrast to 1.78
(reality 1.81, that half of the gate passed) and left sharpness flat, so magnitude was partly
data-limited and localisation is not.

End to end with the full privileged state (`eval/jenga_w3_monitor.py`): the frozen fork rules give
0-34% recall at 7-30% false alarms against 88%/1% on real endings. Ranking states by predicted
ending spread separates fork from quiet at AUC .93 (2x) and .80 (1x) -- real signal, but a ranking
needs a threshold, which costs the calibration-free claim.

## W3 — object-centric prediction target: NOT JUSTIFIED, do not build

The privileged probe fed true block poses, gripper pose, relative geometry, velocities and contact
flags (70 dims) at 10x data. Its fork/quiet jump contrast is 0.88-0.99: camera, blocks-only state
and perfect physics state all fail identically, so better perception is not the binding constraint.

**Scope of that claim.** What is ruled out is a ONE-SHOT map from state to ending, whatever its
input -- three dense layers over a flattened action window, predicting an ending code directly.
That architecture has no integration, therefore no attractors, and must fit a near-discontinuous
function in a single step, while a topple is a process: contact, tipping, past the balance point.
The autoregressive DINO-WM's within-curve jump ratio (5.0) was higher than either one-shot head's
(~2.4), which points the same way.

## W5 — step-wise dynamics model on privileged state (current phase)

This is the other agent's Phase 3 minimum implementation, run oracle-state-first (which that plan
explicitly permits as a marked oracle variant) so perception is not a confound:

1. **Step function** over 4 nodes (3 blocks + gripper): pose, velocity, contact flags and the
   current action -> change in block state, applied recurrently for the H=8 chunk plus the 30-step
   hold. The outcome is read by INTEGRATING, never predicted in one shot.
2. **Order, from that plan:** a single-expert stochastic baseline first, then an action-conditioned
   gate over K=2-4 local experts at matched compute, adopted only if it beats the baseline.
3. **Losses:** pose/velocity regression plus contact classification, a multi-step rollout term, and
   the W1 branch-preservation terms over the 8 probes of each state (non-divergent pairs included).
4. **Training order:** teacher-forced first, then rollout fine-tuned -- reversing this collapsed the
   toy model to "nothing ever happens" (HANDOFF finding 1).

Data: `eval/jenga_state_data.py --per-step` records the state after every action, ~650 MB and ~11
minutes because nothing is rendered; the existing 70,480 rollouts hold ~2.7M transitions.

**Gate:** the W0 response curves first (fork jump ratio >= 3x the same model's quiet value), then
`eval/jenga_w3_monitor.py` end to end against the real-ending reference of 88% recall at 1% false
alarms. The signature that justifies it
is a model that knows a boundary exists but cannot localise it from patch features. Predict object
tokens (slots discovered unsupervised, so universality holds) or an interaction network over them,
instead of 196 patch tokens.

Note the evidence ranking: the ending representation is NOT the demonstrated problem -- real DINO
endings separate outcomes at d' 4.85 and give 88% recall at 1% false alarms. W3 is a hypothesis
about the dynamics being easier over objects, so it comes after W2, not before.

## W4 — structural fallback

A learned rigid-body simulator over object state (FIGNet-style face-interaction graph network)
with a perception front end. Highest accuracy on contact, but it gives up "pixels only" and needs
object meshes, so it changes what the contribution is. Decide deliberately.

## Evaluation rules adopted from the external review (2026-09-18)

An independently written plan (`plan_from_another_agent.md`, assessed 2026-09-18) covered ground
Stages 0-3 already settled -- its Phases 0-2 map onto the answer key, the physics upper bound, the
real-DINO result and the predicted-ending failure, and its branch-margin search was already built
and rejected three times here. Its Gate 2 ("real DINO works, predicted futures fail -> dynamics
training is justified") routes to this plan. Six of its requirements are adopted:

1. **Per-stratum false-alarm budget of 5%**, on moving non-topple and contacting-pick controls
   separately, replacing the earlier ~10% overall figure.
2. **Episode-clustered confidence intervals.** Holdout states share episodes, so the Wilson
   intervals reported for Stage 2 are too narrow; recompute with a cluster bootstrap by episode.
3. **A held-out geometry or friction setting** as an OOD test before any generality claim.
4. **Safe-pair false separation** tracked during training: the branch loss must not win by
   exaggerating every perturbation.
5. **Abstention as its own output**, counted with its own burden, plus a p50/p95 latency profile.
6. **Prevalence-weighted reporting** alongside the stratified numbers.

Not adopted: re-running Phases 0-2, the branch-margin machinery, and the full ablation matrix
before the binding constraint is fixed.

## Kill criteria, fixed now

* W1 restores spread but W2's jump ratio stays under ~5x -> the dynamics model is not the
  bottleneck; go to W3.
* W2 clears the jump gate but Stage 3 recall stays under ~40% -> the split is real but mislocated;
  go to W3.
* Neither W2 nor W3 clears it -> stop. Write up the simulator-based monitor as the result and the
  predicted-ending path as the documented barrier. The holdout numbers already stand on their own.

## References that informed this plan

* Action-insensitivity as a named failure mode, with counterfactual-consistency training and
  action-response metrics: arXiv 2608.04653; intervention-effect supervision: arXiv 2608.24885.
* Proper scoring rules for training generative forecasters (energy score / CRPS), and an
  operational weather model trained on CRPS that keeps realistic ensemble variability:
  npj AI s44387-026-00073-7, arXiv 2504.01781.
* Latent diffusion world model for manipulation built on DINO features: arXiv 2505.11528.
* Object-centric latent dynamics for manipulation: SOLD (ICML 2025), SlotContrast (CVPR 2025).
* Learned rigid-body contact dynamics: FIGNet (arXiv 2212.03574); graph simulators learn
  discontinuous rigid contact (Allen et al., CoRL 2022).
* Neural ODEs cannot uniformly approximate discontinuous mode changes, and event-based variants
  are fragile -- the reason PINN/Neural-ODE routes are not in this plan (arXiv 2512.10117).
