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

## W2 — discrete output space (done)

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

## W3 — object-centric perception as a fix for a one-shot model: superseded by W5

Object tokens are NOT abandoned: they return as W5's deployable input (step 7). What this section
rules out is object-centric perception bolted onto a ONE-SHOT ending predictor as the fix.

The privileged probe fed true block poses, gripper pose, relative geometry, velocities and contact
flags (70 dims) at 10x data. Its fork/quiet jump contrast is 0.88-0.99: camera, blocks-only state
and perfect physics state all fail identically, so better perception is not the binding constraint.

**Scope of that claim.** What is ruled out is a ONE-SHOT map from state to ending, whatever its
input -- three dense layers over a flattened action window, predicting an ending code directly.
That architecture has no integration, therefore no attractors, and must fit a near-discontinuous
function in a single step, while a topple is a process: contact, tipping, past the balance point.
The autoregressive DINO-WM's within-curve jump ratio (5.0) was higher than either one-shot head's
(~2.4), which points the same way.

## W5 — stochastic hybrid world model (current phase)

This phase follows the external plan's Phase 3 ("branch-aware stochastic hybrid world model")
directly. Where this document adds something, it says so.

### Why hybrid

Contact mechanics is a hybrid system: a continuous state (positions, velocities) that flows
smoothly within a mode, and a discrete mode (no contact, sticking, sliding, tipping, toppled) that
switches. Within a mode the dynamics are smooth; at a switch they jump. Every model trained so far
was a single smooth function asked to fake a switch with a steep slope, and all of them ramp across
the boundary instead of jumping. A hybrid model switches between smooth experts instead:

| hybrid-system concept | component |
|---|---|
| the modes | K local dynamics experts, each smooth within one regime |
| which mode applies (the guard) | action-conditioned gate pi_k(s_t, a_t) |
| the mode variable | latent discrete regime / transition variable |
| what triggers a switch | contact / event features |
| an impact's instantaneous jump (reset map) | optional explicit event/reset map, separate ablation |

A small action change can flip the gate, hence a different expert, hence a different trajectory.

### Universality boundary (decided 2026-09-18)

The runtime monitor stays universal: perturb with measured execution error, roll out, test whether
the endings split or spread; no failure labels and no task knowledge. The world model is trained
per deployment on unlabelled interaction data, which is learning how things move, not what failure
is. Within that:

* **Allowed in the deployable model:** object tokens discovered from images WITHOUT labels (slots),
  with identity tracking and uncertainty/occlusion flags. This assumes the world is made of
  objects, not what they are or what failure means.
* **Not allowed in the deployable model:** simulator state, hand-named contacts ("middle-left",
  "neighbour blocks"), or anything that names a failure. These appear only in the explicitly
  marked ORACLE variant, whose job is to answer whether the architecture can represent the
  discontinuity at all when perception is perfect.
* **Transition weighting** (below) is defined on the change in the model's OWN state, not on any
  task quantity, so it is a generic prior, not task knowledge.

### Prediction

Step-wise, per control step: a distribution over next object-token states, rolled out over the
H=8 chunk and the 30-step hold, supporting sampled multi-step trajectories and branch
probabilities. Expert indices are NOT read as physical modes without a post-hoc audit.

### Build order

1. **Oracle, single-expert stochastic baseline -- DONE (2026-09-18).** Graph net over 3 block
   tokens and 1 gripper token from simulator state, Gaussian head, teacher forcing then a
   4 -> 12 -> 38-step rollout curriculum (`src/state_dynamics.py`, `eval/jenga_w5_train.py`,
   `eval/jenga_w5_eval.py`). Rollout error 5.70 -> 0.38; fork/quiet jump contrast 0.84 (reality
   4.17); spread contrast 4.52; spread AUC .90 (1x) and .98 (2x); predicts a NEW topple at 0/16
   fork states at 1x and 1/50 at 2x. With full oracle state the true dynamics are deterministic,
   so this failure is not stochastic averaging: probes a millimetre apart with opposite outcomes
   look almost identical to a smooth network, which interpolates across them.
2. **Gate 3 number for that baseline -- DONE (2026-09-18).** Spread score thresholded on SAFE
   controls only (95th percentile of batch-1 quiet states), evaluated once on batch 2, episode-
   clustered intervals (`eval/jenga_w5_gate3.py`): **19% recall [0-40] at 4% false alarms at 1x**;
   38% [23-54] at 0% at 2x. The same procedure on REAL endings gives 88% [69-100] at 0% (1x) and
   98% at 8% (2x). This is the number every later variant must beat.
3. **Oracle, deterministic baseline** at matched compute, per the ablation table.
4. **Data (additions of this document, labelled as such):**
   * *Finer timestep.* Actions run at 10 Hz with 50 physics substeps each, so a topple onset
     happens INSIDE one recorded step. Record state every 10 substeps as well.
   * *Transition weighting.* 87.4% of recorded steps move a neighbour < 0.1 mm and 0.45% move it
     > 5 mm, so an unweighted loss learns stillness and a gate collapses onto the "nothing
     happens" expert. Weight each step by the magnitude of change in the model's own state (never
     by a task quantity), with the rule fixed before training.
5. **Oracle MoE.** K = 2-4 experts, action-conditioned gate pi_k(s_t, a_t) with contact/event
   features as input, discrete regime latent with hard switching (straight-through Gumbel-softmax,
   annealed temperature), regularisation against expert collapse (load balancing). Grow K only if
   held-out likelihood and branch coverage improve.
6. **Reset-map ablation,** only if measured impact errors justify it, never as a hidden dependency.
7. **The universal model:** the winning architecture on image-derived object tokens discovered
   without labels (frozen DINO backbone, slots with identity tracking and occlusion flags). The
   oracle result says whether this is worth building; this step is the actual claim.

### Losses

* Multi-step proper predictive loss (NLL or a calibrated sample-based score).
* Object pose, velocity and contact terms (contact supervised in the oracle variant only).
* Branch preservation on matched counterfactual pairs over rollout samples paired by common random
  numbers: predicted distance between branches i and j should match the real distance.
* Regime-probability calibration.
* Divergent AND non-divergent pairs, distances normalised by physical scale, large outcomes capped,
  branch weight chosen on development data only.

### Monitored during training

One-step likelihood, rollout likelihood, branch calibration, safe-pair false separation, and expert
usage (collapse or overdispersion).

### Gate 3

Advance only if a variant improves held-out persistent-branch recall at the fixed safe-control
false-alarm budget (5% per stratum, episode-clustered 95% intervals) over the best simpler
baseline, without losing ordinary-motion fidelity. **A stochastic model that emits both futures
everywhere is not a success.** If the simple stochastic model matches the MoE, prefer it and report
that switching was unnecessary. The W0 response curves are reported alongside as a diagnostic.

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

* The oracle MoE (W5 step 5) does not beat the oracle baselines on Gate 3 -> switching does not
  help even with perfect state; do not build the image-token model (step 7). Write up the
  simulator-based monitor as the result and the predicted-ending path as the documented barrier.
* The oracle MoE passes but the image-token model fails -> the barrier is unsupervised perception
  of contact-relevant state; report that gap.
* A variant passes only by emitting both futures everywhere (safe-pair false separation rising,
  quiet-state alarms above budget) -> rejected, per Gate 3.

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
