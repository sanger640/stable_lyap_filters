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

## W1 — data and loss: identifiability and spread

Not expected to produce the jump on its own; every later phase needs it anyway.

1. **K=8 perturbations per simulator state** (`jenga_gtf_data.py --probes-per-chunk 8`, grouped by
   snapshot). With one action per state, "the action caused this" and "this state usually looks
   like that" are not distinguishable.
2. **Difference matching:** `mean over pairs ||(z_i^ - z_j^) - (z_i - z_j)||^2`, the
   intervention-effect term. This is the geometry the monitor reads.
3. **Energy score over the predicted set:**
   `(1/K^2) sum_ij ||z_i^ - z_j|| - (1/2K^2) sum_ij ||z_i^ - z_j^||`, norms not squared. The
   negative term rewards spread, so collapse is penalised. Proper scoring rule: minimised by
   matching the distribution, not the mean.
4. **Reweight the hold steps.** 30 of 38 steps are a held pose where almost nothing moves, so 79%
   of the loss terms currently teach stasis. Weight by true change magnitude or subsample.
5. **Delta targets**, so "nothing happens" stops being free.

**Gate:** predicted spread within 2x of real on the W0 benchmark, with zero-action drift near
zero. Jump ratio is a bonus, not required.

## W2 — discrete output head (the phase aimed at the discontinuity)

Replace continuous latent regression with a categorical prediction:

1. VQ the ending latents into a codebook (a few hundred codes, learned unsupervised from the
   training endings; the count is not supplied by hand).
2. Train the predictor to output a **distribution over codes**, cross-entropy against the true
   code, keeping the W1 terms for the continuous part if it helps.
3. At a fork the distribution goes bimodal and per-probe predictions land on one code or the
   other, so the response is discontinuous by construction.

The monitor then groups code identities instead of continuous endings -- close to the original
basin idea, but learned per model rather than as a global atlas, and still label-free.

**Gate:** jump ratio >= 10x on the W0 benchmark, then Stage 3 rerun unchanged
(`eval/jenga_stage3_predicted_forks.py`), with recall and false alarms compared against the real-
ending numbers on the same states.

## W3 — object-centric prediction target

Only if W2 produces the jump in the wrong place, or produces none. The hypothesis then is that the
bottleneck is representation: a toppling neighbour is a small part of a 224x224 frame, so blurring
it costs almost nothing, while in an object-slot state it is a large fraction. Predict slot states
(SOLD/SlotContrast-style, slots discovered unsupervised so universality holds) instead of 196
patch tokens.

**Gate:** same as W2.

## W4 — structural fallback

A learned rigid-body simulator over object state (FIGNet-style face-interaction graph network,
built for exactly this contact discontinuity) with a perception front end. Highest accuracy on
contact, but it gives up "pixels only" and needs object meshes, so it changes what the
contribution is. Decide deliberately, do not drift into it.

---

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
