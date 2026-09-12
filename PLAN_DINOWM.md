# PLAN — the monitor on DINO-WM, run on the tipping block

**Mode:** phase by phase, same rule as PLAN.md. Do not start a phase until the previous one's
acceptance criteria are met and logged in NOTES.md. **If a phase fails twice, stop and report** —
do not loosen the criteria.

**Environment:** run everything in the `dino_wm` conda env (`/home/sanger/miniforge3/envs/dino_wm`,
torch 2.11+cu128). `tipping_block.py` is pure numpy and runs there unchanged.

---

## STATUS — all phases complete (2026-09-12)

| phase | verdict | key number |
|---|---|---|
| A renderer | **PASS** | ±θ differ 3.4–22.7 grey levels; 10.7 ms/frame |
| B ω recoverable | **PASS** | R² 1.000 (θ) / 0.955 (ω) at stride 10 |
| — latent geometry | **PASS** | pose/lighting 1.58; nearest-centroid 100% on unseen lighting |
| C predictor | **FAIL** on dino_wm's recipe · **PASS** retrained | 75% → **91.7%** basin agreement; RMSE 0.320 → **0.206** |
| D settle tail | **qualified FAIL** | never converges (0.60% floor); expansive (1.299) |
| E discovery | **FAIL** raw · **PASS** after PCA | separation 1.35 → 2.2–11.6; 93.3% agreement |
| F monitor | **PASS** | FP floor 96–97%; **AUC 0.882** vs shPLRNN 0.808 |

**Headline:** the monitor transfers from an exact 2-D state to DINOv2 patch features with **no
retuning** — same ε, n, k_min, probe family — and matches or beats the state-based version while
being scored ~4× more sparsely.

**Two caveats that must travel with that.** The run used the ROLLOUT FINE-TUNED predictor, not
dino_wm's shipped `num_pred: 1` recipe, so it validates the architecture and not the existing Jenga
checkpoint. And the AUC CI is [0.805, 0.949] against the shPLRNN's 0.808 point estimate, with
unpaired runs — so "at least as good, plausibly better", never "better".

**Phase D deserves a second look before Jenga.** The settle tail neither converges to a fixed point
nor contracts, yet Phases E and F worked anyway. That is not understood, and the ending latents
being still in motion when read is a latent fragility rather than a solved problem.

---

## Objective

Swap the world model underneath the existing monitor — shPLRNN on `(theta, omega)` becomes DINO-WM
on rendered images — while holding the system, the actions, the margin oracle, the monitor and
every hyperparameter fixed. Any difference in the result is then attributable to the
representation.

**Why this and not Jenga first.** On the block the true margin is computable by bisection, so a
weak result can be localised. On Jenga it cannot: a bad number could come from the representation,
the predictor, the clustering, or the labels, with no way to tell which.

**Read the outcome asymmetrically.** A FAILURE is highly informative — it localises the problem to
the representation before the Jenga rebuild is paid for. A SUCCESS is weak evidence, because the
toy's images are far simpler than a cluttered tabletop. Do not oversell a pass.

**Two shortcuts this closes.** The fixed-alpha block is easier than Jenga in two ways: (a) action
magnitude predicts the outcome (`sum|a|` -> AUC 0.925), and (b) the observation is Markov, so the
monitor's cold start is legitimate. **The image swap closes (b) for free** — a single frame shows
theta but NOT omega, so history becomes mandatory. Matched-impulse actions (separate work) close
(a).

**Reuse, do not reimplement.** Use `dino_wm/models/visual_world_model.py` (the single-view
`VWorldModel`) and `conf/train.yaml`. A clean-room reimplementation would test a different thing;
the point is to de-risk the actual Jenga code path.

---

## Phase A — Renderer

Rasterise the block at angle theta to a 224x224 RGB frame. Reuse the geometry from
`eval/phase3_live_demo.py:corners()`. Rasterise directly (PIL / numpy polygon fill) — matplotlib
is far too slow at this volume.

**Deliberately include the nuisance that breaks Jenga.** A black polygon on white risks degenerate,
low-variance DINOv2 features, which would make this test easier than the thing it is standing in
for. Add: a textured floor, a soft drop shadow cast by the block, and mild per-episode lighting
variation. The shadow is not decoration — shadow artifacts are the documented precision bottleneck
on Jenga (CLAUDE.md, Limitations 2), so rendering them puts that failure mode inside a system where
the ground truth is known.

**Acceptance**
- `+theta` and `-theta` render visibly differently (the pivot corner changes). Verify by eye on a
  contact sheet — if they are ambiguous the three attractors collapse to two and everything
  downstream is void.
- The three terminal states (`-pi/2`, `0`, `+pi/2`) are visually distinct.
- Throughput is sufficient to render the full corpus in under ~20 min.

---

## Phase B — Encode, and decide the sampling rate

Batch-encode frames with frozen `dinov2_vits14`, cache to LMDB or an npy memmap. Encode ONCE.

**The sampling rate is an open question, not a settled one.** NOTES records that coarsening
450 -> 45 steps was harmless (AUC 0.934 -> 0.971) — but that was measured with omega HANDED to the
model. From images, omega must come from differences between frames, so coarse sampling means large
inter-frame motion and possible aliasing during fast rocking near the topple. Sweep it; do not
assume the 10x saving survives.

**Acceptance — the cheap kill test.** Fit a LINEAR probe from `num_hist=3` frames of patch features
to `(theta, omega)`.
- theta: R^2 > 0.95 (it is directly visible; anything less means the render or the encoder is wrong)
- **omega: R^2 > 0.80.** This is the real gate. If angular velocity cannot be linearly decoded from
  three frames, the representation cannot support the dynamics and no predictor trained on it will
  either. **KILL the plan here** rather than discovering it after training.
- Pick the coarsest sampling rate that clears the omega bar. Log the sweep.

---

## Phase C — Train the predictor

Single-view `VWorldModel`: frozen DINOv2 -> ViT predictor + action -> `z_{t+1}`. `action_dim=1`
(scalar force). No proprio (feed a constant or disable). No decoder — nothing here needs pixels.

Same 600 training trajectories, same `random_push` distribution, same seeds as the shPLRNN run.

**Acceptance**
- Rollout fidelity at the eval horizon: decode theta (linear readout on patch features, fit on
  train) and compare to truth. **RMSE <= 0.265 rad**, the shPLRNN's figure — the bar is parity, not
  superiority.
- Finite everywhere, no divergence over the full rollout+settle length.

---

## Phase D — THE GO/NO-GO: does it settle?

The attractor construction assumes that holding the action drives the latent to a fixed point. A
ViT rolled autoregressively may blur or drift instead of converging. **Nobody has checked this, on
the block or on Jenga, and it gates everything downstream.**

Roll with zero force for the settle tail and measure `||z_{t+1} - z_t||` against the ending scale.

**Acceptance**
- >= 90% of rollouts reach `||z_{t+1} - z_t|| < 0.01 * scale` within the tail (the same criterion
  `phase3_eval100.py` already applies via `step < 0.01 * scale`).
- **KILL: if it does not settle, stop and report.** The basin construction is unavailable on ViT
  predictors, which is a finding about the method's applicability — and a far cheaper one to learn
  here than on Jenga.

---

## Phase E — Attractor discovery

Merge-distance sweep on the settled latents, exactly `find_attractors()` from
`eval/phase3_pipeline.py`. No k supplied.

**Acceptance**
- A plateau exists, and it is at **3**.
- Cross-check assignments against the true basins from `tb.simulate`. shPLRNN reached 87.0%
  agreement; require >= 80%.
- If the plateau is at 5 or 7, or absent, report it — unsupervised attractor discovery is then the
  fragile component, not the monitor, and that distinction matters for the paper.

---

## Phase F — Monitor and evaluation

Port `Monitor` to the DINO-WM latent. Everything else held fixed: `v = a`, `n = 32`, `eps = 0.10`,
`k_min = 2`, `H = 200` (scaled to the chosen sampling rate), continuous scoring, no latching.

**New piece: warm-start.** The cold start was legitimate on `(theta, omega)` and is not legitimate
on images. Before probing, teacher-force through the last `num_hist` observed frames — the inner
loop of `gtf_rollout_loss_latent` with the loss deleted — so the latent carries the history that
encodes omega.

**Run the FALSE-POSITIVE FLOOR test first.** (An earlier draft said "run the eps=0 null". That is
vacuous: the model is deterministic, so 32 probes with identical actions give 32 identical rollouts
and k=0 by construction. Keep eps=0 only as a plumbing check -- it would catch dropout left on at
inference -- never as a measurement.) The real test runs the monitor at the OPERATING eps on
episodes with a LARGE TRUE MARGIN, far enough from any boundary that no probe should be able to
cross one. Any `k > 0` there is model error manufacturing dissent, and that is the floor for the
whole method. If those episodes already reach `k >= 2`, the alarm rule is measuring the model
rather than the boundary.

Then the same 100 episodes, same seed (`np.random.default_rng(777)`), same margin oracle.

**Acceptance**
- false-positive floor: `k < 2` on >= 95% of scores from large-margin episodes.
- Report `k`-vs-margin AUC against the shPLRNN's 0.808, with a paired bootstrap CI. **Parity is
  the bar.** Do not expect to beat a model that sees the exact state.

---

## Phase G — The comparison

One table, one system, one set of labels, two representations, no retuning:

| | shPLRNN on (theta, omega) | DINO-WM on images |
|---|---|---|
| state | exact, Markov | inferred from 3 frames |
| false-positive floor (k, large margin) | ? | ? |
| theta RMSE | 0.265 rad | ? |
| settles? | yes | ? |
| attractor plateau | 3 | ? |
| AUC vs margin | 0.808 | ? |

Plus the baselines already measured on this system (`sum|a|` 0.700, `||a||` 0.764 with S* fitted to
the test labels, both proximity).

**What this buys the paper:** representation-level universality — the same procedure, unchanged,
over two very different state representations — obtainable without a second dynamical system. Pair
it with matched-impulse actions for system-level variation and both axes are covered on a toy where
everything remains exactly measurable.

---

## Risk register

| risk | caught by | cost if missed |
|---|---|---|
| DINOv2 features degenerate on synthetic images | Phase A nuisance rendering; Phase B probe | a pass that means nothing |
| omega not recoverable from frames | **Phase B linear probe** | weeks of training a model that cannot work |
| ViT does not settle under held action | **Phase D** | the entire basin construction, on Jenga too |
| undersampling breaks velocity estimation | Phase B sweep | silent accuracy loss blamed on the method |
| attractors do not plateau at 3 | Phase E | clustering blamed on the monitor |
| model error manufactures dissent | **Phase F false-positive floor** | `k` measuring the model, not the boundary |

The three bolded rows are the cheap kill tests. **Run B, D and the false-positive floor before investing in
anything else** — each can end the plan in under a day, and each would otherwise be discovered
only after the full Jenga rebuild.
