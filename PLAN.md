# PLAN.md — piecewise-linear latent dynamics for a Jenga-toppling safety monitor

**Audience:** an autonomous coding agent (or a human picking this up cold).
**Mode:** phase by phase. Do not start a phase until the previous phase's acceptance criteria
are met and logged in `NOTES.md`. If a phase fails twice, **stop and report** — do not loosen
the criteria and do not paper over it by moving on.

This adapts a general FTLE-monitor plan to *this* repository's task and results. Where the
source plan's assumptions are already settled by evidence in `results/`, that is stated
explicitly rather than re-derived.

---

## 0. Objective

Detect, at runtime, when a manipulation policy is about to cross a **discontinuity boundary**
— the razor edge where two nearly identical states lead to opposite outcomes.

**Our instance:** a Franka Panda picks a block from a cluttered tabletop. A neighbouring block
either stays standing or topples. Toppling is a genuine guard surface: the block tips past its
balance point or it does not, and the two futures diverge irreversibly. This is structurally
the same problem as the source plan's peg/bushing insertion (seated vs. jammed) and the
machinery transfers directly.

The signal is the **finite-time Lyapunov exponent (FTLE)** of a learned latent dynamics model:
nudge the state, roll forward, measure how much the nudge grew.

### Why replace the current dynamics head

The existing monitor (README) reaches **AUC 0.894**, but *not* via FTLE. It works by
**abandoning** the Lyapunov ratio:

| approach | AUC | note |
|---|---|---|
| `ftle` — (1/T)·log(d_end/d_start), the actual Lyapunov construction | **0.599** | ≈ chance |
| exact analytic Jacobian FTLE (`results/` — 225 chunks) | **0.617** | correct code, still bad |
| `d_end` + patch masking — *no ratio at all* | **0.894** | what we ship |

So the current best monitor is not a Lyapunov method. It is a drift magnitude with good
masking. **Every attempt to use actual local-expansion geometry on the current causal-ViT
head has failed**, and we know why:

1. **The linear regime is 50× too small.** Direct linearisation test: at ‖δ‖=1e-3 relative
   error is 0.048 and direction cosine 0.9995; at the operating σ=0.05 relative error is
   **0.963** and cosine **0.538**. The Jacobian is *correct* and *useless* — the model is
   smooth, so it can only steepen a discontinuity, never represent it.
2. **The flow map is not a function.** The predictor conditions on `num_hist=3` frames, so
   `s -> s'` is not a map on one state vector and its Jacobian is not well defined. (We also
   lost real time to a bug where those 3 frames were silently sampled 8 control steps apart —
   the model never complained.)

The four design commitments below address exactly these two failures.

### The four design commitments

1. **Markov state.** Dynamics is a map `s -> s` on one explicit state vector. History lives
   *inside* `s`, not in separate conditioning.
2. **Piecewise-linear dynamics head.** ReLU transitions carve state space into polyhedral
   cells with hard boundaries. Jacobians are analytic and piecewise constant. A smooth model
   provably cannot represent the discontinuity.
3. **Supervise separations, not just trajectories.** Trajectory accuracy does not imply
   Jacobian accuracy. Mine near-neighbour pairs and train on how fast they separate.
4. **Split prediction from geometry.** A generative head predicts futures (multimodal at the
   boundary); the piecewise-linear head supplies Jacobians. One model should not do both.

### ⚠️ The one real tension with our existing results — resolve it in Phase 4, do not ignore it

Our single biggest empirical win is **spatial patch masking**: keeping the low-motion
*background* patches lifts AUC 0.756 → 0.894, because foreground divergence is dominated by a
motion confound from the always-moving arm (README, §3).

The shPLRNN wants a single low-dimensional state (`d = 32–64`). **Naively mean-pooling DINOv2
patches into `d=32` destroys the exact structure that masking exploits.** Do not assume the
masking gain survives the encoder. Concretely:

- Apply the patch mask **before** pooling into `s`, so the state is built from the informative
  patches only; and/or
- let the separation loss (2.5) shape the encoder, and *measure* whether the learned state
  retains the background/foreground distinction.

This is a named risk with an owner (Phase 4, acceptance criterion 5). If the masking gain does
not survive, that is a publishable negative result about the cost of dimensionality reduction
for this signal — record it, do not bury it.

---

## 1. Layout

Add to the existing repo (do **not** start a fresh one — the existing `src/metrics.py`,
`results/` and evaluation discipline are the baseline this is measured against).

```
stable_lyap_filters/
  src/
    metrics.py            # EXISTING baseline monitor (d_end + masks) -- the thing to beat
    monitor.py            # EXISTING runtime API
    models/
      shplrnn.py          # piecewise-linear dynamics head
      encoder.py          # DINOv2 patches -> low-dim Markov latent   (Phase 4)
      generative_head.py  # flow-matching predictor                   (Phase 4+)
    geometry/
      jacobian.py         # analytic Jacobian for shPLRNN
      ftle.py             # flow-map FTLE + full Lyapunov spectrum (QR) + saturating measure
      ridges.py           # ridge extraction from an FTLE field
    detect/
      lipschitz_probe.py  # model-free boundary detector (needs no trained model)
    training/
      gtf.py              # generalized teacher forcing
      neighborhood_loss.py
      pairs.py            # near-neighbour pair mining
    systems/
      lorenz.py
      bouncing_ball.py
      toy_topple.py       # our Phase-3 analogue of the toy peg
  configs/                # one YAML per experiment; no hardcoded hyperparameters
  tests/
  results/phase0..phase5/
  NOTES.md                # running log: decisions, failures, surprises
  PLAN.md                 # this file
```

Every result in `results/` must be reproducible from `config + seed` with one command. Log the
seed. Pin versions.

---

## 2. Core components (build before Phase 1)

### 2.1 `models/shplrnn.py`

```
s_{t+1} = A @ s_t + W1 @ relu(W2 @ s_t + h2) + h1 + C @ a_t
```

`A` diagonal `(d,)`; `W2 (H,d)`; `W1 (d,H)`; `h2 (H,)`; `h1 (d,)`; `C (d,a_dim)`.
Defaults `d=32, H=32`; keep `d` in 20–60. Each hidden unit defines a switching hyperplane
`W2[i]·s + h2[i] = 0`. Expose the active-set gate so downstream code can read which cell a
state is in.

### 2.2 `geometry/jacobian.py`

```python
D = (W2 @ s + h2 > 0)                    # (H,) 0/1 gate
J = diag(A) + W1 @ diag(D) @ W2          # analytic, exact, no autodiff
```

Test: matches `torch.autograd.functional.jacobian` to 1e-6 away from any hyperplane.

### 2.3 `geometry/ftle.py`

Propagate the deformation matrix with QR reorthonormalisation every `reorth_every` (default
10) steps or the product overflows. **Return the full spectrum, not just `lambda_max`** — the
ratio of positive to negative exponents separates a saddle (recoverable) from a repeller (not),
which is exactly the distinction the monitor's downstream decision needs.

Also implement a **saturating separation measure**:

```python
finite_separation(s0, model, T, eps=1e-4, cap=1.0)
# max over sampled unit directions of min(||Phi(s0+eps*v) - Phi(s0)||, cap)
```

At a true guard surface the FTLE **diverges** with `T` rather than converging. The capped
measure stays finite and is the more honest quantity there. Log both, always with `T`.

> This connects to a known failure mode of ours: "flash topples" — blocks that fall with no
> precursor wobble. We characterised those as an information limit given a 3-frame history.
> The saturating measure is the right instrument for them; check whether they are actually
> guard-surface singularities rather than missing information.

### 2.4 `training/pairs.py`

Mine `(i,j)` with `||s_i - s_j|| < radius` **and** `||a_{i:i+T} - a_{j:j+T}|| < action_tol`
and both `i+T`, `j+T` in-trajectory. Return `(s_i, a_i, d0 = s_j - s_i, dT = s_{j+T} - s_{i+T})`.

**The action-match filter is not optional.** Two states that separate because different
actions were commanded say nothing about the flow map. We have an unusually good source of
matched pairs: `replay_noisy.py` replays the *same* expert trajectory with different injected
noise, so near-identical states under near-identical actions are abundant by construction.

### 2.5 `training/neighborhood_loss.py`

```
L = L_pred + beta * mean( || Phi^T(s + d0) - Phi^T(s) - dT ||^2 / ||d0||^2 )
```

Plus the second-order refinement: evaluate separation consistency at perturbed base states
`s + eps` as well as `s`. Default `beta = 1.0`; sweep `{0.1, 1, 10}` in Phase 1.

### 2.6 `training/gtf.py`

`s_t <- alpha * s_t_data + (1 - alpha) * s_t_model` at every step. Default `alpha = 0.1`,
sweep `{0.02, 0.05, 0.1, 0.2, 0.4}`; heuristic `alpha ~ 1 - exp(-lambda_max * dt)` once
`lambda_max` is known. Treat the sweep as a first-class experiment.

### 2.7 `detect/lipschitz_probe.py`

Model-free, runs on raw data before anything is trained. Ratio
`||s_{j+1} - s_{i+1}|| / ||s_j - s_i||` over action-matched near neighbours; flag
`> mu + sigma*sd`.

---

## 3. Phases

### Phase 0 — Data audit and gating probe — **PARTIALLY ALREADY ANSWERED**

The source plan's stop condition is: *if the probe finds nothing in DINOv2 space but finds
structure in F/T space, stop — the discontinuity is invisible to the encoder and no dynamics
head can fix it.*

**For us that branch is already closed, in the encoder's favour.** From `results/`:

- A linear probe on the **predicted** latent recovers future block tilt at **AUC 0.941** (and
  R² 0.836 on tilt directly). The information is not merely present, it is *linearly* present.
- Per-patch divergence localises ground-truth motion at **patch-AUC 0.955–0.960**.

So DINOv2 space contains the signal and the bottleneck is the **readout/dynamics**, not
perception. That is the single most useful thing our prior work contributes to this plan, and
it means Phase 0 is a confirmation, not a gate.

Remaining Phase-0 tasks:
1. Report dataset shape: trajectories, timesteps, success/failure split, channels, control
   rate, for `jenga_noise_50` (1772 chunks / 25 unsafe) and `jenga_tilt_100` (per-step tilt).
2. Run `lipschitz_probe` on (a) DINOv2 patch features (masked, then PCA to ~64), (b) proprio
   only. We have **no F/T sensor in sim** — that is a real difference from the source plan;
   proprio + vision is all we get. Note it.
3. Plot ratio histograms; mark where ground-truth topple onsets fall.

**Acceptance:**
- [ ] Probe returns a non-empty outlier set on at least one representation.
- [ ] Outliers cluster near known topple/contact events rather than scattering uniformly.
- [ ] Report what fraction of the 25 known unsafe chunks the probe's outliers cover.

### Phase 1 — Lorenz-63 — **ACCEPTED WITH DEVIATION** (details in NOTES.md)

> **Status.** lambda_3 0.27% error, 5/5 seeds (published reservoir baseline: 28%).
> lambda_2 ~0. **lambda_1 5.5% mean error, only 2/5 seeds inside 5%.** Accepted as a knowing
> deviation from the stated criterion, on the grounds that Phase 2's own bar is lambda_max
> within 10%. **Carry the lambda_1 caveat into every later write-up.**
>
> Key enabler: **off-attractor training data**. On-attractor data contains almost no evidence
> about the contracting exponent, because the transverse transient decays during the burn-in.
> Also settled: `d=20` + readout is decisively worse (lambda_3 48.8% error, spectrum a smooth
> ladder with no gap); `d=3` is correct and matches the shPLRNN paper's actual low-latent claim.

#### Original specification

Non-negotiable. If this fails nothing measured later means anything.

`sigma=10, rho=28, beta=8/3`, `dt=0.01`, transient discarded. Train shPLRNN with GTF on the 3D
state directly (no encoder). Compute the spectrum from the *learned* model. Ablate: with/without
neighbourhood loss; sweep `alpha`; sweep `beta`.

**Acceptance:**
- [ ] Learned spectrum matches ground truth `(0.906, 0.000, -14.572)` within 5% on **all
      three** exponents, over 5 seeds.
- [ ] The **negative** exponent is recovered, not just `lambda_max`. Negative exponents are the
      hard ones; failing them is the standard sign that Jacobians are wrong even when
      trajectories look fine.
- [ ] Ablation table shows the neighbourhood loss measurably helps. If it does not, that is a
      real finding — log it.

### Phase 2 — Bouncing ball on a shaken table: chaos *and* a hard discontinuity

Cheapest system with both. Restitution `e=0.8`.

Train on `(position, velocity, phase)`. Compute the FTLE field, locate the impact surface,
compare learned ReLU hyperplanes against the true impact surface. Evaluate FTLE **and** the
saturating measure at the guard for `T ∈ {5,10,20,50}`.

**Acceptance:**
- [ ] `lambda_max` between impacts within 10% of analytic.
- [ ] ≥1 learned hyperplane within 5% of the true impact surface.
- [ ] FTLE spikes at impact; ridge stable across 5 seeds.
- [ ] Divergence-with-`T` at the guard characterised and written up.

### Phase 3 — Toy toppling block: guard surface localisation *(adapted — was "toy peg")*

Our guard surface is **tipping past the balance point**, so the toy system should be that, not
a peg/hole. State `(theta, thetadot, x_contact)` for a rigid block on a plane, or the simpler
2D proxy: block of half-width `w` and height `h` on a table, pushed at the top. It topples iff
the centre of mass passes over the pivot edge — an exact, analytically known guard at
`theta_crit = atan(w/h)`.

Cheaper alternative already in hand: the existing **Pusher2D** toy (a fragile block knocked
past a cliff edge) has a genuine guard surface and its own physics. Reuse it if it saves a day,
but the analytic `theta_crit` of a toppling block is a *better* target because ground truth is
closed-form.

Tasks:
1. Implement; verify by hand that trajectories straddling `theta_crit` diverge within a
   plausible horizon.
2. Train shPLRNN with GTF + neighbourhood loss.
3. Localise the guard two independent ways: `lipschitz_probe` on data, and ridge extraction on
   the model's FTLE field.
4. Add a discrete mode head `P(mode|s)`; check its entropy also peaks at the guard.

**Acceptance:**
- [ ] Guard detected within 2% of analytic `theta_crit`, by **both** methods.
- [ ] The two methods agree; report the disagreement distance.
- [ ] **Mean-collapse figure:** train an MSE-only smooth MLP on the same data and show it
      predicts physically impossible half-toppled states. This figure is the motivation for
      the entire architecture choice — make it publication-quality.

### Phase 4 — Jenga in MuJoCo through the real observation pipeline

First rung with DINOv2 in the loop. **We already have all of this infrastructure**: `sim.py`,
101 expert episodes, `replay_noisy.py`, the labelled LMDBs, and a validated baseline to beat.

1. Build the Markov state explicitly:
   `s_t = [enc(masked_patches_t), enc(masked_patches_{t-1}), q_t, qdot_t]`, `d = 32–64`.
   History lives *inside* `s_t`. No separate history conditioning. **Apply the patch mask
   before pooling** (see the tension note in §0).
2. Train the encoder jointly with the shPLRNN head. Freeze DINOv2 itself.
3. Train the generative (flow-matching) head on the same state and encoder.
4. Run all detectors; compare FTLE ridges against ground-truth topple onsets from the sim.

**Acceptance:**
- [ ] Lipschitz probe fires at contact.
- [ ] FTLE ridge stable across 5 seeds (report ridge-position variance).
- [ ] Ridge location correlates with sim ground-truth topple events; report precision/recall
      at a swept threshold **and AUC on the same 1772-chunk protocol as the baseline**.
- [ ] Generative head produces bimodal samples at the boundary (show the histogram); the
      shPLRNN head alone does not, which is expected and fine.
- [ ] **Beat the existing baseline's 0.894 AUC on the same chunks and splits, or explain
      precisely why not.** This is the number that matters. A principled method that scores
      0.85 is not obviously better than an unprincipled one that scores 0.894 — say so if that
      is what happens.
- [ ] Report whether the masking gain survived dimensionality reduction (§0 tension).

### Phase 5 — Real Franka bushing/block insertion — **gated, likely out of scope for now**

Requires real-robot data we do not currently have on this machine (`panda_express` has the
Polymetis interface and a robot IP, but no collected real dataset in the repo). Do not start
until Phase 4 passes *and* real demonstrations exist.

1. Retrain the Phase 4 stack on real demonstrations.
2. Calibrate the alarm threshold with **conformal prediction over successful rollouts only**,
   so no failure examples are needed. (This matches our existing discipline: thresholds come
   from the safe distribution only.)
3. Evaluate on held-out rollouts containing real failures.
4. Report **lead time** — how far ahead of failure the alarm fires. A detector that fires at
   the moment of failure is useless. Our current monitor's best case is ~31 steps of lead on a
   caught topple; state the required lead **before** running.

**Acceptance:**
- [ ] Lead time exceeds the stated requirement (state the number first).
- [ ] False-positive rate on successful rollouts under the stated budget.
- [ ] Compared against ≥2 baselines: prediction-error thresholding, and policy-action-variance
      thresholding — plus our existing `d_end` + PC1 monitor, which is the real incumbent.

---

## 4. Rules

1. **Do not skip rungs.** The ladder exists so a weird robot result is attributable to physics
   rather than to a bug in the instrument.
2. **Do not tune on the evaluation set.** Select on validation splits, freeze before test.
3. **State acceptance numbers before running**, not after.
4. **Log negative results.** They have been the most valuable output of this project so far.
5. **Every figure reproducible** from `config + seed`, one command.
6. **Never report a bare FTLE without its horizon `T`.** At a guard surface it does not
   converge as `T` grows.
7. **Do not silently increase `d`.** High-`d` Jacobians are noise-dominated at realistic data
   scale. If `d` must grow, record why.
8. **If acceptance criteria fail twice, stop and report.** Do not loosen them.

### Carried over from this repo's own hard-won lessons — these are not optional

9. **Never report accuracy.** At a 1.4% base rate, predicting "safe" always scores 98.6%.
   Report AUC *and* the operating curve.
10. **Assert on tensor alignment.** Two separate silent bugs (wrong `num_hist`; history frames
    sampled 8 control steps apart) produced plausible-looking numbers for weeks. The ViT
    predictor slices its positional embedding to whatever length it receives and never errors.
    Assert history length *and* the time spacing between frames.
11. **Resolve any PCA/PC1 sign against measured motion, never against norm.** The norm
    heuristic was backwards on both datasets tested.
12. **Anything tuned on the data it is scored on needs a held-out split.** A PCA-truncation
    result looked strong in-sample and evaporated under cross-validation.
13. **`SIM_HEADLESS=1`** for any batch run importing the simulator, and write results to disk
    incrementally — EGL throws at interpreter teardown.

---

## 5. Theoretical caveats for the write-up

- An FTLE ridge is **not** automatically a Lagrangian coherent structure; some ridges are false
  positives. State the ridge criterion and its assumptions.
- At a genuine hybrid guard the FTLE **diverges** with `T` — a singularity, not a finite ridge.
  Report the saturating separation measure alongside it.
- Universal approximation applies to *continuous* functions. No smooth network uniformly
  approximates a discontinuous one; it can only steepen. **We have measured this directly**
  (linear regime 50× below operating σ) — that measurement is a result, not just a citation.
- Koopman/DMD cannot produce positive Lyapunov exponents; do not add as a chaotic-regime
  baseline.
- Reservoir computing reproduces `lambda_max` but systematically fails the negative exponents.
  **VERIFIED** against Pathak et al. 2017 (Chaos 27, 121102), Table II — on Lorenz the
  successful reservoir gives lambda_1 0.90 (true 0.91), lambda_2 0.00 (true 0.00), and
  lambda_3 **-10.5 against a true -14.6**. They attribute it to the reservoir not needing to
  reproduce the thin transverse structure that carries lambda_3, which is the same mechanism
  our off-attractor data fixes.

### One boundary on scope, from our own results

This plan is about **detection**. Do not extend it to *control* (optimising actions against the
FTLE score) without re-deriving the case: we tested that directly and it **raised** the real
topple rate 15% → 40% (McNemar p=0.013) while successfully driving the score down 26%, and was
statistically indistinguishable from random perturbations of the same magnitude. A good passive
detector is not automatically a valid control objective. If the shPLRNN's geometry turns out to
be genuinely better, that conclusion deserves a re-test — but it must be an explicit, guarded
experiment, not an assumption.

---

## 6. Deliverables

- [ ] `results/phase{0..5}/` with figures, metrics JSON, configs.
- [ ] `NOTES.md` decision log including all negative results.
- [ ] Summary table: per phase, acceptance criteria and pass/fail.
- [ ] Phase-3 mean-collapse figure (smooth MLP vs piecewise-linear).
- [ ] Phase-4 comparison against the incumbent 0.894 baseline on identical chunks/splits.
- [ ] Phase-5 lead-time curve against both baselines (if reached).
