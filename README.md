# stable_lyap_filters

A **zero-shot safety monitor** for robot manipulation. It watches a visuomotor policy's
proposed action chunk and flags the ones likely to cause a failure — without ever training on
labelled failures.

Task: a Franka Panda picks a block from a cluttered tabletop; failure is toppling a
neighbouring block. Everything runs on the latent dynamics of a **frozen DINO world model**.

**Result: AUC 0.894** on 1772 action chunks (100 episodes, 25 unsafe), held-out validated.

The idea is Lyapunov-flavoured — measure whether the predicted future is *stable* — but the
single most useful thing in this repo is which parts of that idea survived contact with data
and which did not.

---

## What works

Score an action chunk by how far the world model's predicted final latent has drifted from
the last observed one, per DINOv2 patch, then take the p90 across a masked set of patches.

| configuration | AUC |
|---|---|
| `ftle` — the original (1/T)·log(d_end/d_start), max over patches *and* perturbations | 0.599 |
| `d_end`, p90 over patches, geometric row mask only | 0.756 |
| `d_end` + low-norm patch mask (k=30) | 0.854 |
| **`d_end` + PC1 background mask (75% keep)** | **0.894** |
| **`ftle_variance` + PC1 background mask** | **0.896** |

Held-out validated: mask type and hyperparameters are chosen on one half of the episodes and
scored on the other, over 20 random splits. Thresholds come from percentiles of the **safe**
score distribution only — no failure labels anywhere in the pipeline.

```python
from monitor import Monitor
mon = Monitor.load()                            # frozen DINO world model
mon.fit_pc1(safe_latents, motion=gt_motion)     # background mask; motion fixes the sign
mon.calibrate(safe_scores, percentile=95)       # threshold from SAFE chunks only
if mon.score(frames, proprio, actions) > mon.threshold:
    halt()
```

### Three findings that made the difference

**1. Drop the FTLE denominator.** The original metric divides by `d_start`, measured one
prediction step in — so it is tiny, noisy, and dominated by how *quiet* a patch happened to
start rather than how unstable it is. Removing it is worth ~0.2 AUC. Taking a second maximum
(over perturbations) makes things worse still: an extremum over ~4100 values per chunk tracks
tail noise, not instability. This change took the metric from 0.599 (near chance) to 0.799.

**2. Mask *low*-norm patches, not high-norm ones.** The intuitive suspect was DINOv2's
high-norm artifact tokens. That was wrong: `corr(‖z‖, d_end) = −0.641` on ground-truth-static
patches. Cosine distance divides by ‖z‖, so a near-featureless patch has a poorly-determined
direction that wobbles under any perturbation. Masking the 30 lowest-norm patches: 0.756 → 0.854.

**3. PC1 masking keeps the *background*, and that is why it works.** This is the
counterintuitive one. On foreground patches (arm, gripper, moving block)
`corr(motion, d_end) = +0.45` — and it is just as strong in **safe** chunks (+0.452) as unsafe
ones (+0.386). Most foreground divergence is a *motion confound* from the always-moving arm,
not a failure signal. Background patches have ~zero baseline motion, so divergence there
actually means something. 0.854 → 0.894.

> This was originally mislabelled in our own notes. The foreground/background sign came from
> an unverified heuristic ("higher mean ‖z‖ = foreground") that turned out to be backwards on
> both datasets tested — `corr(PC1_raw, motion) = −0.479` and `−0.486`. **Always resolve the
> sign against measured patch motion**, never against norm. `Monitor.fit_pc1(motion=...)`
> does this for you.

### Watch it

| | |
|---|---|
| [`media/monitor_catches_topple.mp4`](media/monitor_catches_topple.mp4) | the deployed monitor firing before a real topple |
| [`media/old_ftle_metric_same_episode.mp4`](media/old_ftle_metric_same_episode.mp4) | the original FTLE metric on the **same** episode, for contrast |
| [`media/probe_catches_divergence_misses.mp4`](media/probe_catches_divergence_misses.mp4) | a case the supervised probe catches and divergence misses |
| [`media/patch_grid_reference.png`](media/patch_grid_reference.png) | the 14×14 patch grid and which rows the geometric mask removes |

---

## Honest limits

**Report AUC and the operating curve, never accuracy.** At a 1.4% base rate, always predicting
"safe" scores **98.6%** and beats every real configuration. (An earlier write-up of this work
quoted 96.9% accuracy as a headline; that number was an artifact of class imbalance.)

Best operating point found, `p95` threshold: **recall 0.52, precision 0.13, 0.88 false alarms
per episode.** Precision is capped by arithmetic here — a p95 threshold admits ~88 false
positives against 25 possible true positives, so it cannot exceed ~22%.

**A supervised probe beats it.** A linear probe on the *predicted* latent recovers future
block tilt at AUC 0.941 vs divergence's 0.887 on the same chunks, and reaches 100% recall at
a loose threshold where no divergence configuration does. It needs tilt labels from sim
physics, so it is not zero-shot — but it bounds what the zero-shot framing costs you, and it
shows the information is present in the latent. **The readout, not perception or dynamics, is
the bottleneck.**

**The 50-perturbation "deviator agent" may not earn its cost.** A single unperturbed rollout
(`nominal`, 53 ms) matched the full 50-rollout apparatus (2021 ms) on one dataset — difference
not statistically significant. On a second dataset the ordering reversed. Treat
nominal-vs-perturbed as **dataset-dependent and unsettled**, not a solved question.

**Known failure modes.** "Flash topples" — blocks that fall with no precursor wobble — are an
information limit given a 3-frame history, not a metric weakness. Shadow and reflection
artifacts drive some false positives.

---

## Things that did NOT help

Documented so nobody spends a week rediscovering them. All held-out validated over 20 splits.

| idea | result |
|---|---|
| low-norm mask **and** PC1 mask combined | 0.887 / 0.892 — no gain; the selector picks "no low-norm filtering" in 12–19/20 splits. PC1 already removes the patches low-norm masking would. |
| temporal aggregation (rolling max/mean/EMA over chunk scores) | 0.880 / 0.887 — worse. At a 1.4% positive rate a rolling window mostly imports false-alarm surface from safe neighbours. |
| PCA feature-truncation (as opposed to patch selection) | did not survive cross-validation, despite a promising in-sample number |
| exact Jacobian FTLE (σ_max of ∂z_T/∂a) | correct but worse; the linear regime ends ~50× below the operating perturbation size |
| the FTLE ratio family generally | dominated (0.599 vs 0.894); no amount of masking rescues the denominator |

---

## Does it generalise?

Partially, and the pattern is informative. Two from-scratch 2D toy tasks, each with its own
physics and its own small world model:

- **Pusher2D** — a pusher nudges a block past a fragile one, i.e. the same "moving actor +
  normally-static object that fails discontinuously" structure as Jenga: **AUC 0.89**, and the
  background-beats-foreground finding **replicated** (background-masked 0.916 vs
  foreground-masked 0.649).
- **CartPole** — no second object, so no foreground/background structure at all:
  **AUC ~0.55**, barely above chance.

So the method's strength appears tied to that specific scene structure rather than to
detecting dynamical instability in general. Worth stating that way rather than claiming
either extreme.

---

## Layout

```
src/monitor.py    the monitor: load -> fit_pc1 -> calibrate -> score   (start here)
src/metrics.py    divergence metrics + the patch masks, with the reasoning inline
src/paths.py      external repo locations, from env vars, fails fast
eval/analyze.py   regenerates every table above from results/
results/*.json    raw result files behind each table
media/            videos and the patch-grid reference
docs/HANDOFF.md   state, traps that cost real time, and prioritised next steps
```

Reproduce all tables: `python eval/analyze.py` (needs nothing but the JSON files).

## Setup

```bash
export DINO_WM_DIR=/path/to/dino_wm                # world model + checkpoint
export PANDA_EXPRESS_DIR=/path/to/panda_express    # MuJoCo Jenga sim + episodes
export DINO_WM_CKPT=$DINO_WM_DIR/outputs/model_latest_single.pth
pip install -r requirements.txt
```

Python 3.10+ (DINOv2's hub code uses `X | None`). `SIM_HEADLESS=1` is required for any batch
run that imports the simulator — without it `sim.py` opens a viewer at import and core-dumps
on a machine with no display.

`src/paths.py` resolves everything from those variables and fails fast with a clear message
if something is missing.
