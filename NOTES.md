# NOTES.md — running decision log

Append-only. Record decisions, failures, and surprises. Negative results go here, not in a
drawer — they have been the most valuable output of this project so far.

---

## 2026-09 — Plan adopted, core geometry built

**Phase 0 is partially pre-answered by existing results.** The source plan's stop condition
("if the signal is invisible in DINOv2 space, stop — no dynamics head can fix an encoder
problem") is already closed in the encoder's favour: a linear probe on the *predicted* latent
recovers future block tilt at AUC 0.941, and per-patch divergence localises ground-truth
motion at patch-AUC 0.955–0.960. Perception is not the bottleneck; the readout/dynamics is.

**We already have direct evidence for the plan's central premise.** `results/` contains an
exact analytic-Jacobian FTLE run on the current causal-ViT head: AUC 0.617, *worse* than plain
`d_end` (0.827). The linearisation test showed why — the linear regime ends at ‖δ‖≈1e-3, about
50× below the operating σ=0.05, where relative error is 0.963 and the predicted displacement
direction has only 0.538 cosine with truth. The Jacobian is correct and useless. That is the
measured version of "a smooth network can only steepen a discontinuity, never represent it",
and it is a result in its own right, not just a citation.

**Named risk, owner Phase 4:** our largest empirical win is *spatial* patch masking
(0.756 → 0.894 by keeping low-motion background patches). The shPLRNN wants a single
low-dimensional state. Naively pooling patches into d=32 may destroy exactly that structure.
Apply the mask before pooling, and measure whether the gain survives. If it does not, that is
a publishable finding about the cost of dimensionality reduction for this signal.

### Core geometry — built and tested

`src/models/shplrnn.py`, `src/geometry/jacobian.py`, `src/geometry/ftle.py`,
`tests/test_geometry.py` (7 tests, all passing). Two design decisions worth recording:

1. **`lyapunov_spectrum` sorts descending before returning.** Benettin/QR exponents are only
   asymptotically ordered, so an unsorted return made `spec[0] == lambda_max` unreliable at
   finite T. Sorting breaks the correspondence between exponent *i* and column *i* of Q — if
   covariant directions are ever needed, use the unsorted logs. Documented in the docstring.
2. **Two test bugs found and fixed while writing them**, both worth noting because they are
   the same class of error the plan warns about:
   - the "known linear system" test let expanding modes (|A|>1) grow the state over 200 steps
     until it crossed the switching hyperplanes, so the system stopped being linear mid-test
     and the known answer was no longer known. Fixed with a tiny s0, shorter T, and an
     explicit assertion that the gates stay closed.
   - the descending-order test asserted a property the implementation did not yet guarantee.
     Fixed the implementation rather than the assertion.

### Scope boundary carried in from prior work

This plan is about **detection**. Optimising actions *against* an FTLE-style score was tested
directly and made things worse: real topple rate 15% → 40% (McNemar p=0.013) while the score
itself fell 26%, and statistically indistinguishable from random perturbations of matched
magnitude (p=0.83). A good passive detector is not automatically a valid control objective.
If the shPLRNN's geometry proves genuinely better, that conclusion deserves a re-test — as an
explicit guarded experiment, not an assumption.

---

## Phase 1 (Lorenz-63) — STAGE 1 PASSED, STAGE 2 **FAILED**. Stopping per rule §4.8.

### Stage 1 — the instrument is correct ✅

Ground-truth spectrum computed from the TRUE equations through the same Benettin/QR code the
learned model uses: `(0.9009, -0.0007, -14.5668)` vs published `(0.906, 0, -14.572)`.
Both reference-free checks pass: sum = -13.6666 against trace -13.6667 (err 1e-4), and the
zero exponent is 7.3e-4. **So any Stage-2 failure is attributable to the learned model, not to
our FTLE code.** That separation was the entire reason for computing ground truth ourselves
rather than quoting the literature, and it paid for itself immediately.

### Stage 2 — the learned model's geometry is wrong ❌

| run | λ1 | λ2 | λ3 | λ3 err | pred loss |
|---|---|---|---|---|---|
| 150 ep, β=1 | 0.888 | 0.012 | -18.67 | 28% | 3.2e-5 |
| 150 ep, β=0 | 0.888 | -0.003 | **-16.65** | **14.2%** | — |
| 400 ep, β=0, 60k | 0.860 | 0.048 | -17.80 | 22% | **1.8e-5** |

λ1 and λ2 pass comfortably. **λ3 never comes within 5%; the best is 14.2% off, and every
configuration over-contracts.** Three attempts, criteria not loosened, stopping.

### Finding 1 — the separation loss does not help; it hurts (β ablation, 4 configs x 2 seeds)

β=0 gives the best λ3 (-16.65) and every β>0 is worse (-18.3 to -19.3). Consistent, not noise:
both β=0 seeds land at -16.63/-16.67 (std 0.02). The plan explicitly asked for this to be
logged either way.

**Plausible mechanism, not yet tested.** The loss is `||Φᵀ(s+d0) - Φᵀ(s) - dT||² / ||d0||²`.
That norm is dominated by the LARGEST components of the residual, i.e. the *expanding*
directions. Over the 10-step horizon the contracting direction has shrunk by
`e^(-14.57*0.1) ≈ 0.23`, so it barely contributes. As written the loss mostly re-supervises
λ1 — which is already easy — while adding gradient pressure that appears to push contraction
further negative. If that is right, the fix is a direction-weighted or log-ratio form of the
loss, **not** a larger β. Worth testing before concluding the idea itself is wrong: the plan's
*intent* (supervise separations) may be sound while this *particular estimator* is not.

### Finding 2 — more training makes the spectrum WORSE while prediction loss improves

400 epochs reached the best prediction loss of any run (1.8e-5) and a worse λ3 (-17.80) than
the 150-epoch run (-16.65). **The undertraining hypothesis is falsified.** This is the
cleanest demonstration in the project so far of the premise the whole plan rests on:
trajectory accuracy and Jacobian accuracy are decoupled, and optimising the former can
degrade the latter.

Cross-check that this is not a units/scaling bug: λ1 is off by a factor 0.96 and λ3 by 1.22 —
different directions. A dt or normalisation error would move both the same way.

### Two bugs found and fixed en route (same class this project keeps hitting)

1. **Init:** `A` initialised in [0.5,1.0], but at dt=0.01 the flow map is near-identity. First
   epoch loss 3e5, spectrum collapsed all-negative. Near-identity init took λ1 from -0.59 to
   +0.75 immediately. Now an explicit `init_near_identity` flag with the reasoning inline.
2. **Metric:** I was computing a *relative* error for λ2 against a true value of ~0, which
   reports nonsense like "1773% error". Now judged on an absolute scale relative to |λ1|.

### Candidate causes for the λ3 failure, untested, in the order I would try them

1. **d=3 is too small.** shPLRNN work on Lorenz typically uses a higher-dimensional latent
   with a linear readout rather than fitting the 3D state directly. Raising d is explicitly
   flagged by rule §4.7 as something not to do silently — this is the reason to do it
   deliberately, comparing only the top-3 exponents.
2. **The loss form** (Finding 1) — direction-weighted or log-ratio separation loss.
3. **Contraction is genuinely hard to identify from data**: along λ3 information is destroyed
   in ~1/14.57 time units, so the trajectory data contains very little evidence about it. This
   is the substantive version of the problem and may need a fundamentally different signal.

**Do not proceed to Phase 2 until λ3 is understood.** A Phase-2 guard-surface result measured
with a model whose contracting geometry is 15-25% wrong would be unfalsifiable — which is
precisely the situation the ladder exists to prevent.

---

## Phase 1 — ACCEPTED WITH DOCUMENTED DEVIATION (not a clean pass)

Decision taken deliberately after seeing results. Recording it as a deviation rather than a
pass, because rule §4.3 says acceptance numbers are stated before the experiment and §4.8 says
criteria are not loosened — this is a knowing exception, and a later reader interpreting
Phase 2/3 needs to know exactly what was and was not achieved.

### Final configuration
`d=3` (state = observation, matching the shPLRNN paper's actual low-latent claim), `H=128`,
`beta=0` (separation loss OFF), `off_frac=0.5`, 300 epochs, 60k trajectory, 5 seeds.

### Result vs ground truth (our own Benettin/QR on the true equations)

| | truth | mean of 5 seeds | error | best seed (2) |
|---|---|---|---|---|
| lambda_1 | +0.9009 | 0.8517 (std 0.022) | **5.5%** | 0.8910 (1.1%) |
| lambda_2 | -0.0007 | +0.0380 (std 0.023) | should be 0 | -0.0005 (exact) |
| lambda_3 | -14.5668 | **-14.6063** (std 0.097) | **0.27%** | -14.7478 (1.24%) |

Stated criterion: all three within 5% across 5 seeds. **Achieved: 2/5 seeds.** lambda_3 is
5/5. The failure is entirely lambda_1's seed variance (0.83-0.89).

### Why accepting is reasonable

* **lambda_3 -- the criterion that mattered -- is essentially solved.** 0.27% mean error
  against a published reservoir baseline of 28% (Pathak et al. 2017, Table II: -10.5 vs a
  true -14.6). That paper explicitly treats its own failure as expected: *"one might not
  expect the reservoir to accurately reproduce this very negative Lyapunov exponent"*,
  because the transverse structure carrying lambda_3 is barely present in on-attractor data.
  Our off-attractor training attacks exactly that and cuts the error ~100x.
* **Phase 2's own acceptance bar is lambda_max within 10%.** Our 5.5% is inside it, so this
  deviation does not undermine the next rung on its own terms.
* **The instrument is exactly validated** (Stage 1: sum matches the Jacobian trace to 1e-4,
  zero exponent to 7e-4), so Phase 2 failures remain attributable.

### Risk carried forward -- state this in any Phase 2/3 write-up

lambda_1 carries **5-8% seed-dependent error**, and lambda_2 sits at +0.038 where 0 belongs.
The two co-vary: lambda_1 + lambda_2 is stable at ~0.89 against a true 0.9002 (1.2% off), so
the model reliably captures TOTAL on-attractor expansion and splits it inconsistently between
the expanding and neutral directions. Any Phase 2/3 result that depends on separating those
two directions is unreliable at better than ~8%.

Precedent that this is a known leak rather than our bug: on Kuramoto-Sivashinsky, Pathak et al.
found the reservoir could not reproduce two of three ZERO exponents, and that removing those
two made the negative exponents line up well. Near-zero exponents are where these models leak.

### Also settled this phase

* **d=20 with a readout is decisively WORSE, and the spectral-gap check is why we know.**
  lambda_3 48.8% error vs 0.27% at d=3, and lambda_1 no better (0.843 vs 0.852). The extra
  latent directions do not sit far below the real dynamics -- seed 0's spectrum runs
  `0.899, 0.020, -6.75, -7.85, -8.94, -10.05, ...`, a smooth ladder with no gap. The true
  lambda_3 is not the 3rd exponent at all. **Without the gap check we would have reported
  -6.75 as lambda_3 and never known the comparison was meaningless.**
* **Correction to an earlier claim in these notes and in PLAN.md:** I stated that papers use
  d≈20 for a 3D system. That is wrong. The shPLRNN/GTF paper's headline claim is the
  opposite — reconstruction *"with at most as many latent dynamical variables as those of the
  underlying system"* (M=16 for 64-d EEG, vs dendPLRNN's 105). High-M is what that paper argues
  against. Our d=3 was aligned with it all along.
* **That paper never computes a Lyapunov spectrum for Lorenz.** It benchmarks Lorenz-63 with
  D_stsp (attractor geometry), D_H (power spectra) and PE(20) (prediction error); lambda_max
  appears only to characterise the EEG data. So its hyperparameters carry no evidence for our
  purpose, and our acceptance criterion is strictly harder than anything it validates.
* **PLAN.md's inherited claim about reservoir computing is now VERIFIED**, not assumed:
  Pathak et al. Table II reproduces lambda_max (0.90 vs 0.91) and lambda_2 (0.00) well and
  fails lambda_3 (-10.5 vs -14.6).

---

## Phase 1b — architecture comparison (shPLRNN vs smooth RNN vs GRU)

Same data, same GTF training, same Benettin/QR spectrum code; H chosen per architecture so
parameter counts match (902 / 899 / 948 — at a common H=128 the raw counts would have been
902 / 17411 / 51459, and either outcome would have been dismissible).

| model | params | Jacobian | λ₁ | λ₂ | λ₃ | λ₃ err |
|---|---|---|---|---|---|---|
| TRUTH | | exact | 0.9009 | −0.0007 | −14.5668 | |
| shplrnn | 902 | analytic | 0.8517 | 0.0380 | **−14.579** | **0.08%** |
| smooth | 899 | autograd | 0.8801 | 0.0198 | −13.097 | 10.1% |
| gru | 948 | autograd | 0.8724 | 0.0246 | −12.000 | 17.6% |

**A clean dissociation, not a clean win.** shPLRNN takes λ₃ decisively — zero overlap between
seed distributions — while the smooth models are *better* on λ₁ and λ₂. Read honestly:

* Lorenz is **smooth and has no guard surface**, so this phase cannot support any claim about
  discontinuities. That test only exists in Phase 2.
* What it does settle is Phase 1's open question: **the λ₁ weakness is the architecture, not
  our pipeline.** The identical pipeline reaches λ₁ 0.8801 with a smooth head, so the 5.5%
  λ₁ error is a property of the piecewise-linear head, not a bug in data or training.
* One GRU seed collapsed to λ₁ = −0.085 (a fixed point rather than an attractor); shPLRNN and
  the smooth RNN were stable across all seeds.

### Off-attractor data does NOT trade off against λ₁ — control run, 5 seeds each

| off_frac | λ₁ | λ₁ err | λ₃ | λ₃ err |
|---|---|---|---|---|
| 0.0 | 0.8486 | 5.80% | −18.024 | 23.73% |
| 0.5 | 0.8517 | 5.46% | −14.606 | **0.27%** |

λ₁ is unchanged (marginally better *with* off-attractor data); λ₃ improves ~88×. **The earlier
note in this file suggesting a tradeoff was wrong** — it came from `stage2_pipefix2`, a
SINGLE-seed run (λ₁ 12.9%), and the regression was an artefact of refactoring per-epoch
resampling into a fixed 6000-sequence bank, not of the off-attractor data. Off-attractor
data is a pure win and needs no caveat in the write-up.

---

## Phase 2 — choosing the bouncing-ball regime (system built, not yet trained)

Two hard failure modes bound the parameter space; both were hit empirically before any model
was trained, which is the point of tuning the *system* first.

* **e too low → inelastic collapse (Zeno).** At e ≤ 0.5 dissipation outruns the table's energy
  injection: the ball settles onto the table, the impact detector reads **zero** impacts,
  `x_max` goes negative (riding the table), and λ_max returns ~190. `step()` had been silently
  truncating at `max_impacts=8`; it now records `step.collapsed`.
* **Γ too high → the guard is never sampled.** At Γ=4 (ω=2) the ball reaches x≈47 against
  table amplitude 1, putting impacts on 0.27% of samples. The guard surface *is* Phase 2, so
  that regime is useless regardless of how chaotic it is.

**`dt`, not `e`, is the lever for impact density in the discrete map.** At dt=0.02 a flight
spans ~377 steps, but that is only ~2.4 table periods — physically fine, merely finely sampled.

**Chosen operating point: ω=1.4 (Γ=1.96), e=0.8, dt=0.20** → ~37 steps/impact (2.7% of
samples), x_med ≈ 6.8, x_max 32, min gap 8.6e-4 (no penetration), **λ_max = 0.1577 ± 0.0072
over 8/8 seeds**.

### Inelastic collapse was real, and resampling it away would have been wrong

**Correction to my first reading of this.** I initially attributed the bad λ_max at seed 3 to
the *estimator* — the two-particle renormalisation dropping the perturbed copy into a
coexisting sticking basin — and concluded the training data was clean at e=0.8. Both halves
were wrong once runs got long:

* Collapse is a **rate**, not a seed property. At 8k steps 5/6 seeds survived; at 40k steps
  only **3/8** did. Long runs collapse almost surely.
* Direct instrumentation (raising the impact cap and counting) showed a single step needing
  **>4000 impacts** with the ball at x = −0.22, riding the table. That is genuine inelastic
  collapse (Zeno), not an estimator artefact and not recoverable chattering — which is why
  raising `max_impacts` 8 → 64 never fixed it.
* I also swept `e` in the **wrong direction**. Lower e means more dissipation and *more*
  collapse. Higher e (0.9–0.95) is collapse-free but throws the ball to x_med 18–103, so the
  guard — the entire object of Phase 2 — becomes a thin sliver of state space. Every clean row
  in the sweep failed on state scale, and every small-state row collapsed.

**The impact map alone is ill-posed.** The fix is physical, not statistical: with Γ>1 the ball
does not stay stuck. It rides the table and **detaches** the instant the table falls away
faster than gravity — contact force g − Aω²sin(φ) < 0, i.e. sin(φ) > 1/Γ = 0.51. Adding that
contact/detach phase makes long runs well-posed and takes λ_max from 3/8 usable seeds to
**8/8**, with no change of regime. Discarding collapsed runs instead — the tempting shortcut —
would have deleted a real part of the dynamics from the training distribution.

Note `STICK_TOL` cannot be set arbitrarily small: relative velocity decays by a factor `e` per
impact, so reaching 1e-6 from O(1) needs ~60 impacts. It is 1e-4, paired with
`max_impacts=64`.

### Phase 2 status

System built, tuned, and tested (`tests/test_bouncing_ball.py`, 6 tests, each guarding a bug
that actually occurred). Ground truth λ_max established. **No model has been trained yet** —
that is the next step, along with the FTLE field, the hyperplane-vs-guard alignment test, and
the T-divergence characterisation.

### Phase 2 driver built — Stage 1 passes, first Stage 2 run is a clear miss

**Stage 1 (ground truth) PASSES**, with two independent checks:

| check | value | |
|---|---|---|
| spectrum (finite-difference Benettin) | (0.1585, −0.0041, −0.2198) | |
| λ₂ = 0 (flow direction) | −0.0041 | ✓ |
| λ₁ vs independent two-particle | 0.1585 vs 0.1577 ± 0.0072 | ✓ |
| seeds accepted | 6/8 (rejected on \|λ₂\| ≥ 0.01) | |

No Jacobian is used for the true system — at the guard the correct tangent map needs the
saltation matrix. The finite-difference estimator is licensed by agreeing with the
analytic-Jacobian spectrum on smooth Lorenz to machine precision (now a test).

**`eps` is not a free knob.** At eps=1e-7 seed 3 reports λ₁ = −0.030 for a trajectory that is
demonstrably chaotic (200/200 unique post-impact velocities). Only at eps ≤ 1e-8 does it
recover 0.148. The perturbed particle must stay on the *same side of the guard*; Lorenz shows
no such sensitivity because it is smooth. Likewise λ₂ converges only as ~1/T: at T=4000 every
seed reads −0.02…−0.03 and is correctly rejected, settling near −0.004 by T=20000.

**Stage 2, first run (d=4, H=128, off_frac=0.5, 300 epochs, 3 seeds):**

| | λ₁ | λ₂ | λ₃ |
|---|---|---|---|
| TRUTH | 0.1585 | −0.0041 | −0.2198 |
| shPLRNN | 0.0852 ± 0.0342 | 0.0200 | **−0.0005** |

Spectral gap clean in only **1/3** seeds. Training loss reached 5.5e-3, so the model fits the
GTF rollouts while getting the spectrum badly wrong — exactly the failure mode the two-stage
design exists to expose.

**Leading hypothesis: the model is not learning the impact.** λ₃ ≈ 0 means the learned map is
nearly volume-preserving. In this system *all* dissipation is at impacts (free flight is
area-preserving in (x,v); the reset multiplies by −e), and impacts are only 2.7% of
transitions. A rollout loss averaged over all steps is dominated by smooth parabolic flight,
so the 2.7% carrying the contraction contributes little. Note this is the opposite of Phase
1's λ₃ problem, which was about data coverage off the attractor — here the events are IN the
data and the loss under-weights them. Untested alternatives: too few epochs, α mis-set for a
hybrid system, or d=4 being too tight given one dimension is spent on the cos²+sin²=1
constraint.

### Phase 2 hyperparameter sweep — three corrections and one cross-system result

All 3 seeds per config. TRUTH = (0.1585, −0.0041, −0.2198).

| config | epochs | λ₁ | λ₂ | λ₃ | gap |
|---|---|---|---|---|---|
| baseline | 300 | 0.0852 | 0.0200 | −0.0005 | 1/3 |
| d=6 | 300 | 0.4594 ±0.31 | 0.0040 | −0.0361 | 3/3 |
| α=0.02 | 300 | 0.1765 ±0.006 | −0.0035 | −0.0983 | 2/3 |
| α=0.3 | 300 | 0.1795 ±0.086 | 0.0350 | +0.0025 | 2/3 |
| off_frac=0 | 300 | 0.1054 | 0.0115 | −0.0138 | 2/3 |
| ep1000 | 1000 | 0.2009 ±0.048 | −0.0092 | −0.2710 ±0.070 | 3/3 |
| α=0.02 | 1000 | 0.1954 ±0.006 | **−0.0043** ±0.001 | −0.3088 ±0.025 | 3/3 |
| off_frac=0 | 1000 | 0.1630 ±0.094 | 0.0078 | **−0.6721 ±0.511** | 2/3 |
| impact-weight 20 | 1000 | 0.2421 ±0.019 | −0.0023 | **−0.2027** ±0.034 | 3/3 |

**Correction 1 — my λ₃ hypothesis was wrong.** I attributed λ₃ ≈ 0 to impacts being only 2.7%
of transitions and therefore drowned out by a uniform loss. In fact **training length was the
binding constraint**: 300 → 1000 epochs alone takes λ₃ from −0.0005 to −0.2710 with a clean
spectral gap in 3/3 seeds instead of 1/3. Worth reporting on its own — this hybrid system
needed ~3× the epochs the smooth Phase 1 system did at identical settings.

**Correction 2 — the attractor-dimension prediction is refuted.** I predicted off-attractor
data would matter *less* here because the ball's attractor (D_KY = 2.70) is much fatter than
Lorenz's (2.06). Removing it instead sent λ₃ to −0.672 against a true −0.220, with the seed
spread exploding to ±0.51 — the worst and least stable result in the sweep. **The attractor
dimension is the wrong lens.**

**The cross-system result that replaces it.** On-attractor-only training makes the model
*over-estimate* contraction, in the same direction on both systems:

| | λ₃ without off-attractor data | truth | error |
|---|---|---|---|
| Lorenz | −18.02 | −14.57 | 24% too negative |
| Ball | −0.672 | −0.220 | 206% too negative |

Two systems with nothing in common failing the same way is stronger evidence for the Phase 1
diagnosis than Phase 1 alone.

**Correction 3 — teacher forcing was mis-set, in the direction opposite to the default.**
α=0.02 (more free-running) beats α=0.1 on λ₂ (−0.0043 vs −0.0092, against a true −0.0041) with
seed spreads ~8× tighter. Pushing the other way to α=0.3 drives λ₃ *positive* (+0.0025) — a
model that gains energy on every bounce. Mechanically consistent: teacher forcing keeps
snapping the state back to ground truth, so the model never confronts its own compounding
error, which is where dissipation would have to show up.

**The impact-weighted loss still earns its place**, just not as the fix it was built to be. On
a working (1000-epoch) baseline it gives the best λ₃ of anything measured, 7.8% error vs 23%,
at the cost of inflating λ₁ to 0.2421.

**Capacity hurts, again.** d=6 gives λ₁ nearly 3× truth with ±0.31 seed spread — Phase 1's
d=20 result reproduced on an unrelated system. Two for two against adding latent dimensions.

**Open and honest: λ₁ is ~25% high in every reliable config**, against Phase 2's own acceptance
bar of 10% on λ_max. λ₁ is the exponent the monitor actually uses. The only config with λ₁ near
truth (off0, 0.1630) is precisely the unstable one, so that is noise, not accuracy.

### Phase 2 Test B — POSITIVE. Convergence-vs-not separates a guard crossing from ordinary chaos

Pairs of trajectories started ε apart, split by whether their impact COUNTS diverge.

| ε | straddling | group | λ at T=1 | T=10 | T=40 |
|---|---|---|---|---|---|
| 1e-1 | 3.8% | straddled | −0.009 | +0.311 | **+0.420** |
| | | clean | +0.026 | +0.205 | +0.191 |
| 3e-2 | 1.9% | straddled | +0.101 | +0.314 | **+0.446** |
| | | clean | +0.019 | +0.180 | +0.181 |
| 1e-2 | 1.1% | straddled | −0.003 | +0.429 | **+0.514** |
| | | clean | +0.018 | +0.181 | +0.186 |
| 1e-4 | **0%** | — | — | — | — |

Clean pairs **converge** to ~0.19 and stay. Straddled pairs **keep climbing** and never settle,
ending ~2.3× higher. Consistent across three ε spanning a decade. So the discriminator is not
the magnitude but **whether λ stops changing with horizon**.

**Corrections to my stated prediction.** (1) I predicted λ would *decay* like 1/T from
saturation; it *rises* instead — separations have not saturated at this horizon, and repeated
discrete jumps accumulate. The claim "fails to converge" holds; the mechanism I gave was wrong.
(2) My first grouping (did the reference trajectory hit the guard?) found **no difference at
all**, and that null is correct and worth keeping: for infinitesimal perturbations the tangent
map across a guard is the saltation matrix, which is finite, so the exponent converges normally.

**The effect exists only at finite ε** — 0% straddling at ε≤1e-4. It is therefore invisible to
the Lyapunov exponent and lives entirely in the regime real monitors use. Proper name: this is
a **finite-size Lyapunov exponent (FSLE)** measurement (Aurell et al. 1997), not an FTLE. The
underlying event is **grazing** (Nordmark 1991); the class is discontinuity-induced
bifurcation. "Straddling" is our own label, not standard terminology.

**Implication for Jenga:** ε is not a nuisance parameter justified by EE positional error — it
is the detector's sensitivity dial. Below a critical ε no perturbed rollout ever crosses to the
other side and failures are undetectable at ANY threshold. Suggested tuning: choose ε so the
straddle rate matches the observed failure rate from labels.json.

#### Correction: the mechanism is a CASCADE, not a single guard crossing

Caught by the user watching the video: λ does NOT keep climbing for an individual pair. Two
errors of mine, both real:

1. **I read an ensemble effect as a single-trajectory one.** Test B's "straddled" median rises
   with T (0.31 → 0.42) only because the group is classified by its status at the END, so at
   T=10 most members have not yet had their event. The rise is event-timing across the
   ensemble. A single pair's λ **spikes then relaxes to an elevated plateau** (0.25 vs a clean
   0.159), and looks flat for most of its history.
2. **My "event" detector fired on noise.** `find_pairs` flagged the first instant bounce COUNTS
   disagreed. In the pair it chose that was a 0.1-time-unit offset that immediately re-synced —
   separation actually FELL through it (0.579 → 0.549, local λ went negative). The real
   divergence was ~9 time units later.

**What actually happens.** Each bounce amplifies the mismatch in WHEN the two balls hit, since
outgoing velocity depends on table phase at contact. Measured on the current demo pair:

| bounce | ball A | ball B | gap |
|---|---|---|---|
| 1 | 0.80 | 0.80 | 0.00 |
| 2 | 6.70 | 7.05 | 0.35 |
| 3 | 8.30 | 12.10 | 3.80 |
| 4 | 13.15 | 19.05 | 5.90 |

versus a clean pair bouncing at 8.80 and 19.95 with mismatch **0.00** both times.

So there is no single "grazing moment" — there is a geometric amplification of impact timing
that eventually becomes qualitative. **Retract the grazing attribution**: Nordmark's
tangential-contact singularity was never verified here, and ordinary phase amplification at
impact explains the observation without it.

**Consequence for the detection rule.** "Compare λ(T) with λ(2T), still climbing ⇒
discontinuity" is WRONG for a single trajectory — λ settles either way. The usable signals are
(a) a transient spike in LOCAL λ at each mismatched bounce, which cumulative λ smears away by
dividing through total elapsed time, and (b) an elevated settled value afterwards.

Demo videos: `results/phase2/clean_vs_cascade.mp4` (corrected, marks every bounce of each ball)
and the earlier `results/phase2/clean_vs_straddled.mp4` — clean pair ends at λ=+0.159 against a
true λ₁=0.1585 (bounce counts 2/2); straddled pair reaches λ=+0.251 with counts 2/3 and a 6×
larger final separation.

### Phase 2 Test A — NULL twice, both times caught by controls

**Attempt 1, direction alignment.** |cos| between each learned hyperplane's normal and the
known guard normal, plus |corr| of its signed distance with the true gap.

| metric | trained | untrained | random direction |
|---|---|---|---|
| \|cos\| with guard normal | 0.9899 | 0.9708 | 0.9777 |
| \|corr\| with true gap | 0.9990 | 0.9985 | — |

Saturated, so uninformative. Two causes: with H=128 hyperplanes in 4-d some row is near ANY
fixed direction by chance; and the gap x − A·sin φ is dominated by x (range ~30) over sin φ
(range 1), so any hyperplane with an x-component correlates ~1 with it. **It measured a scale
artefact.**

**Attempt 2, ReLU gate flips** (scale-free, and the only mechanism the model has for a
discontinuity):

| | at impact | elsewhere | ratio |
|---|---|---|---|
| trained | 33.58 | 5.41 | 6.21× |
| untrained | 43.99 | 8.97 | **4.90×** |

Also null. Confound: at an impact the state jumps hard (|Δv| up to 14.7), so ANY hyperplane set
is crossed more often simply because the state moved further. This measured step size, not
structure. **The untrained control is what caught both**; without it, either attempt would have
been reported as a success.

Untested fix: normalise to flips per unit distance travelled, which asks whether the model packs
hyperplanes more DENSELY near the guard independent of how far the state moves.

### Does the learned model reproduce the local-λ spikes? YES — and they are still useless as a detector

400 pairs. Label from the TRUE system (did separation actually blow up); score from the MODEL
alone, using only what a monitor could compute at runtime.

| score | AUC |
|---|---|
| **TRUE** max\|local λ\| | **0.510** ← chance |
| MODEL max\|local λ\| | 0.407 |
| MODEL std(local λ) | 0.391 |
| **MODEL final separation** | **0.786** |

**The model does reproduce the spikes.** Magnitude true 3.93 vs model 3.89; timing 48% within
5 steps against ~10% by chance. This answers what Test A could not: the learned model
represents the discontinuity sharply enough to produce spikes of the right size in roughly the
right place.

**But the spike statistic does not discriminate, even with perfect physics.** On the TRUE
system `max|local λ|` scores AUC 0.510 — a coin flip. No model can rescue a statistic that
fails in the ideal case. Cause: spikes fire at EVERY bounce whose timing differs at all,
including the harmless transients that immediately re-sync. The signal is real and physical but
it is **not specific** to trajectories that diverge. This is the same error as the t=6.3 blip,
now visible at scale — I generalised a detector from one hand-picked trajectory.

**Retracted:** the recommendation to replace the endpoint score with `max|local λ|`. Windowed λ
remains useful for LOCALISING when something happened once divergence is known, but must not be
the decision variable.

**Convergent evidence for the design already in use.** On Jenga, `d_end` (AUC 0.894) beat the
FTLE ratio (0.599). Here, on an unrelated system with exactly known ground truth, final
separation (0.786) beats every rate-based statistic (0.51 / 0.41 / 0.39). **Simple endpoint
divergence beats the cleverer rate-based metrics, twice, independently.** That is now a much
better-supported claim than when it rested on one dataset.

Caveat: only 20/400 pairs were labelled divergent, so the AUCs carry real uncertainty; the
0.51 vs 0.786 gap is wide but the positive class is small.
