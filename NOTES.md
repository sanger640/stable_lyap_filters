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

---

## Phase 3 — tipping block. The first system here that models a FAILURE

**Why the bouncing ball was retired** (user's question: "is this even a good toy problem?").
It is a good model of a *discontinuity* and a bad model of a *safety failure*:

| | Jenga | bouncing ball | tipping block |
|---|---|---|---|
| failure | topples and STAYS toppled | recurrent | **absorbing** |
| reversibility | irreversible | fully reversible | **irreversible** |
| frequency | rare, catastrophic | constant (2.7% of steps) | **rare, tunable** |
| safe vs unsafe | different regions | one attractor, no unsafe region | **two outcomes** |

The ball has no unsafe region at all — every trajectory lives on one attractor and "divergence"
is ordinary chaos happening constantly. That is *why* the detection experiments kept fighting
us, and it is the root cause of the saturation problem below.

### The saturation artefact — also the user's catch, and it invalidated a result

Separation on the ball saturates at the attractor scale (7.25) and then **oscillates**: after
saturation d swings down to 17% of its own max, spending 4 of every 40 steps below 20% of it.
My label was `true final separation > 3× median` = **> 9.27, above the saturation scale**, so it
was not labelling divergence — it was labelling *which pairs happened to sit near the top of
their oscillation at exactly step 99*.

Redone with a physical label (do bounce counts actually decouple?), ε=0.01, 116/600 decoupled:

| T | % saturated | AUC d(T) | AUC max\|local λ\| |
|---|---|---|---|
| 40 | 6% | 0.703 | 0.671 |
| 80 | 23% | 0.844 | 0.715 |
| 100 | 37% | **0.873** | 0.712 |
| 139 | 66% | 0.864 | 0.590 ← saturation eats it |

**Corrects my earlier claim that `max|local λ|` was at chance (0.510).** That was the bad label,
not the statistic — with a physical label it reaches 0.715. The *ranking* conclusion survives
and is now better founded: **d(T) 0.873 beats max|local λ| 0.712.** Also ε=0.01 beats ε=0.1
(0.873 vs 0.778) because a smaller probe leaves more room before saturation — so ε has an upper
bound from saturation as well as the known lower bound from straddling.

### The system (`src/systems/tipping_block.py`)

Classical Housner (1963) rocking block — the standard model for exactly the Jenga question.

    theta      tilt; 0 = flat, sign = which corner it pivots on
    alpha      atan(half-width / half-height); |theta| = alpha is where the CoM passes the pivot
    GUARD 1    theta = 0    base slam, angular velocity scaled by e_r = 1 - 1.5 sin^2(alpha)
    GUARD 2    |theta|=alpha  TOPPLE -- absorbing, falls to lying flat and freezes

Measured at alpha=0.35 rad (20.1°), 50-step push: static lift-off force 0.365, **topple
threshold 0.6311**, and the boundary is sharp — 0.999× survives, 1.001× topples.

| push | max\|θ\| | rocks | fell |
|---|---|---|---|
| 0.50× | 0.0000 | 0 | no (below static lift-off) |
| 0.80× | 0.0832 | 5 | no (rocks and settles) |
| 0.99× | 0.2863 | 1 | no (nearly tips; α=0.35) |
| 1.01× | 1.5708 | 1 | **YES** |

Safe-vs-unsafe separation ends at 1.628 with a **minimum after the topple of 1.362** — it never
reconverges, which is precisely what the ball could not give us.

#### The fall is integrated, not teleported (user caught this in the video)

`step()` originally jumped straight from |θ|≥α to lying flat in ONE step — and my own comment
claimed it "runs it out to lying flat" while doing no such thing. Two problems: it looked
wrong, and it erased dynamics a world model would have to learn.

Fixed by letting the same equation carry the fall. For θ > α, `sin(α − θ)` changes sign, so the
gravity term flips from restoring to driving — no special case is needed, only the removal of
the shortcut. Measured from a 1.05× push: crosses α at step 133, lands flat at step 297, so the
**fall takes 164 steps (3.3 time units)** and accelerates throughout (Δθ grows 0.05 → 0.33).

This changed what two tests should assert, and both were rewritten rather than relaxed:

* the block is **not** frozen when the flag fires — it is *falling*. What must hold is that the
  flag is monotone, |θ| never decreases, and the state freezes once flat.
* safe and failed runs are still **close** at the crossing (one is at α, the other just below);
  the gap opens during the fall. The claim is that it never closes again, so the check now
  applies from the landing, not from the crossing.

`simulate` now reports **|θ| ≥ α ("committed to falling")** rather than "already flat". That is
the safety-relevant event — past α the outcome is decided while the block is still visibly
upright — and it is monotone by construction.

7 tests in `tests/test_tipping_block.py`; the first four guard properties the ball lacked, and
one is a regression guard against the teleporting fall.
Demo: `results/phase3/tipping_block.mp4` (gitignored; regenerate with `eval/tipping_block_demo.py`).

### Phase 3 monitor test — divergence LOSES to simply rolling the model out

shPLRNN (d=4, H=128, action-conditioned), 600 trajectories, 600 epochs, final loss 2.6e-3.
Test: 400 multi-pulse pushes, 179 topple / 221 survive. Label from the true simulator; every
score computed from the learned model alone.

| T | visibly toppled | MODEL divergence | **MODEL max\|θ\|** | TRUE divergence (oracle) |
|---|---|---|---|---|
| 40 | 0% | 0.590 | **0.618** | 0.603 |
| 60 | 1% | 0.613 | **0.642** | 0.615 |
| 80 | 6% | 0.629 | **0.684** | 0.628 |
| 100 | 9% | 0.648 | **0.724** | 0.594 |
| 133 | 18% | 0.643 | **0.842** | 0.589 |
| 160 | 30% | 0.639 | **0.935** | 0.521 |

Giveaway baselines from the ACTION alone: peak force 0.765, net impulse **0.813**.

**Three findings, none flattering to the monitor.**

1. **Direct prediction beats divergence at every horizon**, and the gap widens with T (0.618 vs
   0.590 at T=40; 0.935 vs 0.639 at T=160). If the world model is good enough to support a
   divergence monitor, it is good enough to just roll out and check the failure condition.

2. **Both lose to a one-line action statistic.** Net impulse alone scores 0.813 — better than
   model divergence at EVERY horizon, and better than direct prediction until T=133. A monitor
   has to beat "add up the forces", and this one does not.

3. **Oracle divergence peaks at 0.628 and DECAYS** (0.521 by T=160). With perfect physics. So
   the ceiling on this statistic is ~0.63 and the model is already at it — the limitation is
   the statistic, not model quality. Same verdict as Phase 2's `max|local λ|`, reached the
   same way, by testing the oracle.

**Why divergence is weak here, and it is structural.** Perturbation-divergence measures
*sensitivity* — how much outcomes vary under action noise. That is maximal for pushes NEAR the
boundary, whether they topple or not. A push far above threshold topples robustly and shows
LOW divergence; a marginal safe push shows HIGH divergence. So divergence is roughly a measure
of |distance to the boundary|, which is not the same thing as which SIDE of it you are on. On
this task the two come apart cleanly.

**Caveat on generality.** The tipping block's failure is a smooth monotone function of impulse,
which is exactly the regime where "just predict it" wins. Jenga's failure may be far less
predictable directly, and that -- not divergence being intrinsically good -- would be the
argument for a monitor there. What this phase does establish is that the direct-prediction
control MUST be run before claiming a monitor adds value.

### Is divergence a good UNIVERSAL proxy? Tested properly — and the answer is no, but something else is

The previous comparison was unfair in a specific way, correctly flagged by the user: `max|theta|`
and `net impulse` are **privileged**. The first works only because we know which coordinate
means "tilt"; the second only because we know the action is a force. On DINO latents neither
exists — no dimension means anything and there is no failure predicate to read off. That is
precisely why divergence is attractive: no labels, no failure definition, no interpretable state.

So the real question is: among statistics computable from ONLY latent rollouts and perturbed
actions, is divergence the best, and is it usable? All scores below are norms and differences of
latent vectors; none would need a change to run on DINO features.

**T=100, 400 pushes (179 topple / 221 survive):**

| universal statistic | AUC | prec | recall | F1 |
|---|---|---|---|---|
| **latent final norm** | **0.792** | 0.753 | 0.698 | 0.725 |
| **latent displacement ‖z_T − z₀‖** | **0.789** | 0.687 | 0.760 | 0.721 |
| latent max speed | 0.705 | 0.551 | 0.838 | 0.665 |
| latent path length | 0.688 | 0.541 | 0.804 | 0.647 |
| divergence spread | 0.652 | 0.521 | 0.883 | 0.656 |
| divergence mean | 0.648 | 0.524 | 0.849 | 0.648 |
| divergence max | 0.587 | 0.454 | 0.983 | 0.621 |
| **FTLE ratio** | **0.518** | 0.453 | 1.000 | 0.624 |
| *[privileged] max\|θ\|* | *0.724* | | | |
| *[privileged] net impulse* | *0.720* | | | |

1. **A universal proxy does work** — latent displacement reaches 0.789 knowing nothing about any
   dimension, and at T=80/100 it **beats both privileged oracles**. An interpretable state is
   not required to detect failure.
2. **It is not perturbation-divergence.** That family sits at 0.587–0.652 and the FTLE ratio is
   last at 0.518 — chance — at every horizon tested.
3. **Displacement needs no perturbations**: one rollout vs 33. ~33× cheaper, more accurate,
   equally universal.

**This independently replicates the Jenga result:**

| | Jenga (DINO, real task) | tipping block (shPLRNN, toy) |
|---|---|---|
| displacement metric | `d_end` **0.894** | latent displacement **0.789** |
| FTLE ratio | **0.599** | **0.518** |

Same ordering, same gap, different system / model class / failure mode. `d_end` was never a
quirk of the Jenga pipeline. **Mechanism:** divergence measures SENSITIVITY, which peaks near
the boundary on BOTH sides; displacement measures WHERE THE STATE WENT, which is the side you
are on. A push far past threshold topples robustly and shows LOW divergence.

Mild support for the `ftle_variance` extension: divergence **spread** (0.652) consistently beats
divergence **max** (0.587) — the better member of the divergence family, though both trail
displacement.

**Caveat to state explicitly in any write-up:** in this toy, failure IS the large motion, which
favours displacement by construction. On Jenga the arm moves a great deal regardless, so
displacement could be dominated by arm motion rather than the block; the patch masking is what
addresses this, and the 0.894 is evidence it works. State the assumption rather than leaving it
implicit.

### Proximity to the boundary — the question actually being asked

User's point, and it reverses an earlier conclusion: detecting *failure* is not the goal;
detecting *nearness to failure* is. I had called "divergence measures |distance to boundary|
rather than which side" a defect. For this goal it is the point, and I was grading it on the
wrong task.

**Ground truth (margin).** For an action `a`, scale the whole sequence by `s` and find the
smallest `|s − 1|` that flips the outcome. 400 actions: median margin 0.337, 65 within 10%.

**The two questions are genuinely different:**

| | near boundary | far |
|---|---|---|
| topples | 38 | **141** ← robustly unsafe, nothing marginal |
| survives | **27** ← marginally safe, the dangerous ones | 194 |

**Result at ε=0.10, T=100:**

| statistic | AUC outcome | **AUC proximity** |
|---|---|---|
| **divergence mean** | 0.631 | **0.660** |
| divergence spread | 0.696 | 0.614 |
| latent displacement | **0.789** | 0.545 |
| latent final norm | **0.792** | 0.538 |
| endpoint bimodality | 0.395 | 0.366 |

The rankings nearly **invert**: displacement wins outcome, divergence wins proximity. Two
instruments for two questions.

**ε must match the margin.** Proximity AUC for divergence: 0.548 at ε=0.02 (chance), 0.649 at
ε=0.10, plateauing ~0.657 beyond. Displacement is flat at 0.545 across ε — a good sanity check,
since it uses no perturbations. **Rule: probe at least as far out as the margin you want to
keep.** ε=0.005 on Jenga is a very local probe and may be part of why the FTLE numbers are weak.

**Failed idea, recorded so it is not retried.** "Endpoint bimodality" — the perturbed endpoints
should split into two clusters when straddling a boundary, giving a scale-free (calibration-free)
threshold. It **inverted**: AUC 0.366 with ρ = **+0.236**. Likely because the fallen state is
absorbing, so toppling perturbations collapse onto one frozen latent instead of forming a
distinct second cluster. **No calibration-free threshold was found.** FTLE's λ>0 is the only
principled zero, and in practice δ=0.8 was needed anyway. Everything here requires a percentile
calibrated on safe trajectories — which still needs no failure labels, so zero-shot survives,
but "principled threshold" does not.

### Perturbation DIRECTION matters more than ε — concentration of measure

Testing a direction-matched margin backfired informatively: random directions find the boundary
*harder*, not easier (median radius 0.500 = never found, vs 0.391 for scaling). In a
T-dimensional action space a random perturbation is nearly orthogonal to whichever direction
causes failure, so it spends its magnitude on harmless dimensions. **Jenga's 8×4=32-dim action
means an isotropic probe puts only ~1/√32 ≈ 18% of its magnitude along any critical direction.**

Equal budget (32 probes), only the direction changed:

| probe family | AUC outcome | AUC proximity | ρ vs margin |
|---|---|---|---|
| isotropic (current method) | 0.613–0.617 | 0.629–0.643 | −0.24 |
| **scaled (magnitude)** | **0.734–0.743** | 0.620 | **−0.32** |
| smooth (low-frequency) | 0.624–0.635 | 0.605–0.629 | −0.24 |

**Outcome AUC jumps 0.61 → 0.74 for free**, and the margin correlation strengthens. ε barely
matters for scaled probes (0.734/0.735/0.743 across ε), consistent with direction mattering more
than magnitude. Concrete recommendation for Jenga: perturb the action's scale/magnitude, not
each dimension independently.

### CORRECTION: divergence IS a strong proximity detector — the weak result was a horizon bug

User pushed back that divergence *should* be a good boundary-proximity indicator and that
something was off. It was. Chasing it found a measurement bug that invalidated the whole
proximity section above.

**The bug.** Simulations ran for 200 steps. But the block commits at |θ|≥α around step 133 and
then takes **164 steps to fall**, so at step 200 it is still mid-fall. Measured endpoint norms
for a near-boundary action:

    toppled   |disp|: 0.432 ... 0.525
    survivors |disp|: 0.465 ... 0.498      <- completely overlapping

Every state-based statistic was searching for a split that had not happened yet, which is why
spread, bimodality, 1-D gap and 2-means clustering ALL failed together. The only statistic that
worked was the privileged flip-fraction, because it reads the *committed* flag (fires at α)
rather than the state. **That disagreement was the tell** and should have been chased
immediately instead of theorising about which statistic was better.

**Fixed (horizon 450):**

| probe | ε | divergence std | divergence spread | IDEAL (privileged) |
|---|---|---|---|---|
| isotropic | 0.05 | 0.788 | 0.801 | 0.652 |
| isotropic | 0.20 | 0.925 | 0.888 | 0.914 |
| **scaled** | **0.05** | **0.941** | 0.912 | 0.978 |
| scaled | 0.10 | 0.910 | 0.908 | 0.990 |

**Divergence proximity AUC 0.66 → 0.94**, essentially matching the oracle that knows the failure
predicate. Scaled probes still beat isotropic, most strongly at small ε (0.941 vs 0.788).

**Supersedes** the earlier claim that "divergence reaches only 0.66 on proximity" and the
speculation that spread conflates proximity with ordinary sensitivity. Both were artefacts of
the horizon.

**Direct implication for Jenga.** The monitor uses **T=8**. If a topple takes longer than 8
steps to become visible in the DINO latent, the perturbed rollouts have not separated yet and
the monitor is measuring in exactly the regime that produced the 0.66 here. **Measure how many
steps after the action a topple first shows up in the latents**; if it exceeds 8, extending the
horizon likely matters more than the choice of statistic.

Residual caveat: even at 450 steps the separation is imperfect (toppled min 0.474 vs survivor
max 0.466), because `random_push` scatters pulses across the whole horizon so some topples begin
near the end. Confining the action to early steps and leaving a settling tail should push this
higher still.

### Phase 3 monitor on the LEARNED model, corrected horizon — probe families compared

Everything earlier was either the true simulator or ran at a horizon too short for the failure
to reach the state. Both fixed: model trained AND rolled out at T=450 (fidelity corr(θ)=+0.918,
RMSE 0.265 rad). 250 actions, 151 topple, 58 near-boundary.

**Key algebraic simplification:** `a + c·a = a·(1+c)`, so "multiplicative shared" is just
"additive shared along v = a". There is one mechanism — perturb along a chosen direction `v`
with a single scalar — and the only question is which `v`. That is what makes it transferable:
Jenga's action is an absolute EE position where multiplying is origin-dependent, but perturbing
along a chosen direction is well defined.

**AUC proximity (margin < 10%):**

| probe family | v | ε=0.05 | ε=0.10 | ε=0.20 |
|---|---|---|---|---|
| per-step *(current deviator agent)* | random, fresh per probe | 0.636 | 0.649 | 0.668 |
| shared-const | ones | 0.741 | 0.712 | 0.642 |
| **shared-action** | a (= scaling) | 0.736 | **0.800** | **0.818** |
| **shared-envelope** | sign(a) | 0.705 | 0.769 | **0.793** |

The oracle result transfers to the learned model: **shared probes beat per-step by ~+0.15 AUC**,
same direction and size as on the true simulator (0.916 vs 0.828).

**Deployment numbers at the best config** (shared-action, ε=0.20): precision 0.516, recall
0.845, F1 0.641, accuracy 0.780 — versus per-step's 0.414 / 0.621 / 0.497 / 0.708. It catches
85% of near-boundary sequences but about half its alarms are false. Better on every measure, not
solved.

**Divergence has NO outcome signal.** Every family's outcome AUC was 0.29–0.56 (chance or
inverted) and every outcome F1 was exactly **0.753**, which is precisely the F1 of labelling
everything unsafe at this 60% base rate. AUC caught a degeneracy F1 concealed. **Divergence is a
proximity detector, not a failure detector** — the two need different statistics.

**Transferable recommendation for Jenga.** `shared-action` wins but is multiplicative and does
NOT transfer (scaling an EE position is origin-dependent). `shared-envelope` is additive and
nearly as good (0.793 vs 0.668). The Jenga analogue perturbs the action chunk's DISPLACEMENT
rather than its absolute coordinates:

```python
# now:      a_tilde = a + eps * randn(8, 4)          # 32 independent draws, they cancel
# instead:  D = a[1:] - a[:-1]                       # direction of travel
#           a_tilde = a + eps * randn() * (D/||D||)  # 1 draw, coherent
```

Same N rollouts, same world model, no retraining.

Demo: `results/phase3/monitor_live.mp4` — two sequences that BOTH survive, margins 0.9% and 50%,
scores 0.892 (ALARM) and 0.166 (quiet) against a calibrated threshold of 0.736. Identical
outcomes, so an outcome detector cannot separate them; the proximity monitor can. Note these are
hand-picked clear cases; aggregate performance is the 0.52/0.85 above.

### The calibration-free detector: BASIN ENTROPY — and the k=2 bug that hid it all session

Literature deep dive turned up that we had been reinventing three named quantities:

| concept | reference |
|---|---|
| **basin stability** — probability a perturbation returns to the desired attractor | Menck, Heitzig, Marwan & Kurths, *Nat. Phys.* **9**:89 (2013) |
| **basin entropy** + the **ln 2 criterion** for fractal boundaries | Daza, Wagemakers, Georgeot, Guéry-Odelin & Sanjuán, *Sci. Rep.* **6**:31416 (2016) |
| **uncertainty exponent** α (α=1 smooth, α<1 fractal) | Grebogi, McDonald, Ott & Yorke, *Phys. Lett. A* **99**:415 (1983) |

The scaling exponent measured earlier — **1.06 far from the boundary, 0.615 near** — is a textbook
α measurement, and the 0.5 is Nordmark's grazing law. We derived it from scratch.

**Basin entropy is calibration-free by construction.** S = −Σ pⱼ ln pⱼ over the *terminal states*
reached by the probes. It is built from a probability, so it has no units: S = 0 means every
probe reached the same place (far from any boundary), S = ln2 means a perfect split (on it).
Crucially it is maximal AT the boundary and zero on BOTH sides, unlike any magnitude score.

**THE BUG (mine, all session).** The block topples LEFT or RIGHT, so there are THREE terminal
states: (−π/2,0), (0,0), (+π/2,0). Every clustering attempt used k=2, and:

* the centroid of the two fall modes is **(0,0)** — coinciding exactly with upright, so
  centroid separation is ~0 **by construction**
* 2-means splits **left-fall vs right-fall**, not toppled vs safe

That single error produced every clustering null recorded above — the bimodality ratio (0.366,
inverted), the 1-D gap statistic, the pooled-clustering run (0.658), and the "even the TRUE
system doesn't separate the outcomes" diagnostic (ratio 0.53), which was itself an artefact of
the centroid metric rather than a fact about the system.

**With k=3 it works, on the learned model:**

| | AUC proximity | S>0: P / R / F1 |
|---|---|---|
| MODEL basin entropy (k=3 latent clusters) | **0.836** | 0.327 / **0.927** / 0.483 |
| ORACLE basin entropy (3 true terminal states) | 0.880 | 0.325 / 0.927 / 0.481 |

k=3 clusters recover the true terminal states 97.4% of the time on the true state and 68.9% on
the model latent — and that is enough: the model's operating point is **identical** to the
oracle's (F1 0.483 vs 0.481).

**Versus the calibrated magnitude:** div_std gives AUC 0.770–0.818 with recall 0.794 and needs a
percentile fitted on safe trajectories. Basin entropy gives **AUC 0.836, recall 0.927, and no
calibration at all**. Precision is low (0.33) but that is an ε-matching artefact — ε=0.2 probes
reach boundaries ~20% away while "near" is defined at 10%.

**Recipe, and the warning that transfers to Jenga:**

```
OFFLINE  roll many episodes, pool the final latents, cluster with k > 2
         (pick k by silhouette / gap statistic -- NOT k=2)
RUNTIME  assign each of the N probes to its nearest terminal state
SCORE    S = -sum p_j ln p_j;  flag if S > 0
```

**A Jenga block can topple in several directions, and there are several blocks, so the number of
terminal states is well above 2.** Clustering DINO latents with k=2 will reproduce this exact
failure and lead to the false conclusion that the signal is absent.

### END-TO-END pipeline: automatic attractor discovery + basin entropy

Earlier numbers were stitched from runs with different settings (k supplied by hand, no
settling). `eval/phase3_pipeline.py` assembles the whole thing and measures it once.

**OFFLINE (no k supplied).** Roll actions through the model, then keep rolling with the ACTION
OFF so states settle (555/600 converge). Merge settled endpoints at a swept distance; the
attractor count is the PLATEAU:

```
d/scale   0.05  0.10  0.15  0.20  0.30  0.40  0.60  0.80
groups      15     7     4     4     3     3     3     3
                                     ^^^^^^^^^^^^^^^^^^ plateau -> 3 attractors
```

**RUNTIME.** Coherent `shared-action` probes → roll + settle each → assign to nearest attractor
→ S = −Σ pⱼ ln pⱼ → alarm if **S > 0** (the probes did not all agree).

**Results, 250 actions, 64 within 10% of the boundary:**

| ε | AUC | precision | recall | F1 | flags |
|---|---|---|---|---|---|
| 0.05 | 0.728 | 0.562 | 0.641 | 0.599 | 29% |
| **0.10** | **0.856** | 0.469 | **0.953** | **0.629** | 52% |
| 0.20 | 0.847 | 0.320 | 0.984 | 0.483 | 79% |

**Best at ε = 0.10, exactly the margin being asked about** — the ε-matching rule, now confirmed
on the full pipeline. Note ε is a SPECIFICATION ("warn me within 10%"), not a fitted quantity:
you choose the margin you want, you do not tune it against labels.

**Versus the calibrated-magnitude baseline** (div_std, 80th-percentile of comfortable actions):

| | AUC | recall | F1 | calibration |
|---|---|---|---|---|
| div_std | 0.818 | 0.794 | 0.543 | needs a percentile from safe data |
| **basin entropy (ε=0.10)** | **0.856** | **0.953** | **0.629** | **none** |

Better on every measure while needing no calibration at all.

**Where the residual error comes from — measured, not assumed.** Model's own tilt readout agrees
with truth 86.7%; settled-latent groups agree with the true basins 87.0%. **Those match**, so the
clustering adds essentially zero error and the ~13% loss is entirely the model predicting the
wrong outcome. The cross-tab shows it **over-predicts toppling** (readout counts [231,191,178] vs
true [184,229,187]) — expected, since its RMSE of 0.265 rad is comparable to α = 0.35.

**Limitation of the settling test.** "Has it stopped moving" only finds FIXED-POINT attractors.
A limit cycle or chaotic attractor never stops, so this method would discard it — the bouncing
ball would yield zero basins. Datseris & Wagemakers' recurrence test handles those correctly
(a bounded trajectory revisits cells even while moving). A finished Jenga scene is static, so
the simple test should suffice there; monitoring *during* motion would need recurrence.

### Live-monitor demos (`eval/phase3_live_demo.py`, 5 videos)

Receding-horizon version of the pipeline: every 15 steps the monitor takes the CURRENT state,
perturbs the REMAINING action, rolls each probe, settles, assigns to a discovered attractor and
reports S. Alarm on S > 0. Measures WARNING TIME = steps between first alarm and the block
actually crossing α.

| case | margin | topples | first alarm | crosses α | warning |
|---|---|---|---|---|---|
| A on the edge | 1.2% | yes | t=0 | t=419 | **419 steps** |
| B close to the edge | 7.2% | yes | t=0 | t=373 | **373 steps** |
| C survives, only just | 1.4% | no | t=0 | — | correct proximity warning |
| D survives comfortably | 46.3% | no | t=0 | — | **FALSE POSITIVE** |
| E topples decisively | 43.6% | **yes** | **never** | t=372 | **MISS — the blind spot** |

A and B are the headline: the alarm fires immediately, hundreds of steps before anything is
visible. C is the case an outcome detector cannot produce at all — a survivor flagged for being
marginal.

**Videos now show WHY S is what it is** (4 panels): the block now plus a ghost per probe at its
predicted ENDING coloured by assigned attractor; the same probes in latent space (PCA) with the
attractor centres marked; ground truth; and the entropy trace with ln2/ln3 reference lines. The
attractor readout tilts come out [+1.69, −1.69, −0.00] → falls-right / falls-left / stays-up, so
the discovered clusters do carry the physical meaning without ever being told it.

**Two label/horizon mismatches found while building this, both mine:**

1. **The alarm turns OFF before the topple.** Case A: S = 1.086 at t=0, zero by t=180, and
   silent at t=419 when the block actually crosses α (alarm on for only 43% of calls; case B
   67%). That is *correct* for a proximity detector — once the outcome is committed there is no
   uncertainty left, so S = 0 — and it means **the alarm window is the intervention window**.
   But my first videos used a STICKY banner, so they implied a continuously-lit alarm. The
   "419 steps of warning" is the gap to the FIRST alarm, not a sustained one. The score also
   flickers (ON→off→ON) from 32-probe sampling noise; a latch would fix it.
2. **"Short lookahead fails" (AUC 0.49) was a mislabelled test, not a result.** The monitor saw
   only `a[:H]` while the label came from the margin of the FULL 450-step action — graded on
   information it could not have. Re-labelled per chunk ("apply THIS chunk then stop"), H=200
   gives AUC 0.792 on a 350-step rollout vs 0.831 at H=450/600. Chunks of 25–50 steps give a
   *degenerate* label (0–4% topple) — this system needs sustained pushing, so one short chunk
   cannot hurt it, and there is no per-chunk safety question to ask.

   Note the label used pad=600 while the monitor used settle=150, so those horizons are NOT
   apples-to-apples: the monitor predicts an outcome resolving after its own window ends. Fine
   in principle, but "total horizon 350" is the monitor's cost, not the physics' timescale.

**Fixed-horizon sweep** (lookahead + settle, both constant — better runtime semantics):

| lookahead | settle | total | AUC | recall | F1 |
|---|---|---|---|---|---|
| 450 | 100 | 550 | **0.817** | 0.880 | **0.615** |
| 450 | 400 | 850 | 0.800 | 0.860 | 0.589 |
| 200 | 400 | 600 | 0.708 | 0.440 | 0.473 |

**The settle tail can be cut 4× for free** (400 → 100 costs nothing). Probes only need to be
*distinguishable*, not *finished* — a block 100 steps into a 164-step fall is already
unmistakable. Corrects my earlier estimate that Jenga would need 5–7× its current horizon; on
this scaling (~0.6 × time-to-distinguishable) it is closer to **2.5×**.

**Open question for Jenga, which decides the whole design:** can a SINGLE 8-step chunk topple a
block? If yes, the per-chunk monitor is well posed. If topples need several chunks accumulating,
every chunk is individually safe, the monitor is silent always, and you need multi-chunk
lookahead or a state-based (not action-based) risk measure instead. `labels.json` should settle
it — check whether topples coincide with one chunk or develop across several.

**D and E are the honest half.** D is a false positive at 46% margin, consistent with the
measured precision of 0.469. E is the structural blind spot: a push far PAST the threshold
topples robustly, every probe agrees it falls, so the entropy is ZERO and the monitor stays
silent through a real failure. **That is correct behaviour for a proximity detector and
disqualifying for a failure detector**, and it is why this must be paired with an outcome score
(`d_end`) rather than replacing one.

### 100-episode evaluation of the refined monitor (`eval/phase3_eval100.py`)

Two refinements from analysing the 5-case demo, both of which change the numbers:

* **Score at t=0, no latching.** Latching turned one noisy frame into an episode-level false
  positive (case D was correctly SILENT at t=0, dissent 0/32, and only alarmed at t=15).
* **Alarm on k ≥ 2 dissenting probes, not S > 0.** One dissenter of 32 is a 3% rate, consistent
  with a true rate near zero — the sampling floor. Still calibration-free: a statement about the
  probe sample, not a threshold in latent units. Measured on 250 actions, 1→2 lifts precision
  0.454→0.519 and accuracy 0.696→0.760 while recall only falls 0.922→0.844.

**Result, 100 episodes** (25 near-boundary, 54 topple):

| | near | far |
|---|---|---|
| **ALARM** | 17 | 20 |
| no alarm | 8 | 55 |

```
precision 0.459   recall 0.680   F1 0.548   accuracy 0.720
AUC of the dissent count: 0.769
```

Weaker than the 250-action run at the same settings (AUC 0.769 vs 0.828, recall 0.680 vs 0.844)
— a reminder that these numbers carry real run-to-run variance with a positive class of ~25.

**The blind spot is the headline, and it is large.** **39 of 100 episodes topple while FAR from
the boundary, and 34 of those draw no alarm.** Every probe agrees the block falls, so entropy is
zero and the monitor is silent through a genuine failure. That is correct behaviour for a
proximity detector and disqualifying for a failure detector — **this cannot be deployed alone.**
It must be paired with an outcome score (`d_end`, which reaches AUC 0.894 on real Jenga data).

10 stratified demo videos in `results/phase3/eval100/` (3 TP, 2 FP, 2 FN, 2 BLIND, 1 TN) —
stratified deliberately so every failure mode is visible, not sampled or cherry-picked. The
worst case is `ep019_FN`: margin 0.1%, a one-in-a-thousand near miss, missed entirely.

### CORRECTION: scoring at t=0 was a one-shot GATE, not a monitor

User pushed back that judging the videos at t=0 "seems very premature and wrong". Correct, and
the cost was large. I had found that LATCHING turned one noisy frame into an episode-level false
positive, and overcorrected all the way to scoring once before execution — which stops looking.

Audit of the 10 demo episodes, t=0 verdict vs the full running trace the videos actually show:

| ep | t=0 verdict | k at t=0 | peak k | at t | genuinely risky? |
|---|---|---|---|---|---|
| 8 | "FP" | 1 | **7** | 135 | topples |
| 19 | "FN" | 0 | **10** | 135 | margin 0.1% |
| 2 | "FN" | 2 | **15** | 285 | topples |
| 0 | "BLIND" | 0 | **2** | 30 | topples |

Sensible verdicts: **6/10 at t=0 vs 9/10 continuous.**

**Same 100 episodes, both rules:**

| | precision | recall | F1 | AUC |
|---|---|---|---|---|
| t=0 only *(previously reported)* | 0.395 | 0.600 | 0.476 | 0.769 |
| **continuous** | 0.403 | **1.000** | **0.575** | **0.863** |

**Recall 1.000 — all 25 near-boundary episodes caught, zero misses.** The blind spot also shrinks:
of 39 episodes that topple while far from the boundary, 15 are now caught (24 missed, down from
34), because they pass THROUGH a marginal state on the way over and a monitor that keeps watching
sees it.

**Cost: precision 0.403, flagging 62 of 100 episodes.** Perfect recall is bought with many alarms,
and k≥2 now has 30 chances to trip per episode, so the sampling floor returns in a new form. A
stricter k, or requiring two consecutive alarms, trades recall back for precision.

**These numbers supersede every earlier t=0 figure in this file.**

### Also corrected: the ε/label mismatch

ε=0.10 is the 1σ probe width, but with 32 draws the extreme reaches ~2.3σ, so the ensemble
genuinely straddles boundaries out to ~23%. Labelling "near" at 10% counted correct detections
at 12–22% as false positives — ep007 (margin 21.8%, 8/32 dissenting, S=0.562) fired confidently
and the block DID topple.

| "near" label | precision | recall | F1 |
|---|---|---|---|
| < 10% | 0.459 | 0.680 | 0.548 |
| **< 22%** (matches probe reach) | **0.649** | 0.585 | **0.615** |

**Rule: the label threshold is the probe's REACH (≈2.3ε for 32 draws), not its σ.** For Jenga at
ε=0.005 that means claiming detection out to ~0.0115, not 0.005.

It does not excuse everything: 9 of the 20 false positives had margins 0.32–0.50 (six at the
0.50 search cap, i.e. no flip found within ±50%), which no probe reach explains. And only 5 of
the 20 actually toppled.

**Pattern across this session, worth stating once:** five measurement errors, all the same shape
— an evaluation protocol that could not see what the method produces. Horizon too short; margin
measured along a direction the probes never sample; label from the full action while the monitor
saw one chunk; ε mismatched to reach; scoring at a single instant. Each time the tell was the
same: an oracle and a proxy disagreeing far more than the method's quality could explain.

## Fixed lookahead is now the monitor's default (H=200)

Previously the monitor scored `remaining` — every action from `t` to the end of the episode — so
its horizon shrank from 450 steps to 5 as the episode ran. The score's meaning drifted with time,
which is not what a runtime monitor should do: in Jenga the policy issues an 8-step chunk, runs
it, then issues another. `Monitor.score()` now truncates to `remaining[:H]` with H=200, and
`phase3_eval100.py` takes `--lookahead` (0 restores the receding behaviour).

`--near` also now defaults to 0.23 rather than 0.10, matching the probe reach (2.3ε for 32 draws).

**100 episodes, alarm = k≥2 of 32, continuous, H=200:**

| | near (margin < 23%) | far |
|---|---|---|
| **ALARM** | **38** | 22 |
| no alarm | 4 | 36 |

precision 0.633 · recall 0.905 · F1 0.745 · accuracy 0.740 · **AUC 0.804**

These match the earlier standalone fixed-H sweep at H=200 exactly, which is the consistency check
that the wiring is right.

**Recall is insensitive to where the "near" line is drawn; precision is not:**

| near = margin < | n pos | precision | recall | F1 |
|---|---|---|---|---|
| 10% | 25 | 0.400 | 0.960 | 0.565 |
| 15% | 31 | 0.500 | 0.968 | 0.659 |
| 20% | 38 | 0.583 | 0.921 | 0.714 |
| **23%** *(probe reach)* | 42 | **0.633** | **0.905** | **0.745** |
| 30% | 51 | 0.733 | 0.863 | 0.793 |

So the honest single statement is *recall ≈0.92, precision 0.58–0.73 depending on the margin you
care about* — not one precision number, which is what made the earlier reporting misleading.

**The t=0 gate now fails in the opposite direction.** Fixed H makes a single check at t=0 *precise
but nearly blind*: precision 0.824, recall 0.333. Continuous scoring gets **2.7× the recall**.
Under receding lookahead the gate was imprecise AND blind, so this is a cleaner demonstration that
the gate is simply the wrong device — its failure mode changes with H while continuous scoring's
does not.

**Cost of fixed H: 4 false negatives** (margins 18.7%, 19.8% in the rendered sample), all right at
the edge of the 23% reach where detection is inherently marginal. Receding lookahead had none. The
trade is a little recall for a score whose meaning does not change with time.

Blind spot shrinks to 28 far-margin topples with 16 drawing no alarm (from 39/24), largely because
the 23% label reclassifies some of them as legitimately near. "Sensible" verdicts rise 54% → 70%.

## Does a one-line action statistic shortcut the toy? (outcome yes, proximity no)

A "cheat" baseline tests the BENCHMARK, not the method: if one scalar computed from the action
alone predicts the label, the toy cannot demonstrate anything and a good score on it proves
nothing. Net impulse is the candidate here, and there is an algebraic reason to expect the worst:
impulse is LINEAR in the action, and the margin oracle scales the whole action, so if toppling were
purely "impulse > I*" the flip would land at s* = I*/I(a) and

    margin = |I*/I(a) - 1|

-- the margin in closed form from one number, along exactly the axis the probes explore.

Measured on the same 100 episodes. Every baseline had its S* fitted by grid search to MAXIMISE its
own AUC on the labels it is scored against; the method is fitted to nothing.

| statistic | AUC outcome | AUC proximity |
|---|---|---|
| net impulse \|sum a\| | 0.676 | 0.624 |
| abs impulse sum\|a\| | **0.925** | 0.700 |
| peak force max\|a\| | 0.783 | 0.703 |
| L2 norm \|\|a\|\| | **0.922** | 0.764 |
| **METHOD (dissent k)** | 0.567 | **0.808** |

**The shortcut is real for OUTCOME and absent for PROXIMITY.** Action magnitude nearly solves "will
it topple" (0.925) while the method is near chance (0.567); on "is it near a boundary" that
inverts. So scoping the claim to proximity is not a convenient retreat -- it is the only question
on this toy that is not already answered by adding up the forces. The block is a legitimate
proximity benchmark and an illegitimate outcome benchmark, now measured rather than assumed.

**The caveat that matters more than the table.** Paired bootstrap, 4000 resamples:

```
METHOD       AUC 0.808   95% CI [0.721, 0.887]
L2 (fitted)  AUC 0.764   95% CI [0.667, 0.854]
gap        +0.044   95% CI [-0.069, +0.155]   P(method better) = 0.78
```

**At n=100 the method does NOT demonstrably beat a fitted magnitude baseline on AUC.** The gap
leans right at 78% but straddles zero. Do not put this comparison in a paper as an AUC win.

The claim lives on the CALIBRATION axis instead: 0.764 required fitting S* against the test labels,
a quantity unobtainable on a new task without failure data; 0.808 required fitting nothing. That
difference is categorical and needs no confidence interval. If the AUC gap is wanted too, ~400
episodes at the current effect size would clear zero.

Scripts: `eval/phase3_action_shortcut.py`, `eval/phase3_bootstrap_gap.py`.

**Aside -- a real bug found doing this.** `auc()` in `eval/run_phase3_block.py:46` ranks with
ordinal ranks, not average ranks, so any integer-valued score with many ties (the dissent count k
is exactly that) is slightly misreported: 0.804 vs the correct 0.808. Small, but systematic, and it
affects every k-based AUC recorded above this line.

## The shPLRNN's stated justification has lapsed

PLAN.md §0 picks shPLRNN over the causal-ViT head for one reason: **exact Jacobians**. The
diagnosis there is sound and worth keeping -- FTLE on the ViT head failed because "the linear
regime is 50x too small" (at ||delta||=1e-3, cosine 0.9995; at the operating sigma=0.05,
relative error 0.963 and cosine 0.538), so the Jacobian was correct and useless. A smooth network
can only steepen a discontinuity, never represent it. ReLU switching hyperplanes carve state space
into polyhedral cells that are exactly affine inside, so the Jacobian stays valid at finite
perturbation size.

**But the monitor that actually shipped never computes a Jacobian.** Basin counting rolls out,
settles, and clusters endpoints -- no linearisation anywhere. The exact-Jacobian argument belongs
to Phases 1-2 (spectrum, FTLE); it does not justify the architecture for the current method.

What still does justify it, in order of how much it matters:

1. **Determinism.** The method measures spread among rollouts caused by ACTION perturbations. A
   stochastic latent (Dreamer's RSSM, anything with a KL term) makes two rollouts of the SAME
   action land in different places, and that spread is unrelated to any boundary. `k` would be
   measuring model noise plus action effect, with no way to separate them. This is a real
   constraint on what world models the method can sit on top of, and it should be stated as one.
2. Speed -- one score is 32 probes x 350 steps; the full 100-episode eval is ~34M model steps.
3. A 4-D latent matches a 2-D observation and 600 trajectories.

**Implication for what the toy proves.** Nothing about shPLRNN transfers to Jenga, whose world
model is DINOv2 + a ViT predictor. The toy validates the ALGORITHM, not the architecture. That is
consistent with the universality claim (the monitor is architecture-agnostic by design) but it
means the toy cannot be cited as evidence that the model class works -- only that the procedure
does, GIVEN a model that rolls out accurately and deterministically.

This sharpens the Jenga go/no-go: the question is not only whether toppled and upright scenes
separate in DINOv2 latent space, but whether a ViT predictor's settled latents cluster into
DISCRETE basins at all. The shPLRNN's piecewise-affine structure may be doing more work there than
has been verified.

## Phase A (DINO-WM plan): renderer — PASS

`src/systems/block_render.py`, checked by `eval/phase_a_render_check.py`. Artifacts in
`results/phase_a/`.

Orthographic side view, so the block's geometry is exact and theta is recoverable from the
silhouette without distortion. Only the tabletop and the shadow are faked into pseudo-3D; they are
scenery, never the signal.

**Acceptance, all three met.**

| criterion | result |
|---|---|
| `+theta` vs `-theta` visibly differ | mean\|diff\| 3.4 (theta=0.10) rising to 22.7 (theta=0.60); all > 2 grey levels |
| three terminal states mutually distinct | 20.0-22.7 mean\|diff\| pairwise |
| \|theta\| monotone in distance from upright | 1.8, 3.5, 6.9, 10.4, 14.8, 18.8, 20.8, 21.5 — strictly increasing |

**Two framing decisions worth recording.**

*Crop tighter than the fallen block.* Framing wide enough to contain the fallen block (x = +-2.22)
shrinks the STANDING block to ~38% of frame height, about 6 patches of 14 px. Cropping to
x in [-1.75, 1.75] lets the fallen block run off the edge -- "lying flat toward the left" stays
unmistakable when clipped -- and buys the standing block 57% of the height instead. The first
render used the wide framing and the block was visibly too small.

*The ground plane needed strengthening.* At `_SQUASH = 0.18` the shadow was a thin smudge against
the ground line and the tabletop stripes read as noise, which defeats the purpose: the scene has to
carry real nuisance or a pass proves nothing about a cluttered tabletop. Raised to 0.50 with
stronger stripes and opacity 0.38-0.70.

**The lighting sheet is the artifact to look at.** Same pose (theta=0.25), six episodes: the shadow
swings from hard-left to hard-right and changes length, with ambient, warmth and block value moving
too. That is the documented Jenga precision bottleneck (CLAUDE.md, Limitations 2 -- the world model
predicts shadows inconsistently between original and perturbed rollouts, inflating d_end) placed
inside a system where the true margin is computable. Lighting is sampled PER EPISODE and held fixed
within it, so the shadow moves only as the block tilts -- as it would with a fixed lamp.

If Phase C cannot fit the dynamics, the lighting range is the first thing to narrow; it is a
deliberate difficulty knob, not a fixed property of the system.

**Throughput: 10.7 ms/frame at 2x supersampling (94 fps).**

| corpus | frames | render time |
|---|---|---|
| 600 traj x 45 steps | 27,000 | 4.8 min |
| 600 traj x 90 steps | 54,000 | 9.6 min |
| 600 traj x 150 steps | 90,000 | 16.0 min |
| 600 traj x 450 steps | 270,000 | 48.1 min |

So the ~20 min budget holds up to 150 steps and breaks at the full 450. **The Phase B sampling-rate
decision therefore has a rendering-budget consequence as well as an omega-recoverability one** --
if omega needs the full rate, rendering alone costs the better part of an hour before any encoding.

## Phase B (DINO-WM plan): omega IS linearly recoverable — PASS, at stride 10

**The first run said FAIL at every stride. It was wrong, and the tell was in the control.** theta is
directly visible in a single frame, so a linear probe scoring R^2 = 0.856 on it is evidence that the
PROBE is broken, not the representation — a measurement that fails the easy case cannot be trusted
on the hard one. omega scoring NEGATIVE R^2 (-1.0 to -4.75) said the same thing: worse than
predicting the mean is a generalisation failure, not an absent signal.

Two real bugs, one hypothesis of mine that was wrong.

**Bug 1 — the solve.** Standardising the PCA components divides the low-variance ones by tiny
numbers, turning them into amplified noise and leaving the normal equations ill-conditioned. The
symptom was R^2 moving NON-MONOTONICALLY in lambda (0.972 -> -0.476 -> -0.872 -> 0.717), which ridge
never does. Fixed with an SVD solve, centre-only, no per-feature scaling. This alone lifted
varied-lighting omega from **-3.385 to ~0.28** and theta from 0.854 to 0.962 — so most of the
original catastrophe was arithmetic, not physics.

**Bug 2 — test-set selection.** The first sweep reported test R^2 at every lambda, i.e. picked the
winner by peeking. lambda is now chosen on a validation split of the TRAINING trajectories.

**Wrong hypothesis (mine).** I suspected lambda=1.0 was effectively no regularisation given a
diagonal of order n=14,000. Plausible, and false: raising lambda made trajectory-split theta WORSE
(0.854 -> -0.519). Lighting was the cause the whole time.

**Corrected result — fixed lighting, trajectory split, lambda on validation:**

| stride | dt_eff | steps/ep | theta R^2 | omega R^2 |
|---|---|---|---|---|
| 1 | 0.020 | 450 | 1.000 | 0.874 |
| 2 | 0.040 | 225 | 1.000 | 0.970 |
| 3 | 0.060 | 150 | 1.000 | **0.977** |
| 5 | 0.100 | 90 | 1.000 | 0.918 |
| **10** | 0.200 | **45** | 1.000 | **0.955** |

**Every stride clears the bar (theta > 0.95, omega > 0.80). Take stride 10.** Cheapest to render
(4.8 min), and it independently reproduces the earlier finding that coarsening to 45 steps costs
nothing. The predicted tension is visible but mild: stride 1 is WORST for omega (0.874) because
inter-frame motion approaches sub-pixel; the sweet spot is stride 2-3.

**This shrinks the riskiest phase by 10x.** Rollout+settle goes from 350 steps to 35
(H 200->20, settle 150->15), which is a far easier ask of an autoregressive ViT and makes Phase D
(does it settle?) substantially safer.

**Under VARIED lighting the probe still fails on omega (~0.28) and that is fine.** It saw 32
training lighting draws; Phase C gets 600, with a nonlinear model. A linear map cannot learn
lighting invariance from 32 samples. A sweep of "how many lighting draws does invariance need" was
started and then KILLED as a detour — Phase C answers that question directly and definitively by
training the model, and a linear probe's invariance says little about a ViT's.

## Latent geometry: does lighting dominate pose? — PASS, but the margin is not comfortable

The question Phase C cannot answer. The monitor assigns basins by `argmin_c ||E - C_c||^2` —
nearest centroid, Euclidean — so it needs the SPACE to be metrically organised by pose. Whether a
predictor can decode pose despite lighting is a different question from whether distance is
dominated by it. Fair to test without a trained model, because `VWorldModel.predict()` maps patch
tokens to patch tokens: the predictor's outputs live in the encoder's space.

`eval/phase_b_geometry.py`, 7 poses x 40 lighting draws.

| | within-pose (40 lightings) | between-pose | ratio |
|---|---|---|---|
| 3 terminal states | 276.6 | 437.6 (min 381.1) | **1.58** (worst 1.38) |
| all 7 poses | 280.7 | 466.0 (min 351.2) | 1.66 (worst **1.25**) |

**Nearest-centroid on unseen lighting: 100%**, even with centroids built from only 2 lighting
configs. The clustering step is safe here.

**But 1.58 is a thin margin, not a comfortable one.** Lighting moves latents 63% as far as the gap
between fall-left and fall-right — states that look nothing alike. Across seven poses the worst
case is 1.25.

Two reasons this is an optimistic ceiling rather than the operating value:
1. These are RAW ENCODER outputs. The monitor clusters the PREDICTOR's settled latents, which carry
   model error on top.
2. On Jenga the scene is far busier and the distinction far subtler — a neighbour tipped 15 degrees
   versus standing, not a block flat on its face. **That ratio could easily fall below 1.**

**Action: run `phase_b_geometry.py` on the DINOv2 Jenga latents BEFORE the clustering go/no-go.**
If lighting or shadows dominate distance there, no amount of predictor quality rescues
nearest-centroid, and that is a cheaper thing to learn first than last.

## Phase C: DINO-WM predictor — FAILS as dino_wm trains it, PASSES with rollout fine-tuning

| run | theta RMSE | basin agree | predicted basins |
|---|---|---|---|
| single-step, teacher-forced (**dino_wm's actual recipe**) | 0.3201 | 75.0% | [11, 43, 6] |
| GTF from scratch | 0.4062 | 48.3% | [1, 59, 0] |
| **GTF warm-started from the single-step checkpoint** | **0.2064** | **91.7%** | [15, 31, 14] |
| shPLRNN reference | 0.265 | 87.0% | — |
| ground truth | — | — | [15, 28, 17] |

**Two different questions, and only the first one de-risks Jenga.** `conf/train.yaml` carries
`num_pred: 1 # only supports 1`, so the existing Jenga checkpoint was trained single-step. The
faithful number is therefore **75% / 0.320 / FAIL**, and that is what predicts Jenga. The warm-start
result answers a different question — whether a DINO-WM-architecture model CAN support the monitor
if retrained — and the answer there is yes, comfortably, beating the shPLRNN that sees the exact
state.

I drifted between those two questions without flagging it and was called on it. Recording the
distinction because it decides which number belongs in a paper.

**The failure mode is mean-hedging, and it is visible in the basin counts.** MSE's minimiser is the
conditional mean. Near a bifurcation the model cannot resolve which side of the boundary it is on,
so its conditional distribution over futures is effectively bimodal and the mean sits BETWEEN the
modes — which, with attractors at fall-left / upright / fall-right, is "upright". Predictions
compress toward the middle attractor. Single-step shows it mildly ([11,43,6] vs [15,28,17]);
GTF-from-scratch collapses to it entirely ([1,59,0]).

This is NOT an impossibility result. The block is deterministic: given exact state and actions the
future is unique, so the apparent bimodality is an artefact of the model's finite precision
interacting with sensitive dependence. The uncomfortable part for the paper is that the bias is
**concentrated exactly at decision boundaries** — the model is most accurate where the monitor does
not care and least reliable where it does. It is also why the field builds stochastic world models,
which this method cannot use (it needs determinism to keep `k` measuring action-induced spread).

**Recipe finding, transferable: order matters.** Teacher forcing first to learn the dynamics, THEN
rollout fine-tuning to learn error correction. GTF from scratch saw 8k windows and had not learned
the dynamics before being asked to survive its own errors — it took the safest option and predicted
that nothing ever happens. Warm-started, the same objective fixes the hedging in 19 minutes.
alpha = 0.18 from gtf.py's heuristic `1 - exp(-lambda_max*dt)` at dt_eff = 0.2 s.

**Practical note:** backprop through an 8-step unrolled ViT OOMs a 7.5 GiB card. Gradient
checkpointing (~2x compute, roll-x memory) makes rollout length a modelling choice rather than a
VRAM budget. The Jenga model is larger and dual-view, so this is needed there from the start.

### Two corrections to PLAN_DINOWM, both caught by the user

**1. The eps=0 null is VACUOUS, not a kill test.** The model is deterministic: 32 probes with
identical actions from the same state give 32 identical rollouts and k=0 by construction. It is a
plumbing check (it would catch dropout left on at inference) and nothing more. The measurement I
actually wanted — does model error manufacture spurious dissent — needs the monitor run at the
OPERATING eps on episodes with a LARGE TRUE MARGIN, where no probe should be able to cross a
boundary. Any k > 0 there is the false-positive floor.

**2. The 75% basin figure uses basins I supplied, not discovered.** The diagnostic hardcodes
`where(t < -1.4, 0, where(t > 1.4, 2, 1))` — three bins with boundaries from known physics. That is
fine as an ORACLE score of model fidelity and is labelled as such, but it is not the method. Phase E
must discover the count unsupervised via the merge-distance plateau, and it can fail independently:
the model can land episodes in the right physical state while the latents refuse to form clean
clusters. The 75%/91.7% numbers are evidence about the MODEL, never about the clustering.

## Phase D: the settle tail does NOT converge, and it is EXPANSIVE — qualified FAIL

Two measurements, on both checkpoints. H=20 frames of true action, then L=40 frames of zero force.

**1. Convergence — neither model reaches a fixed point.**

| `||z_t+1 - z_t||` as % of ending scale | step 0 | 4 | 11 | 19 | 29 | 39 | settled (<1%) |
|---|---|---|---|---|---|---|---|
| single-step (dino_wm recipe) | 4.25 | 2.91 | 1.08 | 0.96 | 0.88 | 0.88 | 62% |
| GTF warm-started | 3.68 | 1.80 | 0.75 | 0.63 | 0.60 | 0.60 | 82% |

Both decay and then **flatline** at a nonzero floor rather than converging. The latent never stops
moving; it moves slowly and indefinitely. Bar was 90%, so both FAIL, GTF less badly.

Consequence for Phase E: "settled" latents that keep creeping smear the centroids. Where the tail is
truncated changes the centroid you get, so the merge-distance plateau has to survive drift. That
turns Phase E from "find the clusters" into "find the clusters despite drift".

**2. Contraction — EXPANSIVE.** `||z_model(t) - z_true(t)||` through the tail:

| | step 0 | 19 | 39 | d_end/d_start | contractive on |
|---|---|---|---|---|---|
| single-step | 67.9% | 76.7% | 92.5% | **1.410** | 15% of episodes |
| GTF warm | 62.1% | 66.7% | 75.4% | **1.299** | 20% of episodes |

So the tail amplifies error rather than absorbing it. The hoped-for property -- that errors which do
not cross a basin boundary get absorbed, making Phase C's tracking error survivable -- does not hold.

### Two of my own errors, recorded because both changed the conclusion

**(a) The first contractivity test was invalid.** I compared basin agreement before vs after the
tail (96.7% -> 75.0%) and called it "not contractive". The two tasks are not comparable: at H the
true basins are [2,55,3], so guessing "upright" scores 91.7%, while after the tail they are
[11,35,14]. **The tail is where the outcome is decided -- it changes the true outcome on 20/60
episodes.** The drop measured task difficulty, not contraction. Replaced with the definition:
does `||z_model - z_true||` shrink.

What survives from it: after the tail, against a 58.3% majority baseline, single-step scores 75.0%
and GTF-warm 88.3%, and the hedging persists (predicted [6,50,4] and [9,42,9] vs true [11,35,14]).

**(b) My "displaced manifold" explanation was wrong.** Seeing 62-68% error at the tail's start
alongside 96.7-100% basin agreement, I guessed the predictor's outputs sit on a manifold offset
from the encoder's, which would make the whole comparison irrelevant to a monitor that only
clusters predictions against predictions. Measured: the systematic offset is 5.5-6.5% of scale and
explains **3-4%** of the squared error. Not an offset.

The real explanation is better: **one-step error from a TRUE context is already 26.9% of scale at a
point where theta decodes to 0.040 rad -- essentially perfectly.** The latent is 98,304-dimensional
and theta is one number. The predictor gets the theta-relevant direction right and the rest --
exact shadow pixels, render noise, texture -- wrong. That is irreducible and mostly harmless.

**So the open question is not "how big is the error" but "is it in basin-discriminative
directions".** Nothing measured so far answers that, and Phase E answers it directly: cluster the
PREDICTED ending latents and see whether they form three groups matching the outcomes. Phase D's
numbers bound the difficulty; they do not decide it.

## Phase E: attractor discovery — FAILS in the raw latent space, WORKS after PCA

### The failure, and why it was not the model's fault

The merge-distance plateau from `phase3_pipeline.py` -- the same code that found 3 attractors for
the shPLRNN -- returned **k = 300 at every settle length and every d up to 0.8*scale**. Every
episode its own cluster.

**The CONTROL is what made this diagnosable.** Running the identical discovery on ENCODED TRUE
endings (zero model error) failed the same way. That rules out the predictor and indicts the
procedure.

Cause: single-linkage relies on LOCAL structure (nearest-neighbour chains), which concentrates badly
in 98,304 dimensions. Measured within/between separation in the raw space was **1.09 (encoded truth)
to 1.35 (GTF-warm predictions)** -- no scale gap for a plateau to sit in, so counts decay smoothly
instead of plateauing. The shPLRNN's 4-dimensional latent had entirely different geometry.

Three things were verified healthy first, which is what localised the failure:

| | result |
|---|---|
| physics settles to 3 tight attractors | upright spans theta std **0.012 rad**; 0% "in between" by settle 80 |
| DINOv2 encodes theta | Phase B, R^2 **1.000** |
| nearest-centroid across lighting | Phase B geometry test, **100%** |

Note that nearest-centroid (GLOBAL structure, distance to a mean) worked in the same space where
single-linkage (LOCAL structure) failed completely. High dimensions punish local structure far more.

### The fix: PCA before clustering

| GTF-warm predictions | sep ratio | k-means k=3 agreement |
|---|---|---|
| full (98,304 dim) | 1.354 | 58.7% |
| PCA 2 | **11.644** | 54.3% |
| PCA 8 | 2.165 | **93.7%** |
| PCA 16 | 1.845 | **93.7%** |

**And the ORIGINAL plateau criterion then works**, which matters because it is the calibration-free
one -- the count is whatever survives the widest range of d, nothing supplied:

```
GTF-warm, PCA 2:  0.05:26  0.1:6  0.15:5  0.2:4  0.3:4  0.4:4  0.6:4  0.8:4
                                          ^^^^ plateau at k=4, width 5
```

It reports 4, but the fourth cluster is a **SINGLETON**:

| cluster | n | theta median | true-basin makeup |
|---|---|---|---|
| 0 | 30 | +1.571 | upright 1, fell RIGHT 29 |
| 1 | 225 | +0.000 | fell LEFT 8, upright 209, fell RIGHT 8 |
| 2 | 44 | -1.571 | fell LEFT 42, upright 2 |
| 3 | **1** | +1.571 | fell RIGHT 1 |

So discovery recovers the three real basins plus one stray point that single-linkage never merges.
**With a minimum-cluster-size rule the count is 3, obtained with nothing supplied.** Agreement with
the oracle basins is 93.3% against a 70.7% majority baseline. The 16 misassignments in cluster 1
are the mean-hedging residue, now 5.3%.

### Two findings worth carrying to Jenga

**1. The world model acts as a NUISANCE FILTER, and this is repeatable.** Predicted latents cluster
better than encoded ones at every measurement: separation 1.354 vs 1.091 raw; and in PCA space the
encoded-truth control shows **NO plateau at all** (39, 14, 9, 6, 3, 1, 1, 1 -- a smooth decay) while
the GTF-warm predictions plateau cleanly at width 5. The predictor is trained to model what is
predictable, so it smooths away render noise, exact shadow pixels and texture -- precisely the
nuisance that dominates distance in the encoder's output. **Cluster predictions, never encodings.**
That is also what the monitor does anyway, so the finding is convenient rather than awkward.

**2. Stability-based k selection is unusable here -- it is biased to k=2.** Cross-seed
co-assignment reproducibility was 1.00 at k=2 and 0.70-0.85 at k=3 for every model and every PCA
dimension, so it always picks 2. This is a known weakness of stability criteria. The merge-distance
plateau, in the right space, is the better instrument. Do not substitute stability for it.

### Required additions to the method

- **PCA to ~2-16 dimensions before clustering.** Not optional in a patch-token latent space. The
  dimension is not critical (8 and 16 both give 93.7%) but the raw space does not work at all.
- **A minimum cluster size**, or singletons inflate the discovered count. One stray point out of
  300 turned k=3 into k=4.

Both are unsupervised and neither introduces a tuned threshold, so the calibration-free claim
survives -- but they have to be stated as part of the method rather than discovered per-dataset.

## Phase F: the monitor on DINO-WM — PASSES, and matches the shPLRNN

40 episodes x 3 scoring times, H=20 frames (=200 sim steps, the shPLRNN's horizon), settle 40,
eps=0.10, n=32 probes, k_min=2, clustering in PCA-8 on predicted endings with singletons dropped.
Attractor discovery ran live inside the pipeline: plateau k=4, **3 after dropping singletons,
nothing supplied**.

### 1. False-positive floor — PASS, and this is the structural result

The test that replaced the vacuous eps=0 null. On chunks far from any boundary, no probe should be
able to cross one, so any dissent is model error manufacturing alarm.

| chunks with margin >= | n | mean k | k=0 | k>=2 |
|---|---|---|---|---|
| 0.35 | 83 | 0.29 | **96%** | 4% |
| 0.40 | 80 | 0.30 | 96% | 4% |
| 0.45 | 78 | 0.31 | 96% | 4% |

Bar was k < 2 on >= 95%. **The monitor is measuring boundaries, not the model.** This matters more
than the AUC: Phase D showed the settle tail neither converges (0.60% drift floor) nor contracts
(ratio 1.299), and the obvious worry was that drift would show up as spurious dissent across 32
probes. It does not.

### 2. Detection

| | n near | AUC | 95% CI |
|---|---|---|---|
| per-chunk (120 chunks) | 25 | 0.759 | — |
| per-episode, min CHUNK margin | 18 | 0.833 | [0.714, 0.938] |
| per-episode, full-ACTION margin | 15 | **0.793** | [0.660, 0.924] |

**Only the last row is comparable to the shPLRNN's 0.808**, because that run labelled with
`margin_of(whole 450-step action)` while Phase F labels per chunk. Both monitors SCORE chunks
identically (fixed H, continuous over t); only the label differs, and the shPLRNN's is the flawed
one already recorded above ("label from the full action while the monitor saw one chunk"). The two
labels correlate +0.781 -- related, not interchangeable.

**0.793 vs 0.808 is indistinguishable at n=40.** The operating points differ, though:

| | precision | recall | F1 |
|---|---|---|---|
| DINO-WM (40 eps) | **0.818** | 0.600 | 0.692 |
| shPLRNN (100 eps) | 0.633 | **0.905** | 0.745 |

DINO-WM is precision-favouring, shPLRNN recall-favouring, at the same k_min=2. Not obviously
explained; worth understanding before either is quoted as better.

### What this means

**The monitor transfers from an exact 2-D state to DINOv2 patch features with no retuning** --
same eps, same n, same k_min, same probe family. That is the representation-level universality the
toy was built to test, and it is the strongest result in this plan.

Caveats that must travel with it:
- **n=40, not 100.** CIs are wide and the comparison cannot separate 0.793 from 0.808 either way.
- The DINO-WM model had to be **rollout fine-tuned** (Phase C); dino_wm's shipped single-step recipe
  gives 75% basin agreement and was not carried into Phase F.
- PCA and singleton-dropping are now **required parts of the method**, not incidental.

## Phase F videos: two findings from actually watching them

**1. Phase F UNDERSAMPLED the dissent trace.** The videos score at 8 times per episode
(t=3,6,...,24) against Phase F's 3 (t=3,13,23). On ep565 the 8-time trace peaks at **k=12** while
Phase F recorded **k=4** -- it stepped straight over the spike. `k` spikes and decays rather than
plateauing, so sparse scoring systematically misses peaks.

**Phase F's recall of 0.600 is therefore likely an UNDERESTIMATE**, and the precision-favouring
operating point (0.818/0.600 vs the shPLRNN's 0.633/0.905 at the same k_min) may be partly an
artefact of scoring 3 times rather than a property of the model. Worth a denser rerun before that
asymmetry is explained or quoted. It does NOT affect the false-positive floor, which measured
far-margin chunks where more sampling only adds more zeros.

**2. The false negatives look like the STRUCTURAL BLIND SPOT, not sampling misses.** ep591 at
t=3.0s: block standing in the camera, and all 32 probes unanimous (votes 0/0/32) that it falls
right. Dissent 0.

That is not "the probes failed to reach the boundary". It is confident failure, which dissent
counting cannot see by construction -- unanimity is precisely what k measures the absence of. Same
limitation as the shPLRNN's 28 far-margin topples.

**This corrects a suggestion made earlier in this session.** I proposed larger eps or more probes
as the fix for FNs, reasoning from the binomial sampling floor. That fix addresses misses where
probes fall short of the boundary; it does nothing for unanimity. Only the second alarm clause
would catch these, and that clause was struck as supervised. The honest position is that these FNs
are declared scope.

**Video construction note.** The first attempt rendered 8 frames at 6 fps -- 1.3-second clips,
useless. The monitor updates 8 times per episode but the BLOCK moves continuously and a frame costs
~10 ms, so the camera panel now runs at full temporal resolution (225 frames, 9 s) while the monitor
panels HOLD between scores, with the region past the last scoring time shaded. That is also what a
runtime monitor actually looks like: continuous world, discrete checks. Scores are cached to
`phase_f_video_scores.npz`, so re-rendering never re-runs the model.

## Phase F, dense rerun (100 episodes x 8 scoring times) — SUPERSEDES the sparse numbers

The videos showed `k` spikes and decays, so 3 scoring times per episode was undersampling a
transient. Rerun at 8 times and 100 episodes. **The sparse numbers above are measurement artefacts
and should not be quoted.**

| | sparse (40 eps x 3) | **dense (100 eps x 8)** |
|---|---|---|
| recall | 0.611 | **0.820** |
| precision | 1.000 | 0.872 |
| F1 | 0.759 | **0.845** |
| per-episode AUC | 0.833 | **0.872** |
| per-chunk AUC | 0.759 | **0.804** |

**Recall rose 0.611 -> 0.820.** Sparse scoring was hiding real detections, exactly as predicted from
the ep565 trace (k=12 at 8 times vs k=4 at 3). Precision fell 1.000 -> 0.872 as six false positives
appeared, which is the expected trade: more scoring opportunities, more chances to fire. **So the
precision-favouring operating point flagged earlier as "unexplained" was largely my sampling, not
the model.** That question is closed.

**False-positive floor holds at scale**: 537 far-margin chunks (was 83), k=0 on 96-97%, k>=2 on
3-4%. Unchanged by a 6x larger sample, which is what makes it worth something.

### Like-for-like against the shPLRNN

The per-episode rows above use the min CHUNK margin. The shPLRNN labelled with the FULL-ACTION
margin, so only this row is directly comparable:

| | AUC | 95% CI | precision | recall | F1 |
|---|---|---|---|---|---|
| **DINO-WM** (100 eps, 8 scoring times) | **0.882** | [0.805, 0.949] | 0.766 | 0.857 | **0.809** |
| shPLRNN (100 eps, ~30 scoring times) | 0.808 | — | 0.633 | 0.905 | 0.745 |

**DINO-WM matches or beats the exact-state monitor while being scored ~4x more sparsely.** Since
sparse scoring demonstrably costs recall, its 0.857 is probably still understated.

**Do not overclaim this.** The CI's lower bound is 0.805 and the shPLRNN's point estimate is 0.808,
so the interval just barely includes it. The runs are also not paired (different episode draws), so
no paired test is possible. The supportable statement is **"at least as good, plausibly better"** --
not "better".

### What this settles and what it does not

Settled: the monitor transfers from an exact 2-D state to DINOv2 patch features with no retuning
(same eps, n, k_min, probe family), and the earlier operating-point asymmetry was a sampling
artefact.

Still open: the run used the ROLLOUT FINE-TUNED predictor, not dino_wm's shipped single-step recipe
(75% basin agreement, Phase C). So this validates the architecture, not the existing Jenga
checkpoint. And 8 scoring times is still sparser than the shPLRNN's 30 -- the ceiling has not been
found.

## The settle tail HOVERS — Phase D's "failure" resolved

Phase D recorded that `||z_t+1 - z_t||` decays and then flatlines at 0.60% of ending scale rather
than reaching zero, called it a FAIL, and noted it was unexplained that Phases E and F worked
anyway. **That was the wrong question.** The monitor never reads the latent's position -- only which
attractor it is nearest. So the test should be whether the LABEL is stable while the latent hovers.

Measured, 120 episodes, GTF-warm predictor, basin assignment vs the label at settle step 40:

| settle steps | 10 | 15 | 20 | 25 | 30 | 35 | **40** | 45 | 50 | 60 |
|---|---|---|---|---|---|---|---|---|---|---|
| agreement | 95.0% | 96.7% | 98.3% | 99.2% | 99.2% | 99.2% | **100%** | 100% | 100% | 100% |

**It is a hover INSIDE a basin, not a drift ACROSS basins.** From settle step 25 onward the
assignment is 99%+ stable and it is exactly stable from 40. That is why E and F worked, and the
"unexplained" note above is resolved.

It is also physically faithful rather than a model artefact: a rocking block loses energy at each
base impact (Housner restitution e_r = 1 - 1.5 sin^2(alpha) = 0.824 at alpha = 0.35) and
asymptotically approaches upright without exactly arriving. The model reproduced that.

**Restated acceptance for Phase D:** do not require `||z_t+1 - z_t|| -> 0`. Require that the
ATTRACTOR ASSIGNMENT stops changing. The original criterion measured a quantity the method does not
use, which is the same error shape recorded repeatedly above.

## Tail length: no-tail is ruled out (toy), and Jenga cannot use a held-pose tail at all

**Why this came up.** The Jenga world model has never seen a held pose. Measured over 6771 steps of
demo actions (40 episodes, `jenga_noise_50/jenga_single.lmdb`): per-step EE displacement has a
median of 5.3 mm and a 10th percentile of 2.6 mm, only 0.6% of steps move less than 1 mm, and **the
longest run of consecutive sub-millimetre steps is ONE**. There are zero runs of >= 10 steps. A
40-step zero-action settle tail is therefore pure extrapolation on Jenga -- and Phase D's question
("does it settle?") would be answered by OOD behaviour rather than physics.

This did not arise on the toy because `random_push` fires 3 pulses across 450 steps, leaving ~2/3 of
steps at zero force. I checked that at the time and concluded the settle tail "is not asking for
extrapolation". True for the toy; I had no basis for assuming it carried over.

**Toy measurement, tail = 0** (read the latent at H, no settling), scored against the FULLY SETTLED
outcome:

| tail | separation | plateau k | after singletons | agreement |
|---|---|---|---|---|
| 0 | 1.705 | 2 | 2 | 74.3% |
| 40 (reference) | 2.165 | 4 | **3** | **93.3%** |

**No-tail finds the WRONG NUMBER OF ATTRACTORS (2, not 3)** and scores 74.3% against a 70.7%
majority baseline -- barely above chance. MONITOR.md's warning that mid-flight endings destroy the
attractor structure is confirmed. **Option "drop the tail" is dead.**

The remaining options for Jenga, in order of cost:
1. **Nominal-continuation tail** -- append the policy's own remaining actions instead of freezing.
   The arm lifts and retracts after the grasp in every demo, so a neighbour that was going to topple
   does so while the robot moves away. In-distribution, free, no retraining. One shared tail across
   all probes keeps probe-to-probe differences coming only from the chunk.
2. Fine-tune with held-pose data generated in the MuJoCo sim.

**Bug found and fixed while doing this:** `build_centroids` in `eval/phase_f_monitor.py` crashed with
an obscure `np.stack` error when every cluster fell below `min_size`. It now raises with the plateau
count and cluster sizes. Silently returning the largest few clusters would have been worse -- it
would fabricate attractors out of noise and the caller could not tell.

## JENGA GO/NO-GO: the basins are there, and the arm was hiding them

Encoder only -- no world model, no rollout. 100 labelled episodes, final real frames from
`jenga_noise_50/jenga_single_100.lmdb` against `labels_noise100.json`. `eval/jenga_basins.py`,
`eval/jenga_geometry.py`.

### 1. The basins exist, more cleanly than on the toy

Peak neighbour tilt is sharply bimodal: **75 episodes under 19 deg, 25 above 90 deg, and NOTHING in
between** -- a 72.2 deg gap, zero episodes in the 20-60 deg band. The 45 deg topple threshold sits
in empty space, so the label is not a judgement call. I had hypothesised the opposite (that Jenga
outcomes would be a continuum of partial topples, with no discrete basins to find). **Wrong** -- the
physics is more discrete here than on the tipping block.

### 2. The ARM dominates the raw latents, and removing it is free

PC1 carries 25% of the variance and tracks end-effector pose (|corr| 0.58 with EE z, 0.39 with x);
the topple signal sits in PC2 and PC4 at 11.5% and 5.7%, both entangled with EE x. So keeping the
top components keeps arm motion and dilutes the thing of interest.

Regressing proprio and its quadratic terms out of every latent dimension **uses no topple labels** --
proprio is the robot's own state, available at runtime -- so the method stays calibration-free.
Proprio explains 35.5% of latent variance. The effect:

| | raw | arm removed |
|---|---|---|
| separation (between/within) | 1.502 | **2.146** (toy: 2.2) |
| leave-one-out nearest centroid | 94.0% | **98-99%** (base rate 75%) |
| k-means at k=2 vs topple | **50.0%** (chance) | **98.0%** |
| corr(latent distance, peak tilt) | 0.553 | **0.765** |

**Without arm removal, k-means at k=2 is at chance** -- it clusters purely by arm pose. This is the
single most important preprocessing step found so far for Jenga, and it has no analogue on the toy
(no robot in frame).

The graded number matters as much as the binary one: latent distance from the intact centroid
tracks HOW FAR the block tipped at r = 0.765. That is a proximity signal, which is what this method
actually predicts -- closer to the right target than the topple flag.

### 3. DISCOVERY is the blocker, not the representation

Two independent count-selection methods fail on data where k-means at k=2 gets 98%:

* **merge-distance plateau**: one bridging pair -- ep45 (tilt 94.5 deg) and ep16 (tilt 7.0 deg) --
  sits 0.120*scale apart while the median within-group nearest-neighbour distance is 0.115*scale, a
  ratio of **1.04**. Single-linkage chains straight through it. Raw gives k=1, arm-removed gives
  k=2 but by accident (sizes [99,1]).
* **cross-seed stability**: picks k=3 (0.880) over the true k=2 (0.801), agreeing with the topple
  label only 68% instead of 98%.

**So the structure is findable and the automatic count is not.** That is the one gap between here
and a working Jenga monitor on the representation side.

**Next thing to try:** single-linkage is famously bridge-sensitive; average-linkage or Ward's method
is not. Try those before concluding the count cannot be discovered without supervision.

### What this does NOT yet establish

These are REAL final frames encoded directly. The monitor reads PREDICTED endings from the world
model, which adds prediction error on top, and the settle-tail problem (Jenga demos contain no held
poses at all -- longest sub-millimetre run is ONE step) is still unsolved. This result says the
representation and the physics support the method. It does not say the predictor does.

## Ward linkage discovers the Jenga basins — but breaks the toy. No universal choice yet.

`eval/jenga_linkage.py`. Count criterion is unsupervised throughout: build the dendrogram, cut at
the **largest relative jump in merge height** (many cheap merges inside a clump, then one expensive
merge joining clumps).

**On JENGA (2 true basins, sizes 75/25), arm removed, PCA 2:**

| linkage | k found | agreement | sizes |
|---|---|---|---|
| single | 1 | 75.0% | [100] |
| average | 2 | 74.0% | [99, 1] |
| complete | 2 | 60.0% | [63, 37] |
| **ward** | **2** | **98.0%** | **[75, 25]** |
| *k-means at k=2 (reference)* | *given* | *98.0%* | |

**Ward recovers the exact true split with nothing supplied**, matching k-means that was TOLD k=2.
The clumps are visually unambiguous (`results/jenga/clump*.png`): clump 0 is 25 scenes with the
neighbour flat on the table, mean tilt 89.2 deg; clump 1 is 75 scenes with it standing, mean tilt
12.5 deg. Two errors out of 100 -- ep16 (7 deg, placed with the toppled) and ep36 (91 deg, placed
with the intact). **ep16 is the same episode that bridged the two groups under single-linkage**,
which is a satisfying consistency: the one genuinely ambiguous scene is the one both methods
stumble on.

**But Ward FAILS on the toy**, where single-linkage + plateau succeeded:

| PCA | single-linkage plateau | ward | ward agreement |
|---|---|---|---|
| 2 | 4 | 2 | 83.7% |
| 8 | **4** (3 after singletons, 93.3%) | 1 | 70.7% (= majority baseline) |

**Cause is Ward's known bias toward EQUAL-SIZED clusters** -- it merges to minimise the increase in
within-group variance. The toy's basins are 212/50/38, badly imbalanced, so Ward splits the big
cluster rather than isolating the two small ones. Jenga's 75/25 is balanced enough that it works.

### The honest consequence

**There is no single clustering choice that works for both systems**, and picking the linkage by
which one gives the right answer IS using the labels. The calibration-free claim survives for
thresholds (there is still no tuned delta) but takes a real dent here: the *algorithm* is now a
per-task decision.

Options, none yet tested:
1. A linkage robust to BOTH bridging and size imbalance. Average is the usual compromise and it
   failed here ([99,1]), so this needs actual search rather than assumption.
2. Run several linkages and require consensus; report disagreement as low confidence.
3. Accept a per-task choice and justify it from cluster-size balance, which is observable without
   labels -- but that is close to circular and should be argued carefully if used.

**For Jenga specifically, Ward works and that is what matters right now.** The universality claim
needs the above resolved before it goes in a paper.

## HDBSCAN discovers the basins on BOTH systems — the per-task algorithm choice is gone

`eval/cluster_shootout.py`. Same PCA'd endings as before; only the clustering changes.

| method | TOY (k=3, sizes 212/50/38) | JENGA (k=2, sizes 75/25) |
|---|---|---|
| single-linkage + merge plateau | k=4 -> 3 after singletons, **93.3%** | bridged, [99,1] |
| Ward + merge-height gap | k=1, **70.7%** (= baseline) | k=2, **98.0%** |
| **HDBSCAN, min 5% of n** | **k=3, 93.7%**, 100% coverage | **k=2, 98.8%**, 83% coverage |
| **HDBSCAN, min 10% of n** | **k=3, 93.3%**, 100% coverage | **k=2, 98.8%**, 82% coverage |

**Right count on both, nothing supplied, and it beats the best per-task method on each** (93.7 vs
93.3 on the toy; 98.8 vs 98.0 on Jenga).

**Why it works where the others failed.** HDBSCAN keeps the clusters that persist over the widest
range of DENSITY levels -- the same principle as our merge-distance plateau, but on density rather
than raw distance -- and it has an explicit NOISE label. The bridging point that destroyed
single-linkage gets discarded instead of used as a bridge. It calls 17% of Jenga episodes noise;
that is the mechanism working, not a defect.

**The knob is structural, not fitted.** `min_cluster_size` says "a basin must hold at least this
share of the episodes". 5% and 10% both work on both systems, so it is not knife-edge. At 15% it
returns NOTHING on the toy rather than inventing structure -- correct, since 15% of 300 is 45 and
the smallest true basin has 38.

**For the monitor**, a probe whose ending lands in noise reached no known basin, so it should count
as DISSENT. That is a small addition to the alarm rule and is now recorded in MONITOR.md.

**Caveat:** two systems is not universality. But the previous position -- single-linkage for the
toy, Ward for Jenga, chosen by which matched the labels -- was a genuine hole in the
calibration-free claim, and this closes it for the cases in hand.
