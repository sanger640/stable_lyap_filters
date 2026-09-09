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
