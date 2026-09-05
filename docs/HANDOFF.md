# Handoff

For whoever picks this up — human or agent. README has the results; this is state, traps, and
what to do next.

## State

The **monitor is done and validated**: AUC 0.894 on 1772 chunks, held-out over 20 episode
splits, thresholds calibrated on safe chunks only. `src/monitor.py` is the usable API.
Nothing is mid-flight; every number in the README regenerates from `results/` via
`python eval/analyze.py`.

Two things are *deliberately* not here:

- **Active filtering / steering.** We tried optimising against the divergence score to
  *correct* unsafe actions before execution. It does not work: in paired MuJoCo trials it
  raised the topple rate 15% → 40% (McNemar p = 0.013) while successfully driving the score
  itself down 26%, and a matched random-direction control caused statistically
  indistinguishable harm (p = 0.83) — so the gradient carries no usable control signal. **A
  good passive detector is not automatically a valid control objective.** Do not re-attempt
  this with the same objective; see next steps for the version worth trying.
- **World-model architecture experiments** (Dreamer/RSSM, actor-critic). No usable result for
  the monitor, so they are out of scope for this repo.

## Traps that already cost real time

1. **Silent shape/alignment mismatches.** `ViTPredictor` does `x + self.pos_embedding[:, :n]`
   — it *slices* to whatever token count it gets. Feed it 2 history frames when it was trained
   on 3 and nothing errors; the numbers just quietly get worse. `Monitor._rollout` asserts on
   this now. Also assert that consecutive history frames are ~1 control step apart, not
   several: a loop that refreshes observations once per action *chunk* silently hands the model
   frames 8 steps apart.
2. **Actions must be aligned with frames.** `rollout(obs, act)` treats the first `num_hist`
   entries of `act` as the actions taken *at* the history frames, and the rest as the future.
   Getting this off by a chunk is invisible and corrupts the score.
3. **`sim.py` core-dumps on import without a display** (it builds `SimContext` at module scope
   and opens a viewer). Set `SIM_HEADLESS=1`. EGL then throws at interpreter teardown —
   harmless, it happens after the run, but it is why long evaluations should write results to
   disk incrementally rather than at the end.
4. **Never report accuracy.** 1.4% base rate: predicting "safe" always scores 98.6%.
5. **Resolve the PC1 sign against measured motion, not norm.** The norm heuristic is backwards
   (see README). `fit_pc1(motion=...)` handles it; without `motion` it falls back to a
   convention that may not hold for a new fit.
6. **Anything tuned on the data it is scored on needs a held-out split.** A PCA-truncation
   result looked strong in-sample and evaporated under cross-validation.

## Next steps, in the order I would do them

### 1. Close the readout gap (highest value)
The probe result says the information is *there* and the readout is what is losing it: a
linear probe on the predicted latent hits 0.941 vs divergence's 0.887 on identical chunks.
Options, cheapest first:
- Fit the probe on the *frozen* latents and use it as the monitor score, accepting that it
  needs tilt labels from sim physics and is therefore not zero-shot. Quantify what that buys
  on the operating curve, not just AUC.
- Look for an unsupervised readout that recovers more of the probe's signal — the probe is
  linear, so whatever it is reading is linearly present in the latent and something better
  than "mean cosine drift over masked patches" should exist.

### 2. Settle nominal vs 50-perturbation
Currently unresolved and it is a 40× latency difference (53 ms vs 2021 ms). The two datasets
disagree on the ordering. Run both on a third dataset before quoting either as the default.

### 3. Reduce false alarms
Precision is 0.13 at the p95 operating point, and part of that is arithmetic (ceiling ~0.22)
but part is shadow/reflection artifacts. Per-patch calibration against a safe-trajectory
baseline was scoped but never run.

### 4. Test generalisation properly
Pusher2D replicated the core finding, CartPole did not, and the difference tracks whether the
scene has a "moving actor + static fragile object" structure. A third task with that structure
but different visuals would either confirm or kill that explanation.

### If you want to revisit active filtering
Do not reuse the divergence objective. The one version worth trying is minimising the
**probe's predicted tilt** — a directly physical quantity rather than a latent-space proxy —
in the same paired-trial harness, with a trust region and task-progress guards (pick rate,
endpoint error). A filter that "improves safety" by stalling the arm scores perfectly on
safety metrics alone; that failure mode has already produced one convincing-looking false
positive in this project.

## Do not bother with

- **The FTLE ratio family.** Dominated 0.599 vs 0.894; the denominator is the problem.
- **Combining the low-norm and PC1 masks.** PC1 subsumes it (see README table).
- **Temporal aggregation over chunk scores.** Hurts at this base rate.
- **Exact Jacobian FTLE.** Correct, but the linear regime ends ~50× below the operating
  perturbation magnitude.
