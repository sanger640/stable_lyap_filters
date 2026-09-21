# Active plan (revised 2026-09-21)

`PLAN_WORLDMODEL.md` is now the historical record of W0-W6. This file is the plan being executed.

## Terminology: V1 and V2 are different pipelines

**V1 — vision as state estimator (the current working system).**

```
image -> frozen DINO -> estimated object state -> state-space GNN dynamics -> fork monitor
```

An object-centric spatial readout that predicts block centres, orientations, corners or gaps is
STILL V1, as long as those quantities are fed into the existing state-space GNN. Changing how the
state is read off the image does not make it a visual world model.

**V2 — direct visual/latent world model (not started).**

```
video -> DINO or V-JEPA tokens -> action-conditioned latent dynamics -> predicted future tokens
      -> fork score
```

V2 removes the explicit 61-dim state bottleneck. Proprioception stays available directly in both
pipelines and is never reconstructed from vision. Conceptually analogous to V-JEPA 2-AC [2].

**V2 does not begin until the V1 dynamics limitations are characterised**, because V1 already
matches privileged performance at the realistic 1x operating point.

## Where the bottleneck actually is

| | 1x recall | blind forks |
|---|---|---|
| privileged state + W6 dynamics | 67% | **24/84 (29%)** |
| DINO + proprioception + same dynamics | 67% | 25/84 (30%) |

Perception is no longer the 1x bottleneck. The dominant remaining 1x limitation is **dynamics**:
roughly 24 of 84 fork states are blind even when the initial physical state is exact. The question
for the next phase is therefore:

> Why does the learned model fail to respond to some real action interventions even when its
> initial physical state is exact?

The one place perception still loses is 2x: 83% vs 99% privileged, which Step E handles separately.

---

## Step A — Characterise the privileged blind forks

For every privileged-state blind fork, record: nominal state; nominal action; the counterfactual
perturbations; real and predicted trajectory spread OVER TIME; the time real trajectories begin to
diverge; contact-state changes; pose and contact configuration at divergence; perturbation
direction; whether the real branch is transient or persistent; and whether the model predicts
**zero** response or merely an **insufficient** one.

Cluster by physical mechanism where possible: contact creation/breaking, stick/slip, support loss,
tipping onset, or another repeated event.

**No architecture changes before this analysis.**

## Step B — A CoCo-inspired counterfactual objective

Reference [1] (CoCo, Counterfactual Consistency). Its motivation matches a failure this project has
already seen: an action-conditioned model can score well by exploiting inertia rather than
responding to the action, so different actions give nearly identical futures and a zero action
still produces drift.

CoCo's components are Multi-Step Counterfactual Consistency (reference, inverse-action and
zero-action rollouts), Action-Spatial Counterfactual Consistency, and Action Response / Drift
Energy metrics.

**Adapt the intervention principle rather than copying it.** This project has stronger supervision
than CoCo assumes: genuine simulator counterfactual rollouts from exactly the same state.

## Step C — Counterfactual consistency vs the old branch loss

The existing branch loss matches predicted pairwise branch MAGNITUDE to real pairwise magnitude. It
does not distinguish appropriate action response from artificial drift, and says nothing about the
direction or structure of the intervention.

For state `x` and actions `a`, `a + d`, `a - d`, and optionally a no-op:

```
real  intervention effect:  D_real = traj(x, a + d) - traj(x, a)
pred  intervention effect:  D_pred = pred_traj(x, a + d) - pred_traj(x, a)
```

Train `D_pred` to agree with the STRUCTURE of `D_real`, evaluating at minimum: whether the
intervention should produce little change; whether it should produce large change; WHEN in the
rollout divergence occurs; and whether it persists. **Not endpoint Euclidean distance alone.**

Keep the successful one-step + 0.5 input-noise baseline. **Do not reintroduce the long rollout
curriculum** — it halved recall and drove fork separation to ~0.05 (NOTES.md, W6).

| arm | training |
|---|---|
| D0 | one-step + noise (the current winner) |
| D1 | one-step + noise + old branch-distance loss |
| D2 | one-step + noise + counterfactual consistency |

Primary evaluation: recall at 1/3/5% FPR, privileged blind-fork count, quiet p99, fork spread, AUC.

**Success condition: recover blind forks WITHOUT recreating runaway quiet rollouts.** Every quiet
tail so far has been the thing that costs recall at strict thresholds.

## Step D — Perception frozen during the dynamics experiment

Run Steps B and C on **privileged state only**. Do not introduce DINO initially. This preserves
causal attribution:

```
perfect state -> does the dynamics improve?   then   vision state -> does the improvement survive?
```

If D2 improves privileged recall, re-evaluate that exact trained model through the existing
DINO + proprioception V1 pipeline.

## Step E — Diagnose the residual 2x visual gap

1x is at parity; 2x is not (83% vs 99%). Before building anything, inspect the vision-only missed
2x forks with ground-truth substitutions for block position, orientation, position+orientation, and
contacts — the same intervention method that showed velocity was irrelevant and joint pose was not.

**Build the object-centric spatial readout only if accurate geometry demonstrably recovers those
states.** Evaluate both image resolutions through downstream fork recall and blind forks, never
through reconstruction RMSE alone. Note the readout must beat the ~3.6 mm relative-pose floor that
a global-PCA readout saturates at, not merely match it.

## Step F — V2: a direct latent world model

After V1 and the dynamics experiments. One baseline, V-JEPA 2 / 2-AC style [2]: frozen visual
encoder + proprioception + action -> action-conditioned latent future predictor. The aim is not to
reproduce V-JEPA 2-AC but to answer:

> Does a direct visual latent world model preserve the fork signal better or worse than the
> explicit-state V1 pipeline?

Same fork benchmark, same perturbations.

## Step G — Dense visual representation baseline

V-JEPA 2.1 [3] targets spatially structured, temporally grounded dense features. Because DINO V1
already matches privileged 1x performance, this is a **publication baseline, not an emergency
replacement**. Run it after the main dynamics experiments, unless the Step E diagnosis points back
at representation quality.

## Step H — Contact/tactile extension

ContactWorld [4] reports that spatially structured representations and compatible tactile
information improve predictive planning on contact-rich tasks. **Do not add force/torque merely
because it is available.** Add it as a controlled sensory ablation after the visual and dynamics
baselines exist, and evaluate whether it reduces blind forks, improves recall at fixed FPR, and
reduces ambiguity specifically around contact transitions.

## Step I — Counterfactual evaluation methodology

Twin Rollouts [5] formalises evaluation where two trajectories share the same generated history and
the same exogenous noise and differ only in the action after an intervention time. This is closely
aligned with the same-state / action-perturbation protocol already used here, and with the unused
common-random-numbers support in `state_dynamics.rollout`. Use as related work and methodology
justification. **Not an implemented robotics baseline** — that paper's experiments are stated as
forthcoming.

## Step J — Safety positioning

[6] argues predictive likelihood and visual fidelity alone are insufficient for safety-critical
world models, distinguishing likelihood from risk, prediction from intervention, and single-step
error from accumulated consequence. Use for paper positioning. This project supplies an empirical
robotics realisation of the related question: does the world model preserve action-conditioned
branches accurately enough for runtime safety monitoring?

---

## Immediate execution order

1. Characterise the privileged-state blind forks (Step A).
2. Implement a CoCo-inspired counterfactual objective on the privileged-state GNN (Step B).
3. Compare D0 / D1 / D2 directly (Step C).
4. Keep perception frozen throughout (Step D).
5. Diagnose the residual 2x visual misses (Step E).
6. Build an object-centric readout only if GT-pose interventions show it would solve them.
7. Re-evaluate the improved dynamics through the DINO + proprioception V1 pipeline.
8. Add V-JEPA 2.1 and V-JEPA 2-AC-style V2 later as publication baselines (Steps F, G).
9. Add ContactWorld-inspired force/tactile sensing as a subsequent ablation (Step H).

## References

Supplied with the revised plan. **Not verified against the papers themselves** — several postdate
what is available here, so titles, arXiv identifiers and claimed contents should be checked before
they appear in a submission.

[1] Shi et al., 2026. *Overcoming Statistical Bias in Action-Controllable World Models* (CoCo /
    Counterfactual Consistency). arXiv:2608.04653. — counterfactual training, action-response and
    drift motivation.
[2] Assran et al., 2025. *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction
    and Planning.* arXiv:2506.09985. — direct latent action-conditioned world-model baseline;
    proprioception alongside vision.
[3] Mur-Labadia et al., 2026. *V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised
    Learning.* arXiv:2603.14482. — dense/spatial visual representation baseline.
[4] Zhang et al., 2026. *ContactWorld: What Representations Matter in Vision-Tactile World Models
    for Contact-Rich Manipulation.* arXiv:2606.13877. — object/spatial representation, later
    tactile and force/torque experiments.
[5] Ma, Shi, Xu, 2026. *Twin Rollouts: Noise-Coupled Counterfactual Branching in Interactive Video
    World Models.* arXiv:2608.08982. — counterfactual branching evaluation methodology.
[6] Ma et al., 2026. *Rethinking World Models for Safety-Critical Embodied Systems.*
    arXiv:2609.03774. — safety-oriented motivation and positioning.

## Standing rules carried forward

* Grade on batch 3 (84 forks at 1x) with the frozen protocol; report recall at MATCHED FPR
  alongside the pre-declared Gate 3 point.
* >= 10 seeds per arm. Three-seed estimates have been off by ~10 points three separate times.
* Never select an arm on validation: it comes from the training episode pool and understated the
  rollout-curriculum damage by an order of magnitude.
* Judge readouts and representations by fork recall, blind forks and quiet-tail statistics, never
  by reconstruction RMSE alone — velocity was the worst-reconstructed group and the least relevant.
* Take end-effector pose from proprioception, never from vision.
