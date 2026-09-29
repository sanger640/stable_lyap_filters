# Active plan — after D21

## Objective

Develop a universal, task-label-free monitor that warns when realistic action uncertainty places a
robot near a consequential physical regime boundary.

The detector must not require task identity, unsafe-state labels, object names or a threshold fitted
on safe task data. Exact simulator state and cloned physical rollouts are allowed only as diagnostic
oracles. The deployable path must eventually consume generic observations, robot proprioception and
the proposed action.

## Established result

Frozen Regime Monitor v0 works when supplied accurate counterfactual trajectories:

| Benchmark | Boundary recall | Quiet alarms |
|---|---:|---:|
| Jenga physical TEST | 55/84 (65.5%) | 4/113 (3.5%) |
| Upright-push physical TEST | 14/20 (70.0%) | 0/15 (0%) |

The Jenga intervention wrapper also reduced neighbouring-block failures from 34/100 to 25/100.
This supports the monitoring principle, not yet a deployable implementation.

## Unresolved requirement

A real system needs a model to predict nearby futures. Current learned rollouts do not preserve the
physical neighbourhood topology or persistent branch geometry reliably enough. The strongest
conventional baseline, D2, reaches 31.1% mean fork recall across ten seeds. Later models altered
losses, routing, outputs, history and state, but none passed the staged evaluation.

The final bounded combination, D21, improved D2 one-step motion and contacts but failed to retain
D20's configuration-held-out contact F1:

| Axis | D21/D2 continuous NRMSE | D21 contact F1 | Required floor | Gate |
|---|---:|---:|---:|---|
| episode | 0.961× | 0.499 | 0.472 | pass |
| configuration | 0.972× | 0.508 | 0.535 | **fail** |

Per protocol, later rollout and monitor panels stayed closed. This ends the incremental D2–D21
model-tuning line.

D21 protocol/result SHA-256:

- protocol: `d2cad06779201ee07bdf0c3c38d1281f6008966288b2921563ade4a9fa73393c`
- result: `5042545bf579a8b7a9f4a3ab8d827aeff09bbde8fa20517141a29d4816c96b97`
- checkpoint: `d7984994153e18ef9a58e5af4a54ee85eb6db6eb5e36074731ea99fe06957a69`

## Decision

Do **not** run D22 as another residual, router, contact loss, hold-window variation or
Jenga-specific feature change. That would be continued selection on the same evidence rather than a
new test of the research hypothesis.

Choose one of the following programs and freeze its protocol before implementation.

### Program A — online re-observation and active sensing (recommended first)

Replace one long imagined rollout with a short receding interaction:

```text
observe → predict short nearby futures → act a small amount → re-observe → update
```

Why this is the recommended first test:

- It attacks the observed failure directly: topology and branch evidence degrade during open-loop
  prediction.
- It can reuse the frozen monitor and current simulator/world-model interfaces.
- It tests whether fresh observations rescue boundary evidence before funding a much larger model.

Proposed gates:

1. Freeze a causal re-observation schedule and compute budget without reading TEST labels.
2. On held-out TRAIN configurations, verify that repeated short rollouts improve physical pair
   selection and branch-evidence agreement over open-loop D2.
3. Require improvement on both episode- and configuration-held-out axes, with no material increase
   in quiet alarms.
4. Only then open DEV once. Keep physical TEST immutable until the whole protocol passes.
5. Measure latency and intervention delay, because re-observation trades model error for time.

This is not permission to tune the observation interval on topple labels. Use trajectory agreement
and frozen monitor evidence for selection; task labels remain grading-only.

Interpretation is deliberately binary. If frequent exact-state correction materially restores
pair selection and monitor evidence, long open-loop drift is a major cause and the next stage is
rendered re-observation. If correction fails even at one- or two-step segments, local contact
branching is already absent and further scheduling work should stop in favour of Program B.

### Program B — pretrained physical/visual world model

Start a new data/model budget rather than extending D21:

- broad contact-rich interaction pretraining across scenes and mechanisms;
- generic visual or object-centric state with robot proprioception;
- stochastic futures or explicit uncertainty rather than one averaged trajectory;
- objectives that preserve neighbouring-action geometry and contact transitions;
- frozen representation/model selection before Jenga and pushing evaluation.

This program is more expensive and should proceed only if Program A shows that short-horizon
re-observation still cannot preserve the signal, or if suitable pretrained infrastructure becomes
available.

Do not combine the two programs in the first experiment. Keeping them separate makes the result
diagnostic: Program A tests the interaction horizon while holding D2 fixed; Program B changes the
learned physical prior and data scale while preserving the frozen evaluation.

## Evaluation hierarchy for either program

1. **Integrity:** exact data alignment, causal inputs and no task-label leakage.
2. **One-step dynamics:** ordinary motion and contact creation/loss.
3. **Short rollout:** H8 or the preregistered shorter horizon.
4. **Neighbour topology:** does the model select the same nearby pairs as physics?
5. **Response geometry:** does separation remain smooth or persist as a branch at the same scales?
6. **Frozen monitor agreement:** boundary, commitment, persistence and final decisions.
7. **Prospective DEV/TEST:** only after all earlier gates pass.

Average reconstruction or rollout error is never sufficient by itself.

## Standing rules

- Monitor v0 and existing TEST results remain frozen.
- Report recall, quiet alarms and intervention burden, never headline accuracy.
- Use at least two independent held-out axes: episode and configuration.
- Stop at a failed preregistered gate; do not tune from later panels.
- Separate universal inputs from privileged diagnostics and grading labels in every result.
- Record protocol, result and checkpoint hashes for every accepted run.
- Keep detailed chronology in [NOTES.md](NOTES.md), synthesis in
  [WORLD_MODEL.md](WORLD_MODEL.md), and superseded plans in
  [docs/archive/README.md](docs/archive/README.md).

## Intended claim

> Accurate counterfactual trajectories support a useful task-label-free signal of proximity to
> execution-sensitive physical regime boundaries. Standard learned world models can erase that
> signal despite reasonable predictive accuracy; deployability therefore requires either models
> trained to preserve local intervention geometry or interaction designs that repeatedly
> re-observe and limit open-loop prediction.
