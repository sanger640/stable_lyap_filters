# World models in the stability-monitor project

## Why the monitor needs one

The monitor asks a counterfactual question before the robot acts:

> If the next command is executed with several small, realistic errors, do the resulting futures
> remain one smooth family, or split into persistent physical regimes?

In simulation, those futures can be obtained by cloning the exact state and replaying every nearby
action. A real robot cannot clone reality. A **world model** must therefore predict the same set of
nearby futures from the current observation and proposed actions.

```text
current observation + nearby action probes
                    ↓
             learned world model
                    ↓
          counterfactual trajectories
                    ↓
       frozen calibration-free monitor
                    ↓
        smooth region or branch boundary
```

The world model is not asked to classify topples, failures or task success. It must preserve local
action-response geometry: two nearly identical commands that lead to different persistent physical
regimes must remain different in its predictions. This is harder than obtaining low average
next-state error. A model can predict plausible-looking futures while smoothing away exactly the
rare contact transition the monitor needs.

## What counts as success

The evaluation hierarchy is deliberately staged:

1. **One-step dynamics:** predict ordinary motion and contact creation/loss on held-out TRAIN
   episodes and configurations.
2. **H8 rollout:** preserve motion over the eight commanded steps without compounding drift.
3. **Neighbouring-action geometry:** reproduce how trajectory separation changes as action probes
   approach one another.
4. **Monitor agreement:** reproduce the boundary, commitment, persistence and final decisions of
   the frozen physical monitor.
5. **DEV/TEST:** opened only after all earlier gates pass.

Task labels are grading-only. A model is never selected because it predicts “topple” or “success.”

## The reference ceiling and the deployment gap

With exact simulator trajectories, Regime Monitor v0 detects **55/84** Jenga topple forks with
**4/113** quiet alarms. On the independent upright-push TEST it detects **14/20** mixed topple
forks and remains quiet on **15/15** stable centered pushes. This establishes that useful structural
evidence exists when trajectories are accurate.

The strongest established learned baseline is D2, a privileged-state graph dynamics model trained
with intervention-consistency losses. Across ten Jenga seeds it recovers only **31.1%** of topple
forks on average, versus **65.5%** for physical rollouts. It is also not visually deployable: it
receives exact simulator poses, velocities and contacts.

The problem is therefore not simply “predict the next state.” It is:

> Learn futures whose *differences under neighboring actions* match physical differences closely
> enough for the same unsupervised monitor to make the same decision.

## Iterations and what each taught us

| Iteration family | Main change | Result and lesson |
|---|---|---|
| Visual DINO-WM baseline | Predicted future image latents from recent frames and actions | Predicted branches collapsed. Rollout fine-tuning reduced latent MSE roughly tenfold but barely increased stable/topple separation. Average prediction loss was not the right target. |
| W5/W6, D0 | Step-wise privileged-state graph dynamics | Supplied a physically interpretable baseline and contacts, but long rollouts smoothed sharp response changes. |
| D1–D2 | Branch loss, then intervention-consistency loss on paired neighboring actions | D2 became the strongest conventional baseline. It improved action response but retained only about half the physical monitor's useful fork signal. |
| D3 and switching-edge studies | Boundary-scale supervision, hard contacts and latent interaction modes | Additional boundary losses did not pass; latent switching modes collapsed or became nearly constant. Merely adding a discrete gate did not make modes identifiable. |
| D4–D5 | Relational counterfactual training and multimodal complete-trajectory prediction | D4 improved TRAIN geometry but failed matched DEV. D5's mixture did not improve commitment at its frozen gate. Multimodality without reliable mode selection was insufficient. |
| D6–D9 | Set-conditioned response models, explicit curves and temporal/action encoders | These models could memorize small training sets and represent the curves. Held-out specificity and final alarm recall still failed, showing that raw capacity was not the main problem. |
| D10–D12 | Multimodal curve experts, fixed routing and a larger generic interaction corpus | Oracle components contained better alternatives, but learned routing could not select them reliably. Another 576 balanced contact/motion/quiet groups did not repair held-out geometry. |
| D13 | Predict continuous monitor evidence directly | False positives fell, but commitment and final recall collapsed. Predicting the score was easier to regularize but did not recover missed physical branches. |
| D14 | Anonymous object/contact graph instead of an ordered Jenga vector | Evidence error improved, but final recall remained **29.5%/34.3%** on episode/configuration validation. Better structure did not solve held-out commitment. |
| D15–D17 | Alias audits, short causal history, richer proprioception and force/impulse state | History did not resolve aliases. Proprioception helped local purity; force features did not. These were observability diagnostics, not successful rollout models. |
| D18–D19 | Audited and shortened the physical commitment target | Commitment has real tail sensitivity, but instability did not explain model misses. A five-held-step version retained only **23.6%/23.7%** of full positives and was rejected. Physical v0 stayed unchanged. |
| D20 | From-scratch enhanced-state graph model with supervised contact modes | Contact-event F1 improved, but continuous one-step error became **1.262×/1.205×** D2. Rich state helped contact recognition but the replacement model lost ordinary dynamics. |
| D21 | Frozen D2 plus zero-initialized enhanced residual | Continuous error improved by **3.9%/2.8%** and contact F1 improved over D2. Configuration contact F1 was **0.508**, below D20's frozen **0.535** floor, so the combined gate failed and no rollout evaluation was opened. |

## The most important diagnostic

Oracle crossover experiments separated pair selection from trajectory prediction:

- D2-selected paths evaluated with physical trajectories still produced **0/23** topple alarms.
- Physically selected paths evaluated with D2 trajectories produced only **4/23** alarms.
- Physical selection with physical trajectories produced **18/23**.

So there are two independent failures:

1. the learned neighborhood topology chooses the wrong nearby branch pairs;
2. even on the correct pairs, learned trajectories lose physical boundary and commitment evidence.

Late settling is not the main issue. D2 usually preserves persistence after a split; it fails to
locate and represent the split itself.

## Current conclusion

The project has established a useful **monitoring algorithm under accurate counterfactual
trajectories**. It has not established a deployable learned source of those trajectories.

D21 was the final bounded incremental test. It improved D2 but failed one predeclared held-out gate.
The correct conclusion is not that world models are impossible, nor that the extra state is useless.
It is narrower:

> At the present data and model scale, small supervised architecture, routing and loss changes do
> not preserve local intervention geometry reliably enough to support the monitor.

No D22 residual, router, contact loss, hold-window change or Jenga-specific feature sweep is
authorized. Continuing that sequence would be result-driven tuning on the same benchmark.

## Materially different future directions

Further work should begin only as a new program with its own protocol and evaluation budget. The
two directions test different explanations for the deployment gap.

### Direction 1 — substantially larger or pretrained physical/visual world model

**Hypothesis:** local physical branches are learnable, but the present Jenga-scale data and model
family do not contain enough varied contact experience or the right physical prior.

This direction changes both scale and training distribution. It is not D22 with a wider GNN. A
candidate should be pretrained on broad contact-rich interaction data and should represent object
persistence, 3-D geometry, contact creation/loss and multiple plausible futures. Ordinary
deterministic regression can minimize average error by predicting between two outcomes; a
stochastic or multimodal model must preserve the alternatives instead.

Two stages keep perception and dynamics identifiable:

1. **Pretrained physical/object model:** consume generic object state and robot proprioception to
   test whether broader pretraining repairs counterfactual dynamics under good perception. Exact
   simulator state makes this a privileged diagnostic, not a deployment claim.
2. **Visual model:** consume camera history, proprioception and action, then predict generic visual
   or object-centric futures. This is the deployable target, attempted only after the physical
   model preserves local branches.

The model must be frozen before Jenga/pushing evaluation. Success requires more than low video or
next-state error: it must recover physical neighbouring pairs, preserve separation-versus-action
scale curves, and reproduce frozen-v0 boundary, commitment, persistence and final decisions on
both episode- and configuration-held-out data.

This is the closest route to a fully predictive monitor, but it needs a genuinely new data and
compute budget and risks becoming a foundation-world-model project. It is justified if short
re-observation cannot recover the signal or suitable pretrained infrastructure becomes available.

### Direction 2 — online re-observation and active sensing

**Hypothesis:** asking one learned model to predict the full action-plus-settle trajectory is
unnecessarily difficult; repeated short predictions corrected by reality may preserve enough local
geometry for the same frozen monitor.

The current open-loop path feeds each prediction into the next prediction. A small contact error
can therefore put the rollout in the wrong regime and contaminate every later state. Re-observation
periodically replaces that imagined state with a real observation:

```text
observe → predict short nearby futures → execute a small segment
        → re-observe → correct the state → predict again
```

The model is still necessary because only one action is executed and the alternatives remain
counterfactual. What changes is its burden: it predicts short local alternatives rather than a
38-step future from one initial state. Active sensing extends this idea by slowing, pausing, making
a bounded reversible probe, changing viewpoint, or reading force/torque when the current regime is
ambiguous. Those actions gather generic physical information; they must not encode a Jenga-specific
topple rule.

The first bounded test should reuse frozen D2 and v0. Split H8 into a preregistered short schedule,
replace predicted state with the exact simulator observation after each executed segment, and
recompute the remaining nearby futures. Start on held-out TRAIN episode and configuration axes.
Measure:

- agreement with physical neighbouring-pair selection;
- boundary, commitment, persistence and final-decision agreement;
- fork recall and quiet-alarm burden, used only for frozen grading;
- observation count, latency and delay before intervention.

Do not tune the observation interval on topple labels. Advance to DEV only if both held-out TRAIN
axes improve over open-loop D2 without a material quiet-alarm regression; keep TEST closed until
the complete protocol passes. If correction helps, replace exact state progressively with rendered
observation and proprioception. If even one- or two-step corrected predictions miss the branch, the
problem is local representation rather than accumulated rollout drift, strengthening the case for
Direction 1.

### Decision rule

| Direction | Question it answers | Cost | Recommended order |
|---|---|---:|---:|
| Re-observation/active sensing | Can feedback remove the need for accurate long open-loop futures? | lower | **first** |
| Larger/pretrained model | Can broad physical learning represent the local branches directly? | high | second or parallel if infrastructure exists |

A third valid outcome is to publish the monitor, physical cross-task evidence and learned-rollout
failure as a benchmark for whether world models preserve local causal geometry. None of these paths
should be implemented as another small Jenga-specific model variation.

## Canonical artifacts

- Physical monitor: `results/jenga/regime_monitor_v0/test_ground_truth.json`
- D2 learned monitor: `results/jenga/regime_monitor_v0/d2/`
- Pair-selection crossover: `results/jenga/pair_selection_crossover_summary.json`
- D20 enhanced trajectories: `results/jenga/d20_enhanced_trajectory_data.npz`
- D20 result: `results/jenga/d20_enhanced_hybrid_result.json`
- D21 result: `results/jenga/d21_d2_enhanced_residual_result.json`
- Active decision: [`PLAN_NEXT.md`](PLAN_NEXT.md)
- Chronological record: [`NOTES.md`](NOTES.md)
