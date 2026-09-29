# stable_lyap_filters

A research testbed for a **task-label-free runtime safety monitor**. The monitor asks whether small,
realistic changes to a proposed robot action produce one smooth family of futures or split into
persistent physical regimes.

The central result is now clear:

- With accurate simulator counterfactuals, the frozen monitor detects useful execution-sensitive
  boundaries in Jenga picking and upright block pushing.
- With the current learned world models, those local branches are often smoothed away or the wrong
  neighbouring actions are selected.
- D2–D21 explored conventional dynamics, branch-aware losses, multimodality, routing, explicit
  response curves, richer state and supervised contact modes. None passed the full staged gate.
  Incremental tuning of this model family is closed.

See [WORLD_MODEL.md](WORLD_MODEL.md) for the model history and evidence, and
[PLAN_NEXT.md](PLAN_NEXT.md) for the active decision.

## Research question

Can a robot detect that its proposed action lies near a consequential physical regime boundary
without being told the task, what failure looks like, or a threshold fitted from safe examples?

```text
current observation + intended action
                 ↓
      nearby execution-error probes
                 ↓
       counterfactual trajectories
                 ↓
  smooth response or persistent branch?
                 ↓
             quiet / alarm
```

An alarm means that nearby realizations of the action have persistently different consequences.
It does **not** mean the nominal action must fail. An action that fails under every nearby probe can
also be quiet; this is a boundary monitor, not a universal failure classifier.

## Frozen monitor

Regime Monitor v0 uses the same task-independent procedure on both 3-D tasks:

1. Apply measured execution residuals to the next eight commands.
2. Generate 64 nearby action trajectories and hold for 30 steps.
3. Fit smooth-versus-branched action-response explanations with a complexity penalty.
4. Refine candidate boundaries by repeatedly bisecting nearby action pairs.
5. Alarm when a majority of three selected pairs retain boundary, commitment and persistence
   evidence as their actions become nearly identical.

No topple label, goal label, safe calibration set or fitted alarm threshold is used by the monitor.
Task labels are used only after freezing the method to evaluate it.

## Main evidence

| Evaluation | Result |
|---|---|
| Jenga physical TEST | 55/84 topple forks detected; 4/113 quiet alarms |
| Upright-push physical TEST | 14/20 mixed topple forks; 0/15 stable-centred alarms |
| Jenga intervention study | neighbour failures 34/100 → 25/100; safe completions 51/100 → 59/100 |
| D2 learned Jenga rollouts, 10 seeds | 31.1% mean fork recall vs 65.5% physical; 1.95% quiet alarms |

The physical results validate the monitoring idea under accurate futures. They do not establish a
deployable system, because simulator cloning is unavailable on a real robot and D2 is a privileged-
state model. The learned-rollout gap is the active scientific limitation.

## Documentation map

| File | Purpose |
|---|---|
| [HANDOFF.md](HANDOFF.md) | Short operational handoff and exact current state |
| [MONITOR.md](MONITOR.md) | Monitor definitions, assumptions and frozen v0 |
| [WORLD_MODEL.md](WORLD_MODEL.md) | Why a model is needed, D0–D21 history and conclusion |
| [PLAN_NEXT.md](PLAN_NEXT.md) | Current decision and allowed next research programs |
| [JENGA_3D_TASKS.md](JENGA_3D_TASKS.md) | Presentation-ready Jenga pick and upright-push results |
| [PANDA_BLOCK_PUSH.md](PANDA_BLOCK_PUSH.md) | Upright-push benchmark details |
| [TOY_EXPERIMENT.md](TOY_EXPERIMENT.md) | Original 2-D basin/dissent experiment |
| [CONTACT_BENCHMARKS.md](CONTACT_BENCHMARKS.md) | Earlier pushing and insertion mechanism prototypes |
| [NOTES.md](NOTES.md) | Append-only chronological experiment record |
| [docs/archive/README.md](docs/archive/README.md) | Index of superseded plans and legacy formulations |

## Repository layout

```text
src/                 monitor, dynamics and system implementations
eval/                frozen protocols, data builders and evaluations
tests/               unit and integrity tests
results/             immutable protocols, metrics, checkpoints and media
media/               figures and short visual explanations
docs/archive/        superseded plans and legacy method documents
```

## Setup and verification

```bash
cd /home/corey/wksp/stable_lyap_filters
./scripts/setup_env.sh
source .venv/bin/activate
python -m pytest tests/ -q
```

External simulator/model paths are resolved by `src/paths.py`:

```bash
export DINO_WM_DIR=/path/to/dino_wm
export PANDA_EXPRESS_DIR=/path/to/panda_express
export DINO_WM_CKPT=$DINO_WM_DIR/outputs/model_latest_single.pth
export SIM_HEADLESS=1
```

Verify the immutable physical monitor before using its evaluator:

```bash
python eval/jenga_regime_monitor_v0_freeze.py verify
```

Frozen v0 protocol SHA-256:
`4e1a20ca3c64bf18733315fe31df5850529027f8f48cda2c5f547ce43ecdc317`.

## Current boundary

Do not create a D22 residual, router, contact-loss variant, hold-window sweep or Jenga-specific
feature tweak. D21 was the declared final incremental combination test. A legitimate continuation
must be a materially different program: either a substantially larger/pretrained physical or
visual world model with new data, or an online re-observation/active-sensing formulation that
reduces dependence on long open-loop prediction.
