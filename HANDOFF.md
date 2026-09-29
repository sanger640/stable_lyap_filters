# Handoff — stability-based safety monitor

## Read this first

The project has validated a calibration-free structural monitor when counterfactual trajectories
are accurate. It has **not** produced a deployable learned world model that preserves the required
local intervention geometry.

The latest bounded experiment, D21, is the stopping point for incremental state-world-model tuning.
It improved D2 but failed one predeclared held-out contact-retention gate. Do not continue with a
D22 variant without defining a materially new research program.

Recommended reading order:

1. [README.md](README.md) — question and headline evidence.
2. [MONITOR.md](MONITOR.md) — current monitor and its scope.
3. [WORLD_MODEL.md](WORLD_MODEL.md) — complete model synthesis.
4. [PLAN_NEXT.md](PLAN_NEXT.md) — permitted next directions.
5. [NOTES.md](NOTES.md) — chronological evidence when exact provenance is needed.

## Current scientific state

### What worked

- Frozen Regime Monitor v0 uses no task/failure labels, safe calibration set or fitted alarm
  threshold.
- Physical Jenga TEST: **55/84** topple forks and **4/113** quiet alarms.
- Independent upright-push TEST: **14/20** mixed topple forks and **0/15** stable-centred alarms.
- In 100 Jenga episodes, the conservative intervention wrapper reduced neighbouring-block failures
  **34 → 25** and increased safe completions **51 → 59**, while pick success stayed **80 → 79**.
- On common successful episodes, monitoring added about **0.23 s mean / 0.20 s median** to first
  pick success.

These results establish useful boundary sensitivity and intervention value under exact simulator
rollouts. They do not imply every alarm is a failure prediction.

### What did not work

The learned model must preserve *differences between nearby actions*, not merely predict plausible
average motion. D2 remains the strongest conventional privileged-state baseline but reaches only
**31.1% mean Jenga fork recall** across ten seeds, versus **65.5%** for physical trajectories.

The decisive crossover diagnostic found two independent failures:

- D2-selected pairs evaluated physically: **0/23** topple alarms — wrong neighbourhood topology.
- Physically selected pairs evaluated through D2: **4/23** — branch dynamics are also smoothed.
- Physical selection and physical trajectories: **18/23**.

D3–D21 tested boundary-scale losses, switching modes, relational objectives, trajectory mixtures,
direct response prediction, learned routing, explicit curves, direct monitor evidence, anonymous
graphs, short history, richer Markov state, supervised contact transitions and a D2 residual. The
full staged gate was never passed.

### Final bounded model result: D21

D21 starts as an exact frozen-D2 identity and learns only enhanced-state/contact residuals.

| Held-out TRAIN axis | D2 NRMSE | D21 NRMSE | D2 contact F1 | D21 contact F1 | Required contact floor |
|---|---:|---:|---:|---:|---:|
| episode | 0.462 | **0.444** | 0.426 | **0.499** | 0.472 |
| configuration | 0.466 | **0.453** | 0.431 | **0.508** | **0.535** |

It improves D2 on all measured quantities but misses D20's best configuration contact floor. The
frozen gate therefore fails. H8, response curves, monitor agreement, DEV and TEST were not opened.

Canonical files:

- `results/jenga/d21_d2_enhanced_residual_protocol.json`
- `results/jenga/d21_d2_enhanced_residual_result.json`
- `results/jenga/d20_enhanced_hybrid_result.json`
- `results/jenga/regime_monitor_v0/d2/aggregate.json`

Hashes are recorded in [PLAN_NEXT.md](PLAN_NEXT.md) and [NOTES.md](NOTES.md).

## Operational state

Repository:

```bash
cd /home/corey/wksp/stable_lyap_filters
git status --short
source .venv/bin/activate
python -m pytest tests/ -q
```

The suite currently contains **400 tests**. Batch simulator jobs must use `SIM_HEADLESS=1`. Keep
worker counts conservative: exact replay is memory- and simulator-heavy, and prior high-parallelism
runs froze the machine.

Verify v0 before any physical evaluation:

```bash
python eval/jenga_regime_monitor_v0_freeze.py verify
```

Protocol SHA-256:
`4e1a20ca3c64bf18733315fe31df5850529027f8f48cda2c5f547ce43ecdc317`.

## Experimental discipline

- Keep task labels grading-only; never train or select the monitor with topple/success labels.
- Keep physical monitor v0 frozen. Its TEST panels are immutable.
- Evaluate learned models in order: one-step dynamics → H8 rollout → local action-response geometry
  → monitor agreement → DEV/TEST.
- Stop at the first failed gate. Do not inspect later panels to rescue an arm.
- Report both recall and quiet-alarm rate; accuracy is misleading under class imbalance.
- Mark every feature as universal/deployable, privileged diagnostic, or grading-only.
- Do not reinterpret simulator cloning or exact block state as deployable perception.

## What the next agent should do

No automatic model run is pending. First choose and preregister one of two genuinely different
programs:

1. **Pretrained physical/visual world model:** new interaction data and broader pretraining,
   stochastic futures, object persistence and 3-D geometry; freeze before benchmark evaluation.
2. **Online re-observation/active sensing:** execute smaller increments, observe again and update
   the counterfactual set, reducing open-loop horizon and treating repeated disagreement/model
   uncertainty as evidence.

The second is the lower-cost next scientific test because it directly attacks compounded rollout
error without another Jenga-specific architecture sweep. A detailed rationale and falsification
logic for both directions is in [WORLD_MODEL.md](WORLD_MODEL.md).

For a re-observation takeover, do not begin with a new model. First freeze a simulator-state
diagnostic using existing D2 and v0: split H8 into short causal segments, replace D2's imagined
state with the exact observed state between segments, and compare pair-selection and monitor-stage
agreement against open-loop D2 on both held-out TRAIN axes. Keep DEV and TEST closed. A positive
result licenses a rendered-observation version; a negative result at one- or two-step horizons
points toward the larger/pretrained-model direction instead.

## Documentation policy

Active documents stay short and represent the current truth. Detailed experiment results are
append-only in [NOTES.md](NOTES.md). Superseded plans are indexed in
[docs/archive/README.md](docs/archive/README.md); do not restore their “next step” sections as
current guidance.
