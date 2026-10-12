# Full paper plan (draft 0, 2026-10-12) — not the submitted workshop versions

Working title: **Learned Intuitive Physics for Fragile-Action Detection: World Models that Branch Where Physics Does**

Target: RSS 2027 (deadline to verify; typically late Jan/early Feb), fallback RA-L / IROS.

## One-sentence claim
A world model trained per task from images, with deterministic dynamics rolled from sampled perception
errors, label-free rare-branch rebalancing and an error-floor split test, flags fragile robot actions as
well as a physics engine does (with fewer false alarms), and as a safety filter prevents failures;
generative and large pretrained world models do not, and lower prediction error does not imply branching.

## Contributions
1. **Branching fidelity**: a goal-free test (64 futures under the robot's own uncertainty; split test)
   and the observation that average-error metrics miss it (transformer result: lowest loss, worst branching).
2. **A recipe that works**: deterministic dynamics + object-free start sampler (Gauss-TE) + label-free
   rare-branch rebalancing (US) + hold-30 split test with model-error floor. 3-block task: 85-89% / 4-5%
   on two fresh panels vs MuJoCo probe 89% / 15%.
3. **Multiple tasks, each with its own monitor and filter** (T1 pick-from-row, T2 push-past-tall-object,
   T3 pick-from-stack): monitor accuracy and failure-rate reduction from the filter.
4. **Real robot** (Franka Panda): repetition-based real fork labels, monitor and filter on T1.
5. **Negative results / limits**: stochastic generative WMs (incl. a FlowWM-recipe baseline), V-JEPA 2-AC
   zero-shot, and cross-scene generalisation (Phase 2 diagnosis: readout fine, dynamics do not branch).

## Experiments (status)
| # | Experiment | Status |
|---|---|---|
| E1 | T1 monitor, selection on H, fresh panels 6 and 7 | done (Phase 1e + addendum 3) |
| E2 | Baselines on T1: MuJoCo probe, cell E, R (labels), generative cells A-D, V-JEPA zero-shot | mostly done; FlowWM-recipe baseline TODO |
| E3 | T1 safety filter (offset-variant search, monitor-in-the-loop) vs no filter vs MuJoCo-probe filter | protocol + code: in progress |
| E4 | T2 push-past-tall-object: data, monitor, panel, filter | generator: in progress |
| E5 | T3 pick-from-stack: data, monitor, panel, filter | reuse Phase-2 F2 generator |
| E6 | Real robot T1: setup, real fork labels by repetition, perception/execution error, monitor, filter | planning |
| E7 | Phase 2 generalisation study (limits section) | in progress (hybrid, data scale-up) |
| E8 | Ablations: rebalancing (E vs U vs US), readout (P/BOTH/H30/H30_floor), sampler (G/Gauss/Gauss-TE) | data exist |

## Figures
1. Pipeline (image -> DINOv2 -> latent -> 64 rollouts from sampled starts -> split test -> filter).
2. Branching fidelity across model families (generative vs deterministic vs ours) + loss-vs-branching plot.
3. Per-task monitor results vs MuJoCo probe (bars with Wilson CIs).
4. Filter: failure rate and action deviation per task.
5. Real robot: setup photo, real fork labels, monitor/filter results.
6. Phase 2: where the gap is (ceiling vs models, per family).

## Sections
1 Intro (IPE motivation, Battaglia; the question; contributions) · 2 Related work (intuitive physics & Physion /
Physics-IQ; world models DINO-WM, V-JEPA 2-AC, Cosmos, FlowWM, Foresight Diffusion; uncertainty-aware WMs and
latent safety filters; ensemble/scoring-rule training; Yu et al.; Toso et al.) · 3 Problem & branching fidelity ·
4 Method (sampler, dynamics, rebalancing, readout, filter) · 5 Experiments (E1-E6, E8) · 6 Limits (E7) · 7 Conclusion.

## Timeline (weeks from 2026-10-12)
1-2: T2/T3 data + monitors; T1 filter. 3-4: T2/T3 filters; FlowWM baseline; real-robot setup.
5-7: real fork labels, sim-to-real fine-tune, real eval. 8-10: writing, ablations, buffer.
