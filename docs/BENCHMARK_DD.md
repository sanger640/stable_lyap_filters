# Benchmark due diligence (before building anything on a new benchmark)

Rule (2026-10-19, after SafeLIBERO): first measure (1) how often real outcomes FORK under small perturbations and
(2) whether our fork test detects those real forks; report to the user before building a pipeline.

## SafeLIBERO (closed: negative)
Real contact forks 3.8% of untouched-obstacle decision points (~21% of contact moments); world model caught 0 of 13
real forks. See docs/FINDINGS.md.

## OopsieBench (RoboCasa pick_egg) — check 1 PASSES (fork prevalence)
Code: eval/oopsie_fork_check.py (copy of ~/wksp/oopsie/dd/fork_check.py); results/benchmark_dd/fork_*.json.
Recorded teleop demos, decision points every 20 steps; restore full state (sim state, solver warm start, gripper's
internal command, damage-evaluator velocities), reset egg health, replay the next 20 recorded actions (1 s) 9 times:
exact + 8 with smoothed Gaussian arm noise (0.05 xyz, 0.017 rot). Damage = egg health drop >= 10.
| split | decision points | never | always | any split | forks 2-6/8 | fork share of damage moments | exact vs recorded |
|---|---|---|---|---|---|---|---|
| without live feedback | 222 | 176 | 23 | 23 | 15 | 50% | 97% agree |
| with live feedback | 376 | 366 | 2 | 8 | 4 | 80% | 100% agree |
Caveats: human teleop, not a policy; one noise level (sensitivity at 0.02 / 0.10 next). Check 2 (does our fork test
detect these forks?) needs a world model on this environment; not done.

## Physion — check 1 in progress
Dominoes v1 used a config whose outcome is always positive (8/8 originals positive) -> 0 forks, invalid sample.
v2 runs the four mixed-outcome configs that make up most of the Dominoes test set.
