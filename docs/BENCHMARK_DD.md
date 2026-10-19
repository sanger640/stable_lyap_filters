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
Noise sensitivity (without-feedback split): fork share of damage moments 28% at 0.02 (11 split / 28 always), 50% at
0.05, 82% at 0.10 (80 split / 18 always). Forks exist at every level; their share grows with the assumed uncertainty,
which is what the barrier h measures (the smallest uncertainty scale at which futures split).
Caveats: human teleop, not a policy. Check 2 (does our fork test
detect these forks?) needs a world model on this environment; not done.

## Physion — check 1 PASSES for Dominoes (fork prevalence)
Code: ~/wksp/physion/dd (tdw_physics noisy Dominoes controller via dd/noisy_run.py; scene RNG fixed per trial,
noise RNG varied). Perception-scale noise on all dynamic objects: position sd 5 mm (x, z), yaw von Mises kappa 1000
(~1.8 deg). 8 scenes x 16 noise draws per config; outcome = Physion label (target ever contacts zone).
v1 used a config whose outcome is always positive (8/8 originals positive) -> 0 forks, invalid sample (my selection).
v2, the four mixed-outcome configs that make up most of the Dominoes test set:
| config | contacts of 16, per scene | forks (2-14) |
|---|---|---|
| 1mid_J025R45_boxroom | 7 16 12 16 11 12 16 0 | 4/8 |
| 2mid_J020R15_d3chairs_o1plants_tdwroom | 16 16 0 12 0 1 15 0 | 1/8 |
| 2mid_J025R30_tdwroom | 0 15 3 0 0 7 15 7 | 3/8 |
| SJ020_d3chairs_o1plants_tdwroom | 2 16 16 0 0 7 12 1 | 3/8 |
Total 11 of 32 scenes (34%) fork under 5 mm noise (plus 4 near-forks at 1 or 15 of 16). Other scenarios not yet checked.

## Physion — check 2 (does our fork test detect real forks?) — spec fixed 2026-10-19 before any data
- Scenario: Dominoes, the four mixed-outcome configs. Training trials from new seeds (no overlap with test scenes).
- World model: same recipe as Jenga (frozen DINOv2-392 -> channel PCA 8 fitted on training frames; deterministic
  residual FiLM transition via eval/fast_train.py), passive (no actions), one model step = 10 frames, from frame 0.
- Perception uncertainty (the monitor's only uncertainty): object-free Gauss-TE latent field fitted on
  z(noisy re-render, frame 0) - z(clean, frame 0) of training scenes, noise = the same 5 mm / ~2 deg as check 1.
- Monitor on each test scene: clean frame 0 -> 64 Gauss-TE starts at s in (0.5, 1, 2, 3) -> roll out to the end ->
  two-mode split test on the final latents (as Jenga); h = log2 of the smallest splitting s, capped at log2 3.
- Test scenes: the 32 labelled in check 1 + 32 more labelled the same way (new seeds), real fork = 2..14 of 16.
- Gate: AUROC(-h, real fork) >= 0.7 AND alarm rate (h < cap) on real forks >= 2x that on non-forks.
