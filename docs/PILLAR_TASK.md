# Pillar task: a custom second task for the filter (started 2026-10-22 00:00)

User (2026-10-22 ~00:00): "develop a custom test environment in MuJoCo to test our filter ... similar and relatively
simple but different enough we can claim generalisability".

**Task: precarious placement** (eval/pillar_task.py). It keeps Jenga's robot, controller, action format, camera and
execution-noise model. Pick a cube (3 cm) from the table and set it on top of a narrow upright pillar (10 cm tall),
then release and retreat. Success: the cube rests on the pillar top, and the pillar is upright and has not moved
> 2 cm. Failures: the cube falls off, or the pillar tips.

Difference from Jenga: placement instead of extraction; the failure is the load falling off / the support tipping
instead of a neighbour toppling; the critical moment is the release, not the pull.

Scripted expert (expert_plan): straight-line waypoints at <= 12 mm per 0.1 s action. With no noise it succeeds 4/4
(cube within 1 mm of centre). Under Jenga's closed-loop execution noise (2x snippets): 85% (20 episodes);
at 3x: 75%.

## Quick fork check (eval/pillar_task.py fork; results/pillar/)
Along the expert's noisy (2x) closed loop, branch at every 8-action chunk 33 ways (exact + 32 with fresh 2x snippets on
the chunk). Two look-aheads:
- HOLD: chunk + 20 hold, generic label vs the exact replay (cube > 2 cm lower or pillar > 45 deg more tilted vs both
  within 1 cm). This is what the filter imagines.
- REST: chunk + the rest of the expert plan + settle, task-success label. Placement outcomes are decided at the
  release, after the descent; CPU MuJoCo is deterministic, so a zero-noise control is meaningful.

Zero-noise control (8 episodes): 0 forks in both modes.

Pillar half-width 12.5 mm (2.5 cm top), 32 episodes, 277 points:
- HOLD: 11 forks / 13 points with any bad (85%) / 0 all-bad / 4.0%.
- REST: 13 / 25 / 3 / 4.7%.
The failure profile is Jenga's (noise-decided), but there are too few failures.

**Width selection rule (fixed 00:25, before trying any other width):** the narrowest pillar half-width in {12.5, 10,
8} mm at which the expert succeeds >= 60% under 2x noise (40 episodes). The fork check is then rerun at that width
(64 episodes). It passes if the HOLD or REST label meets the three usual bars. The rule picks the task's difficulty
from the expert's success rate, not from fork counts.

## Width selection + fork check RESULT (2026-10-22 00:30)
Expert success under 2x noise (40 episodes): half-width 10 mm 85%, 8 mm 80% -> both >= 60% -> **8 mm (1.6 cm top)
selected by the rule.** Fork check at 8 mm, 64 episodes, 551 points (results/pillar/fork_w8.json):
- **HOLD: 42 forks / 50 points with any bad (84%) / 0 all-bad / 7.6% -> PASS** (all three bars).
- **REST: 114 / 163 (70%) / 21 / 20.7% -> PASS.**
- Zero-noise control (8 episodes, 68 points): 0 forks in both modes (results/pillar/fork_w8_control.json).
The pillar task is a second fork benchmark with Jenga's failure profile, from a different contact mode.
Next: a chunked closed-loop policy (expert re-planning every chunk; a Diffusion Policy from expert demos later),
then world-model data from the fixed camera, the world-model check, and closed-loop filter tests, as for Jenga.
World-model camera (chosen 00:45 from 4 rendered candidates): `cam_fixed` at (0.5, -0.55, 0.62) aimed at (0.5, -0.02,
0.49), fovy 45: side-on to the pillar, so the cube on top and any tilt / fall are visible in profile and the arm only
enters the top of the frame. A front view puts the gripper over the pillar top at the critical moment; the views from
the robot's side and the far corner are dominated by the arm. Videos: dashboard "Pillar (custom)".

## Closed-loop policy (01:00)
ExpertPolicy (eval/pillar_task.py): Jenga's interface (16-target plan per decision, first 8 executed), re-plans from the
current state every chunk, recovers drops. First version oscillated (re-planned "rise first" during the descent; fixed).
With TRUE object poses it succeeds 40/40 with and without 2x noise: re-planning corrects every error before the release,
so there is nothing for a filter to fix. Real policies see the scene through a camera, so the expert gets a perception
error on the pillar's xy: per-episode offset ~ N(0, sigma) + per-decision jitter ~ N(0, sigma/3).
**Rule (fixed 01:00, before trying any sigma):** the smallest sigma in {2, 3, 4, 5} mm whose closed-loop success under
2x noise (40 episodes) is in [60%, 90%] (the policy-success band used for ToolHang / StackThree). Then the closed-loop
fork check on this policy (HOLD + REST-of-episode-closed-loop labels, zero-noise control).
Sigma sweep (2x noise, 40 episodes): 2 mm 90%, 3 mm 92.5%, 4 mm 77.5%, 5 mm 60% -> **2 mm selected** (smallest in
[60%, 90%]).

## Closed-loop fork check RESULT (2026-10-22 01:00; results/pillar/closed_fork.json, closed_control.json)
ExpertPolicy with 2 mm perception error, pillar 8 mm half-width, 2x execution noise, 64 episodes, 1097 points; at every
chunk 33 copies (the policy's internal random state copied, so only the chunk's noise differs):
- **HOLD (chunk + 20 hold, generic label): 61 forks / 105 with any bad (58%) / 0 all-bad / 5.6% -> PASS.**
- **REST (the policy re-plans to the episode's end, task label): 95 / 171 (56%) / 49 all-bad / 8.7% -> PASS.**
- Zero-noise control (8 episodes, 138 points): 0 forks in both modes.
The pillar task has a closed-loop chunked policy whose own rollouts fork. Next: world-model data from the side camera
(label-free noisy rollouts of this policy), world-model check (does the imagined fork test see the real forks?), then
closed-loop filter tests (G-CEM), as for Jenga.

## World-model check (spec fixed 2026-10-22 01:25, before any data; eval/pillar_wm.py)
SafeManip 2b-WM recipe on the pillar side camera. Training data (label-free): closed-loop policy episodes, seeds
3000-3099, every 2nd decision, exact + 4 noisy replays of chunk (8) + hold (22), frames every 2 actions with robot masks.
Held-out eval: seeds 4000-4023, every decision, exact + 32 noisy replays, generic hold labels. DINOv2 -> channel PCA 8
(fitted on training start frames, robot patches excluded) -> 28x28x8; FiLM transition (fast_train), input = targets
relative to the start gripper position + gripper + start proprio. Imagined fork score h with the real branching noise
model at s in {0.5, 1, 2} (cap log2 3). **Gate (as SafeManip): AUROC(-h, real fork) >= 0.7 AND imagined split rate at
s = 1 on forks >= 2x non-forks.** If it passes: closed-loop filter tests (G-CEM with this world model), spec first.

**Result (2026-10-22 03:50): FAIL.** 362 held-out points, 23 real forks; AUROC(-h, fork) = 0.54; imagined split rate at
s = 1: forks 0.26 vs non-forks 0.14 (1.84x; bar 2x). Training: 7500 steps, final loss ~0.006.
Diagnostic, after the result and not a gate: the same split test applied to the REAL latents of the 32 noisy replays
also fires at non-forks (end of hold: forks 0.35 vs non-forks 0.31), while the largest real latent distance from the
exact replay at the end of the hold separates forks well (AUROC 0.93). So the representation sees the outcome; the
split statistic is the weak link on this task.

### World-model check v2 (spec fixed 2026-10-22 03:52, commit 85b0470, before computing it; same model, no retraining)
Score d = the largest distance, over 32 imagined noisy replays (real branching noise, s = 1), between the imagined
end-of-hold latent and the imagined zero-noise replay's end-of-hold latent. Fresh held-out eval seeds 4100-4123 (data
generated after this spec; same protocol as seeds 4000-4023). **Gate: AUROC(d, real fork) >= 0.7 on seeds 4100-4123.**
Also reported, not gated: the same on seeds 4000-4023. If it passes: closed-loop filter spec next.

**Result v2 (2026-10-22 04:00): PASS.** Seeds 4100-4123: 405 points, 27 real forks, **AUROC(d, fork) = 0.78**. Seeds
4000-4023 (not gated): 0.84. The world model trained label-free on the pillar camera ranks the policy's real forks.

## Closed-loop filter: universality test (spec fixed 2026-10-22 04:06, commit cc554ad, before any test episode; eval/pillar_filter.py)
This is the test pre-registered in docs/OPT_FILTER_REDESIGN.md ("the frozen family and ranges, with no retuning, on a
second task"). Frozen from Jenga: the G family (speed, pause, amplitude, back-off along the chunk's own reverse
direction; gripper on the policy's clock; anchored at the current target), its box, CEM settings (population 16, 3
iterations, 6 elites, 4 finals, sd (0.4, 3, 0.4, 12 mm)), opt2's cost (eta 4), and the fallback order (cheapest
certified, else hold if certified, else best). Task-specific, by necessity: the world model (pillar camera) and its
validated score d (v2 above) in place of Jenga's (h, o). Certificate: d <= TAU on 32 fresh noise draws.
TAU = 23.48, the 90th percentile of d over the policy's own chunks at 912 training decisions (label-free;
results/pillar/filter/tau.json). Trigger: d(policy chunk) > TAU.
- Arms: record (policy alone; d logged, never acted on), hold (trigger -> hold one chunk, re-plan), g (trigger -> G-CEM).
- Episodes: fresh seeds 5000-5299 (300), paired by seed (same layout, perception bias, noise sequence). Execution noise
  2x Jenga snippets, pillar 8 mm, perception sigma 2 mm (the configuration that passed the fork check).
- Failure = cube_fell OR pillar_tipped at the end; success = `outcome()["success"]`.
- **H1 (primary): G reduces failures vs record**, one-sided exact McNemar on discordant pairs, alpha 0.05.
- **H2: G's success is non-inferior to record**: paired bootstrap (10,000) one-sided 95% lower bound on the success
  difference > -5 pp.
- Universality is claimed only if H1 and H2 both pass. Secondary (reported, not gated): hold vs record (same tests),
  G vs hold (two-sided McNemar on failure and success), trigger rate, intervention kinds, chosen theta, and on the record
  arm the AUROC of the episode's largest d vs failure.
- Smoke (seeds 9000-9001, not test seeds; checked only for crashes and timing): runs; 1-2 triggers per episode.

**Result (2026-10-22 ~05:30): FAIL; universality NOT claimed.** 300 paired episodes (results/pillar/filter_result.json).
Failures / successes: policy 33 / 267, hold-and-replan 63 / 237, G-CEM 43 / 257 (every failure is a cube fall; pillar
tipped 18 / 27 / 26). H1 (G fewer failures): 30 G-only vs 20 policy-only failures, p = 0.94 -> FAIL. H2 (success
non-inferior): -3.3 pp, lower bound -7.0 -> FAIL. Secondary: hold is worse than the policy (failures 63 vs 33; success
-10 pp); G beats hold (p = 0.015) but not the policy. Trigger rate 9-14% of decisions; G: 607 certified, 46 hold, 62 best
fallbacks; median theta (v 1.0, k 0, s 0.92, b 4.5 mm); 0.4 s per decision. On the policy arm the episode's largest d
does not predict failure (AUROC 0.50; failure 12% in triggered vs 5.6% in untriggered episodes).
Exploratory (after the result, not tests): the perception bias does not explain failures (|bias| vs failure AUROC
0.46-0.55). Held chunks also receive execution noise, and the hold arm ran 6261 decisions vs the policy's 5109; on this
task waiting near the pillar adds noisy exposure instead of removing risk. Hypothesis, untested: a fork-ranking world
model (moment-level AUROC 0.78) is not enough when the fix (slower / smaller / back off / hold) does not remove the
noise's effect at the moment of placement.

## Diagnostic of the failed filter test (exploratory, 2026-10-22; eval/pillar_diag.py, results/pillar/diag/)
True episode-failure risk of one intervention (16 continuation draws, policy alone afterwards) on the record arm.
- A, fixes at the 459 triggered moments: mean risk policy 12.0%, G 14.3%, hold 15.1%, still (hold, no noise) 13.6%;
  G better / worse than the policy at 79 / 149 moments. At the 84 triggered moments with policy risk >= 25%: 36.5% ->
  G 31.7%, hold 31.7%, still 28.5%. The fixes help a little only where the risk is high, and the trigger rarely
  selects such moments.
- B, trigger vs true risk (all moments of the 33 failed + 33 control episodes): AUROC(d, risk >= 25%) 0.55; the trigger
  catches 9% of the risky moments; in 32 of 33 failed episodes the moment where the risk first reaches 50% is not
  triggered.
- Where failures are decided: in 31 of 33 failed episodes the risk is <= ~0.2 through the carry and jumps to ~1.0 at the
  RELEASE decision: a rare noise draw in the last descent chunk leaves the cube off-centre, and opening the gripper
  then makes the fall certain. At the release every future fails, so there is no spread for d (and the fork score) to
  see ("no fork is not safe"), and the filter cannot act there because the gripper column is never changed.
- Release check (scratchpad script, results/pillar/diag/release_check.json): at 28 release moments (27 doomed, policy
  risk 97%), keeping the gripper closed and lifting 3 cm for one chunk, then the policy alone: risk 68%.
Conclusion: on the pillar the world model's fork ranking is fine (0.78), but the pipeline fails at three points: (1) the
score sees spread, not certain failure (needs Jenga's disturbance readout o); (2) the decisive action is the gripper,
which G never touches; (3) holding is not a safe backup.
