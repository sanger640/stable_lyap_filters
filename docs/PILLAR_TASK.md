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
