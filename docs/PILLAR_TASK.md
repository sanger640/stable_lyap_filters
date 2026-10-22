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
