# Benchmark recheck (2026-10-21 night)

The user suspected that our "no forks" verdicts on SafeLIBERO, SafeManip, ToolHang and ManiSkill came from errors on
our side. This file checks every benchmark against the same list; results are filled in as the checks finish.

Checklist per benchmark:
1. **Noise level.** Is the execution noise comparable to Jenga's? Reference: in the Jenga closed-loop tests (scale 2)
   the gripper target is offset by a median 3.2 mm at the chunk's end (scale 3: 4.8 mm).
2. **Restore exactness.** Does a restored state replay exactly? This includes hidden state the flattened simulator
   state omits: robosuite's ramped gripper command, MuJoCo's solver warm start, and controller state.
3. **Zero-noise control.** Identical replays must not "fork".
4. **Labels.** Do the outcome labels capture the task's failure, or only damage events?
5. **Look-ahead.** How long after the chunk is the outcome judged?
6. **Power.** Are there enough replays per moment to see a split?

| benchmark | 1 noise | 2 restore | 3 control | 4 labels | 5 look-ahead | 6 power |
|---|---|---|---|---|---|---|
| SafeLIBERO | sigma 0.15 (3x the robosuite checks); not low | gripper and warm-start bug found and fixed 2026-10-19; 600/800 episodes pre-fix, with f1-f7 starting from a slightly different hidden state (this adds spread, i.e. biases toward MORE splits) | not run | contact / damage | chunk + 10 hold | **weak: 3 futures (f0-f2) per moment**, so a 20%-bad moment shows as a split only ~half the time; the 3.8% fork rate may be understated ~2x |
| SafeManip | sigma 0.05 normalised OSC (same as ToolHang) | snapshot keeps grippers, controllers, warm start, step counter; exact-replay check | not run | **damage only** (slip / unstable release / knocked) | chunk + 20 hold | 32 replays (2b-powered) |
| ToolHang | same as SafeManip; realised gripper deviation: see the recheck | same snapshot; exact replay <= 0.6 mm | recheck running (seed 1, scale 0) | **damage only**; recheck adds a generic pose label + success-flag split | chunk + 20 hold | 32 |
| ManiSkill StackPyramid | joint-target sigma 0.004 rad | GPU set_state_dict; controllers stateless (absolute joint targets) | **0 forks** (4 demos, 94 points) | generic pose label + success flag | chunk + 20 hold | 32 |
| ManiSkill LiftPeg (PPO demos) | realised gripper deviation 3.7 mm at the chunk's end (about Jenga scale 2.3) | GPU; parallel envs differ by ~0.01 mm (float) | **NOT clean: 2 strict forks with zero noise vs 5 at 1x on the same 10 demos (46 points)**, so ~40% of the strict forks are GPU float chaos and the "20 / 192 PASS" is marginal (noise-attributable ~12) | strict stand vs flat | chunk + 20 hold | 32 |
| ManiSkill LiftPeg DP (planner demos) | 3.7 mm | as above | **0 forks** (1024 points) | strict | chunk + 20 hold | 32 |
| MimicGen StackThree | sigma 0.05 normalised OSC | **bug found and fixed before any result**: restoring the dataset's saved states lost the gripper's ramp state (zero-noise control showed 32/32 replays differing at one point); now rebuilt from the demo's gripper commands, and the warm start is zeroed for every replay | **0 / 0 after the fix** (2 demos, 67 points) | cube falls > 2 cm or tilts > 45 deg vs all within 1 cm | chunk + 20 hold | 32 |

## Findings so far
- Noise was not too small anywhere; SafeLIBERO used 3x more.
- No restore bug affected a reported result. SafeManip and ToolHang restored the hidden state from the start;
  SafeLIBERO's bug was fixed and documented on 10-19. Tonight's MimicGen bug was caught by the control before results.
- Real weaknesses:
  - damage-only labels (SafeManip, ToolHang);
  - the short look-ahead, which is the measure aligned with what the filter can act on (see BENCHMARK_DD.md
    "Re-check");
  - low power in SafeLIBERO;
  - ManiSkill GPU float nondeterminism, which inflated LiftPeg's strict fork count by ~40%; its pass is marginal.
