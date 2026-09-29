# Cross-task contact benchmark prototypes

> **Current-project note (2026-09-29):** these pages establish physical mechanism benchmarks, not
> successful learned rollouts. The incremental Jenga world-model line ended at D21 without opening
> a pushing/insertion predicted-future test. See [`WORLD_MODEL.md`](WORLD_MODEL.md).

These environments are now both visual prototypes and the subjects of the frozen ground-truth U1
universality test. The visual scenes are valid, but U1 rejects the claim that Regime Monitor v0
transfers unchanged from Jenga. The separate natural-episode intervention distribution is
uninformative about failure reduction because it accidentally contains no terminal failures; that
negative benchmark-design result is preserved rather than silently replaced.

Both environments command a controlled tool directly rather than simulating a full robot arm. This
is intentional for the first ground-truth mechanism study: it isolates contact-regime sensitivity
from inverse kinematics, arm collisions, camera perception and controller error. A full arm can be
added after the physical signal passes unchanged across mechanisms.

## Pushing

**Task:** move a free red cube into a green goal while keeping it supported by the table.

**Action:** planar pusher target `(x, y)` at 10 Hz.

**Relevant physical regimes:**

- static/sticking contact versus sliding;
- centered pushing versus glancing deflection;
- supported sliding versus crossing the table edge and falling.

The nominal preview stops the cube in the goal. A stronger nearby push continues through the same
contact and sends it over the edge. This creates a meaningful action-conditioned boundary without
reusing Jenga geometry or a topple-specific definition.

## Insertion

**Task:** insert an orange rectangular peg into a tight square socket.

**Action:** tool target `(x, y, z, yaw)` at 10 Hz.

The peg half-width is 20 mm and the socket half-width is 25 mm, giving 5 mm clearance per side.

**Relevant physical regimes:**

- free approach motion;
- first rim contact;
- aligned insertion;
- one-sided or two-sided contact;
- wedging/rim jam under lateral or angular execution error.

The nominal preview inserts centrally. A 12 mm lateral execution offset contacts the rim and remains
jammed above the socket despite the continued downward command.

## Why these tasks are complementary

| Environment | Desired regime | Sensitive boundary | Failure mechanism |
|---|---|---|---|
| Jenga extraction | controlled grasp/extraction | neighboring contact/topple | loss of support |
| pushing | supported object transport | slide/edge transition | object falls from support |
| insertion | constrained assembly | contact/alignment transition | jam or failed insertion |

A detector that works unchanged on all three has a much stronger universality claim than one that
only recognizes toppling.

## U1 frozen experiment and result (2026-09-23)

U1 uses the exact 64 frozen Jenga tracking-error snippets at 1x, H=8, hold=30, three refined pairs
and five bisections. The monitor sees only three anonymous rigid-body slots represented by position,
rotation-6D, linear velocity and angular velocity. It never sees task outcomes.

The label-only panel was generated and audited before scoring, then frozen with protocol SHA-256
`3f2386af5537ceddfc70ea6ef9959ac3f049ddda16d8243651b6eb8758df641c`. It contains 240 states per
task, split by episode into 120 DEV and 120 TEST states.

| TEST task | consequential forks | initial candidates | refined boundaries | final v0 alarms | quiet alarms |
|---|---:|---:|---:|---:|---:|
| pushing | 12 | 12 | 8 | **0** | 3/108 |
| insertion | 11 | 11 | 3 | **0** | 0/109 |

Both predeclared gates fail. Pushing's refined fork pairs pass persistence but not commitment.
Insertion sometimes has commitment evidence, but not aligned with boundary evidence on two of three
pairs. Candidate discovery also fires on most quiet states, so its dominant unsupervised partition
is not reliably the task-outcome boundary. The canonical detector result has SHA-256
`37d7d00b8dc1404aae5c385795d7c8868a3974707e0365f756cd107b46e4e1c8`.

The frozen 100-episode-per-task intervention follow-up produced no terminal failures in either
baseline: pushing had 48 goal successes and insertion 100 successes. The wrapper therefore had no
failure-prevention opportunity. It reobserved 27 pushing and 9 insertion episodes, changed two
pushing chunks, and changed no outcomes. This is **inconclusive utility evidence**, not a safety
success. The next intervention benchmark must freeze only after a label-only pilot confirms both
successes and failures after an explicit terminal settle.

## Historical pre-freeze checklist

1. Define realistic execution-error snippets in each action space from controller tracking error or
   a prospectively declared proxy. Do not tune them to produce desired failure counts.
2. Sample dense action neighborhoods and confirm that each dataset contains successful, benign
   alternative-contact and consequential branch states.
3. Define task labels for grading only: goal/support for pushing and insertion depth/jam for
   insertion. They must not enter Regime Monitor v0.
4. Freeze development/test seeds, action chunks, simulator source, perturbations and complete state
   snapshots before evaluating v0.
5. Run the unchanged v0 detector and the unchanged 100/75/50/25/0% intervention wrapper.
6. Report branch detection, completion, failures, reobservations, action modification and time to
   success with no-monitor and physical-oracle controls.

Counterfactual snapshots in these prototypes use MuJoCo `mjSTATE_INTEGRATION`, including solver
warm-start state. This prevents planning rollouts from subtly changing the subsequently executed
trajectory.

## Visual outputs

- `results/contact_benchmarks/previews/pushing_goal_vs_edge_fall.mp4`
- `results/contact_benchmarks/previews/insertion_success_vs_rim_jam.mp4`
- `results/contact_benchmarks/previews/preview.jpg`
- `results/contact_benchmarks/previews/manifest.json`
