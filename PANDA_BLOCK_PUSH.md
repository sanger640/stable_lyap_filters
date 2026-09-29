# Panda single-block pushing environment

This replaces the abstract blue-pusher prototype. A Panda end effector uses the same operational-
space controller, assets, table, camera convention and Jenga dimensions as the existing simulator.
One red Jenga block stands upright and must be pushed into a non-colliding green goal without ever
toppling.

The environment, physical-state v0 diagnostic, and rendered-future representation diagnostics are
complete. No world-model-predicted-future result has been run. The Jenga state-world-model ladder
is now closed at D21, so Panda prediction should not be opened with those rejected checkpoints.
It resumes only under a materially new pretrained/large-scale model program; see
[`WORLD_MODEL.md`](WORLD_MODEL.md).

## Task and grading

- Action: absolute end-effector `(x, y, z)` target plus gripper command at 10 Hz.
- Object: one free 75 × 25 × 20 mm Jenga block standing upright.
- Success: centre within 25 mm of goal, current tilt at most 10°, still supported, and never toppled.
- Failure: peak tilt reaches 45° at any time, even if the fallen block later enters the goal.
- Mechanisms: no-contact/contact, maintained/lost contact, centered/off-center pushing,
  translation/rotation and goal/overshoot.
- Counterfactual snapshots: MuJoCo integration state, including solver warm-start.

## Visual validation

| case | result |
|---|---|
| low centered push | success; 11.3 mm goal error; peak tilt 13.5° |
| excessive low push | incomplete but not toppled; overshoots goal by 40.8 mm |
| higher centered push | failure; peak tilt 96.2°; fallen block ends inside goal area |
| peel-away path | incomplete but not toppled; contact lost; block stops 35.0 mm away |

Outputs are in `results/panda_block_push/previews/`; implementation is in
`src/systems/panda_block_push.py` and `eval/panda_block_push_previews.py`.

Do not launch another 100-episode intervention benchmark yet. First create a small DEV mechanism
panel from this task and Jenga, develop a more general ground-truth transition-law monitor, and test
whether rendered and world-model trajectories preserve its decisions. Reserve new pushing episodes
before freezing a detector.

## Ground-truth DEV panel and v0 diagnostic (2026-09-24)

The requested small mechanism panel is complete and frozen before monitor scoring under protocol
SHA-256 `20dc10ea8eb455fceb958e0c409f94fdeafacddfb577ef479db6b443742d1aed`.
It contains 56 states selected using physical labels only:

- 20 mixed upright/topple forks (21–43 of 64 probes topple);
- 15 stable centered pushes;
- 10 states where all 64 probes topple;
- five non-toppling contact-loss controls;
- six non-toppling overshoot controls.

Unchanged Regime Monitor v0 gives:

| cohort | states | candidates | refined boundaries | commitment majority | persistence majority | final alarms |
|---|---:|---:|---:|---:|---:|---:|
| mixed upright/topple | 20 | 20 | 19 | 20 | 20 | **19** |
| stable centered | 15 | 0 | 0 | 0 | 0 | **0** |
| all probes topple | 10 | 8 | 7 | 3 | 5 | **1** |
| contact-loss control | 5 | 5 | 5 | 5 | 5 | **5** |
| upright overshoot | 6 | 4 | 4 | 0 | 4 | **0** |

This is strong DEV evidence that v0 can recognize a second real robot/contact mechanism: it has
95% recall on local upright/topple forks and is completely quiet on stable centered pushes. The
contact-loss alarms are not false positives under the Layer-1 definition: nearby futures enter
different maintained/lost-contact regimes even though none topple. Conversely, one unanimous-
topple state alarms because its already-failing futures still branch physically; v0 remains a
regime detector, not an outcome classifier.

The only missed fork passes commitment and persistence but fails boundary refinement (0/3 refined
pairs), so commitment is not the bottleneck here. This is DEV evidence only and must not be used as
a pushing TEST claim. Result SHA-256:
`5168479e7a058955b49b48748d6fa9338385057a365bee1b909cdeab24fee47e`.

## Rendered-future visual transfer (2026-09-24)

All 3,584 exact frozen initial counterfactuals were re-simulated and rendered at hold steps
5/10/20/29/30. Visual-selected refinement midpoints and both final refined endpoints were then
rendered, with the endpoints recorded densely over all 38 H8+hold30 steps. Every rendered initial
replay reproduced its frozen physical topple count.

The predeclared visual adapter uses the existing frozen DINOv2-S/14 encoder, fixed 4x4 spatial
pooling, and a seeded 45-D Gaussian projection. It uses no task labels, fitted threshold, reference
states, or outcome data. The v0 monitor functions are unchanged. The 45-D projection is an adapter
to v0's physical-state-shaped consequence input, so this is a representation diagnostic rather
than a final visual-monitor formulation.

| cohort | states | candidates | refined | commitment majority | persistence majority | visual alarms | physical alarms |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed upright/topple | 20 | 20 | 19 | 8 | 19 | **7** | **19** |
| stable centered | 15 | 15 | 15 | 1 | 14 | **1** | **0** |
| all probes topple | 10 | 10 | 9 | 4 | 10 | **4** | **1** |
| contact-loss control | 5 | 5 | 5 | 2 | 5 | **1** | **5** |
| upright overshoot | 6 | 6 | 6 | 1 | 6 | **1** | **0** |

Final visual/physical decisions agree on 33/56 states (58.9%). The visual adapter recovers 8/25
physical alarms overall and 6/19 mixed upright/topple alarms (31.6%), while producing six alarms
where physical v0 is quiet. Candidate discovery, refinement and persistence are almost saturated;
the principal loss is commitment. Full-frame visual changes from the robot/controller are too easy
to branch, while the random 45-D projection does not preserve the physical action-to-consequence
geometry needed by the final stage.

This fails representation Gate 4 for the baseline full-frame adapter. Do **not** evaluate the world
model yet: predicted features cannot repair a decision already lost in encoded true frames. Next
run the predeclared representation ladder on this same DEV panel: fixed background-suppressed DINO
patch features, then a simulator-segmented block-region diagnostic upper bound. Select only by
agreement with frozen physical v0, freeze the preprocessing, and evaluate once on a fresh Panda
TEST panel before moving to predicted futures.

Artifacts are in `results/panda_block_push/visual_v0/`; the result SHA-256 is
`8a84de40e877bc3f3adab33f74b2d42d067c1ab96af61d62afb27f1f43e4bb33`.

### PCA45 adapter follow-up

Replacing only the random projection with per-state PCA45 improves—but does not pass—the transfer
test. PCA is fitted online to the current state's 64 unlabeled probe trajectories. It uses no
physical decisions, outcomes, reference population, or alarm threshold, and captures a median
93.4% of the pooled DINO variance.

| metric | random projection | PCA45 |
|---|---:|---:|
| final physical/visual agreement | 33/56 (58.9%) | **37/56 (66.1%)** |
| recovered physical alarms | 8/25 (32.0%) | **11/25 (44.0%)** |
| recovered mixed-fork alarms | 6/19 (31.6%) | **10/19 (52.6%)** |
| visual-only alarms | 6 | **5** |
| mixed-fork commitment majorities | 8/20 | **16/20** |

This confirms that the fixed random projection discarded useful consequence structure. It does not
resolve the larger visual problem: all 56 states still become initial candidates, including every
stable-centered state, and PCA recovers 0/5 physical contact-loss alarms. Full-frame DINO remains
dominated by nuisance robot/scene changes and lacks a clean object-centric consequence geometry.
Proceed to fixed background suppression and then the segmented block-region upper bound. PCA result:
`results/panda_block_push/visual_pca45/visual_pca45_diagnostic.json`, SHA-256
`f4c6905e02c42d9f6096fd15fff9724d81d5e62b400e78eb639c743f476ee464`.

### Background suppression and block-region upper bound

Two further PCA45 arms replayed and refined their own boundaries under the same frozen monitor:

| representation | agreement | recovered physical alarms | mixed-fork recall | visual-only alarms | total visual alarms |
|---|---:|---:|---:|---:|---:|
| full-frame PCA45 | **37/56 (66.1%)** | 11/25 | 10/19 | **5** | 16 |
| motion-weighted PCA45 | 30/56 (53.6%) | 12/25 | 9/19 | 13 | 25 |
| segmented-block PCA45 | 31/56 (55.4%) | **21/25** | **16/19** | 21 | 42 |

Motion weighting multiplies each DINO patch by its RGB change from the common pre-action frame. It
does not help: moving robot/contact silhouettes remain dominant and false alarms increase. The
segmented-block arm is an oracle representation upper bound using MuJoCo object masks. It recovers
84% of physical alarms, including 16/19 mixed forks and 4/5 contact-loss alarms, showing that the
relevant information is visible when the representation isolates the object. But it alarms on
10/15 stable-centered pushes and 5/6 overshoots, so it is not specific enough.

The conclusion is narrower than “DINO cannot see instability”: object-centric DINO contains much
of the signal, but the current adapter and physical-state-shaped consequence test do not distinguish
smooth safe object motion from true physical branching. Every visual arm still marks all 56 states
as initial candidates. The next formulation should be visual-native rather than forcing PCA
coordinates into v0's semantic position/rotation/velocity channel slices. First test tracked
object geometry or patch correspondence with a single representation-distance consequence curve;
keep simulator masks diagnostic-only and do not train the world model yet.

Results:

- motion arm SHA-256: `2328bb220d851637c1b0ac74bd5e4832c40008420904584310da06efe80d2278`;
- block oracle SHA-256: `a09f7c092e2d8bd6a0971a020286fce45668dda91edb57c73afb0f01f9769d4a`.

### Proper visual-native consequence implementation

The compatibility caveat above is now removed. `src/visual_consequence_monitor.py` treats the
complete visual latent as one metric space and constructs one internally RMS-normalized Euclidean
pair-separation curve. It then applies the same v0 boundary evidence, action-exposure model,
commitment BIC, persistence BIC, zero decision boundary and pair majority. It does not assign PCA
axes fake position/rotation/velocity meanings and uses no calibration population or labels.

| representation | agreement | recovered physical alarms | mixed-fork recall | visual-only alarms |
|---|---:|---:|---:|---:|
| full-frame PCA45 | 36/56 | 10/25 | 9/19 | **5** |
| motion-weighted PCA45 | 33/56 | 11/25 | 9/19 | 9 |
| segmented-block PCA45 | **37/56** | **24/25** | **18/19** | 18 |

The object-centric upper bound now has near-complete physical-regime recall: 18/19 mixed forks and
5/5 contact-loss regimes. This validates the visual-native consequence port and shows that rendered
images contain the required signal. Its remaining failure is specificity: it also alarms on 8/15
stable pushes, five physically quiet unanimous-topple states and 5/6 overshoots. Every visual arm
still marks all 56 states as an initial smooth-vs-branch candidate.

Therefore the next bottleneck is upstream visual response geometry. DINO patch trajectories make
ordinary object translation/appearance changes look piecewise-branched even when physical v0 models
them smoothly. Next test tracked block geometry or patch correspondence in coordinates where smooth
image motion remains smooth. Do not alter the now-correct visual consequence logic or move to
world-model predictions until specificity improves. Result SHA-256:
`b5afd81f0b22e056897fd52a3a1e690e844d17c268147c315b76ee2b960c4c5a`.

### DINO patch-correspondence geometry

The first universal tracking attempt uses DINO only for identity matching. All 256 start-frame
patches are matched directly into each later frame by fixed cosine top-4 soft correspondence;
normalized image position and finite-difference velocity are passed through unlabeled PCA45 and the
proper visual-native monitor. It uses no object mask, identity, confidence threshold or labels.

It does not improve the result: agreement is 33/56, physical-alarm recall is 10/25, mixed-fork
recall is 8/19, and eight visual-only alarms remain. All 56 states are still initial candidates.
The 16x16 DINO grid is too coarse and appearance-ambiguous for stable dynamical tracking: similar
patches can exchange identity across the robot, block and background, while sub-patch motion is
poorly resolved. Do not tune top-k or matching softness on this DEV result. The next generic
geometric baseline should use higher-resolution temporally constrained point tracking or optical
flow, then be frozen before comparison. Result SHA-256:
`7c0a15cf50607dab3a18b6bbf0bdc8c44f825fd3c39b417efe205d408a15c2e0`.

### Generic optical-flow geometry

The predeclared high-resolution follow-up detects up to 128 generic Shi--Tomasi corners in each
common start image and tracks them through every control frame with pyramidal Lucas--Kanade flow.
A fixed one-pixel forward/backward check marks losses; lost tracks retain their last position and
visibility becomes zero. Only normalized image position, one-step velocity and visibility enter
per-state unlabeled PCA45 and the unchanged visual-native monitor. There is no object mask,
identity, simulator state, task label or outcome calibration.

This arm recovers **24/25 physical alarms**, including **18/19 mixed forks** and all five
contact-loss regimes, but alarms on **54/56 states**. Its 30 visual-only alarms comprise all 15
safe-centered states, nine physically quiet unanimous-topple states, and all six overshoots. All
56 states remain initial candidates. Tracking itself is not simply collapsing: states retain a
median 46--51 points by cohort and 85--94% mean final visibility. Instead, full-image 2-D motion
turns ordinary robot, contact and perspective changes into committed branches under the current
response test. High-resolution flow repairs recall but catastrophically fails specificity.

Do not tune corner or flow constants on this DEV panel. This closes the hand-designed full-frame
tracking ladder. The next representation should be trained without failure labels to preserve
generic object-centric dynamics: temporal correspondence, 3-D/SE(3)-like geometry, velocity and
local action smoothness. Freeze that representation and evaluate it on a newly frozen Panda TEST
before any world-model prediction experiment. Result SHA-256:
`6742d28cfd937d804b10d21e5516a5545bbb760ed5c0934cd1012c869925a4c0`.

### Label-free slot-dynamics representation gate

The next representation track is implemented without opening monitor DEV or TEST. A separate
unlabeled dataset contains 80 paired Panda interactions (64 TRAIN, 16 validation), 2,720 RGB
frames, three cuboid geometries, randomized block pose, push direction/height, camera, light and
colour, and symmetric neighboring actions capped at 3 mm. Stored fields are only RGB, actions,
split and pair membership. Physical state, contact, success, topple, object identity and monitor
decisions are excluded. Frozen DINO patch descriptors are mapped to 64 fixed coordinates; a small
CPU slot model is trained with reconstruction, action-conditioned one-step prediction and
neighboring-action separation losses. Data SHA files and all training manifests are in
`results/panda_block_push/slot_dynamics/`.

Three prospectively separated attempts were rejected before monitor scoring:

- **v0 appearance slots:** validation objective improves from 1.0122 to 0.7768, but mask entropy
  remains near `ln(8)`: all eight masks are effectively uniform.
- **v1 motion-conditioned slots:** reconstructing descriptor change makes masks sharp, but one slot
  monopolizes essentially every patch (`slot_balance = 0.10935`, the eight-slot monopoly value).
- **v2 balanced motion slots:** structural balancing gives 8.0 effective slots, maximum mass 0.125,
  and motion-map correlation 0.337. It still fails the frozen geometry gate: temporal slot-geometry
  RMS is 0.000240 versus the required 0.002. The masks are balanced but nearly static/uniform.

Therefore no slot checkpoint was run through the Panda monitor and no fresh TEST was opened. This
is the intended purpose of the pre-monitor gate: avoid using alarm results to repair a collapsed
representation. The next architecture must receive a generic correspondence signal stronger than
feature reconstruction—dense optical/scene flow or pretrained depth/3-D tracks on unlabeled data—
and must predict persistent entity trajectories, not infer object masks from reconstruction alone.
Only a model that passes effective-slot, non-monopoly, temporal-geometry and motion-correlation
gates may proceed to frozen true-future monitor evaluation.

### Fresh ground-truth pushing TEST

A new physical TEST was generated independently with continuous random specifications (seed 1640),
selected using only 64-probe physical coverage, and frozen before the unchanged monitor ran once.
The 56 states comprise 20 mixed topple forks, 15 safe-centered pushes, 10 unanimous topples, five
contact-loss regimes and six overshoots. Protocol SHA-256:
`bfc1490690f15a0e879c04684250808813700b8e666e6ee5affa7496257f2696`.

| cohort | alarms / states |
|---|---:|
| mixed upright/topple fork | **14/20 (70%)** |
| safe-centered | **0/15** |
| unanimous topple | 1/10 |
| contact-loss | **3/5** |
| overshoot | **0/6** |

This is weaker than Panda DEV's 19/20 fork recall but close to held-out Jenga TEST's 65.5%.
Candidate discovery finds 18/20 forks; all 18 retain refined boundaries, but only 14 pass
commitment majority. The principal generalization loss is therefore consequence/commitment, not
boundary localization. Result SHA-256:
`dd406d053e1789a01b1b1574a083276ea3da995d85c64fa8b85ea8d2fd47c238`.

Ten exact consequential alarm-pair videos for this TEST and ten for immutable Jenga TEST are under
`results/ground_truth_monitor_alarm_videos/`. Each shows the two refined nearby executions side by
side through H=8 plus hold30; manifests record measured toppling and tilt outcomes.
