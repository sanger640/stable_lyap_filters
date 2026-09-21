# PLAN — the monitor on Jenga, offline then live

> **Superseded.** The plan being executed is [`PLAN_NEXT.md`](PLAN_NEXT.md) (revised 2026-09-21: task-agnostic counterfactual safety monitor, Track A mechanism study vs Track B universal monitor, cross-task validation). This file is kept as a historical record; its phase numbers are still referenced by older code and NOTES.md entries.

**Mode:** phase by phase, same rule as the other plans. Do not start a phase until the previous
one's acceptance criteria are met and logged in `NOTES.md`. **If a phase fails twice, stop and
report.**

**Plain-language background:** [`TOY_EXPERIMENT.md`](TOY_EXPERIMENT.md). Method spec:
[`MONITOR.md`](MONITOR.md). Current state and commands: [`HANDOFF.md`](HANDOFF.md).

---

## What is already established

**On real Jenga frames, with the encoder alone — no world model:**

| question | answer |
|---|---|
| do toppled and intact scenes separate in DINOv2 space? | **yes** — 98-99% nearest centroid, separation 2.15 |
| are the basins real, or a continuum of partial topples? | **real** — peak tilt is bimodal, 72.2 deg gap, 0 episodes in 20-60 deg |
| can the count be discovered with no labels? | **yes** — HDBSCAN finds k=2 at 98.8% |
| does latent distance track HOW FAR the block tipped? | **yes** — r = 0.77 with `peak_tilt_deg` |
| what has to happen first? | **regress proprio out.** PC1 of the raw latents is the ARM. Without removal, k-means at k=2 sits at CHANCE (50%) |

**So the representation and the physics both support the method.** Everything below is about
whether the PREDICTOR does, and whether it survives a live loop.

## The one problem that needs solving before anything else

**Jenga demos contain no held poses.** Over 6771 steps of demo actions: median per-step EE
displacement 5.3 mm, only 0.6% of steps under 1 mm, and the longest run of consecutive
sub-millimetre steps is **ONE**. A zero-action settle tail is therefore pure extrapolation.

Dropping the tail entirely is already **ruled out** -- measured on the toy against the fully settled
outcome, tail=0 finds the wrong count (2 instead of 3) and scores 74.3% against a 70.7% baseline.

**Current gate (2026-09-14): STOP before J5.** The corrected chunk-local simulator experiment
selects a five-step held tail physically, but only 12% of real or predicted short-tail endpoints
fall inside the terminal-basin support. Details are in `HANDOFF.md` and
`results/jenga/short_held_tails.json`.

---

## J1 — Load and check fidelity

Load `data/jenga/world_model.pt` (stripped, 136 MB) via `src/models/dinowm.install_import_shim()`.
Roll it along real episodes from the LMDB and compare predicted latents to encoded truth.

**Acceptance**
- Rolls 20+ steps without diverging or producing NaNs.
- One-step prediction error is small relative to the between-basin distance measured in J3.
- **Report error GROWTH with horizon, not just the mean.** On the toy, one-step error was excellent
  (0.040 rad) while 40-step error was useless (0.577). The mean hides that.

## J2 — Decide the settle tail (THE design decision)

Three candidates, in order of cost:

1. **Nominal-continuation tail.** Append the policy's own remaining actions instead of freezing. The
   arm lifts and retracts after every grasp, so a neighbour that was going to topple does so while
   the robot moves away. In-distribution, free, no retraining. One shared tail across all probes, so
   probe-to-probe differences still come only from the chunk.
2. **Short held-pose tail** (~5-10 steps) and accept mild extrapolation. Cheap to test; check
   whether the latent does something sane or explodes.
3. **Fine-tune on held-pose data** generated in the MuJoCo sim. Most faithful, most work.

**Acceptance:** for whichever tail is chosen, the **basin ASSIGNMENT stops changing** as the tail
lengthens. Do NOT require `||z_t+1 - z_t|| -> 0`; the toy showed the latent hovers indefinitely
while the label is stable from settle step 40 (99.2% by step 25). Requiring motion to stop measures
a quantity the method never uses.

**Superseded run 1:** nominal continuation remained finite for all 100 trajectories (80-330 tail steps).
Late 90%-to-full assignment agreement was 93-96% only for several non-default configurations and
only on 55-72% joint known-basin coverage; the default configuration was 61.8%. Because basin
count and membership were not robust, no tail was accepted. This was later recognized as the wrong
operational test: later policy chunks confound the consequence of the current eight actions.

**Corrected chunk-local run:** replayed 2,049 non-overlapping chunks from 100 action trajectories in
MuJoCo, branching after exactly H=8 and repeating only the final pose/gripper command. Among 1,702
upright starts, tail 0 exposed 36 new topples, tail 5 exposed 54, and tail 10 exposed 57. **Choose
tail 5 provisionally:** it recovers 18 delayed topples over tail 0; another five steps adds only 3.
The remaining blocker is representation coverage, not physical tail length.

## J3 — Do PREDICTED endings cluster like REAL ones?

The headline result so far used real final frames. The monitor reads predictions. Run the identical
pipeline -- regress proprio out, PCA, HDBSCAN at min_cluster_size 5-10% of n -- on predicted
endings.

**Acceptance**
- HDBSCAN finds **k=2**, agreeing with the topple label at **>= 90%** (real frames: 98.8%).
- Separation ratio **>= 1.5** (real frames: 2.15).
- If it fails, J4 says why.

**Full-trajectory stress test — not the chunk-local verdict:** at the full nominal tail, 2-D/5% produced k=3, 67.7% agreement, 96% coverage,
and 1.02 separation. Raising the minimum cluster size to 10% produced k=2, but agreement was only
74.1%, coverage 85%, and separation remained 1.02. Across 2/4/8/16 PCs and 5/10%, no setting
reached 90% agreement or 1.5 separation. One-step RMSE was 0.685; its ratio to the discovered
between-basin RMS was 1.05-1.89, so J1's relative-fidelity criterion also fails.

**Chunk-local held-tail result:** in a label-free PCA16 basis, real/predicted supervised separation
at tails 0/5/10 is 2.36/2.25, 2.45/2.41, and 2.42/2.25; the predicted physical outcome axis keeps
72-81% magnitude with cosine 0.99. However, HDBSCAN finds no persistent basins when fit across all
mixed trajectory stages, and only 11.6-13.4% of endpoints land in the existing terminal basin
support (real endpoints: 12.5-12.8%). J3 therefore remains gated for a structural reason shared by
real and predicted data: intermediate chunk endpoints are not terminal attractors.

## J4 — The hedging check (one line, no retraining)

Compare **predicted basin COUNTS** against the true 75/25 split. MSE-trained models predict the
conditional mean, which near a bifurcation sits BETWEEN the basins, so predictions compress toward
"nothing happened". On the toy, dino_wm's shipped single-step recipe gave [11,43,6] against a true
[15,28,17] and only 75% basin agreement.

**The Jenga checkpoint uses that same `num_pred: 1` recipe, so expect this.**

**If counts are compressed:** rollout fine-tune. ~19 min on the toy; more here, with gradient
checkpointing. **Order matters** -- teacher-force first, THEN rollouts. Reversed, the model collapses
to predicting that nothing ever happens ([1,59,0] on the toy).

**Run 1 — general rollout failure, not count-only hedging:** real-frame separation is only 1.06 at
H=8 because the physical outcome has not emerged yet, then rises to 2.21 by 75% of the nominal
continuation and remains 2.12 at the end. Predicted separation stays at 0.96-1.04. At the final
endpoint the predicted outcome axis is 38.7% as large as the real one and has cosine 0.40 with it.
Nearest-real-centroid counts are 72/28, superficially close to 75/25, but only 67% of episodes are
correct. Thus counts alone would hide the failure: the model loses and rotates the outcome signal.
Next attempt is rollout fine-tuning, then rerun J1-J3. The short held-pose and held-pose-fine-tune
tail candidates have not been tested; they cannot fix geometry already lost under real actions.

## J5 — Offline monitor on replayed episodes

Now the monitor proper. Per chunk: 50 probes along `v = a - a[0]` (the action is an ABSOLUTE EE
position, so scaling it is origin-dependent and meaningless -- perturb the DISPLACEMENT), settle
tail from J2, HDBSCAN centroids from J3, `k >= 2` alarm. Probes landing in NOISE are excluded from
the basin vote and reported separately as reduced known-basin coverage.

**Run the FALSE-POSITIVE FLOOR first**: on chunks far from any boundary, `k` should be 0 on >= 95%.
If not, the alarm is measuring the model rather than the scene, and nothing downstream is
meaningful.

**Local delta-z attempt — FAIL at the false-positive floor:** 50 event-adjacent and 50 quiet
controls were each evaluated with 50 coherent probes at H=8 + held tail 5. Physical branching found
10 true boundary chunks. HDBSCAN on predicted local delta-z detected all 10 but alarmed on 84/90
non-boundaries. The encoded-real control was worse: it alarmed on all 100 chunks, including all 50
quiet controls. Mean known-cluster coverage was 68.5% predicted and 78.3% real, so this is not a
noise-vote artifact. Density clustering partitions the smooth nonlinear response curve and cannot
distinguish curvature from a bifurcation. Do not adopt this score or fix it by retraining the model.

**Ordered function-fitting attempt — large improvement, but the real-data gate still fails:**
`eval/jenga_ordered_change_points.py` compares a global cubic and a continuous piecewise cubic
against the same piecewise model with a step discontinuity. BIC penalizes parameters and the search
over split locations; an alarm additionally requires the observed adjacent latent increment to be
at least 3x the median increment. At the predeclared PCA4 setting, encoded-real results are 7 TP,
8 FP, 3 FN, 82 TN (precision 46.7%, recall 70%, F1 0.56). Quiet-control alarms fall from HDBSCAN's
50/50 to 5/50, but 10% still exceeds the <=5% gate. PCA2/8/16 quiet-control rates are 6/8/6%, so
none passes. Prediction scoring is deliberately withheld. Before changing thresholds, save
continuous peak tilt, block pose, and contact state per probe to determine whether the remaining
real-latent jumps are genuine sub-threshold physical transitions.

**Physical attribution completed:** only 1/5 quiet alarms coincides with a material neighboring-
block change (2.48 mm translation and 2.01 deg rotation); none changes contact category. Of the
other four, one moves only the manipulated middle block by 0.33 mm and three have no meaningful
block motion. Targeted paired renders show only tiny robot/raster differences. Thus the 45-degree
binary label was too coarse for one alarm, but does not explain the remaining four. The next clean
test is a frozen 4x adjacent-jump ratio on a new held-out quiet set, alongside a neighbor-block
spatial latent that suppresses robot and manipulated-block pixels. Do not claim the gate passed on
the present data: choosing 4x after inspecting these controls would be post-hoc.

**Frozen 4x episode-held-out validation — FAIL by one alarm:** 50 new nominally quiet chunks were
selected from the 25 episodes never used by the original local-probe set, with two temporal
interior points per episode (starts 18-146, median 58). All 50 remained physical non-boundaries.
The frozen detector produced 3/50 false alarms = 6%, just above the <=5% gate (exact 95% interval
1.25-16.55%). An initial implementation accidentally selected only starts 2 and 10 and obtained
2/50; that easy-prefix result is invalid and superseded. On the original mixed set, 4x gives
7 TP / 3 FP / 3 FN / 87 TN. Prediction scoring remains formally withheld. An exploratory score,
seen after the flawed preliminary pass but before the selector correction, found 0/10 predicted
boundary detections and 4 false alarms; treat this as a warning that DINO-WM smooths the signal,
not as a passed-gate model result. Next: neighbor-block spatial latents, then rerun the real gate.

**Score against `peak_tilt_deg`, not the binary outcome.** It is a continuous near-miss measure and
therefore much closer to the PROXIMITY quantity this method predicts. The binary flag is the wrong
target and the toy work repeatedly showed what that costs.

**All-three-block extension (steps 1-5, 2026-09-15):** failure now includes left/right tilt >=45
deg, middle tilt >=45 deg, or middle COM falling below the 0.425 m tabletop; controlled upright
middle translation/lifting is not failure. Re-labeling the original cache leaves the union at 10
boundary chunks and the frozen-4x score at 7 TP / 3 FP / 3 FN / 87 TN. It exposes two specifically
middle-topple boundaries, both missed; they contain only 5 and 2 safe probes and are outside the
fitter's minimum-eight-per-side supported range.

An all-100-episode physical scan across 2,049 chunks gives new all-block failures from safe starts
of 48/69/77 at tails 0/5/10. Tail 5 remains the main knee, but tail 10 adds eight failures, six of
them middle topples; use tail 10 for all-block safety if latency permits. No middle COM fell below
the table in this corpus. A balanced 80-scenario set (20 middle failure, 20 neighbor failure, 20
safe extraction, 20 quiet) was tested at tail 10. At operational eps=0.10 it yielded no supported
middle boundaries. A fixed eps=0.30 stress test yielded 18 physical mixed-outcome chunks, but only
seven had >=8 probes on each side: three middle and six neighbor boundaries with overlap.

On those seven supported eps=0.30 boundaries, full-frame DINO detects 6/7 with 2 false alarms
(precision .75, recall .857, F1 .80; quiet 0/20); a simulator-oracle all-three-block patch pool
detects 5/7 with 1 false alarm (precision .833, recall .714, F1 .769; quiet 0/20). Both detect only
1/3 supported middle boundaries, while full-frame detects all 6/6 supported neighbor boundaries.
Therefore spatial suppression does not repair middle sensitivity. The original fixed red ROI is
invalid—it misses the blocks at some stages—and must not be cited. The green segmentation-derived
mask is the valid upper-bound test. The next blocker is a middle-specific probe/response signal and
more naturally near-boundary middle data, not further crop tuning.

**Neighbor-only steps 1-3 completed — FAIL:** the frozen operational detector was evaluated on 250
new temporally spread quiet states and every supported eps=.10 neighbor boundary available after
excluding prior probe states. Quiet FPR is 16/250 = 6.4% (exact 95% CI 3.70-10.19%), still above
the <=5% gate. Exhaustively screening 1,302 eligible states produced only 11 boundaries with at
least eight probes per side, so the planned >=30 benchmark is infeasible in the present corpus.
Real encoded recall on those 11 is 5/11 = 45.5% (exact 95% CI 16.75-76.62%), below the >=80% gate.
Seven physical outcome curves have multiple safe/toppled switches, showing that a single split is
often the wrong structural model rather than merely a badly tuned threshold.

The matched DINO-WM audit detects 0/11 boundaries and 5/250 quiet controls. Predicted boundary jump
ratios have median 0.81 versus 6.64 in real encodings, only 14.8% of the paired real ratio, with
correlation -0.075. The checkpoint preserves average short-horizon geometry but suppresses the
local bifurcations needed by this detector. J6 is blocked: do not tune the present score or wire it
live. Next choose between targeted counterfactual/multimodal retraining with new near-boundary data
and an object-centric pose/physics or supervised risk model. Full metrics are in
`results/jenga/neighbor_validation.json`.

**Research-aligned next experiment 1-4 completed:** `src/local_expansion.py` replaces the single
global split with a sliding local continuity test. Every supported adjacent pair gets a local
continuous piecewise-linear fit and an otherwise identical stepped fit; BIC plus a search penalty
provides the fixed zero-evidence rule, and several separated positive regions may be reported. No
failure label or fitted percentile enters the score.

A broad-to-local stress-set constructor screened 300 untouched states at eps=.30 only to locate
action-space transitions, then recentered operational 50-probe eps=.10 families. It found 59
supported local neighbor boundaries; 30 episode-spread cases and 100 untouched quiet controls were
evaluated in all representations. After excluding three accidentally included middle-only contact
channels, physical neighbor pose/orientation/contact is the oracle: 26/30 detections, 0/100 control
alarms, recall .867 (95% CI .693-.962), F1 .929, AUC .868. It passes both physical point gates;
14/30 outcome curves remain multi-switch.

Real DINO detects 30/30 but alarms on all 100 controls (AUC .599). Current DINO-WM detects 12/30
and alarms on 49/100 (AUC .469). On this static-control benchmark alone, representation learning
looked like the next gate; expanded hard negatives below overturn that priority. Full endpoint
results: `results/jenga/multipeak_oracle.json`.

**Trajectory-level measurement completed — physical oracle PASS:** recorded the strictly neighbor-
only physical state at all 13 action times for the same 100 controls and 30 boundaries. Per-time
local BIC evidence is combined with a temporal-search-corrected Bayes-factor mean. The result is
30/30 boundary detections and 0/100 control alarms: precision, recall, F1, and AUC all 1.0. The
same cache's endpoint score is 26/30, so all four endpoint misses exhibit earlier transient
expansion. Median temporal support is 6.5/13 times on detected boundaries. This passes the
constructed boundary/static-control benchmark only. Expanded hard negatives below overturn a
general physical gate-pass claim. Results: `results/jenga/trajectory_oracle.json`.

**Visual trajectory test and expanded negatives — FAIL (2026-09-16):** the same 130-state benchmark
was rendered and DINO-encoded at all 13 times. Full-frame PCA4 detects 30/30 boundaries but alarms
on 100/100 quiet controls (AUC .489). A simulator-oracle neighbor-patch pool also gives 30/30 and
100/100 (AUC .331). On quiet controls the visual evidence begins by time 2 in 97/100 cases and is
positive at a median 12/13 times, despite zero paired physical alarms. Simple cropping does not
remove visual nuisance.

More importantly, the physical oracle was evaluated on all 38 chunks from the only two strictly
silent episodes and 100 nominal moving non-topples. It falsely alarms on 5/38 safe silent chunks.
Of the moving set, 88 stay safe under all 50 probes and **87/88** alarm; six are supported
boundaries, five are low-support mixed outcomes, and one is certain failure. Pose-only and position-
only ablations also alarm on all 88 moving-safe states. This defeats a gate defined purely by
toppling inside the operational probe interval, but does not prove the states are *far* from a
topple boundary. The next test is to estimate physical failure margin by widening and refining
the probe range without using those labels to fit the alarm. Do not retrain a visual or predictive
model against the current score until proximity specificity is known. Results:
`results/jenga/visual_trajectories.json` and `results/jenga/expanded_controls.json`.

**No-alarm pick check:** among 67 nominal chunks where the middle/red block rises >=2.5 cm and
travels sideways >=2 cm with no nominal neighbor topple, 66 remain safe under all 50 probes;
55 have no physical alarm. Three videos from separate episodes verify ~8 cm lift, 3-4 cm sideways
motion, zero neighbor tilt, and no alarm on displayed probes. This is evidence of selectivity
against *red-block motion alone*. It does not establish whether moving-neighbor alarms are
premature. Crucially, those first clips show carrying after the grasp. Two zoomed clips at table
height show safe no-alarm lifts close to the neighbors but **without** robot-neighbor contact.
All 11 safe/alarm cases in the 67-pick screen had such contact; none of its 55 safe/no-alarm cases
did. Among a wider 20 nominal contact-and-lift chunks, 19 stay safe on all probes and all 19
alarm. So the requested no-alarm *contacting-neighbor* pick was not found. See
`results/jenga/pick_noalarm_search.json`, `results/jenga/contact_pick_check.json`, and
`results/jenga/near_grasp_noalarm_videos/`.

**Full episodes resolve the ambiguity:** of 100 nominal replays, 50 meet the post-settle red-pick
and neighbor-safety criterion; all 50 have at least one verified alarm decision. No entirely
alarm-free successful episode was found. Four episodes have been exhaustively scored at every
non-overlapping H=8 decision and rendered from start to finish. Ep24 alarms at pick chunk 106
(all 50 probes safe at that chunk); ep74/ep91 are quiet at the lift chunks but alarmed earlier.
The earlier no-alarm pick clips were therefore local snapshots, not full-episode successes without
warnings. See `results/jenga/nominal_success_scan.json`,
`results/jenga/full_episode_alarm_audit.json`, and `results/jenga/full_episode_alarm_videos/`.

**Sizing:** `eps` follows from the margin to be detected -- `m* = 1.56*eps` at n=50. Pick the margin
first. Note the current Jenga deviator uses per-timestep noise at eps=0.005, which is the WRONG
probe family: independent per-step nudges cancel, costing a factor of ~sqrt(T).

**2026-09-16 physical-basin revisit:** using a reference atlas that excludes test episodes 24/74,
held tails 5/10/20 yield 10/30, 20/30, 19/30 detections of recentered boundary families, but
zero dissent alarms across all 37 full-episode decisions. The 30 boundary families contribute to
the atlas, so those detection rates are in-sample/optimistic. Coverage is imperfect and the ep24
pick decision is unassigned at all tails. Tail 10 is preferable among those tested, but global
basin dissent is not validated. See `results/jenga/physical_basin_tails.json` and full videos in
`results/jenga/physical_basin_episode_videos/`.

**Strength is not the sole fix:** eps=.10 reaches 23.3% of nominal chunk span, which can mean
many millimeters of command change. A matched 50-probe, 20-state physical-BIC pilot at eps
.10/.035/.0165 kept all 10 moving-safe alarms even at the narrowest range, while two boundary
families lost their within-range topple branch. Do not choose eps by fitting these Jenga labels;
derive it from plausible action uncertainty and report sensitivity. See
`results/jenga/action_strength_resim.json`.
The same matched pilot scored by frozen physical-basin dissent gets 2/10, 1/10, 0/10 boundary
alarms as eps narrows, versus 0/10 moving-safe alarms at each strength. Its moving-safe coverage
is only 29%/21%/20%; quiet is often abstention, and narrower probes do not repair this atlas.

**Nearest persistent branch margin implemented:** `eval/jenga_persistent_branch_margin.py`
probes from the original nominal action on a fixed signed coefficient grid, not from recentered
benchmark centers. A branch must survive both held tails 5/10 and two neighboring strengths.
The physical endpoint-topple oracle brackets 13/15 held-out boundary families (median upper
margin 3.01 mm), 1/20 moving-safe states, and 0/10 each silent/contacting-pick controls; two
boundary and three moving cases only have edge-of-grid candidates. However, the global physical
basin estimator cannot supply a margin on any boundary case: all 15 nominal endpoints are
unassigned. These are single-direction, selected-cohort diagnostics, not a validated unlabeled
monitor. See `results/jenga/persistent_branch_margin.json`.

**Next gate:** develop a local branch test without global basin assignment on the cached physical response curves;
compare its nominal-centered margin brackets with the held-out physical oracle and abstention
rate. Keep the 5/10 tails and fixed grid, avoid tuning on topple labels, and only proceed to a
visual representation after physical-state specificity and coverage pass. Separately estimate
the controller's actual action-error envelope; the current ±.6 grid is a diagnostic search range.

**First local test completed; gate failed (2026-09-16):** local continuous-vs-step BIC on unlabeled
PCA6 physical responses, requiring the *same gap* at held tails 5 and 10, produces 15/15 boundary
alarms but also 20/20 moving-safe and 10/10 safe contacting-pick alarms (1/10 silent). It finds
apparent margins around 0.3-0.5 mm that do not match the ~3 mm physical boundary margins.
Temporal persistence and local BIC are insufficient. Keep `results/jenga/local_persistent_branch.json`
as a negative baseline. The next version should add *action-resolution persistence*: bisect a
candidate gap and check whether the physical response separation remains finite as the action
gap halves, rather than shrinking as a smooth response would. Evaluate it on the same frozen
panel, then a new episode holdout; no visual training until that physical gate passes.

**Action-resolution persistence tested; gate still fails (2026-09-16):** two adaptive bisections
of each frozen local-BIC gap retain separation on 11/15 boundary states, but also 14/20 moving-safe
states and 9/10 safe contacting picks. These counts use freshly replayed endpoints at both original
gap edges, avoiding a cache/replay mismatch; pose-only ablation still retains 15/20 moving and
9/10 contacting-pick cases. The one silent candidate is rejected. For contacting
picks, physical endpoints at tails 5 and 10 are practically identical, so the temporal condition
adds little independent evidence. See `results/jenga/branch_scale_refine.json` and
`results/jenga/branch_scale_ablation.json`. No fresh holdout
was run because the development-panel specificity gate failed. Next investigate the *magnitude
and persistence of downstream consequence*, not merely the mathematical retention ratio of a
small contact/pose fluctuation. Define a task-agnostic uncertainty/noise envelope from the
controller and observation process; test multiple action directions and a real short-horizon
behavioral difference before any visual or live deployment claim.

**Controller-scaled multi-direction experiment completed (2026-09-17):** The next proposal was
implemented in `eval/jenga_tracking_uncertainty.py`, `eval/jenga_predictive_regime_probe.py`,
and `eval/jenga_predictive_variation.py`. Held-out tracking residual median/p90 are
1.465/3.56 mm; training radial q90 gives 92.3% held-out coverage. Three residual-covariance
directions, 17 strengths each, perturb H=8 and share one nominal continuation; pose is read at
H, H+5, H+10. Persistent unlabeled split alarms on 9/20 moving non-topple, 3/10 contacting
picks, 9/15 selected boundaries and 1/10 silent states. Only 8/15 boundaries physically mix
within the new envelope, and 5 of these alarm. Eight residual-snippet replay variations show
tail-10 median candidate-gap/jitter ratios of 0.81 moving, 0.08 contacting and 0.75 boundary.
This is a *negative development-panel result*, not a validated filter; the snippet injection
is only a proxy for execution noise. No new holdout was run. Results are in
`results/jenga/tracking_uncertainty.json`, `predictive_regime_probe.json`, and
`predictive_variation.json`. The next gate is a task-general, independently measurable
consequence criterion with actual execution/observation variability; freeze it, then test on
new episodes before moving to pixels. Keep Jenga topple labels strictly evaluation-only.

**Score densely.** `k` spikes and decays; on the toy, 3 scoring times per episode understated recall
at 0.611 where 8 gave 0.820.

## J6 — Live during policy rollout

Wire it into the loop: `panda_express/sim.py` -> `diff_server_dual.py` (policy, ZMQ 5555) ->
monitor (ZMQ 5556). The policy checkpoint is
`diffusion_policy/.../21.59.04_train_franka_dual_jenga_jenga_image_dual/latest.ckpt` (815 MB,
`n_action_steps: 8`).

**Note the asymmetry:** the POLICY is dual-camera, the WORLD MODEL is single-camera (196 patches,
`model_latest_single.pth`). That is fine and intentional -- they are separate consumers. Do not
"fix" it by making the monitor dual-view: the wrist camera moves WITH the action, so probe
perturbations would change the wrist view for reasons unrelated to the scene outcome, injecting
action magnitude straight into `k`.

**Timing budget:** a chunk is 8 steps at ~10 Hz = 0.8 s. One score is 50 probes x (8 + tail) steps,
batched. At 588 tokens per frame this is smaller per sample than the toy's 768, so real-time is
plausible. If it is tight, fewer probes or a shorter tail buys headroom -- at the cost of detectable
margin, via the sizing rule.

**Acceptance:** the monitor produces a score per chunk within the control period, and alarms
correlate with `peak_tilt_deg` on the episodes where a neighbour is disturbed.

---

## Transfer

Code, including the vendored dino_wm models, is all in git. Everything else:

```bash
./scripts/bundle_jenga.sh          # ~1.2 GB: world model + LMDB + labels + sim
./scripts/bundle_jenga.sh --live   # ~2.1 GB: adds the diffusion policy
```

| asset | size | needed for |
|---|---|---|
| `data/jenga/world_model.pt` | 136 MB | everything (stripped from 378 MB; optimizers dropped) |
| `jenga_single_100.lmdb` | 1.1 GB | J1-J5 |
| `labels_noise100.json` | tiny | scoring (tracked in git) |
| `panda_express/sim.py` + jenga assets | small | J6 |
| diffusion policy `latest.ckpt` | 815 MB | J6 only |

**Not needed:** the `dino_wm` repo (its model code is vendored at `src/models/dinowm/`), the 102 raw
episodes (2.4 GB, only for rebuilding an LMDB), the 13 GB of toy caches.

---

## Risk register

| risk | caught by | cost if missed |
|---|---|---|
| no in-distribution settle tail | **J2** | the whole basin construction on Jenga |
| predictor hedges toward "nothing toppled" | **J4** | a blind monitor -- high precision, no recall |
| predicted endings do not cluster like real ones | **J3** | J5-J6 measure noise |
| model error manufactures dissent | **J5 false-positive floor** | `k` measures the model, not the scene |
| wrong probe family (per-timestep noise) | J5 sizing check | ~sqrt(T) of the probe reach thrown away |
| scoring too sparsely | J5 | recall understated, as on the toy |
