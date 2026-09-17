# HANDOFF — stability-based safety monitor

> **Current handoff (2026-09-17, end of day).** The live line of work is the Stage 0-2b
> sequence at the END of this document, not the basin/dissent method described below. In one
> line: wobble the chunk 64 times with measured execution error, let the scene settle, and
> alarm if the endings split into separated groups that persist over time.
>
> Where it stands on a FRESH 140-state holdout, from image latents of rendered endings:
> with the arm converging to a shared hold pose, the two-group rule gets 96% recall on
> mixed-outcome states at 1x with 3% false alarms on unanimous-safe states; with the arm ending
> in each probe's own pose it falls to 78%, where the coarse-persistence multi-group rule gets
> 93% with 5%. Neither rule wins both regimes, and no rule may be chosen per case.
> **Nothing here is validated as a safety filter.** Stage 3 ran the same rules on world-model
> PREDICTED endings and FAILED: alarm rates on forks and on quiet states are equal within noise
> (0-12% either way). The cause is the predictor -- the separation between toppling and stable
> probes is d' 0.60 predicted versus 4.85 real -- so the deployable path is blocked on rollout
> fine-tuning the Jenga world model, not on the detector.
> Read section 6's Stage 0-3 entries, then NOTES.md from the bottom up.

Written 2026-09-12. Self-contained brief for picking this up cold, human or agent.
Read this, then `MONITOR.md` (the method), then `NOTES.md` (the append-only log, newest at the
bottom). `PLAN.md` and `PLAN_DINOWM.md` are the phase plans with acceptance criteria.
`TOY_EXPERIMENT.md` is a plain-language writeup of the toy result with figures -- start there if
you want the story before the detail.

---

## 1. What this is

A **runtime safety monitor** for robot manipulation policies. The target task is a Franka Panda
picking a block from a cluttered tabletop without toppling its neighbours (`~/wksp/dino_wm`,
`~/wksp/panda_express`, `~/wksp/diffusion_policy`). This repo is the **toy-system testbed** where
the method is developed and falsified cheaply.

### The claim — read this before anything else

**It detects that the policy is operating where small errors change the outcome. NOT that a failure
is coming.** The signal is that a *nearby alternative action reaches a different terminal state*.

Three consequences, all non-negotiable once the claim is fixed:

1. **Evaluate on PROXIMITY (margin < m), never on outcome.** On outcome the method scores AUC
   0.29–0.56 and loses to summing the forces (0.925). On proximity it wins. When a result looks
   bad, check first whether it is an outcome-task number being read as a proximity claim.
2. **The blind spot is declared scope.** Dissent counting cannot see an action that fails with
   CERTAINTY — if every probe agrees, k=0. Report it; do not patch it.
3. **Any alarm clause needing a basin named "failure" is supervision and is struck.** The monitor
   must drop onto a new task with no labels.

### The method in one block

```
ONLINE, every chunk:
  1. truncate the action chunk to a FIXED lookahead H   (never receding)
  2. 32 probes: a_j = a + eps * z_j * v,  ONE SCALAR z_j ~ N(0,1) per probe
  3. append a SETTLE TAIL of zero/hold action
  4. roll each probe through the world model, take the ENDING latent
  5. PCA -> assign to nearest supported attractor centroid, or NOISE
  6. k = number of KNOWN-BASIN probes NOT in the plurality
     coverage = fraction of probes assigned to a known basin (reported separately)
  7. ALARM if k >= 2.   No threshold. No labels. No calibration.

OFFLINE, once: discover the attractors with HDBSCAN on PREDICTED endings, PCA'd first
               (min_cluster_size = 5-10% of n). The count is never supplied. A probe
               landing in NOISE is excluded from the basin vote and reported as reduced coverage.
```

**Sizing rule — this replaces threshold tuning.** `E[k] = n * Phi(-m/eps)`, so the detectable
margin follows from the probe budget: **m\* = 1.33 eps at n=32**, 1.56 at n=50. Reach grows only
logarithmically in n, so raising eps is far cheaper than adding probes. Pick the margin you must
detect; solve for eps.

---

## 2. Where it stands

| track | status |
|---|---|
| **shPLRNN on the exact state** (toy) | done — AUC 0.808, prec 0.633, rec 0.905 |
| **DINO-WM on images** (toy) | known-only dissent rerun — chunk-label **AUC 0.869**, prec 0.889, rec 0.800, F1 0.842 |
| **Jenga** (the real task) | tail 5 selected; ordered delta-z cuts false alarms sharply but misses the <=5% real-control gate; stop before J5 |

**The current result:** HDBSCAN removes the per-task clustering choice and discovers the correct
three toy basins (93.7% agreement, 100% coverage). Excluding noise from the dissent vote restores
the runtime false-positive floor: on 100 episodes it scores AUC 0.869 on the preferred per-chunk
label, with precision 0.889, recall 0.800, F1 0.842, and 96-97% of large-margin chunks below alarm.
Against the old full-action label, its like-for-like AUC is 0.872 versus 0.882 for nearest-centroid
and 0.808 for the exact-state monitor. Mean known-basin coverage is 92.8% (median 100%); 29 of
800 chunks have no known-basin probes and therefore produce k=0 by definition, not evidence of
safety.

**The caveat that must travel with it:** this used a ROLLOUT FINE-TUNED predictor. dino_wm's
shipped `num_pred: 1` recipe gives 75% basin agreement and was NOT used. So this validates the
architecture, not the existing Jenga checkpoint.

---

## 3. Environment and how to run things

**Everything runs in the `dino_wm` conda env** (named after the sibling repo, but the toy pipeline
no longer depends on it -- see below). Versions are pinned in `requirements.txt`.

```bash
PY=/home/sanger/miniforge3/envs/dino_wm/bin/python
cd ~/wksp/stable_lyap_filters
$PY -m pytest tests/ -q            # 61 tests, ~12 s, all should pass
```

The current machine has an RTX 3090 with **24 GiB**. The monitor still chunks to <=128 to preserve
the validated execution path and remain portable to the original RTX 5060 Ti (7.5 GiB):
* batch 256 x 768 tokens needs ~9.6 GB for attention alone on the original machine.
* backprop through an unrolled 8-step ViT OOMs. Use gradient checkpointing.

### Pipeline, in order (DINO-WM track)

```bash
$PY eval/phase_c_data.py            # render + DINOv2-encode 600 eps  (~5 min, writes 5.3 GB)
$PY eval/phase_c_train.py           # teacher-forced predictor        (~8 min)
$PY eval/phase_c_train_gtf.py --tag gtf_warm   # rollout fine-tune    (~19 min)  <- REQUIRED
$PY eval/phase_d_settle.py          # settle + contraction tests      (~2 min)
$PY eval/phase_e_attractors.py      # attractor discovery             (~4 min)
$PY eval/phase_f_monitor.py         # the monitor, 100 eps x 8 times  (~34 min on RTX 3090)
$PY eval/phase_f_videos2.py --rescore  # select/rescore/render 10 demos (~14 min on RTX 3090)
```

### Jenga scripts

```bash
# --- current line of work (Stage 0-2b, 2026-09-17); see the banner at the top ---
$PY eval/jenga_stage0_noise_oracle.py    # 64 execution-noise runs/state -> answer key (~2.5 min)
$PY eval/jenga_stage1_outcome_modes.py   # physics fork test on those endings (CPU only)
$PY eval/jenga_stage1_split_images.py    # render the non-topple splits it flags
$PY eval/jenga_stage2_visual_forks.py    # same test on full-frame DINO latents (~9 min/55 states)
$PY eval/jenga_stage2_visual_forks.py --own-hold  # arm-movement check (own hold target)
$PY eval/jenga_stage2b_multimode.py      # multi-group + revised persistence on cached latents
$PY eval/jenga_holdout_select.py         # pick holdout states from a screen (rule in docstring)
# the holdout end to end: screen 1,168 chunks (~37 min, 30 workers), then select, render, score
$PY eval/jenga_stage0_noise_oracle.py --panel results/jenga/holdout_screen_panel.json \
    --cache results/jenga/holdout_stage0_cache.npz --output results/jenga/holdout_stage0.json \
    --workers 30   # --snippet-panel keeps the SAME 64 snippets; do not change it

$PY eval/jenga_j1_fidelity.py       # bundled model + LMDB, error vs horizon (J1)
$PY eval/jenga_j2_j3_predicted_basins.py  # nominal tail + predicted basins (J2/J3)
$PY eval/jenga_j2_j3_predicted_basins.py --reuse-cache  # reanalyse without GPU rollout
$PY eval/jenga_j4_hedging.py        # paired real/predicted horizon diagnostic (J4)
$PY eval/jenga_j4_hedging.py --reuse-cache  # reanalyse without encoding/rollout
$PY eval/jenga_short_held_tails.py  # 100-episode simulator branches: H=8 + tail 0/5/10
$PY eval/jenga_short_held_tails.py --reuse-cache  # reanalyse cached 2,049 chunks
$PY eval/jenga_local_delta_probes.py  # 100-chunk paired local delta-z diagnostic
$PY eval/jenga_local_delta_probes.py --reuse-cache  # reanalyse from compact raw cache
$PY eval/jenga_ordered_change_points.py  # ordered continuity test; real gate before predictions
$PY eval/jenga_physical_jump_diagnostics.py  # attribute real-latent alarms to simulator state
$PY eval/jenga_quiet_alarm_inspection.py  # paired renders/robot pose for five quiet alarms
$PY eval/jenga_ordered_holdout.py  # frozen 4x on temporally spread unseen episodes
$PY eval/jenga_ordered_holdout.py --reuse-cache  # re-score held-out real latents
$PY eval/jenga_all_block_relabel.py  # re-label cached probes for middle + neighbors
$PY eval/jenga_all_block_tails.py  # 2,049 chunks, all blocks, tail 0/5/10
$PY eval/jenga_all_block_tails.py --reuse-cache
$PY eval/jenga_spatial_latents.py --reuse-cache  # balanced eps=.10 representations
$PY eval/jenga_spatial_latents.py --eps .30 --reuse-cache --cache results/jenga/spatial_latents_eps03_masked_cache.npz --output results/jenga/spatial_latents_eps03.json
$PY eval/jenga_neighbor_validation.py --reuse-cache  # frozen neighbor-only validation/audit
$PY eval/jenga_near_boundary_discovery.py  # broad locator -> centered operational families
$PY eval/jenga_multipeak_oracle.py --reuse-cache  # physical/real/predicted multi-peak gate
$PY eval/jenga_trajectory_oracle.py --reuse-cache  # 13-time physical expansion/persistence gate
$PY eval/jenga_visual_trajectories.py --reuse-cache  # 13-time real DINO and oracle-mask audit
$PY eval/jenga_expand_controls.py --reuse-cache  # complete silent episodes and moving-safe negatives
$PY eval/jenga_geometry.py          # do toppled/intact separate? raw vs PCA   (~2 min)
$PY eval/jenga_basins.py            # bimodality + arm removal, the key result (~2 min)
$PY eval/jenga_linkage.py           # linkage comparison + contact sheets      (~2 min)
$PY eval/cluster_shootout.py        # HDBSCAN vs the rest, BOTH systems        (~5 min)
```

Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for anything that trains or rolls out.

### The Jenga bundle (built and verified 2026-09-14)

`jenga_bundle_live.tar`, **2.0 GB**, holds everything the Jenga work needs that git cannot carry.
Rebuild with `./scripts/bundle_jenga.sh --live`.

```bash
git clone git@github.com:sanger640/stable_lyap_filters.git && cd stable_lyap_filters
sha256sum -c data/jenga/BUNDLE.sha256      # verify the transfer
tar -Sxf /path/to/jenga_bundle_live.tar    # -S IS REQUIRED, see below
./scripts/setup_env.sh
source .venv/bin/activate
python -m pytest tests/ -q                 # 61 tests
python eval/jenga_j1_fidelity.py --episodes 10
python eval/jenga_j2_j3_predicted_basins.py
```

**Extract with `-S`.** The LMDB's `data.mdb` is **20 GB apparent against 1.1 GB of real blocks** --
LMDB preallocates its map file. Without `-S` you write 20 GB to disk for 1.1 GB of data. (The same
trap made the first bundle 21 GB until `tar -S` was added on the build side.)

Verified by extracting to a clean directory: LMDB opens with 100 episodes, world model loads at
epoch 88, label keys match the LMDB episodes exactly, the sim tar contains
`panda_jenga_setup.xml`, policy checkpoint intact.

### Moving to another machine

Everything needed for the TOY work is in this repo. Verified-working versions are pinned in
`requirements.txt` (python 3.11.15, torch 2.11.0+cu128, numpy 2.4.6). **torch must be a cu128
build** -- the GPUs used here are Blackwell (sm_120) and cu121 wheels will not run.

```bash
git clone git@github.com:sanger640/stable_lyap_filters.git && cd stable_lyap_filters
python -m pytest tests/ -q                       # 61 tests, ~12 s -- verifies the port
```

**The repo is self-contained for the toy work.** `ViTPredictor` -- the only thing ever imported
from `dino_wm` -- is vendored at `src/models/dinowm_vit.py`. You do NOT need the dino_wm repo to
run phases A-F. (It was previously behind a hard-coded `sys.path` entry pointing at
`/home/sanger/wksp/dino_wm`, which made a fresh clone fail immediately.)

**Training from scratch is the expected path** and takes ~32 min end to end:

```bash
python eval/phase_c_data.py                      # render + encode 600 eps   ~5 min
python eval/phase_c_train.py                     # teacher-forced            ~8 min
python eval/phase_c_train_gtf.py --tag gtf_warm  # rollout fine-tune         ~19 min  <- REQUIRED
```

The trained predictors are also recoverable from git history if you ever want to reproduce exact
numbers rather than retrain -- they were tracked until `b48efbc`, so a full clone carries the
blobs: `git show b48efbc^:results/phase_c/predictor.pt > results/phase_c/predictor.pt`.

**Regenerate, do not copy**, the large caches -- `results/phase_c/latents.npy` (5.3 GB, ~5 min via
`phase_c_data.py`), `results/phase_a/feat_*.npz` (13 GB), `phase_e_endings_*.npz`.

**Reproduce one known result to confirm the port**: `eval/phase_e_attractors.py` should find a
k=3 on the toy at 93.7%, ~4 min.

**For the JENGA work only** -- not needed for the toy -- you additionally need assets from outside
this repo. **The inventory below was verified on 2026-09-14 and corrects RESUME.md, which claims
the checkpoint, the LMDB and the labels are all missing. None of them are.**

| what | where | size | state |
|---|---|---|---|
| world model checkpoint | `dino_wm/outputs/model_latest_single.pth` | 360 MB | **exists** (epoch 88, single-view, 196 patches) |
| eval LMDB (smallest usable) | `panda_express/tasks/jenga_noise_50/jenga_single.lmdb` | 534 MB | **exists**, 8837 entries, opens clean |
| ground-truth labels | `panda_express/labels_noise100.json` | tiny | **exists**, 100 eps: 75 success / 25 failure |
| ground-truth labels (smaller set) | `panda_express/labels_noise50.json` | tiny | **exists**, 50 eps: 38 / 12 |
| the `dino_wm` repo | `~/wksp/dino_wm` | — | needed for its model/config code |
| 102 raw episodes (only if rebuilding) | `panda_express/tasks/jenga_mujoco/episodes/` | 2.4 GB | dual-cam PNGs + `trajectory_*.json` |

**So the minimum transfer for the Jenga go/no-go is ~1 GB** (checkpoint + one LMDB + labels), not
the ~27 GB of Jenga task data on disk, and **no rebuild step is required.** The raw episodes are
only needed if you want to regenerate an LMDB with different preprocessing.

Four other LMDBs exist and all open clean, if you need more or cleaner data:
`jenga_mujoco/jenga_single_clean.lmdb` (1.3 GB, 16903 entries),
`jenga_noise_50/jenga_single_100.lmdb` (1.1 GB), `jenga_noise_50/jenga_unified.lmdb` (6.8 GB),
`jenga_tilt_100/jenga_tilt.lmdb` (944 MB).

The labels carry more than a boolean -- `outcome`, `failure_step`, `peak_tilt_deg`, `failure_block`
-- at a 45 deg topple threshold. `peak_tilt_deg` is the useful one: it is a continuous
near-miss measure, which is much closer to the PROXIMITY quantity this method actually predicts
than a success/failure flag is. Median peak tilt is 12.7 deg against a 45 deg threshold.

### Disk

`results/phase_a/feat_*.npz` (13 GB) and `results/phase_c/latents.npy` (5.3 GB) are gitignored
caches. **Both are regenerable** — delete freely if you need the space; `phase_c_data.py` rebuilds
the latter in ~5 minutes.

---

## 4. Findings worth carrying to Jenga

**1. dino_wm's single-step training recipe is inadequate for a monitor.** `conf/train.yaml` has
`num_pred: 1`. One-step prediction is excellent (0.040 rad, at the readout's floor) and compounds
to 0.577 over 40 autoregressive steps, because the model never sees its own output as input. Fixed
by rollout fine-tuning — but **order matters**: teacher-force first to learn the dynamics, THEN
fine-tune on rollouts. Reversed, it collapses to predicting that nothing ever happens
([1,59,0] against a true [15,28,17]).

**2. MSE hedges at bifurcations.** MSE's minimiser is the conditional mean; near a boundary the
mean of a bimodal future sits BETWEEN the basins, which with attractors at fall-left / upright /
fall-right is "upright". Predictions compress toward the middle attractor. **Diagnostic: compare
predicted vs true basin COUNTS.** If the Jenga model's predicted terminal states are compressed
toward "nothing toppled", the same thing is happening there.

**3. The same MSE property is an ASSET against nuisance.** The predictor replaces unpredictable
components (shadow pixels, render noise) with their mean, tightening clusters 21% without moving
them closer. One mechanism: liability at decision boundaries, asset against appearance.

**4. Clustering needs PCA in patch-token space.** Single-linkage returned k=300 of 300 in the raw
98,304-dim space — and so did the control on encoded truth, which is what ruled out the model.

**6. JENGA'S BASINS ARE REAL AND DISCOVERABLE — but only after removing the arm.** Verified on
100 labelled episodes, real final frames, encoder only. Peak neighbour tilt is sharply bimodal (75
episodes under 19 deg, 25 above 90, NOTHING between -- a 72.2 deg gap). PC1 of the raw latents is
end-effector pose, not the blocks, so **regress proprio out first** -- it uses no outcome labels,
and without it k-means at k=2 sits at CHANCE (50%). With it: separation 1.50 -> 2.15,
nearest-centroid 94% -> 98-99%, correlation with peak tilt 0.55 -> 0.77. HDBSCAN then finds k=2 at
98.8% with nothing supplied.

**What this does NOT establish:** those are REAL frames. The monitor reads PREDICTED endings, and
the settle-tail problem below is still open.

**5. The settle tail HOVERS rather than freezing, and that is fine.** `||z_t+1 - z_t||` decays then
flatlines at 0.60% of scale -- it never reaches zero. That was recorded as a failure until the right
question was asked: the monitor reads only WHICH attractor is nearest, so what matters is whether
the LABEL is stable. Measured over 120 episodes it is 99.2% stable by settle step 25 and **100% from
step 40**. A hover INSIDE a basin, not a drift ACROSS basins. It is also physically faithful -- a
rocking block loses speed at each impact (e_r = 0.824) and approaches upright geometrically without
exactly arriving.

**Acceptance for any settle test should be label stability, not `||dz|| -> 0`.** The original
criterion measured a quantity the method does not use.

---

## 5. Traps that cost real time here

* **Check the CONTROL before blaming the model.** Phase E looked like a model failure until the
  same discovery run on encoded truth failed identically.
* **A measurement that fails the EASY case cannot be trusted on the hard one.** Phase B's first
  verdict was FAIL; the tell was theta (directly visible) scoring 0.856. Two probe bugs, not a
  representation limit.
* **Label per (state, chunk), never per episode.** The monitor asks "from here, does THIS chunk
  straddle a boundary?". A whole-episode margin answers something else.
* **Score densely when MEASURING.** k spikes and decays. 3 scoring times gave recall 0.611; 8 gave
  0.820. Deployment rate (one score per action chunk) and evaluation rate are different questions.
* **`auc()` in `eval/run_phase3_block.py` uses ordinal ranks, not average ranks** — it misreports
  tied integer scores like k (0.804 vs the correct 0.808). Unfixed.
* **Verify every patch, and chain the run behind `&&`.** Silent no-op patches followed by a launch
  on the stale file cost three runs in one session.

---

## 6. Next steps, ranked

### The Jenga go/no-go is PARTLY DONE. What is settled:

| check | status |
|---|---|
| do toppled/intact scenes separate in DINOv2 space? | **YES** — 98-99% nearest-centroid, separation 2.15 (`eval/jenga_geometry.py`) |
| are the basins real, or a continuum? | **REAL** — peak tilt bimodal, 72.2 deg gap, 0 episodes in 20-60 deg |
| can the count be found with no labels? | **YES** — HDBSCAN finds k=2 at 98.8% (`eval/jenga_linkage.py`, `eval/cluster_shootout.py`) |
| does latent distance track HOW FAR it tipped? | **YES** — r = 0.77 with `peak_tilt_deg` |

**All of that used REAL final frames and the encoder only.** The first predicted-ending run below
shows that the checkpoint does not preserve this structure.

### 1. Diagnose/fix the predictor before J5

The method reads the ending after "stop acting and let the scene settle". **Jenga demos contain no
held poses at all**: over 6771 steps of demo actions the median per-step EE displacement is 5.3 mm,
only 0.6% of steps move under 1 mm, and the longest run of consecutive sub-millimetre steps is
**ONE**. So a zero-action tail is pure extrapolation, and asking "does it settle?" would be
answered by OOD behaviour rather than physics.

Ruled out already: **dropping the tail entirely**. Measured on the toy against the fully-settled
outcome, tail=0 finds the WRONG count (2 instead of 3) and scores 74.3% against a 70.7% baseline.

The first nominal-continuation run used all 100 episodes and remained finite for 80-330 tail steps.
It did not yield a robust stable assignment: 2-D/5% HDBSCAN changed from five to three clusters
between the 90% and full checkpoints and retained only 61.8% of joint known assignments. Some
higher-dimensional settings reached 93-96% late agreement, but only on 55-72% joint coverage and
with memberships that do not match the physical outcome. No tail is accepted yet.

### 2. Predicted-ending structure — run 1 failed

Everything above is the encoder on real photographs. The monitor reads the world model's PREDICTED
endings, which carry error. Phase C predicts trouble: the Jenga checkpoint uses dino_wm's shipped
`num_pred: 1` recipe, which on the toy gave 75% basin agreement and hedged toward "nothing
happened".

At the full nominal tail, 2-D/5% HDBSCAN finds k=3, 67.7% agreement, 96% coverage, and separation
1.02. The 2-D/10% setting finds k=2 but reaches only 74.1% agreement, 85% coverage, and separation
1.02. No setting in the 2/4/8/16-PC by 5/10% sweep meets the >=90% agreement and >=1.5 separation
gate. Its two clusters mix outcomes (54 intact / 12 toppled and 10 intact / 9 toppled), so this is
not just a mildly compressed 75/25 count.

The full-trajectory J4 stress test confirms general long-rollout failure rather than simple count compression. Real separation grows
from 1.06 at H=8 to 2.21 at 75% of the nominal continuation; predicted separation remains near
1.0. At the end, the predicted outcome axis has only 38.7% of the real magnitude and cosine 0.40
with the real direction. Nearest-real-centroid counts are 72/28, deceptively close to 75/25, while
episode accuracy is only 67%. Rollout-fine-tune the existing checkpoint, then rerun J1-J3; do not
start the monitor while J3 fails.

The corrected chunk-local simulator test changes the immediate diagnosis. Across 2,049 H=8 chunks,
tail 0/5/10 reveals 36/54/57 newly initiated topples from 1,702 upright starts, so tail 5 is the
physical knee. Short-horizon predicted geometry remains close to real geometry (tail-5 separation
2.41 vs 2.45; outcome-axis cosine 0.99). But only 11.6% of predicted and 12.8% of real tail-5
endpoints are supported by the terminal basin model. The blocker is now how to represent mixed
mid-trajectory endpoints without silently turning terminal attractors into context clusters.
Because the LMDB lacks MuJoCo state snapshots, action replay uses deterministic re-sampled reset
variation; it matches original episode outcomes at 79%. Tail comparisons are paired and causal,
but the simulated absolute failure rate is not the original dataset's rate.

The first proposed fix—cluster local endpoint changes `delta-z` per chunk after subtracting a
quadratic response to probe magnitude—also fails. On a stratified 100-chunk x 50-probe simulator
test, it finds all 10 physical boundaries but false-alarms on 84/90 non-boundaries. More decisively,
the encoded-real control alarms on all 100 chunks. HDBSCAN is partitioning ordinary smooth
nonlinear response geometry, not detecting discontinuity. Retraining DINO-WM cannot fix a detector
that fails on real latents. The next formulation must use the probes' one-dimensional ordering to
test continuity directly rather than clustering an unordered point cloud.

That ordered formulation is now implemented. A global cubic and a continuous piecewise cubic are
compared with an otherwise identical piecewise cubic that permits a jump. BIC penalizes model
complexity and the split search; a candidate also needs an adjacent latent increment at least 3x
ordinary increments. On the same encoded-real 100 chunks it detects 7/10 physical boundaries and
produces 8/90 non-boundary alarms (precision 46.7%, recall 70%, F1 0.56). Quiet-control alarms drop
from 50/50 to 5/50. This is a major correction, but 10% misses the predeclared <=5% gate;
PCA2/8/16 also miss at 6-8%. Predicted curves were therefore not scored. Before tuning the jump
ratio, cache continuous peak tilt, block pose, and contact state per probe; some apparent false
positives contain abrupt encoded-real changes and may be sub-threshold physical events. Results:
`results/jenga/ordered_change_points.json` and `results/jenga/ordered_change_points.png`.

Physical attribution is now complete. Per-probe peak/end tilt, all block poses, transient and
endpoint contact signatures, and end-effector position were cached from the same deterministic
branches. Only one of the five quiet alarms has a material neighboring-block change: ep67/chunk66
moves a side block 2.48 mm and rotates it 2.01 degrees without reaching 45 degrees. None changes a
contact category. Ep99 moves the manipulated middle block 0.33 mm; the other three have effectively
fixed blocks and only tiny robot/raster changes. The coarse topple label therefore explains one
residual alarm, not the detector's remaining false-positive floor. Do not retroactively call 4x a
passing threshold on these controls, even though it would numerically leave 2/50 alarms. Freeze it
and test new controls, preferably together with a spatial readout restricted to neighboring blocks.
See `results/jenga/physical_jump_diagnostics.json` and
`results/jenga/quiet_alarm_inspection.png`.

The frozen 4x validation is also complete. It uses 50 new chunks from the 25 episodes absent from
the original probe diagnostic, with two temporally spread points per episode. All were physical
non-boundaries; 3 alarmed, for 6% versus the <=5% gate (exact 95% interval 1.25-16.55%). An initial
selector bug used only action starts 2 and 10 and produced 2/50; that easy-prefix result is invalid.
On the original mixed set, 4x retains 7/10 real boundary detections with 3/90 false alarms. A model
score was exposed after the flawed preliminary pass: predictions detected 0/10 boundaries and made
4 false alarms. Because the corrected external gate fails, this is exploratory rather than formal,
but it strongly warns that DINO-WM smooths away real jumps (predicted adjacent ratios are <=2.41 on
all ten boundary chunks). Next test a neighboring-block-only spatial latent on real endpoints before
reopening prediction scoring. See `results/jenga/ordered_holdout.json` and
`results/jenga/ordered_change_points_4x.json`.

**Correction and extension: the middle block is now part of failure.** Failure is either neighbor
at 45 degrees, middle at 45 degrees, or middle COM below the table; safe upright extraction is
allowed. Re-labeling leaves the original union at 10 boundaries (7 detected), but exposes two
middle-specific edge transitions with only 5 and 2 safe probes—outside the fitter's minimum-eight-
per-side support. The 2,049-chunk all-block tail scan finds 48/69/77 new failures from safe starts
at tails 0/5/10. Tail 10 adds eight over tail 5, including six middle topples, so it is preferred if
the five extra rollout steps meet the live budget.

The balanced 80-scenario experiment uses 20 each of middle failure, neighbor failure, safe
extraction, and quiet control. At eps=.10 there are no supported middle boundaries. At eps=.30
there are three supported middle and six supported neighbor boundaries (seven in the union).
Full-frame detects 6/7 union and 1/3 middle; a valid simulator-oracle all-three-block mask detects
5/7 union and also 1/3 middle. Both produce 0/20 quiet alarms. Spatial masking therefore does not
solve middle sensitivity. The originally drawn red fixed ROI misses the blocks at some stages and
is invalid; only the green segmentation-derived mask is a valid upper-bound test. The next blocker
is a middle-relevant probe/response signal and more naturally near-boundary middle data, not crop
tuning. Results: `results/jenga/all_block_relabel.json`, `results/jenga/all_block_tails.json`,
`results/jenga/spatial_latents.json`, and `results/jenga/spatial_latents_eps03.json`.

**Expanded neighbor-only validation fails (2026-09-15).** With middle events excluded, the frozen
H=8/tail-5/eps=.10/PCA4/cubic/min-side-8/4x detector was tested on 250 temporally spread quiet
states absent from both earlier probe caches. It alarms on 16/250 = **6.4%** (exact 95% CI
3.70-10.19%), so the <=5% real-control point gate still fails. An exhaustive physical screen of
all 1,302 remaining eligible states found only **11** supported neighbor boundaries, not the
planned >=30. On all 11, encoded-real recall is **5/11 = 45.5%** (exact 95% CI 16.75-76.62%): the
>=80% recall gate also fails. Seven of the 11 physical outcome sequences switch more than once as
probe strength increases, so the assumed single-change-point structure is often false.

The paired world-model audit is conclusive for this checkpoint: DINO-WM detects **0/11** supported
boundaries and makes 5/250 quiet alarms. Its median adjacent-jump ratio on boundary states is 0.81
versus 6.64 for real encodings; the median predicted/real ratio is **14.8%**, and real/predicted
jump-ratio correlation is -0.075. Low predicted FPR is smoothing, not success. Do not deploy or
threshold-tune this monitor. The next defensible choices are (a) collect more naturally near-
boundary trajectories and train a counterfactual/multimodal model with a local-jump objective, or
(b) replace image-latent bifurcation detection with an object-pose/physics or supervised neighbor-
topple risk model. Results: `results/jenga/neighbor_validation.json`; the paired cache and physical
screen are gitignored.

**Multi-peak physical-oracle gate (steps 1-4, 2026-09-15):** a sliding local expansion detector
tests every adjacent probe pair with a continuous piecewise-linear null versus the same local fit
plus a step. BIC pays for the step and a `2 log(K)` search penalty; positive evidence is the fixed
alarm rule. Adjacent positive locations are grouped into multiple peaks, so the score no longer
assumes one global transition. A constant-response invariant is essential: otherwise BIC compares
round-off-sized residuals and invents evidence in deterministic states.

To obtain enough boundary cases without widening the evaluated perturbation, 300 untouched states
were screened with 21 broad eps=.30 probes; the nearest transition recentered a fresh 50-probe
eps=.10 family. Sixty broad families crossed a transition and **59** recentered families retained
at least eight outcomes per side. The frozen evaluation uses 100 untouched quiet controls plus 30
episode-spread recentered boundaries. Labels construct and evaluate this stress set only; the
score does not use them.

After correcting the physical vector to exclude three middle-only contact channels, endpoint
neighbor position/orientation/contact deltas produce **26 TP / 0 FP / 4 FN / 100 TN**: 86.7%
recall (95% CI 69.3-96.2%), 0% FPR (upper 95% bound 3.62%), F1 .929, AUC .868. This passes both
physical-oracle point gates; 14/30 boundaries still switch physical outcome more than once. On
exactly the same probes, real DINO gives 30/30 recall but alarms on 100/100 controls
(AUC .599), proving the representation injects local visual variation unrelated to block state.
DINO-WM detects 12/30 and alarms on 49/100 controls (AUC .469). This initially suggested a
representation bottleneck; expanded hard negatives below show that the physical score also needs
revision before representation training. Results:
`results/jenga/multipeak_oracle.json`; raw discovery and paired caches are gitignored.

**Trajectory-level physical oracle — PASS:** the same 130 states were rerun while recording the
neighbor-only physical vector after each of H=8 actions and all five held steps. A local expansion
score is computed independently at each of 13 times, then combined as
`2 log(mean(exp(evidence_t / 2)))`; this is the BIC/Bayes-factor evidence for expansion at an
unknown time and automatically penalizes temporal search while rewarding persistence. The fixed
zero-evidence alarm gives **30 TP / 0 FP / 0 FN / 100 TN**: precision, recall, F1, and AUC all 1.0.
The endpoint from the same cache reproduces 26/30, proving the four endpoint misses contain earlier
transient expansion. Detected boundaries have positive evidence at a median 6.5/13 times. This
passes only the constructed boundary/static-control benchmark; expanded hard negatives below
invalidate a general gate-pass claim. Results:
`results/jenga/trajectory_oracle.json`.

**Visual trajectories and expanded controls (2026-09-16) — generalization FAIL:** all 13 real
frames were rendered and DINO-encoded for the same 100 quiet controls and 30 recentered boundaries.
Frozen full-frame PCA4 yields 30/30 detections but **100/100 quiet alarms** (AUC .489). Even a
privileged simulator-segmentation neighbor-patch pool, reduced by unlabeled transductive PCA4,
yields 30/30 and **100/100** (AUC .331). All neighbor masks were visible. On quiet states, false
visual evidence starts by time 2 in 97/100 full-frame cases and lasts a median 12/13 times; the
paired physical score is zero. Block-pixel pooling alone does not remove action/raster/occlusion
nuisance. Results: `results/jenga/visual_trajectories.json`.

The physical dataset was expanded with **all 38 chunks** from the only two episodes whose nominal
neighbor tilt stays <5 degrees throughout, plus 100 episode-spread nominal moving non-topples
(neighbor displacement >=2 mm, nominal tilt 5-45 degrees). All 50 operational probes were
re-simulated. The trajectory oracle alarms on **5/38** safe silent chunks (13.2%). Among moving
examples, 88 remain safe under all probes, six have supported mixed outcomes, five have low-
support mixed outcomes, and one has certain failure. It alarms on **87/88** verified moving-safe
chunks (98.9%). Pose-only and position-only ablations still alarm on 88/88 moving-safe chunks, so
discrete contact features are not the sole cause. These are alarms on outcomes safe *within the
operational probe range*, not proof that the states are far from a topple boundary. The old
static-control gate does not establish proximity specificity. Do not retrain the visual
representation or world model against this score yet. First measure each moving-safe state's
physical distance to a topple boundary with a wider adaptive probe search, then compare alarm
against actual margin without fitting to failure labels. Results: `results/jenga/expanded_controls.json`.

**No-alarm red-block picks (2026-09-16):** screened 2,049 nominal chunks for middle-block rise
>=2.5 cm and sideways travel >=2 cm, with no nominal neighbor topple. Of 67 candidates, 66 are
safe under all 50 operational probes and 55 of those produce no physical alarm. Three rendered
examples from different episodes show ~8 cm middle-block lift, 3-4 cm sideways motion, zero
neighbor tilt, and no alarm. This confirms that the physical score can remain quiet during a
successful pick; it does not settle whether alarms on moving neighbors indicate genuine proximity.
Search: `eval/jenga_pick_noalarm_search.py`, result: `results/jenga/pick_noalarm_search.json`;
videos: `results/jenga/pick_noalarm_videos/`.

**Grasp/contact clarification:** the carry-phase videos above do not show the critical approach.
Two further no-alarm clips start with the red block at table height between its neighbors; it
lifts 4.2/7.5 cm while the end effector is ~4.8/5.1 cm from the nearest neighbor body center.
These clips have no recorded robot-neighbor contact. In the 67 screened pick chunks, all 55
safe/no-alarm cases had no nominal robot-neighbor contact; the 11 safe/alarm cases all did.
A broader check of all nominal chunks with robot-neighbor contact and >=2.5 cm red-block lift found
20 candidates, 19 safe under all 50 probes, and **0/19 no-alarm**. Thus there is currently no
verified no-alarm *contacting-neighbor* successful pick in this tested set. Results:
`results/jenga/contact_pick_check.json`; zoomed near-grasp videos:
`results/jenga/near_grasp_noalarm_videos/`.

**Full-episode alarm audit:** nominally replayed all 100 episodes. Fifty satisfy the explicit
successful-pick proxy (after settling: red lift >=2.5 cm and lateral travel >=2 cm; neighbor peak
tilt <45 degrees). Every one of these 50 has at least one independently verified physical alarm
decision. Therefore no verified *entirely alarm-free* successful episode exists under this
criterion, even though 55/66 selected pick chunks themselves were quiet. Four complete episodes
were scored at every non-overlapping H=8 decision point and rendered start-to-finish. Episode 24
alarms at the pick chunk 106 despite 50/50 probes staying safe; episodes 74 and 91 show no alarm
at their lift chunks 106/98 but did alarm earlier during approach. The video overlay is the most
recent decision's result, held until the next decision; it is not a continuously recomputed alarm.
Code: `eval/jenga_full_episode_alarm_audit.py`, `eval/jenga_full_episode_noalarm_screen.py`,
`eval/jenga_nominal_success_scan.py`, `eval/jenga_full_episode_videos.py`; results:
`results/jenga/nominal_success_scan.json`, `results/jenga/full_episode_alarm_audit.json`,
`results/jenga/full_episode_alarm_videos/`.

**Physical basin/dissent revisit and action-strength check (2026-09-16):** 50 ordered probes,
H=8, eps=.10, and held tails 5/10/20 were re-simulated at 30 recentered boundary states and
every H=8 decision in held-out successful episodes 24 and 74. The unlabeled reference atlas uses
the original diverse physical endpoints plus the 30 boundary families, with episodes 24/74
excluded. PCA4/HDBSCAN finds 2/3/3 basins at tails 5/10/20. It alarms on 10/30, 20/30, and
19/30 boundary families, respectively, but on 0/19 and 0/18 episode decisions at every tail.
Boundary detection is optimistic because the same boundary families contribute to the atlas;
episode results are held out. Noise is excluded from dissent. Reference coverage is 72%/85%/85%,
but key pick decisions can be completely unassigned (ep24 chunk 106 at all three tails). Thus
quiet does not mean a verified stable basin. Tail 10 is the most informative tested short tail,
yet the current global-basin atlas is not an adequate general proximity monitor. Full nominal
videos with all three scores: `results/jenga/physical_basin_episode_videos/`; analysis:
`results/jenga/physical_basin_tails.json`.

The original eps=.10 grid reaches a maximum coefficient .2326 times the within-chunk command
span (e.g. 8.5/11.8/18.2 mm in three sample chunks). A cached central-subset sensitivity check
suggested fewer alarms with narrower reach but confounded reach with probe count. A controlled
20-state pilot re-simulated the same 50 probes at eps=.10/.035/.0165 (maximum coefficients
.2326/.0814/.0384), with H=8+5 and frozen physical BIC rule. All 10 selected moving-safe states
alarmed at every strength; boundary alarms stayed 10/10, 10/10, 10/10, while true mixed-topple
responses fell 10/10, 10/10, 8/10. Oversized probes therefore do not alone explain this score's
poor specificity, although this pilot is too small to settle the appropriate physical uncertainty
scale. On the **same matched pilot**, frozen physical-basin dissent alarms on only 2/10, 1/10,
0/10 boundary families as strength narrows, with 0/10 moving-safe alarms throughout; mean
moving-safe basin coverage is only 29%/21%/20%. Lowering strength does not rescue basin detection;
it increasingly hides the boundary while leaving many points unassigned. Results:
`results/jenga/action_strength_sensitivity.json`,
`results/jenga/action_strength_resim.json`.

**Nominal-centered nearest persistent branch margin (2026-09-16):** implemented
`eval/jenga_persistent_branch_margin.py`. A fixed 31-strength grid from coefficient -.6 to +.6
perturbs the original nominal 8-action chunk, then records physical neighbor endpoints after
5 and 10 held steps. A branch must differ from nominal at both tails and have the same alternative
at two adjacent strengths; the first sustained change gives a coefficient and commanded-motion
millimeter bracket. The evaluation-only physical oracle uses endpoint neighbor tilt >=45 degrees,
separate from the unlabeled detector. On 15 held-out boundary families, **13/15** oracle margins
are bracketed (median upper bracket 3.01 mm); the remaining two change only at the grid edge and
are unconfirmed. Among 20 previously operationally safe moving states, **1/20** has a bracketed
physical margin (4.36-5.45 mm) and three have edge-only candidates. None of 10 silent or 10
contacting-pick controls has a bracketed physical branch on this grid. These are selected cohorts
and one action direction, not population rates or a full robustness certificate.
The previously fitted trajectory/BIC score alarmed on 19/20 of these moving states despite only
one confirmed persistent topple branch on the wider grid; this strengthens the specificity
concern, but unseen directions or unsampled narrow branches remain possible.

The unlabeled physical-basin estimator produced **0/15** boundary margins: all 15 nominal
boundary endpoints were unassigned by the held-out global atlas (also 20/20 moving nominal
endpoints); mean grid coverage on boundary/moving states is only 7.7%/11.5% despite 82-84%
reference coverage. This directly identifies the atlas as unusable for nominal-centered margin
estimation. Do not reinterpret its `not_found`/quiet outputs as large margins. The next method
must compare local physical response regimes without relying on global terminal-cluster coverage,
then be evaluated against these frozen oracle brackets before visual translation. Results:
`results/jenga/persistent_branch_margin.json` (cached simulator arrays ignored by git).

**Label-free local persistent-branch test (2026-09-16) — FAIL on specificity:** implemented
`src/local_persistent_branch.py` and `eval/jenga_local_persistent_branch.py` on the frozen
55-state physical response cache. At each neighboring-strength gap, a continuous local spline
competes with the same spline plus a jump under BIC; the gap must favor a jump at both held tails
5 and 10. A 6D PCA projection is learned from unlabeled physical reference endpoints with all
test episodes excluded. There is no topple label, global basin label, or fitted Jenga alarm
threshold in the detector. Nevertheless it brackets **15/15** held-out boundaries *and*
**20/20** moving-safe states, **10/10** safe contacting picks, and **1/10** silent controls.
Detected upper-margin medians are only ~0.30/0.28/0.51 mm for boundary/moving/contact groups,
versus the independent physical-oracle boundary median of 3.01 mm. Matching an oracle case in
binary terms is misleading: median absolute upper-margin error on the 13 confirmed boundaries
is 2.54 mm. Persistent local model preference is still reacting to smooth/contact-driven motion,
not isolating a persistent outcome regime. Do not claim the label-free margin works. Next test:
refine candidate action intervals at multiple resolutions; a genuine jump retains finite response
separation as the strength gap shrinks, while an ordinary smooth response shrinks with the gap.
Freeze this failed run as a baseline before any revision. Result:
`results/jenga/local_persistent_branch.json`.

**Two-level action-resolution refinement (2026-09-16) — still fails specificity:**
`src/branch_scale.py` and `eval/jenga_branch_scale_refine.py` replay the frozen local-BIC candidate
gap at its midpoint, follow the half with the larger physical endpoint difference, and bisect
again. A one-parameter constant-separation model competes with a one-parameter
width-proportional-shrinking model, independently at held tails 5 and 10. No topple label or
Jenga-fitted alarm threshold enters the decision. On 46 frozen candidates, constant separation
wins for **11/15** boundary states (10/13 with independently confirmed persistent topple
margin), but also **14/20** moving-safe states and **9/10** safe contacting picks; the one silent
candidate is rejected. These are the verified numbers after replaying the original endpoints
alongside each new midpoint. Median retained quarter-width separation is ~0.86 boundary, ~0.84
moving, and ~1.02 contacting-pick, not the .25 expected for a locally linear smooth response. The
5- and 10-step endpoints are effectively identical for the contacting-pick probes (median raw
endpoint difference <1e-9), so requiring both tails is not independent evidence. Cached versus
freshly replayed endpoints sometimes differ by binary contact-feature flips (errors up to 2 in
27D physical state), but **pose-only** analysis on the same refinement intervals still retains
separation on 15/20 moving and 9/10 contacting-pick cases. This is not just a contact-channel
cache artifact. Tiny or nonmonotonic pose response can retain separation without a confirmed
topple branch. Ablation: `results/jenga/branch_scale_ablation.json`.
This refinement was rejected on the development panel; no fresh-holdout success claim. It also
refines only the nearest BIC gap, so rejecting that gap does not rule out a farther true branch.
Result: `results/jenga/branch_scale_refine.json`.

**Controller-scaled predictive consequence probe (2026-09-17) — specificity gate FAIL:**
`src/action_uncertainty.py` and `eval/jenga_tracking_uncertainty.py` fit a one-step linear
command-to-proprioception lag and residual covariance on 57 training episodes, excluding the
43 episodes used by the frozen 55-state panel. On 7,224 held-out samples, median/p90 residual
norms are 1.465/3.56 mm; the training residual radial q90 scale covers 92.3% of held-out
samples. This is an empirical proxy for execution variation, not a certified uncertainty bound.
`eval/jenga_predictive_regime_probe.py` probes three covariance axes at 17 strengths each,
perturbs H=8 commands, then holds the *same nominal target* for every probe. It records the
pose of all three blocks at H=8, H+5, H+10. An unlabeled PCA6 ordered continuous-vs-step fit
must pick the same neighboring strength gap at both tails. Frozen development-panel alarms:
9/20 moving non-topple, 3/10 contacting lifts, 9/15 selected boundaries, 1/10 silent.
The independent neighbor-topple oracle is mixed within this controller-scaled envelope on
only 8/15 selected boundary states, of which 5 alarm. Thus reduced probe reach explains some
misses, but the detector still confuses ordinary consequential motion/contact with the desired
regime transition. Safe controls may have non-topple regimes; these counts specifically reject
using this as a *topple-proximity filter*, not every possible general-regime interpretation.
Exact replay variation is ~1e-15, so deterministic repeats cannot supply a noise floor.
`eval/jenga_predictive_variation.py` injects eight held-out training-episode residual snippets
as a diagnostic proxy. The median detected adjacent-gap size divided by the median pairwise
injected-jitter response at tail 10 is 0.81 moving, 0.08 contacting, 0.75 boundary, and 7.14
for the single silent alarm. The candidate gap is often no larger than plausible execution
variation, but this proxy is *not* measured repeated execution. No failure label was used to
fit the detector; its fixed BIC/jump rules still contain structural hyperparameters, so do
not call it fully calibration-free. No untouched holdout was run after this failed panel gate.
Results: `results/jenga/tracking_uncertainty.json`, `predictive_regime_probe.json`, and
`predictive_variation.json` in the same directory. Large simulator caches are intentionally
git-ignored; rerun the scripts in that order with the Jenga bundle/LMDB installed.

**Next agent:** first audit whether an independently observable, task-general *future
consequence* can separate routine contact/motion from distinct outcome regimes within a
realistic uncertainty set. Acquire or model actual executed-action variation rather than
treating target/proprio lag residuals as hardware noise. Freeze a new decision rule before a
new episode-level holdout, explicitly report reach/censoring and abstention, and only then
attempt visual representation transfer or live runtime integration. Preserve the main goal:
failure-label-free, zero-shot proximity to changed future behavior; Jenga topple labels are
evaluation-only. Do not tune a Jenga-specific score to these 55 states.

**Stage 0 answer key under realistic noise (2026-09-17):** `eval/jenga_stage0_noise_oracle.py`
replays each panel state's chunk with 64 contiguous tracking-residual snippets from non-panel
episodes, at 0.5x/1x/2x, then a 30-step hold. No detector is scored. Topple labels are settled by
hold step 10 (100% agreement with step 30). Mixed outcomes (>=2 per side) at 1x/2x:
boundary 7/11 of 15, moving non-topple 2/8 of 20, contact lift 0/0 of 10, silent 0/0 of 10.
Contact and silent states are true negatives at every scale. The "boundary" and "moving" names
are not reliable labels under realistic noise, so Stage 1 must be graded against these
per-scale grades. Details: NOTES.md and `results/jenga/stage0_noise_oracle.json`.

**Stage 1 two-mode test on true endings (2026-09-17):** `src/outcome_modes.py` and
`eval/jenga_stage1_outcome_modes.py`. Recall on mixed states is 100% at every scale. False
alarms on unanimous-safe states: 9/52, 8/43, and 0/29 at 0.5x/1x/2x; contact-lift and silent
states never alarm. The false alarms are real non-topple splits (a neighbor slides 3.5-10 mm or
rests tilted in some runs and not in others). Ending spread alone is perfect on this panel
(AUC 1.0), so the panel cannot show the two-mode test adds anything. Open decision: do such
non-topple splits count as "different outcomes"? A harder benchmark with small-consequence
positives is also needed.

**Fresh holdout, 140 unseen states (2026-09-17):** 1,168 chunks in the 57 non-development
episodes were screened with the Stage 0 physics, then 60 topple-fork / 20 physical-fork /
60 quiet states were selected by a pre-declared rule. Three frozen rules were scored
(`eval/jenga_stage2b_multimode.py`, `results/jenga/holdout_stage2b.json`). Recall on mixed
states at 1x / false alarms on quiet states: with a shared hold, the PC1 two-group rule gets
96% / 3%; with the arm moving, it drops to 78% and the coarse-persistence multi-group rule gets
93% / 5%. The strict multi-group rule fails everywhere (17-30%). At 2x the coarse rule reaches
14% false alarms with the arm moving. No single rule wins both hold regimes; do not select per
case. Details in NOTES.md.

**Stage 2, universal image version (2026-09-17):** `eval/jenga_stage2_visual_forks.py` runs the
same split test on full-frame DINO latents of the rendered endings, using no object knowledge.
At 1x it alarms on 8/9 topple forks and 0/23 quiet states, and at 2x on 19/19 and 0/13. The
nudge forks mostly go unseen (1/8). The persistence check (same split at hold steps 10 and 30)
is essential: without it, 10/23 quiet states alarm on image-latent noise. Two alarms that looked
false are real forks on the grasped block (in ep25 c90 the red block drops in 4/64 runs). Next:
the same test on world-model *predicted* endings, then a fresh holdout. Details: NOTES.md.

### 3. Competing baselines on PROXIMITY

Never measured, and "same performance, no threshold" is the entire paper claim. Tuned-delta FTLE
(the predecessor), div_std (0.818, needs a fitted percentile), Mahalanobis/kNN on safe-trajectory
latents.

### 4. The warm-start ablation

GTF from scratch at MATCHED compute. "Warm start required" is currently a claim, not a result — the
successful run changed initialisation AND training volume together.

### 5. Matched-impulse actions (system-level universality)

The toy is shortcut on OUTCOME (`sum|a|` -> AUC 0.925) but NOT on proximity (best fitted baseline
0.764 vs the method's 0.808). Action pairs with matched total impulse differing only in timing would
force magnitude baselines to chance by construction.

---

## 7. Paper position, honestly

Toy-only is not a robotics paper — no reviewer accepts a rocking block as the evaluation for a
manipulation safety monitor. The defensible contribution is **deleting the threshold** plus the
**sizing rule** that states a detectable margin a priori, which nothing in the runtime-monitoring
literature does.

That literature is crowded (Sentinel CoRL'24, PATCH 2026, SAFE, Rewind-IL) but **every one of them
is a failure detector and every one needs calibration**, mostly conformal prediction with a
labelled calibration set. The proximity framing plus a counting rule is differentiated on both
axes. FTLE-for-safety is taken (arXiv 2508.15588) but offline, on the true environment, without
action perturbation.

Jenga is what makes it a paper. The toy makes it a methods contribution.
