# HANDOFF — stability-based safety monitor

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
| **Jenga** (the real task) | encoder checks done; J1 runtime implemented; J2-J6 pending |

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
$PY -m pytest tests/ -q            # 27 tests, ~12 s, all should pass
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
$PY eval/jenga_j1_fidelity.py       # bundled model + LMDB, error vs horizon (J1)
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
python -m pytest tests/ -q                 # 27 tests
python eval/jenga_j1_fidelity.py --episodes 10
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
python -m pytest tests/ -q                       # 27 tests, ~12 s -- verifies the port
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

**All of that used REAL final frames and the encoder only.** The two things it does not touch are
exactly what remains.

### 1. THE SETTLE TAIL — the live blocker, and it needs a design decision

The method reads the ending after "stop acting and let the scene settle". **Jenga demos contain no
held poses at all**: over 6771 steps of demo actions the median per-step EE displacement is 5.3 mm,
only 0.6% of steps move under 1 mm, and the longest run of consecutive sub-millimetre steps is
**ONE**. So a zero-action tail is pure extrapolation, and asking "does it settle?" would be
answered by OOD behaviour rather than physics.

Ruled out already: **dropping the tail entirely**. Measured on the toy against the fully-settled
outcome, tail=0 finds the WRONG count (2 instead of 3) and scores 74.3% against a 70.7% baseline.

The live option is a **nominal-continuation tail**: append the policy's own remaining actions
instead of freezing. The arm lifts and retracts after every grasp, so a neighbour that was going to
topple does so while the robot moves away — in-distribution, free, no retraining. One shared tail
across all probes keeps probe-to-probe differences coming only from the chunk. Fallback is
fine-tuning on held-pose data generated in the MuJoCo sim.

### 2. Does the PREDICTOR preserve the structure the encoder has?

Everything above is the encoder on real photographs. The monitor reads the world model's PREDICTED
endings, which carry error. Phase C predicts trouble: the Jenga checkpoint uses dino_wm's shipped
`num_pred: 1` recipe, which on the toy gave 75% basin agreement and hedged toward "nothing
happened".

**The diagnostic is one line and needs no retraining:** roll the checkpoint, cluster the endings,
and compare the predicted basin COUNTS against the true 75/25 split. Compressed toward "nothing
toppled" means it is hedging, and the fix is ~19 min of rollout fine-tuning on top of the existing
checkpoint (teacher-force first, THEN rollouts — the order matters, reversed it collapses).

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
