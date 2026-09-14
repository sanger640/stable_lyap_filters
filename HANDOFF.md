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
  5. PCA -> assign to nearest attractor centroid
  6. k = number of probes NOT in the plurality
  7. ALARM if k >= 2.   No threshold. No labels. No calibration.

OFFLINE, once: discover the attractors by merge-distance PLATEAU on PREDICTED endings
               (PCA first, drop singletons). The count is never supplied.
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
| **DINO-WM on images** (toy) | phases A–F done — **AUC 0.882 [0.805, 0.949]**, prec 0.766, rec 0.857 |
| **Jenga** (the real task) | **not started** |

**The headline result:** the monitor transfers from an exact 2-D state to DINOv2 patch features
with **no retuning** — same eps, n, k_min, probe family — and matches or beats the state-based
version while being scored ~4x more sparsely.

**The caveat that must travel with it:** this used a ROLLOUT FINE-TUNED predictor. dino_wm's
shipped `num_pred: 1` recipe gives 75% basin agreement and was NOT used. So this validates the
architecture, not the existing Jenga checkpoint.

---

## 3. Environment and how to run things

**Everything runs in the `dino_wm` conda env.** `tipping_block.py` is pure numpy and runs there too.

```bash
PY=/home/sanger/miniforge3/envs/dino_wm/bin/python
cd ~/wksp/stable_lyap_filters
$PY -m pytest tests/ -q            # 23 tests, ~12 s, all should pass
```

GPU is an RTX 5060 Ti with **7.5 GiB** — small. Two traps it causes, both hit during development:
* batch 256 x 768 tokens needs ~9.6 GB for attention alone. Chunk to <=128.
* backprop through an unrolled 8-step ViT OOMs. Use gradient checkpointing.

### Pipeline, in order (DINO-WM track)

```bash
$PY eval/phase_c_data.py            # render + DINOv2-encode 600 eps  (~5 min, writes 5.3 GB)
$PY eval/phase_c_train.py           # teacher-forced predictor        (~8 min)
$PY eval/phase_c_train_gtf.py --tag gtf_warm   # rollout fine-tune    (~19 min)  <- REQUIRED
$PY eval/phase_d_settle.py          # settle + contraction tests      (~2 min)
$PY eval/phase_e_attractors.py      # attractor discovery             (~4 min)
$PY eval/phase_f_monitor.py         # the monitor, 100 eps x 8 times  (~53 min)
$PY eval/phase_f_videos2.py         # demo videos (scores are cached) (~5 min)
```

Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for anything that trains or rolls out.

### Moving to another machine

Everything needed for the TOY work is in this repo. Verified-working versions are pinned in
`requirements.txt` (python 3.11.15, torch 2.11.0+cu128, numpy 2.4.6). **torch must be a cu128
build** -- the GPUs used here are Blackwell (sm_120) and cu121 wheels will not run.

```bash
git clone git@github.com:sanger640/stable_lyap_filters.git && cd stable_lyap_filters
python -m pytest tests/ -q                       # 23 tests, ~12 s -- verifies the port
```

**The trained predictors are not in the working tree but ARE recoverable from git history** --
they were tracked until commit `b48efbc`, so a full clone still carries the blobs:

```bash
mkdir -p results/phase_c
git show b48efbc^:results/phase_c/predictor_gtf_warm.pt > results/phase_c/predictor_gtf_warm.pt
git show b48efbc^:results/phase_c/predictor.pt          > results/phase_c/predictor.pt
```

(Or just retrain: 8 min + 19 min. The recovery is only worth it to reproduce exact numbers.)

**Regenerate, do not copy**, the large caches -- `results/phase_c/latents.npy` (5.3 GB, ~5 min via
`phase_c_data.py`), `results/phase_a/feat_*.npz` (13 GB), `phase_e_endings_*.npz`.

**Reproduce one known result to confirm the port**: `eval/phase_e_attractors.py` should find a
plateau at k=4 dropping to 3 singletons-removed, ~4 min.

**For the JENGA work you also need, from outside this repo:**
| what | where | size |
|---|---|---|
| the `dino_wm` repo | `~/wksp/dino_wm` | — |
| the world model checkpoint | `dino_wm/outputs/model_latest_single.pth` | 360 MB |
| the 102 raw Jenga episodes | `~/wksp/panda_express` | — |

`labels.json` and the eval LMDB are absent but rebuildable from those episodes.

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

**1. The Jenga go/no-go.** The only step that can invalidate everything. The world model checkpoint
EXISTS at `~/wksp/dino_wm/outputs/model_latest_single.pth` (378 MB) — `RESUME.md` still wrongly
claims it is missing. `labels.json` and the eval LMDB are absent but rebuildable from the 102 raw
episodes.

Run in this order, each step cheap and able to end the plan:
  a. `eval/phase_b_geometry.py` adapted to Jenga latents — **does pose dominate nuisance?** The toy
     measured 1.58. If Jenga falls below 1, nearest-centroid fails regardless of predictor quality.
     A neighbour tipped 15 degrees is a far subtler distinction than a block flat on its face.
  b. Roll the checkpoint with a settle tail — **does the basin LABEL stop changing?** (not
     whether motion stops; see finding 5)
  c. PCA + plateau on predicted endings — **does a plateau appear?**
  d. Only then check clusters against `labels.json`, as a CHECK, never a fit.

Phase C predicts trouble at (b)/(c): the Jenga model has the same single-step recipe that failed.

**2. Competing baselines on PROXIMITY.** Never measured, and "same performance, no threshold" is
the entire paper claim. Tuned-delta FTLE (the predecessor), div_std (0.818, needs a fitted
percentile), Mahalanobis/kNN on safe-trajectory latents.

**3. The warm-start ablation.** GTF from scratch at MATCHED compute. Currently "warm start
required" is a claim, not a result — the successful run changed initialisation AND training volume
together.

**4. Matched-impulse actions** (system-level universality). The toy is shortcut on OUTCOME
(sum|a| -> AUC 0.925) but NOT on proximity (best fitted baseline 0.764 vs the method's 0.808).
Generating action pairs with matched total impulse that differ only in timing would force magnitude
baselines to chance by construction.

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
