# Why systematically missed forks are invisible to the dynamics model

PLAN_NEXT.md Phase 2 deliverable (2026-09-21). Track A: privileged simulator state, Jenga-specific
physics used for diagnosis only. Model: the frozen best dynamics, GNN one-step + 0.5 input noise
(`w6_gnn_n5`, 10 seeds). Benchmark: frozen batch 3 at 1x (84 topple forks).

**In one line.** The model misses forks in which the gripper pushes an **upright** neighbour past its
tipping angle. It responds to the push but under-delivers the rotation by about 30% (11 deg vs 16 deg),
lands short of the ~15-18 deg tipping angle, and lets the block settle back upright. Its tipping physics
past that angle are correct, so the failure is confined to contact-driven rotation during the push,
a transition that is rare in the training data.

---

## 1. Which forks are systematically missed

Per fork, `q` = fraction of the 10 seeds that miss it, at the pre-declared operating point and at a
matched 5% test FPR (each seed's development threshold realises 2-8% FP on test, so the matched
version equalises operating points). `eval/jenga_blind_freq.py`, `results/jenga/blind_freq_gnn_n5.json`.

| scale | forks | robust blind (q >= 0.8) pre-declared / matched | robust under both |
|---|---|---|---|
| 0.5x | 42 | 7 / 10 | 6 |
| **1x** | 84 | 15 / 13 | **11** |
| 2x | 82 | 0 / 1 | **0** |

One seed's misses are mostly not systematic: seed 1 misses 28 forks at 1x, of which 12 are robust blind,
13 seed-sensitive and 3 caught by most other seeds. At 2x no fork is systematically missed.

## 2. Method

* **Groups.** 17 forks robust blind under either operating point, and 17 usually-detected controls
  (q < 0.3 under both), each matched to a blind fork on the number of probes that topple. A mechanism
  explains blindness only if it separates the two.
* **Dense re-simulation** (`eval/jenga_blind_resim.py`): every fork from its exact frozen start state,
  all 64 probes at 1x, full 61-dim state five times per control step.
* **Mechanism analysis** (`eval/jenga_blind_mechanism.py`, `results/jenga/blind_mechanism.json`):
  tilt course real vs predicted, a true-state handoff test, a reproducible class per fork, and
  training-data coverage. Response regimes and contact timing come from `eval/jenga_blind_analyse.py`
  (`results/jenga/blind_analysis.json`).
* **Population check**: start tilt against miss count for all 84 forks.

All of it reproduces from committed code; the mechanism script gives identical output on re-run.

## 3. Taxonomy

Classes are assigned by rule (`classify` in `jenga_blind_mechanism.py`) from the toppling neighbour's
start tilt, when the toppling probes pass 15 deg, and the topple count.

| class | mechanism | blind | control |
|---|---|---|---|
| **A1** | upright neighbour (start < 8 deg) pushed past 15 deg **during the perturbed chunk** | **10** | 4 |
| **A2** | upright neighbour pushed past 15 deg **after the chunk, early in the hold** (steps 10-13) | **4** | 0 |
| B | near-unanimous: >= 60 of 64 probes topple | 1 | 1 |
| C | fewer than 2 of 64 topple from the clean start state (weak by the benchmark's own mixed rule) | 2 | 1 |
| **D** | **pre-leaning** neighbour (start >= 8 deg) | **0** | **11** |

**14 of 17 blind forks are upright blocks pushed past tipping; none are pre-leaning. 11 of 17 detected
forks are pre-leaning.**

The same split holds across **all 84 forks**, not just the matched sample:

| | upright start (< 8 deg) | pre-leaning |
|---|---|---|
| forks | 45 | 39 |
| mean miss rate over 10 seeds | **0.52** | **0.18** |
| robust blind | **15** | **0** |
| usually detected | 10 | 28 |

Spearman(start tilt, miss count) = -0.45 over the 84. The upright penalty holds within narrow branches
(<= 8/64 topple: miss rate 0.65 vs 0.23) and within wide ones (0.40 vs 0.17). Branch width and
uprightness each contribute independently.

## 4. Mechanism

**The trigger is the same in both groups.** In blind and detected forks alike, the first contact flag
that separates toppling from stable probes is the gripper touching the neighbour (`left-robot` /
`right-robot` / `middle-robot`), divergence begins at about control step 4, the gripper starts about 50 mm
from the neighbour, and the toppling probes are equally identifiable by perturbation direction
(separability AUC 0.82 blind vs 0.79 control). What differs is the state of the block being pushed.

**Real vs predicted tilt of the toppling neighbour** (median over the probes that truly topple and over
seeds):

| | start | real, end of chunk | real, end | predicted peak | predicted end |
|---|---|---|---|---|---|
| blind | 3.9 deg | 15.8 deg | 90 deg | **11.1 deg** (about step 9) | **3.6 deg** |
| control | 9.5 deg | 44.0 deg | 90 deg | 90.0 deg | 89.7 deg |

A 25 x 20 x 75 mm block tips at roughly atan(10/37.5) = 15 deg to atan(12.5/37.5) = 18 deg. The real
push carries an upright block to about 16 deg and it falls; the model carries it to about 11 deg and it
settles back. A pre-leaning block crosses the tipping angle even with an under-delivered push.

**Response regimes** across the 17 blind forks x 10 seeds (spread over probes, 1 mm onset):

| predicted response | blind | control |
|---|---|---|
| responds and stays diverged | 36 / 170 | 160 / 170 |
| **responds, then reconverges** (final < half the peak) | **127 / 170** | 10 / 170 |
| never responds | 7 / 170 | 0 / 170 |

**True-state handoff: the tipping physics are correct and the push is not.** The model is handed the
true simulator state at control step k and rolls out the rest (fraction of truly-toppling probes it
then topples, median over forks):

| handoff step | true tilt, blind | model completes, blind | true tilt, control | model completes, control |
|---|---|---|---|---|
| 0 | 3.9 | **0.00** | 9.5 | 0.75 |
| 4 | 4.3 | **0.00** | 16.4 | 0.93 |
| 6 | 8.3 | 0.20 | 20.0 | 1.00 |
| **8** | **15.8** | **0.79** | 44.0 | 1.00 |
| 10 | 24.1 | 1.00 | 90.0 | 1.00 |

Given a block the physics has already carried to ~16 deg, the model finishes the topple almost every time;
from any state before the push has delivered that rotation, it never does. The failure is confined to
control steps ~4-10, while the gripper is in contact with an upright block. It fails even from the true
mid-push state at step 6, so this is a **systematic few-step bias in contact-driven rotation**, not error
compounding over a long rollout.

**Training coverage.** Of 70,480 training rollouts, 2,417 topple a neighbour; the toppling block's
median start tilt is 10.3 deg, and only **400 (16.5%)** start upright (< 5 deg), i.e. 0.6% of all
rollouts. The failing transition is rare but present.

**Exceptions.** B (ep 69 step 58): 61 of 64 probes topple and the model topples all 64, so the fork is a
3-probe stable minority it erases; its matched control (ep 84 step 54) is caught. C (ep 61 step 74, ep 71
step 106): from the clean start state the model is evaluated from, only 1 of 64 probes topples (Stage 0 had
2, from its drifted start states), so by the benchmark's own rule these are weak rather than mixed.

## 5. The plan's six questions

1. **Does the model ignore the perturbation completely?** No. 7 of 170 seed-fork runs never respond.
2. **Does it respond and then reconverge?** Yes. This is the dominant signature (127 / 170).
3. **Does it miss specific contact transitions?** It sees the contact; the gripper-neighbour contact is
   the separating event in blind and detected forks alike. It misses the **rotation** that contact
   imparts to an upright block.
4. **Are blind forks later-horizon events?** Partly. 4 of 17 (A2) are pushed in the hold after the
   perturbed chunk; otherwise divergence begins at about step 4, the same as controls.
5. **Are they associated with small initial geometric margins?** The opposite. They start FAR from the
   tipping angle (about 4 deg against 15-18 deg). Pre-leaning forks, which start close to it, are never
   robust blind (0 / 39).
6. **Are certain perturbation directions disproportionately missed?** No. Topple-inducing directions are
   as identifiable in blind forks as in detected ones (AUC 0.82 vs 0.79).

## 6. What this implies for D2

The taxonomy constrains the counterfactual objective more tightly than the plan's generic version:

1. **Where to supervise: the contact window, not the chunk start.** The decisive rotation happens over
   control steps ~4-10 (A1) and ~10-13 (A2); the median A1 fork passes 15 deg at step 6.8 and A2 at 11.7.
   A 2-5 step unroll anchored at the chunk start ENDS before the failure for most forks. Lengthening
   the horizon from the chunk start would reach it, but reintroduces the long-rollout gradient that halved
   recall. The better fit is **short unrolls (2-5 steps) anchored at true states inside the contact
   window**.
2. **That needs new counterfactual data.** Pairs sharing an exact state exist only at the chunk start in
   the current data. Same-state / different-action branches from states inside the contact window must
   be simulated, which the re-simulation code already nearly does. In Track A the branch points can be
   chosen at contact-state changes; a Track B equivalent would need a generic trigger (for example, states
   where the model's own predicted response to action perturbation peaks), since contact labels are not
   available there.
3. **What to compare: orientation, not only position.** Before a block passes the tipping angle almost all
   of the fork signal is in its rotation; positions barely move. `Phi` must include each object's
   orientation and angular velocity.
4. **Supervise each branch's response, not only the difference between branches.** The model
   under-rotates EVERY pushed upright block by about 30%, a level bias. Matching `Delta_pred` to
   `Delta_real` alone can leave a bias that is shared by both branches in place, so D2 should also match
   each branch's rotation response to its real counterpart over the unroll. Keep the invariance term for
   sub-critical pushes, which should stay sub-critical and relax.
5. **Persistence past the tipping angle does not need supervising.** The handoff shows the model's
   dynamics beyond ~16 deg are right. Getting the push right should be enough.
6. **Test data against objective.** The transition is rare in training (0.6% of rollouts), so part of
   the fix may be DATA rather than the objective. Add an arm with the same one-step + noise loss but with
   the new contact-window branches added as ordinary training data. Proposed arms: D0; D0 + contact-window
   data; D1; D2 (with the same data). If D0 + data recovers most blind forks, the lever is coverage; if
   only D2 does, it is the objective.

Success criterion unchanged: fewer robust-blind forks with quiet p99 and fixed-FPR recall at least as good
as D0. The primary target is the upright-start class (45 of 84 forks, mean miss rate 0.52).

## 7. Simulator reproducibility finding

`DirectJengaSim.snapshot()` saves `mjSTATE_FULLPHYSICS`, which excludes the constraint solver's warm start.
A restore also leaves kinematics consistent with the restored positions, where an uninterrupted replay
reads them one physics substep late. So restoring and then continuing a replay is never bit-identical to
not interrupting it. Stage 0 restored and continued at every probed chunk start, and its replays drifted:
its start states differ from the clean-replay start states in 182 of 214 test states (median 0.04 mm,
up to 52 mm). Consequences, checked:

* **No fork is affected materially.** Every fork drifts < 1 mm (median ~0.02 mm) and the drift is the
  same in blind and detected forks. All 13 states drifting > 1 mm are quiet states.
* **The quiet tail is not explained by it.** Dropping every drifted state leaves quiet p99 essentially
  unchanged (8.08 -> 8.33 mm).
* **The branches are physical.** From a fixed start state, the toppling set is identical under a restored,
  zeroed or carried-over warm start for all 34 analysed forks.
* **Two forks are borderline** (class C): from the clean start the model sees, only 1 of 64 probes
  topples.

The frozen benchmark evaluates models from clean-replay start states but carries labels from Stage 0's
drifted ones. For 201 of 214 states the two agree to under a millimetre; the 13 quiet states that drifted
further are documented in NOTES.md. Any future oracle generation should reach each chunk start by an
uninterrupted replay and never continue a replay after probing (as `jenga_blind_resim.py` now does).

## 8. Caveats

* 17 blind forks from about 13 situations; several are consecutive chunk starts in one episode.
* One model family (one-step + noise GNN). Other dynamics may fail differently.
* The 8 deg "upright" cutoff is analyst-chosen, and 15 deg comes from the block geometry. The threshold-free
  population result (Spearman -0.45 over 84 forks) does not depend on either.
* Everything here is Track A. It uses tilt, contact labels and a known tipping angle, none of which the
  universal monitor may use. It explains a failure; it is not a monitor feature.

## 9. Outcome (Step 7, 2026-09-22)

The D2 design in section 6 was tested against a coverage-only control, 10 paired seeds per arm
(NOTES.md, `results/jenga/step7_compare.json`). Adding the contact-window branches as ordinary
one-step data leaves the upright class mostly blind (miss rate at matched 5% FPR 0.48 -> 0.40; 8 of
the 15 robust-blind forks unchanged). Training on the same branches with a counterfactual objective
halves it (0.26 with the branch-distance loss, 0.25 with the intervention-consistency loss) and
leaves 2 upright forks robust-blind, while the other forks barely change. The mechanism in section
4 was an objective problem, not a data-coverage problem.
