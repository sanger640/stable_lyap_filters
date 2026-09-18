# The execution-noise fork monitor on Jenga — what works, what does not

Plain-language writeup of the 2026-09-17 line of work. Companion to
[`TOY_EXPERIMENT.md`](TOY_EXPERIMENT.md) (the toy result), [`MONITOR.md`](MONITOR.md) (the original
method spec) and [`HANDOFF.md`](HANDOFF.md) (state and commands). Every number below comes from
`results/jenga/`; the scripts that produce them are named inline.

---

## The question

A robot is about to pull a block from between two neighbours. Before it executes the next eight
commands, we want to know: **would a slightly sloppier execution of this same plan end differently?**

Not "will this fail" — that is a different and later question. Proximity to a fork is actionable
while there is still room to slow down or ask for help.

The monitor must use **no failure labels, no tuned threshold, and nothing task-specific**, because
the point is that it drops onto a new task unchanged.

## The method, end to end

```
1. take the planned 8-action chunk
2. build 64 perturbed copies: subtract 8 consecutive real tracking-error
   snippets (measured, not chosen) from the commanded xyz targets
3. run each copy, then hold the chunk's final target for 30 steps so the scene settles
4. read each ending (image latents, or physical state for the upper-bound test)
5. alarm if the 64 endings fall into separated groups that persist over time
```

There is no eps to pick. The perturbation size is a property of the robot: median 1.6 mm per step,
p90 4.0 mm, measured as the part of the command-versus-achieved error that a linear lag model does
not explain (`src/action_uncertainty.py`). Everything is reported at 0.5x, 1x and 2x that size
because the true scale is uncertain.

The sizing rule from the original method survives in a simpler form: with 64 executions and a
minimum of 2 per side, an outcome that occurs 5% of the time is detected with probability 0.84.

## How it is graded

Physics is used **only after scoring**, never by the detector. A state is a **fork** when at least
2 of the 64 executions topple a neighbour and at least 2 do not; it is **quiet** when all 64 agree
and no physical split is found at any scale. Outcomes are read after a 30-step hold, and they are
settled: the topple label at hold step 10 agrees with step 30 in 99.997% of 224,256 runs
(`eval/jenga_stage0_noise_oracle.py`).

Two conditions are always reported:

* **shared hold** — every execution ends holding the same pose, so the arm looks identical in
  every image and only the blocks differ;
* **arm moving** — every execution holds its own perturbed pose, so the arm ends 1-4.5 mm apart
  across runs. This is the harder and more realistic case.

## What the answer key showed first

Screening the development panel changed what the old cohorts meant (`stage0_noise_oracle.json`):

* safe grasps that touch a neighbour, and quiet states, **never** topple at any noise scale — so
  every alarm any earlier detector raised on them was a genuine false alarm;
* 6 of 15 "near-boundary" states never topple under realistic error; they were built with a wider
  perturbation grid;
* "moving non-topple" is not a clean negative set: at 2x, 14 of 20 have at least one toppling run.

**Names from earlier cohorts cannot be used as labels.** Everything below grades against the
measured per-scale outcome instead.

## Result 1 — on real endings, it works

Four decision rules were written, each frozen before the data it was scored on (`src/outcome_modes.py`):

| rule | what it does |
|---|---|
| **old** | project onto the single biggest direction of variation, cut into two groups |
| **strict** | allow 1-4 groups over 5 directions, require the whole grouping to match at both times |
| **revised** | same groups, require only that *some* split persists |
| **dominant** | same groups, merge to the widest-gap two-way split, require *that* to persist |

Recall on forks / false alarms on quiet states, from image latents of **real rendered endings**,
on a 107-state holdout drawn from episodes never used in development
(`holdout2_stage2b.json`):

| | old | strict | revised | dominant |
|---|---|---|---|---|
| shared hold, 1x | **88% / 1%** | 6% / 0% | 56% / 1% | 81% / 0% |
| shared hold, 2x | **84% / 2%** | 12% / 0% | 74% / 2% | 66% / 2% |
| arm moving, 1x | 40% / 2% | 35% / 0% | **75% / 3%** | 75% / 3% |
| arm moving, 2x | 67% / 2% | 23% / 2% | 79% / 8% | **73% / 2%** |

Read this as three findings:

1. **False alarms are 0-3% almost everywhere.** Every earlier detector in this project failed
   here, several alarming on 100% of quiet states. This is the first clean holdout specificity
   result the project has produced.
2. **No single rule wins both conditions.** The old rule is best when the arm converges and
   collapses when it does not; the revised rule is the reverse. The dominant rule is the best
   compromise (66-81% recall at 0-3% false alarms everywhere) and was built to be, but it still
   misses the pre-declared gate of ~85% recall under 10% false alarms in both conditions.
3. **Batch-to-batch variance exceeds the differences between rules** at these sample sizes (the
   old rule, arm moving at 1x: 78% on the first holdout batch, 40% on the second). Further rule
   tuning on single batches is not justified.

Two things the image version does **not** see: 4 mm nudges (a finger dragging a neighbour and then
releasing it) alarm in only 1-4 of 20 cases. Those splits are real — they are rendered in
`results/jenga/stage1_split_images/` — but nothing topples, and the camera barely resolves them.

## Result 2 — on predicted endings, it fails

A deployable monitor cannot render the future; it must imagine it with the world model. Stage 3
changes exactly that one thing (`eval/jenga_stage3_predicted_forks.py`): the three real frames at
the chunk start warm-start the model, each perturbed action window is rolled out 8 chunk steps plus
the 30-step hold, and the same frozen rules read the predicted latents.

**Alarm rates on forks and on quiet states become equal within noise.** There is no signal at
all, in any condition, with either checkpoint (recall / false alarms, the three rules of Result 1):

| | old | revised | dominant |
|---|---|---|---|
| shipped, shared hold 1x | 0% / 8% | 12% / 8% | 6% / 7% |
| shipped, arm moving 1x | 0% / 0% | 0% / 6% | 0% / 5% |
| fine-tuned, shared hold 1x | 0% / 7% | 0% / 3% | 0% / 3% |
| fine-tuned, arm moving 1x | 0% / 0% | 0% / 0% | 0% / 0% |
| fine-tuned, arm moving 2x | 4% / 2% | 2% / 4% | 2% / 4% |

against 40-88% recall at 0-3% false alarms on the same states from real endings.

The cause is measurable and is not the detector. In each state's own latent space, the separation
between toppling and non-toppling executions:

| endings | 1x | 2x |
|---|---|---|
| real | 4.85 | 3.11 |
| predicted, shipped checkpoint | 0.60 | 0.55 |
| predicted, after rollout fine-tuning | 0.68 | 0.83 |

The model predicts nearly the same ending whether or not a block topples.

## Result 3 — the obvious fix does not close the gap

The shipped checkpoint was trained to predict one step ahead, and on the toy system that
compressed distinct outcomes toward the middle attractor until rollout fine-tuning fixed it. So
the same fix was applied here, with the missing ingredient regenerated: the recorded demos contain
**no held poses at all**, so 1,762 sequences of "perturbed chunk plus 30-step hold" were rendered
and encoded from the 43 development episodes (`eval/jenga_gtf_data.py`), and the predictor alone
was trained on its own 38-step rollout (`eval/jenga_gtf_train.py`).

Rollout prediction improved by an order of magnitude — validation latent MSE 2.095 to 0.207 — and
the separation moved from 0.60 to 0.68. Detection stayed at zero.

**Why this is the expected outcome in hindsight.** The loss rewards predicting the *average*
future. For a chunk that topples in 11 of 64 executions, that average is "mostly standing", and a
toppling neighbour occupies a small part of the frame, so blurring it costs almost nothing in MSE.
On the toy system the outcome dominated the image; here it does not.

One recipe was tried. Longer training, more data, or unfreezing more of the model are untested.
But a 10x drop in loss moving the quantity that matters by 13% is the wrong shape for a fixable
optimisation problem.

## Where this leaves the method

**Validated:** given accurate endings, a label-free, threshold-free test detects that a chunk sits
on an outcome fork, with 0-3% false alarms on held-out states and no task-specific input. That
covers the setting where a simulator or an accurate rollout is available.

**Not validated:** the same test from camera images through this world model. That path is blocked
on a predictor that does not collapse distinct futures, which is a research problem, not a
configuration change.

**Not claimed anywhere:** that this is a safety filter. It has never run in a live loop, it has
only ever seen this Jenga scene, and its blind spot is declared — an action that fails with
certainty produces no disagreement and therefore no alarm.

## What the next person should do

1. **Do not tune a fifth decision rule.** Four were tried, three frozen in advance; batch variance
   already exceeds the gaps between them.
2. **Attack the predictor's averaging**, with a loss that penalises collapsing distinct futures:
   sample-based or distributional prediction, or training directly against the separation
   diagnostic above (which is label-free — it needs only the spread of predicted endings, not
   topple labels).
3. **Re-run Stage 3 unchanged** after any predictor change. It is the gate, it takes ~55 minutes,
   and the d' diagnostic tells you the answer before the alarm rates do.
4. **Keep physics for grading only.** Every collapse in this project came from letting a
   task-specific quantity into the detector.

## Reproducing

```bash
PY=.venv/bin/python
$PY eval/jenga_stage0_noise_oracle.py     # answer key, 55 development states (~2.5 min)
$PY eval/jenga_stage1_outcome_modes.py    # physics upper bound
$PY eval/jenga_stage2_visual_forks.py     # image latents of real endings (~9 min)
$PY eval/jenga_stage2_visual_forks.py --own-hold   # arm-movement condition
$PY eval/jenga_stage2b_multimode.py       # all four rules on cached latents
$PY eval/jenga_stage3_predicted_forks.py --stage2 <stage2 json> --cache <npz> --output <json>
```

The holdout end to end (screen 1,168 chunks, select, render, score) and the fine-tuning pipeline
are documented in `HANDOFF.md` section 3; the append-only detail, including every failed variant
and why it failed, is at the end of `NOTES.md`.
