# The toy experiment, explained simply

**What we were trying to find out:** does our safety monitor still work when the robot's world
model only gets to look at *camera images*, instead of being handed the exact state of the world?

That matters because on the real Jenga task there is no exact state — only pixels. But on the real
task we also cannot check our answers, because nobody knows the true "how close to disaster was
that?" number. So we built a toy where we *can* check, swapped the world model, and looked.

**Short answer: yes.** The image-based version matched the version that cheats by reading the exact
state — using identical settings, no re-tuning.

---

## 1. The toy: a block that can tip over

A rectangular block standing on a table. You push it. It rocks. Push hard enough and it topples —
left or right — and once it's down, it stays down.

![the block through its whole range](results/phase_a/sweep.png)

*The block from fallen-left, through upright, to fallen-right. It pivots on a corner, exactly like
a real toppling object.*

We picked this because it has the one property the real Jenga task has and most toy problems don't:
**failure is irreversible**. A toppled block never un-topples. That makes "did it fail?" a genuine
fork in the road rather than a temporary wobble.

There are exactly **three ways it can end up**: fallen left, still standing, fallen right.

![the three endings](results/phase_a/terminals.png)

### We deliberately made it harder than it needed to be

Every episode gets its own lighting: a different lamp direction, shadow length, brightness, and
block shade. Same block, same pose — completely different picture.

![lighting variation](results/phase_a/lighting.png)

*All six of these are the **same block at the same angle**. Only the lighting changed.*

Why sabotage ourselves? Because shadows are the known weak point of the real Jenga monitor — the
world model predicts them inconsistently and that causes false alarms. Putting that same problem
into a system where we *can* measure the truth is the whole point of having a toy.

---

## 2. How the monitor works, in plain terms

The monitor never asks *"will this fail?"* It asks:

> **"If I nudge this action slightly, does the world end up somewhere different?"**

If yes, you're balanced on a knife edge — small errors change the outcome, and that's worth
knowing *now*, while there's still time to slow down or ask for help.

The recipe:

1. Take the action the robot is about to perform.
2. Make **32 slightly different versions** of it — nudged a little stronger, a little weaker.
3. Run each one through the world model, imagining the future.
4. Let each imagined future **settle** (stop pushing, let the block come to rest).
5. Look at where all 32 ended up. **If they disagree, raise the alarm.**

That's it. There is **no threshold to tune** and **no training on examples of failure**. The alarm
is just "did at least 2 of the 32 disagree with the majority?" — a count, not a calibrated number.

### What that looks like when it fires

![alarm](media/fig_toy_alarm.png)

Reading left to right: the **camera image** is the only thing the monitor gets. The second panel
shows all 32 imagined endings — **12 have fallen over (orange), 20 are still standing (green)**.
They disagree, so the alarm fires. The third panel is the same disagreement in the world model's
internal "map", and the fourth is the alarm over time.

### And when it stays quiet

![quiet](media/fig_toy_quiet.png)

Here all 32 imagined futures agree. The action is nowhere near a tipping point, so no alarm.

**Watch them run:** [`results/phase_f_videos/`](results/phase_f_videos/) — nine 9-second clips.
`ep565_TP` and `ep570_TP` are good catches; `ep591_FN` and `ep592_FN` are the misses;
`ep571_TN` is a comfortably safe one.

---

## 3. What we actually did

We already had a working monitor using a **tiny model that reads the exact tilt and speed of the
block**. The experiment was to replace that with **DINO-WM** — the same kind of image-based world
model used on the real robot — and change nothing else.

| step | question | answer |
|---|---|---|
| A | Can we draw the block convincingly? | yes |
| B | Can the image features tell us how fast it's rocking? | yes — a single photo shows the angle but not the speed, so it needs 3 frames |
| C | Can the world model predict the future accurately? | **not at first** — see below |
| D | Do the imagined futures settle down? | **no, and we still don't fully understand why it worked anyway** |
| E | Can we find the three endings automatically? | **not at first** — see below |
| F | Does the monitor work? | **yes** |

---

## 4. The result

![results](media/fig_toy_results.png)

**Left:** the image-based monitor (blue) matches the one that reads the exact state (grey), on
identical settings. It is *slightly* ahead on three of four measures, but the error bar shows the
two are close enough that we say **"at least as good"** rather than "better".

**Right:** something we got wrong and had to fix. The monitor's alarm signal **spikes and fades**.
When we only checked 3 times per episode we were missing spikes entirely, and it looked like the
monitor was failing to catch things. Checking 8 times instead pushed recall from 0.61 to 0.82. The
monitor hadn't changed at all — **we were just looking too rarely.**

One number we're pleased with: on actions that were genuinely nowhere near a tipping point, the
monitor stayed silent **96–97% of the time** across 537 checks. It isn't jumpy.

---

## 5. Two things that broke, and what they taught us

### The standard way of training the world model doesn't work here

![training](media/fig_toy_training.png)

DINO-WM is normally trained to predict **one step ahead**. It's superb at that — but our monitor
needs it to imagine *40 steps* into the future, feeding its own predictions back in. It had never
practised that, so small errors snowballed (red line).

The fix is to make it practise on its own predictions. But **the order matters**: teach it the
physics first, *then* the self-correction. Done in the wrong order, the model discovers a lazy
shortcut — predict that nothing ever happens — and gets stuck there.

**This matters for the real robot:** the existing Jenga world model was trained the standard
one-step way, so we should expect the same problem there.

### The grouping step needed the data squashed down first

![pca](media/fig_toy_pca.png)

To tell "these futures disagree" you first have to group them into the three endings. Our grouping
method worked on the small model and **completely failed** on the image model — it called all 300
examples separate groups.

The cause is that the image model describes the world with 98,304 numbers, and in that many
dimensions *everything looks equally far apart*. Squashing down to ~8 numbers first made the three
groups obvious.

We only found this because we ran a **control**: we fed the grouping step *real photographs* of the
endings instead of imagined ones. It failed identically — which proved the world model wasn't at
fault, the grouping method was.

**Unexpected bonus:** grouping worked *better* on the world model's imagined endings than on real
photographs. The world model only learns what's predictable, so it quietly discards the lighting
and shadow noise that real photos faithfully record. **It acts as a filter for exactly the nuisance
we deliberately added.**

---

## 6. What this does and doesn't tell us

**It tells us** the monitor survives the jump from perfect information to camera images without any
re-tuning, and it tells us two specific things to fix before trying the real task: the training
recipe, and the grouping step.

**It doesn't tell us the method works on Jenga.** A block on a plain table is a much easier picture
than a cluttered tabletop, and "fell flat on its face vs standing" is a much easier distinction
than "neighbour tipped 15 degrees vs not". The next step is a one-minute test on real Jenga images
that measures whether the difference between block positions is bigger than the difference between
lighting conditions. If it isn't, none of this transfers.

**Honest caveats:** the headline comes from 100 episodes; the two models were compared on separate
runs rather than head to head; and the image model had to be re-trained with the better recipe, so
this validates the *approach*, not the world model checkpoint currently sitting on disk.

---

*Full detail in [`NOTES.md`](NOTES.md) (append-only log), the method spec in
[`MONITOR.md`](MONITOR.md), the plan and per-step results in [`PLAN_DINOWM.md`](PLAN_DINOWM.md),
and a handover brief in [`HANDOFF.md`](HANDOFF.md).*
