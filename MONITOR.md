> **Current plan: [`PLAN_NEXT.md`](PLAN_NEXT.md) (2026-09-21).** The monitor construction below
> (execution-noise probes, common-target hold, spread of endings) is still the one in use.

> Plain-language version with figures: [`TOY_EXPERIMENT.md`](TOY_EXPERIMENT.md). Current status and next steps: [`HANDOFF.md`](HANDOFF.md).

> **Which variant is current (2026-09-17).** Everything below describes the ORIGINAL method:
> Gaussian probes that scale the chunk's own motion, one global attractor atlas discovered
> offline, and dissent counted against that atlas. It is validated on the toy system and it
> FAILED on Jenga. The Jenga line of work now uses a different construction, kept deliberately
> close in spirit:
>
> * **probes** = 64 contiguous 8-step tracking-residual snippets from held-out episodes,
>   subtracted from the commanded targets (a measured execution-error scale, no `eps`), at
>   0.5x/1x/2x; see `src/action_uncertainty.py` and `eval/jenga_stage0_noise_oracle.py`;
> * **endings** = the settled scene after a 30-step hold on a common target, read as full-frame
>   DINO latents (`eval/jenga_stage2_visual_forks.py`);
> * **alarm** = the 64 endings split into separated groups, tested PER STATE with no global
>   atlas and no supplied count, and the split must persist between hold steps 10 and 30
>   (`src/outcome_modes.py`).
>
> The sizing rule survives in the form "at least 2 of 64 runs in the minority", i.e. a minority
> outcome of 5% is detected with probability .84. Holdout numbers and the open problem (no single
> rule works both when the arm converges and when it does not) are in `HANDOFF.md` and `NOTES.md`.

A **calibration-free** monitor for an action chunk. No tuned threshold: the alarm is a *counting*
statement about disagreement among probes, so nothing plays the role that δ=0.8 plays in the FTLE
monitor. No labels of any kind, so it drops onto a new manipulation task as-is.

## What it claims, and what it does not

**It detects that the policy is operating where small errors change the outcome.** Not that a
failure is coming. The signal is that a *nearby alternative action has a different terminal state*.

That is the useful thing to know, and the timing is the reason: proximity is actionable while there
is still room to slow down, re-plan, or ask for help. Failure prediction fires when it is already
too late to act. And confidently-catastrophic actions are the easy case that any crude check
catches -- on the tipping block, net impulse alone scores AUC 0.813 on outcome, beating oracle
divergence at every horizon. The failures that actually bite a demonstration-trained policy come
from OOD drift where the actions are **marginal**. That is the regime this targets.

Three consequences, and they are not negotiable once the claim is fixed:

1. **Evaluate on proximity (margin < m), never on outcome.** On outcome the method scores AUC
   0.29-0.56 and loses to a one-line action statistic. On proximity it scores 0.856 with no
   calibration against div_std's 0.818, which needs a percentile fitted to safe data.
2. **The blind spot is declared scope, not a defect.** Dissent counting is blind by construction to
   certain failure: if every probe agrees the block falls, k=0. 28 of 100 tipping-block episodes
   topple while far from any boundary, 16 silently. Report it; do not patch it.
3. **Any alarm clause that needs a basin named "failure" is supervision, and is struck.** See
   clause 1' below for the label-free substitute and its own limitation.

---

## Offline, once: discover the attractors

You need the terminal states of the system in latent space. You do **not** supply how many.

```
1. Roll M actions (M ~ 200-300) through the world model for H steps, then append a
   SETTLE TAIL of L steps of zero/hold action, so each rollout comes to rest.
2. Collect the ENDING latents E_j (after the settle tail, not after H).
   *** Use the PREDICTED endings, never encoded observations. See below. ***
3. *** PCA the endings to ~2-16 dimensions. NOT OPTIONAL in a patch-token space. ***
4. *** HDBSCAN with min_cluster_size = 5-10% of M. Keeps the clusters that PERSIST over the
   widest range of DENSITY levels, and labels sparse points as NOISE. ***
5. The attractor count is whatever HDBSCAN returns. Nothing is supplied.
6. Take the cluster centroids as the attractor centres C. Record each cluster's density support
   as the 99th percentile of within-cluster nearest-neighbour distance. At runtime, use the
   nearest centroid only when the ending lies inside that cluster's support; otherwise retain
   the NOISE label. Noise is excluded from DISSENT and reported as reduced coverage.
```

### Why HDBSCAN and not the merge-distance plateau

The original recipe used a single-linkage merge-distance sweep and took the count that survived the
widest range of d. **That is not robust, and two systems broke it in opposite directions:**

| method | toy (3 basins, 212/50/38) | Jenga (2 basins, 75/25) |
|---|---|---|
| single-linkage + plateau | works (93.3%) | **fails** -- one bridging pair welds the basins |
| Ward + merge-height gap | **fails** (k=1, 70.7%) | works (98.0%) |
| **HDBSCAN (5-10% of M)** | **works (93.7%)** | **works (98.8%)** |

*Single-linkage* merges on the NEAREST pair, so one coincidental close pair chains two clumps into
one. On Jenga, ep16 (block barely moved, 7 deg) and ep45 (fully over, 94.5 deg) sit 0.120*scale
apart against a median within-group nearest-neighbour distance of 0.115 -- a ratio of 1.04.

*Ward* merges to minimise the increase in within-cluster VARIANCE, which costs more for large
clusters, so it biases toward EQUAL-SIZED groups. At 75/25 that is fine; at 212/50/38 it splits the
majority instead of isolating the minorities.

*HDBSCAN* keeps clusters that persist across density levels -- the same idea as the plateau, built
on density instead of raw distance -- and crucially it has a NOISE label, so a bridging point is
discarded rather than used as a bridge. It discarded 17% of Jenga episodes as noise, which is the
intended behaviour and not a defect.

**Runtime result from the 100-episode toy rerun.** Discovery and online assignment are separate
tests. HDBSCAN finds the correct three toy basins at 93.7% agreement and 100% training coverage.
Excluding support noise from the basin vote restores the large-margin floor: 96-97% of far chunks
stay below alarm. Preferred chunk-label episode AUC is 0.869; applying the old full-action label
gives AUC 0.872. Mean known-basin coverage is 92.8% (median 100%); 29/800 chunks are all-noise
and therefore have k=0 by definition. Noise remains visible as reduced coverage, not as basin
dissent, and zero-coverage chunks should be treated as no-evidence rather than as a safety claim.

`min_cluster_size` is the only knob and it is a structural statement, not a fitted one: *a basin
must hold at least this share of the episodes to count as a basin.* 5% and 10% both work on both
systems, so it is not knife-edge. Setting it ABOVE the smallest true basin correctly returns
nothing rather than inventing structure (15% asks for 45 points on the toy, whose smallest basin
has 38).

**Two systems is not proof of universality.** But it removes the per-task algorithm choice, which
was the thing that made the calibration-free claim shaky.

### Steps 3 and 6 were added after the DINO-WM run and are not optional

**PCA first.** Single-linkage relies on LOCAL structure (nearest-neighbour chains), which
concentrates in high dimensions. In the raw 98,304-dim DINOv2 patch space the within/between
separation was **1.09-1.35** -- no scale gap for a plateau to sit in -- and the sweep returned
k = 300 out of 300 at every d. PCA to 2-16 dims raises separation to **2.2-11.6** and the plateau
appears immediately. The dimension is not critical (8 and 16 both worked); the raw space does not
work at all. This was diagnosed with a CONTROL: running the same discovery on encoded TRUE endings
failed identically, which ruled out the model and indicted the procedure.

**Minimum cluster size.** One stray point out of 300 never merges and turns k=3 into k=4.

**Cluster PREDICTIONS, never encodings.** Predicted endings separate better than encoded ones at
every measurement -- and in PCA space the encoded-truth control shows NO plateau at all while
predictions plateau cleanly. Measured in absolute distance (same space, so comparable):

| | within-basin | between-basin |
|---|---|---|
| encoded truth | 559 | 610 |
| predicted | **441** (-21%) | 597 (-2%) |

The clusters get TIGHTER without getting CLOSER. That is denoising, not collapse. The predictor is
trained to model what is PREDICTABLE, so it replaces unpredictable components -- render noise,
exact shadow pixels, this episode's lighting -- with their conditional mean, which is roughly
constant across episodes. Pose survives because pose follows from the dynamics.

**PCA cannot substitute for this.** PCA selects by VARIANCE; the world model selects by
PREDICTABILITY. Lighting is high-variance (63% of a full pose change), so PCA keeps it. The
predictor discards it anyway. The two are complementary, which is why PCA'ing the encoded control
still produced no plateau.

This is convenient rather than awkward: the monitor only ever has predictions, since it is
predicting a future for which no observation exists.

Why the plateau and not silhouette or the gap statistic: both of those were tried on the tipping
block and both pick 5-6. The plateau picks 3, which is correct (fall-left, upright, fall-right).
Datseris & Wagemakers (Chaos 2022) use recurrence for the same job; the merge-distance plateau is
simpler and needs no basin sampling.

**"Settled" means the LABEL stops changing, not that motion stops.** The latent will generally keep
creeping -- on the tipping block it decays to ~0.6% of scale per step and then flatlines forever,
never reaching a fixed point. That is not a defect and it is not worth engineering away: the monitor
reads only which attractor is nearest, and the assignment is stable long before the motion is
(measured: 99.2% stable by settle step 25, 100% from step 40). It is also physically honest, since a
damped oscillator approaches its rest state geometrically without arriving. **Size the settle tail by
when the assignment stops changing; requiring `||dz|| -> 0` measures a quantity this method never
uses.**

**The settle tail is essential and is NOT the lookahead.** Without it, endings are mid-flight
states and the "attractors" are just wherever the rollout happened to be at step H. On the tipping
block the failure commits at ~step 133 but takes 164 steps to actually fall: measuring at H=200
put toppled and surviving endpoints in overlapping ranges (0.432-0.525 vs 0.465-0.498) and cost a
whole section of results.

---

## Online, every step: score the chunk

```
INPUT   current observation o_t, action chunk a from the policy
CONST   H (fixed lookahead), L (settle), n (probes), eps, k_min, C (attractor centres)

1. TRUNCATE        a <- a[:H]                       # FIXED horizon, never receding
2. PROBE           for j in 1..n:
                       z_j ~ N(0, 1)                # ONE SCALAR per probe
                       a_j <- a + eps * z_j * v     # v = probe direction (see below)
3. SETTLE          a_j <- concat(a_j, zeros(L))
4. ROLL            z_0 <- lift(encode(o_t))
                   E_j <- world_model.rollout(z_0, a_j)[-1]
5. ASSIGN          lab_j <- nearest supported basin, or NOISE
6. COUNT           known <- {j : lab_j != NOISE}
                   k <- |known| - max_c #{j in known : lab_j = c}  # known-basin dissent
                   coverage <- |known| / n                        # report separately
                   S <- -sum_c p_c ln p_c                         # entropy over known basins

7. ALARM if   k >= k_min                               <- near a boundary. THE METHOD.
   (optional)   lab(nominal) != lab(null action)       <- clause 1': the action CHANGES
                                                          the terminal state
```

**The method is clause 2 alone: k >= k_min.** Nothing else. It requires no labels of any kind --
not failure labels, not attractor names, not a tuned threshold -- which is the entire point, since
a monitor that needs to be told what failure looks like cannot be dropped onto a new manipulation
task.

**The price is declared scope: this is a PROXIMITY monitor, not a failure detector.** Dissent
counting is blind by construction to actions that fail with certainty -- if every probe agrees the
block falls, k=0 and S=0. On the 100-episode tipping-block eval, 28 episodes topple while far from
any boundary and 16 draw no alarm. Those are OUT OF SCOPE, not errors. Evaluate on proximity and
report the blind spot as a limitation; do not evaluate on outcome, where a one-line action
statistic (net impulse, AUC 0.813) beats oracle divergence at every horizon anyway.

The argument for why proximity is the signal worth having: it says a nearby alternative action has a
DIFFERENT outcome, i.e. the policy is operating where small errors matter. That is actionable while
there is still time to slow down, re-plan or ask for help. Failure prediction fires too late, and
confidently-catastrophic actions are the easy case that any crude check catches. The failures that
actually bite a demo-trained policy come from OOD drift where actions are marginal.

**Clause 1' is optional and was NOT part of the original design.** An earlier draft of this file had
a clause "alarm if the majority attractor is a FAILURE attractor". That is supervision -- it needs
someone to say which basin means toppled -- so it is struck. The label-free replacement compares the
nominal rollout's ending basin against a NULL-ACTION reference rolled from the same state: if doing
nothing settles upright and executing settles fall-right, the action changed the terminal state.
That needs no labels, only the unsupervised attractor set and one extra rollout, and it recovers the
16 silent topples.

Its limitation is real: it also flags INTENDED terminal-state changes. On Jenga a successful grasp
changes the scene, so every good pick trips it. The fix is not failure labels but the expert demos
already used to train the policy -- the safe terminal basins are the ones the demos reach. On the
toy there is no intended change, so the clause is clean there and misleading about how clean it
would be elsewhere. Treat it as an extension, not part of the core claim.

**Report k, not S.** S is a scalar summary; k is the statistic that carries the sampling
properties, and the sizing rule below is written in k.

**Score continuously, and do not latch.** Scoring only at t=0 is a gate, and the gate is the wrong
device: at H=200 it scores precision 0.824 / recall 0.333 while continuous scoring gets 2.7x the
recall. Latching hides recovery and makes every later frame unreadable.

---

## What eps, z and v actually are

```
a_j  =  a  +  eps * z_j * v
       ^      ^     ^      ^
       |      |     |      the DIRECTION in action space to perturb along
       |      |     ONE SCALAR per probe, z_j ~ N(0,1). Not one per timestep,
       |      |     not one per action dimension. 32 probes = 32 numbers.
       |      the SIZE of one standard deviation, in whatever units v carries
       the nominal chunk from the policy
```

`z_j` says how many eps this probe moves; `eps` says how far one eps is. With eps=0.10 and
z_7=1.82, probe 7 sits 18% out along v.

### Choosing v

The algebra that matters:  `a + c*a == a*(1 + c)`.  So "multiplicative shared" is just "additive
shared along the direction v = a". There is only ONE mechanism -- perturb along a chosen direction
with a single scalar -- and the only question is which v.

| system | v | why |
|---|---|---|
| tipping block | `a` | the action is a FORCE, so doubling it means something |
| Jenga | `a - a[0]` | the action is an ABSOLUTE EE POSITION (see below) |

**Jenga cannot use `v = a`.** If the gripper sits at x=0.457 m, `a * 1.1` sends the arm to x=0.503
-- that is not "push 10% harder", it is "go somewhere else", and the answer changes if you move the
world origin. `v = a - a[0]` involves only differences, so it is origin-free; it means *travel 10%
further along the path you were already going to travel*:

```python
v   = a - a[0]                      # displacement from the chunk's start, shape (T, 4)
v[:, 3] = 0.0                       # gripper is BINARY -- scaling it is meaningless
a_j = a + eps * z_j * v             # equivalently: a[0] + (1 + eps*z_j) * (a - a[0])
```

The start is pinned and the reach is stretched or shortened. A 30 mm reach in x at eps=0.10 becomes
35.5 mm at z=+1.82 and 23.7 mm at z=-2.10. That is the axis the topple boundary lives on.

(An earlier draft wrote `v = diff(a)`. That is the right mechanism but the wrong shape -- diff(a) is
(T-1, 4) and will not broadcast. `a - a[0]` is the integrated form of the same thing.)

### One scalar per probe, never one per timestep

Per-step isotropic noise changes the chunk's *shape* while barely changing its *size*: T
independent nudges largely cancel, so the net moves by only ~eps/sqrt(T) even though every timestep
moved by eps. Fawzi et al. (NeurIPS 2016, Thm 1): a random direction needs Theta(sqrt(d)) times the
magnitude to reach the same boundary, because its expected squared overlap with the boundary normal
is 1/d. At T=450 that is a factor of ~21.

The current Jenga deviator agent (eps=0.005 added independently per timestep) is exactly this wrong
family.

### Units: pick one convention and stay in it

`eps` is dimensionless or metres depending on whether v is raw or normalised. Both work; mixing
them is how the "7.8 mm" figure below got stated without its precondition.

| convention | v | eps | detectable margin m* (n=50) |
|---|---|---|---|
| **fractional** (validated on the block) | `a - a[0]` | dimensionless | 1.56*eps as a FRACTION of the chunk's motion |
| **absolute** | `(a - a[0]) / \|\|a - a[0]\|\|` | metres | 1.56*eps in metres |

Fractional with eps=0.10 on a 3 cm chunk gives m* ~ 4.7 mm, and scales with how far the chunk
travels -- usually what you want. Absolute with eps=0.005 m gives a flat 7.8 mm.

## Sizing: eps, n and k_min from the margin you want to detect

This is the part that replaces threshold tuning. Let **m** be the margin: the fractional change in
the action that flips the outcome. A single probe crosses the boundary with probability
`p = Phi(-m/eps)`, so the dissent count is `k ~ Binomial(n, p)` and

```
E[k] = n * Phi(-m / eps)
```

Largest margin still caught with probability >= 0.80, for k_min=2:

| n | detectable margin m* |
|---|---|
| 16 | 0.93 eps |
| 32 | **1.33 eps** |
| 50 | 1.56 eps |
| 100 | 1.88 eps |
| 200 | 2.17 eps |
| 400 | 2.43 eps |

**Read this the right way.** The most extreme of n draws reaches ~2.3 eps, but the k>=2 RULE only
reaches ~1.33 eps at n=32, because two crossings are much harder than one. Sizing on the
single-draw reach overstates the monitor's range by nearly 2x -- it is what produced three of the
four false negatives in the 100-episode eval (margins 18.7-22.5%, where P(alarm) is 0.05-0.26) and
they are a binomial floor, not a method failure.

**So: pick the margin you must detect, then set `eps >= m* / 1.33` (n=32) or buy reach with n.**
Reach grows only logarithmically in n -- doubling probes from 50 to 100 buys 20% more reach -- so
raising eps is far cheaper than raising n.

For Jenga in the ABSOLUTE convention at eps=0.005 m (~the positional error of the EE),
with v normalised to unit length:

| n | detectable margin |
|---|---|
| 32 | 6.7 mm of EE displacement |
| 50 | **7.8 mm** |
| 100 | 9.4 mm |
| 200 | 10.8 mm |

If the Jenga margins of interest are tighter than ~8 mm, n=50 at eps=0.005 is already adequate. If
they are 2 cm, eps must roughly double; adding probes will not get there.

`k_min=2` rather than 1 because one dissenter out of 32 is a 3% rate, which is within sampling
noise of an ensemble that all lands in one basin. Do not raise k_min to buy precision without
re-deriving m* -- k_min=3 at n=32 drops the reach to ~1.1 eps.

---

## Labelling, for anyone evaluating this

Two rules, both learned the hard way:

1. **Label at the rule's reach, not the probe's reach** -- 1.33 eps for k>=2 at n=32, not 2.3 eps.
2. **Label per (state, chunk), not per episode.** The monitor at step t asks "from the state I am
   in now, does the next H-step chunk straddle a boundary?" A whole-episode margin measured at t=0
   answers a different question, and they diverge badly late in an episode: on the tipping block,
   alarms at t=180-315 came from a block already tilted, where the remaining chunk is genuinely
   marginal even though scaling the full action by +-50% changes nothing. 12 of 22 "false
   positives" in the 100-episode eval actually toppled.

Every measurement error in this project had the same shape: **an evaluation protocol that could
not see what the method produces.** Horizon too short for the failure to express itself; margin
measured along a direction the probes never sample; label from the full action while the monitor
saw one chunk; eps mismatched to reach; scoring at a single instant; label at the single-draw reach
under a two-draw rule. The tell was the same every time -- an oracle and a proxy disagreeing far
more than the method's quality could explain.

---

## Settings used for the tipping block

| param | value | note |
|---|---|---|
| H (lookahead) | 200 | fixed; ~4.0 s at dt=0.02 |
| L (settle) | 150 | must exceed the 164-step fall time... see caveat below |
| rollout total | 350 | H + L |
| n (probes) | 32 | reach 1.33 eps = 13% margin |
| eps | 0.10 | 1-sigma of the scalar |
| k_min | 2 | inclusive: k=2 known-basin dissent IS an alarm |
| v | `a` | scale the push |
| attractors | 3 | fall-left, upright, fall-right, discovered by HDBSCAN |

Caveat on L: 150 is shorter than the 164-step fall and was chosen empirically; the fall completes
because the settle tail starts from an already-committed state. Worth re-checking if H changes.

Undersampling is safe and slightly helps: coarsening dt from 450 to 45 steps moved AUC 0.934 ->
0.971, so the monitor does not need the fine time grid.
