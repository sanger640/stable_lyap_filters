# The runtime monitor — final recipe

A **calibration-free** proximity-and-failure monitor for an action chunk. No tuned threshold: the
alarm is a *counting* statement about disagreement among probes, so nothing plays the role that
δ=0.8 plays in the FTLE monitor.

The question it answers is not "will this fail?" but **"is this action near a boundary where the
outcome changes?"** — plus, for free, "has it already committed to failing?"

---

## Offline, once: discover the attractors

You need the terminal states of the system in latent space. You do **not** supply how many.

```
1. Roll M actions (M ~ 200) through the world model for H steps, then append a
   SETTLE TAIL of L steps of zero/hold action, so each rollout comes to rest.
2. Collect the ENDING latents E_j (after the settle tail, not after H).
3. Sweep a merge distance d over fractions of the ending scale
   (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8) x scale(E).
   Single-linkage merge at each d, count the clusters.
4. The attractor count is the PLATEAU -- the count that survives the widest range of d.
   Take the cluster centroids as the attractor centres C.
```

Why the plateau and not silhouette or the gap statistic: both of those were tried on the tipping
block and both pick 5-6. The plateau picks 3, which is correct (fall-left, upright, fall-right).
Datseris & Wagemakers (Chaos 2022) use recurrence for the same job; the merge-distance plateau is
simpler and needs no basin sampling.

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
5. ASSIGN          lab_j <- argmin_c || E_j - C_c ||^2
6. COUNT           k <- n - max_c #{j : lab_j = c}          # dissent count
                   S <- -sum_c p_c ln p_c                   # basin entropy (a summary of k)

7. ALARM if   lab(majority) is a FAILURE attractor     <- clause 1: already committed
           or k >= k_min                               <- clause 2: near a boundary
```

**Two clauses, and they cover different things.** Clause 2 is blind by construction to actions that
fail with certainty: if every probe agrees the block falls, k=0 and S=0. Clause 1 catches exactly
those — it is free, because step 5 already labels the nominal rollout. On the 100-episode tipping
block eval, 28 episodes topple while far from the boundary and 16 draw no alarm from clause 2
alone; those are clause 1's job. **Clause 1 is specified here but has not yet been measured on the
100 episodes — that is the next run.**

**Report k, not S.** S is a scalar summary; k is the statistic that carries the sampling
properties, and the sizing rule below is written in k.

**Score continuously, and do not latch.** Scoring only at t=0 is a gate, and the gate is the wrong
device: at H=200 it scores precision 0.824 / recall 0.333 while continuous scoring gets 2.7x the
recall. Latching hides recovery and makes every later frame unreadable.

---

## Choosing the probe direction v

The algebra that matters:  `a + c*a == a*(1 + c)`.  So "multiplicative shared" is just "additive
shared along the direction v = a". There is only ONE mechanism -- perturb along a chosen direction
with a single scalar -- and the only question is which v.

| system | v | why |
|---|---|---|
| tipping block | `v = a` (i.e. scale the push) | action is a force; scaling is meaningful |
| Jenga | `v = diff(a)` (the EE *displacement* sequence) | action is an ABSOLUTE EE position, so scaling is origin-dependent and geometrically meaningless; the displacement sequence is origin-free |

**One scalar per probe, never one per timestep.** Per-step isotropic noise changes the chunk's
*shape* while barely changing its *size*: T independent nudges largely cancel, so the net moves by
only ~eps/sqrt(T) even though every timestep moved by eps. Fawzi et al. (NeurIPS 2016, Thm 1): a
random direction needs Theta(sqrt(d)) times the magnitude to reach the same boundary, because its
expected squared overlap with the boundary normal is 1/d. At T=450 that is a factor of ~21. The
per-step family is what the current deviator agent does, and it is the wrong family.

---

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

For Jenga at eps=0.005 (~the positional error of the EE):

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
| k_min | 2 | inclusive: k=2 IS an alarm |
| v | `a` | scale the push |
| attractors | 3 | fall-left, upright, fall-right, from the plateau |

Caveat on L: 150 is shorter than the 164-step fall and was chosen empirically; the fall completes
because the settle tail starts from an already-committed state. Worth re-checking if H changes.

Undersampling is safe and slightly helps: coarsening dt from 450 to 45 steps moved AUC 0.934 ->
0.971, so the monitor does not need the fine time grid.
