# The toy experiment, explained simply

**What we were trying to find out:** does our safety monitor still work when the world model only
gets to look at *camera images*, instead of being handed the exact state of the world?

That matters because on the real Jenga task there is no exact state — only pixels. But on the real
task we also cannot check our answers, because nobody knows the true "how close to disaster was
that?" number. So we built a toy where we *can* check, swapped the world model, and looked.

**Short answer: yes.** The image-based version matched the version that reads the exact state,
using identical settings and no re-tuning.

---

## 1. The toy: a block that can tip over

A rectangular block standing on a table. You push it horizontally. It rocks on one bottom corner,
slams back down, rocks the other way. Push hard enough and it topples — and once it's down, it
stays down.

![the block through its whole range](results/phase_a/sweep.png)

*From fallen-left, through upright, to fallen-right. It pivots on a corner, like a real toppling
object.*

We picked this because it has the one property the real Jenga task has and most toy problems
don't: **failure is irreversible**. A toppled block never un-topples. That makes "did it fail?" a
genuine fork in the road rather than a temporary wobble.

### The exact dynamics

This is the classical **rocking block** of Housner (1963) [1], the standard model for exactly the
Jenga question: how hard can you push a free-standing block before it goes over?

While the block is up on one corner, it obeys a single equation:

$$
I_O \ddot{\theta} = F R \cos(\alpha - \theta) - m g R \sin(\alpha - \theta)
$$

Every symbol in it:

| symbol | what it is | value here |
|---|---|---|
| $\theta$ | **tilt angle.** $\theta = 0$ is flat on the base, positive is tilted right | *state* |
| $\omega = \dot{\theta}$ | **angular velocity** — how fast it is rocking | *state* |
| $\ddot{\theta}$ | angular acceleration — what the equation solves for | — |
| $F$ | **the push**: a horizontal force at the centre of mass | *the action* |
| $\alpha$ | **slenderness**, $\arctan(\text{half-width} / \text{half-height})$. Tall thin block = small $\alpha$ | $0.35$ rad ($20°$) |
| $m, g, R$ | mass, gravity, corner-to-centre distance | all set to $1$ |
| $I_O$ | moment of inertia about the pivoting corner | $\frac{4}{3} m R^2$ |

The two terms are a tug of war: $F R \cos(\alpha - \theta)$ is the push trying to tip it over, and
$m g R \sin(\alpha - \theta)$ is gravity trying to pull it back down. Putting in
$I_O = \frac{4}{3} m R^2$ and $m = g = R = 1$:

$$
\ddot{\theta} = \frac{3}{4} \left[ F \cos(\alpha - \theta) - g \sin(\alpha - \theta) \right]
$$

**Two guards interrupt this equation** — moments where the smooth physics stops and something
discrete happens:

| guard | when | what happens |
|---|---|---|
| **impact** | $\theta = 0$ | the block slams back onto its base and starts rocking the other way, losing speed. Housner's restitution: $\omega \rightarrow e_r \omega$ with $e_r = 1 - 1.5 \sin^2 \alpha = 0.824$ |
| **topple** | $\lvert \theta \rvert = \alpha$ | the centre of mass crosses over the pivot. Beyond this, $\sin(\alpha - \theta)$ turns negative, so the gravity term flips sign and starts *driving* the rotation instead of resisting it. The fall is committed |

Two more numbers worth knowing. The block does not move at all unless the push beats the
**critical force** $g \tan \alpha = 0.365$. And once committed, it still takes about 164
simulation steps (3.3 s) to actually hit the ground — that gap between *committed* and *fallen* is
why the monitor has to imagine far enough ahead.

### The state, for the model that gets to cheat

The whole system is described by **two numbers**, $(\theta, \omega)$ — tilt and rotation speed.
That is exactly what the state-based model is handed, and it is a *complete* description: given
$(\theta,\omega)$ and the forces, the future is fully determined.

**The image-based model gets neither.** It gets a picture. A single photo shows $\theta$ but says
nothing about $\omega$ — you cannot see speed in a still image — which is why it needs **three consecutive
frames** to work out how fast the block is rocking.

### There are exactly three ways it can end

![the three endings](results/phase_a/terminals.png)

Fallen left, still standing, fallen right. These are the **attractors** — the states the system
comes to rest in. The monitor's entire job is to notice when nearby actions lead to *different*
ones.

### We deliberately made it harder than it needed to be

Every episode gets its own lighting: different lamp direction, shadow length, brightness, block
shade.

![lighting variation](results/phase_a/lighting.png)

*All six are the **same block at the same angle**. Only the lighting changed.*

Why sabotage ourselves? Because shadows are the known weak point of the real Jenga monitor — the
world model predicts them inconsistently and that causes false alarms. Putting that problem into a
system where we *can* measure the truth is the point of having a toy.

---

## 2. The monitor, explicitly

It never asks *"will this fail?"* It asks:

> **"If I nudge this action slightly, does the world end up somewhere different?"**

If yes, you're balanced on a knife edge — small errors change the outcome, and that is worth
knowing *now*, while there's still time to slow down or ask for help.

### The algorithm

```
GIVEN:  the last 3 camera frames, and the action chunk the robot is about to execute
        a set of attractor centres C (found offline, see §5)

 1.  a  <-  the next H = 20 steps of the planned action         (fixed horizon, never shrinking)

 2.  for j = 1 ... 32:                                          <- 32 probes
         z_j ~ Normal(0, 1)                                     <- ONE random number per probe
         a_j <-  a * (1 + eps*z_j)    with eps = 0.10           <- same push, scaled up or down

 3.  append 40 steps of ZERO force to every a_j                 <- the "settle tail": stop
                                                                   pushing, let the block rest

 4.  roll each a_j through the world model from the current
     frames; keep only the FINAL latent state E_j

 5.  squash every E_j down to 8 numbers (PCA)                   <- see section 5 for why

 6.  label_j  <-  index of the nearest attractor centre in C    <- which ending did it reach?

 7.  k  <-  32 - (size of the largest label group)              <- how many probes DISAGREE

 8.  ALARM  if  k >= 2
```

**There is no threshold to tune and no training on examples of failure.** Step 8 is a *count*, not
a calibrated score. That is the property we care about most: it can be dropped onto a new task
without anyone first collecting failures to calibrate against.

Steps 2 and 7 as equations. Each probe rescales the whole push by one random factor:

$$
\tilde{a}_j = a \left( 1 + \varepsilon z_j \right), \qquad z_j \sim \mathcal{N}(0, 1)
$$

and the alarm statistic counts how many probes end up outside the most popular group:

$$
k = n - \max_c n_c
$$

where $n = 32$ probes, $\varepsilon = 0.10$ is the nudge size, and $n_c$ is how many probes landed
in ending $c$.

**Why one random number per probe, not one per timestep?** Because nudging every timestep
independently mostly cancels out — the net change is smaller by roughly √T. A single scalar scales
the *whole* push coherently, which is what actually moves you toward or away from tipping over.
This is the concentration-of-measure result of Fawzi et al. [8]: a random direction needs
$\Theta(\sqrt{d})$ times the magnitude to reach the same boundary.

**How far can it see?** Say the true *margin* is $m$ — the fraction you would have to rescale the
push by to flip the outcome. A single probe crosses that boundary only if its own random factor is
big enough, which happens with probability $p = \Phi(-m / \varepsilon)$, where $\Phi$ is the normal
CDF. So the dissent count is binomial:

$$
k \sim \text{Binomial}(n, p), \qquad p = \Phi\!\left( -\frac{m}{\varepsilon} \right)
$$

Requiring $k \ge 2$ to fire reliably works out to a **detectable margin of about
$1.33 \varepsilon$** with $n = 32$ probes, or $1.56 \varepsilon$ with $n = 50$. **The reach is a
design choice, not a mystery** — decide what margin you must catch, then solve for $\varepsilon$.
And since it grows only logarithmically in $n$, widening the nudge is far cheaper than adding more
probes.

### What it looks like when it fires

![alarm](media/fig_toy_alarm.png)

Left to right: the **camera image** is the only input. The second panel shows all 32 imagined
endings — **12 have fallen over (orange), 20 are still standing (green)**. They disagree, so the
alarm fires. Third is the same disagreement in the model's internal map; fourth is the alarm over
time.

### And when it stays quiet

![quiet](media/fig_toy_quiet.png)

All 32 imagined futures agree. Nowhere near a tipping point, no alarm.

**Watch them run:** [`results/phase_f_videos/`](results/phase_f_videos/) — nine 9-second clips.
`ep565_TP` and `ep570_TP` are catches; `ep591_FN`/`ep592_FN` are misses; `ep571_TN` is safe.

---

## 3. The two world models being compared

**The one that cheats — shPLRNN.** A *shallow Piecewise-Linear Recurrent Neural Network* [4], from
the dynamical-systems-reconstruction literature. It's small and deliberately simple:

$$
s_{t+1} = A s_t + W_1 \operatorname{relu}(W_2 s_t + h_2) + h_1 + C a_t
$$

Here $s_t$ is the model's 4-number internal state, $a_t$ is the action, and $A, W_1, W_2, C$ and
$h_1, h_2$ are what it learns.

We used it because ReLU units make the model **exactly piecewise-linear**: state space is carved
into flat-sided regions, and inside each region the map is exactly linear, so its derivatives are
exact rather than approximate. That was needed for earlier work computing stability exponents, where
a smooth network gives derivatives that are correct but useless at the perturbation sizes actually
probed. It reads the exact $(\theta, \omega)$ — the two numbers from the table above.

**The one we're testing — DINO-WM** [3]. The same architecture used on the real robot. A frozen
**DINOv2** image encoder [2] turns each 196×196 camera frame into **196 patch descriptors of 384
numbers each** — so one frame is $196 \times 384 = 75{,}264$ numbers — and a small transformer
learns to predict the *next* set of descriptors given the current ones plus the action. Nothing is ever decoded back to pixels — prediction happens entirely
in feature space.

The experiment was simply: **replace the first with the second, change nothing else.**

---

## 4. What we did, step by step

| step | question | answer |
|---|---|---|
| **A** | Can we draw the block convincingly? | **Yes** |
| **B** | Can image features recover tilt *and* rotation speed? | **Yes** — $\theta$ at $R^2\!=\!1.000$, $\omega$ at $R^2\!=\!0.955$, from 3 frames |
| **C** | Can the world model predict well enough? | **Yes — after changing how it is trained** (§6) |
| **D** | Do the imagined futures settle down? | **Yes — they hover around an equilibrium** rather than freezing (§5) |
| **E** | Can we find the three endings automatically? | **Yes — after squashing the latent down first** (§5) |
| **F** | Does the monitor work? | **Yes** (§7) |

---

## 5. The two things that surprised us

### The imagined futures hover — they don't freeze

We expected the settle tail to bring each imagined future to a dead stop. It doesn't. The latent
keeps creeping at about **0.6% of the scene's scale per step**, forever — it decays and then
flatlines instead of reaching zero.

At first we called this a failure. It isn't, because **the monitor never reads the exact position —
only which of the three endings it's nearest**. So we measured whether the label changes while the
latent hovers:

| settle steps | agrees with the label at step 40 |
|---|---|
| 10 | 95.0% |
| 25 | 99.2% |
| **40** | **100%** |
| 60 | 100% |

**It's a hover *inside* a basin, not a drift *across* basins.** The block in the real system behaves
the same way — a rocking block loses a fraction of its speed at every base impact
($\omega \leftarrow e_r\,\omega$, $e_r = 0.824$), so it approaches upright geometrically and never
exactly arrives. The model reproduced that faithfully; we were
asking the wrong question.

### Finding the three endings needed the data squashed down first

![pca](media/fig_toy_pca.png)

**Left panel:** every dot is one imagined ending, reduced to its two most important numbers.
Colour shows what *really* happened — which the monitor never sees. **The three endings form three
obvious clumps.** The handful of off-colour dots are the model's genuine mistakes (a few futures it
imagined as toppling that actually stayed up).

**Right panel:** why we had to reduce the numbers at all.

The bar measures **how far apart the groups are, divided by how wide each group is**. Above 1 means
the groups are further apart than they are wide — separable. Near 1 means they overlap into mush.

DINO-WM describes each imagined ending with **98,304 numbers** (196 patches × 384 features × …).
In that many dimensions a well-known thing happens: *everything is roughly the same distance from
everything else*. The ratio is 1.4 — technically above 1, nowhere near enough — and our clustering
step responded by declaring all 300 examples to be 300 separate groups.

Keeping only the few directions that vary the most fixes it, because **the biggest source of
variation IS which ending you reached** — falling left versus standing is a huge visual change,
while lighting is a smaller one. Drop to 8 numbers and the ratio is 2.2; drop to 2 and it's 11.6.

We only knew the clustering was at fault, rather than the world model, because we ran a **control**:
we fed the same clustering step *real photographs* of the endings instead of imagined ones. It
failed identically. That ruled out the model.

**A later correction, from trying this on real Jenga data.** The grouping method used here — merge
things that are close, and keep the count that survives the widest range of "close" — turned out to
be fragile. On Jenga, two scenes that happened to look alike (one block fully toppled, one barely
moved) acted as a bridge and welded the two groups into one. We replaced it with **HDBSCAN**, which
follows the same principle (keep what survives the widest range of scales) but works on density
rather than raw distance, and which can label an odd scene as *noise* instead of letting it bridge.
It finds the right number of groups on **both** systems — 3 on the block at 93.7%, 2 on Jenga at
98.8% — where each of the older methods worked on one and failed on the other.

**An unexpected bonus.** Clustering worked *better* on imagined endings than on real photographs.
Measured in raw distance, imagined endings sit **21% closer together within a group** while the
gaps *between* groups barely move (−2%). The world model only learns what's *predictable*, so it
quietly discards the lighting and shadow noise that real photos faithfully record. **It filters out
exactly the nuisance we deliberately added.** PCA cannot do this on its own — PCA drops whatever
*varies least*, and lighting varies a lot.

---

## 6. The standard training recipe doesn't work here

![training](media/fig_toy_training.png)

DINO-WM is normally trained to predict **one step ahead**, and it is superb at that — 0.04 radians
of error. But our monitor needs it to imagine **40 steps** ahead, feeding its own predictions back
in as input. It had never practised that, so small errors snowballed to $0.577$ rad (red) — larger
than the topple angle $\alpha = 0.35$ rad itself, which means guessing.

This is a textbook case of **exposure bias** [9,10]: train on real data, test on your own output,
and errors compound. Ross et al. [11] showed the error grows as $\mathcal{O}(T^2)$ in the horizon when you do not train
on your own induced states, against $\mathcal{O}(T)$ when you do.

The fix is to make it practise on its own predictions [5,6]. **But the order matters**: teach the
physics first with real data, *then* fine-tune on rollouts. Done in the wrong order the model finds
a lazy shortcut — predict that nothing ever happens — and gets stuck. We saw exactly that: it
predicted "still standing" for 59 of 60 test episodes.

**This matters for the real robot:** the existing Jenga world model was trained the standard
one-step way, so expect the same problem there.

---

## 7. The result

![results](media/fig_toy_results.png)

**Left is the historical nearest-centroid result, not the current algorithm.** After replacing
per-task clustering with shared HDBSCAN and excluding out-of-support endings from the dissent vote,
the 100-episode result is episode AUC 0.869 on the preferred per-chunk label, versus 0.872 for the
previous nearest-centroid run. On the old full-action label, the like-for-like comparison is 0.872
now versus 0.882 previously.

**Right records an earlier sampling correction.** The alarm signal **spikes and fades**. Checking
only 3 times per episode missed spikes entirely; checking 8 times pushed recall from 0.61 to 0.82
without changing that historical monitor. The later HDBSCAN change is a separate experiment.

The revised dissent rule fixes the false-positive regression. On actions genuinely nowhere near a
tipping point, both the previous monitor and the current HDBSCAN runtime stay below alarm 96–97%
of the time. The tradeoff is a small recall drop (0.820 → 0.800) because noise no longer contributes
to alarms. Basin discovery remains HDBSCAN, while noise is shown separately as coverage.

---

## 8. What this does and doesn't tell us

**It tells us** unsupervised HDBSCAN can discover the correct attractors in both the toy and Jenga
representations without selecting a linkage per task. With noise excluded from the dissent vote,
the runtime monitor passes the large-margin false-positive test and is close to the previous
nearest-centroid monitor on AUC. It still does not prove transfer to the full Jenga runtime.

**It does not tell us the runtime method works on Jenga.** Real-frame geometry has since passed:
HDBSCAN finds two Jenga groups at 98.8% agreement. The next test is J2/J3: roll the Jenga world
model through a settle tail, cluster its *predicted* endings, and compare one-step error with
between-basin distance. Only then is an end-to-end Jenga monitor justified.

**Honest caveats:** the headline comes from 100 episodes; the two models were compared on separate
runs rather than head to head; and the image model had to be re-trained with the better recipe, so
this validates the *approach*, not the world-model checkpoint currently on disk.

---

## References

**The physical system**
[1] Housner, G.W. (1963). *The behavior of inverted pendulum structures during earthquakes.*
Bulletin of the Seismological Society of America 53(2). — the rocking-block model, its impact
restitution $e_r = 1 - 1.5\sin^2\alpha$, and the $\lvert\theta\rvert = \alpha$ overturning
condition.

**The world models**
[2] Oquab, M. et al. (2024). *DINOv2: Learning Robust Visual Features without Supervision.* TMLR.
[3] Zhou, G. et al. (2025). *DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot
Planning.* ICML. [arXiv:2411.04983](https://arxiv.org/abs/2411.04983)
[4] Brenner, M. et al. (2022). *Tractable Dendritic RNNs for Reconstructing Nonlinear Dynamical
Systems.* ICML. — the shPLRNN.

**Training dynamics models on long rollouts**
[5] Hess, F. et al. (2023). *Generalized Teacher Forcing for Learning Chaotic Dynamics.* ICML.
[arXiv:2306.04406](https://arxiv.org/abs/2306.04406)
[6] Mikhaeil, J., Monfared, Z., Durstewitz, D. (2022). *On the difficulty of learning chaotic
dynamics with RNNs.* NeurIPS. — exploding gradients are intrinsic to chaos, not an optimiser bug.

**Basins, attractors and the monitor's statistic**
[7] Daza, A. et al. (2016). *Basin entropy: a new tool to analyze uncertainty in dynamical
systems.* Scientific Reports 6. — the entropy-over-outcomes statistic, of which our dissent count
is a simplification.
[8] Fawzi, A., Moosavi-Dezfooli, S.-M., Frossard, P. (2016). *Robustness of classifiers: from
adversarial to random noise.* NeurIPS. — why one coherent perturbation beats many independent ones.
[12] Datseris, G., Wagemakers, A. (2022). *Effortless estimation of basins of attraction.* Chaos
32(2). — automatic attractor discovery without supplying the count.

**Exposure bias (§6)**
[9] Ranzato, M. et al. (2016). *Sequence Level Training with Recurrent Neural Networks.* ICLR. —
names the problem.
[10] Bengio, S. et al. (2015). *Scheduled Sampling for Sequence Prediction with RNNs.* NeurIPS. —
the teacher-force-then-rollout curriculum we ended up needing.
[11] Ross, S., Gordon, G., Bagnell, D. (2011). *A Reduction of Imitation Learning and Structured
Prediction to No-Regret Online Learning.* AISTATS. — the O(T²) compounding result.

---

*Full detail in [`NOTES.md`](NOTES.md) (append-only log), method spec in [`MONITOR.md`](MONITOR.md),
per-phase results in [`PLAN_DINOWM.md`](PLAN_DINOWM.md), handover brief in
[`HANDOFF.md`](HANDOFF.md).*
