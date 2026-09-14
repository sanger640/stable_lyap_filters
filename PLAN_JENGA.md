# PLAN — the monitor on Jenga, offline then live

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

## J3 — Do PREDICTED endings cluster like REAL ones?

The headline result so far used real final frames. The monitor reads predictions. Run the identical
pipeline -- regress proprio out, PCA, HDBSCAN at min_cluster_size 5-10% of n -- on predicted
endings.

**Acceptance**
- HDBSCAN finds **k=2**, agreeing with the topple label at **>= 90%** (real frames: 98.8%).
- Separation ratio **>= 1.5** (real frames: 2.15).
- If it fails, J4 says why.

## J4 — The hedging check (one line, no retraining)

Compare **predicted basin COUNTS** against the true 75/25 split. MSE-trained models predict the
conditional mean, which near a bifurcation sits BETWEEN the basins, so predictions compress toward
"nothing happened". On the toy, dino_wm's shipped single-step recipe gave [11,43,6] against a true
[15,28,17] and only 75% basin agreement.

**The Jenga checkpoint uses that same `num_pred: 1` recipe, so expect this.**

**If counts are compressed:** rollout fine-tune. ~19 min on the toy; more here, with gradient
checkpointing. **Order matters** -- teacher-force first, THEN rollouts. Reversed, the model collapses
to predicting that nothing ever happens ([1,59,0] on the toy).

## J5 — Offline monitor on replayed episodes

Now the monitor proper. Per chunk: 50 probes along `v = a - a[0]` (the action is an ABSOLUTE EE
position, so scaling it is origin-dependent and meaningless -- perturb the DISPLACEMENT), settle
tail from J2, HDBSCAN centroids from J3, `k >= 2` alarm. Probes landing in NOISE count as dissent.

**Run the FALSE-POSITIVE FLOOR first**: on chunks far from any boundary, `k` should be 0 on >= 95%.
If not, the alarm is measuring the model rather than the scene, and nothing downstream is
meaningful.

**Score against `peak_tilt_deg`, not the binary outcome.** It is a continuous near-miss measure and
therefore much closer to the PROXIMITY quantity this method predicts. The binary flag is the wrong
target and the toy work repeatedly showed what that costs.

**Sizing:** `eps` follows from the margin to be detected -- `m* = 1.56*eps` at n=50. Pick the margin
first. Note the current Jenga deviator uses per-timestep noise at eps=0.005, which is the WRONG
probe family: independent per-step nudges cancel, costing a factor of ~sqrt(T).

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
