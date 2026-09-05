# stable_lyap_filters

Stability-based safety filtering for robot manipulation, using the latent dynamics of a
frozen DINO world model. Task: a Franka Panda picks a block from a cluttered tabletop without
toppling its neighbours.

Two questions, two answers:

| | question | answer |
|---|---|---|
| **Detection** | can latent-divergence *detect* unsafe action chunks, zero-shot? | **Yes** — AUC 0.894 |
| **Active filtering** | can we *optimise against* that score to prevent topples? | **No** — it roughly triples the topple rate |

The headline result is the second one, and it is negative in an informative way: **a metric
that is a good passive detector is not automatically a valid control objective.** Optimising
the divergence score reliably drives the score down (−26%) while making real physics
*worse*, and the correction it produces is statistically indistinguishable from random noise
of the same magnitude (p = 0.83).

---

## The negative result

40 paired trials, expert trajectories replayed in MuJoCo with per-waypoint Gaussian position
noise (`σ = 2 mm`) — a setting where topples are real and the task actually completes. Within
each pair, both conditions get an identical noise realisation and identical initial scene.

| filter | trust | topple ctl → filtered | prevented/caused | McNemar p | correction | predicted div | pick rate |
|---|---|---|---|---|---|---|---|
| gradient (world model) | 5 mm | 15.0% → **40.0%** | 2 / **12** | **0.013** | 3.09 mm | **−26.4%** | 98% → 100% |
| random, matched magnitude | 5 mm | 12.5% → **35.0%** | 1 / **10** | **0.012** | 2.55 mm | −2.2% | 98% → 95% |
| gradient (pre-fix history) | 1 mm | 12.5% → 20.0% | 1 / 4 | 0.375 | 0.92 mm | −13.7% | 98% → 100% |

Three things make this interpretable rather than just a null:

1. **The optimiser works.** Gradient mode drives its objective down 26.4%; the random control
   leaves it flat (−2.2%). The machinery is sound — the objective is the problem.
2. **It is not "safe by doing nothing."** Pick rate holds at 100% and endpoint error stays at
   3 → 8 mm, so the filter executes the task and *still* topples more. (This guard exists
   because a frozen policy scores a perfect safety record; that failure mode has bitten this
   project before.)
3. **The gradient carries no signal.** Gradient-directed corrections caused 12 topples,
   random-direction corrections of matched magnitude caused 10 — **p = 0.83**. The harm is
   explained entirely by perturbation magnitude, not by direction.

> The honest claim is therefore *"optimising this objective does not improve real safety, and
> its gradient is indistinguishable from noise"* — **not** "the gradient actively points the
> wrong way," which the data does not support.

### Watch it

| control (unmodified) — stays safe, 11.1° | filtered (safety filter on) — topples, 90° |
|---|---|
| [`media/replay_control_safe.mp4`](media/replay_control_safe.mp4) | [`media/replay_filtered_topples.mp4`](media/replay_filtered_topples.mp4) |

Same seed, same noise, same initial scene. The only difference is the safety filter, and the
filtered run is the one that knocks the block over. *(Individual trials are not
bit-reproducible — see Caveats — so treat this as an illustration of the aggregate, not a
proof on its own.)*

[`media/world_model_imagination_ep3.mp4`](media/world_model_imagination_ep3.mp4) shows what
the filter *thinks* it is doing: real observation | world model's imagined future under the
original actions | under corrected actions | amplified difference | the two action chunks.
Both middle panels are the world model dreaming — **no physics runs in that video**, which is
precisely why it looked like the filter was working.

---

## The detection result (what does work)

Scored on 1772 chunks / 25 unsafe, 100 episodes. Thresholds are percentiles of the *safe*
distribution only, so the zero-shot claim holds.

| metric | AUC |
|---|---|
| `ftle` — original, (1/T)·log(d_end/d_start), max over patches *and* perturbations | 0.599 |
| `d_end`, p90 over patches | 0.799 |
| `d_end` + low-norm patch mask (k=30) | 0.854 |
| `d_end` + PC1 background mask (75% keep) | **0.894** |
| `ftle_variance` + PC1 background mask | **0.896** |
| *linear tilt probe on the predicted latent (uses tilt labels)* | *0.941* |

Three findings worth carrying forward, all in [`src/metrics.py`](src/metrics.py):

- **Drop the FTLE denominator.** `d_start` is measured one step in — tiny and noisy — so
  dividing by it ranks patches by how quiet they started, not how unstable they are. Worth
  ~0.2 AUC. A second max (over perturbations) makes it worse: an extremum over ~4100 values
  per chunk tracks tail noise.
- **Mask low-‖z‖ patches, not high-‖z‖ ones.** The original suspect was DINOv2's high-norm
  artifact tokens. Wrong: `corr(‖z‖, d_end) = −0.641` on ground-truth-static patches. Cosine
  distance divides by ‖z‖, so featureless patches have poorly-determined directions that
  wobble under any perturbation.
- **PC1 masking keeps the *background*, and that is why it works.** On foreground patches
  `corr(motion, d_end) = +0.45`, and it is just as strong in safe chunks (+0.452) as unsafe
  (+0.386) — most foreground divergence is a motion/phase confound from the always-moving arm.
  Background patches have ~zero baseline motion, so divergence there means something. *This
  was originally mislabelled* because the foreground/background sign came from an unverified
  norm heuristic that was backwards on both datasets tested. Always resolve the sign against
  measured patch motion.

---

## Layout

```
src/metrics.py     divergence metrics + patch masks (the detection core)
src/steering.py    classifier-guidance hook into a diffusion policy's DDIM loop,
                   and the differentiable divergence used as a control objective
eval/eval_active_filter.py   THE MAIN EXPERIMENT: noisy expert replay in MuJoCo,
                             paired control vs filtered, with the task-progress guards
eval/eval_live_policy.py     live diffusion-policy rollouts (vacuous — see Caveats)
eval/check_policy_quality.py policy sanity check / DDIM inference-step sweep
eval/analyze.py              regenerates every table above from results/*.jsonl
results/*.jsonl              raw per-trial records
media/*.mp4                  videos referenced above
docs/HANDOFF.md              state, gotchas, and next steps for whoever picks this up
```

Reproduce the tables: `python eval/analyze.py`

## Dependencies

This repo holds the safety-filter logic and evaluation. It expects two external pieces:

- **`dino_wm`** — the world model (frozen DINOv2 encoder + trained ViT predictor) and its
  checkpoint. `src/steering.py` imports its loader.
- **`panda_express`** — the MuJoCo Jenga sim (`sim.py`) and the expert teleop episodes.
  Requires `SIM_HEADLESS=1` for batch runs (EGL, no viewer); without it, importing `sim.py`
  core-dumps on a machine with no display.

Paths are currently absolute (`/home/sanger/wksp/...`) — parameterising them is the first
chore for anyone porting this. Python 3.10+ is required (DINOv2's hub code uses `X | None`
syntax), and `diffusers==0.11.1` / `huggingface_hub==0.23.0` are pinned by the diffusion
policy snapshot.

## Caveats

- **Trials are not bit-reproducible.** The MuJoCo physics thread advances on wall-clock, so
  identical seeds can diverge. Pairing controls the noise realisation and initial scene but
  not timing jitter; aggregate statistics are sound, individual trials are not.
- **The live-policy experiment is vacuous** and is included only to document why. The trained
  diffusion policy never grasps the block (gripper RMSE 0.48–0.67 on a ±1 signal), so across
  16 trials it produced 0 topples and 0 picks, peak tilt 5.97° against a 45° threshold. A
  steered-vs-unsteered table there reads 0% vs 0% and proves nothing.
- **The 1 mm row predates a history-alignment fix** (observation history was sampled once per
  chunk instead of once per step, so the world model saw frames 8 control steps apart paired
  with mismatched actions). Re-running the 5 mm gradient condition after the fix moved the
  numbers by ~2 points and changed nothing qualitatively (42.5% → 40.0% topple, p 0.004 →
  0.013), so the conclusion survives; the 1 mm row was not re-run.
- Detection AUCs are quoted from the wider research log; this repo carries the metric
  implementations and the active-filtering evidence, not the full detection sweep.

## Quick start

```bash
export DINO_WM_DIR=/path/to/dino_wm
export PANDA_EXPRESS_DIR=/path/to/panda_express
export DIFFUSION_POLICY_DIR=/path/to/diffusion_policy
export DINO_WM_CKPT=$DINO_WM_DIR/outputs/model_latest_single.pth

python eval/analyze.py                      # reproduce every table above from results/

SIM_HEADLESS=1 python eval/eval_active_filter.py \
    --n-trials 40 --episodes 1 2 3 4 5 \
    --mode gradient --max-delta 0.005 \
    --out results/my_run.jsonl              # the main experiment (~1 h)

SIM_HEADLESS=1 python eval/eval_active_filter.py --mode random_matched ...   # the control
```

`src/paths.py` resolves all external locations from those environment variables and fails
fast with a clear message if something is missing.
