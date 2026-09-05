# Handoff

For whoever (human or agent) picks this up next. Read the README first for results; this file
is state, traps, and what to do next.

## Where things actually stand

**Detection works and is validated.** `d_end` / `ftle_variance` with the PC1 background mask
reach AUC ~0.894 on 1772 chunks, held-out validated (hyperparameters chosen on one half of
the episodes, scored on the other, 20 random splits). Nothing here is in doubt.

**Active filtering does not work, and the investigation is complete.** Three 40-paired-trial
MuJoCo runs, a matched random-direction control, and a trust-region sweep all agree. The
gradient of the divergence score carries no usable control signal (p = 0.83 against random
noise of matched magnitude). This line is closed unless the objective changes — see next steps.

**Nothing is mid-flight.** All runs referenced in the README are complete and their raw
per-trial records are in `results/`.

## Traps that have already cost time here

1. **Silent shape/alignment mismatches.** `ViTPredictor` does `x + self.pos_embedding[:, :n]`
   — it *slices* to whatever token count it receives. Feeding it 2 history frames when it was
   trained on 3 raises no error and produces plausible-looking numbers. Two separate bugs of
   this shape occurred (wrong `num_hist`; observation history sampled per-chunk instead of
   per-step). **Assert on history length and on the time spacing between history frames**, and
   sanity-check that consecutive proprio entries differ by ~1–4 mm (a 10 Hz step) rather than
   ~10–25 mm.
2. **`sim.py` core-dumps on import without a display.** It builds `SimContext` at module scope,
   which calls `mujoco.viewer.launch_passive`. Set `SIM_HEADLESS=1` (EGL, viewer skipped).
   EGL then throws at interpreter teardown — harmless, it happens after the run, but it is why
   every eval script writes results to disk **per trial** rather than at the end.
3. **`num_inference_steps` must be ≤ `num_train_timesteps`.** The diffusion policy config ships
   with 100 vs 50, so DDIM's `step_ratio` floors to 0, every sampling timestep collapses to 0,
   and the sampler silently emits garbage (action MSE 0.383 — worse than predicting the mean,
   0.155). This is what made `train_action_mse_error` look flat across all of training.
4. **Trials are not bit-reproducible** (physics thread on wall-clock). Never conclude anything
   from a single trial; the paired design is over aggregates.
5. **A "safe" filter that stalls the arm scores perfectly.** Always report task progress
   (block lift, endpoint error) next to any safety number. This has produced a
   convincing-looking false positive in this project before.

## Next steps, in the order I would do them

### 1. Tilt probe as the control objective (highest value, ~2–3 h)
The natural follow-up. A linear probe on the *predicted* latent recovers future block tilt at
AUC 0.941 — a directly physical quantity, unlike latent divergence. The negative result here
is specifically about *this* objective, and the obvious question is whether a better-grounded
objective would behave differently. Minimise predicted tilt instead of predicted divergence,
same harness (`eval/eval_active_filter.py`, swap the objective inside `correct_chunk`).
Caveat: needs tilt labels from sim physics, weakening the zero-shot claim — worth stating
explicitly rather than glossing.

### 2. Fix the diffusion policy, then redo the live-policy experiment (~1 day)
Currently blocked: the policy never grasps (gripper RMSE 0.48–0.67 on a ±1 signal), so
`eval/eval_live_policy.py` is vacuous. Validation loss plateaued by epoch 20, so more epochs
of the same recipe will not help. Most likely cause is the 10× capacity cut (306M → 31M
params, `down_dims` [512,1024,2048] → [128,256,512]) forced by an 8 GB GPU. Options: lower
image resolution to afford a bigger UNet, or weight the gripper dimension in the loss (it is a
±1 discrete signal competing with mm-scale positions under plain MSE).

### 3. Only if 1 and 2 both fail
Reconsider whether *any* learned-world-model score is a viable control objective here, versus
using the monitor purely as a halting detector — which is what it is demonstrably good at.

## What I would not spend time on

- **Re-running the 1 mm trust-region condition** with the fixed history. The 5 mm rerun moved
  by ~2 points and changed nothing; the 1 mm condition was already the weakest effect.
- **Tuning the trust region further.** The sweep is monotonic: smaller corrections do less
  damage but never produce a benefit. There is no magnitude at which this objective helps.
- **The FTLE ratio family.** Thoroughly dominated (0.599 vs 0.894). The denominator is the
  problem and no amount of masking rescues it.
