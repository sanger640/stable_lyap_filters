# Deep dive: why the filter's fix adds little over the trigger, and the rebuild (Jenga)

User mandate 2026-10-22 night: keep improving the monitor and filter on Jenga until it converges. Constraints:
- the filter is an optimisation (CBF / HJ-inspired) that edits the policy's actions, with no hand-made strategies;
- the world model is off-the-shelf and reliable, working in latent space;
- the monitor uses sensitivity of the dynamics to actions;
- both monitor and filter stay task-agnostic.

## 1. Evidence so far (2026-10-22; all exploratory unless marked)

### 1a. The check spots risky states, but ranks candidate chunks badly
- **Relative-pass test (pre-registered, docs/JENGA_RESIDUAL.md): H1 FAIL.** At 403 decisions where the policy's plan was
  forked and holding was forked too, real one-chunk neighbour risk was:
  - policy 5.1%, hold 6.8%, G-CEM with the absolute rule 7.7%.
- Split by G-CEM's outcome:

  | G-CEM outcome | n | policy | G-CEM |
  |---|---|---|---|
  | certified, chunk essentially unchanged | 84 | 0.1% | 0.1% |
  | certified and actually changed (median 5.5 mm) | 136 | 4.3% | 5.7% |
  | no candidate certified (fell back to best) | 168 | 8.5% | 13.8% |

- **The check is a good detector of risky STATES** (these 403 have 5.1% risk vs 0.3% for decisions that pass outright),
  **but a poor ranker of nearby CHUNKS**: a modified chunk the check certifies is, on average, riskier than the
  policy's own.
- This matches benchmark v2 (dev), real one-chunk risk:
  - policy 2.83%;
  - free mm-offset optimiser 5.73% (CEM) / 7.17% (gradient), worse than doing nothing;
  - G-CEM 0.67%;
  - plain hold 0.48%.
  
  An optimiser with more freedom exploits the model's errors more.
- Within-state ranking test on benchmark v2 (all chunks with real risk): running (eval/rank_check.py).

### 1b. The latent discards most of what a chunk changes
results/jenga/deep_dive/pca_check.json, pca_curve.json:
- The 8-channel PCA of DINOv2 ViT-S tokens (fitted for appearance) keeps:
  - 69% of the static token energy;
  - 28% of the energy of the CHANGE a chunk causes;
  - 24% of that change outside the robot's own region (106k patches, 150 dev decisions).
- PCA fitted on the change itself keeps 38% / 52% / 63% / 75% / 87% / 96% at 8 / 16 / 32 / 64 / 128 / 256 channels.
- Every world model trained since early October lives in this 8-channel space:
  - conv FiLM;
  - ViT predictor (DINO-WM design, vit3);
  - more fork data.

  On the fork-selection set H, all of them plateau at about 70% caught / 10% false (phase1c_selection.json:
  E 69 / 8, conv3 71 / 12, vit3 68 / 12; Phase 1b: more fork data stopped helping).
  
  A full-token model (DINO-WM option 2, docs/PLAN_BRANCHING_WM.md) was planned and never run.

### 1c. World-model size and data
- model_OIL: 2.27M parameters (6 FiLM conv blocks, width 128).
- Trained on PCA8 latents:
  - the counterfactual set (cf_v1: 1700 states x 4 perturbations x 3 plans: the policy's plan + 2 random nudges);
  - plus 200 states of the optimiser's own proposals (oil_v1).
- Earlier ladder: more, and more counterfactual, data monotonically helped the optimiser. b50 real risk:
  threeblock_v3 24.8% -> closed-loop 15.1% -> CF 12.4% -> OIL 11.8%.

### 1d. Off-the-shelf models tried before
- V-JEPA 2-AC zero-shot (ViT-g + AC predictor, DROID-trained) on H: caught 2.6% of forks vs ours 38%.
- Its encoder on REAL futures caught 79%, so its predictor erases the branching.
- V-JEPA 2's encoder localises blocks ~3x worse than DINOv2 (11.8 vs 4.1 mm).
- The DINO-WM checkpoint bundled with the project (data/jenga/world_model.pt) has:
  - a ViT predictor (6 layers, dim 404, 20M parameters);
  - full DINOv2 ViT-S tokens at 224 px (16x16 patches), 3-frame history, proprio + action inputs.
  
  We use its encoder; its predictor was trained on the early single-block Jenga data.

### 1e. Noise model: consistent in simulation
The truth rollouts and the world model's futures use the same execution-noise snippets (subtracted from the commanded
targets), so the speed-independence of the noise is not a sim mismatch. It only matters on a real robot.

### 1f. Decomposition (running): is it the dynamics, the representation or the criterion?
eval/real_fork.py runs the certificate's exact criterion on REAL futures from the simulator:
- same perturbations, scales and split test;
- the monitor's close camera at the same two times;
- encoded (a) as PCA8, (b) as full DINOv2 tokens.

Compared with the world model's scores (rank_check) against real risk, on the same chunks:
- real-PCA8 ranks well, model ranks badly -> the dynamics model is the bottleneck;
- real-full ranks better than real-PCA8 -> the representation is too;
- even real-full ranks badly -> the criterion (fork + disturbance) is wrong for ranking chunks.

### 1g. Decomposition results (2026-10-22 night; results/jenga/deep_dive/)
Within-state concordance: the share of chunk pairs from the same state, differing by >= 2 of 32 futures in real risk,
that a readout orders correctly (ties excluded). Every chunk is scored as chunk + hold tail, the actions the truth
executes.

| set | readout | world model (model_OIL) | real futures, PCA8 | real futures, full tokens |
|---|---|---|---|---|
| dev (150 decisions, 1350 chunks; real: 10 decisions) | h | 0.72 (190/74) | 0.71 | 0.14 (2/12, few pairs) |
| dev | o | 0.80 (476/116) | 0.73 | 0.59 |
| relpass, same chunks (38+ decisions) | h | 0.83 (72/15) | 0.70 (52/22) | 0.68 (58/27) |
| relpass, same chunks | o | 0.67 (82/40) | 0.66 (81/41) | 0.65 (79/43) |

- relpass, all 403 decisions, world model: h concordance 0.80; pooled AUROC 0.76 (h), 0.70 (o). Certified
  (absolute) chunks carry real risk 2.8%, uncertified ones 11.4%.
- Reading 1: the world model is NOT worse than real futures at ranking chunks under this criterion. The criterion
  itself (fork + disturbance from about 32 futures) is a noisy, moderate ranker even on exact futures (0.65-0.75).
- Reading 2 (dev, by arm): the free-offset optimisers' chunks (s2_cem / s2e_cem / s2e_grad) have real risk 5.7-7.6%,
  yet the world model certifies 87-90% of them (hold: 0.5% risk, 92% certified). The model is fooled by large
  off-distribution edits; G's coarse family stays nearer the data.
- Reading 3 (dev, 10 decisions with both): real futures fork much more than imagined ones (hold h 1.0 real vs 1.585
  imagined). The learned model under-imagines branching, the "erased branching" seen before.
- Reading 4 (relpass): of G-CEM's 136 certified-and-changed edits, 124 also certify as chunk + hold (real risk 4.6%,
  about the policy's); the 12 certified only through the 15-step plan carry 16.9%. Choosing among {policy, edit,
  hold} by the chunk + hold certificate gives 5.95% vs policy 5.06%. At these hard states, a ranking accuracy of
  0.7-0.8 is not enough to find a reliably safer chunk.

**Where this leaves the diagnosis.**
1. Not the representation: full tokens don't rank better than PCA8 on real futures.
2. Not mainly a dynamics bug: the model ranks about as well as real futures.
3. The limits are:
   - (a) the criterion's statistical resolution: a fork test on about 32 futures is a coarse risk estimate;
   - (b) model exploitation by unconstrained edits, which needs a trust region / data support;
   - (c) under-imagined branching, which makes the model optimistic.

**Implications for the rebuild.**
- Use continuous readouts with a margin rather than a binary certificate.
- Stay inside the edit support the model was trained on.
- Correct the optimism: calibrate thresholds on the model's own scale, and/or train the model on more branching
  (counterfactual) data.
- The full-token DINO-WM is still worth testing as the off-the-shelf model (step 2). The gate is improvement on
  these ranking numbers, not on token fidelity.

## 2. Plan (each step gated offline on benchmark v2 real risk before any closed loop)
1. Decomposition (1f) + ranking test -> decide where to spend.
2. World model: DINO-WM recipe (off-the-shelf architecture and code, vendored), full DINOv2 tokens, proprio + actions.
   - Initialised from the bundled checkpoint.
   - Fine-tuned on counterfactual Jenga data: the policy's plan + nudged plans, paired perturbations, close camera.
   - Gate: within-state ranking vs real risk better than model_OIL, and fork detection on H no worse.
3. Monitor: sensitivity of the outcome to action perturbations only (no object-pose sampler, so it is task-agnostic).
   - Continuous score: spread / bimodality of the imagined end states under execution-noise draws.
   - Checked on real futures first (oracle) and then with the new world model.
4. Filter: optimisation over the policy's own action chunk, CBF-style.
   - Objective: minimise the distance to the policy's chunk.
   - Constraints: barrier from the continuous sensitivity score, plus the disturbance readout.
   - Robust to model error: pessimistic over an ensemble, and a trust region around the policy's actions.
   - No hold, primitive or backup heuristics.
   - Gates: benchmark v2 dev, then holdout, then closed loop vs G-CEM.
