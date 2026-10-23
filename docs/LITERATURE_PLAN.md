# Literature deep dive (2026-10-23) and the plan it grounds (Jenga only)

Scope (user mandate):
- optimisation filter on the policy's own actions, CBF / HJ-inspired;
- off-the-shelf world model in latent space;
- monitor based on sensitivity of the dynamics to actions;
- task-agnostic and label-free.

Every next step below cites the work it builds on and the evidence from our own runs that motivates it.

## 1. What the field does, and where our results sit

### Latent safety filters (HJ reachability in a world model's latent space)
- **Nakamura, Peters, Bajcsy, "Generalizing Safety Beyond Collision-Avoidance via Latent-Space Reachability Analysis"**
  (RSS 2025, arXiv 2502.00935).
  - DINO-WM (hardware) / RSSM; failure margin = a classifier trained on labelled failures; HJ value learned by RL in
    imagination; least-restrictive switching.
  - Franka, toppling and spilling tasks.
- **Seo, Nakamura, Bajcsy, "UNISafe"** (CoRL 2025, arXiv 2505.00779): adds epistemic uncertainty (ensemble), a conformal
  threshold, and reachability in the augmented space. This is our head-to-head 2 baseline, which G-CEM beat 19 vs 55
  topples.
- **AnySafe** (arXiv 2509.19555): constraint-parameterised latent filters.
- **Tabbara et al., "Designing Latent Safety Filters using Pre-Trained Vision Models"** (arXiv 2509.14758): DINOv2
  backbones perform consistently well (we use DINOv2).
- **Nakamura et al., "How to Train Your Latent CBF"** (L4DC 2026, arXiv 2511.18606).
  - Classifier margins give saturated, discontinuous values that are useless for smooth CBF filtering.
  - Smooth margins need gradient penalties, and values must be trained on nominal-policy data too.
- **"When World Models Lie"** (arXiv 2609.34300): adaptive conformal inference on the latent one-step prediction error
  sets a pessimistic ball around the predicted latent; the filter takes the min of V over it. Baselines: LatentSafe,
  UNISafe, AcoFi.
- **Nath et al., "Pixels to Proofs" (SLS^2)** (arXiv 2606.15594): Le-WM-style JEPA latent model; conformal ellipsoidal
  error sets; robust MPC via system level synthesis; an **in-domain latent set** (conformal Mahalanobis ellipsoid) the
  reachable tube must stay in (a data-support trust region).

**Where we differ:** all of these need failure labels or a hand-specified failure set for the margin. Ours is label-free.
The margin is the fork / disturbance structure of imagined futures under action perturbation. Our HJ ideas 2-3 (viability
value learned from our own certificate) failed because cheap backups certify almost everywhere (docs/HJ_REACHABILITY.md).

### Failure prediction with pretrained world models
- **Foresight** (arXiv 2606.23085): V-JEPA 2-AC predicted latents plus a causal transformer trained on episode-level
  success/failure labels, with functional conformal thresholds. ROC-AUC 0.79-0.93 on real arms.
  - Label-based, so it is a baseline class rather than our method.
  - Our zero-shot V-JEPA 2-AC test caught 2.6% of forks: its predictor erases branching.

### Early-warning signals of critical transitions (bifurcations)
- **Scheffer et al. 2009 (Nature, "Early-warning signals for critical transitions")**: near a tipping point the basin
  flattens (critical slowing down), so variance and autocorrelation RISE before the transition, and the TREND warns
  before the level does.
- Limits: noise-induced and rate-induced transitions give little warning; false-alarm rates must be quantified.
- **Mapping to us:** the fork scale h = log2 of the smallest perturbation scale whose futures split is a distance to the
  bifurcation in units of noise. Its decrease over successive decisions is the early-warning trend. A topple is a
  bifurcation (block stays vs falls).

### Discrete-time CBFs, margins, predictive / backup filters
- Discrete-time CBF (Agrawal & Sreenath, RSS 2017): require h(x_{k+1}) >= (1 - gamma) h(x_k), i.e. do not approach the
  boundary faster than a class-K rate. A larger margin / smaller gamma acts earlier, at the cost of conservatism.
- **Backup CBFs, model predictive shielding, gatekeeper** (Agrawal, Chen, Panagou, T-RO 2024, arXiv 2211.14361):
  - accept the nominal trajectory up to a horizon only if, stitched to a backup trajectory, it stays safe;
  - conservatism is set by the backup, lead time by the horizon.
  - **Our trigger is this structure:** the policy's 15-step plan stitched to a hold tail, certified by the fork test, so
    hold is the backup. That is the formal framing for the paper.

### Risk-aware (CVaR) barriers
- CVaR barrier functions (Ahmadi et al.; Kishida 2023; sample-based belief CBFs, arXiv 2504.04097; survey arXiv
  2403.18972): constrain the tail of the barrier's distribution, estimated from samples, rather than its mean.
- **Mapping:** our resolution check found that the ranking information sits in the extreme tail of the imagined
  disturbance (max > q90 > mean), so a tail-risk barrier is the principled form of o.

### Off-the-shelf world models
- DINO-WM (frozen DINOv2 + ViT predictor) and V-JEPA 2-AC (1.2B, latent) are the latent models; Cosmos and other
  pixel-generating models are minutes per candidate (too slow for 64 futures x 9 candidates per decision).
- LaDi-WM (latent diffusion over DINO + CLIP, CoRL 2025) is stochastic, but no evidence that it represents
  outcome bifurcations.
- "Certified World Models" (arXiv 2606.13092): learned models can match one-step error yet get multi-step growth
  rates wrong, and local sensitivity spectra can be uninformative in near-neutral contact tasks.
- **Our evidence:**
  - V-JEPA 2-AC zero-shot erases branching;
  - DINO-WM full tokens fine-tuned on 1100 counterfactual states ranks chunks no better than model_OIL
    (docs/FILTER_DEEP_DIVE.md section 4);
  - real futures rank no better than model_OIL either (section 1g).

  Capacity and representation are therefore not the limit on Jenga. A longer-trained or larger WM is not the
  highest-value next step unless a gate below points to the model.

## 2. Our evidence that defines the problem (docs/FILTER_DEEP_DIVE.md, FINDINGS.md)
- The trigger's timing works: fork-timed vs random holds, 8 vs 34 topples.
- The residual is LATE warnings. The trigger fires before nearly all residual topples, but often only 0-1 chunks ahead.
- A larger margin warns earlier. Exploratory, policy-alone episodes (results/jenga/deep_dive/margin_check.json + the
  trend analysis below):

| alarm rule (fork scale d = smallest splitting scale; relative = plan forks earlier than hold) | alarm rate, safe episodes | topples warned >= 3 chunks ahead | median lead |
|---|---|---|---|
| G trigger (relative, d <= 2x) | 6.6% | 48% | 2 |
| relative, d <= 3x (under test: margin3_protocol) | 12.0% | 72% | 4 |
| relative, d <= 4x | 15.7% | 89% | 6 |
| **relative: d <= 2x, OR d fell since the previous decision and d <= 4x (CBF decrease / early-warning trend)** | **12.7%** | **87%** | **6** |

The trend rule gets the lead of the 4x margin at the alarm rate of the 3x margin. Post hoc on the same episodes, so it
needs a fresh pre-registered test.

## 3. Plan (each step gated; closed-loop tests pre-registered on fresh episodes)
1. **3x margin closed loop** (running; results/jenga/margin3_protocol.json).
2. **Fork-distance CBF with a decrease condition (early-warning trend).**
   - Barrier: h_k = fork scale of the policy's plan at decision k (on scales 0.5 / 1 / 2 / 3 / 4).
   - Trigger: h_k < log2 3 (current), OR h_k < h_{k-1} with h_k <= log2 4, i.e. the discrete-time CBF condition
     h_{k+1} >= h_k is violated near the boundary; relative to hold, as now.
   - Fix certificate: candidate chunk must satisfy h >= min(h_{k-1}, h_hold, H_SAFE), i.e. not move closer to the
     bifurcation than the last accepted plan.
   - Same optimiser (G-CEM).
   - Gate: closed loop vs the winner of step 1.
3. **Tail-risk (CVaR) disturbance barrier in the fix:** replace max-over-futures o by CVaR_alpha over futures x
   ensemble models (alpha from dev).
   - Gate: within-state concordance on relpass / dev (section 5a numbers), then benchmark v2 holdout risk.
4. **Data-support trust region** (SLS^2 in-domain set; "When World Models Lie"): constrain edits so imagined latents
   stay inside a conformal Mahalanobis region of the WM's training latents.
   - This lets the fix use a richer family than G's 4 knobs without the exploitation seen with free offsets (1g reading
     2).
   - Gate: benchmark v2 dev / holdout real risk vs G-CEM.
5. **World model:** only if a gate points to the model. Candidates:
   - stochastic latent model (latent diffusion over DINOv2, LaDi-WM style) to represent branching;
   - error-adaptive pessimism ("When World Models Lie").

## 4. Results log
- **Step 3 (CVaR tail barrier), offline gate: no gain** (results/jenga/deep_dive/res_{dev,relpass}.json).
  Within-state concordance at 64 / 256 futures:

  | readout | dev | relpass |
  |---|---|---|
  | max | 0.787 / 0.809 | 0.684 / 0.737 |
  | CVaR 2% | 0.787 / 0.804 | 0.684 / 0.707 |
  | CVaR 5% | 0.794 / 0.802 | 0.684 / 0.678 |
  | CVaR 10% | 0.791 / 0.787 | 0.671 / 0.671 |

  The single most extreme imagined future is the most informative; the max stays. Closed.
