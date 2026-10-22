# HJ-reachability ideas for the optimisation filter (Jenga), vs G-CEM

User request 2026-10-22: try ideas 2, 3 and 4 on Jenga and compare with the current G-CEM filter. All stay label-free and
optimisation-only. Our own certificate (fork score h + disturbance o, eval/cbf_filter.py ok()) plays the role of HJ's
failure margin; labels are used only to evaluate.

## Idea 2: reachability trigger (spec fixed 2026-10-22, before any run; eval/hj_trigger.py)
Question asked by HJ: "after the policy's next chunk, can I still escape?" instead of "is the policy's plan safe?".
- Current trigger C (Stage 4, model_OIL, h_safe log2 3, TRIGGER both): score [plan16, hold] with 64 futures, fire if
  not ok(h_nom, o_nom, h_now).
- Reachability trigger R: candidates = the plan's first chunk (steps 0-8, step 0 = current target) followed by a backup
  for steps 9-15. Backups (task-agnostic, from the chunk alone): hold at step 8; retreat 10 / 20 / 30 mm along the
  reverse of the chunk's net direction (plan[8] - target; skipped if < 2 mm, as G), ramped over the 7 steps; lift 20 mm
  (gravity up). Gripper at steps 9-15 = the chunk's last command. Fire if NO backup candidate is ok() (same h_now).
- Ablation R-hold: only the hold backup.
- Data: Stage 4 policy-alone episodes 14000-14299 (saved states, results/program_b/policy_rollouts/paired_record_s4).
  Decisions (seeded, rng 20261022): every decision from 8 chunks before the first neighbour topple up to the topple
  chunk in the 69 toppled episodes; plus 600 random decisions from non-toppled episodes. Decisions after a topple excluded.
- Label (evaluation only): a neighbour topple starts at this chunk or within the next 3.
- Metrics: false-alarm rate on the non-toppled-episode decisions (FPR); recall = share of toppled episodes flagged at
  least once in the 4 decisions up to and including the topple chunk; median lead (chunks before the topple of the first
  flag within the 9-decision window); per-decision TPR on the labelled positives.
- **Advance R to closed loop if (recall_R >= recall_C - 0.03 AND FPR_R <= FPR_C) OR (recall_R >= recall_C + 0.05 AND
  FPR_R <= FPR_C + 0.02).** Otherwise idea 2 stops here.

**Result (2026-10-22 12:15; results/jenga/hj/trigger_report.json): FAIL, idea 2 stops here.** 1221 decisions (600
negatives, 69 toppled episodes).

| trigger | FPR | recall (4 decisions to topple) | median lead (chunks) | TPR |
|---|---|---|---|---|
| C (current, plan vs hold) | 5.2% | 92.8% | 3 | 51.4% |
| R (any backup certifies) | 1.0% | 73.9% | 1 | 25.4% |
| R-hold (hold backup only) | 3.7% | 88.4% | 2 | 39.1% |

R fires less often: false alarms fall from 5.2% to 1.0%, but recall falls 19 points and the warning comes later
(1 chunk vs 3). Neither advance condition holds. Reading: after the policy's first chunk, some backup usually still
certifies even 1-2 chunks before a topple. The cheap retreats keep the certificate happy, so "can I still escape?"
answers yes too late. The current trigger already gives a 3-chunk lead at 93% recall; the gap to close is in the fix,
not the trigger.

## Idea 3: learned viability value V(z) (spec fixed 2026-10-22, before any target is computed; eval/hj_value.py)
V(z) = probability that a safe BACKUP exists from latent state z (HJ viability, sampled). Targets are label-free: our own
certificate, absolute form (h >= h_safe = log2 3 AND o <= O_MAX; model_OIL, 64 futures, Stage 4 settings), applied to
state-only backups (no policy plan needed, so V is a function of the state): hold; lift 10 / 20 mm; retreat 10 / 20 mm
toward the robot base in xy (the base is at the origin), all ramped over 15 steps, gripper unchanged. Target = 1 if
any backup certifies.
- Training states: 3000 decisions sampled (seeded) from Stage 4b, all five arms (episodes 14300-14599). Held-out:
  300 decisions from Stage 4 policy-alone (14000-14299), a subset of idea 2's decisions. conf1 / benchmark v2 never used.
- Model: small conv net on the 28x28x8 latent (+ gripper position, gripper state), BCE, 20% of training episodes for
  early stopping. No outcome labels anywhere in training.
- **Gate 3a (fidelity): AUROC(V, certificate target) >= 0.85 on the 300 held-out states.**
- Gate 3b (usefulness, evaluated with idea 2's metrics on ALL of idea 2's decisions; V is cheap to evaluate): trigger V3 = fire if
  V(imagined state after the policy's first chunk, world-model mean) < 0.5. Reported against C and R; it advances to
  idea 4 if 3a passes (3b is descriptive, because idea 4 uses V as a barrier, not as a trigger).

## Idea 4: V as the CBF barrier with a gradient solver (spec to be written after idea 3's gate, before any run)
