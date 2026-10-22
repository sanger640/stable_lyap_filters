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
