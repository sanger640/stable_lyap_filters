# Findings ledger (fork monitor + CBF safety filter)

Every entry has a source file. Check this ledger before quoting any comparison. Newest first within each section.
Last updated 2026-10-16.

## Headline (closed loop, the user's diffusion policy, three-block pick)

| Result | Numbers | Source |
|---|---|---|
| **Final filter passes final test 2** (pre-registered, 600 fresh eps 10000-10599, Alliance) | h_safe = log2 3: topples 114 -> 30 (95 prevented / 11 caused, p = 4e-18); success 374 -> 420; vs first filter 52 vs 9 (p = 9e-9); acts on 7.1% of decisions, median change 10 mm, 15.7 s per intervention; median lift time 18.8 -> 21.6 s | results/jenga/cbf_final2_result.json, cbf_final2_protocol.json |
| First filter (h_safe = 1) final test 1 | 65 -> 47 (27/9, p = 0.002), success 179 -> 187 | results/jenga/cbf_final_test_result.json |
| First filter pooled replications (final test 1 + dev3/4/5/6) | 1,500 eps: 315 -> 206 (-35%), 143 prevented / 34 caused (p = 2e-17), success 893 -> 990 | results/jenga/cbf_pooled.json |
| Success definition | success = lifted 3 cm AND no neighbour toppled (a topple is a failure even if the block was picked) | eval/cbf_final2.py, policy_rollout.py |

## What made the filter work (in order found)

1. Stress test must be in-distribution for the world model: light (0.3x) neighbours were OOD (barrier AUC 0.68 vs 0.90 in-distribution); use normal mass + 2x tracking error. (results/jenga/cbf_barrier_panels_summary.json)
2. Look ahead along the policy's own 15-step plan (H15).
3. Retrain the world model on the policy's closed-loop states: barrier AUC 0.69 -> 0.88 (H15) on held-out closed-loop decisions; real futures in the latent split 97.5%/3.7% (the ceiling) vs 35%/9% for the old model. (results/jenga/closedloop_train_eval.json, lookahead_ceiling.json)
4. Act one step earlier: h_safe = log2 3 (no fork even at 2x errors). Most remaining failures of the first filter were detected only at the topple decision (dev6: 53 of 78 within 3 decisions, 41 at the topple decision). dev7: 113 / 87 / 45 topples (policy / first / final). (results/jenga/cbf_dev7_result.json)

## World models: which is used where (also memory: project-model-registry)

- **Filter (both final tests, all dev runs)** = PLAIN deterministic model (closedloop_v1/model/model_US.pt) + object-centric start sampler G (per-block pose noise, 3 blocks). Never the hybrid, never Gauss-TE.
- **Monitor panels ("our monitor 80%/5%")** = three-block HYBRID HYB_T1 (deterministic + residual flow head), H30_floor readout: any outcome 79.6% caught / 4.6% false vs plain 86.1% / 11.6%; topple 85.6/6.4 vs 88.7/14.9. Hybrid halves false alarms. (results/jenga/panels_v2_result.json, hybrid_T1_result.json)
- phase2_hybrid / hybrid_T2 "hybrid_helps: false" are CROSS-TASK generality tests, not the three-block monitor.
- **Object-free latent uncertainty EXISTS**: Phase 2 P2-0 Gauss-TE (spatially correlated Gaussian noise added directly to the latent, fitted from data; no objects) matched/beat the object-centric G: score 0.709 vs 0.672 (83% caught / 12% false at R0), selected sampler. G* (learned) 0.58. Not yet used in the filter. (results/jenga/phase2_p0_result.json, eval/phase2_p0.py, phase2_models.GaussTE)
- Closed-loop hybrid HYB_CF (2026-10-15): high-precision trigger (fork: 62% of risky caught at 4.6% false vs barrier 85% / 20%); pooled ranking of plans 0.56 vs 0.34 but within-decision pick no better than random. (results/program_b/cbf_qp/hyb_trigger.json, reliability_hyb.json)
- Counterfactual fine-tune model_CF: fork detection up (H15 AUC 0.884 -> 0.896, forks caught 45 -> 64%, false 1.3 -> 0%). (results/jenga/cf_train_eval.json)
- Ensemble (4 det models): averaging improves the pick (15.7% vs ~19% single); disagreement flags 2x larger errors (spearman 0.46). ens_v2 with new data: fixed-readout pick 13.1%. (results/program_b/cbf_qp/ens_check*.json, readout_fix_imagined_ensv2.json)

## Readouts and gradients

- **"No fork" is not "safe"**: the fork readout treats "every future topples" as safe. Fixed readout (no fork -> disturbed if mean outside-hand change > 0.5 x O_MAX) on REAL futures picks 9.0% real risk among the policy's 13 plans (random 19.9%, oracle 6.7%, fork readout 19.3%); on imagined futures 13-15%. (results/program_b/cbf_qp/readout_fix_real.json, readout_fix_imagined.json)
- Fixed readout is a good within-scene selector but a poor trigger across scenes (AUC 0.57 vs barrier 0.88). (qfix_trigger.json)
- Fork-barrier gradients don't track real sensitivity (nudge r = 0.12-0.15); smooth mean disturbance away from the hand does (r = 0.76). Real risk is mm-sensitive (5 mm nudge: 23 points) and smooth (5 vs 10 mm r = 0.91). (checks/report.json, qsoft_check.json)
- Calibration of imagined topple fraction: ECE 0.155; calibrated for 20-50%, overconfident above 50%; "no fork" plans topple 23% in risky scenes (det), 17% (hybrid). (reliability.json, reliability_hyb.json)

## Optimisation filter (the user's chosen direction)

- dev8 interim (75 eps 12075-12149, Nibi): clean CBF optimisation (cbf_opt.optimise: path bends + speed, relative disturbance barrier alpha 0.5, multi-start from policy samples, exact fork check at h_safe log2 3, NO primitives) behind the final trigger: 7 topples vs selection 6 vs policy 13; success 50 vs 55 vs 47; head-to-head 4 vs 5; median change 16.8 vs 9.8 mm; 24.8 vs 9.2 s per intervention. On par for safety, slightly behind on success (larger changes). (results/jenga/cbf_dev8_result.json)
- dev8b (same 75 eps, replays verified identical 75/75): alpha 0.3 (smaller required cut): 9 topples, success 47, median change 11.8 mm, 24.2 s. Worse than alpha 0.5 and selection (head-to-head vs selection 5 vs 8). Smaller changes did not buy success; they cost safety. (results/jenga/cbf_dev8b_result.json)
- Offline optimiser variants (75 risky decisions, real simulator, fork check ON as in closed loop; policy 27.6%, selection 13.8%, old optimiser b50 12.4%; results/program_b/cbf_qp/cbfopt_report_<tag>.json):
  - sep50fc: path change on geometry + sigma inits 0.99/0.55 (only 4 distinct starting plans): 19.4%; worse also where it did not slow (19.0 vs 11.5%) -> fewer distinct starting plans is the likely cause, not slowing per se.
  - sep50 / sep50s1 (22.5% / 25.9%) were run WITHOUT the fork check -> not comparable (my error).
  - CEM v1 on the exact rule (32 futures, 24x3): 24.3%; returned the flagged policy plan unchanged in 45/75 (a noisy re-test passed it; winner's curse) -> CEM v2 (common random numbers, exclude the flagged plan, verify + fallback).
  - al (augmented Lagrangian on CVaR_0.25 disturbance, 16 futures, <=15 steps, 8 starts, fork check): 23.1% (helped 22 / hurt 15 vs policy; median change 6.1 mm; slowed 3).
  - Every new variant today lands at 19-24% vs the single old run b50 12.4% -> control queued: b50 configuration rerun exactly (b50rep) to test whether 12.4% reproduces.
  - CORRECTION (2026-10-16 afternoon): a30/b50 ran with model_CF (cf_v1), but sep50, sep50s1, sep50fc, CEM v1/v2, al, chunk and b50rep ran with the cem_filter DEFAULT world model threeblock_v3 (no CBF_WM set; my error). So those variants are NOT comparable to b50 (solver and model both changed). cbf_opt CFG now records WM. b50rep = b50 settings with threeblock_v3 = a world-model comparison, not a reproduction.
  - World-model ladder, b50 optimiser settings unchanged: threeblock_v3 24.8% (b50rep; helped 31 / hurt 16; verified 84%) vs model_CF 12.4% (b50) -> the world model, not the solver, explains most of the 19-24% vs 12.4% gap. Closed-loop model_US (b50cl): 15.1% (helped 43 / hurt 9; vs selection better 13 / worse 17). Ladder threeblock_v3 24.8 -> closed-loop 15.1 -> CF 12.4: more (and more counterfactual) data monotonically helps the optimiser. dev8 closed loop used the closed-loop model (7 vs 6 topples), consistent with 15.1 vs 13.8 offline. (cbfopt_report_b50rep.json, cbfopt_report_b50cl.json)
  - Optimiser-in-the-loop round 1 (pre-registered, results/jenga/oil_round1_protocol.json): model_OIL = model_CF + 200 states of the optimiser's own proposals (3200 futures, no labels). b50 settings: 11.8% vs model_CF 12.4% (paired: better 23 / worse 17, diff -0.6 +- 1.3 points) -> within noise; rule (<= 10.4%) NOT met, no round 2. vs selection 13.8%: better 14 / worse 10; helped 47 / hurt 5 vs policy; verified 94.7%. (cbfopt_report_b50oil.json)
  - CEM v2 (threeblock_v3): 22.5% (helped 30 / hurt 14 vs policy; worse than selection on 33, better on 7; verified 100%).
  - Learned barrier (eval/learned_barrier.py) is trained on real neighbour-topple labels -> NOT task-agnostic, not admissible inside the filter; its chunk run was stopped. CV ranking: spearman 0.59 vs 0.37 for mean disturbance (diagnostic only: the latent statistics carry the risk signal).

## Variants that did NOT beat the first/final filter in closed loop (all pre-registered dev tests)

- dev4: single gradient CBF optimisation (no primitives): 25 vs 27 topples, success 87 vs 93 -> keep. PROTOCOL DEVIATION: ran with model_US and 15 steps (EXTRA_ENV comma bug), not model_CF/10. (cbf_dev4_result.json)
- dev5: fixed-readout selection among policy samples (cbfq): 37 vs 34 topples, success 182 vs 209 (large changes, mean 33 mm) -> keep. (cbf_dev5_result.json)
- dev6: readout veto on the filter's choice: 85 vs 78 -> keep. (cbf_dev6_result.json)
- Offline only: hybrid "selection + gradient refinement" 4.4% vs selection 5.1% (211 dev decisions); single CBF optimisation 12.4% vs selection 13.8% on 75 risky decisions. (cbf_qp/report_hybrid.json, cbfopt_report_b50.json)

## Thresholds

- O_MAX = 0.484: 99th percentile of the policy's own outside-hand change o over 300 random dev decisions (label-free); no extra margin; regime-only triggers rarely matter (first final test: 197 interventions, 7 helped / 2 hurt). (results/jenga/cbf_calibration.json, cbf_diagnose.json)
- h_safe: 1 (first filter: no fork at 1x errors) -> log2 3 (final: no fork even at 2x). h_safe was chosen with dev topple counts (not label-free).

## Infrastructure facts

- Alliance: user ali313, account def-soojeon; container torch 2.9 + MuJoCo 3.13; MIG slices have no OpenGL -> OSMesa CPU rendering; OSMesa vs EGL latents differ by 2.4% of between-state spread (not a cause of differences). EXTRA_ENV must be ';'-separated (sbatch splits commas). (cluster/README.md)
- Nibi had maintenance 2026-10-15/16 (drained nodes); Rorqual most available.

## Open items (as of 2026-10-16)

- Matched-rate baselines (random interventions at 7% with the same fallback moves; always-cautious) - needed to show foresight, not caution.
- Swap the object-centric G for object-free Gauss-TE in the final filter (fully task-agnostic runtime).
- Second environment (LIBERO + SafeVLA bystander metrics); literature baselines (Sentinel, FAIL-Detect).
- Speed (15.7 s per intervention) and a real-robot pilot.
