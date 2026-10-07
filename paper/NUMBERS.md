# Where every number in the draft comes from

| Claim in paper/main.tex | Source |
|---|---|
| Exact-physics split test: 23/23 DEV topple forks, 10% quiet | results/jenga/monitor_v1_result.json (NOTES "Monitor v1 fails its frozen gates") |
| Execution noise median chunk max 3.2 mm | NOTES monitor v1 interpretation |
| Perception 2.5 mm relative; error model | results/jenga/visual_perception_error_model.json |
| Panel 2: ceiling 0.85, naive 0.32, v1 0.76, v2 0.73, v2 harm 0.90; token vs state 0.90 aware; 22/24 small splits | results/jenga/visual_pipeline_result.json; NOTES panel-2 token diagnostic |
| Panel 3: ceiling 0.92, naive 0.51/0.65, v1 0.70/0.77, v2 0.74/0.90, v1 tokens 0.70/0.77; token vs state 0.86; 14 misses, 12 wrong side | results/jenga/visual_pipeline_panel3_result.json; NOTES panel-3 miss analysis |
| Panel 4: ceiling 0.92, naive 0.42/0.67, v1 0.84/0.91, v2 0.82/0.89, v1 tokens 0.77/0.88, v2 tokens 0.83/0.88, 7% non-harm | results/jenga/visual_v2_panel4_result.json |
| Compression choice (channel PCA 8) | results/jenga/token_compression_result.json |
| Token WM: stochastic 0.18/0.18, deterministic 0.42/0.60; no-motion baseline 19-step latent MSE 0.176 vs 0.055 (stoch.) / 0.042 (det.) on 40 val states (wm_*_log.json, computed 2026-10-03) | results/jenga/step3_token_world_model_result.json |
| Diagnostics: spread 0.98, PC1 0.15 vs 0.57, D 0.04/0.69, D_pert 0.37 vs D 0.17, fork/non-fork 1.3x / 4.8x / 32x | results/jenga/step3_diagnostics.json |
| Iteration 2 | TODO: results/jenga/step3b_p1_result.json, step3b_panel5_result.json |
| Figure 1 (a) panel-3 state 73:34 aware endings; (b) D0 state ep63_seed104_s74 (most balanced real split, minority 31/64), models wm_stochastic/wm_deterministic, seeds as step3_diagnostics; (c) kappa from panel 2/3/4 result files | paper/make_figure.py |
| Naming in paper v5: test set A/B/C = fresh panels 2/3/4; "fork probe" = aware v1; "5 hypotheses" = Bayesian aware v2; "no perception sampling" = naive; "topple fork" = harm fork; 112 development states = DEV (23 topple forks, 89 quiet); latent-vs-state split agreement 0.93 = gate D0 | results/jenga/token_compression_result.json, results/program_b/step3/d0_result.json |
| Paper v6 (uncertainty & safe deployment rework): every kappa/recall in the abstract, Sec. 3 and Tables 1-2 is against the physical fork label (2-62/64 oracle topples); ceiling 0.99; probes 0.70-0.81, recall 0.77-0.91; perception ignored 0.42-0.54; WM stochastic 0.19/0.18, deterministic 0.52/0.60; set C 5-hyp images 0.81, recall 50/57 = 0.88 (Wilson 0.77-0.94), false alarms 0.07 | results/jenga/fork_label_reanalysis.json |
| Iteration 2: P1 kappa 0.29 vs real-latent splits, oracle starts 0.19, spread 0.28x, main-axis share 0.55, 16 vs 35 real-fork alarms | results/jenga/step3b_p1_result.json |
| Paper v7: kappa dropped; forks caught / false alarms vs the physical fork label (Table 1 from fork_label_reanalysis recall/false_alarm); D0: latent split test caught 97% (17% false), iteration 2 G starts 48% (3%), rendered starts 45% (10%); oracle runs agree on 99% of states | results/jenga/fork_label_reanalysis.json, results/jenga/d0_fork_rates.json |
| main_phase1.tex (version WITH Phase 1): Table 2 rows "5 generative variants" 0-3% caught / 0-1% false, spread 1.1-1.6, main axis 0.07-0.10 (Phase-1 cells A,B,C,D,F on H); "uncertain starts + deterministic, 4x forks" 69% / 8%, spread 0.35, main axis 0.48 (cell E); 52% / 7% = iteration-2 recipe on H | results/jenga/phase1_selection.json, results/jenga/phase1_iter2_ref_on_H.json |
| main_phase1.tex (2026-10-07, physics-sim/WM workshop edits): "giving a generative model the same perturbed starts does not help (0%)" = Phase-1b arm d; "plateaus at 68-72% with 3x training, more fork data, frame history or DINO-WM's predictor" = a1 72%, a2 68%, conv3 71%, vit3 68% (E 69%); the refuted "hidden pose error" sentence removed | results/jenga/phase1b_selection.json, results/jenga/phase1c_selection.json |
