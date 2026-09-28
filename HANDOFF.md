# HANDOFF — stability-based safety monitor

> **Active handoff (2026-09-28):** the current decision is the read-only re-observation/interface
> analysis in `PLAN_NEXT.md`. D4 and D5 close the present privileged-Jenga architecture search;
> do not tune them, select a favorable seed, or open TEST. The dated blocks below are a reverse-
> chronological audit trail, so their older “Next” statements are historical rather than active.

> **Latest result (2026-09-27): the complete-trajectory multimodal D5 architecture is implemented
> but fails its frozen TRAIN pilot gate; DEV and TEST were not touched.** D5 warm-starts the D2 GNN,
> predicts three coherent 38-step residual futures, and routes from initial state plus action sequence
> only. Proper mixture likelihood, D4 relational losses and task-free balance avoid collapse:
> deployed occupancy is **41.5/20.1/38.5%**. Against deterministic full-D4 seed 1 on the identical
> 36-state validation set, D5 keeps candidate recall/quiet alarms at **66.7%/16.7%**, improves
> topology error **0.04066 -> 0.04028** (-0.95%), but worsens commitment **0.03853 -> 0.03955**
> (+2.67%) and response **0.03101 -> 0.03213** (+3.63%). The preregistered lower-commitment gate
> fails, so do not replicate seeds, tune D5, or open another DEV run. The deterministic D4 and
> explicit multimodal D5 results together close the current privileged-Jenga architecture search.
> Next reassess the monitor/world-model interface—especially whether demanding branch decisions
> from an open-loop learned rollout is the right deployable decomposition—before Panda or vision
> transfer. Canonical result: `results/jenga/d5_multimodal_pilot_summary.json`.

> **Latest result (2026-09-27): D4 passes replicated TRAIN structural gates but fails its frozen
> matched DEV monitor gate; TEST was not touched.** The expanded label-free corpus contains 256
> TRAIN states x 64 actions x 38 steps, with 220 training and 36 episode-held-out states. Across
> fixed seeds 1-5, full D4 lowers median topology/commitment error versus matched continuation by
> **8.1% / 3.9%**, retains frozen candidate recall at **62.5%**, and stays within the response
> guardrail (+4.8% median); every TRAIN gate passes. Under unchanged Regime Monitor v0 on all 140
> DEV states, however, matched continuation gives median **8/23 topple, 2/89 quiet** alarms and
> **77.1%** physical-v0 agreement, while full D4 gives **8/23, 3/89, 75.7%**. The prospectively
> required strict recall improvement and nondecreasing physical agreement both fail. Do not select
> seed 5 (11/23, 1/89) post hoc and do not open TEST. This deterministic GNN can improve average
> relational geometry without reliably preserving the discrete branch realization needed by the
> monitor. Next test exactly one multimodal trajectory architecture under the same universal D4
> supervision and matched controls. TRAIN result: `results/jenga/d4_replication_summary.json`;
> DEV result: `results/jenga/d4_dev_summary.json`.

> **Latest result (2026-09-27): D4 full-neighbourhood supervision is implemented and passes a
> promising one-seed TRAIN-only architecture gate; DEV and TEST were not touched.** A new bounded
> corpus records 48 label-free TRAIN states x 64 nearby actions x the complete H=8 + hold-30
> trajectory. Matched continuation improves absolute response error but damages topology and cuts
> physical-candidate recall from **0.80 to 0.40**. Full D4 jointly matches trajectories, complete
> pairwise response geometry, nearest-action commitment curves and nested D3 boundary scaling. It
> retains **0.80 recall / 0.714 candidate agreement**, while reducing response/topology/commitment
> validation error by **34.1% / 4.5% / 6.7%** versus frozen D2. Relative to matched continuation,
> it accepts 14.4% more response error for **12.6% less topology error, 19.6% less commitment error,
> and twice the candidate recall**. This is seven episode-held-out TRAIN states (five candidates,
> two quiet), so do not claim monitor performance or open DEV yet. Next replicate the frozen
> comparison over several seeds on a larger TRAIN-held-out neighborhood panel; advance to matched
> DEV only if the structural gain is stable. Protocol: `results/jenga/d4_protocol.json`; result:
> `results/jenga/d4_one_seed_summary.json`.

> **Latest result (2026-09-27): the preregistered pair-selection crossover shows that both D2
> response topology and D2 trajectory dynamics fail; TEST was not touched.** On physically selected
> candidates and nested paths, physical trajectories give **18/23 topple alarms, 7/89 quiet**,
> whereas unchanged D2 gives only **4/23, 5/89**; the diagnostic target was >=14 topples and <=7
> quiet. On 69 physically selected topple-state pairs, D2 recovers only 29/68 physical boundaries
> and shares 28; boundary agreement is **40.6%**. Commitment agreement is **59.4%** (D2 43,
> physical 51, shared 33), while persistence agreement remains **95.7%**. Combined with the prior
> result that D2-selected paths replayed physically give 0/23, this proves two independent defects:
> D2 chooses the wrong pairs and predicts the right physical pairs incorrectly. The next model must
> be trained on whole probe-neighbourhood relational topology and dense commitment trajectories
> jointly; pair-level post-hoc corrections are closed. Judge it by agreement with the task-label-
> free physical monitor before using topple labels. Protocol SHA-256:
> `2ac21243b944e15af458eef01d4d335286e38fdec7826c2d715263f4391e33fc`; result SHA-256:
> `f3d8fa7c38d027700d4c64a2aaf09a1727b8ba9770aa48d4a598f352ba3eb630`.

> **Latest result (2026-09-27): the preregistered trajectory-level 2x2 DEV oracle ablation shows
> that D2 selects the wrong counterfactual pairs; TEST was not touched.** Holding D2 candidate
> discovery and all three nested pair paths fixed gives: D2 boundary+D2 consequence **8/23 topples,
> 2/89 quiet**; physical boundary+D2 consequence **9/23, 1/89**; D2 boundary+physical consequence
> **0/23, 1/89**; physical boundary+physical consequence **0/23, 0/89**. On the 60 selected pairs
> from topple states, D2/physical agreement is only **46.7% for boundary** and **25.0% for
> commitment**. D2 marks 49/60 committed while physics marks only 8/60; 43 are D2-only. Physical
> persistence is usually present (54/60), so rollout length is not the main issue. The all-physical
> failure means the frozen D2-selected pairs do not straddle physically committed topple outcomes;
> correcting their gaps or tails cannot recover the monitor. Next run a pair-selection crossover:
> physical-selected paths evaluated with D2 and physical trajectories, versus these D2-selected
> paths. If physical selection restores the ceiling, train full-neighbourhood response-topology/
> pair-selection supervision rather than another post-hoc router. Protocol SHA-256:
> `450869610a5cf14bd257a8152bad8e7826aa38f6e1037e981f9662ddef66ff8b`; result SHA-256:
> `a796be647234cd1a94f94412cc28105ffaea0caca9f38bfaebf44a96495c39fd`.

> **Latest result (2026-09-27): the prospectively frozen selective-router DEV experiment fails;
> TEST was not touched.** On the original frozen D2 seed 1, identity gives **8/23 topple alarms,
> 2/89 quiet alarms**. Ten forced-router seeds give median **15/23, 4/89**; ten selective cost-
> router seeds give **15.5/23, 5/89** (ranges 14-16 and 3-5). The gate required >=10 topples,
> <=3 quiet, and no degradation versus the forced router: selective routing fails both quiet
> conditions. Constant correction gives 15/23 and 8/89. Most importantly, a grading-only oracle
> that replays the exact D2-selected nested paths in MuJoCo and chooses the option minimizing true
> physical scale-curve cost gives only **9/23 and 4/89**. Thus the learned recall gain comes from
> aggressive boundary restoration, not faithful approximation of a selective physical oracle;
> the four post-hoc curve corrections cannot supply the required end-to-end tradeoff even with
> future information. Stop this correction-router line and do not tune on DEV. Next run a
> trajectory-level oracle ablation to localize whether D2's remaining error is in boundary scaling,
> commitment/persistence trajectories, or both before choosing another world-model architecture.
> Protocol SHA-256: `aa1472e51f76f1bd7fd4bd5a45e3b072a0db3419e56bfc35dc6234fbea712a17`;
> result SHA-256: `1b00df73098d6a55a5ad45227d4e6fe7add8c2b35fd10115ec81768c022f096f`.

> **Latest result (2026-09-27): the TRAIN-only selective cost router passes every prospectively
> frozen gate; DEV and TEST were not touched.** Adding an explicit identity/no-correction option
> and predicting task-label-free physical reconstruction cost fixes much of the prior forced-
> correction failure. Across ten fixed seeds, median cost-oracle option accuracy is **60.0%**,
> global/local oracle-gain recovery is **74.9%/73.8%**, persistent recall is **81.3%**, and smooth
> alarm FPR is **37.5%** (1.5/4 median; seed range 25-50%). Identity or weak mode 0 is selected for
> **62.5%** of smooth cases. Every fixed gate passes, but the smooth sample is only four groups and
> the result remains privileged-state TRAIN validation—not deployable evidence. The next justified
> action is to freeze a matched DEV protocol comparing D2, the prior forced three-mode router, this
> selective router, the constant-option control and the future-informed cost oracle. Do not touch
> TEST. Protocol SHA-256: `d05dfc5073420e456f02b836dd303f6d2ec46cf9370c17810b3958667fd10cb6`;
> result SHA-256: `a9772b73bf2255d37fbbfc4cb0c497bdbab487403db49d6efa0d51c5bd66f0b9`.

> **Latest result (2026-09-26): the TRAIN-only causal router fails the prospective specificity
> gate; DEV and TEST were not touched.** Across ten fixed seeds, the action-conditioned MLP reaches
> median **56.7%** oracle-mode accuracy, recovers **68.8%/70.8%** of the oracle's global/local
> curve-loss gain, and raises persistent recall from D2's **20.8% to 75.0%**. But its smooth-case
> alarm false-positive rate is **75% median** (range 50-100%, only four smooth held-out groups),
> above the frozen <=50% veto. The future-informed oracle has 91.7% persistent recall and **0%
> alarm FPR**: it routes all smooth cases to weak mode 0, whereas the causal router usually sends
> them to stronger modes 1/2. This is a regime-identification/smooth-abstention failure, not a lack
> of useful prototypes. Do not run DEV or TEST. Next add an explicit task-label-free no-correction
> or smooth-abstention option and train routing by physical curve reconstruction/cost, not only
> nearest-prototype classification. Protocol SHA-256:
> `0228e9d15cc316781e36236054dfaad28af7d13cb7f0621984890eff438c695f`; result SHA-256:
> `ff68e2a62458926c6620e84550df460ed4c7a3760019b8f492f675f976eb6332`.

> **Latest result (2026-09-26): the TRAIN-only three-regime oracle-routing upper bound passes its
> prospectively frozen feasibility gate; DEV and TEST were not touched.** Three capacity-balanced
> correction prototypes were fit to physical-minus-D2 nested scale-curve residuals from 226 TRAIN
> groups. On 30 episode-disjoint held-out TRAIN groups, a non-deployable router that sees the
> physical future reduces global scale Huber loss by **83.0%** and adjacent-level local loss by
> **65.4%**; held-out mode occupancy is **26.7/43.3/30.0%**. Thus D2's missing branch geometry has
> reusable low-cardinality structure, and a causal router is worth testing. This is not a monitor
> result: the oracle uses the answer to select a mode. Its descriptive persistent recall rises
> **20.8% -> 91.7%**. Strict smooth fidelity falls **3/4 -> 0/4**, but those four predictions stay
> below the persistent/alarm threshold, so alarm FPR is **0/4**, not 4/4. The next causal test must
> gate both branch recovery and smooth-case specificity rather than optimize mean curve error alone.
> Protocol SHA-256: `2418bca3cf0d8ddef64e0213d4a5087cafd1965b50b3c5600a6fb75d879520cc`;
> canonical result: `results/jenga/oracle_routing_upper_bound_summary.json` (SHA-256
> `1533473b307cb219e0a8670a3d4a8072a0d76d864d4befa6ca6c4aa69eac6981`). **Next:** on TRAIN
> only, predict the oracle assignment from causal pre-rollout inputs using balanced hard routing;
> require held-out assignment skill plus persistent/smooth fidelity before any new DEV run.

> **Latest result (2026-09-26): the prospective one-seed four-arm switching-edge DEV screen is
> complete and both learned switching arms fail the feasibility gate; TEST was not touched.** All
> arms start from D2 seed 1 and receive three matched continuation epochs. Frozen-v0 DEV results are:
> soft D2 **10/23 topples, 1/89 quiet**; hard-contact D2 **11/23, 9/89**; switching D2 **4/23,
> 1/89**; switching+D3 **4/23, 2/89**. The gate required >=10 topple alarms and <=3 quiet alarms.
> Switching D2's soft occupancy looks balanced (0.368/0.253/0.376), but entropy is 0.988 and
> deterministic argmax selects one mode on 99.994% of edge-steps. Switching+D3 collapses completely
> to one mode (occupancy 0/1/0, entropy approximately zero). Their topple early/full gap ratios are
> only 0.043/0.046 and 0.047/0.049, versus soft D2's 0.082/0.085: neither preserves physical branch
> topology. Do not expand seeds or run TEST. The next architecture change must make discrete mode
> allocation identifiable (for example balanced hard assignment/optimal-transport routing), not
> merely increase the present regularizer weights. Canonical result:
> `results/jenga/switching_edge_four_arm_summary.json` (SHA-256
> `1190e67b3df3b6df85ca46ee7969b06784e918138bad802e065ba130c027bc5f`).

> **Latest implementation (2026-09-26): persistent switching-edge GNN is implemented and passes
> an end-to-end D2+D3 smoke test; no scientific performance result exists yet.**
> `StepSwitchingEdgeGNN` assigns a straight-through categorical latent mode to each of the 12
> directed interaction edges, conditions message passing on mode-specific experts, and carries the
> selected modes through the rollout. A shared action-context pass lets block-block gates see robot
> action information. Modes have no semantic or task labels. Balance, entropy and temporal-
> persistence regularizers address collapse, indecision and flicker. Trainer kind `edge_switch`,
> checkpoint loading, frozen-v0 learned rollout and D3 all support it. Five focused architecture
> tests plus a real-data one-batch H=8+hold30 smoke run pass. The smoke result is not evidence of
> monitor quality. **Next:** prospectively run matched soft D2, naïve hard-contact, switching-edge
> without D3, and switching-edge+D3 arms on DEV; do not touch TEST until a multi-seed DEV gate.

> **Latest experiment (2026-09-25): the first scientific D3 pilot fails on DEV; frozen TEST was not
> touched.** A prospectively fixed 256-state TRAIN dataset contains 60 fully smooth curves, 167 with
> a persistent phase and 100 persistent into the late hold. Starting from D2 seed 1, matched
> three-epoch continuations compare D2-only against D2+D3. Frozen v0 on DEV gives the control
> **10/23 topple forks and 1/89 quiet alarms**, but D3 only **3/23 and 3/89**. D3 raises coarse
> topple candidates from 19 to 21, then collapses refined boundaries from 11 to 3. Median DEV
> topple final/initial gaps fall from 0.082/0.085 (early/full) to 0.040/0.042, close to smooth
> 1/32 scaling. D3 did lower its TRAIN scale loss, especially on quiet curves, so this is a
> generalization/representation failure rather than an inactive objective. A no-training hard-
> contact diagnostic recovers 11/23 topples but alarms on 13/89 quiet states: discrete mode feedback
> restores branches indiscriminately. **Do not expand D3 seeds or run TEST. Next test a learned
> selective hybrid/contact mode with smooth and persistent physical scale curves as supervision.**
> Canonical artifact: `results/jenga/d3_pilot_summary.json` (SHA-256
> `e9dc68cc352bad837791f7de34bfe68843e2ec0bf0c31203d9d3075f223dd612`).

> **D1 control completed (2026-09-25): D1 preserves more of v0's branch geometry than D2, but with
> a sensitivity/specificity tradeoff.** Running the same frozen evaluator on all ten existing D1
> checkpoints gives **44.6% mean topple-fork recall** (29.8-60.7%) and **4.78% quiet alarms**
> (1.8-8.8%), versus D2's 31.1% / 1.95% and physical v0's 65.5% / 3.5%. Mean D1 topple stage counts
> are 66.6 initial candidates, 46.1 refined boundaries, 55.9 commitment majorities, 63.6 persistence
> majorities and 37.5 alarms. D1 has 13 robust alarms and 26 robust blind topples, versus D2's 3 and
> 35. Median final/initial gap ratios at topple forks are 0.263/0.727 (early/full) for D1 versus
> 0.074/0.097 for D2 and 0.997/0.997 physically; quiet ratios remain near smooth 1/32 for both
> (D1 0.0341/0.0341, D2 0.0326/0.0329). Therefore D1's pair-distance loss partly preserves true
> plateaus but also creates some false plateaus: it is evidence for D3's nested physical scale-curve
> target, not a deployable winner. Result: `results/jenga/regime_monitor_v0/d1/aggregate.json`
> (SHA-256 `7011941ed32424210d97e9a1c03ec42580e45dac7f2358b8197993034fc9589a`).

> **Latest learned-world-model result (2026-09-25): frozen v0 on the existing D2 GNN fails the
> deployment gate.** This is not a new model: `eval/jenga_regime_monitor_v0_d2.py` evaluates all ten
> already-trained intervention-consistency GNN checkpoints (`w6_cw_d2_s1..s10`) on the frozen Jenga
> TEST states. Every original probe, adaptive midpoint and final refined endpoint is predicted by
> D2; MuJoCo supplies no future trajectory. The unchanged calibration-free v0 gets **31.1% mean
> topple-fork recall** (range 8.3-48.8%) and **1.95% mean quiet alarms** (0-5.3%), versus physical
> v0's 65.5% and 3.5%. Mean topple stage counts are 63.3/84 initial candidates, 32.5 refined
> boundaries, 52.4 commitment majorities, 60.7 persistence majorities and 26.1 final alarms. Thus
> the dominant learned-model loss is local boundary scaling/refinement, not the final consequence
> test. Only three topple states alarm in >=8/10 seeds, while 35 are missed in >=8/10. Mean agreement
> with physical v0 decisions is 71.0%; only 25.0% of physical alarms are recovered and 8.45% new
> alarms are added among physical non-alarms. Result:
> `results/jenga/regime_monitor_v0/d2/aggregate.json` (SHA-256
> `fa9ba7e24dda73f0fd140ead5868a5db61fd8f6d1b38889e00d7ec8759865d94`). **Next:** do not retrain
> another generic state GNN or tune v0 on TEST. Diagnose D2's midpoint scaling error on TRAIN/DEV,
> then prospectively train a boundary-consistent model and rerun this exact evaluator.

> **Latest implementation (2026-09-23): timing audit plus pushing/insertion prototypes.** Replaying
> the frozen intervention decisions gives median +0.2 s time-to-pick for reobserve, wrapper and
> oracle; paired mean delays are +0.249, +0.230 and +0.252 s. Mean episode duration changes 16.974 s
> -> 17.244/17.249 s. This is physical control time, not compute latency: exhaustive ground-truth v0
> is still far from real-time (~18 s for an isolated quiet decision). One neighbor-threshold replay mismatch occurs in each monitored arm because
> the old Jenga snapshot omits MuJoCo solver warm-start; future benchmarks now use integration-state
> snapshots. `src/systems/contact_benchmarks.py` adds relevant mechanism prototypes: pushing
> goal-versus-edge-fall and tight insertion-versus-rim-jam. Inspect
> `results/contact_benchmarks/previews/` and `CONTACT_BENCHMARKS.md`. These are environment-design
> previews, not frozen v0 results. **Next:** prospectively define execution noise and mechanism
> coverage, then freeze pushing DEV/TEST before running the unchanged detector/wrapper.

> **Latest result (2026-09-23): the frozen intervention experiment passes its Jenga utility gate.**
> On 100 matched fresh-reset episodes, no monitor gives 34 neighboring-block failures, 80 picks and
> 51 safe completions. Reobservation alone gives 31/79/53. The label-blind v0 wrapper gives
> **25 failures, 79 picks and 59 safe completions**; the privileged local oracle gives 20/79/63.
> Wrapper versus baseline is -9 failure points (paired 95% CI -15 to -3, p=0.0117) and +8 safe-
> completion points (CI +1 to +15, p=0.0386). It also prevents six failures beyond reobservation
> alone (p=0.0313). Cost: 275 one-step reobservations across 2,049 decisions, touching 97/100
> episodes, and 173 modified chunks. Thus utility is positive but conservative intervention breadth
> remains high. Frozen result: `results/jenga/intervention_v0/ground_truth_100.json`; analysis:
> `results/jenga/intervention_v0/analysis.json`. **Next:** do not tune on Jenga; build the pushing
> ground-truth benchmark and carry the unchanged detector/wrapper to pushing and insertion.

> **Latest result (2026-09-23): frozen Regime Monitor v0 TEST is complete.** The one-time,
> verifier-gated ground-truth evaluation gets **55/84 topple forks (65.5%)** and **4/113 quiet
> committed branches (3.5%)**. Initial candidate discovery and refined boundary evidence retain
> 80/84 and 78/84 topples; the final commitment/persistence conjunction is the recall bottleneck,
> removing 23 more. DEV-to-TEST topple recall changes 78.3% -> 65.5%; quiet branch rate changes
> 7.9% -> 3.5%. The signal generalizes, but its recall is not sufficient for a standalone stop
> filter. The immutable result is `results/jenga/regime_monitor_v0/test_ground_truth.json` (SHA-256
> `9a7b4004647f09a165bd52d2ea959dc0ef81e93cadd760eb256327917cadf0f5`) and the descriptive report
> is `results/jenga/regime_monitor_v0/test_ground_truth_analysis.json`. Do not alter v0 from TEST.
> That result motivated the intervention experiment reported above; the wrapper is now implemented,
> prospectively frozen and evaluated. Preserve this detector TEST paragraph as the detector-only
> result, but follow the newer intervention result and cross-task next step at the top of this file.

> **Current plan: [`PLAN_NEXT.md`](PLAN_NEXT.md) (status 2026-09-23).** Two tracks: Track A is the
> privileged-state mechanism study (an oracle; Jenga features allowed for diagnosis), Track B the
> universal deployable monitor (generic vision + proprioception + action, no Jenga state).
>
> **Where it stands.** Phase 1 done: the benchmark is frozen behind one evaluator that refuses to run
> if the cache or any constant changes. Phase 2 done: 11 forks at 1x are missed by >= 8 of 10 seeds,
> none at 2x; they are upright blocks pushed past their tipping angle, which the model under-rotates
> and lets settle back ([`blind_fork_analysis.md`](blind_fork_analysis.md)). Step 7 done
> (4 arms x 10 seeds): contact-window data alone does not remove the blind forks; adding a
> counterfactual loss on the same branches does. The intervention-consistency loss (D2,
> `--cw-loss intervention`) is the best-supported training arm. The calibrated spread analysis gets
> matched recall 70/80/84/91 at 1/3/5/10% FPR and AUC 0.974, but that is now a diagnostic, not the
> monitor. The deployed-rule candidate is calibration-free again: `src/counterfactual_monitor.py`
> requires a BIC-supported, Ashman-separated two-mode split to persist from hold 10 to hold 30.
> Across 10 D2 seeds it gets 50.4% fork recall, 7.8% quiet alarms and 16 robust blind forks at 1x.
> The next calibration-free formulation conditions directly on the sampled action errors and asks
> whether two smooth action-to-trajectory surfaces compress the futures better than one, after a
> split-search and graph-complexity charge (`src/action_branch_monitor.py`). The matched control is
> decisive: simulator block-pose trajectories get 95.2% fork recall / 15.9% quiet alarms, while ten
> D2 seeds get 75.4% mean recall / 33.0% quiet alarms (range 61.9-89.3% / 27.4-42.5%), with three
> robust-blind forks and twelve robust quiet alarms. The same probes, pose features, five times and
> rule are used on both sides. Therefore this formulation is sensitive to real branching, but D2
> both smooths true branches and manufactures spurious ones. It is not yet a validated safety
> filter, and its exhaustive GP reference implementation is not real-time. Details in NOTES.md and
> PLAN_NEXT.md.
> A stricter eight-fold held-out-probe variant was implemented and rejected on DEV: it cuts
> ground-truth fork recall from 100% to 60.9% while quiet alarms move only 30.3% to 29.2%. This rules
> out simple in-sample split overfitting as the main false-alarm cause; do not run/promote it on TEST.
> Adaptive midpoint refinement is more useful but still misses its gate: five bisections on three
> candidate boundaries retain all 23/23 DEV forks and reduce quiet alarms 27/89 -> 21/89
> (30.3% -> 23.6%). It is universal and calibration-free, but most quiet alarms also retain a local
> response plateau. TEST remains untouched; next diagnose what generic physical regime changes those
> retained cases represent before changing the alarm again.
> That generic audit is now done. Retained quiet alarms are physically non-null versus matched
> controls, but much weaker than forks: persistent contact branching 28.6% vs 91.3%, late contact
> branching 4.8% vs 87.0%, median final pose gap 0.011 vs 1.67 and peak velocity gap 0.28 vs 8.49.
> BIC detects branch existence, not consequence. Do not adopt the tempting post-hoc contact gate;
> the next universal formulation must require task-independent, multi-channel effect significance.
> The first such attempt is now complete and rejected on DEV. `src/action_branch_monitor.py` and
> `eval/jenga_shared_dynamics_ground_truth.py` remove each refined endpoint's static pose offset,
> then compare one pooled nonlinear pose/velocity transition law with two branch-specific laws over
> hold 1-10 and hold 11-30. It alarms on only **4/23 topple forks and 1/89 quiet states**. Even much
> looser structural votes cannot approach the frozen >=21/23, <=4/89 gate. This is a conceptual
> result: with a sufficient Markov state, both sides of a topple boundary still obey the same
> physical law; consequential branching need not mean different dynamics parameters. TEST remains
> untouched. Do not carry this rule to D2. The next formulation should measure persistent growth or
> irreversibility of the *counterfactual separation*, normalized internally to the applied action
> separation, rather than ask whether the governing law changes.
> That boundary + amplification + persistence experiment is also now complete on DEV
> (`src/consequence_monitor.py`, `eval/jenga_consequence_ground_truth.py`). Synthetic mechanisms
> were frozen first. On Jenga it gets **3/23 topple forks and 2/89 quiet alarms**, failing the recall
> gate. Boundary and persistence each pass all 23 forks; post-H=8 amplification passes only 3/23.
> Conversely it passes 9/17 nudge forks. Most target topples therefore become committed during the
> eight perturbed action steps and then persist/settle; they need not keep amplifying after the
> intervention. The next test must measure action-to-consequence amplification over the complete
> H=8 + hold trajectory while separately checking late commitment, not demand post-action growth.
> Keep the action-width scaling and leave TEST/D2 untouched until that ground-truth rule passes.
> The complete-trajectory version is now implemented (`eval/jenga_whole_trajectory_consequence.py`).
> Its passive null explains separation from cumulative action exposure; a searched event may begin
> during H=8 or the first five hold steps and pays a description-length charge. The same pair must
> pass boundary scaling, commitment and late persistence. It improves to **18/23 topple forks and
> 7/89 quiet alarms**, but misses both sides of the >=21/23, <=4/89 gate. All 23 forks pass boundary
> and persistence; 18 pass commitment. Of 27 initial quiet candidates, 8 pass commitment and 7 the
> full conjunction. This validates whole-trajectory timing but does not yet separate all meaningful
> commitments from real benign regime changes. TEST/D2 remain untouched. Next collect dense curves
> at every bisection level and test whether event timing/shape itself converges as action width
> shrinks; do not tune a BIC margin on these seven quiet cases.
> That multi-resolution experiment is complete and rejected. Dense replay at all six widths
> (original through 32x) requires commitment at the last three widths, onset agreement within one
> control step, convergent normalized curve shape, final persistence and the original boundary.
> It yields **2/23 topple forks and 1/89 quiet alarms**. Shape convergence alone reaches a >=2/3
> majority on only 6/23 forks (versus 1/27 initial quiet candidates); stable commitment reaches
> 13/23 (versus 1/27). Thus it gains specificity by discarding most real boundaries. A hybrid
> boundary need not have stable finite-resolution timing/shape as it is approached, and 32x is not
> demonstrably asymptotic. Do not weaken this rule on DEV. The best frozen result remains the
> one-resolution whole-trajectory rule at 18/23 and 7/89. The next decision is conceptual: either
> introduce a task-independent physical-severity notion and validate it across tasks, or treat all
> real committed regime changes as monitor positives and stop calling the seven quiet cases false
> alarms. TEST/D2 remain untouched.
> A prospective local-recoverability test is now also complete. From each final boundary endpoint,
> it applies neutral and +/- three one-sigma execution-error axes for five steps, returns to nominal
> for ten, and compares the two anonymous pose/velocity reachable sets at their own sampling
> resolution (`src/recoverability_monitor.py`). Result: **15/23 topple forks and 12/89 quiet
> alarms**. It fails both sides of the gate. Many harmless static/contact changes are genuinely not
> reversible by millimetre-scale local EE corrections, while 8 topple states have overlapping local
> reachable sets. AND with whole-trajectory commitment gives 12/23 and 5/89; OR gives 21/23 and
> 14/89. Recoverability is therefore neither necessary nor sufficient here. Do not tune correction
> radius on DEV. A generic severity representation now needs either explicit physical scale/control
> cost validated cross-task, or the project should adopt the conservative all-regime-change framing.
> **Plan decision:** adopt both as separate layers. Freeze the one-resolution whole-trajectory rule
> as **Regime Monitor v0**: it detects committed execution-sensitive physical branches, not failures.
> Put task cost in a separate intervention policy evaluated by failures prevented, completion,
> delay and intervention rate. Immediate work is: checksum/freeze v0, run its untouched Jenga TEST
> once, then build a low-cost reobserve/replan wrapper and carry the identical detector to pushing
> and insertion. Only afterward test rendered representations and D2. See the revised execution
> order and gates in `PLAN_NEXT.md`.
> **Regime Monitor v0 is now frozen.** Manifest:
> `results/jenga/regime_monitor_v0/manifest.json`; protocol SHA-256
> `4e1a20ca3c64bf18733315fe31df5850529027f8f48cda2c5f547ce43ecdc317`. Run
> `python eval/jenga_regime_monitor_v0_freeze.py verify` before any TEST, visual or world-model
> evaluation. Verification binds implementation files, explicit constants, DEV evidence/result,
> benchmark/cache inputs and simulator archive, and fails on any byte mismatch. Step 1 is complete;
> The one-time TEST run is complete; next is the fixed intervention-policy experiment above.
>
> **Commands.**
> `python eval/jenga_bench.py verify` -- check the frozen benchmark is intact.
> `python eval/jenga_bench.py eval --model results/jenga/<checkpoint>.pt` -- score a checkpoint
> (add `--probe results/jenga/v1_probe_px196_c4096.pt` for the DINO + proprioception pipeline).
> `python eval/jenga_blind_freq.py` -- cross-seed miss frequency per fork.
> `python eval/jenga_step7_compare.py` -- the Step 7 arm comparison.
> `python eval/jenga_calibration_free.py --model results/jenga/w6_cw_d2_s1.pt` -- runtime rule.
> `python eval/jenga_calibration_free_compare.py` -- aggregate its 10 D2 seeds.
> `python eval/jenga_action_branch_ground_truth.py` -- action-branch control on simulator states.
> `python eval/jenga_action_branch.py --model results/jenga/w6_cw_d2_s1.pt` -- matched D2 run.
> `python eval/jenga_action_branch_compare.py` -- aggregate the 10 matched D2 runs.
> `python eval/jenga_action_branch_cv_ground_truth.py --splits dev` -- rejected held-out-probe
> diagnostic (`results/jenga/action_branch_cv_ground_truth_dev.json`).
> `python eval/jenga_action_boundary_refine.py` -- adaptive midpoint refinement on DEV
> (`results/jenga/action_boundary_refine_dev.json`).
> `python eval/jenga_generic_regime_audit.py` -- full-trace generic physical audit on DEV
> (`results/jenga/generic_regime_audit_dev.json`).
> `python eval/jenga_shared_dynamics_ground_truth.py` -- rejected DEV-only shared-vs-branch dynamics
> test (`results/jenga/shared_dynamics_ground_truth_dev.json`); do not run TEST/D2.
> `python eval/jenga_consequence_ground_truth.py` -- rejected DEV-only boundary + amplification +
> persistence test (`results/jenga/consequence_ground_truth_dev.json`); do not run TEST/D2.
> `python eval/jenga_whole_trajectory_consequence.py` -- DEV-only whole-trajectory commitment test
> (`results/jenga/whole_trajectory_consequence_dev.json`); promising but gate failed, no TEST/D2.
> `python eval/jenga_multiresolution_consequence.py` -- rejected dense six-width convergence test
> (`results/jenga/multiresolution_consequence_dev.json`); no TEST/D2.
> `python eval/jenga_recoverability_ground_truth.py` -- rejected local corrective-reachability test
> (`results/jenga/recoverability_ground_truth_dev.json`); no TEST/D2.
> `python eval/jenga_regime_monitor_v0_freeze.py verify` -- required integrity gate for frozen
> Regime Monitor v0.
> `python eval/jenga_regime_monitor_v0_test_analysis.py` -- reproduce the descriptive analysis of
> the immutable TEST result; this cannot change alarms or the frozen rule. Do not rerun the TEST
> evaluator: its canonical output already exists and overwrite is deliberately forbidden.
> `python eval/jenga_intervention_v0.py verify` -- verify the prospectively frozen intervention
> experiment. Its canonical 100-episode run is complete and cannot be overwritten.
> `python eval/jenga_intervention_v0_analysis.py` -- descriptive paired analysis; the canonical
> analysis already exists and cannot be overwritten.
> `python eval/jenga_intervention_time.py` -- immutable-decision time-to-success replay; canonical
> output already exists and cannot be overwritten.
> `python eval/contact_benchmark_previews.py` -- regenerate pushing/insertion design-validation
> videos (not benchmark results).
> Training: `eval/jenga_w6_simple.py` (defaults = the D0 recipe; `--cw-data results/jenga/cw_data`
> `--cw-loss intervention` = D2+CW; add `--fast-branch` for any `--cw-loss branch` run).
> Contact-window data: `python eval/jenga_cw_data.py` (writes `results/jenga/cw_data/`; npz gitignored).
>
> Everything below this banner is the historical record.

> **Current handoff (2026-09-17, end of day).** The live line of work is the Stage 0-2b
> sequence at the END of this document, not the basin/dissent method described below. In one
> line: wobble the chunk 64 times with measured execution error, let the scene settle, and
> alarm if the endings split into separated groups that persist over time.
>
> Where it stands on a FRESH 140-state holdout, from image latents of rendered endings:
> with the arm converging to a shared hold pose, the two-group rule gets 96% recall on
> mixed-outcome states at 1x with 3% false alarms on unanimous-safe states; with the arm ending
> in each probe's own pose it falls to 78%, where the coarse-persistence multi-group rule gets
> 93% with 5%. Neither rule wins both regimes, and no rule may be chosen per case.
> **Nothing here is validated as a safety filter.** Stage 3 ran the same rules on world-model
> PREDICTED endings and FAILED: alarm rates on forks and on quiet states are equal within noise
> (0-12% either way), because the predictor collapses the branches (separation d' 0.60 predicted
> versus 4.85 real). Rollout fine-tuning was then done and did NOT fix it: validation rollout MSE
> fell 10x (2.095 -> 0.207) while d' moved 0.60 -> 0.68 and recall stayed at 0%. The blocker is a
> loss that rewards the average future, not an under-trained model.
> **Plain-language writeup of this whole line of work: [`JENGA_EXPERIMENT.md`](JENGA_EXPERIMENT.md).**
> **The plan for fixing the predictor is [`PLAN_WORLDMODEL.md`](PLAN_WORLDMODEL.md)**; its W0
> scoreboard (`eval/jenga_w0_response_curves.py`) replaces rollout MSE as the grading metric.
>
> **2026-09-18:** W1 (set-based losses) and W2 (discrete ending head) both failed their gates, and
> the privileged-state probe showed that perception is NOT the binding constraint for a one-shot
> ending predictor (W3 superseded; object tokens return as W5's deployable input). What is ruled
> out is a one-shot map from state to ending; the
> current phase is W5, the external plan's Phase 3 stochastic HYBRID world model: K smooth
> experts switched by an action-conditioned gate and a discrete regime latent, so a small action
> change can switch the dynamics instead of ramping across the boundary. The oracle single-expert
> baseline is done (spread AUC .98 at 2x, but predicts a new topple at 0/16 fork states); the
> build order, universality boundary, losses and Gate 3 are in PLAN_WORLDMODEL.md W5.
> Read that, then section 6's Stage 0-3 entries, then NOTES.md from the bottom up.

Written 2026-09-12. Self-contained brief for picking this up cold, human or agent.
Read this, then `MONITOR.md` (the method), then `NOTES.md` (the append-only log, newest at the
bottom). `PLAN.md` and `PLAN_DINOWM.md` are the phase plans with acceptance criteria.
`TOY_EXPERIMENT.md` is a plain-language writeup of the toy result with figures, and
`JENGA_EXPERIMENT.md` is the same for the Jenga execution-noise work -- start there if you want
the story before the detail.

---

## 1. What this is

A **runtime safety monitor** for robot manipulation policies. The target task is a Franka Panda
picking a block from a cluttered tabletop without toppling its neighbours (`~/wksp/dino_wm`,
`~/wksp/panda_express`, `~/wksp/diffusion_policy`). This repo is the **toy-system testbed** where
the method is developed and falsified cheaply.

### The claim — read this before anything else

**It detects that the policy is operating where small errors change the outcome. NOT that a failure
is coming.** The signal is that a *nearby alternative action reaches a different terminal state*.

Three consequences, all non-negotiable once the claim is fixed:

1. **Evaluate on PROXIMITY (margin < m), never on outcome.** On outcome the method scores AUC
   0.29–0.56 and loses to summing the forces (0.925). On proximity it wins. When a result looks
   bad, check first whether it is an outcome-task number being read as a proximity claim.
2. **The blind spot is declared scope.** Dissent counting cannot see an action that fails with
   CERTAINTY — if every probe agrees, k=0. Report it; do not patch it.
3. **Any alarm clause needing a basin named "failure" is supervision and is struck.** The monitor
   must drop onto a new task with no labels.

### The method in one block

```
ONLINE, every chunk:
  1. truncate the action chunk to a FIXED lookahead H   (never receding)
  2. 32 probes: a_j = a + eps * z_j * v,  ONE SCALAR z_j ~ N(0,1) per probe
  3. append a SETTLE TAIL of zero/hold action
  4. roll each probe through the world model, take the ENDING latent
  5. PCA -> assign to nearest supported attractor centroid, or NOISE
  6. k = number of KNOWN-BASIN probes NOT in the plurality
     coverage = fraction of probes assigned to a known basin (reported separately)
  7. ALARM if k >= 2.   No threshold. No labels. No calibration.

OFFLINE, once: discover the attractors with HDBSCAN on PREDICTED endings, PCA'd first
               (min_cluster_size = 5-10% of n). The count is never supplied. A probe
               landing in NOISE is excluded from the basin vote and reported as reduced coverage.
```

**Sizing rule — this replaces threshold tuning.** `E[k] = n * Phi(-m/eps)`, so the detectable
margin follows from the probe budget: **m\* = 1.33 eps at n=32**, 1.56 at n=50. Reach grows only
logarithmically in n, so raising eps is far cheaper than adding probes. Pick the margin you must
detect; solve for eps.

---

## 2. Where it stands

| track | status |
|---|---|
| **shPLRNN on the exact state** (toy) | done — AUC 0.808, prec 0.633, rec 0.905 |
| **DINO-WM on images** (toy) | known-only dissent rerun — chunk-label **AUC 0.869**, prec 0.889, rec 0.800, F1 0.842 |
| **Jenga** (the real task) | tail 5 selected; ordered delta-z cuts false alarms sharply but misses the <=5% real-control gate; stop before J5 |

**The current result:** HDBSCAN removes the per-task clustering choice and discovers the correct
three toy basins (93.7% agreement, 100% coverage). Excluding noise from the dissent vote restores
the runtime false-positive floor: on 100 episodes it scores AUC 0.869 on the preferred per-chunk
label, with precision 0.889, recall 0.800, F1 0.842, and 96-97% of large-margin chunks below alarm.
Against the old full-action label, its like-for-like AUC is 0.872 versus 0.882 for nearest-centroid
and 0.808 for the exact-state monitor. Mean known-basin coverage is 92.8% (median 100%); 29 of
800 chunks have no known-basin probes and therefore produce k=0 by definition, not evidence of
safety.

**The caveat that must travel with it:** this used a ROLLOUT FINE-TUNED predictor. dino_wm's
shipped `num_pred: 1` recipe gives 75% basin agreement and was NOT used. So this validates the
architecture, not the existing Jenga checkpoint.

---

## 3. Environment and how to run things

**Everything runs in the `dino_wm` conda env** (named after the sibling repo, but the toy pipeline
no longer depends on it -- see below). Versions are pinned in `requirements.txt`.

```bash
PY=/home/sanger/miniforge3/envs/dino_wm/bin/python
cd ~/wksp/stable_lyap_filters
$PY -m pytest tests/ -q            # 61 tests, ~12 s, all should pass
```

The current machine has an RTX 3090 with **24 GiB**. The monitor still chunks to <=128 to preserve
the validated execution path and remain portable to the original RTX 5060 Ti (7.5 GiB):
* batch 256 x 768 tokens needs ~9.6 GB for attention alone on the original machine.
* backprop through an unrolled 8-step ViT OOMs. Use gradient checkpointing.

### Pipeline, in order (DINO-WM track)

```bash
$PY eval/phase_c_data.py            # render + DINOv2-encode 600 eps  (~5 min, writes 5.3 GB)
$PY eval/phase_c_train.py           # teacher-forced predictor        (~8 min)
$PY eval/phase_c_train_gtf.py --tag gtf_warm   # rollout fine-tune    (~19 min)  <- REQUIRED
$PY eval/phase_d_settle.py          # settle + contraction tests      (~2 min)
$PY eval/phase_e_attractors.py      # attractor discovery             (~4 min)
$PY eval/phase_f_monitor.py         # the monitor, 100 eps x 8 times  (~34 min on RTX 3090)
$PY eval/phase_f_videos2.py --rescore  # select/rescore/render 10 demos (~14 min on RTX 3090)
```

### Jenga scripts

```bash
# --- world-model line (W0-W5, 2026-09-18); see PLAN_WORLDMODEL.md ---
$PY eval/jenga_w0_response_curves.py     # the scoreboard: response curves, jump ratio and spread
$PY eval/jenga_bulk_data.py --seeds 10   # 10x image data, only the 5 frames used (~25 min, 24 GB)
$PY eval/jenga_state_data.py --seeds 10  # 10x privileged state, no rendering (~10 min, 62 MB)
$PY eval/jenga_state_data.py --per-step --seeds 10   # + state after every action (W5 training data)
$PY eval/jenga_w1_train.py               # W1: set-based losses (FAILED its gate)
$PY eval/jenga_w2_discrete.py --bulk --data results/jenga/bulk_data   # W2a discrete ending head
$PY eval/jenga_w2_curves.py --head k64=<ckpt>        # W2 gate on the response curves
$PY eval/jenga_w3_privileged.py --data results/jenga/state_data --state-key start_state
$PY eval/jenga_w3_priv_curves.py --head full_state=<ckpt>   # privileged response curves
$PY eval/jenga_w3_monitor.py             # the monitor end to end on predicted endings

# --- current line of work (Stage 0-2b, 2026-09-17); see the banner at the top ---
$PY eval/jenga_stage0_noise_oracle.py    # 64 execution-noise runs/state -> answer key (~2.5 min)
$PY eval/jenga_stage1_outcome_modes.py   # physics fork test on those endings (CPU only)
$PY eval/jenga_stage1_split_images.py    # render the non-topple splits it flags
$PY eval/jenga_stage2_visual_forks.py    # same test on full-frame DINO latents (~9 min/55 states)
$PY eval/jenga_stage2_visual_forks.py --own-hold  # arm-movement check (own hold target)
$PY eval/jenga_stage2b_multimode.py      # multi-group + revised persistence on cached latents
$PY eval/jenga_holdout_select.py         # pick holdout states from a screen (rule in docstring)
# the holdout end to end: screen 1,168 chunks (~37 min, 30 workers), then select, render, score
$PY eval/jenga_stage0_noise_oracle.py --panel results/jenga/holdout_screen_panel.json \
    --cache results/jenga/holdout_stage0_cache.npz --output results/jenga/holdout_stage0.json \
    --workers 30   # --snippet-panel keeps the SAME 64 snippets; do not change it

$PY eval/jenga_j1_fidelity.py       # bundled model + LMDB, error vs horizon (J1)
$PY eval/jenga_j2_j3_predicted_basins.py  # nominal tail + predicted basins (J2/J3)
$PY eval/jenga_j2_j3_predicted_basins.py --reuse-cache  # reanalyse without GPU rollout
$PY eval/jenga_j4_hedging.py        # paired real/predicted horizon diagnostic (J4)
$PY eval/jenga_j4_hedging.py --reuse-cache  # reanalyse without encoding/rollout
$PY eval/jenga_short_held_tails.py  # 100-episode simulator branches: H=8 + tail 0/5/10
$PY eval/jenga_short_held_tails.py --reuse-cache  # reanalyse cached 2,049 chunks
$PY eval/jenga_local_delta_probes.py  # 100-chunk paired local delta-z diagnostic
$PY eval/jenga_local_delta_probes.py --reuse-cache  # reanalyse from compact raw cache
$PY eval/jenga_ordered_change_points.py  # ordered continuity test; real gate before predictions
$PY eval/jenga_physical_jump_diagnostics.py  # attribute real-latent alarms to simulator state
$PY eval/jenga_quiet_alarm_inspection.py  # paired renders/robot pose for five quiet alarms
$PY eval/jenga_ordered_holdout.py  # frozen 4x on temporally spread unseen episodes
$PY eval/jenga_ordered_holdout.py --reuse-cache  # re-score held-out real latents
$PY eval/jenga_all_block_relabel.py  # re-label cached probes for middle + neighbors
$PY eval/jenga_all_block_tails.py  # 2,049 chunks, all blocks, tail 0/5/10
$PY eval/jenga_all_block_tails.py --reuse-cache
$PY eval/jenga_spatial_latents.py --reuse-cache  # balanced eps=.10 representations
$PY eval/jenga_spatial_latents.py --eps .30 --reuse-cache --cache results/jenga/spatial_latents_eps03_masked_cache.npz --output results/jenga/spatial_latents_eps03.json
$PY eval/jenga_neighbor_validation.py --reuse-cache  # frozen neighbor-only validation/audit
$PY eval/jenga_near_boundary_discovery.py  # broad locator -> centered operational families
$PY eval/jenga_multipeak_oracle.py --reuse-cache  # physical/real/predicted multi-peak gate
$PY eval/jenga_trajectory_oracle.py --reuse-cache  # 13-time physical expansion/persistence gate
$PY eval/jenga_visual_trajectories.py --reuse-cache  # 13-time real DINO and oracle-mask audit
$PY eval/jenga_expand_controls.py --reuse-cache  # complete silent episodes and moving-safe negatives
$PY eval/jenga_geometry.py          # do toppled/intact separate? raw vs PCA   (~2 min)
$PY eval/jenga_basins.py            # bimodality + arm removal, the key result (~2 min)
$PY eval/jenga_linkage.py           # linkage comparison + contact sheets      (~2 min)
$PY eval/cluster_shootout.py        # HDBSCAN vs the rest, BOTH systems        (~5 min)
```

Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for anything that trains or rolls out.

### The Jenga bundle (built and verified 2026-09-14)

`jenga_bundle_live.tar`, **2.0 GB**, holds everything the Jenga work needs that git cannot carry.
Rebuild with `./scripts/bundle_jenga.sh --live`.

```bash
git clone git@github.com:sanger640/stable_lyap_filters.git && cd stable_lyap_filters
sha256sum -c data/jenga/BUNDLE.sha256      # verify the transfer
tar -Sxf /path/to/jenga_bundle_live.tar    # -S IS REQUIRED, see below
./scripts/setup_env.sh
source .venv/bin/activate
python -m pytest tests/ -q                 # 61 tests
python eval/jenga_j1_fidelity.py --episodes 10
python eval/jenga_j2_j3_predicted_basins.py
```

**Extract with `-S`.** The LMDB's `data.mdb` is **20 GB apparent against 1.1 GB of real blocks** --
LMDB preallocates its map file. Without `-S` you write 20 GB to disk for 1.1 GB of data. (The same
trap made the first bundle 21 GB until `tar -S` was added on the build side.)

Verified by extracting to a clean directory: LMDB opens with 100 episodes, world model loads at
epoch 88, label keys match the LMDB episodes exactly, the sim tar contains
`panda_jenga_setup.xml`, policy checkpoint intact.

### Moving to another machine

Everything needed for the TOY work is in this repo. Verified-working versions are pinned in
`requirements.txt` (python 3.11.15, torch 2.11.0+cu128, numpy 2.4.6). **torch must be a cu128
build** -- the GPUs used here are Blackwell (sm_120) and cu121 wheels will not run.

```bash
git clone git@github.com:sanger640/stable_lyap_filters.git && cd stable_lyap_filters
python -m pytest tests/ -q                       # 61 tests, ~12 s -- verifies the port
```

**The repo is self-contained for the toy work.** `ViTPredictor` -- the only thing ever imported
from `dino_wm` -- is vendored at `src/models/dinowm_vit.py`. You do NOT need the dino_wm repo to
run phases A-F. (It was previously behind a hard-coded `sys.path` entry pointing at
`/home/sanger/wksp/dino_wm`, which made a fresh clone fail immediately.)

**Training from scratch is the expected path** and takes ~32 min end to end:

```bash
python eval/phase_c_data.py                      # render + encode 600 eps   ~5 min
python eval/phase_c_train.py                     # teacher-forced            ~8 min
python eval/phase_c_train_gtf.py --tag gtf_warm  # rollout fine-tune         ~19 min  <- REQUIRED
```

The trained predictors are also recoverable from git history if you ever want to reproduce exact
numbers rather than retrain -- they were tracked until `b48efbc`, so a full clone carries the
blobs: `git show b48efbc^:results/phase_c/predictor.pt > results/phase_c/predictor.pt`.

**Deleted on 2026-09-18 to free 51 GB, all regenerable:** `results/jenga/w1_data`,
`w2_data_phase1`, `w2_data_combined`, `gtf_data` (full-frame rollout data, `eval/jenga_gtf_data.py`,
~20 min each), `visual_trajectories_cache`, `j4_truth_and_short_predictions.npz`,
`j2_j3_predicted_endings.npz`, the 453-state W2 heads and `w2_setup.npz`. Kept: `bulk_data` (24 GB,
the 10x image latents), `state_data`, and the W1/GTF checkpoints as documented negative baselines.

**Regenerate, do not copy**, the large caches -- `results/phase_c/latents.npy` (5.3 GB, ~5 min via
`phase_c_data.py`), `results/phase_a/feat_*.npz` (13 GB), `phase_e_endings_*.npz`.

**Reproduce one known result to confirm the port**: `eval/phase_e_attractors.py` should find a
k=3 on the toy at 93.7%, ~4 min.

**For the JENGA work only** -- not needed for the toy -- you additionally need assets from outside
this repo. **The inventory below was verified on 2026-09-14 and corrects RESUME.md, which claims
the checkpoint, the LMDB and the labels are all missing. None of them are.**

| what | where | size | state |
|---|---|---|---|
| world model checkpoint | `dino_wm/outputs/model_latest_single.pth` | 360 MB | **exists** (epoch 88, single-view, 196 patches) |
| eval LMDB (smallest usable) | `panda_express/tasks/jenga_noise_50/jenga_single.lmdb` | 534 MB | **exists**, 8837 entries, opens clean |
| ground-truth labels | `panda_express/labels_noise100.json` | tiny | **exists**, 100 eps: 75 success / 25 failure |
| ground-truth labels (smaller set) | `panda_express/labels_noise50.json` | tiny | **exists**, 50 eps: 38 / 12 |
| the `dino_wm` repo | `~/wksp/dino_wm` | — | needed for its model/config code |
| 102 raw episodes (only if rebuilding) | `panda_express/tasks/jenga_mujoco/episodes/` | 2.4 GB | dual-cam PNGs + `trajectory_*.json` |

**So the minimum transfer for the Jenga go/no-go is ~1 GB** (checkpoint + one LMDB + labels), not
the ~27 GB of Jenga task data on disk, and **no rebuild step is required.** The raw episodes are
only needed if you want to regenerate an LMDB with different preprocessing.

Four other LMDBs exist and all open clean, if you need more or cleaner data:
`jenga_mujoco/jenga_single_clean.lmdb` (1.3 GB, 16903 entries),
`jenga_noise_50/jenga_single_100.lmdb` (1.1 GB), `jenga_noise_50/jenga_unified.lmdb` (6.8 GB),
`jenga_tilt_100/jenga_tilt.lmdb` (944 MB).

The labels carry more than a boolean -- `outcome`, `failure_step`, `peak_tilt_deg`, `failure_block`
-- at a 45 deg topple threshold. `peak_tilt_deg` is the useful one: it is a continuous
near-miss measure, which is much closer to the PROXIMITY quantity this method actually predicts
than a success/failure flag is. Median peak tilt is 12.7 deg against a 45 deg threshold.

### Disk

`results/phase_a/feat_*.npz` (13 GB) and `results/phase_c/latents.npy` (5.3 GB) are gitignored
caches. **Both are regenerable** — delete freely if you need the space; `phase_c_data.py` rebuilds
the latter in ~5 minutes.

---

## 4. Findings worth carrying to Jenga

**1. dino_wm's single-step training recipe is inadequate for a monitor.** `conf/train.yaml` has
`num_pred: 1`. One-step prediction is excellent (0.040 rad, at the readout's floor) and compounds
to 0.577 over 40 autoregressive steps, because the model never sees its own output as input. Fixed
by rollout fine-tuning — but **order matters**: teacher-force first to learn the dynamics, THEN
fine-tune on rollouts. Reversed, it collapses to predicting that nothing ever happens
([1,59,0] against a true [15,28,17]).

**2. MSE hedges at bifurcations.** MSE's minimiser is the conditional mean; near a boundary the
mean of a bimodal future sits BETWEEN the basins, which with attractors at fall-left / upright /
fall-right is "upright". Predictions compress toward the middle attractor. **Diagnostic: compare
predicted vs true basin COUNTS.** If the Jenga model's predicted terminal states are compressed
toward "nothing toppled", the same thing is happening there.

**3. The same MSE property is an ASSET against nuisance.** The predictor replaces unpredictable
components (shadow pixels, render noise) with their mean, tightening clusters 21% without moving
them closer. One mechanism: liability at decision boundaries, asset against appearance.

**4. Clustering needs PCA in patch-token space.** Single-linkage returned k=300 of 300 in the raw
98,304-dim space — and so did the control on encoded truth, which is what ruled out the model.

**6. JENGA'S BASINS ARE REAL AND DISCOVERABLE — but only after removing the arm.** Verified on
100 labelled episodes, real final frames, encoder only. Peak neighbour tilt is sharply bimodal (75
episodes under 19 deg, 25 above 90, NOTHING between -- a 72.2 deg gap). PC1 of the raw latents is
end-effector pose, not the blocks, so **regress proprio out first** -- it uses no outcome labels,
and without it k-means at k=2 sits at CHANCE (50%). With it: separation 1.50 -> 2.15,
nearest-centroid 94% -> 98-99%, correlation with peak tilt 0.55 -> 0.77. HDBSCAN then finds k=2 at
98.8% with nothing supplied.

**What this does NOT establish:** those are REAL frames. The monitor reads PREDICTED endings, and
the settle-tail problem below is still open.

**5. The settle tail HOVERS rather than freezing, and that is fine.** `||z_t+1 - z_t||` decays then
flatlines at 0.60% of scale -- it never reaches zero. That was recorded as a failure until the right
question was asked: the monitor reads only WHICH attractor is nearest, so what matters is whether
the LABEL is stable. Measured over 120 episodes it is 99.2% stable by settle step 25 and **100% from
step 40**. A hover INSIDE a basin, not a drift ACROSS basins. It is also physically faithful -- a
rocking block loses speed at each impact (e_r = 0.824) and approaches upright geometrically without
exactly arriving.

**Acceptance for any settle test should be label stability, not `||dz|| -> 0`.** The original
criterion measured a quantity the method does not use.

---

## 5. Traps that cost real time here

* **Check the CONTROL before blaming the model.** Phase E looked like a model failure until the
  same discovery run on encoded truth failed identically.
* **A measurement that fails the EASY case cannot be trusted on the hard one.** Phase B's first
  verdict was FAIL; the tell was theta (directly visible) scoring 0.856. Two probe bugs, not a
  representation limit.
* **Label per (state, chunk), never per episode.** The monitor asks "from here, does THIS chunk
  straddle a boundary?". A whole-episode margin answers something else.
* **Score densely when MEASURING.** k spikes and decays. 3 scoring times gave recall 0.611; 8 gave
  0.820. Deployment rate (one score per action chunk) and evaluation rate are different questions.
* **`auc()` in `eval/run_phase3_block.py` uses ordinal ranks, not average ranks** — it misreports
  tied integer scores like k (0.804 vs the correct 0.808). Unfixed.
* **Verify every patch, and chain the run behind `&&`.** Silent no-op patches followed by a launch
  on the stale file cost three runs in one session.

---

## 6. Historical Jenga handoff snapshot (superseded)

This section records the early encoder/world-model handoff and is retained for provenance. It is
not the active plan; see the document header and `PLAN_NEXT.md`.

### The Jenga go/no-go is PARTLY DONE. What is settled:

| check | status |
|---|---|
| do toppled/intact scenes separate in DINOv2 space? | **YES** — 98-99% nearest-centroid, separation 2.15 (`eval/jenga_geometry.py`) |
| are the basins real, or a continuum? | **REAL** — peak tilt bimodal, 72.2 deg gap, 0 episodes in 20-60 deg |
| can the count be found with no labels? | **YES** — HDBSCAN finds k=2 at 98.8% (`eval/jenga_linkage.py`, `eval/cluster_shootout.py`) |
| does latent distance track HOW FAR it tipped? | **YES** — r = 0.77 with `peak_tilt_deg` |

**All of that used REAL final frames and the encoder only.** The first predicted-ending run below
shows that the checkpoint does not preserve this structure.

### 1. Diagnose/fix the predictor before J5

The method reads the ending after "stop acting and let the scene settle". **Jenga demos contain no
held poses at all**: over 6771 steps of demo actions the median per-step EE displacement is 5.3 mm,
only 0.6% of steps move under 1 mm, and the longest run of consecutive sub-millimetre steps is
**ONE**. So a zero-action tail is pure extrapolation, and asking "does it settle?" would be
answered by OOD behaviour rather than physics.

Ruled out already: **dropping the tail entirely**. Measured on the toy against the fully-settled
outcome, tail=0 finds the WRONG count (2 instead of 3) and scores 74.3% against a 70.7% baseline.

The first nominal-continuation run used all 100 episodes and remained finite for 80-330 tail steps.
It did not yield a robust stable assignment: 2-D/5% HDBSCAN changed from five to three clusters
between the 90% and full checkpoints and retained only 61.8% of joint known assignments. Some
higher-dimensional settings reached 93-96% late agreement, but only on 55-72% joint coverage and
with memberships that do not match the physical outcome. No tail is accepted yet.

### 2. Predicted-ending structure — run 1 failed

Everything above is the encoder on real photographs. The monitor reads the world model's PREDICTED
endings, which carry error. Phase C predicts trouble: the Jenga checkpoint uses dino_wm's shipped
`num_pred: 1` recipe, which on the toy gave 75% basin agreement and hedged toward "nothing
happened".

At the full nominal tail, 2-D/5% HDBSCAN finds k=3, 67.7% agreement, 96% coverage, and separation
1.02. The 2-D/10% setting finds k=2 but reaches only 74.1% agreement, 85% coverage, and separation
1.02. No setting in the 2/4/8/16-PC by 5/10% sweep meets the >=90% agreement and >=1.5 separation
gate. Its two clusters mix outcomes (54 intact / 12 toppled and 10 intact / 9 toppled), so this is
not just a mildly compressed 75/25 count.

The full-trajectory J4 stress test confirms general long-rollout failure rather than simple count compression. Real separation grows
from 1.06 at H=8 to 2.21 at 75% of the nominal continuation; predicted separation remains near
1.0. At the end, the predicted outcome axis has only 38.7% of the real magnitude and cosine 0.40
with the real direction. Nearest-real-centroid counts are 72/28, deceptively close to 75/25, while
episode accuracy is only 67%. Rollout-fine-tune the existing checkpoint, then rerun J1-J3; do not
start the monitor while J3 fails.

The corrected chunk-local simulator test changes the immediate diagnosis. Across 2,049 H=8 chunks,
tail 0/5/10 reveals 36/54/57 newly initiated topples from 1,702 upright starts, so tail 5 is the
physical knee. Short-horizon predicted geometry remains close to real geometry (tail-5 separation
2.41 vs 2.45; outcome-axis cosine 0.99). But only 11.6% of predicted and 12.8% of real tail-5
endpoints are supported by the terminal basin model. The blocker is now how to represent mixed
mid-trajectory endpoints without silently turning terminal attractors into context clusters.
Because the LMDB lacks MuJoCo state snapshots, action replay uses deterministic re-sampled reset
variation; it matches original episode outcomes at 79%. Tail comparisons are paired and causal,
but the simulated absolute failure rate is not the original dataset's rate.

The first proposed fix—cluster local endpoint changes `delta-z` per chunk after subtracting a
quadratic response to probe magnitude—also fails. On a stratified 100-chunk x 50-probe simulator
test, it finds all 10 physical boundaries but false-alarms on 84/90 non-boundaries. More decisively,
the encoded-real control alarms on all 100 chunks. HDBSCAN is partitioning ordinary smooth
nonlinear response geometry, not detecting discontinuity. Retraining DINO-WM cannot fix a detector
that fails on real latents. The next formulation must use the probes' one-dimensional ordering to
test continuity directly rather than clustering an unordered point cloud.

That ordered formulation is now implemented. A global cubic and a continuous piecewise cubic are
compared with an otherwise identical piecewise cubic that permits a jump. BIC penalizes model
complexity and the split search; a candidate also needs an adjacent latent increment at least 3x
ordinary increments. On the same encoded-real 100 chunks it detects 7/10 physical boundaries and
produces 8/90 non-boundary alarms (precision 46.7%, recall 70%, F1 0.56). Quiet-control alarms drop
from 50/50 to 5/50. This is a major correction, but 10% misses the predeclared <=5% gate;
PCA2/8/16 also miss at 6-8%. Predicted curves were therefore not scored. Before tuning the jump
ratio, cache continuous peak tilt, block pose, and contact state per probe; some apparent false
positives contain abrupt encoded-real changes and may be sub-threshold physical events. Results:
`results/jenga/ordered_change_points.json` and `results/jenga/ordered_change_points.png`.

Physical attribution is now complete. Per-probe peak/end tilt, all block poses, transient and
endpoint contact signatures, and end-effector position were cached from the same deterministic
branches. Only one of the five quiet alarms has a material neighboring-block change: ep67/chunk66
moves a side block 2.48 mm and rotates it 2.01 degrees without reaching 45 degrees. None changes a
contact category. Ep99 moves the manipulated middle block 0.33 mm; the other three have effectively
fixed blocks and only tiny robot/raster changes. The coarse topple label therefore explains one
residual alarm, not the detector's remaining false-positive floor. Do not retroactively call 4x a
passing threshold on these controls, even though it would numerically leave 2/50 alarms. Freeze it
and test new controls, preferably together with a spatial readout restricted to neighboring blocks.
See `results/jenga/physical_jump_diagnostics.json` and
`results/jenga/quiet_alarm_inspection.png`.

The frozen 4x validation is also complete. It uses 50 new chunks from the 25 episodes absent from
the original probe diagnostic, with two temporally spread points per episode. All were physical
non-boundaries; 3 alarmed, for 6% versus the <=5% gate (exact 95% interval 1.25-16.55%). An initial
selector bug used only action starts 2 and 10 and produced 2/50; that easy-prefix result is invalid.
On the original mixed set, 4x retains 7/10 real boundary detections with 3/90 false alarms. A model
score was exposed after the flawed preliminary pass: predictions detected 0/10 boundaries and made
4 false alarms. Because the corrected external gate fails, this is exploratory rather than formal,
but it strongly warns that DINO-WM smooths away real jumps (predicted adjacent ratios are <=2.41 on
all ten boundary chunks). Next test a neighboring-block-only spatial latent on real endpoints before
reopening prediction scoring. See `results/jenga/ordered_holdout.json` and
`results/jenga/ordered_change_points_4x.json`.

**Correction and extension: the middle block is now part of failure.** Failure is either neighbor
at 45 degrees, middle at 45 degrees, or middle COM below the table; safe upright extraction is
allowed. Re-labeling leaves the original union at 10 boundaries (7 detected), but exposes two
middle-specific edge transitions with only 5 and 2 safe probes—outside the fitter's minimum-eight-
per-side support. The 2,049-chunk all-block tail scan finds 48/69/77 new failures from safe starts
at tails 0/5/10. Tail 10 adds eight over tail 5, including six middle topples, so it is preferred if
the five extra rollout steps meet the live budget.

The balanced 80-scenario experiment uses 20 each of middle failure, neighbor failure, safe
extraction, and quiet control. At eps=.10 there are no supported middle boundaries. At eps=.30
there are three supported middle and six supported neighbor boundaries (seven in the union).
Full-frame detects 6/7 union and 1/3 middle; a valid simulator-oracle all-three-block mask detects
5/7 union and also 1/3 middle. Both produce 0/20 quiet alarms. Spatial masking therefore does not
solve middle sensitivity. The originally drawn red fixed ROI misses the blocks at some stages and
is invalid; only the green segmentation-derived mask is a valid upper-bound test. The next blocker
is a middle-relevant probe/response signal and more naturally near-boundary middle data, not crop
tuning. Results: `results/jenga/all_block_relabel.json`, `results/jenga/all_block_tails.json`,
`results/jenga/spatial_latents.json`, and `results/jenga/spatial_latents_eps03.json`.

**Expanded neighbor-only validation fails (2026-09-15).** With middle events excluded, the frozen
H=8/tail-5/eps=.10/PCA4/cubic/min-side-8/4x detector was tested on 250 temporally spread quiet
states absent from both earlier probe caches. It alarms on 16/250 = **6.4%** (exact 95% CI
3.70-10.19%), so the <=5% real-control point gate still fails. An exhaustive physical screen of
all 1,302 remaining eligible states found only **11** supported neighbor boundaries, not the
planned >=30. On all 11, encoded-real recall is **5/11 = 45.5%** (exact 95% CI 16.75-76.62%): the
>=80% recall gate also fails. Seven of the 11 physical outcome sequences switch more than once as
probe strength increases, so the assumed single-change-point structure is often false.

The paired world-model audit is conclusive for this checkpoint: DINO-WM detects **0/11** supported
boundaries and makes 5/250 quiet alarms. Its median adjacent-jump ratio on boundary states is 0.81
versus 6.64 for real encodings; the median predicted/real ratio is **14.8%**, and real/predicted
jump-ratio correlation is -0.075. Low predicted FPR is smoothing, not success. Do not deploy or
threshold-tune this monitor. The next defensible choices are (a) collect more naturally near-
boundary trajectories and train a counterfactual/multimodal model with a local-jump objective, or
(b) replace image-latent bifurcation detection with an object-pose/physics or supervised neighbor-
topple risk model. Results: `results/jenga/neighbor_validation.json`; the paired cache and physical
screen are gitignored.

**Multi-peak physical-oracle gate (steps 1-4, 2026-09-15):** a sliding local expansion detector
tests every adjacent probe pair with a continuous piecewise-linear null versus the same local fit
plus a step. BIC pays for the step and a `2 log(K)` search penalty; positive evidence is the fixed
alarm rule. Adjacent positive locations are grouped into multiple peaks, so the score no longer
assumes one global transition. A constant-response invariant is essential: otherwise BIC compares
round-off-sized residuals and invents evidence in deterministic states.

To obtain enough boundary cases without widening the evaluated perturbation, 300 untouched states
were screened with 21 broad eps=.30 probes; the nearest transition recentered a fresh 50-probe
eps=.10 family. Sixty broad families crossed a transition and **59** recentered families retained
at least eight outcomes per side. The frozen evaluation uses 100 untouched quiet controls plus 30
episode-spread recentered boundaries. Labels construct and evaluate this stress set only; the
score does not use them.

After correcting the physical vector to exclude three middle-only contact channels, endpoint
neighbor position/orientation/contact deltas produce **26 TP / 0 FP / 4 FN / 100 TN**: 86.7%
recall (95% CI 69.3-96.2%), 0% FPR (upper 95% bound 3.62%), F1 .929, AUC .868. This passes both
physical-oracle point gates; 14/30 boundaries still switch physical outcome more than once. On
exactly the same probes, real DINO gives 30/30 recall but alarms on 100/100 controls
(AUC .599), proving the representation injects local visual variation unrelated to block state.
DINO-WM detects 12/30 and alarms on 49/100 controls (AUC .469). This initially suggested a
representation bottleneck; expanded hard negatives below show that the physical score also needs
revision before representation training. Results:
`results/jenga/multipeak_oracle.json`; raw discovery and paired caches are gitignored.

**Trajectory-level physical oracle — PASS:** the same 130 states were rerun while recording the
neighbor-only physical vector after each of H=8 actions and all five held steps. A local expansion
score is computed independently at each of 13 times, then combined as
`2 log(mean(exp(evidence_t / 2)))`; this is the BIC/Bayes-factor evidence for expansion at an
unknown time and automatically penalizes temporal search while rewarding persistence. The fixed
zero-evidence alarm gives **30 TP / 0 FP / 0 FN / 100 TN**: precision, recall, F1, and AUC all 1.0.
The endpoint from the same cache reproduces 26/30, proving the four endpoint misses contain earlier
transient expansion. Detected boundaries have positive evidence at a median 6.5/13 times. This
passes only the constructed boundary/static-control benchmark; expanded hard negatives below
invalidate a general gate-pass claim. Results:
`results/jenga/trajectory_oracle.json`.

**Visual trajectories and expanded controls (2026-09-16) — generalization FAIL:** all 13 real
frames were rendered and DINO-encoded for the same 100 quiet controls and 30 recentered boundaries.
Frozen full-frame PCA4 yields 30/30 detections but **100/100 quiet alarms** (AUC .489). Even a
privileged simulator-segmentation neighbor-patch pool, reduced by unlabeled transductive PCA4,
yields 30/30 and **100/100** (AUC .331). All neighbor masks were visible. On quiet states, false
visual evidence starts by time 2 in 97/100 full-frame cases and lasts a median 12/13 times; the
paired physical score is zero. Block-pixel pooling alone does not remove action/raster/occlusion
nuisance. Results: `results/jenga/visual_trajectories.json`.

The physical dataset was expanded with **all 38 chunks** from the only two episodes whose nominal
neighbor tilt stays <5 degrees throughout, plus 100 episode-spread nominal moving non-topples
(neighbor displacement >=2 mm, nominal tilt 5-45 degrees). All 50 operational probes were
re-simulated. The trajectory oracle alarms on **5/38** safe silent chunks (13.2%). Among moving
examples, 88 remain safe under all probes, six have supported mixed outcomes, five have low-
support mixed outcomes, and one has certain failure. It alarms on **87/88** verified moving-safe
chunks (98.9%). Pose-only and position-only ablations still alarm on 88/88 moving-safe chunks, so
discrete contact features are not the sole cause. These are alarms on outcomes safe *within the
operational probe range*, not proof that the states are far from a topple boundary. The old
static-control gate does not establish proximity specificity. Do not retrain the visual
representation or world model against this score yet. First measure each moving-safe state's
physical distance to a topple boundary with a wider adaptive probe search, then compare alarm
against actual margin without fitting to failure labels. Results: `results/jenga/expanded_controls.json`.

**No-alarm red-block picks (2026-09-16):** screened 2,049 nominal chunks for middle-block rise
>=2.5 cm and sideways travel >=2 cm, with no nominal neighbor topple. Of 67 candidates, 66 are
safe under all 50 operational probes and 55 of those produce no physical alarm. Three rendered
examples from different episodes show ~8 cm middle-block lift, 3-4 cm sideways motion, zero
neighbor tilt, and no alarm. This confirms that the physical score can remain quiet during a
successful pick; it does not settle whether alarms on moving neighbors indicate genuine proximity.
Search: `eval/jenga_pick_noalarm_search.py`, result: `results/jenga/pick_noalarm_search.json`;
videos: `results/jenga/pick_noalarm_videos/`.

**Grasp/contact clarification:** the carry-phase videos above do not show the critical approach.
Two further no-alarm clips start with the red block at table height between its neighbors; it
lifts 4.2/7.5 cm while the end effector is ~4.8/5.1 cm from the nearest neighbor body center.
These clips have no recorded robot-neighbor contact. In the 67 screened pick chunks, all 55
safe/no-alarm cases had no nominal robot-neighbor contact; the 11 safe/alarm cases all did.
A broader check of all nominal chunks with robot-neighbor contact and >=2.5 cm red-block lift found
20 candidates, 19 safe under all 50 probes, and **0/19 no-alarm**. Thus there is currently no
verified no-alarm *contacting-neighbor* successful pick in this tested set. Results:
`results/jenga/contact_pick_check.json`; zoomed near-grasp videos:
`results/jenga/near_grasp_noalarm_videos/`.

**Full-episode alarm audit:** nominally replayed all 100 episodes. Fifty satisfy the explicit
successful-pick proxy (after settling: red lift >=2.5 cm and lateral travel >=2 cm; neighbor peak
tilt <45 degrees). Every one of these 50 has at least one independently verified physical alarm
decision. Therefore no verified *entirely alarm-free* successful episode exists under this
criterion, even though 55/66 selected pick chunks themselves were quiet. Four complete episodes
were scored at every non-overlapping H=8 decision point and rendered start-to-finish. Episode 24
alarms at the pick chunk 106 despite 50/50 probes staying safe; episodes 74 and 91 show no alarm
at their lift chunks 106/98 but did alarm earlier during approach. The video overlay is the most
recent decision's result, held until the next decision; it is not a continuously recomputed alarm.
Code: `eval/jenga_full_episode_alarm_audit.py`, `eval/jenga_full_episode_noalarm_screen.py`,
`eval/jenga_nominal_success_scan.py`, `eval/jenga_full_episode_videos.py`; results:
`results/jenga/nominal_success_scan.json`, `results/jenga/full_episode_alarm_audit.json`,
`results/jenga/full_episode_alarm_videos/`.

**Physical basin/dissent revisit and action-strength check (2026-09-16):** 50 ordered probes,
H=8, eps=.10, and held tails 5/10/20 were re-simulated at 30 recentered boundary states and
every H=8 decision in held-out successful episodes 24 and 74. The unlabeled reference atlas uses
the original diverse physical endpoints plus the 30 boundary families, with episodes 24/74
excluded. PCA4/HDBSCAN finds 2/3/3 basins at tails 5/10/20. It alarms on 10/30, 20/30, and
19/30 boundary families, respectively, but on 0/19 and 0/18 episode decisions at every tail.
Boundary detection is optimistic because the same boundary families contribute to the atlas;
episode results are held out. Noise is excluded from dissent. Reference coverage is 72%/85%/85%,
but key pick decisions can be completely unassigned (ep24 chunk 106 at all three tails). Thus
quiet does not mean a verified stable basin. Tail 10 is the most informative tested short tail,
yet the current global-basin atlas is not an adequate general proximity monitor. Full nominal
videos with all three scores: `results/jenga/physical_basin_episode_videos/`; analysis:
`results/jenga/physical_basin_tails.json`.

The original eps=.10 grid reaches a maximum coefficient .2326 times the within-chunk command
span (e.g. 8.5/11.8/18.2 mm in three sample chunks). A cached central-subset sensitivity check
suggested fewer alarms with narrower reach but confounded reach with probe count. A controlled
20-state pilot re-simulated the same 50 probes at eps=.10/.035/.0165 (maximum coefficients
.2326/.0814/.0384), with H=8+5 and frozen physical BIC rule. All 10 selected moving-safe states
alarmed at every strength; boundary alarms stayed 10/10, 10/10, 10/10, while true mixed-topple
responses fell 10/10, 10/10, 8/10. Oversized probes therefore do not alone explain this score's
poor specificity, although this pilot is too small to settle the appropriate physical uncertainty
scale. On the **same matched pilot**, frozen physical-basin dissent alarms on only 2/10, 1/10,
0/10 boundary families as strength narrows, with 0/10 moving-safe alarms throughout; mean
moving-safe basin coverage is only 29%/21%/20%. Lowering strength does not rescue basin detection;
it increasingly hides the boundary while leaving many points unassigned. Results:
`results/jenga/action_strength_sensitivity.json`,
`results/jenga/action_strength_resim.json`.

**Nominal-centered nearest persistent branch margin (2026-09-16):** implemented
`eval/jenga_persistent_branch_margin.py`. A fixed 31-strength grid from coefficient -.6 to +.6
perturbs the original nominal 8-action chunk, then records physical neighbor endpoints after
5 and 10 held steps. A branch must differ from nominal at both tails and have the same alternative
at two adjacent strengths; the first sustained change gives a coefficient and commanded-motion
millimeter bracket. The evaluation-only physical oracle uses endpoint neighbor tilt >=45 degrees,
separate from the unlabeled detector. On 15 held-out boundary families, **13/15** oracle margins
are bracketed (median upper bracket 3.01 mm); the remaining two change only at the grid edge and
are unconfirmed. Among 20 previously operationally safe moving states, **1/20** has a bracketed
physical margin (4.36-5.45 mm) and three have edge-only candidates. None of 10 silent or 10
contacting-pick controls has a bracketed physical branch on this grid. These are selected cohorts
and one action direction, not population rates or a full robustness certificate.
The previously fitted trajectory/BIC score alarmed on 19/20 of these moving states despite only
one confirmed persistent topple branch on the wider grid; this strengthens the specificity
concern, but unseen directions or unsampled narrow branches remain possible.

The unlabeled physical-basin estimator produced **0/15** boundary margins: all 15 nominal
boundary endpoints were unassigned by the held-out global atlas (also 20/20 moving nominal
endpoints); mean grid coverage on boundary/moving states is only 7.7%/11.5% despite 82-84%
reference coverage. This directly identifies the atlas as unusable for nominal-centered margin
estimation. Do not reinterpret its `not_found`/quiet outputs as large margins. The next method
must compare local physical response regimes without relying on global terminal-cluster coverage,
then be evaluated against these frozen oracle brackets before visual translation. Results:
`results/jenga/persistent_branch_margin.json` (cached simulator arrays ignored by git).

**Label-free local persistent-branch test (2026-09-16) — FAIL on specificity:** implemented
`src/local_persistent_branch.py` and `eval/jenga_local_persistent_branch.py` on the frozen
55-state physical response cache. At each neighboring-strength gap, a continuous local spline
competes with the same spline plus a jump under BIC; the gap must favor a jump at both held tails
5 and 10. A 6D PCA projection is learned from unlabeled physical reference endpoints with all
test episodes excluded. There is no topple label, global basin label, or fitted Jenga alarm
threshold in the detector. Nevertheless it brackets **15/15** held-out boundaries *and*
**20/20** moving-safe states, **10/10** safe contacting picks, and **1/10** silent controls.
Detected upper-margin medians are only ~0.30/0.28/0.51 mm for boundary/moving/contact groups,
versus the independent physical-oracle boundary median of 3.01 mm. Matching an oracle case in
binary terms is misleading: median absolute upper-margin error on the 13 confirmed boundaries
is 2.54 mm. Persistent local model preference is still reacting to smooth/contact-driven motion,
not isolating a persistent outcome regime. Do not claim the label-free margin works. Next test:
refine candidate action intervals at multiple resolutions; a genuine jump retains finite response
separation as the strength gap shrinks, while an ordinary smooth response shrinks with the gap.
Freeze this failed run as a baseline before any revision. Result:
`results/jenga/local_persistent_branch.json`.

**Two-level action-resolution refinement (2026-09-16) — still fails specificity:**
`src/branch_scale.py` and `eval/jenga_branch_scale_refine.py` replay the frozen local-BIC candidate
gap at its midpoint, follow the half with the larger physical endpoint difference, and bisect
again. A one-parameter constant-separation model competes with a one-parameter
width-proportional-shrinking model, independently at held tails 5 and 10. No topple label or
Jenga-fitted alarm threshold enters the decision. On 46 frozen candidates, constant separation
wins for **11/15** boundary states (10/13 with independently confirmed persistent topple
margin), but also **14/20** moving-safe states and **9/10** safe contacting picks; the one silent
candidate is rejected. These are the verified numbers after replaying the original endpoints
alongside each new midpoint. Median retained quarter-width separation is ~0.86 boundary, ~0.84
moving, and ~1.02 contacting-pick, not the .25 expected for a locally linear smooth response. The
5- and 10-step endpoints are effectively identical for the contacting-pick probes (median raw
endpoint difference <1e-9), so requiring both tails is not independent evidence. Cached versus
freshly replayed endpoints sometimes differ by binary contact-feature flips (errors up to 2 in
27D physical state), but **pose-only** analysis on the same refinement intervals still retains
separation on 15/20 moving and 9/10 contacting-pick cases. This is not just a contact-channel
cache artifact. Tiny or nonmonotonic pose response can retain separation without a confirmed
topple branch. Ablation: `results/jenga/branch_scale_ablation.json`.
This refinement was rejected on the development panel; no fresh-holdout success claim. It also
refines only the nearest BIC gap, so rejecting that gap does not rule out a farther true branch.
Result: `results/jenga/branch_scale_refine.json`.

**Controller-scaled predictive consequence probe (2026-09-17) — specificity gate FAIL:**
`src/action_uncertainty.py` and `eval/jenga_tracking_uncertainty.py` fit a one-step linear
command-to-proprioception lag and residual covariance on 57 training episodes, excluding the
43 episodes used by the frozen 55-state panel. On 7,224 held-out samples, median/p90 residual
norms are 1.465/3.56 mm; the training residual radial q90 scale covers 92.3% of held-out
samples. This is an empirical proxy for execution variation, not a certified uncertainty bound.
`eval/jenga_predictive_regime_probe.py` probes three covariance axes at 17 strengths each,
perturbs H=8 commands, then holds the *same nominal target* for every probe. It records the
pose of all three blocks at H=8, H+5, H+10. An unlabeled PCA6 ordered continuous-vs-step fit
must pick the same neighboring strength gap at both tails. Frozen development-panel alarms:
9/20 moving non-topple, 3/10 contacting lifts, 9/15 selected boundaries, 1/10 silent.
The independent neighbor-topple oracle is mixed within this controller-scaled envelope on
only 8/15 selected boundary states, of which 5 alarm. Thus reduced probe reach explains some
misses, but the detector still confuses ordinary consequential motion/contact with the desired
regime transition. Safe controls may have non-topple regimes; these counts specifically reject
using this as a *topple-proximity filter*, not every possible general-regime interpretation.
Exact replay variation is ~1e-15, so deterministic repeats cannot supply a noise floor.
`eval/jenga_predictive_variation.py` injects eight held-out training-episode residual snippets
as a diagnostic proxy. The median detected adjacent-gap size divided by the median pairwise
injected-jitter response at tail 10 is 0.81 moving, 0.08 contacting, 0.75 boundary, and 7.14
for the single silent alarm. The candidate gap is often no larger than plausible execution
variation, but this proxy is *not* measured repeated execution. No failure label was used to
fit the detector; its fixed BIC/jump rules still contain structural hyperparameters, so do
not call it fully calibration-free. No untouched holdout was run after this failed panel gate.
Results: `results/jenga/tracking_uncertainty.json`, `predictive_regime_probe.json`, and
`predictive_variation.json` in the same directory. Large simulator caches are intentionally
git-ignored; rerun the scripts in that order with the Jenga bundle/LMDB installed.

**Next agent:** first audit whether an independently observable, task-general *future
consequence* can separate routine contact/motion from distinct outcome regimes within a
realistic uncertainty set. Acquire or model actual executed-action variation rather than
treating target/proprio lag residuals as hardware noise. Freeze a new decision rule before a
new episode-level holdout, explicitly report reach/censoring and abstention, and only then
attempt visual representation transfer or live runtime integration. Preserve the main goal:
failure-label-free, zero-shot proximity to changed future behavior; Jenga topple labels are
evaluation-only. Do not tune a Jenga-specific score to these 55 states.

**Stage 0 answer key under realistic noise (2026-09-17):** `eval/jenga_stage0_noise_oracle.py`
replays each panel state's chunk with 64 contiguous tracking-residual snippets from non-panel
episodes, at 0.5x/1x/2x, then a 30-step hold. No detector is scored. Topple labels are settled by
hold step 10 (100% agreement with step 30). Mixed outcomes (>=2 per side) at 1x/2x:
boundary 7/11 of 15, moving non-topple 2/8 of 20, contact lift 0/0 of 10, silent 0/0 of 10.
Contact and silent states are true negatives at every scale. The "boundary" and "moving" names
are not reliable labels under realistic noise, so Stage 1 must be graded against these
per-scale grades. Details: NOTES.md and `results/jenga/stage0_noise_oracle.json`.

**Stage 1 two-mode test on true endings (2026-09-17):** `src/outcome_modes.py` and
`eval/jenga_stage1_outcome_modes.py`. Recall on mixed states is 100% at every scale. False
alarms on unanimous-safe states: 9/52, 8/43, and 0/29 at 0.5x/1x/2x; contact-lift and silent
states never alarm. The false alarms are real non-topple splits (a neighbor slides 3.5-10 mm or
rests tilted in some runs and not in others). Ending spread alone is perfect on this panel
(AUC 1.0), so the panel cannot show the two-mode test adds anything. Open decision: do such
non-topple splits count as "different outcomes"? A harder benchmark with small-consequence
positives is also needed.

**Fresh holdout, 140 unseen states (2026-09-17):** 1,168 chunks in the 57 non-development
episodes were screened with the Stage 0 physics, then 60 topple-fork / 20 physical-fork /
60 quiet states were selected by a pre-declared rule. Three frozen rules were scored
(`eval/jenga_stage2b_multimode.py`, `results/jenga/holdout_stage2b.json`). Recall on mixed
states at 1x / false alarms on quiet states: with a shared hold, the PC1 two-group rule gets
96% / 3%; with the arm moving, it drops to 78% and the coarse-persistence multi-group rule gets
93% / 5%. The strict multi-group rule fails everywhere (17-30%). At 2x the coarse rule reaches
14% false alarms with the arm moving. No single rule wins both hold regimes; do not select per
case. Details in NOTES.md.

**Stage 2, universal image version (2026-09-17):** `eval/jenga_stage2_visual_forks.py` runs the
same split test on full-frame DINO latents of the rendered endings, using no object knowledge.
At 1x it alarms on 8/9 topple forks and 0/23 quiet states, and at 2x on 19/19 and 0/13. The
nudge forks mostly go unseen (1/8). The persistence check (same split at hold steps 10 and 30)
is essential: without it, 10/23 quiet states alarm on image-latent noise. Two alarms that looked
false are real forks on the grasped block (in ep25 c90 the red block drops in 4/64 runs). Next:
the same test on world-model *predicted* endings, then a fresh holdout. Details: NOTES.md.

### 3. Competing baselines on PROXIMITY

Never measured, and "same performance, no threshold" is the entire paper claim. Tuned-delta FTLE
(the predecessor), div_std (0.818, needs a fitted percentile), Mahalanobis/kNN on safe-trajectory
latents.

### 4. The warm-start ablation

GTF from scratch at MATCHED compute. "Warm start required" is currently a claim, not a result — the
successful run changed initialisation AND training volume together.

### 5. Matched-impulse actions (system-level universality)

The toy is shortcut on OUTCOME (`sum|a|` -> AUC 0.925) but NOT on proximity (best fitted baseline
0.764 vs the method's 0.808). Action pairs with matched total impulse differing only in timing would
force magnitude baselines to chance by construction.

---

## 7. Paper position, honestly

Toy-only is not a robotics paper — no reviewer accepts a rocking block as the evaluation for a
manipulation safety monitor. The defensible contribution is **deleting the threshold** plus the
**sizing rule** that states a detectable margin a priori, which nothing in the runtime-monitoring
literature does.

That literature is crowded (Sentinel CoRL'24, PATCH 2026, SAFE, Rewind-IL) but **every one of them
is a failure detector and every one needs calibration**, mostly conformal prediction with a
labelled calibration set. The proximity framing plus a counting rule is differentiated on both
axes. FTLE-for-safety is taken (arXiv 2508.15588) but offline, on the true environment, without
action perturbation.

Jenga is what makes it a paper. The toy makes it a methods contribution.

## Cross-task U1 executed — unchanged v0 does not transfer (2026-09-23)

The panel is frozen in `results/contact_benchmarks/u1/manifest.json` (`3f2386af...`). On untouched
TEST, v0 alarms on 0/12 consequential pushing forks and 0/11 insertion forks; quiet alarms are
3/108 and 0/109. Persistence is usually present, but the Jenga-derived autonomous-commitment clause
rejects the forks. Do not tune v0 from these TEST outcomes.

The matched intervention protocol is frozen in
`results/contact_benchmarks/u1/intervention/manifest.json` (`44e5e035...`). Its baseline generator
yields zero terminal failures in both tasks, so the wrapper's unchanged outcomes are inconclusive,
not a pass. It modifies two pushing chunks and no insertion chunks.

Immediate handoff: create a new version, never overwrite U1. First require success/failure coverage
after terminal settling using grading labels only. In parallel, develop a task-general
transition-law/contact-graph branch test on new DEV data; U1 shows that persistent autonomous
amplification after action exposure is not universal across quasi-static contact tasks. Reserve new
TEST episodes before choosing the rule.

```bash
python eval/contact_regime_u1.py verify
python eval/contact_intervention_u1.py verify
```

## Panda block push built (2026-09-24)

`src/systems/panda_block_push.py` replaces the abstract pushing prototype for future work. The
existing Panda operational-space controller physically pushes one upright Jenga block to a green
non-colliding goal. Success requires reaching the area without ever crossing 45° tilt. A low push
succeeds at 11.3 mm goal error; an excessive low push overshoots while upright; a 20 mm higher push
topples to 96.2° even though the fallen block ends in the goal; and a peel-away trajectory loses
contact and stops short. See `PANDA_BLOCK_PUSH.md` and
`results/panda_block_push/previews/`.

This is not a monitor result. Use it only as a compact shared-domain DEV mechanism panel while
fixing the ground-truth monitor and then the world model. Do not start another large intervention
benchmark yet.

That DEV panel has now been built and scored without changing v0. Protocol SHA is `20dc10ea...` and
the result is `results/panda_block_push/dev_panel/v0_stage_diagnostic.json` (SHA `5168479e...`).
Coverage is 20 mixed upright/topple forks, 15 safe centered, 10 unanimous topple, five contact-loss
and six overshoot states. Final v0 alarms are respectively 19, 0, 1, 5 and 0. The five contact-loss
alarms are real regime changes, not toppling false positives. All 20 mixed forks pass commitment and
persistence; the sole miss is boundary refinement. Next compare physical, rendered-feature and
world-model-predicted decisions on these exact frozen trajectories. Do not tune v0 or run another
large rollout first.

```bash
python eval/panda_push_dev_panel.py verify
```

The first rendered-future transfer is also complete. `eval/panda_push_visual_v0.py` renders all
56 x 64 exact initial futures at hold 5/10/20/29/30 and caches visual-selected refinement and dense
endpoint futures. Its fixed baseline adapter is frozen DINOv2-S/14, 4x4 spatial pooling and a
seeded 45-D Gaussian projection; it fits no threshold or representation on labels/calibration
states. Frozen v0 agrees with physical-v0 final decisions on only 33/56 states. Visual recall is
8/25 physical alarms overall and 6/19 mixed upright/topple physical alarms, with six visual-only
alarms. Every state becomes an initial visual candidate and 54/56 survive refinement/persistence;
commitment majority falls to 16/56. This is a representation failure before world-model prediction,
not evidence that the physical monitor failed.

Do not run the Panda world model next. Run two readout-only DEV diagnostics on the same cached true
frames: fixed background-suppressed DINO patches, then simulator-segmented block pooling as an
oracle diagnostic upper bound. Judge agreement with physical-v0 stages, never topple labels. If one
works, freeze it and reserve a fresh Panda TEST panel before predicted-future evaluation. The 45-D
adapter is only a compatibility bridge to v0's semantic physical-state channel contract; a later
universal visual consequence formulation should consume visual geometry directly.

```bash
.venv/bin/python eval/panda_push_visual_v0.py verify-render
.venv/bin/python eval/panda_push_visual_v0.py run --device cpu --batch-size 16
```

Result: `results/panda_block_push/visual_v0/visual_v0_diagnostic.json`, SHA
`8a84de40e877bc3f3adab33f74b2d42d067c1ab96af61d62afb27f1f43e4bb33`.

The direct PCA check is now complete in `eval/panda_push_visual_pca.py`. Per-state PCA45 is fitted
only to each current state's 64 unlabeled rendered probe trajectories and then held fixed for its
newly selected bisections/endpoints. Compared with random projection, agreement improves 33/56 ->
37/56, all-physical-alarm recall 8/25 -> 11/25, and mixed upright/topple recall 6/19 -> 10/19.
Mixed-fork commitment majorities improve 8/20 -> 16/20. This establishes that random projection
lost useful structure. It is not a pass: every state remains an initial candidate, five visual-only
alarms remain, and PCA recovers 0/5 physical contact-loss alarms. Next run fixed background
suppression, then segmented block pooling as an upper bound; do not start world-model prediction.
PCA result SHA: `f4c6905e02c42d9f6096fd15fff9724d81d5e62b400e78eb639c743f476ee464`.

The remaining representation ladder is complete in `eval/panda_push_visual_regions.py`. A
label-free motion-weighted readout (DINO patches weighted by RGB change from the common start)
worsens agreement to 30/56, with 12/25 physical alarms recovered and 13 visual-only alarms. The
MuJoCo-segmented block-region oracle recovers 21/25 physical alarms—16/19 mixed forks and 4/5
contact-loss—but emits 21 visual-only alarms, including 10/15 stable-centered and 5/6 overshoot
states. Thus object isolation recovers most relevant visual signal but not specificity. Full-frame
PCA45 remains the best agreement baseline (37/56, five visual-only alarms).

Do not move to predicted futures yet. The next experiment should stop forcing visual PCA axes into
v0's semantic 45-channel physical-state contract. Define a visual-native consequence curve from
tracked object geometry or patch correspondence, normalize it internally, and retain the frozen
action-branch/refinement logic for comparison. Simulator masks are diagnostic-only. The experiment
is resource-capped to one simulator process and at most eight Torch threads; do not restore the
discarded eight-EGL-worker configuration because it froze the desktop.

Results: `results/panda_block_push/visual_regions/`, motion SHA `2328bb220d85...`, block SHA
`a09f7c092e2d...`.

The proper visual-native port is implemented in `src/visual_consequence_monitor.py` and rescored by
`eval/panda_push_visual_native.py`. It replaces only physical semantic channel slicing with one
internally normalized Euclidean curve over the complete visual latent; all v0 BIC/majority logic is
unchanged and calibration-free. The segmented-block oracle now recovers 24/25 physical alarms,
including 18/19 mixed forks and all five contact-loss regimes, but creates 18 extra alarms (eight
stable, five physically quiet unanimous-topple, five overshoot). Full-frame PCA is more specific
(five extras) but recovers only 10/25 physical alarms. Every visual representation remains an
initial candidate on all 56 states.

This localizes the open problem to upstream visual geometry: ordinary object motion is not smooth
in the current DINO/PCA action-response coordinates. Next hold the visual-native consequence stage
fixed and test tracked object geometry or patch correspondence. Do not return to the invalid
physical-channel adapter and do not evaluate world-model predictions yet. Result:
`results/panda_block_push/visual_native/visual_native_diagnostic.json`, SHA `b5afd81f0b22...`.

`eval/panda_push_patch_correspondence.py` then tested the first universal geometry arm without
retuning: DINO identifies all 256 start patches, and only matched x/y coordinates plus velocities
enter PCA45 and the proper visual-native monitor. It fails to improve transfer: 33/56 agreement,
10/25 physical-alarm recall, 8/19 mixed-fork recall, eight extras, and 56/56 initial candidates.
Treat this as a frozen negative formulation; do not tune top-k/softness from DEV. The likely failure
is coarse 16x16 semantic correspondence and identity swapping, not the visual consequence stage.
Next use a higher-resolution temporally constrained point tracker or optical flow under a new
predeclared protocol. Result SHA `7c0a15cf5060...`.

`eval/panda_push_optical_flow.py` completes that predeclared generic geometry test. It tracks up to
128 unlabeled Shi--Tomasi corners through every control frame with pyramidal Lucas--Kanade flow and
a fixed one-pixel forward/backward check. The monitor receives only normalized x/y position,
velocity and visibility through per-state PCA45; masks, identities, simulator state and labels are
absent. Recall rises to 24/25 physical alarms (18/19 mixed forks and all five contact-loss), but
specificity collapses: 54/56 total alarms and 30 extras, including every safe-centered and
overshoot state. All 56 are still initial candidates. Track retention is high, so this is not a
missing-track artifact; benign full-image robot/contact/perspective motion is itself represented as
branching. Do not tune optical-flow constants on this DEV panel. Result:
`results/panda_block_push/optical_flow/optical_flow_diagnostic.json`, SHA `6742d28cfd93...`.

**Next agent:** stop the hand-designed full-frame tracking ladder. Predeclare and train a universal
dynamics representation from unlabeled video plus actions, using generic temporal correspondence,
object-centric or 3-D/SE(3)-like geometry, velocity preservation and local action smoothness. Do not
use failure/topple/success labels, object identity, or physical-v0 decisions in training. Freeze the
representation and interface, create a fresh Panda TEST split, and require a visual true-future
Pareto improvement before moving to world-model-predicted futures.

That label-free representation track is now implemented through its pre-monitor quality gate.
`eval/panda_push_slot_data.py` generates 80 paired interactions/2,720 frames completely separate
from monitor DEV and stores only RGB, actions, split and pair index. `src/models/slot_dynamics.py`
and its motion/balanced successors provide anonymous action-conditioned slots. Training protocol
SHAs are `c3f711031420...` (v0), `ad51ac336d6f...` (v1) and `512ef487d342...` (v2). All checkpoint
selection is based on label-free validation loss.

The pre-monitor result is negative but clean. V0 keeps uniform masks. V1 sharpens by one-slot
monopoly. V2 mathematically balances assignments and passes effective-slot count (8.0), max mass
(0.125) and motion correlation (0.337), but fails temporal geometry motion: 0.000240 RMS versus the
frozen 0.002 gate. `results/panda_block_push/slot_dynamics/quality_report_v2.json` therefore has
`passed: false`. Do not run these checkpoints on monitor DEV or create/open Panda TEST; that would
use monitor outcomes to tune a representation already known to be collapsed/static.

**Next agent:** add generic flow/depth correspondence supervision on the unlabeled TRAIN set and
predict persistent entity trajectories (position, velocity, visibility, uncertainty), while
retaining action-conditioned and neighboring-action losses. Add forward/backward identity and
non-static geometry gates before training. No task labels or simulator state. Only after all
label-free gates pass may the frozen visual-native monitor be opened, followed by a new TEST and
then predicted futures.

## Fresh physical Panda TEST and alarm videos (2026-09-25)

The requested fresh ground-truth Panda TEST is complete independently of the paused visual track.
`eval/panda_push_fresh_test.py` generated continuous random candidates with seed 1640, selected the
same 56-state mechanism coverage using physical outcomes only, froze protocol SHA
`bfc1490690f15...`, and ran unchanged Regime Monitor v0 once. Result: 14/20 mixed topple forks,
0/15 safe-centered, 1/10 unanimous-topple, 3/5 contact-loss and 0/6 overshoot alarms. Candidate and
refinement stages retain 18/20 forks; commitment majority reduces this to 14. Canonical result:
`results/panda_block_push/fresh_test/ground_truth_monitor.json`, SHA `dd406d053e17...`.

`eval/ground_truth_monitor_alarm_videos.py` rendered ten exact consequential alarm pairs from
immutable Jenga TEST and ten from fresh Panda TEST. Videos and outcome manifests are in
`results/ground_truth_monitor_alarm_videos/jenga/` and `.../pushing/`. A few selected pairs share
the same binary topple outcome while differing in another persistent physical regime; v0 detects
committed branches, not topple labels.

Recommended next: keep visual representation work paused. Run pushing intervention utility on a
separate episode distribution with the frozen detector, reporting topples prevented, goal
completion, delay and intervention rate, analogous to Jenga's 100-episode wrapper evaluation.
