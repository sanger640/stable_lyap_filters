# Latent safety filter (LSF) baseline on Jenga — protocol (fixed 2026-10-20 before any LSF code ran)

Baseline: Nakamura, Peters & Bajcsy, "Generalizing Safety Beyond Collision-Avoidance via Latent-Space Reachability
Analysis" (RSS 2025, arXiv 2502.00935), with the design choices of the follow-up "Designing Latent Safety Filters using
Pre-Trained Vision Models" (arXiv 2509.14758) where the original is silent. Goal: a fair head-to-head against our
label-free G-CEM filter on the same world model, policy, episodes and seeds.

## The method (as published)
1. Failure classifier l(z): MLP on world-model latents, hinge loss with margin delta:
   L = mean_safe ReLU(delta - l(z+)) + mean_fail ReLU(delta + l(z-)), tanh on the output; failure set {z : l(z) < 0}.
   Labels from privileged simulator state in simulation (their block task: block fallen).
2. Safety value V(z): discounted latent reachability backup learned in imagination,
   V(z) = (1 - g) l(z) + g min{ l(z), max_a V(f(z, a)) }, g = 0.9999; continuous actions -> DDPG (critic = V/Q, actor =
   safety policy maximising V of the next latent); imagination starts from encoded dataset frames; 4 x 512 ReLU MLPs,
   AdamW, actor lr 1e-4, critic lr 1e-3, batch 512.
3. Least-restrictive runtime filter: imagine the task action one step ahead; execute it if V(next) > eps, else the safety
   actor's action. Published eps: 0.4 (simulation), 0.3 (hardware).

## Adaptation to Jenga (stated, not hidden)
- World model: model_OIL (the one G-CEM uses in Stage 4/4b; deterministic, DINOv2-392 latents 28x28x8). One model step
  = 2 control steps; action input = the 2 absolute targets + gripper (8-d), as in all our filters.
- Latent for the heads: the 28x28x8 latent (a small conv trunk -> MLP, since our latent is a patch grid, not a CLS
  vector) plus proprioception (current target xyz + gripper), which the follow-up paper found necessary.
- Labels (privileged, simulator): failure = a neighbour block toppled (the same topple rule as all Jenga results,
  eval/fresh_fragility_panel.topple_count), sticky from the step it happens; computed per imagined step from the
  stored traces of results/program_b/{closedloop_v1, cf_v1, cf_v2, oil_v1} training splits (validation splits held out).
- Safety actor action: a bounded change of the target per model step (|delta xyz| <= 1 cm per control step, the policy's
  own range), gripper held; DDPG with target networks (tau 0.005) and a replay buffer of imagined transitions; imagined
  rollouts of 15 model steps (the look-ahead G-CEM uses) from dataset latents.
- Action chunks (not covered by the paper): at each decision, imagine the executed part of the policy's chunk (8
  control steps = 4 model steps); if V > eps at every imagined step, execute it; else the safety actor drives the next
  8 control steps, re-encoding the real frame every model step.
- eps tuned on dev episodes only (grid {0.1, 0.2, 0.3, 0.4, 0.5}); selection rule fixed now: the eps with the fewest
  neighbour topples among those whose v4 success is not more than 5 pp below the policy alone; ties -> larger eps.
- Head-to-head (pre-registered separately before it runs): fresh episodes, arms policy | G-CEM | LSF, same seeds,
  metrics as Stage 4 (topples, v4, v4b), exact McNemar.

## What differs between the two methods (for the paper)
LSF needs failure labels, a trained classifier, value function and safety policy; G-CEM needs none of these (no labels,
no extra training beyond the world model). Same world model, so differences come from the safety logic, not the model.

## Clarifications (fixed 2026-10-20 ~17:00, after the data stage, before any training)
- Data stage result: train 231,748 labelled latents (93,484 failed), 5,969 imagination resets; val 32,796 (13,606
  failed), 729 resets (validation episodes of the same sources). Failure share is 40% (not 25%) because many files have
  few safe latents to sample; the hinge loss averages safe and failed terms separately, so balance does not matter.
- Classifier margin DELTA = 0.25 on the tanh output (the paper does not report its value). Model selection: the epoch
  with the lowest validation hinge loss.
- Runtime with chunks: our harness re-decides every 8 control steps (0.8 s), as for every other filter. If the policy's
  segment fails the check, the safety actor plans the 8-step segment in imagination (4 model steps) from the current
  real latent and that segment is executed; the next decision re-encodes the real frame. (The paper re-plans every
  control step at 15 Hz; per-step re-encoding is not possible in this harness without changing the comparison.)
- Compute (2026-10-20 ~17:40, before training): training runs on Nibi (H100 slice) where cf_v1 / cf_v2 already live.
  The data are built in two parts with the same stage and rules: cf_v1 + cf_v2 on Nibi (tag cf), closedloop_v1 +
  oil_v1 locally (tag clo: train 59,618 latents / 25,013 failed / 2,569 resets; val 5,251 / 2,206 / 329) and
  uploaded; merged with eval/lsf.py merge. Exact counts differ slightly from the earlier single local build (separate
  random streams per part); the merged summary files record the final numbers.

## Training run 1 (Nibi job 24697324, 2026-10-20 17:44-17:56) and sanity check
Classifier: validation accuracy ~98.7% (best val hinge 0.018, epoch 3). Value (DDPG, 300 iterations): COLLAPSED -
on 729 held-out decision points V = Q(z, g, pi) is about -0.47 everywhere (5-95% range -0.50 to -0.43; 100% below 0),
AUROC(-V, any real topple) 0.45 (no signal), while the classifier at the current state alone gives 0.75. Likely cause:
with g = 0.9999 the value approaches the worst classifier score along imagined rollouts, and 15-step imagined rollouts
under the untrained actor drift to latents the classifier calls failed. Not usable; next: measure l along imagined
rollouts (hold still / dataset policy actions / actor) to locate the drift, then fix training (no test data used).
results/jenga/lsf_value_diag.json.
- Drift check (same day, validation states only): the classifier calls 2-3% of non-toppling imagined futures failed
  under hold-still, dataset policy actions and the actor alike - the same as on real frames - so imagination is not the
  problem. Diagnosis: with a fixed g = 0.9999 a safe state's value can rise only (1-g) l = 0.0001 per backup while
  the min propagates pessimism at once, so the critic drifts down from its initialisation. Fix (run 2): anneal g from
  0.9 to 0.9999 over the first 80% of training (as in the reachability-RL toolbox the paper builds on, to my
  understanding), 600 iterations; nothing else changed. Selection of the run uses the same validation sanity check.
- Run 2 (gamma annealed 0.9 -> 0.9999, 600 iterations) also collapsed: critic ~-0.47 for every input although the
  classifier scores safe resets ~0.8 and failed latents ~-0.33. The log shows the value rising until the buffer fills,
  then sinking steadily: the min-backup with function approximation spreads the ~3% failed states' low values to all
  similar latents, and with g near 1 nothing pulls it back.
- **Tuning rule for g (fixed before running, validation data only, never test episodes):** fixed g in {0.9, 0.95,
  0.99} (effective horizons ~1, 2, 10 s at 0.1 s per model step), 600 iterations each; choose the g with the highest
  AUROC of -V for "any real future topples" on the 729 held-out validation decision points (eval/lsf_diag.py).
  If none exceeds the classifier-only score at the current state (0.75), report that the learned value adds nothing.
- Reference code checked (2026-10-20 ~18:35, before any of the gamma runs finished): the DINO-WM-based reference
  implementation (github.com/tabz23/Latent-Safety-Filters, train_HJ_mani_configs.yaml) uses gamma = 0.98, batch 128,
  actor lr 1e-4, critic lr 1e-3, tau 0.005; the Bajcsy group's UNISafe (github.com/CMU-IntentLab/UNISafe) anneals gamma
  0.85 -> 0.9999 in steps every 4,000 updates, uses twin critics (SAC), and truncates imagined episodes (target = l at
  the final step). Added: LSF_TRUNC=1 (final imagined step terminal) and gamma = 0.98. Runs: fixed gamma 0.9 / 0.95 /
  0.99 without truncation (running) + 0.9 / 0.95 / 0.98 / 0.99 with truncation. Selection rule unchanged: highest
  validation AUROC across all seven; the chosen setting then goes to the dev eps sweep.

## Value selection (Nibi job 24703953, 2026-10-20 19:33; rule above)
AUROC(-V, any real topple) on the 729 held-out validation decision points (classifier at the current state: 0.755):
| run | AUROC | share V < 0 |
|---|---|---|
| g 0.9999 (run 1) | 0.45 | 100% |
| g annealed 0.9 -> 0.9999 (run 2) | 0.39 | 100% |
| g 0.9 | 0.782 | 4% |
| **g 0.95 (selected)** | **0.785** | 16% |
| g 0.99 | 0.781 | 67% |
| g 0.9 / 0.95 / 0.98 / 0.99 + truncation | 0.72 / 0.66 / 0.74 / 0.71 | 2 / 2 / 2 / 47% |
Selected: fixed g = 0.95, no truncation (results/program_b/lsf_v1/value_g0.95.pt). The fixed-g runs are within noise of
each other; all three beat the classifier alone, so the learned value adds a little foresight. Next: dev eps sweep.
results/jenga/lsf_value_diag_value*.json.

## Dev eps sweep spec (fixed 2026-10-20 ~19:45, before any closed-loop LSF episode ran)
Episodes 15000-15099 (100 fresh dev episodes; no earlier run used 15000+), Stage 4 paired setup (stress layouts,
settled start, execution noise x2, policy seeded per episode), arms record (policy alone) and lsfe1..lsfe5 (eps 0.1 -
0.5), world model model_OIL, value_g0.95.pt. Packed on Nibi (paired_group.sbatch, 8 workers per GPU). Selection rule
as above (fewest topples with v4 no more than 5 pp below the policy; ties -> larger eps); topples by the same replay
outcome audit as Stage 4. Smoke first: episodes 15900-15901, record + lsfe3 (not part of any analysis).
- **Smoke result and grid amendment (2026-10-20 19:50, before any dev episode ran).** Smoke (15900-15901, eps 0.3): the
  LSF intervened on 60/60 chunks in both episodes (V of the policy's segment 0.05-0.28, never above 0.3); its safety
  actor then kept moving the target away (up to ~1.7 m from the policy's plan; least-restrictive switching never hands
  back while V < eps), so the block was never lifted. With g = 0.95 the validation V quantiles are 5/25/50/75/95% =
  -0.20 / 0.09 / 0.15 / 0.24 / 0.43 (83% below 0.3), so the published eps (0.3-0.4, on that paper's own value scale)
  means "always intervene" here. Amended grid, covering our value's range: eps in {0.00, 0.05, 0.10, 0.15, 0.20, 0.30}
  (arms lsfe000 ... lsfe030); selection rule unchanged. Behaviour of the safety actor recorded as is (method as published).
- Scheduling note (21:05, nothing had run): the packed whole-GPU job (24705559) was held behind a node in maintenance
  (est. start 02:37); cancelled and resubmitted unchanged in content as a 100-task slice array (paired_array.sbatch,
  1g.10gb, job 24709409; same episodes, arms, code).
- Scheduling note 2 (22:10): Nibi ran dev tasks 0-69; the never-started tasks 70-99 were cancelled there and run on Fir
  (job 66192391; same container, environment and code; Fir mirrors Nibi's user-site packages). Reproduction check:
  episode 15000 (record + lsfe010) re-run on Fir (tag _firrepro) must match Nibi's logged outcome; if not, tasks 70-99
  are re-run on Nibi instead.
  Reproduction check PASSED (22:20): episode 15000 on Fir vs Nibi, record: 32 chunks, lifted, no topple, max lift
  52.7 vs 52.6 mm, 25.6 s both; lsfe010: 60 chunks, no lift, neighbour toppled, peak tilt 95.0 vs 96.1 deg (same
  outcomes; sub-mm floating-point differences as in the Narval check). Fir's tasks 70-99 are kept.

## Dev sweep RESULT (2026-10-20 23:33; results/jenga/lsf_dev_result.json, outcome_audit_lsfdev.json)
92 episodes complete in every arm (some arms lost 1-8 episodes to the 2 h job limit; analysis on the common 92; replay
topple flag matches the log in 91-99 per arm). Topples / v4 / v4b: policy 25 / 62 / 54; eps 0.00: 80 / 2 / 1;
0.05: 83 / 0 / 0; 0.10: 64 / 0 / 0; 0.15: 0 / 0 / 0; 0.20: 0 / 0 / 0; 0.30: 0 / 0 / 0. No eps keeps v4 within 5 pp
of the policy -> by the rule NO eps is selected; the LSF as published fails on Jenga.
Diagnosis (not a switching bug): the value falls sensibly as the policy nears the block (ep 15003: 0.28 -> ~0.1, then
< 0 at contact), but the DDPG safety actor is saturated: once in control it moves the target 10 mm per step in -x, -y
and -z every step (ep 15003, eps 0: (485, -1, 497) -> (-523, -1045, -542) mm over 13 chunks), i.e. down through the
blocks and the table. Imagined training let it exploit world-model blind spots for large off-distribution moves. At
eps >= 0.15 it takes control early (before any contact), so nothing topples but the task is never done.

## Variant LSF-hold (spec fixed 2026-10-20 23:40, before running; agreed direction: strongest fair version of the baseline)
Same learned value V and least-restrictive switch; the fallback is HOLD (target held for the chunk, gripper column
untouched, exactly the Stage 4 hold-and-replan fix) instead of the DDPG actor. Isolates the learned value (the method's
trigger) from its actor. Arms lsfh000 ... lsfh030 (same 6 eps), same 100 dev episodes 15000-15099 (record arm reused
from the sweep above: same episodes, seeds, code), Fir, same selection rule. Both LSF (as published) and LSF-hold go
to the head-to-head if selected; LSF as published is reported regardless.
- LSF-hold dev, partial (2026-10-21 00:23): 12 of 100 Fir tasks failed at start-up on one node (fc10713, "CUDA device
  busy or unavailable"), unrelated to episode content. On the 87 complete episodes the rule selects eps 0.10 (topples /
  v4 / v4b: policy 27 / 54 / 49, eps 0.10: 17 / 50 / 46, eps 0.00: 18 / 51 / 42; results/jenga/lsfh_dev_result_partial87.json).
  The 12 tasks are re-run with that node excluded; the selection is made on the full set.
- **LSF-hold dev RESULT, full set (2026-10-21 01:06; results/jenga/lsfh_dev_result.json):** 99 episodes complete in
  every arm (replay flags match 99-100/100). Topples / v4 / v4b: policy 27 / 66 / 58; eps 0.00: 20 / 60 / 50; 0.05:
  19 / 58 / 50; 0.10: 19 / 59 / 54; 0.15: 3 / 17 / 16; 0.20: 0 / 3 / 3; 0.30: 0 / 0 / 0. v4 bar = policy - 5 pp =
  61.7%; best is eps 0.00 at 60.6% (-6.1 pp) -> by the rule NO eps is selected (the 87-episode partial had picked
  0.10; the 12 re-run episodes moved it below the bar). Per the head-to-head launch rule the head-to-head is NOT
  launched automatically; decision on an exploratory head-to-head left to the user.
  Reading: the learned value works as a trigger (topples -26 to -30%), but it costs 6-8 pp of success; for comparison
  (different episodes, Stage 4) G-CEM cut topples 69 -> 19 while RAISING v4 206 -> 246 out of 300.

## Open item (user, 2026-10-21 morning)
The user expects the LSF as published (with its DDPG safety actor) to work without the hold variant and suspects an
implementation problem. To revisit AFTER the SafeManip benchmark work: re-check the actor (saturation, action
bounds/scaling, exploration noise, training length, imagined-rollout horizon, data distribution of resets) against the
reference implementations (tabz23/Latent-Safety-Filters, CMU-IntentLab/UNISafe) before any further LSF claims.

## Re-audit (2026-10-21 afternoon): our DDPG safety actor had collapsed
Audit (eval/lsf_audit.py, results/jenga/lsf_audit_value_g0.95.json; 729 held-out decision points): the deployed actor
(value_g0.95.pt) is saturated on 100% of outputs and outputs the SAME action everywhere (-10 mm/step straight down);
ensemble disagreement of its imagined transitions is 3.6x the policy's (99% above the policy's 95th percentile); and its
own critic rates it WORSE than holding still (median V after its plan 0.01 vs 0.15 hold / 0.15 policy). So the actor
optimised nothing: a DDPG actor-saturation failure plus model exploitation. Comparison with the authors' code:
- Original LSF (CMU-IntentLab/latent-safety, franka-wm.py): each imagined step samples the stochastic world model N
  times and keeps the MOST PESSIMISTIC transition; the world model is trained on 50% random-exploration data; resets from
  random offline timesteps. Ours: deterministic model, no pessimism, policy-only data.
- UNISafe (CMU-IntentLab/UNISafe, same group, follow-up): SAC (twin critics, entropy in the target, learned alpha),
  gamma annealed 0.85 -> 0.9999, episodes truncated as terminal, and an epistemic-uncertainty margin (ensemble
  disagreement above an OOD threshold counts as failure) to stop the actor exploiting the model.
Fix (eval/lsf.py): `sac` stage = UNISafe recipe in our world model (uncertainty = disagreement of model_OIL + ENS0-3,
threshold = 95th percentile under the policy's own actions); `LSF_PESS=1` = original LSF's pessimistic transition with
the 5 ensemble members as samples (both trainers). Training run 1: UNISafe (value_unisafe.pt), 60k updates.
Gate before any closed-loop use (fixed now): actor not saturated (< 20%), its imagined V after its own plan >= hold's,
value AUROC (eval/lsf_diag.py) >= 0.7.

## Re-audit RESULT, UNISafe recipe (2026-10-21 17:37; results/jenga/lsf_audit_value_unisafe.json, lsf_value_diag_value_unisafe.json)
value_unisafe.pt (SAC + ensemble-uncertainty margin, 60k updates) PASSES the gate on all three counts: actor saturated
on 0.03% of outputs (< 20%; was 100%); median imagined V after its own plan 0.85 vs hold 0.76 / policy 0.76 (>= hold;
was 0.01 vs 0.15); value AUROC for "any real future topples" 0.81 (>= 0.7; g0.95 DDPG was in the same range). Its
plans are also the most in-distribution of the three (median ensemble disagreement 0.0051 vs policy 0.0066 / hold
0.0027; 0.1% above the policy's 95th percentile, was 99%). So the earlier LSF failure was our DDPG actor collapsing,
as the user suspected.

## Dev eps sweep, UNISafe value (spec fixed 2026-10-21 17:50, before any closed-loop episode with this value)
Same protocol as the sweep above: episodes 15000-15099, Stage 4 paired setup, world model model_OIL, least-restrictive
switch to the SAC actor (the method as published, no hold), record arm REUSED from the lsfdev sweep (policy-only
episodes do not touch the filter code; same episodes and seeds). Value value_unisafe.pt. The value's scale differs
(validation V quantiles 5/25/50/75/95% = 0.15 / 0.67 / 0.74 / 0.88 / 0.94), so the grid is eps in {0.00, 0.10, 0.20,
0.30, 0.50, 0.70} (includes the published 0.3; 0.70 is below the median), arms lsfe000 ... lsfe070, tag _unidev, Fir
paired_array. Selection rule unchanged (fewest replay-audited topples among eps with v4 no more than 5 pp below the
policy; ties -> larger eps). If an eps is selected, it goes to a head-to-head with G-CEM on fresh episodes (spec to be
fixed before that run). Smoke first: episodes 15900-15901, lsfe030 (not part of any analysis).

## Re-audit RESULT, original-LSF pessimism on our DDPG (2026-10-21 18:35; results/jenga/lsf_audit_value_pess_g0.95.json)
value_pess_g0.95.pt = the selected g0.95 DDPG run with ONLY the original LSF's pessimistic transition added (worst of
the 5 ensemble members per imagined step). FAILS the gate: actor saturated on 100% of outputs; median imagined V after
its own plan -0.05 vs hold 0.12; ensemble disagreement of its plans 3.4x the policy's (93% above the policy's p95);
value AUROC 0.78. Pessimism over the ensemble does not stop the DDPG actor collapsing; the UNISafe recipe (SAC +
uncertainty margin) does. Only value_unisafe.pt goes to closed loop.
- Scheduling note (18:30): Fir and Nibi GPU partitions down (PartitionDown); the smoke and sweep move to Rorqual (code
  synced; src/ and model_OIL identical). Reproduction check PASSED (18:38): episode 15000, record arm, Rorqual vs Nibi:
  32 chunks, lifted, max lift 52.6 mm, neighbour tilt 6.2 deg, 25.6 s, success - identical. The reused record arm stands.
- Smoke (Rorqual 24879926, 15900-15901, lsfe030, not analysed): runs cleanly (~0.22 s per chunk); V of the policy's
  segment stays 0.54-0.94, so at eps 0.3 it never intervened, and both episodes equal the policy-alone smoke exactly
  (15900 success, 31 chunks; 15901 timeout, 60 chunks). Dev sweep submitted unchanged: Rorqual job 24880394 (50 tasks x
  2 episodes x 6 arms, tag _unidev).

## Dev sweep RESULT, UNISafe value (2026-10-21 20:25; results/jenga/lsf_dev_result_unidev.json, outcome_audit_unidev.json)
99 episodes complete in every arm (replay flags match 99-100/100). Topples / v4 / v4b: policy 27 / 66 / 58;
eps 0.00: 24 / 70 / 59; **0.10: 23 / 70 / 59 (selected)**; 0.20: 24 / 69 / 59; 0.30: 24 / 66 / 56; 0.50: 26 / 66 / 53;
0.70: 8 / 5 / 4. Eligible (v4 within 5 pp): 0.00-0.50; fewest topples -> eps 0.10.
Reading: trained correctly, the LSF as published no longer collapses (success up 4 points instead of to ~0), but at
every eps that keeps success it removes only 1-4 of 27 topples (-4 to -15%); at eps 0.70 it stops topples by
stopping the task. For comparison only (different episodes): G-CEM in Stage 4 cut topples 69 -> 19 with v4 206 -> 246
of 300. Per the spec, eps 0.10 goes to a head-to-head with G-CEM on fresh episodes 17000-17299
(results/jenga/lsf_h2h2_protocol.json, written before any episode). Rorqual cbfg reproduction check (Stage 4 ep 14000)
first.
- Head-to-head 2 NOT launched (2026-10-21 20:50): the pre-set cbfg reproduction check FAILED on Rorqual. Stage 4 ep 14000
  (Nibi): 22 chunks, lifted, success; Rorqual: 60 chunks, never lifted. Same trigger decisions (interventions at chunks
  13, 15, 17, 20), but G-CEM's seeded search picked a slightly different edit already at chunk 13 (theta differs) and
  the episodes diverged. Rorqual repeats ITSELF exactly (second run identical), so it is a cross-cluster numerical
  difference amplified by the search, not a broken setup. A Fir check was submitted (66693831), but the Fir and Nibi
  sessions expired (need the user's Duo login), so its result is unread. Decision for the user: run head-to-head 2 on
  Rorqual (all arms per episode on the same cluster, so pairing stays valid, but the rule as written is not met), or
  wait for Fir/Nibi.

## Audit 3 (2026-10-24 evening, user: "we did not properly implement the latent safety filter")
Diagnostic, exploratory (eval/lsf_trigger_check.py -> results/jenga/deep_dive/lsf_trigger_check.json; logs of the LSF
arms, which at eps <= 0.1 act on only 1.7-2.0 chunks per episode, so these are nearly the policy's own episodes):
- **As an alarm the learned value is weak.** At the fork monitor's matched safe alarm rate (11.3%), V_min of the policy's
  segment warns >= 3 chunks ahead of 43.6% of topples (h2h2, 55 topples; 25-26% on the 100 unidev episodes), vs 77-79%
  for the fork test on other held-out policy episodes (monrep, 5l). At 20%: 76% (h2h2), 54-57% (unidev).
- **The dev-selected eps 0.10 never sees a topple coming.** Safe decisions: V_min 5/50/95% = 0.60 / 0.82 / 0.94;
  the 3 decisions before a topple: 0.40 / 0.68 / 0.74; only 1.2% of those are below 0.10.
- **It acts, but too late.** h2h2 LSF arm, 55 topple episodes: 24 had no intervention in the 8 chunks before the topple,
  29 had one only in the last 2 chunks (after contact had started: V crashes at contact), 2 had one >= 3 ahead.
- No new code bug found (encoding, action format, gripper column, V(f(z, a)) check, actor output all re-read). The gaps are
  fidelity choices, listed against the published method:
  1. **Switching rate.** The paper re-checks every control step at 15 Hz, so a value that only drops at contact can
     still save the scene; ours re-checks every 0.8 s and then runs the actor open-loop for 8 control steps. The
     "too late" pattern above is exactly what a per-step filter would catch.
  2. **Stochastic imagination.** The original LSF samples its stochastic world model N times per step and keeps the
     most pessimistic transition; our model is deterministic and the only pessimism is over 5 ensemble members
     (epistemic). Jenga topples are noise-driven forks (aleatoric), which a deterministic imagination averages away.
     G-CEM gets the robot's measured perception and execution noise; the LSF did not. A faithful, fair version gives it
     the same noise model.
  3. **eps grid.** The useful range of this value is ~0.55-0.75 (alarm rates 5-25%); the unidev grid jumped 0.50 -> 0.70.

### LSF-step (stage 1, spec fixed 2026-10-24 ~19:30, before any LSF-step code ran)
- Same value value_unisafe.pt (SAC actor + twin critics), same world model model_OIL, same encoder.
- Least-restrictive switch at EVERY model step (2 control steps = 0.2 s, the finest our world model allows): encode the
  real frame now; imagine the policy's next 2 commands one model step ahead; if V(z', g') > eps execute them, else
  execute the safety actor's 2-step action (mean, as deployed). Control returns to the policy at the next model step
  whenever its commands pass. The policy still plans every 8 control steps (its own chunking, unchanged).
- Modes lsfs<eps>, dev episodes 15000-15099, Stage 4 paired setup, record arm reused from unidev (policy-only episodes),
  Fir paired_array. eps grid {0.10, 0.30, 0.50, 0.60, 0.65, 0.70} (0.10 = the chunk-level choice, isolating the
  switching rate; 0.30 = published; the rest span the value's useful range). Selection rule unchanged (fewest
  replay-audited topples among eps with v4 no more than 5 pp below the policy; ties -> larger eps). Smoke 15900-15901 first.
- If an eps is selected: a head-to-head vs G-CEM on fresh episodes, pre-registered separately.

### LSF-stochastic (stage 2, fixed now, runs after stage 1)
Retrain the UNISafe value with the original LSF's pessimistic sampling translated to our noise model: each imagined
step draws N = 8 transitions of model_OIL under the robot's measured execution noise (the episodes' snippets, scale 2,
the same snippets the monitor uses) and keeps the one with the lowest classifier margin; resets get the monitor's
perception-noise start sampler (G at scale 1). Runtime check: V after each of 8 noisy one-step imaginations, the minimum
must exceed eps. Same offline gate as the re-audit (actor not saturated, V after its own plan >= hold, AUROC >= 0.7),
plus the early-warning number above reported (not gated). Then the stage 1 sweep protocol with the better switching rate.
