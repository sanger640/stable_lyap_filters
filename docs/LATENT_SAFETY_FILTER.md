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

### LSF-step dev RESULT (2026-10-24 20:22; results/jenga/lsfs_dev_result_lsfsdev.json, outcome_audit_lsfsdev.json)
Fir, 100 dev episodes x 6 eps, 99 common with the reused record arm (replay flags 99-100/100). Topples / v4 / v4b:
policy 27 / 66 / 58; eps 0.10: 26 / 66 / 58; 0.30: 29 / 61 / 53; 0.50: 32 / 56 / 49; 0.60: 32 / 55 / 40; 0.65: 28 / 53 / 34;
**0.70: 13 / 63 / 26 (selected by the rule: fewest topples with v4 within 5 pp)**.
- Switching at every model step makes the learned value useful: topples 27 -> 13 (-52%) with v4 kept, where the
  chunk-level switch never removed more than 4 of 27 topples without losing the task.
- The cost moves elsewhere: at eps 0.70 the safety actor drives 31.9 of ~229 model steps per episode, about half of
  them after the first grasp (1681 before / 1507 after); red block knocked over after extraction 39 vs 9 episodes
  (v4b 26 vs 58), time-outs 20 vs 1, episodes 57 vs 42 chunks. The actor does not know it is holding the block.
- Mid-range eps (0.30-0.65) are worse than the policy on topples (28-32): the actor takes over in brief bursts that
  disturb the scene without saving it.
- Next per the plan: stage 2 (noisy value + noisy check), then one head-to-head vs G-CEM with the better of the two on
  fresh episodes (pre-registered separately; v4 and v4b both reported).

### LSF stage 2 offline gate RESULT (2026-10-25 05:40; value_unisafe_alea.pt, 60k updates, trained locally)
- Actor saturated on 0.006% of actions (gate: not saturated) -> pass.
- Imagined V after 4 model steps, median over the 729 held-out decision points: actor 0.839, policy 0.734, hold 0.734 (gate: actor >= hold) -> pass.
- AUROC of -V vs any toppling future 0.829 (gate >= 0.7; stage 1 value 0.810) -> pass. Spearman vs topple share 0.37 (0.34).
- Sources: results/jenga/lsf_audit_value_unisafe_alea.json, lsf_value_diag_value_unisafe_alea.json.
- Early-warning number: taken from the dev sweep's own logs (the runtime logs the noisy V_min at every decision, as stage 1).
- Dev sweep running (Fir 337330): modes lsfa010-lsfa070, episodes 15000-15099, the stage 1 protocol unchanged. Smoke 15900-15901 ran (both time-outs at eps 0.70, ~29 actor overrides per episode).

### LSF stage 2 dev RESULT (2026-10-25 08:38; results/jenga/lsfa_dev_result_lsfadev.json, outcome_audit_lsfadev.json)
Fir, 100 dev episodes x 6 eps, 99 common with the reused record arm. Topples / v4 / v4b: policy 27 / 66 / 58;
eps 0.10: 16 / 69 / 58; **0.30: 9 / 76 / 59 (selected: fewest topples with v4 within 5 pp)**; 0.50: 9 / 50 / 28;
0.60: 4 / 19 / 7; 0.65: 1 / 3 / 1; 0.70: 1 / 0 / 0.
- The faithful version works: at the published eps 0.30 it cuts dev topples by two thirds AND raises v4 (+10 of 99),
  v4b equal. Stage 1 (per-step, deterministic value) needed eps 0.70 for 13 topples and lost v4b (26).
- Its actor now acts rarely: 7.9 overrides of ~203 model steps per episode at eps 0.30 (stage 1 at 0.70: 31.9); time-outs
  5 (policy 1); red falls before / after extraction 10 / 17. Cheap: no sampling or certificate, one V per model step.
- Above eps 0.5 the value is too pessimistic and the actor takes over (v4 collapses).
- So the user's suspicion was right: LSF-as-run was unfaithful in two ways (switching rate, deterministic imagination),
  and fixing both makes it a strong baseline. Head-to-head 3 vs psel pre-registered (results/jenga/lsf_h2h3_protocol.json,
  600 fresh episodes 25600-26199).


## lflsf: a LABEL-FREE latent safety filter (pre-registered 2026-10-25 12:02, results/jenga/lflsf_protocol.json)
User 2026-10-25: retry the learned value and the learned actor. LSF stage 2 works in closed loop, and its only scope
violation is its margin (a classifier trained on topple labels). lflsf swaps only that: a margin net l_hat(z) regressed on
our label-free hold-plan fork statistic (neg_log_sens_4, HJ v2 stage A), proprio zeroed, mapped by t = tanh((m - q10) / s)
with q10 / s from the training decisions' own distribution (the most sensitive tenth of visited states = the failure set).
Then the stage 2 recipe unchanged (UNISafe SAC in model_OIL's imagination, ALEA 8, 60k updates) -> value_lf_alea.pt.

### LF1 RESULT (2026-10-25 12:04, commit 939a024; results/jenga/lflsf_LF1.json): PASS
Held out (paired_record_m3, 13798 decisions; trained on 69247): Spearman(l_hat, t) 0.635 (gate >= 0.6); AUC of -l_hat for
decisions 1-3 chunks before a topple vs safe-episode decisions 0.875, ABOVE its own target's 0.804 (gate >= 0.754).
Share of states with l_hat < 0: 7.6% (target 9.5%). Labels used for grading only.

### LF2 training (2026-10-25 13:33-17:25, local RTX 3090)
Healthy and close to the labelled stage 2 run: actor saturated on 0.13% of actions at 60k updates, uncertainty-fail share
0, mean margin over imagined states 0.92 (labelled run's classifier ~0.76 at the same point): the label-free margin calls
fewer imagined states unsafe, so the value may be less conservative.

### LF2 offline gate RESULT (2026-10-25 17:45; results/jenga/lflsf_LF2.json): PASS
Actor saturated 0.22% (< 5%); imagined V after the actor's plan 0.968 >= after hold 0.870 (policy 0.864); AUROC of -V vs
any toppling future 0.885 (>= 0.70; labelled stage 2: 0.829; the label-free margin itself 0.883). V quantiles over
held-out decisions (5/25/50/75/95%): -0.43 / 0.33 / 0.85 / 0.995 / 0.999; 24% below 0.30. So on REAL states it is
lower than the labelled value: the LF4 smoke test (2 episodes 99000-99001 per edit arm at eps 0.30) overrode on 17-29% of
model steps (labelled stage 2 at eps 0.30: ~4%), 5 of 6 time-outs. Amendment_1 (17:31, before any LF3 episode): eps 0.00
added to LF3 (lfa000); LF4's fallback eps 0.30 -> 0.00.

### LF4: least-restrictive EDITS instead of a switch (pre-registered 2026-10-25 17:15, results/jenga/lflsf_LF4_protocol.json)
User 2026-10-25 ~17:05: psel's "pick the closest certified policy sample" is too simple and does not generalise; the
direction is LSF's methodology or other reachability / Lyapunov / CBF methods. LSF's switch discards the policy's command
whenever V <= eps; a CBF-style filter edits it as little as needed. With the LF2 value / actor, at the LF3-selected eps,
same dev episodes: lfm (smallest blend toward the actor with min V > eps), lfg (gradient ascent of the soft-min V through
model_OIL on the policy's own xyz, <= 1 cm, gripper the policy's), lfc (lfg under the discrete-time CBF condition
h(z') >= 0.5 h(z), h = V - eps). All candidates see common noise. Code: eval/cbf_filter.py lsf_edit.

### Amendment 2 (2026-10-25 17:49, after only LF3 record episodes): rate-matched label-free eps
Held-out paired_record_m3 decision states (exploratory, scratchpad an/vreal.py), state value V(z, g):
- The label-free value is the BETTER state alarm (grading only): at matched safe-state alarm rates it flags 31 / 51 / 76%
  of states 1-3 chunks before a topple at 5 / 10 / 20% safe alarms; the labelled stage 2 value 23 / 44 / 68%. AUC 0.868 vs 0.845.
- But its scale differs: it is low (< 0.3) on 30% of safe states with the gripper closed (labelled: 3%) and on 26% of
  chunks 10-29 (the approach / grasp), so eps 0.30 means a different rule for each value.
- Label-free calibration (no labels: match the share of ALL visited states the labelled value flags, 5.4% at eps 0.10,
  6.9% at 0.30) -> label-free eps -0.25 / -0.15. Both added to LF3 (lfa-25, lfa-15). Selection rule unchanged.

### LF5: the 8x margin (pre-registered 2026-10-25 18:07, results/jenga/lflsf_LF5_protocol.json)
First LF3 arm (lfa-15, logged outcomes, unaudited): topples 27 -> 14 but logged successes 40 -> 13, lifts 50 -> 16 of 99:
the switch blocks the grasp. Training runs (m3 untouched, an/margins.py): the 4x margin flags 18% of safe gripper-closed
states vs 45% of states 1 chunk before a topple; the 8x margin 9% vs 33%. Toppled scenes are calm at every scale (0-9%
flagged), so no hold-plan spread marks the failed state itself; the min-over-time backup has to carry it.
- **LF5a RESULT (18:09; results/jenga/lflsf_LF1_x8.json): PASS.** Held-out Spearman 0.836, AUC 0.854 (target 0.837). The
  net is smoother than its target: 3.3% of held-out states below 0 (target 8.8%). SAC value value_lf8_alea.pt training.

### Why the fragility margin over-triggers (exploratory, 2026-10-25 19:50-20:20; scratchpad an/noise_trig.py, an/margin_choice.py, an/sidefx*.py)
- Not the 2x execution noise. Hold command on 4,707 held-out m3 states, min V over 8 noisy one-step imaginations, share of
  states from SAFE episodes that would trigger, execution noise 0 / 1 / 2x: label-free value (eps 0) 10.2 / 11.2 / 12.9%,
  labelled stage 2 value (eps 0.30) 1.4 / 2.1 / 3.5%. Noise adds a little to both; the label-free value is ~7x higher at any noise.
- No hold-fork statistic is a failure set: at 50% recall of states 1-3 chunks before a topple every statistic and scale
  flags >= 12% of safe-episode states (best: spread at 8x), and states AFTER a topple score as safe (AUC 0.226 for the 8x
  spread, post-topple vs safe: a fallen scene is calm). The value therefore learns "avoid fragile states" (incl. the grasp),
  not "avoid topples". Imagined drift under a hold is no better (AUC 0.78; the model drifts while the gripper is closed).
- A side-effect statistic is: c(z_k) = 98th percentile over the 28x28 patches of mean_channels |z_k - z_0| (the scene has
  changed somewhere, relative to the episode's first view). Post-topple vs safe AUC 0.992 (LF1 training runs) / 0.989 (m3);
  0% above threshold in the 4 chunks before a topple starts, 12% at the topple chunk, 83% one chunk later (absorbing);
  0 of 1,110 safe states with the red block lifted. Label-free threshold (98th percentile over decisions before the gripper
  first closes): flags 75.6 / 72.7% of post-topple states, 0.9% of safe states, 1.7% of safe gripper-closed (train / m3).

### LF6: side-effect failure set (pre-registered 2026-10-25 20:23, commit dd9352a; results/jenga/lflsf_LF6_protocol.json)
Margin t = tanh((thr - c) / s), thr / s from the pre-grasp reference; margin net from a single frame (no z_0 input), then
the stage 2 recipe (value_lfse_alea.pt), the LF2 gate, then dev arms lfa010 / lfa030 / lfa050 / lfc030 (tag _lfsedev,
local 3090 only, amendment 1), selection over all lflsf arms. Relative-reachability / side-effect safety in latent space:
"do not change what you are not acting on"; label-free and task-agnostic.
- **LF6a RESULT (20:25; results/jenga/lflsf_LF1_se.json): PASS.** Held out (m3): Spearman 0.967 with its target; AUC
  post-topple vs safe 0.988 (target 0.989); l_hat < 0 on 0.36% of safe-episode states and 78.4% of post-topple states.
  A single frame suffices: the net recognises a changed (toppled) scene without the episode's first view.

### LF3 absolute-switch arms, replay-audited (2026-10-25 20:34; results/jenga/outcome_audit_lflsfdev_partial.json; lfr arms pending)
Topples / v4 / v4b of 100 (local record arm 27 / 64 / 58): lfa-25 11 / 53 / 43; lfa-15 14 / 35 / 27; lfa000 4 / 26 / 23;
lfa010 4 / 19 / 14; lfa030 3 / 4 / 4; lfa050 0 / 2 / 2. None within 5 pp of the policy's v4. Overrides per episode 11.7-25.7
(labelled stage 2 at eps 0.30 on the same episodes: 7.8, topples 9, v4 76). Note: the rollout's logged success flag is far
stricter than v4 (record 40 vs 64), never compare on it.

### LF3 RESULT (2026-10-25 21:24; results/jenga/lflsf_LF3_dev.json, outcome_audit_lflsfdev.json): no arm selected
Relative switch (latent-CBF form) topples / v4 / v4b / time-outs: lfr010 5 / 27 / 19 / 64; lfr030 3 / 6 / 5 / 90; lfr050
0 / 4 / 4 / 96 (record 27 / 64 / 58 / 2). Time-outs for the absolute arms: lfa-25 24, lfa-15 42, lfa000 62, lfa010 74,
lfa030 93, lfa050 98. No arm within 5 pp of the policy's v4 -> LF4 runs at the fallback eps 0.00 (started 21:25).

### CF1 RESULT (2026-10-25 22:01; results/jenga/cf1_result.json, protocol cf1_protocol.json): FAIL
Counterfactual fork (plan vs hold over the same 64 futures, outside the hand path), 2359 held-out m3 decisions (180
pre-topple, 979 safe gripper-closed). AUC / safe gripper-closed flagged at 50% pre-topple recall: P_d 2x 0.719 / 7.7%,
P_d 4x 0.716 / 10.3%, max_d 4x 0.859 / 9.7%, max_d 2x 0.805 / 7.2%; plan vs now (o) P_o 2x 0.745 / 5.2%; hold-spread
reference 4x 0.811 / 10.4%, 2x 0.694 / 23.8%. Gate (AUC >= 0.85 and <= 5.2%) not met. The plan-vs-hold reading removes a
little of the grasp's false alarms, not half; every per-state imagined statistic tried so far (spread, split, drift,
counterfactual change, label-free and labelled values) flags 5-10% of safe grasping states at 50% recall, which points to
the world model's foresight of contact outcomes as the shared limit rather than the readout. The labelled LSF works in
closed loop at similar per-state discrimination because it acts rarely (1.4% of safe states trigger at eps 0.30) and its
actor, trained against topples only, does not abandon the grasp.

### CF2 RESULT (2026-10-25 23:16; results/jenga/cf2_result.json, protocol cf2_protocol.json): FAIL as registered
Time-irreversibility read of the fork (plan, then the reversed plan, vs hold, over the same 64 futures, outside the hand
path), same 2359 m3 decisions as CF1. AUC / safe flagged / safe gripper-closed flagged at 50% pre-topple recall:
P_irr 2x 0.753 / 6.8% / 3.0%, P_irr 4x 0.723 / 7.1% / 2.7%; mean_r 2x 0.794 / 9.9% / 13.1%, 4x 0.866 / 7.9% / 7.7%;
**max_r 2x 0.850 / 5.4% / 1.9%, max_r 4x 0.890 / 6.8% / 2.5%** (hold-spread reference 10.4%). The registered statistic
P_irr meets the false-alarm half (<= 5.2%) but not AUC >= 0.85 -> FAIL. The secondary max over draws (worst case over
the engine's noise, the robust-reachability reading) meets both on m3, and is the first imagined statistic to bring the
grasp's false alarms well under the 5-10% floor of CF1. Selected after looking, so it needs a confirmatory test (CF3).
Exploratory notes (scratchpad an/cf2_look*.py): 95% of pre-topple decisions have the gripper OPEN (the approach knocks a
neighbour), so the pre-grasp reference contains dangerous states (its q98 gives 8% recall); r is bimodal (about 0.17 in
free space, above 1 near contact), so an early-episode reference is too tight (flags 50% of safe). The m3 50%-recall cut
for max_r 2x (1.336) coincides with LF6's label-free scale thr = 1.3356 (q98 of REAL latent change before the first
grasp, training runs): "the imagined undo leaves more change than anything real changes before contact". At that fixed
threshold on m3: 2x recall 50.0%, safe 5.4%, safe closed 1.9%; 4x recall 69.4%, safe 14.5%, safe closed 8.6%. Recall by
lead (2x): 65 / 50 / 35% at 1 / 2 / 3 chunks. Background motion common to both branches cancels by construction.

### LF4 RESULT (2026-10-25 23:26; results/jenga/lflsf_LF4_dev.json, outcome_audit_lflsfdev4.json): lfc000 selected (dev)
Least-restrictive edits with the label-free value (value_lf_alea.pt, eps 0.00), same 100 dev episodes, replay-audited.
Topples / v4 / v4b / time-outs / red falls before, after extraction; overrides per episode, edit size:
record 27 / 64 / 58 / 2 / 12, 9; **lfc000 (gradient edit under the latent CBF condition V' >= 0.5 V) 2 / 64 / 57 / 21 /
13, 7; 41.2 overrides, 3.6 mm**; lfg000 (gradient edit, level test) 10 / 74 / 64 / 13 / 7, 11; 22.6, 4.2 mm; lfm000 (blend
toward the actor) 8 / 51 / 43 / 28 / 16, 8; 28.0, 8.0 mm. Eligible (v4 >= 59): lfg000, lfc000 -> lfc000 (fewest topples).
Labelled LSF stage 2 (lsfa030) on the same episodes: 9 / 76 / 59, 7.8 overrides. The same label-free value that failed as a
switch (LF3: best lfa-25 11 / 53 / 43) works as an edit: keeping the policy's own command and moving it a few mm along
the value's gradient avoids the actor's retreat from the grasp. Cost: time-outs 2 -> 21 (lfc), 13 (lfg). Exploratory dev
selection on 100 episodes; the overall selection (dev_all, with LF5 / LF6) then goes to a fresh-episode head-to-head.
