# Optimisation-only safety filter: redesign and evaluation plan

Written 2026-10-17, before any redesign code or result. Every change to this plan is appended in the "Amendments" section
with its date and reason; nothing is edited silently.

## Why a redesign (evidence, all in docs/FINDINGS.md)

conf1 (pre-registered, 300 fresh episodes, model_OIL): optimisation filter 31 topples vs selection 19 (non-inferiority
failed), success (v4) 226 vs 252 vs policy 216. Diagnosis from the conf1 logs:

1. **No backup.** 12% of the optimiser's interventions executed a chunk that FAILED its own safety rule; such
   interventions are ~10x over-represented in the 4 decisions before a topple (9 of 31 topples vs a 3% base rate).
2. **Optimiser's curse.** 18 of 31 topples followed interventions that PASSED the rule: optimising against one model's
   check finds that model's errors. (Held-out-verifier test: eval/verifier_check.py, result pending at writing.)
3. **Not single-chunk.** 7 of 8 starts are extra policy samples (the given chunk was the start used in 7% of
   interventions): not policy-agnostic, and its diversity came from the policy.
4. **Constraint mismatch.** Optimised a surrogate (relative mean disturbance, alpha) but accepted on the exact fork rule.
5. **Benchmark flaw.** The offline "selection 13.8%" baseline was the OLD first filter (dev3A logs), not the current
   selection filter on the same world model; offline promise (11.8% vs 13.8%) did not carry to closed loop.
   Ruled out: the hand mask hiding neighbours (optimiser outputs cover 1.2% of neighbour volume vs 1.6% for the policy).

## Formulation F (the method; fixed, solver-independent)

- **Input:** one action chunk from any policy: steps 1-8 executed (xyz target + gripper), plus whatever continuation the
  policy provides for the look-ahead (steps 9-15; if none, hold the last point). Nothing else from the policy.
- **Decision variable:** xyz offsets D of steps 1-8 (24 numbers); look-ahead shifted by the step-8 offset; gripper fixed
  (no counterfactual gripper data in the world model; see the gripper note below).
- **Certificate C(u):** the exact rule used by the trigger: h(u) >= min(h_hold, log2 3) and o(u) <= O_MAX, evaluated on
  FRESH imagined futures (64, not the ones used in the search) with the certifier world model(s) (decided by Stage 0).
- **Problem:** minimise cost(D) = ||D||^2 + eta * ||second differences of the path||^2 (mm^2 / 100), subject to C(u(D)).
- **Fallback (backup action):** if no candidate is certified: hold still for one chunk if holding still is certified;
  otherwise the candidate with the best (h, -o). The policy re-plans at the next decision.

## Solvers (two, interchangeable, evaluated separately; no hybrid unless Stage 2 motivates one)

- **S-CEM:** Gaussian over D in a smooth 4-knot basis (12 numbers), mean 0, sd 8 mm (z: 4 mm), population 32, 6 elites,
  3 iterations; candidates ranked by (certified on the search draws, then cost); common random numbers within a decision.
- **S-GRAD:** augmented Lagrangian on a smooth surrogate of C (mean outside-hand disturbance <= O-level of the hold
  chunk, plus the smooth fork margin), Adam, 20 steps, trust region 40 mm, 8 starts = D = 0 plus 7 random smooth
  perturbations (sd 8 mm), all from the one given chunk.
- Both: final candidates certified on fresh draws (+ certifier models); choose the cheapest certified, else fallback.

## Benchmark v2 (offline)

- **Decisions:** states where the trigger fired in conf1 (both filter arms, model_OIL). Dev set: 150 from episodes
  13000-13149; holdout set: 150 from 13150-13299, untouched until Stage 3.
- **Real risk:** exact simulator, 32 futures with the run's perturbation/noise model (as truth_32).
- **Baselines on the same decisions and model:** the policy's chunk; hold still; the CURRENT selection filter (15 fresh
  policy samples + 3 primitives, h_safe log2 3, exactly the closed-loop code path).
- **Metrics:** real risk (primary), certified rate, fallback rate, median change (mm), seconds per decision.

## Stages and gates (decided now)

- **Stage 0, certifier choice:** from eval/verifier_check.py. If chunks optimised against model_OIL and accepted by
  OIL AND all 4 ensemble members have real risk at least 3 points lower than those accepted by OIL alone, with the
  rejected share <= 50%: certifier = OIL + ensemble (all must pass). Otherwise certifier = OIL with fresh draws only.
- **Stage 1, benchmark v2 build + baseline check:** the selection baseline must reproduce its closed-loop advantage over
  the policy chunk offline (real risk clearly lower). If not, the benchmark is not trustworthy: stop and diagnose.
- **Stage 2, solvers on the dev set:** a solver passes if its real risk is <= selection's + 2 points (paired one-sided
  95% upper bound of the difference <= +4 points) with seconds per decision <= 30. Pick the better passing solver.
  Any tuning happens only on the dev set and is logged.
- **Stage 3, holdout:** the chosen solver, frozen, on the holdout set; same criterion. No tuning after this.
- **Stage 4, closed loop:** pre-registered Nibi run, 300 fresh episodes, arms: policy, selection, chosen optimiser, all on
  model_OIL; outcome definition fixed at the time (v4 or v4b per the user); same non-inferiority rule as conf1.

## Notes

- Gripper timing: a later extension (needs counterfactual gripper data first).
- Success definition: v4 (extracted at some point, no neighbour topple, red block not knocked over first); the user
  decides whether a later tip-over when set down is a failure (v4b) before Stage 4.

## Amendments

- **A1 (2026-10-17, before Stage 1 ran):** benchmark decisions are interventions of the SELECTION arm of conf1 only (not
  both arms): its logged executed chunk is then exactly the closed-loop selection output (model_OIL, current settings), so
  the baseline needs no re-run of the policy. Two real-risk labels per chunk: neighbour topple and red block fallen
  (eval/bench2.py). Selection: at most 2 interventions per episode, seeded.
- **Stage 1 result (2026-10-17):** dev set, 150 decisions: neighbour risk policy 2.83%, selection 0.71% (28 better /
  3 worse, upper bound -1.3 points), hold still 0.48%; red-fall risk 7.6% / 7.0% / 4.75%. Gate passed
  (results/program_b/bench2/baseline_dev.json).
- **A2 (2026-10-17, after Stage 1, BEFORE any solver ran):** the Stage 2/3 gate was written for the old benchmark's ~14%
  base risk; here per-decision risks are ~1-3%, so "+2 points" would allow tripling selection's risk. New gate: a solver
  passes if neighbour risk <= selection + 0.5 points AND the paired one-sided 95% upper bound of (solver - selection)
  <= +1.5 points AND red-fall risk <= selection + 1 point, with seconds per decision <= 30. Everything else unchanged.
- **Stage 0 result (2026-10-17 ~10:45):** eval/verifier_check.py (results/jenga/verifier_check.json). Chunks optimised
  against model_OIL: accepted by OIL alone -> real risk 11.0% (91); accepted by OIL AND all 4 ensemble members -> 7.0%
  (54); accepted by OIL but rejected by some member -> 16.8% (37), worse than chunks OIL itself rejects (14.4%).
  Rejected share 37/91 = 41%. Gate (>= 3 points lower, rejected <= 50%) PASSED -> certifier = OIL + ENS0-3 (all must
  pass). Same pattern for chunks optimised against other models (16.3% -> 12.8%) and for non-optimised plans (14.7% -> 12.8%).
- **A3 (2026-10-17, after Stage 0):** Stage 2 is run with OPT2_CERT=OIL+ENS for both solvers (as the plan prescribes).
  The CEM run started earlier with OIL-only certification is kept as an ablation (certifier effect), not as the Stage 2
  result; the OIL-only gradient run is dropped to save time.
- **Ablation result, CEM with OIL-only certification (2026-10-17, dev 150):** neighbour risk 7.6% vs selection 0.71%,
  policy 2.83%, hold 0.48% (worse than the uncertified policy chunk; 31 worse / 7 better vs selection); red-fall 8.3%.
  136 certified, 13 hold, 1 best. DOSE-RESPONSE among certified fixes: change 0-6 mm risk 0.3% (policy 0.3%), 6-10 mm 4.8%
  (2.1%), 10-15 mm 15.1% (4.8%), > 15 mm 12.8% (3.9%). The model certifies mm-scale bends it cannot resolve; real risk is
  mm-sensitive (earlier: a 5 mm nudge moves real risk by 23 points on average). Hold fallback: 0.5% (13).
- **Stage 2 result (2026-10-17 ~15:10), certifier OIL + 4 ensemble members, dev 150:** BOTH SOLVERS FAIL the A2 gate.
  CEM: neighbour risk 5.73% (selection 0.71%, policy 2.83%, hold 0.48%; 23 worse / 7 better vs selection), red-fall 7.5%,
  127 certified / 20 hold / 3 best, 46 s. GRAD: 7.17% (30 worse / 7 better), red-fall 7.6%, 96 certified / 47 hold / 7 best,
  58 s. Dose-response persists with the ensemble certifier: certified fixes of 0-10 mm are about as risky as the policy
  chunk (CEM 6-10 mm: 1.4% vs 1.6%), fixes of 10-15 mm are 2-4x riskier (CEM 8.1% vs 3.7%; GRAD 28% vs 7.3%), > 15 mm
  worse still. Hold-still fallback: 0.3%. The ensemble certifier helped (CEM 7.6% -> 5.7%) but does not remove it.
  Per the plan, no solver goes to Stage 3. Next step to be decided with the user.

## Formulation G: optimise coarse, universal transformations of the one given chunk (pre-registered 2026-10-17)

Written after Stage 2 and BEFORE any G code ran. Motivation: Stage 2 dose-response (the world model judges THAT a moment
is fragile but cannot resolve which chunk 10 mm away is safe; free mm offsets are where it is blind) and the human
intuitive-physics view (a rough engine judges fragility; caution is expressed by changing HOW a motion is executed,
not by re-planning it to the mm). Agreed with the user: no hand-engineered or task-specific safety moves, no extra
policy samples.

- **Input:** ONE chunk (16-step plan p, p[0] = current target), exactly as in F. No other policy candidates or samples.
- **Decision variable theta (4 numbers, box-constrained), all defined from the chunk alone, none assumed safe:**
  - v in [0, 1]: speed (time-scale). Step t is placed at plan time tau_t = v * max(0, t - k) along the plan's own path
    (linear interpolation, clipped to 15).
  - k in [0, 8] steps: pause before moving (continuous, interpolated).
  - s in [0, 1]: amplitude ("do less"): q_t = p0 + s * (P(tau_t) - p0).
  - b in [0, 30] mm: back-off along the REVERSE of the chunk's own net direction u = (p[15] - p[0]) / |.|, ramped
    linearly over steps 1-8 and held after; inactive if |p[15] - p[0]| < 2 mm (no direction to reverse).
  - Gripper column untouched (as in F). Identity = (1, 0, 1, 0); hold still = v = 0 (or s = 0, b = 0).
  - Dropped vs the earlier chat sketch: the free "coarse shift" (a translation is a free spatial offset, the variable
    Stage 2 showed the model cannot resolve).
- **Certificate, cost, fallback:** identical to F (cost on the resulting xyz change of steps 1-8; certificate on fresh
  64 futures with OIL + ENS0-3, all must pass; fallback hold if certified else best (h, -o)).
- **Solvers (separate, evaluated separately):**
  - G-CEM: Gaussian over theta, mean = identity, sd (0.4, 3, 0.4, 12 mm), clipped to the box; population 16,
    3 iterations, 6 elites, 4 finals to certify; common random numbers; same ranking as F's CEM.
  - G-GRAD: augmented Lagrangian on F's smooth surrogate (same constraint code), projected Adam on theta in the box,
    starts = identity + 5 uniform random in the box, 15 steps.
- **Evaluation:** benchmark v2 dev (150) with the A2 gate (vs selection; seconds <= 30), then holdout (Stage 3), then
  closed loop (Stage 4) with an extra arm "hold and re-plan" (trigger fires -> hold one chunk) as the bar to beat.
  Additional offline metric, reported but not gating: progress kept = |q_8 - p0| / |p_8 - p0| (median), and the
  share of decisions whose output is hold-equivalent (progress < 0.1). Note in advance: holding still alone (0.48%)
  already meets the risk part of the gate, so a solver that always holds would pass offline; whether G keeps task
  progress is decided by Stage 4 success (v4/v4b), not offline.
- **Universality test (later, pre-registered now):** the frozen family and ranges, with no retuning, on a second task.
  The task-agnostic claim is only made if it transfers.
- **A4 (2026-10-17, after a 3-decision smoke run, before any G result):** the transformations are anchored at the
  CURRENT target (sim.target), not plan step 0 (the policy's first point; median 4.9 mm away), so that v = 0 is exactly
  holding still as pre-registered. Identity unchanged (steps 1-15 are untouched; the scorer never reads step 0). Smoke
  timing: G-CEM ~35 s, G-GRAD ~64 s per decision (the gate is 30 s); no tuning before the dev run.
- **G-CEM dev result (2026-10-17), certifier OIL + ENS0-3, 150 decisions:** neighbour risk 0.67% (selection 0.71%,
  policy 2.83%, hold 0.48%; 7 better / 10 worse vs selection, paired one-sided upper bound +0.29 points); red-fall 6.98%
  (selection 6.96%, policy 7.62%). Risk parts of A2: all PASS. Seconds: median 34.6 > 30 -> gate FAILS on time only.
  125 certified (risk 0.68% vs policy 2.97% at the same decisions), 17 hold (0.37%), 8 best-fallback (1.17%; red-fall
  32%). Progress kept median 0.64; hold-equivalent 12.7%. Chosen theta median (v 0.98, k 0, s 0.85, b 0.6 mm); s < 0.9
  in 60%, v < 0.9 in 35%, pause in 15%, back-off > 3 mm in 24%. Contrast F (Stage 2): free mm offsets 5.73%.
  results/program_b/bench2/report_dev_g_g_cem.json. Per the plan, not advanced to Stage 3 until the time criterion is
  resolved with the user.
- **A5 (2026-10-17 evening; user asleep, mandate "speed up, then holdout, then closed loop"):**
  1. Exact speed-up: CBF.score skips the largest noise scale (3x). It can never change h (a split only at 3x and no
     split both give h = log2 3) nor o (scale 1), so certificates are unchanged. Verified: 4 + 6 dev decisions re-run,
     chunks bit-identical; G-CEM 36.0 -> 27.0 s per decision on an idle GPU (CBF_SKIP_TOP=1, default now; =0 restores).
     Dev outputs are therefore identical to the reported run -> G-CEM dev, all A2 criteria: PASS (27 s <= 30).
  2. G-GRAD dev run stopped at decision ~15/150: >= 61 s per decision alone on the GPU (65-96 s shared), so it cannot
     meet the 30 s criterion even with the 25% speed-up and cannot be the chosen solver; it was blocking the GPU for
     ~3 h. Its dev risk will be run later for completeness. Chosen solver for Stage 3: G-CEM (the only one that can pass).
- **Stage 3 result (2026-10-17 ~21:45), G-CEM frozen, holdout 150 (untouched until now): PASS on every A2 criterion.**
  Neighbour risk 2.25% (selection 2.38%, policy 5.75%, hold 1.77%; 8 better / 10 worse vs selection, upper bound
  +0.66 points <= +1.5); red-fall 6.21% (selection 6.08%, policy 7.42%, hold 4.08%); 25.9 s per decision; 129 certified,
  15 hold, 6 best; progress kept median 0.67, hold-equivalent 11.3%. Holdout baseline (Stage 1 check on the holdout):
  selection vs policy -3.4 points (33 better / 4 worse). -> Stage 4 submitted on Nibi (job 24351148, 38 x 8 workers,
  episodes 14000-14299, arms record / cbfmsoil / cbfh / cbfg; results/jenga/stage4_protocol.json). Nibi smoke (episode
  19000): G-CEM ~8 s per intervention on an H100.
- **G-GRAD dev (completed 2026-10-18, deferred under A5):** neighbour risk 0.65% (selection 0.71%), red-fall 6.27%,
  certified 135 / hold 10 / best 5, 54.6 s per decision (fails time only), progress kept 0.44, hold-equivalent 27%.
- **Stage 4 result (2026-10-18 ~02:00), 300 fresh episodes 14000-14299 on Nibi, outcome audit by replay
  (results/jenga/outcome_audit_s4.json, eval/stage4_analysis.py -> results/jenga/stage4_result.json):**
  topples / v4 / v4b: policy 69 / 206 / 177; selection 20 / 250 / 212; hold-and-replan 16 / 241 / 209; G-CEM 19 / 246 / 220.
  - G vs selection (primary, non-inferiority): topples -0.3 pp (upper +2.2 < +3: PASS); success v4 -1.3 pp, lower bound
    -5.59 (> -5 required: FAILS by 0.6 pp); success v4b +2.7 pp, lower bound -2.05 (PASS). The verdict therefore depends
    on v4 vs v4b, which the protocol left to the user; this choice is now made after seeing the result and must be
    reported as such.
  - G vs hold-and-replan (secondary): topples +1.0 pp (upper +3.74: NOT non-inferior to simply holding); success v4
    +1.7 pp, v4b +3.7 pp (both n.s.). G is not demonstrably better than holding still for one chunk.
  - Every filter vs the policy: topples -16 to -18 pp (p ~ 1e-11); success v4 +11.7 to +14.7 pp, v4b +10.7 to +14.3 pp.

## G2: gripper time-warped with the arm (pre-registered 2026-10-18, before any G2 code or result)

Found after Stage 4 (user question): in G (and hold-and-replan) the gripper column was copied step-by-step from the
policy, NOT time-warped with the arm. A gripper switch occurs inside 18.2% of G's interventions (13.1% hold, 16.6%
selection; 7.5% of all policy decisions), so slowing / pausing / holding the arm while the gripper switches on the
original clock can grasp or release out of sync with the arm.

- **G2 transformation:** identical to G for xyz; gripper command at step t = the policy's gripper command at plan step
  round(tau_t) (tau_t = v * max(0, t - k), as for xyz), i.e. the gripper follows the arm's progress along the plan.
  Amplitude s and back-off b do not change gripper timing. v = 0 (hold) keeps the step-0 gripper command throughout.
  Still no new gripper decisions (only the policy's own commands, re-timed), so no counterfactual gripper data needed.
- **Hold-and-replan 2 (cbfh2):** hold xyz at the current target AND the gripper at the policy's step-0 command.
- Everything else (certificate, certifier OIL + ENS0-3, cost, solver G-CEM settings, fallback, A5 speed-up) unchanged.
- **Offline gates:** G2-CEM on benchmark v2 dev, then holdout, each with the A2 gate (as G). Fail -> stop and report.
- **Closed loop (Stage 4b), only if both pass:** 300 fresh episodes 14300-14599 on Nibi (as Stage 4), arms record,
  cbfmsoil (selection), cbfh2, cbfg (G, unchanged), cbfg2. Outcomes v4 and v4b (both reported, user's choice).
  Primary: G2 vs selection, non-inferiority as conf1 (success lower bound > -5 pp, topple upper bound < +3 pp), under
  v4 and v4b. Key secondary: G2 vs G (does the gripper fix help: paired difference in topples, v4, v4b, red falls after
  extraction); G2 vs hold-and-replan 2 (same bounds); all vs policy.

**G2 offline results (2026-10-18): both gates PASS.** Neighbour risk dev 0.73% (selection 0.71, policy 2.83, hold 0.48;
G 0.67), holdout 1.94% (selection 2.38, policy 5.75, hold 1.77; G 2.25). Red fall dev 6.6 vs selection 7.0, holdout
6.3 vs 6.1. 26 s per decision; progress kept 0.62 / 0.66; hold-equivalent 13% / 10%.
(results/program_b/bench2/report_{dev,holdout}_g_g2_cem.json)

**Stage 4b amendment 1 (2026-10-18, before any Stage 4b outcome was looked at):** Nibi completed tasks 0-79
(episodes 14400-14479) then stalled ~11 h (ReqNodeNotAvail, nodes draining for an image update). The pending tasks
80-299 were cancelled and resubmitted unchanged (same code as Nibi, copied from Nibi; same models, verified by md5;
same per-episode seeds and arms) on Rorqual. Smoke check on Rorqual, ep 14400 record + cbfg2 vs Nibi: identical
outcomes, chunk counts, intervention chunks (20-23) and monitor scores; max lift differs by <= 2.8 mm (GPU
non-determinism, as seen before between clusters). Two cbfg2 episodes on Nibi (14459, 14465) crashed at start-up
(torch.hub GitHub check returned an HTTP error) and are rerun on Rorqual; not counted as filter failures. Episode
set, arms and analysis unchanged.
Addendum (21:40): the full-H100 Rorqual array also sat in ReqNodeNotAvail for 3 h; it was cancelled before any task
started and resubmitted as one H100 1g.10gb slice job per episode (as the smoke check), same everything else.
