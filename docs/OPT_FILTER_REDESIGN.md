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
