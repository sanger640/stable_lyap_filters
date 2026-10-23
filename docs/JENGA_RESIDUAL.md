# Jenga: where G-CEM's remaining topples come from, and the relative-pass gap

User decision 2026-10-22: pillar dropped, all time on Jenga.

## Residual-topple breakdown (exploratory, 2026-10-22 15:30; from the Stage 4 + 4b G-CEM logs, no new runs)
G-CEM (cbfg), 600 episodes (14000-14599), 37 neighbour topples. Window = the 4 decisions up to and including the chunk
where the topple starts.
- 24 of 37 topples had at least one triggered decision in the window; 13 had none.
- The 39 triggered decisions in those windows: 28 certified, 10 fallback_best (no candidate certified), 1 fallback_hold.
- **Relative pass.** The certificate accepts a plan if h >= min(h_now, h_safe) - gamma, where h_now is holding still's
  score. When holding is itself forked, a forked plan passes untouched. Over all 27,780 pre-topple decisions:

| decision type | decisions | followed by a topple within 3 chunks |
|---|---|---|
| passed with h >= h_safe | 25,065 | 80 (0.3%) |
| passed only by the relative rule (h_nom < h_safe) | 403 | 29 (7.2%) |
| triggered (G-CEM acted) | 2,312 | 39 (1.7%) |

  Relative passes are 1.5% of decisions but 24x riskier than absolute passes, and 22 of the 37 topple windows
  contain one. These are exactly the states where neither the plan nor hold is safe and the filter does nothing.

## Relative-pass test (offline; spec fixed 2026-10-22, before any G-CEM run or truth on this set; eval/relpass.py)
Question: at these 403 decisions, does running G-CEM with the ABSOLUTE requirement (h >= h_safe = log2 3, same
o <= O_MAX; OPT3_ABS=1 in eval/opt3.py) produce a chunk with lower real risk than what closed loop executed (the
policy's chunk)?
- Set "relpass": every G-CEM (Stage 4 + 4b) decision with intervened = false and h_nom < log2 3, excluding decisions
  after a topple started (403). Saved states; z0 re-encoded (rng [20261030, ep, k]).
- G-CEM exactly as in closed loop otherwise: opt3 CEM settings, OPT2_CERT = OIL+ENS, model_OIL, anchored at the
  current target, fallback as opt2.finish (cheapest certified, else hold if certified, else best (h, -o)).
- Real risk: benchmark-v2 truth (exact simulator, 32 paired futures per chunk, neighbour topple; red fall secondary).
  Arms: policy chunk (= executed), hold, G-abs.
- **H1 (primary): G-abs has lower mean neighbour risk than the policy chunk, paired bootstrap one-sided p < 0.05.**
- H2: G-abs not worse than hold by more than 0.5 pp (one-sided 95% upper bound of G-abs - hold <= +0.005).
- If H1 passes: pre-register a closed-loop test of G-CEM with the absolute requirement vs current G-CEM on fresh
  episodes. If H1 fails: the relative rule stays, and the gap is reported as a limitation (the fix has no safer move
  from those states).
- Descriptive: certification rate, kinds, progress kept, red-fall risk, hold vs policy.

**Result (2026-10-22 ~21:00; results/program_b/bench2/report_relpass_abs.json): H1 FAIL, H2 FAIL.** 403 decisions.
- Real neighbour risk: policy 5.06%, hold 6.82%, G-abs 7.72%.
  - G-abs minus policy: +2.65 pp (p one-sided 0.9995; better 75 / worse 73).
  - G-abs minus hold: +0.90 pp (upper +2.42 > +0.5).
- G-abs kinds: certified 220, fallback_best 168, fallback_hold 15. Median progress kept 0.93.
- By kind (exploratory):
  - certified and unchanged (< 1 mm, 84): 0.1% vs 0.1%;
  - certified and changed (136, median 5.5 mm): G 5.7% vs policy 4.3%;
  - fallback_best (168): G 13.8% vs policy 8.5%.
- Reading: the relative rule stays. These states are genuinely risky (5.1% vs 0.3% for absolute passes), but the
  certificate cannot find safer chunks: chunks it certifies after modification are riskier than the policy's own.
  The bottleneck is the certificate's ranking of nearby chunks, not the rule. Follow-up: docs/FILTER_DEEP_DIVE.md.
