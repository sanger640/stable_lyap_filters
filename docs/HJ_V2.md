# HJ reachability, take 2: what our first attempt missed, and a literature-grounded redesign (Jenga)

User, 2026-10-23: "i think we didnt properly implement it ... hj-reachability has very rich history and results; look into
hj reachability i feel there is some serious potential, ofc do deep literature review". Scope as always: label-free,
task-agnostic, latent space, off-the-shelf WM, the filter optimises the policy's own actions.

## 1. Literature (what a working learned-HJ safety filter needs)

| work | key point for us |
|---|---|
| Fisac et al., ICRA 2019, "Bridging HJ safety analysis and RL" | Safety value V(x) = max_u min_t l(x_t) (avoid). The discounted safety Bellman equation V = (1-g) l + g min{l, max_u V(x')} is a contraction, so V can be learned by value iteration / RL from sampled transitions instead of grids. |
| Hsu, Hu, Fisac, Annual Review 2023, "The Safety Filter: A Unified View" | Every filter = a monitor (is the nominal action safe w.r.t. V?) + an intervention (least-restrictive switch, or a CBF-like projection). Our trigger + G-CEM fits; what was missing is a V carrying multi-step information. |
| So, ..., Fan, ICRA 2024, "How to Train Your Neural CBF" (PNCBF) | The value of the NOMINAL policy, V^{h,pi}(x) = sup_t h(x_t^pi), is itself a CBF (Thm 1). It is learned from rollouts of pi with target max{max_{t<=s<=T} h(x_s), V(x_T)} (Monte-Carlo max + bootstrap). Filtering with it can only enlarge pi's safe set (Thm 2); iterating = policy iteration. No optimal-control HJ solve, no exploration needed. |
| Nakamura, ..., Bajcsy, L4DC 2026, "How to Train Your Latent CBF" (arXiv 2511.18606) | (a) Thm 3: Lipschitz(V) <= Lipschitz(l) x const: a classifier-like, saturated margin gives a saturated value with jumps, useless for ranking actions; a smooth margin (gradient penalty) fixes it. (b) Train the value on a MIX of nominal- and safety-policy data, or Q is wrong exactly at the actions the filter evaluates. (c) Filter: min ||a - a_nom|| s.t. Q(z,a) - eps >= alpha (Q(z, pi_safe) - eps), solved by sampling. Franka + diffusion policy: success 38% -> 80%, fails 62% -> 0%. |
| lambda-Reachability (arXiv 2606.16022) | Multi-step safety Bellman target y = max{l_t..l_{t+n-1}, s V(x_{t+n})}, n ~ Geom(1-lambda) (TD(lambda)-like). One-step TD (Fisac DPE) propagates danger poorly over long horizons: lambda 0.99 vs one-step, sim false-positive rate 0.2-6% vs 49-60%; hardware 20-31% vs 45-72%. Offline data from rollouts. |
| Nakamura, Peters, Bajcsy, RSS 2025 (latent safety filter); Seo et al. CoRL 2025 (UNISafe) | Latent HJ with a LABELLED failure classifier as margin; our head-to-head baseline (G-CEM 19 vs 55 topples). |
| ISAACS (Hsu et al., L4DC 2023); robust / game HJ | Worst-case disturbance player; our fork test's perturbation scales are the sampled analogue. |
| EigenSafe (RSS 2026); stochastic reach-avoid RL (arXiv 2605.11975) | Probabilistic safety values for stochastic dynamics (safety probability via a linear DP operator). |
| Solanki et al. 2026 (arXiv 2601.08050, 2602.16475) | Formal link RL value iteration -> HJ viscosity solution; certification of RL-learned reachable tubes. |

## 2. What our first attempt (docs/HJ_REACHABILITY.md, ideas 2-3) got wrong
1. **One-step, not dynamic programming.** Idea 2 asked "after the policy's next chunk, does some backup certify?" (a
   one-step reach check); idea 3 regressed a binary one-step label. Neither did a Bellman recursion, so neither could carry
   danger information more than one chunk back. Multi-step propagation is the whole point of HJ (lambda-Reachability:
   one-step targets have FPR 49-60%).
2. **Binary, saturated margin.** Our margin was the certificate's yes/no. The logged fork distance h_now is at its cap
   (log2 3) in 94% of 29,022 G-CEM decisions and takes only 4 values; along topple windows it jumps (1.6, 1.6, -1, 1.6, ...).
   Latent-CBF Thm 3 predicts exactly the flat value we got (idea 3: V at the 98% base rate).
3. **Optimal-control viability instead of the policy's value.** "Does ANY backup survive?" is almost always yes on Jenga
   (hold is nearly always viable): least-restrictive by construction, so it can only act at the last moment. The
   PNCBF quantity, danger along what THIS policy (+ filter) will actually do, is the one that carries lead time, and it is
   learnable from logged rollouts without running the policy in imagination (our structural limit).
4. **No data mix.** Idea 3 trained on 3000 decisions; the logged closed-loop data (policy-alone + filtered arms) is ~100x
   larger and contains both nominal and filter actions.

## 3. Redesign (each stage gated; pre-registered before its numbers are seen)
**Stage A: a smooth, continuous, label-free margin l(z).** From the same fork test on the HOLD plan at the current state
(state-only), keep the continuous statistics instead of the yes/no:
- l_D(s) = 2 - Ashman D of the imagined endings at scale s (D > 2 = split), s in {2, 4};
- l_h = continuous fork distance: log2 of the scale where D crosses 2 (log-linear interpolation over 0.5, 1, 2, 3, 4, 8),
  capped at log2 8;
- l_sens = -log of the futures' spread per unit perturbation at s = 2 (raw sensitivity of the outcome to the action
  perturbation, our monitor's core idea).
Gate A (dev episodes): smoothness = median |l_{k+1} - l_k| / IQR(l) along safe trajectories (lower is better) and
information = AUC(-l, decisions 1-3 chunks before a topple vs safe-episode decisions). A margin passes if AUC >= 0.70 and it
is smoother than the binary fork distance; the best passing margin goes to stage B.
**Stage B: the policy's safety value V(z)** (PNCBF + lambda-Reachability), a small conv net on the 28x28x8 latent (+ gripper
state), trained offline on logged closed-loop trajectories (policy-alone AND G-CEM arms, all past tests, held-out tests
excluded) with targets y = min{l_t, ..., l_{t+n-1}, V(z_{t+n})}, n ~ Geom(1 - lambda), lambda = 0.9 (decisions are 8
actions, ~10 chunk horizon). Labels never used in training.
Gate B (held-out policy-alone episodes): the trigger "V(z) < tau" with tau set to the trend rule's alarm rate (12.7%) must
warn >= 3 chunks ahead in more topple episodes than the trend rule (87%), or reach the same lead at a lower alarm rate.
**Stage C: V in the filter.** Model-based discrete-time CBF condition on the imagined next state: candidate chunk a is
admissible if V(f(z, a)) >= min(V(f(z, hold)), (1 - alpha) V(z)) (Latent-CBF form, our hold backup), added to the G-CEM
certificate; trigger = nominal fails it. Offline on benchmark v2, then closed loop vs the trend winner.

## 4. Stage A result (results/jenga/hj2/margin_A.json; 3366 decisions, 55 topple episodes, 2761 safe decisions)

| margin | distinct values (safe) | median step / IQR | AUC 1-3 before | AUC 4-6 | AUC 7-10 |
|---|---|---|---|---|---|
| h_bin | 7 | 0.0 | 0.726 | 0.588 | 0.522 |
| h_cont | 744 | 0.0 | 0.725 | 0.589 | 0.520 |
| negG_2 | 2744 | 0.64 | 0.542 | 0.446 | 0.477 |
| negG_4 | 2753 | 0.71 | 0.687 | 0.559 | 0.544 |
| negG_max | 2751 | 0.57 | 0.775 | 0.657 | 0.578 |
| neg_log_sens_2 | 2410 | 0.5 | 0.648 | 0.611 | 0.594 |
| neg_log_sens_4 | 2492 | 0.26 | 0.787 | 0.653 | 0.565 |

Gate as written: **FAIL**. The smoothness reference is degenerate: h_bin's median consecutive step is 0 because it sits at
its cap, so nothing can be "smoother" by that statistic. On information, two continuous margins beat the binary fork
distance: the max BIC gain over scales (negG_max, AUC 0.775 / 0.657) and the outcome sensitivity at 4x (neg_log_sens_4,
0.787 / 0.653, also the smoothest). Stage B continues as an exploratory step, flagged, with negG_max (pre-declared
tie-break: highest AUC 4-6) and neg_log_sens_4 as secondary; stage B's held-out gate decides.
