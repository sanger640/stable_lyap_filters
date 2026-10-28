# Handoff (2026-10-27): label-free safety filter on Jenga

For the next agent continuing this project on the same machine. Read this file, then `docs/AGENT_CONTEXT.md`
(the user's decisions and working rules), then `docs/FINDINGS.md` (every result with its source file). The previous
handoff (2026-09-29, an earlier phase) is in `docs/history/`; README.md and the top-level *.md notes are historical too.

## 1. The project in one paragraph

Goal (user's framing): give a robot a rough, noisy "intuitive physics engine" (Battaglia et al. 2013): imagine many
slightly-off futures of an action in a learned world model and use them to judge fragility, then filter a learned
policy's actions with a reachability / control-barrier-style rule. No failure labels, task-agnostic. Testbed: a
simulated Franka (MuJoCo) picks the MIDDLE of three upright Jenga-like blocks without toppling its two neighbours
(not a tower). Policy = a diffusion policy (`data/jenga/policy/latest.ckpt`). World model = `model_OIL` (DINOv2 patch
latents 28x28x8, deterministic, `results/program_b/oil_v1/model/model_OIL.pt`). Stress setting = 2x the arm's measured
execution noise; the arm's real noise = 1x.

## 2. Where things stand

**Method (frozen by the pre-registered rule, 2026-10-27):** an *always-on* least-restrictive edit. Every 0.2 s (one
world-model step) it imagines the policy's next 2 commands under 8 noisy draws and checks a control-barrier condition
on a learned label-free value: min_j V(z'_j) >= (1 - gamma) V(z), with gamma = 0.1 (barrier "90%"). If it fails, a
gradient step nudges the policy's xyz targets by at most 1 cm (gripper unchanged; safety-actor fallback as in all runs).
The value `value_irr8_alea.pt` is learned in imagination (UNISafe-style SAC recipe) with a **fragile-irreversibility**
failure set: an action fails if, in the worst of 64 noisy imagined futures, doing it for 0.8 s and undoing it leaves
lasting latent change outside the hand vs holding still (threshold 1.3356 = label-free calibration on real latent change
before the first grasp). Code: `eval/policy_rollout.py paired lfc000` with `LSF_VALUE_NAME=value_irr8_alea.pt
LSF_CBF_GAMMA=0.1` (and `LSF_ALEA_SCALE` = the condition's noise); filter logic `eval/cbf_filter.py` `lsf_edit`;
value training `eval/lsf.py sac` with `LSF_IRR_MARGIN` / `LSF_MARGIN_MODE=irr`; margin network `eval/irrm2.py`.

**Dev evidence (100 episodes 15000-15099; topples / v4 / v4b):** 2x: policy 27 / 64 / 58, ours (90%) 7 / 72 / 58,
ours (80%) 14 / 78 / 68, UNISafe + noise-aware pessimism 9 / 76 / 59, latent safety filter as published 26 / 65 / 58.
1x: policy 5 / 85 / 76, ours (90%) 1 / 90 / 80, ours (80%) 1 / 92 / 82. Real-time on the RTX 3090.

**Why this method (the story of the last week, details in `docs/REVIEW_2026-10-27.md`):** the earlier failure set
(fragility = most fragile 10% of visited states) called ~13% of the policy's normal states failures and was blind to
the red block dropping, so the always-on edit stalled and interfered at 1x; fragile irreversibility fixed both.

**Labelled baselines (fidelity audit `docs/LSF_FIDELITY_AUDIT.md`):** "LSF as published" (faithful DDPG re-implementation,
`value_pub_code.pt`, eps 0.2; failed its training gate: saturated actor; no effect in closed loop) and "UNISafe +
noise-aware pessimism" (`value_unisafe_alea.pt`, mode `lsfa030`; the strongest labelled baseline). psel (an older
policy-sample-selection filter) is NOT reported: not real-time (user decision).

## 3. The next step: the decisive fresh test (H2H5) - needs the user's go

Pre-registration draft: `results/jenga/h2h5_protocol.json` (600 fresh episodes 26200-26799, 2x and 1x noise, 96 s cap,
arms policy / ours90 / ours80 / UNISafe / LSF-as-published, primary hypotheses H1-H4, task-time outcomes).
Runner: `scripts/run_h2h5.sh` (~10-12 h locally). Analysis: `eval/h2h5_analysis.py` (written before any H2H5 data;
dry-run tested on dev data). **Do not launch without the user's explicit go** (it consumes the fresh episodes).
Dev-scale caution: at n = 100 the primary arm (90%) did not meet H1's success bound and the 80% arm did; at n = 600 the
power is higher - report as registered, do not swap arms after seeing results.

After H2H5: record in FINDINGS + dashboard; then the paper (RSS 2027 Stage 1, Dec 4, 2026); ablations at the end
(list in `docs/AGENT_CONTEXT.md` section "Deferred ablations"); real robot after Jenga.

## 4. How to run things

- Python: `/home/corey/wksp/dp_venv/bin/python` (see `docs/ENVIRONMENT.md`; torch lives in `~/.local`, fragile).
- One closed-loop dev arm: see `scripts/chains/irrm_lib.sh` (`cl tag exec_noise imagined_noise value gamma`): 6 shards of
  `eval/policy_rollout.py paired <mode> <n>` with `EP_OFFSET`, `SHARD=i/6`, `PAIRED_TAG`, `PER_EPISODE_SEEDS=1`,
  `PAIRED_PER_EP=1`, `HARD_OFF=1`, `EXEC_NOISE_SCALE`, `CBF_WM`, `CBF_CALIB=results/jenga/cbf_calibration_OIL.json`,
  then `cluster/merge_paired.py <runs>` and `eval/outcome_audit.py runs <runs>` with `AUDIT_OUT=...` and
  `AUDIT_NOISE_SCALE` = the run's noise. Outcomes: ALWAYS the replay audit (v4 and v4b), never the logged success flag.
- Modes: `record` (policy alone), `lfc000` (always-on edit), `lsfa030` (UNISafe + noise pessimism), `lsfp<eps>`
  (LSF as published), `eval/irr_rollout.py lfci000|lfcj000|lfck000` (gated edits, ablations).
- Selection rules as code: `eval/irrm_select.py`; offline gates: `eval/irrm_gate.py`, `eval/lsf_pub_gate.py`.
- Every past experiment's exact launch recipe: `scripts/chains/` (historical; PIDs in wait loops are dead).
- Videos: `eval/filter_videos.py A|B [ep...]` (annotated four-way videos; outcomes from the audit).
- Dashboard source: `dashboard/index.html` (the live page belongs to the user's claude.ai account; a new account must
  publish its own copy). Videos it references: `results/program_b/filter_videos/`.

## 5. Repo map (what matters now)

- `docs/FINDINGS.md`: the ledger. Newest first. Quote numbers only from here or the JSON it cites.
- `docs/REVIEW_2026-10-27.md`: step-back analysis, root causes, self-reflection.
- `docs/LSF_FIDELITY_AUDIT.md`, `docs/LATENT_SAFETY_FILTER.md`: baselines.
- `results/jenga/*_protocol.json`: every pre-registration (irrm_protocol.json = the method's development, with amendments).
- `results/jenga/outcome_audit_*.json`: replay audits per run; `paired_*.jsonl`: per-episode logs.
- `docs/MANIFEST.md`: models / data that are not in git (all on this machine) with sizes and checksums.
- `eval/`: all code (flat); `src/`: world model + simulator glue; `cluster/`: Alliance sbatch scripts.

## 6. Loose ends

- 7 files were modified in the working tree before the 2026-10-26/27 sessions (eval/cbf_qp.py, eval/hyb_reliability.py,
  paper/submissions/physwm_oct7/paper.pdf, results/jenga/outcome_audit_runs.json, results/program_b/bench2/truth_*.json,
  results/program_b/cbf_qp/reliability_hyb.json). Their provenance is unclear; they were left uncommitted on purpose.
  Ask the user before committing or reverting them.
- Raw per-episode rollouts (`results/program_b/policy_rollouts/`, ~2 GB) and other large arrays stay untracked (manifest).
- Cluster (Alliance: Fir/Nibi/Rorqual) needs the user's Duo login; H2H5 does not need the cluster.
