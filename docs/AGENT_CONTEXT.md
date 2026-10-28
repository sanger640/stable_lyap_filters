# Agent context: the user's decisions, scope and working rules (as of 2026-10-27)

Distilled from the previous agent's working notes so a new agent starts with them. Dates are when the user said it.
When in doubt, ask the user; these are standing instructions, not guesses.

## Research goal and scope

- **Goal (2026-10-01):** a robot "intuitive physics engine" (Battaglia, Hamrick & Tenenbaum 2013): rough noisy
  simulations of an uncertain scene; if imagined outcomes disagree, the scene is precarious. Judge fragility at the
  robot's own uncertainty scale, not millimetre-exact prediction.
- **Core must stay the imagined futures (2026-10-25):** every method reads its judgement off the world model's noisy
  imagined futures (perception + execution noise). Allowed machinery: reachability values, CBF conditions,
  Lyapunov / CLF certificates, time irreversibility, catastrophe theory. Prefer counterfactual statistics (plan vs hold
  from the same state) over references to the past (e.g. the first frame), so background motion cancels.
- **Label-free and task-agnostic:** failure labels are for grading only, never inside the filter or anything it learns
  (barriers, critics, calibrators count as the filter). No Jenga-specific state (block poses, named contacts) in the
  deployable method; proprioception is allowed.
- **Filter = optimisation on the policy's actions** (edit the action; no hand-made primitives or hold heuristics as
  the method; no "pick among policy samples").
- **Main method (2026-10-26/27):** the ALWAYS-ON edit ("cleaner") with the fragile-irreversibility value; gated
  variants are ablations.
- **Scope: Jenga only** (2026-10-22). The task is three upright blocks; pick the middle red one without toppling the
  neighbours (not a tower). Pillar / ManiSkill / LIBERO / SafeManip / ToolHang were closed. A second task
  (pouring or handover) and ManiGuard (BEHAVIOR-1K, on hold; setup at ~/wksp/maniguard) only if the user reopens them;
  any new benchmark starts with cheap due diligence (do real forks occur? does the monitor see them?) before building.
- **Real robot (Franka Panda) comes after Jenga.** Go/no-go for RSS: real-robot data underway by ~2026-11-15.

## Baselines and reporting

- **psel is NOT reported** (2026-10-27: "not runtime worthy", fixes take ~55 s). Keep its code/results; run nothing new.
- Labelled baselines: "LSF as published" (faithful re-implementation) and "UNISafe" (our re-implementation, with
  noise-aware pessimism; also mention the plain version from dev), as UNISafe itself reports LatentSafe.
- Report **v4 and v4b** successes everywhere (replay audit), never the logged lift flag; topples; red-block drops;
  time-outs; task time (time to lift, episode duration) and compute per move.
- **Deferred ablations (2026-10-27: "do at the end, remind me for the paper"):** fragility-value always-on edit and
  gated edit vs the irreversibility always-on edit; barrier strength 0.5 / 0.8 / 0.9 V (dev done); pessimism once vs
  twice at runtime; 0.8 s vs 1.6 s irreversibility horizon (1.6 s dev done; 1.6 s at 0.8 V untested); margin
  conditioned on the policy's real 15-step plan; longer value horizon (only after a model-accuracy check beyond 3.8 s).

## Venue

RSS 2027 Stage 1 extended abstract, Dec 4, 2026 AoE (6 pages, double-blind, judged on hypothesis + methodology);
fallback IROS / RA-L Mar 1, 2027; later full version (CoRL 2027) with real robot; extended journal version (T-RO / IJRR)
after a conference paper. No dual submission. The user decides the venue.

## Working rules (each learned the hard way)

- **Pre-register every test** (protocol JSON in results/jenga/, committed BEFORE any data; selection rules fixed in
  advance, as code where possible). Dev episodes for choosing, fresh episodes for claims. Amend protocols explicitly;
  never change a rule after seeing results. Gates must test the object that will be deployed.
- **docs/FINDINGS.md is the ledger.** Read it before quoting any number; add every result with its source file.
- **Real timestamps:** take them from `date`, never guess.
- **One DINOv2 worker pool on the GPU at a time** (24 GB RTX 3090; OOM otherwise). Value training (one process) can
  share with label generation; not with a 6-shard closed-loop pool.
- **Never use the machine "gpu5090"** (the user's other machine, 2026-10-25).
- **Never start multiprocessing spawn pools from a stdin heredoc** (`python3 - <<EOF`): workers crash-loop for hours.
  Put such code in a file under eval/.
- **Waiting on jobs:** use `$!` from the launching command or verify the PID; `pgrep -f` / `ps | grep` patterns match
  your own command or its wrapper shell (two silent failures on 2026-10-26/27).
- **Do not edit files a running chain imports**; make new behaviour env-gated with the old default unchanged.
- **Other installers** (conda, setup.sh, pip for other projects): run with `PYTHONNOUSERSITE=1` and `PYTHONPATH=` unset;
  the experiment's torch lives in `~/.local` and was uninstalled once by a third-party setup script.
- **Git:** stage explicit paths (never `git add -A`; jobs write into results/). The branch
  `execution-noise-fork-monitor` is pushed: never rewrite its history. Check the largest blobs before pushing.
  Commit messages end with the co-author line the session specifies. Push only when the user says so.
- **Cluster (Alliance):** small runs first, the allocation is scarce; the user does the Duo login (`! ssh -fN fir`);
  never retry failed logins. Never sync filter code to a cluster while a test runs there. Details: cluster/README.md.
- **Dashboard:** minimal; one card per result (headline, one-line takeaway, small chart); detail stays in FINDINGS.
- **Communication:** the user wants plain explanations, honest negatives, and to be asked before costly or
  irreversible steps (fresh test episodes, long cluster runs, deleting data). Report results as they land.

## Known pitfalls in this codebase

- Always set `CBF_WM` and the matching `CBF_CALIB`; otherwise some code paths silently use an older world model.
- Local multi-shard rollouts are not bit-identical to single-process ones (GPU non-determinism); outcomes match.
  Filter arms re-run from scratch diverge from the original run within the first moves.
- `outcome_audit.py` defaults to 2x replay noise: set `AUDIT_NOISE_SCALE=1` for 1x runs.
- The value check pessimism: `LSF_ALEA_SCALE` must match the condition's noise (1 at 1x); the gate's imagined noise is
  `IRR_S` in eval/irr_rollout.py.
- `policy_rollout.py` mode prefixes set some env vars themselves (e.g. `lsfa*` forces value_unisafe_alea.pt); check
  the mode's branch before overriding.
