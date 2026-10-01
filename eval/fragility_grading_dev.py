"""Fragility at the agent's own uncertainty scale: is the aware monitor's caution warranted?

Frozen protocol: results/jenga/fragility_grading_protocol.json (development evidence, Jenga DEV).
Ground truth for (state, sigma): aware sampling centred on the TRUE state (probe j = execution
snippet j + state sample s + e_j), judged by monitor v1's split rule, computed with two independent
seeds (A, B). The option-2 rollouts (results/program_b/uncertainty) are graded against truth A.
Replay is the same bit-identical stage-0 chain as eval/uncertainty_monitor_dev.py.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

import uncertainty_monitor_dev as um  # noqa: E402

PROTOCOL = ROOT / "results/jenga/fragility_grading_protocol.json"
OUTPUT_DIR = ROOT / "results/program_b/fragility"
OPTION2_DIR = ROOT / "results/program_b/uncertainty"
RESULT = ROOT / "results/jenga/fragility_grading_result.json"
SIGMAS_MM = (1.0, 2.65)
TRUTH_SEEDS = {"A": 20261011, "B": 20261012}     # disjoint from option 2's 20260929/20261001/20261002


def truth_sample(key, sigma, probe, seed):
    rng = np.random.default_rng([seed, *key, probe, int(round(sigma * 1e6))])
    import program_b_reference as pr
    return rng.normal(0, sigma, (3, 3)), rng.normal(0, sigma / pr.HALF_LENGTH, (3, 3))


def simulate_episode(job):
    episode_id, all_starts, targets, output = job
    path = Path(output) / f"ep{episode_id}.npz"
    if path.exists():
        return path.name
    from jenga_runtime import DEFAULT_LMDB, JengaReplay
    from jenga_short_held_tails import DirectJengaSim, extract_sim
    snippets = np.load(um.BENCH)["snippets"]
    replay = JengaReplay(DEFAULT_LMDB)
    actions = np.asarray(replay.episode(episode_id).actions, np.float32)
    replay.close()
    sim = DirectJengaSim(str(extract_sim(um.SIM_ARCHIVE, tempfile.mkdtemp(prefix="fragility_"))))
    result, last_target = {}, max(targets)
    try:
        sim.reset(int(episode_id))
        for index, action in enumerate(actions):
            if index in all_starts:
                chunk = actions[index:index + um.HORIZON]
                snapshot = sim.snapshot()
                um.rollout(sim, snapshot, chunk, np.zeros((um.HORIZON, 3)))
                for scale in um.SCALES:
                    traces = [um.rollout(sim, snapshot, chunk, noise) for noise in snippets * scale]
                    if scale == 1.0 and index in targets:
                        result[f"s{index}_sigma_0"] = np.stack(traces).astype(np.float32)
                warm_end = sim.data.qacc_warmstart.copy()
                if index in targets:
                    key = (int(episode_id), index)
                    for sigma_mm in SIGMAS_MM:
                        sigma = sigma_mm / 1000
                        for name, seed in TRUTH_SEEDS.items():
                            traces = [um.rollout(sim, snapshot, chunk, noise,
                                                 [truth_sample(key, sigma, j, seed)])
                                      for j, noise in enumerate(snippets * um.SCALES[1])]
                            result[f"s{index}_truth_{sigma_mm:g}_{name}"] = np.stack(traces).astype(np.float32)
                sim.restore(snapshot)
                sim.data.qacc_warmstart[:] = warm_end
                if index == last_target:
                    break
            sim.execute(action)
    finally:
        sim.close()
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **result)
    tmp.replace(path)
    return path.name


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = np.mean(a == b)
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def grade(decision, truth):
    from monitor_v1_eval import wilson
    decision, truth = np.asarray(decision, bool), np.asarray(truth, bool)
    fragile, robust = truth.sum(), (~truth).sum()
    hit, false = int((decision & truth).sum()), int((decision & ~truth).sum())
    return {"agreement": float(np.mean(decision == truth)), "kappa": kappa(decision, truth),
            "fragile_states": int(fragile), "recall": hit / fragile if fragile else None,
            "recall_wilson": wilson(hit, int(fragile)), "robust_states": int(robust),
            "alarm_on_robust": false / robust if robust else None,
            "alarm_on_robust_wilson": wilson(false, int(robust)), "alarms": int(decision.sum())}


def score():
    from monitor_v1_eval import NON_ROBOT, decide, trace_endings
    bench = np.load(um.BENCH)
    classes = {(str(e), int(c)): cls for e, c, cls in
               zip(bench["dev_episode"], bench["dev_chunk"], bench["dev_classes"][:, 1])}
    cache = np.load(um.CACHE)
    rows = json.loads(um.STAGE0.read_text())["rows"]
    cache_index = {(r["episode_id"], int(r["chunk_start"])): i for i, r in enumerate(rows)}
    truth, monitors, floor = {}, {}, 0.0
    for path in sorted(OUTPUT_DIR.glob("ep*.npz")):
        episode = path.stem[2:]
        with np.load(path) as f, np.load(OPTION2_DIR / path.name) as g:
            for key in sorted({k.split("_")[0] for k in f.files}):
                state = (episode, int(key[1:]))
                i = cache_index[state]
                for step, slot in ((17, 1), (37, 4)):
                    floor = max(floor, 1000 * float(np.abs(
                        f[f"{key}_sigma_0"][:, step, :9] - cache["start_pose"][i][:9]
                        - cache["pose"][i, 1, :, slot, :9]).max()))
                for sigma in SIGMAS_MM:
                    for seed in TRUTH_SEEDS:
                        truth[(sigma, seed, state)] = decide(*trace_endings(
                            f[f"{key}_truth_{sigma:g}_{seed}"], NON_ROBOT["jenga"]))[0]
                for name in ("sigma_0", "naive_1", "aware_1", "naive_2.65", "aware_2.65"):
                    monitors[(name, state)] = decide(*trace_endings(g[f"{key}_{name}"], NON_ROBOT["jenga"]))[0]
    if floor > 0.0:
        raise SystemExit(f"sigma_0 does not reproduce the cache: {floor} mm")
    states = sorted({k[2] for k in truth})
    out = {"states": len(states), "replay_floor_mm": floor, "by_sigma": {}}
    for sigma in SIGMAS_MM:
        a = [truth[(sigma, "A", s)] for s in states]
        b = [truth[(sigma, "B", s)] for s in states]
        row = {"truth_fragile_A": int(np.sum(a)), "truth_fragile_B": int(np.sum(b)),
               "ceiling_A_vs_B": grade(b, a), "monitors": {}}
        for name in ("sigma_0", f"naive_{sigma:g}", f"aware_{sigma:g}"):
            row["monitors"][name] = grade([monitors[(name, s)] for s in states], a)
        crosstab = {}
        for cls in ("topple_fork", "nudge_fork", "small_split", "quiet", "weak"):
            members = [s for s in states if classes[s] == cls]
            crosstab[cls] = {"states": len(members),
                             "truth_fragile_A": int(sum(truth[(sigma, "A", s)] for s in members))}
        row["truth_vs_exact_classes"] = crosstab
        quiet_alarms = [s for s in states if classes[s] == "quiet" and monitors[(f"aware_{sigma:g}", s)]]
        row["aware_quiet_alarms"] = {"count": len(quiet_alarms),
                                     "truth_fragile_A": int(sum(truth[(sigma, "A", s)] for s in quiet_alarms))}
        out["by_sigma"][f"{sigma:g}"] = row
    r = out["by_sigma"]["2.65"]
    aware, ceiling = r["monitors"]["aware_2.65"], r["ceiling_A_vs_B"]["kappa"]
    out["decision"] = {"calibrated": bool(aware["kappa"] >= 0.8 * ceiling and aware["recall"] is not None
                                          and aware["recall"] >= 0.80 and aware["alarm_on_robust"] <= 0.20),
                       "aware_kappa": aware["kappa"], "ceiling_kappa": ceiling}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default=str(OUTPUT_DIR))
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("simulator work is capped at four workers")
    if not json.loads(PROTOCOL.read_text())["status"].startswith("frozen"):
        raise SystemExit("protocol not frozen")
    Path(args.output).mkdir(parents=True, exist_ok=True)
    if args.simulate:
        todo = um.jobs(args.output)[:args.limit or None]
        with ProcessPoolExecutor(args.workers) as pool:
            for number, name in enumerate(pool.map(simulate_episode, todo), 1):
                print(f"{number}/{len(todo)} {name}", flush=True)
    if args.score:
        result = score()
        result.update({"protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
                       "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "created": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        RESULT.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
