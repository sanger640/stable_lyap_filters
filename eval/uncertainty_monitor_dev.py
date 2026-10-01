"""Option 2: an uncertainty-aware boundary monitor on Jenga DEV (frozen protocol, development only).

Protocol: results/jenga/uncertainty_monitor_protocol.json. The monitor sees a misperceived state
s_hat (one perception draw per state, as in R). The NAIVE monitor rolls all 64 execution probes
out from s_hat; the AWARE monitor also gives probe j its own state sample s_hat + e_j from the same
perception-noise model. Both are judged by monitor v1's split rule on non-robot positions at hold
10 and 30 (and v0's 64-probe candidate stage), against the TRUE state's frozen benchmark class.

States are reached exactly as eval/jenga_stage0_noise_oracle.py reached them: reset(episode id),
replay, and at every screened chunk start snapshot -> nominal -> 0.5x/1x/2x probes -> restore ->
continue. Only the rollout immediately before a probe (or before the continuation) sets the solver
warm start, so we run 0.5x probe 63 before the 1x probes and 2x probe 63 before restoring. That
makes sigma = 0 bit-identical to the benchmark's cached endings.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

PROTOCOL = ROOT / "results/jenga/uncertainty_monitor_protocol.json"
BENCH = ROOT / "results/jenga/bench/jenga_bench.npz"
STAGE0 = ROOT / "results/jenga/holdout_stage0.json"
CACHE = ROOT / "results/jenga/holdout_stage0_cache.npz"
OUTPUT_DIR = ROOT / "results/program_b/uncertainty"
RESULT = ROOT / "results/jenga/uncertainty_monitor_result.json"
SIM_ARCHIVE = ROOT / "vendor/panda_express_sim.tar"
HORIZON, HOLD = 8, 30
SCALES = (0.5, 1.0, 2.0)
SIGMAS_MM = (1.0, 2.65)
NOISE_SEED = 20261002


def conditions():
    out = [("sigma_0", 0.0, False)]
    for s in SIGMAS_MM:
        out += [(f"naive_{s:g}", s / 1000, False), (f"aware_{s:g}", s / 1000, True)]
    return out


def perception_draw(key, sigma):
    """One perceived-state error (positions m, rotation vectors) per (state, sigma)."""
    import program_b_reference as pr
    return pr.pose_noise(key, sigma)


def probe_sample(key, sigma, probe):
    """The aware monitor's own state sample for probe j, from the same noise model."""
    rng = np.random.default_rng([NOISE_SEED, *key, probe, int(round(sigma * 1e6))])
    import program_b_reference as pr
    return rng.normal(0, sigma, (3, 3)), rng.normal(0, sigma / pr.HALF_LENGTH, (3, 3))


def rollout(sim, snapshot, chunk, noise, perturbations=()):
    """Stage-0 rollout (probe actions, then hold on the nominal last target), recording the
    61-D state after every command. ``perturbations`` are applied to the blocks after restore."""
    import program_b_reference as pr
    from jenga_state_data import step_state
    sim.restore(snapshot)
    for position, rotation in perturbations:
        pr.perturb_blocks(sim, position, rotation)
    actions = np.asarray(chunk, np.float32).copy()
    actions[:, :3] -= noise
    states = []
    for action in actions:
        sim.execute(action); states.append(step_state(sim))
    for _ in range(HOLD):
        sim.execute(chunk[-1]); states.append(step_state(sim))
    return np.stack(states)


def simulate_episode(job):
    episode_id, all_starts, targets, output = job
    path = Path(output) / f"ep{episode_id}.npz"
    if path.exists():
        return path.name
    from jenga_runtime import DEFAULT_LMDB, JengaReplay
    from jenga_short_held_tails import DirectJengaSim, extract_sim
    snippets = np.load(BENCH)["snippets"]
    replay = JengaReplay(DEFAULT_LMDB)
    actions = np.asarray(replay.episode(episode_id).actions, np.float32)
    replay.close()
    sim = DirectJengaSim(str(extract_sim(SIM_ARCHIVE, tempfile.mkdtemp(prefix="unc_mon_"))))
    result = {}
    last_target = max(targets)
    try:
        sim.reset(int(episode_id))
        for index, action in enumerate(actions):
            if index in all_starts:
                chunk = actions[index:index + HORIZON]
                snapshot = sim.snapshot()
                # The original stage-0 chain, in order: each rollout's warm start comes from the
                # previous one, so the whole chain is replayed. sigma_0 = its own 1x probes.
                rollout(sim, snapshot, chunk, np.zeros((HORIZON, 3)))
                for scale in SCALES:
                    traces = [rollout(sim, snapshot, chunk, noise) for noise in snippets * scale]
                    if scale == 1.0 and index in targets:
                        result[f"s{index}_sigma_0"] = np.stack(traces).astype(np.float32)
                warm_end = sim.data.qacc_warmstart.copy()
                if index in targets:
                    key = (int(episode_id), index)
                    for name, sigma, aware in conditions()[1:]:
                        hat = perception_draw(key, sigma)
                        traces = []
                        for j, noise in enumerate(snippets * SCALES[1]):
                            perturbations = [hat] + ([probe_sample(key, sigma, j)] if aware else [])
                            traces.append(rollout(sim, snapshot, chunk, noise, perturbations))
                        result[f"s{index}_{name}"] = np.stack(traces).astype(np.float32)
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


def jobs(output):
    bench = np.load(BENCH)
    rows = json.loads(STAGE0.read_text())["rows"]
    starts = {}
    for r in rows:
        starts.setdefault(r["episode_id"], set()).add(int(r["chunk_start"]))
    targets = {}
    for e, c in zip(bench["dev_episode"], bench["dev_chunk"]):
        targets.setdefault(str(e), set()).add(int(c))
    return [(e, starts[e], targets[e], str(output)) for e in sorted(targets, key=int)]


def probe_actions(chunk, snippets):
    """(64, 38, 4) commanded actions: the 1x probe chunk, then the nominal last target held."""
    out = []
    for noise in snippets:
        a = np.asarray(chunk, np.float32).copy()
        a[:, :3] -= noise
        out.append(np.concatenate([a, np.repeat(np.asarray(chunk[-1:], np.float32), HOLD, 0)]))
    return np.stack(out)


def score(output, workers):
    from monitor_v1_eval import NON_ROBOT, decide, trace_endings, wilson
    from program_a_feasibility import _init_worker, evaluate_candidates
    from jenga_runtime import DEFAULT_LMDB, JengaReplay
    bench = np.load(BENCH)
    snippets = bench["snippets"]
    classes = {(str(e), int(c)): cls for e, c, cls in
               zip(bench["dev_episode"], bench["dev_chunk"], bench["dev_classes"][:, 1])}
    # Exactness is checked against the stage-0 cache itself (its own start pose); the bench's
    # dev_start comes from a different (rendering) replay and differs by ~18 micrometres.
    cache = np.load(CACHE)
    cache_rows = json.loads(STAGE0.read_text())["rows"]
    cache_index = {(r["episode_id"], int(r["chunk_start"])): i for i, r in enumerate(cache_rows)}
    replay = JengaReplay(DEFAULT_LMDB)
    names = [c[0] for c in conditions()]
    decisions = {n: {} for n in names}
    candidate = {n: {} for n in names}
    floor = 0.0
    from concurrent.futures import ProcessPoolExecutor as Pool
    pool = Pool(workers, initializer=_init_worker)
    for path in sorted(Path(output).glob("ep*.npz")):
        episode = path.stem[2:]
        chunk_actions = np.asarray(replay.episode(episode).actions, np.float32)
        with np.load(path) as f:
            stored = {k: f[k] for k in f.files}
        for key in {k.split("_")[0] for k in stored}:
            index = int(key[1:])
            state = (episode, index)
            actions = probe_actions(chunk_actions[index:index + HORIZON], snippets)
            for name in names:
                traces = stored[f"{key}_{name}"]
                if name == "sigma_0":
                    i = cache_index[state]
                    for step, slot in ((17, 1), (37, 4)):
                        floor = max(floor, float(np.abs(traces[:, step, :9] - cache["start_pose"][i][:9]
                                                        - cache["pose"][i, 1, :, slot, :9]).max()) * 1000)
                decisions[name][state] = decide(*trace_endings(traces, NON_ROBOT["jenga"]))[0]
            batch = evaluate_candidates(pool, [actions] * len(names),
                                        [stored[f"{key}_{n}"] for n in names])
            for name, c in zip(names, batch):
                candidate[name][state] = bool(c is not None and c[0])
    pool.shutdown()
    replay.close()
    if floor > 0.0:
        raise SystemExit(f"sigma_0 does not reproduce the benchmark endings: max {floor} mm")

    def table(choice):
        out = {}
        for cls in ("topple_fork", "quiet", "nudge_fork", "small_split", "weak"):
            states = [s for s in classes if classes[s] == cls]
            k = int(sum(choice[s] for s in states))
            out[cls] = {"states": len(states), "alarms": k,
                        "rate": k / len(states) if states else None, "wilson": wilson(k, len(states))}
        return out
    summary = {n: {"v1": table(decisions[n]), "v0_candidate": table(candidate[n])} for n in names}
    n265, a265 = summary["naive_2.65"]["v1"], summary["aware_2.65"]["v1"]
    useful = bool(a265["topple_fork"]["rate"] >= 0.70 and n265["topple_fork"]["rate"] < 0.50
                  and a265["quiet"]["rate"] <= 0.25)
    return {"replay_floor_mm": floor, "summary": summary, "decision": {"useful": useful}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--score-workers", type=int, default=16)
    parser.add_argument("--output", default=str(OUTPUT_DIR))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("simulator work is capped at four workers")
    if not json.loads(PROTOCOL.read_text())["status"].startswith("frozen"):
        raise SystemExit("protocol not frozen")
    Path(args.output).mkdir(parents=True, exist_ok=True)
    if args.simulate:
        todo = jobs(args.output)[:args.limit or None]
        with ProcessPoolExecutor(args.workers) as pool:
            for number, name in enumerate(pool.map(simulate_episode, todo), 1):
                print(f"{number}/{len(todo)} {name}", flush=True)
    if args.score:
        result = score(args.output, args.score_workers)
        result.update({"protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
                       "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "created": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        RESULT.write_text(json.dumps(result, indent=2) + "\n")
        for name, row in result["summary"].items():
            v1, v0 = row["v1"], row["v0_candidate"]
            print("%-11s v1: forks %2d/%d quiet %2d/%d nudge %2d/%d | v0-candidate: forks %2d/%d quiet %2d/%d" % (
                name, v1["topple_fork"]["alarms"], v1["topple_fork"]["states"], v1["quiet"]["alarms"],
                v1["quiet"]["states"], v1["nudge_fork"]["alarms"], v1["nudge_fork"]["states"],
                v0["topple_fork"]["alarms"], v0["topple_fork"]["states"], v0["quiet"]["alarms"], v0["quiet"]["states"]))
        print("replay floor mm", result["replay_floor_mm"], "decision", result["decision"])


if __name__ == "__main__":
    main()
