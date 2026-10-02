"""Fresh-panel confirmation of the uncertainty-aware fragility monitor (frozen protocol).

Protocol: results/jenga/fresh_fragility_protocol.json. Jenga episodes replayed with reset seed
episode_id + 5000 (never used before). Pass 1 screens every chunk start of 40 seeded episodes with
64 exact-state probes and grades neighbour topples; a panel of topple forks and safe states is
selected; pass 2 replays the identical chain and adds naive, aware and two truth conditions at the
selected states. Grading labels never enter the monitor.
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
import fragility_grading_dev as fg  # noqa: E402

PROTOCOL = ROOT / "results/jenga/fresh_fragility_protocol.json"
OUT = ROOT / "results/program_b/fresh_fragility"
RESULT = ROOT / "results/jenga/fresh_fragility_result.json"
RESET_BASE = 5000
EPISODES = 40
SELECT_SEED = 20261003
TOPPLE_DEG = 45.0
SIGMA = 0.00265


def neighbour_tilts(trace_end):
    """Tilt (deg) of the two neighbour blocks from a 61-D state's 6D rotations."""
    tilts = []
    for b in (1, 2):
        r = trace_end[9 + 6 * b:15 + 6 * b]          # r00 r01 r10 r11 r20 r21
        z = np.clip(r[0] * r[3] - r[2] * r[1], -1, 1)
        tilts.append(np.degrees(np.arccos(z)))
    return np.array(tilts)


def topple_count(traces, start_tilt):
    eligible = start_tilt < TOPPLE_DEG
    return int(sum(np.any((neighbour_tilts(t[-1]) >= TOPPLE_DEG) & eligible) for t in traces))


def chain_episode(job):
    """Replay one episode; at every chunk start run the 64 exact probes (the chain). At selected
    states also run the extra conditions. Returns screening labels, and traces if selected."""
    episode_id, selected, output = job
    from jenga_runtime import DEFAULT_LMDB, JengaReplay
    from jenga_short_held_tails import DirectJengaSim, chunk_starts, extract_sim
    snippets = np.load(um.BENCH)["snippets"]
    replay = JengaReplay(DEFAULT_LMDB)
    actions = np.asarray(replay.episode(episode_id).actions, np.float32)
    replay.close()
    starts = set(chunk_starts(len(actions)))
    sim = DirectJengaSim(str(extract_sim(um.SIM_ARCHIVE, tempfile.mkdtemp(prefix="fresh_"))))
    screen, traces = {}, {}
    last = max(selected) if selected else max(starts)
    try:
        sim.reset(int(episode_id) + RESET_BASE)
        for index, action in enumerate(actions):
            if index in starts and index + 8 <= len(actions):
                chunk = actions[index:index + 8]
                snapshot = sim.snapshot()
                start_tilt = sim.block_diagnostics()["tilt"][1:].copy()
                exact = [um.rollout(sim, snapshot, chunk, noise) for noise in snippets]
                warm_end = sim.data.qacc_warmstart.copy()
                screen[index] = {"topples": topple_count(exact, start_tilt),
                                 "start_tilt": start_tilt.tolist()}
                if index in selected:
                    key = (int(episode_id) + RESET_BASE, index)
                    hat = um.perception_draw(key, SIGMA)
                    traces[f"s{index}_exact"] = np.stack(exact).astype(np.float32)
                    traces[f"s{index}_naive"] = np.stack([um.rollout(sim, snapshot, chunk, n, [hat])
                                                          for n in snippets]).astype(np.float32)
                    traces[f"s{index}_aware"] = np.stack([um.rollout(sim, snapshot, chunk, n,
                                                                     [hat, um.probe_sample(key, SIGMA, j)])
                                                          for j, n in enumerate(snippets)]).astype(np.float32)
                    for name, seed in fg.TRUTH_SEEDS.items():
                        traces[f"s{index}_truth_{name}"] = np.stack([um.rollout(
                            sim, snapshot, chunk, n, [fg.truth_sample(key, SIGMA, j, seed)])
                            for j, n in enumerate(snippets)]).astype(np.float32)
                sim.restore(snapshot)
                sim.data.qacc_warmstart[:] = warm_end
                if selected and index == last:
                    break
            sim.execute(action)
    finally:
        sim.close()
    if traces:
        path = Path(output) / f"ep{episode_id}.npz"
        np.savez_compressed(path.with_suffix(".tmp.npz"), **traces)
        path.with_suffix(".tmp.npz").replace(path)
    return episode_id, screen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not json.loads(PROTOCOL.read_text())["status"].startswith("frozen"):
        raise SystemExit("protocol not frozen")
    OUT.mkdir(parents=True, exist_ok=True)
    from jenga_runtime import DEFAULT_LMDB, JengaReplay
    replay = JengaReplay(DEFAULT_LMDB)
    all_ids = list(replay.episode_ids)
    replay.close()
    rng = np.random.default_rng(SELECT_SEED)
    episodes = sorted(rng.choice(all_ids, EPISODES, replace=False).tolist(), key=int)

    screen_file = OUT / "screen.json"
    if screen_file.exists():
        screen = json.loads(screen_file.read_text())
    else:
        with ProcessPoolExecutor(args.workers) as pool:
            screen = {e: s for e, s in pool.map(chain_episode, [(e, set(), str(OUT)) for e in episodes])}
        screen_file.write_text(json.dumps(screen) + "\n")
        screen = json.loads(screen_file.read_text())          # str keys, as when resuming
    states = [(e, int(i), v["topples"]) for e, rows in screen.items() for i, v in rows.items()]
    forks = [s for s in states if 2 <= s[2] <= 62]
    safe = [s for s in states if s[2] == 0]
    pick = np.random.default_rng(SELECT_SEED + 1)
    forks = [forks[i] for i in sorted(pick.choice(len(forks), min(60, len(forks)), replace=False))]
    safe = [safe[i] for i in sorted(pick.choice(len(safe), min(100, len(safe)), replace=False))]
    panel = {(e, i): ("topple_fork" if t >= 2 else "safe") for e, i, t in forks + safe}
    print(f"screened {len(states)} states: {len(forks)} forks and {len(safe)} safe selected", flush=True)

    by_episode = {}
    for e, i in panel:
        by_episode.setdefault(e, set()).add(i)
    todo = [(e, idx, str(OUT)) for e, idx in sorted(by_episode.items(), key=lambda x: int(x[0]))
            if not (OUT / f"ep{e}.npz").exists()]
    with ProcessPoolExecutor(args.workers) as pool:
        for number, (e, rows) in enumerate(pool.map(chain_episode, todo), 1):
            mismatch = [i for i in by_episode[e] if rows[i]["topples"] != screen[e][str(i)]["topples"]]
            if mismatch:
                raise SystemExit(f"episode {e}: replay differs from screen at {mismatch}")
            print(f"panel {number}/{len(todo)} ep{e}", flush=True)

    from monitor_v1_eval import NON_ROBOT, decide, trace_endings, wilson
    keys = sorted(panel, key=lambda k: (int(k[0]), k[1]))
    dec = {c: [] for c in ("exact", "naive", "aware", "truth_A", "truth_B")}
    harm = []
    for e, i in keys:
        with np.load(OUT / f"ep{e}.npz") as f:
            for c in dec:
                dec[c].append(decide(*trace_endings(f[f"s{i}_{c}"], NON_ROBOT["jenga"]))[0])
            start_tilt = np.array(screen[e][str(i)]["start_tilt"])
            t = topple_count(f[f"s{i}_truth_A"], start_tilt)
            harm.append(2 <= t <= 62)
    truth_a, truth_b = np.array(dec["truth_A"]), np.array(dec["truth_B"])
    ceiling = fg.grade(truth_b, truth_a)
    graded = {c: fg.grade(dec[c], truth_a) for c in ("exact", "naive", "aware")}
    harm = np.array(harm)
    aware = np.array(dec["aware"])
    harm_hits = int((aware & harm).sum())
    by_class = {}
    for cls in ("topple_fork", "safe"):
        mask = np.array([panel[k] == cls for k in keys])
        by_class[cls] = {"states": int(mask.sum()),
                         **{c: int(np.array(dec[c])[mask].sum()) for c in dec},
                         "harm_fork": int(harm[mask].sum())}
    g1 = bool(graded["aware"]["kappa"] >= 0.8 * ceiling["kappa"] and graded["aware"]["recall"] >= 0.80
              and graded["aware"]["alarm_on_robust"] <= ceiling["alarm_on_robust"] + 0.10)
    g2 = bool(graded["aware"]["kappa"] - graded["naive"]["kappa"] >= 0.15)
    harm_recall = harm_hits / harm.sum() if harm.sum() else None
    g3 = bool(harm_recall is not None and harm_recall >= 0.80)
    result = {
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "episodes": episodes, "screened_states": len(states), "panel_states": len(keys),
        "panel_by_class": by_class, "truth_fragile_A": int(truth_a.sum()),
        "ceiling_A_vs_B": ceiling, "graded_vs_truth_A": graded,
        "harm": {"harm_fork_states": int(harm.sum()), "aware_recall": harm_recall,
                 "aware_recall_wilson": wilson(harm_hits, int(harm.sum())),
                 "aware_alarm_on_non_harm": float((aware & ~harm).sum() / max((~harm).sum(), 1))},
        "gates": {"G1_calibration": g1, "G2_beats_naive": g2, "G3_harm": g3},
        "confirmed": bool(g1 and g2 and g3),
    }
    RESULT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("panel_by_class", "truth_fragile_A", "ceiling_A_vs_B",
                                            "graded_vs_truth_A", "harm", "gates", "confirmed")}, indent=1))


if __name__ == "__main__":
    main()
