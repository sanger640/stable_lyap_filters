"""Trajectory-level finite-time expansion on the neighbor-only physical oracle."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from jenga_runtime import DEFAULT_LMDB, JengaReplay  # noqa: E402
from local_expansion import detect_local_expansion  # noqa: E402
from ordered_change_point import robust_component_scale  # noqa: E402

sys.path.insert(0, str(ROOT / "eval"))
from jenga_multipeak_oracle import (TAIL, original_physical_responses,
                                    physical_state)  # noqa: E402
from jenga_near_boundary_discovery import offset_probe_chunks  # noqa: E402
from jenga_ordered_change_points import confusion, physical_boundary  # noqa: E402
from jenga_ordered_holdout import binomial_interval  # noqa: E402
from jenga_short_held_tails import DirectJengaSim, HORIZON, TOPPLE_DEG, extract_sim  # noqa: E402


def temporal_log_mean_evidence(scores):
    """BIC approximation to evidence for expansion at an unknown trajectory time."""
    values = np.asarray(scores, np.float64) / 2.0
    maximum = float(np.max(values))
    return float(2.0 * (maximum + np.log(np.mean(np.exp(values - maximum)))))


def simulate_trajectory(sim, snapshot, probes):
    start = physical_state(sim); trajectories = []; outcomes = []
    for probe in probes:
        sim.restore(snapshot); states = []
        trace = {"peak_tilt": sim.block_diagnostics()["tilt"].copy()}
        for action in probe:
            sim.execute(action, trace); states.append(physical_state(sim) - start)
        for _ in range(TAIL):
            sim.execute(probe[-1], trace); states.append(physical_state(sim) - start)
        trajectories.append(states)
        outcomes.append(bool(np.max(trace["peak_tilt"][1:]) >= TOPPLE_DEG))
    sim.restore(snapshot)
    return np.asarray(trajectories, np.float32), np.asarray(outcomes, bool)


def collect(metadata, replay, xml, scalars, eps):
    wanted = {}
    for row in metadata:
        wanted.setdefault(row["episode_id"], {})[row["chunk_start"]] = row
    rows, trajectories, outcomes = [], [], []
    sim = DirectJengaSim(xml)
    try:
        for number, episode_id in enumerate(sorted(wanted, key=int)):
            episode = replay.episode(episode_id); sim.reset(int(episode_id))
            for start, action in enumerate(episode.actions):
                if start in wanted[episode_id]:
                    row = wanted[episode_id][start]; snapshot = sim.snapshot()
                    probes = offset_probe_chunks(
                        episode.actions[start:start + HORIZON], scalars, eps,
                        row.get("center_offset", 0.0))
                    trajectory, outcome = simulate_trajectory(sim, snapshot, probes)
                    rows.append(row); trajectories.append(trajectory); outcomes.append(outcome)
                sim.execute(action)
            print(f"  trajectory {number + 1}/{len(wanted)} ep{episode_id}", flush=True)
    finally:
        sim.close()
    return rows, np.stack(trajectories), np.stack(outcomes)


def score_rows(scalars, trajectories, outcomes, metadata, scale):
    rows = []
    for meta, trajectory, outcome in zip(metadata, trajectories, outcomes):
        time_results = [detect_local_expansion(
            scalars, trajectory[:, time], half_window=8, scale=scale)
                        for time in range(trajectory.shape[1])]
        evidence = [result.score for result in time_results]
        score = temporal_log_mean_evidence(evidence)
        positive_times = [i + 1 for i, result in enumerate(time_results) if result.alarm]
        row = dict(meta); row.update({
            "physical_boundary": physical_boundary(outcome, 8),
            "physical_counts": {"intact": int((~outcome).sum()),
                                "toppled": int(outcome.sum())},
            "physical_switches": int(np.sum(outcome[1:] != outcome[:-1])),
            "alarm": bool(score > 0.0), "trajectory_score": score,
            "time_scores": evidence, "positive_times": positive_times,
            "endpoint_alarm": bool(time_results[-1].alarm),
            "endpoint_score": float(time_results[-1].score),
        })
        rows.append(row)
    return rows


def summarise(rows):
    truth = np.asarray([row["physical_boundary"] for row in rows], bool)
    alarms = np.asarray([row["alarm"] for row in rows], bool)
    controls = np.asarray([row["stratum"] == "quiet_control" for row in rows], bool)
    boundary = truth
    fp = int(alarms[controls].sum()); tp = int(alarms[boundary].sum())
    endpoint = np.asarray([row["endpoint_alarm"] for row in rows], bool)
    return {"confusion": confusion(truth, alarms),
            "controls": int(controls.sum()), "control_alarms": fp,
            "control_fpr": fp / max(int(controls.sum()), 1),
            "control_fpr_exact_95_percent": binomial_interval(fp, int(controls.sum())),
            "boundaries": int(boundary.sum()), "boundary_detections": tp,
            "boundary_recall": tp / max(int(boundary.sum()), 1),
            "boundary_recall_exact_95_percent": binomial_interval(tp, int(boundary.sum())),
            "score_roc_auc": float(roc_auc_score(truth, [row["trajectory_score"] for row in rows])),
            "endpoint_confusion_same_trajectory_cache": confusion(truth, endpoint),
            "multi_switch_boundaries": int(sum(row["physical_switches"] > 1
                                                for row in rows if row["physical_boundary"])),
            "median_positive_times_on_detected_boundaries": float(np.median([
                len(row["positive_times"]) for row in rows
                if row["physical_boundary"] and row["alarm"]]))}


def write_cache(path, metadata, scalars, trajectories, outcomes):
    output = Path(path); output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, metadata_json=np.asarray(json.dumps(metadata)), scalars=scalars,
                        physical_trajectory=trajectories, physical_outcomes=outcomes)


def read_cache(path):
    data = dict(np.load(path, allow_pickle=False))
    return (json.loads(str(data["metadata_json"])), data["scalars"],
            data["physical_trajectory"], data["physical_outcomes"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    ap.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    ap.add_argument("--oracle-cache", default=str(ROOT / "results/jenga/multipeak_oracle_cache.npz"))
    ap.add_argument("--original-cache", default=str(ROOT / "results/jenga/local_delta_probe_arrays.npz"))
    ap.add_argument("--cache", default=str(ROOT / "results/jenga/trajectory_oracle_cache.npz"))
    ap.add_argument("--output", default=str(ROOT / "results/jenga/trajectory_oracle.json"))
    ap.add_argument("--eps", type=float, default=.10); ap.add_argument("--reuse-cache", action="store_true")
    args = ap.parse_args()
    source = dict(np.load(args.oracle_cache, allow_pickle=False))
    metadata = json.loads(str(source["metadata_json"])); scalars = source["scalars"]
    scale = robust_component_scale(original_physical_responses(args.original_cache))
    if args.reuse_cache:
        metadata, scalars, trajectories, outcomes = read_cache(args.cache)
    else:
        replay = JengaReplay(args.lmdb)
        try:
            with tempfile.TemporaryDirectory(prefix="jenga_trajectory_oracle_") as temp:
                metadata, trajectories, outcomes = collect(
                    metadata, replay, extract_sim(args.sim_archive, temp), scalars, args.eps)
        finally:
            replay.close()
        write_cache(args.cache, metadata, scalars, trajectories, outcomes)
    rows = score_rows(scalars, trajectories, outcomes, metadata, scale)
    summary = summarise(rows)
    passed = bool(summary["control_fpr"] <= .05 and summary["boundary_recall"] >= .80)
    result = {"protocol": {"states": len(rows), "probes": len(scalars), "horizon": HORIZON,
                           "held_tail": TAIL, "trajectory_times": HORIZON + TAIL,
                           "state": "neighbor position/orientation/neighbor-involving contacts",
                           "time_aggregation": "2*log(mean(exp(local_BIC_evidence/2)))",
                           "alarm": "aggregate evidence > 0"},
              "summary": summary, "physical_oracle_passed": passed, "rows": rows}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
