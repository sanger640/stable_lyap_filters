"""DEV-only ground-truth test of shared versus branch-specific post-action dynamics.

This is the consequence stage after the frozen action-branch and 32x boundary-refinement stages.
For each retained boundary, replay its three final brackets densely through H=8 plus hold=30 and
ask whether one anonymous pose/velocity transition law explains both sides. Jenga classes are
joined only for grading. Contact channels are recorded for later diagnostics but never fitted.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from action_branch_monitor import result_dict, shared_vs_branch_dynamics_alarm  # noqa: E402
from jenga_action_branch import summarise  # noqa: E402
from jenga_bench import verify  # noqa: E402
from jenga_generic_regime_audit import reconstruct_endpoints, rollout_trace  # noqa: E402
from jenga_runtime import DEFAULT_LMDB, JengaReplay  # noqa: E402
from jenga_short_held_tails import DirectJengaSim, HORIZON, extract_sim  # noqa: E402


def _run_episode(job):
    episode_id, targets, lmdb, xml, snippets = job
    replay = JengaReplay(lmdb)
    episode = replay.episode(episode_id)
    replay.close()
    sim = DirectJengaSim(xml)
    out = {}
    try:
        sim.reset(int(episode_id))
        for step, action in enumerate(episode.actions):
            if step in targets:
                snapshot = sim.snapshot()
                pair_noises = reconstruct_endpoints(snippets, targets[step]["pair_details"])
                traces = np.stack([[rollout_trace(
                    sim, snapshot, episode.actions[step:step + HORIZON], noise)
                    for noise in pair] for pair in pair_noises])
                evidence = shared_vs_branch_dynamics_alarm(traces[:, :, :, :45])
                out[step] = result_dict(evidence)
                sim.restore(snapshot)
            sim.execute(action)
    finally:
        sim.close()
    return episode_id, out


def prepare(refinement_path):
    """Use only frozen DEV refinement survivors; all other rows remain non-alarms."""
    verify()
    refinement = json.loads(Path(refinement_path).read_text())
    if refinement["protocol"].get("split") != "dev only; test remains untouched":
        raise ValueError("refinement input is not the frozen DEV-only result")
    rows, wanted = [], {}
    for prior in refinement["rows"]:
        row = {key: prior[key] for key in (
            "split", "episode_id", "chunk_start", "scale", "class", "topple_count")}
        row.update({"initial_alarm": bool(prior["initial_alarm"]),
                    "refined_alarm": bool(prior["alarm"]), "alarm": False})
        rows.append(row)
        if prior["alarm"]:
            wanted.setdefault(prior["episode_id"], {})[prior["chunk_start"]] = {
                "pair_details": prior["pair_details"], "row": len(rows) - 1}
    # The snippets are part of the frozen ground-truth cache and are identical for all DEV rows.
    from jenga_bench import SPLITS
    cache = np.load(ROOT / SPLITS["dev"]["cache"], allow_pickle=False)
    return rows, wanted, cache["snippets"]


def evaluate(lmdb, xml, workers, refinement_path, limit=0):
    rows, wanted, snippets = prepare(refinement_path)
    if limit:
        keep = []
        for episode_id, targets in wanted.items():
            for step, target in targets.items():
                keep.append((episode_id, step, target))
        keep = keep[:limit]
        wanted = {}
        for episode_id, step, target in keep:
            wanted.setdefault(episode_id, {})[step] = target
    jobs = [(episode_id, targets, lmdb, xml, snippets)
            for episode_id, targets in wanted.items()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_episode, job): job[0] for job in jobs}
        for number, future in enumerate(as_completed(futures), 1):
            episode_id, results = future.result()
            for step, result in results.items():
                row = rows[wanted[episode_id][step]["row"]]
                row.update(result)
            print(f"dynamics episode {number}/{len(jobs)}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    ap.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    ap.add_argument("--refinement", default=str(
        ROOT / "results/jenga/action_boundary_refine_dev.json"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output", default=str(
        ROOT / "results/jenga/shared_dynamics_ground_truth_dev.json"))
    args = ap.parse_args()
    with tempfile.TemporaryDirectory(prefix="jenga_shared_dynamics_") as temp:
        xml = str(extract_sim(args.sim_archive, temp))
        rows = evaluate(args.lmdb, xml, args.workers, args.refinement, args.limit)
    summary = summarise(rows, [1.0])
    gate = summary["dev"]["1.0"]
    fork = gate["topple_fork"]
    quiet = gate["quiet"]
    result = {
        "protocol": {
            "calibration_free": True,
            "split": "DEV only; TEST untouched",
            "trajectory_source": "dense MuJoCo ground-truth state",
            "candidate_stage": "frozen action-branch plus five-level boundary refinement",
            "signals": "anonymous block pose and velocity only; contacts excluded",
            "static_offset_handling": "subtract each endpoint's pose at H=8",
            "null": "one nonlinear GP transition law shared by both bracket endpoints",
            "alternative": "the same GP prior with cross-branch covariance removed",
            "decision": "at least two of three brackets favour separate laws during both "
                        "hold 1-10 and disjoint hold 11-30",
            "gate": ">=21/23 topple forks and <=4/89 quiet alarms",
            "task_information": "none in alarm; Jenga classes are grading only",
        },
        "gate_passed": bool(fork["alarms"] >= 21 and quiet["alarms"] <= 4),
        "summary": summary,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"gate_passed": result["gate_passed"], **gate}, indent=2))


if __name__ == "__main__":
    main()
