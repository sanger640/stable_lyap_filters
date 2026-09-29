"""Replay frozen D3+D12 TRAIN paths and record D17 enhanced Markov state every step."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "eval"))

from jenga_d6_set_response import sha256
from jenga_d17_enhanced_state_data import (execute_with_impulse, instantaneous_contacts,
                                           robot_proprioception)
from jenga_state_data import step_state

PROTOCOL = ROOT / "results/jenga/d20_enhanced_hybrid_protocol.json"
OUTPUT = ROOT / "results/jenga/d20_enhanced_trajectory_data.npz"
REPORT = ROOT / "results/jenga/d20_enhanced_trajectory_integrity.json"


def enhanced_state(sim, recent_impulse):
    return np.concatenate((step_state(sim), robot_proprioception(sim),
                           instantaneous_contacts(sim).reshape(-1),
                           np.asarray(recent_impulse, np.float32).reshape(-1))).astype(np.float32)


def contact_transition_modes(current, following):
    """Encode absent/persistent/created/lost as 0/1/2/3 for each generic contact."""
    current = np.asarray(current) > .5; following = np.asarray(following) > .5
    result = np.zeros(current.shape, np.int64)
    result[current & following] = 1
    result[~current & following] = 2
    result[current & ~following] = 3
    return result


def collect(protocol, limit=None):
    from jenga_runtime import DEFAULT_LMDB, JengaReplay
    from jenga_short_held_tails import DirectJengaSim, extract_sim

    manifest = json.loads((ROOT / protocol["inputs"]["manifest"]).read_text())
    records = manifest["records"][:limit]
    grouped = defaultdict(list)
    for global_index, row in enumerate(records):
        grouped[(row["dataset"], row["file"])].append((global_index, row))
    count = len(records); enhanced = np.empty((count, 6, 2, 38, 181), np.float32)
    starts = np.empty((count, 181), np.float32); maximum_error = 0.; start_error = 0.
    d17 = np.load(ROOT / protocol["inputs"]["d17_start_state"], allow_pickle=False)
    d17_start = np.concatenate((d17["base_state"], d17["proprioception"],
                                d17["instant_contact"].reshape(-1, 60),
                                d17["recent_contact_impulse"].reshape(-1, 36)), axis=1)
    replay = JengaReplay(DEFAULT_LMDB)
    with tempfile.TemporaryDirectory(prefix="jenga_d20_") as temporary:
        sim = DirectJengaSim(str(extract_sim(ROOT / protocol["inputs"]["simulator"], temporary)))
        try:
            for file_number, ((dataset_name, filename), rows) in enumerate(sorted(grouped.items()), 1):
                directory = ROOT / protocol["inputs"][f"{dataset_name}_directory"]
                with np.load(directory / filename, allow_pickle=False) as nested, np.load(
                        ROOT / protocol["inputs"]["trace_directory"] / filename,
                        allow_pickle=False) as trace:
                    episode_id, reset_seed = str(trace["episode_id"]), int(trace["seed"])
                    episode_actions = np.asarray(replay.episode(episode_id).actions, np.float32)
                    needed = {int(trace["starts"][int(row["state_index"])]) for _, row in rows}
                    clean = {}; sim.reset(reset_seed)
                    for step, action in enumerate(episode_actions):
                        if step in needed:
                            clean[step] = (sim.snapshot(), sim.data.qacc_warmstart.copy())
                            if len(clean) == len(needed): break
                        sim.execute(action)
                    for global_index, row in rows:
                        state_index = int(row["state_index"]); probe_index = int(row["probe_index"])
                        contact_step = int(row["contact_step"]); source_row = int(row["index"])
                        start_step = int(trace["starts"][state_index]); snapshot, warmstart = clean[start_step]
                        sim.restore(snapshot); sim.data.qacc_warmstart[:] = warmstart
                        controls = trace["actions"][state_index, probe_index, 2:2 + contact_step]
                        for action in controls[:-1]: sim.execute(action)
                        recent = execute_with_impulse(sim, controls[-1])
                        starts[global_index] = enhanced_state(sim, recent)
                        start_error = max(start_error, float(np.max(np.abs(
                            starts[global_index] - d17_start[global_index]))))
                        group_snapshot = sim.snapshot(); group_warmstart = sim.data.qacc_warmstart.copy()
                        for level in range(6):
                            for endpoint in range(2):
                                sim.restore(group_snapshot); sim.data.qacc_warmstart[:] = group_warmstart
                                for time_index, action in enumerate(nested["actions"][source_row, level, endpoint]):
                                    impulse = execute_with_impulse(sim, action)
                                    value = enhanced_state(sim, impulse)
                                    enhanced[global_index, level, endpoint, time_index] = value
                                    maximum_error = max(maximum_error, float(np.max(np.abs(
                                        value[:61] - nested["traces"][source_row, level, endpoint,
                                                                            time_index]))))
                if file_number % 10 == 0 or file_number == len(grouped):
                    done = sum(len(value) for value in list(grouped.values())[:file_number])
                    print(f"D20 replay files {file_number}/{len(grouped)}, groups {done}/{count}", flush=True)
        finally:
            sim.close(); replay.close(); d17.close()
    return starts, enhanced, maximum_error, start_error


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--protocol", default=str(PROTOCOL))
    parser.add_argument("--output", default=str(OUTPUT)); parser.add_argument("--report", default=str(REPORT))
    parser.add_argument("--workers", type=int, default=1); parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true"); args = parser.parse_args()
    if args.workers != 1: raise SystemExit("D20 is frozen to one simulator worker")
    output, report = Path(args.output), Path(args.report)
    if (output.exists() or report.exists()) and not args.force: raise SystemExit("refusing to overwrite D20 data")
    protocol_path = Path(args.protocol); protocol = json.loads(protocol_path.read_text())
    for key, hash_key in (("manifest", "manifest_sha256"),
                          ("d17_start_state", "d17_start_state_sha256"),
                          ("d2_checkpoint", "d2_checkpoint_sha256"),
                          ("monitor_targets", "monitor_targets_sha256"),
                          ("simulator", "simulator_sha256")):
        if sha256(ROOT / protocol["inputs"][key]) != protocol["inputs"][hash_key]:
            raise SystemExit(f"frozen input differs: {key}")
    for key, hash_key in (("d3_directory", "d3_meta_sha256"),
                          ("d12_directory", "d12_meta_sha256"),
                          ("trace_directory", "trace_meta_sha256")):
        if sha256(ROOT / protocol["inputs"][key] / "meta.json") != protocol["inputs"][hash_key]:
            raise SystemExit(f"frozen input differs: {key}/meta.json")
    starts, enhanced, trajectory_error, start_error = collect(protocol, args.limit)
    expected = args.limit or protocol["trajectory_collection"]["groups"]
    checks = {"group_count": len(starts) == expected,
              "start_shape": starts.shape == (expected, 181),
              "trajectory_shape": enhanced.shape == (expected, 6, 2, 38, 181),
              "finite": bool(np.all(np.isfinite(starts)) and np.all(np.isfinite(enhanced))),
              "base_trajectory_exact": trajectory_error == 0., "d17_start_exact": start_error == 0.}
    result = {"protocol_sha256": sha256(protocol_path), "created": datetime.now(timezone.utc).isoformat(),
              "limit": args.limit, "checks": checks, "maximum_base_trajectory_error": trajectory_error,
              "maximum_d17_start_error": start_error, "passes": all(checks.values())}
    report.parent.mkdir(parents=True, exist_ok=True); report.write_text(json.dumps(result, indent=2) + "\n")
    if not result["passes"]: raise SystemExit(f"D20 replay integrity failed: {checks}")
    np.savez_compressed(output, start=starts, trajectory=enhanced)
    try:
        output_name = str(output.relative_to(ROOT))
    except ValueError:
        output_name = str(output)
    result["output"] = output_name; result["output_sha256"] = sha256(output)
    report.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
