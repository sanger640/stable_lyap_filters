"""Audit and collect broader task-free nested responses by generic interaction-event coverage."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "eval"))

from jenga_cw_data import residual_windows
from jenga_d3_data import simulate_file
from jenga_d6_set_response import sha256


PROTOCOL = ROOT / "results/jenga/d12_coverage_protocol.json"
AUDIT = ROOT / "results/jenga/d12_coverage_audit.json"
OUTPUT = ROOT / "results/jenga/d12_coverage_data"
STRATA = ("contact_create", "contact_loss", "velocity_impulse", "pose_motion",
          "persistent_motion", "quiet")


def prior_keys(directory):
    keys = set()
    for path in sorted(Path(directory).glob("ep*_seed*.npz")):
        data = np.load(path, allow_pickle=False)
        for state, probe, step in zip(data["state_index"], data["probe_index"], data["contact_step"]):
            keys.add((path.stem, int(state), int(probe), int(step)))
    return keys


def transition_rows(trace_directory, excluded):
    rows = []
    for path in sorted(Path(trace_directory).glob("ep*_seed*.npz")):
        data = np.load(path, allow_pickle=False); traces = data["traces"]
        episode = str(data["episode_id"]); seed = int(data["seed"])
        for state_index in range(traces.shape[0]):
            for probe_index in range(traces.shape[1]):
                path_trace = traces[state_index, probe_index]
                for step in range(3, 11):
                    key = (path.stem, state_index, probe_index, step)
                    if key in excluded:
                        continue
                    before, after = path_trace[step - 2], path_trace[step - 1]
                    contact_delta = after[45:57] - before[45:57]
                    future = path_trace[step - 1:min(step + 2, len(path_trace)), 27:45]
                    rows.append({"path": str(path), "source": path.stem, "episode": episode,
                                 "seed": seed, "state_index": state_index,
                                 "probe_index": probe_index, "step": step,
                                 "contact_create": bool(np.any(contact_delta > .5)),
                                 "contact_loss": bool(np.any(contact_delta < -.5)),
                                 "velocity_change": float(np.sqrt(np.mean(
                                     (after[27:45] - before[27:45]) ** 2))),
                                 "pose_change": float(np.sqrt(np.mean(
                                     (after[:27] - before[:27]) ** 2))),
                                 "persistent_motion": float(np.sqrt(np.mean(future ** 2)))})
    return rows


def thresholds(rows, quantile=.8, quiet_quantile=.4):
    values = {name: np.asarray([row[name] for row in rows])
              for name in ("velocity_change", "pose_change", "persistent_motion")}
    return {"event_quantile": quantile, "quiet_quantile": quiet_quantile,
            **{f"{name}_high": float(np.quantile(value, quantile))
               for name, value in values.items()},
            **{f"{name}_quiet": float(np.quantile(value, quiet_quantile))
               for name, value in values.items()}}


def classify(row, cuts):
    if row["contact_create"]:
        return "contact_create"
    if row["contact_loss"]:
        return "contact_loss"
    if row["velocity_change"] >= cuts["velocity_change_high"]:
        return "velocity_impulse"
    if row["pose_change"] >= cuts["pose_change_high"]:
        return "pose_motion"
    if row["persistent_motion"] >= cuts["persistent_motion_high"]:
        return "persistent_motion"
    if all(row[name] <= cuts[f"{name}_quiet"]
           for name in ("velocity_change", "pose_change", "persistent_motion")):
        return "quiet"
    return None


def select_balanced(rows, cuts, per_stratum, seed):
    candidates = defaultdict(list)
    for row in rows:
        stratum = classify(row, cuts)
        if stratum:
            candidates[stratum].append(row)
    selected = []; used_paths = set()
    for stratum in STRATA:
        ordered = sorted(candidates[stratum], key=lambda row: hashlib.sha256(
            f"d12:{seed}:{stratum}:{row['source']}:{row['state_index']}:{row['probe_index']}:{row['step']}".encode()
        ).hexdigest())
        episode_seed_counts = Counter()
        # First pass caps repeated configurations; second pass fills only if a stratum is scarce.
        for cap in (2, per_stratum):
            for row in ordered:
                path_key = (row["source"], row["state_index"], row["probe_index"])
                config = (row["episode"], row["seed"])
                if path_key in used_paths or episode_seed_counts[config] >= cap:
                    continue
                chosen = dict(row); chosen["stratum"] = stratum
                selected.append(chosen); used_paths.add(path_key); episode_seed_counts[config] += 1
                if sum(item["stratum"] == stratum for item in selected) == per_stratum:
                    break
            if sum(item["stratum"] == stratum for item in selected) == per_stratum:
                break
    return selected, {stratum: len(candidates[stratum]) for stratum in STRATA}


def coverage_summary(rows, cuts, selected=None):
    classified = Counter(filter(None, (classify(row, cuts) for row in rows)))
    output = {"candidate_transitions": len(rows), "classified_counts": dict(classified),
              "thresholds": cuts}
    if selected is not None:
        output.update({"selected_groups": len(selected),
                       "selected_counts": dict(Counter(row["stratum"] for row in selected)),
                       "selected_episodes": len({row["episode"] for row in selected}),
                       "selected_configurations": len({(row["episode"], row["seed"])
                                                       for row in selected})})
    return output


def simulate_coverage(job):
    path, rows, xml, residuals, directions, seed = job
    points = [(row["state_index"], row["probe_index"], row["step"],
               row["stratum"] in ("contact_create", "contact_loss")) for row in rows]
    _, arrays = simulate_file((path, points, xml, residuals, directions, seed))
    if len(arrays["start"]) != len(rows):
        raise RuntimeError("coverage simulation dropped a preselected point")
    arrays["event_stratum"] = np.asarray([row["stratum"] for row in rows])
    return path, arrays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", default=str(ROOT / "results/jenga/trace_data"))
    parser.add_argument("--prior", default=str(ROOT / "results/jenga/d3_data"))
    parser.add_argument("--protocol", default=str(PROTOCOL)); parser.add_argument("--audit", default=str(AUDIT))
    parser.add_argument("--output", default=str(OUTPUT)); parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--workers", type=int, default=1); parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.workers != 1:
        raise SystemExit("D12 is frozen to one simulator worker to protect desktop resources")
    protocol = json.loads(Path(args.protocol).read_text())
    if sha256(args.trace + "/meta.json") != protocol["sources"]["trace_meta_sha256"]:
        raise SystemExit("trace metadata hash differs")
    if sha256(args.prior + "/meta.json") != protocol["sources"]["prior_meta_sha256"]:
        raise SystemExit("prior metadata hash differs")
    excluded = prior_keys(args.prior); rows = transition_rows(args.trace, excluded)
    cuts = thresholds(rows, protocol["selection"]["event_quantile"],
                      protocol["selection"]["quiet_quantile"])
    selected, available = select_balanced(rows, cuts, protocol["selection"]["groups_per_stratum"],
                                          protocol["selection"]["seed"])
    audit = {"protocol": protocol, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "protocol_sha256": sha256(args.protocol), "excluded_prior_groups": len(excluded),
             "available_by_stratum": available, **coverage_summary(rows, cuts, selected)}
    audit_path = Path(args.audit)
    if audit_path.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {audit_path}")
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2), flush=True)
    if args.audit_only:
        return
    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.force:
        raise SystemExit(f"refusing to overwrite nonempty {output}")
    output.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for row in selected:
        grouped[row["path"]].append(row)
    residuals = residual_windows(protocol["simulation"]["residual_windows"], 8,
                                 protocol["simulation"]["seed"] + 1)
    from jenga_short_held_tails import extract_sim
    with tempfile.TemporaryDirectory(prefix="jenga_d12_") as temporary:
        xml = str(extract_sim(ROOT / "vendor/panda_express_sim.tar", temporary))
        jobs = [(path, grouped[path], xml, residuals, protocol["simulation"]["directions"],
                 protocol["simulation"]["seed"]) for path in sorted(grouped)]
        total = 0
        with ProcessPoolExecutor(max_workers=1) as pool:
            for number, (path, arrays) in enumerate(pool.map(simulate_coverage, jobs), 1):
                np.savez_compressed(output / Path(path).name, **arrays); total += len(arrays["start"])
                if number % 20 == 0 or number == len(jobs):
                    print(f"D12 files {number}/{len(jobs)}, groups {total}/{len(selected)}", flush=True)
    meta = {"protocol_sha256": sha256(args.protocol), "audit_sha256": sha256(audit_path),
            "source": str(Path(args.trace).resolve()), "prior_excluded": str(Path(args.prior).resolve()),
            "groups": len(selected), "strata": dict(Counter(row["stratum"] for row in selected)),
            "directions": protocol["simulation"]["directions"], "workers": 1,
            "task_or_failure_labels": False,
            "layout": "D3 nested responses plus event_stratum; start (G,61), actions (G,6,2,38,4), traces (G,6,2,38,61)"}
    (output / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
