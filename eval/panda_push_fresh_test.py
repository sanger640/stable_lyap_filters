"""Prospectively generated, frozen one-time TEST for physical Panda upright pushing.

Physical outcomes are used only to stratify mechanism coverage before the monitor is run.  Monitor
scores never enter selection.  The evaluator verifies the frozen Regime Monitor v0 and this TEST
panel, runs once, and refuses to overwrite its result.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "eval"))

from jenga_regime_monitor_v0_freeze import sha256_file, verify as verify_v0
from jenga_short_held_tails import extract_sim
import panda_push_dev_panel as dev
from systems.panda_block_push import PandaBlockPush, write_push_xml

OUT = ROOT / "results/panda_block_push/fresh_test"
PANEL = OUT / "panel.json"
CACHE = OUT / "panel_cache.npz"
MANIFEST = OUT / "manifest.json"
RESULT = OUT / "ground_truth_monitor.json"
GRADE_CACHE = OUT / "grades_cache.json"
SEED = 1640
TARGET_COUNTS = dict(dev.TARGET_COUNTS)
PROTOCOL = {
    "id": "panda-upright-push-fresh-test-v0",
    "status": "prospectively frozen one-time TEST; monitor scores absent from selection",
    "seed": SEED,
    "candidate_families": {"boundary": 80, "safe_centered": 20, "topple": 15,
                           "contact_loss": 20, "overshoot": 30},
    "selection": TARGET_COUNTS,
    "selection_information": "64-probe physical topple coverage and planned mechanism family only",
    "freshness": "continuous random specifications with seed 1640; no reused DEV state",
    "probes": "exact frozen 64xH8 Jenga execution-error snippets at 1x",
    "monitor": "unchanged frozen Regime Monitor v0",
    "task_labels_in_monitor": False,
    "workers": 1,
}
FROZEN_FILES = ("eval/panda_push_fresh_test.py", "results/panda_block_push/fresh_test/panel.json",
                "results/panda_block_push/fresh_test/panel_cache.npz",
                "results/jenga/holdout_stage0_cache.npz",
                "results/jenga/regime_monitor_v0/manifest.json", "vendor/panda_express_sim.tar")


def candidate_specs(seed=SEED):
    rng = np.random.default_rng(seed); rows = []
    counts = PROTOCOL["candidate_families"]
    for family, count in counts.items():
        for _ in range(count):
            if family == "boundary":
                z, end_x = rng.uniform(.439, .454), rng.uniform(.545, .625)
                contact_y, block_y, veer = rng.uniform(-.008, .008), rng.uniform(-.012, .012), 0.
            elif family == "safe_centered":
                z, end_x = rng.uniform(.424, .439), rng.uniform(.535, .605)
                contact_y, block_y, veer = rng.uniform(-.003, .003), rng.uniform(-.008, .008), 0.
            elif family == "topple":
                z, end_x = rng.uniform(.451, .465), rng.uniform(.555, .635)
                contact_y, block_y, veer = rng.uniform(-.005, .005), rng.uniform(-.008, .008), 0.
            elif family == "contact_loss":
                z, end_x = rng.uniform(.418, .429), rng.uniform(.540, .580)
                contact_y, block_y = 0., rng.uniform(-.012, .012)
                veer = rng.choice((-1., 1.)) * rng.uniform(.055, .110)
            else:
                z, end_x = rng.uniform(.420, .438), rng.uniform(.640, .700)
                contact_y, block_y, veer = rng.uniform(-.004, .004), rng.uniform(-.008, .008), 0.
            rows.append({"candidate_id": len(rows), "family": family, "z": float(z),
                         "end_x": float(end_x), "contact_y": float(contact_y),
                         "block_y": float(block_y), "veer_y": float(veer)})
    return rows


def _select(rows):
    selected = []
    forks = [row for row in rows if row["mixed_topple"]]
    forks.sort(key=lambda row: (abs(row["topple_count"] - dev.PROBES / 2), row["candidate_id"]))
    selected.extend(("topple_fork", row) for row in forks[:TARGET_COUNTS["topple_fork"]])
    safe = [row for row in rows if row["family"] == "safe_centered" and row["topple_count"] == 0]
    selected.extend(("safe_centered", row) for row in safe[:TARGET_COUNTS["safe_centered"]])
    topple = [row for row in rows if row["family"] == "topple" and row["topple_count"] == dev.PROBES]
    selected.extend(("topple_unanimous", row) for row in topple[:TARGET_COUNTS["topple_unanimous"]])
    contact = [row for row in rows if row["family"] == "contact_loss" and row["topple_count"] == 0]
    selected.extend(("contact_loss_control", row)
                    for row in contact[:TARGET_COUNTS["contact_loss_control"]])
    overshoot = [row for row in rows if row["family"] == "overshoot" and row["topple_count"] == 0]
    selected.extend(("overshoot_control", row) for row in overshoot[:TARGET_COUNTS["overshoot_control"]])
    observed = {name: sum(cohort == name for cohort, _ in selected) for name in TARGET_COUNTS}
    if observed != TARGET_COUNTS:
        raise RuntimeError(f"fresh candidate pool misses predeclared coverage: {observed}")
    return [{**row, "cohort": cohort, "panel_index": index}
            for index, (cohort, row) in enumerate(selected)]


def prepare(archive):
    verify_v0()
    if PANEL.exists() or CACHE.exists() or MANIFEST.exists() or RESULT.exists():
        raise SystemExit("fresh Panda TEST already exists; refusing overwrite")
    snippets = np.load(ROOT / "results/jenga/holdout_stage0_cache.npz", allow_pickle=False)["snippets"]
    with tempfile.TemporaryDirectory(prefix="panda_fresh_test_") as temp:
        xml = str(write_push_xml(extract_sim(archive, temp).parent))
        env = PandaBlockPush(xml, render=False)
        try:
            specs = candidate_specs()
            graded = json.loads(GRADE_CACHE.read_text()) if GRADE_CACHE.exists() else []
            if len(graded) > len(specs) or any(
                    any(saved[key] != spec[key] for key in spec)
                    for saved, spec in zip(graded, specs)):
                raise RuntimeError("physical grade cache belongs to a different candidate protocol")
            OUT.mkdir(parents=True, exist_ok=True)
            for index, spec in enumerate(specs[len(graded):], start=len(graded)):
                snapshot, chunk = dev._initialise(env, spec)
                count = sum(dev._run_probe(env, snapshot, chunk, noise)[0]["failure"]
                            for noise in snippets)
                graded.append({**spec, "topple_count": int(count),
                               "mixed_topple": bool(2 <= count <= dev.PROBES - 2)})
                GRADE_CACHE.write_text(json.dumps(graded, indent=2) + "\n")
                print(f"grade {index + 1}/{len(specs)}", flush=True)
            selected = _select(graded); collected = []
            for index, row in enumerate(selected):
                snapshot, chunk = dev._initialise(env, row); traces = []; count = 0
                for noise in snippets:
                    outcome, trace = dev._run_probe(env, snapshot, chunk, noise, record=True)
                    count += int(outcome["failure"]); traces.append(trace)
                if count != row["topple_count"]: raise RuntimeError("TEST collection changed grade")
                collected.append((row, snapshot, chunk, np.stack(traces)))
                print(f"collect {index + 1}/{len(selected)}", flush=True)
        finally: env.close()
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, snippets=snippets,
                        snapshots=np.stack([item[1][0] for item in collected]),
                        snapshot_ctrl=np.stack([item[1][1] for item in collected]),
                        snapshot_target=np.stack([item[1][2] for item in collected]),
                        snapshot_gripper=np.asarray([item[1][3] for item in collected]),
                        snapshot_peak_tilt=np.asarray([item[1][4] for item in collected]),
                        chunks=np.stack([item[2] for item in collected]),
                        traces=np.stack([item[3] for item in collected]).astype(np.float32))
    rows = [item[0] for item in collected]
    coverage = {name: {"states": sum(row["cohort"] == name for row in rows),
                       "topple_count_min": min(row["topple_count"] for row in rows
                                               if row["cohort"] == name),
                       "topple_count_max": max(row["topple_count"] for row in rows
                                               if row["cohort"] == name)}
                for name in TARGET_COUNTS}
    PANEL.write_text(json.dumps({"protocol": PROTOCOL, "coverage": coverage, "rows": rows},
                                indent=2) + "\n")
    print(json.dumps(coverage, indent=2))


def _digest(files):
    payload = json.dumps({"protocol": PROTOCOL, "files_sha256": files},
                         sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def freeze():
    verify_v0()
    if MANIFEST.exists(): raise SystemExit("fresh TEST is already frozen")
    files = {name: sha256_file(ROOT / name) for name in FROZEN_FILES}
    value = {"protocol": PROTOCOL, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "files_sha256": files, "protocol_sha256": _digest(files)}
    MANIFEST.write_text(json.dumps(value, indent=2) + "\n")
    print(value["protocol_sha256"]); return value


def verify():
    verify_v0(); value = json.loads(MANIFEST.read_text())
    if value.get("protocol") != PROTOCOL or tuple(value.get("files_sha256", {})) != FROZEN_FILES:
        raise SystemExit("fresh Panda TEST protocol mismatch")
    for name, expected in value["files_sha256"].items():
        if sha256_file(ROOT / name) != expected: raise SystemExit(f"fresh TEST file changed: {name}")
    if _digest(value["files_sha256"]) != value["protocol_sha256"]:
        raise SystemExit("fresh TEST digest mismatch")
    return value


def evaluate(archive):
    manifest = verify()
    if RESULT.exists(): raise SystemExit("refusing to overwrite one-time fresh TEST result")
    panel = json.loads(PANEL.read_text()); cache = np.load(CACHE, allow_pickle=False); rows = []
    with tempfile.TemporaryDirectory(prefix="panda_fresh_test_eval_") as temp:
        xml = str(write_push_xml(extract_sim(archive, temp).parent))
        for index, row in enumerate(panel["rows"]):
            job = (xml, index, row, dev._snapshot(cache, index), cache["chunks"][index],
                   cache["traces"][index], cache["snippets"])
            rows.append(dev._diagnose_state(job))
            print(f"TEST monitor {index + 1}/{len(panel['rows'])}", flush=True)
    summary = dev.summarise(rows)
    value = {"protocol": PROTOCOL, "protocol_sha256": manifest["protocol_sha256"],
             "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "regime_monitor_v0_sha256": verify_v0()["protocol_sha256"],
             "summary": summary, "rows": rows}
    RESULT.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2)); return value


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "evaluate"):
        child = sub.add_parser(command)
        child.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    sub.add_parser("freeze"); sub.add_parser("verify")
    args = parser.parse_args()
    if args.command == "prepare": prepare(args.sim_archive)
    elif args.command == "freeze": freeze()
    elif args.command == "verify": print(verify()["protocol_sha256"])
    else: evaluate(args.sim_archive)


if __name__ == "__main__": main()
