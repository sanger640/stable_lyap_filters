"""Program B addendum 1: choose the model timestep from missed contact onsets.

Reads only the generic validation corpus (no Jenga data, no model). A new contact at substep s is
represented at graph period k if the pair's minimum geom distance was <= 5 mm at the last graph
build b = k * floor((s - 1) / k). Rule: if more than 10% of new contacts are missed at k = 50
(100 ms), the model steps at 20 ms; otherwise at 100 ms.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from program_b_corpus import (ARM, FLOOR, HAND, LEFT_FINGER, RECORD_EVERY,  # noqa: E402
                              RIGHT_FINGER, TABLE)

CORPUS = ROOT / "results/program_b/corpus/validation"
ADDENDUM = ROOT / "results/jenga/program_b_stage1_addendum_1.json"
OUTPUT = ROOT / "results/jenga/program_b_timestep_audit_result.json"
RADIUS = 0.005
PERIODS = {"20ms": 10, "100ms": 50}
THRESHOLD = 0.10


def category(other):
    if other < TABLE:
        return "object/object"
    if other == TABLE:
        return "object/table"
    if other in (HAND, LEFT_FINGER, RIGHT_FINGER):
        return "object/gripper"
    return "object/floor" if other == FLOOR else "object/arm" if other == ARM else "other"


def build_frame(substep, period):
    return period * ((substep - 1) // period) // RECORD_EVERY


def audit_episode(onsets, audit, radius=RADIUS):
    near = {(int(f), int(a), int(b)) for f, a, b, d in audit if d <= radius}
    rows = []
    for substep, a, b in onsets:
        row = {"category": category(int(b))}
        for name, period in PERIODS.items():
            row[name] = (build_frame(int(substep), period), int(a), int(b)) in near
        rows.append(row)
    return rows


def summarise(rows):
    result = {"new_contacts": len(rows)}
    for name in PERIODS:
        missed = sum(not r[name] for r in rows)
        result[f"missed_{name}"] = missed / max(len(rows), 1)
    by_category = {}
    for key, count in Counter(r["category"] for r in rows).items():
        subset = [r for r in rows if r["category"] == key]
        by_category[key] = {"new_contacts": count, **{
            f"missed_{name}": sum(not r[name] for r in subset) / count for name in PERIODS}}
    result["by_category"] = by_category
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=str(CORPUS))
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    files = sorted(Path(args.corpus).glob("ep*.npz"))
    if len(files) != 100:
        raise SystemExit(f"expected the 100 validation trajectories, found {len(files)}")
    rows, digest = [], hashlib.sha256()
    for path in files:
        data = np.load(path, allow_pickle=False)
        if not len(data["main_audit"]):
            raise SystemExit(f"{path.name} has no distance audit; regenerate with --audit")
        digest.update(path.read_bytes())
        rows.extend(audit_episode(data["main_onsets"], data["main_audit"]))
    summary = summarise(rows)
    choice = "20ms" if summary["missed_100ms"] > THRESHOLD else "100ms"
    result = {
        "addendum_sha256": hashlib.sha256(ADDENDUM.read_bytes()).hexdigest(),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corpus_files": len(files), "corpus_sha256": digest.hexdigest(),
        "radius_m": RADIUS, "threshold": THRESHOLD, "summary": summary,
        "decision": {"model_timestep": choice,
                     "rule": "20 ms if missed fraction at 100 ms > 0.10, else 100 ms"},
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
