"""Freeze and verify the exact Regime Monitor v0 ground-truth protocol.

This manifest is deliberately separate from the older frozen benchmark manifest. It binds the
monitor implementation, structural constants, DEV selection/refinement evidence, benchmark inputs,
and the accepted DEV result. Future TEST/cross-task evaluators must call ``verify()`` before running.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
FREEZE_DIR = ROOT / "results/jenga/regime_monitor_v0"
MANIFEST_FILE = FREEZE_DIR / "manifest.json"

PROTOCOL = {
    "id": "regime-monitor-v0",
    "claim": "nearby realistic execution errors commit the scene to distinct persistent "
             "physical futures; this is not a failure or severity classifier",
    "trajectory_source": "dense ground-truth pose and velocity",
    "execution_probes": 64,
    "action_horizon": 8,
    "hold_steps": 30,
    "candidate": "action-conditioned smooth-versus-branch evidence",
    "candidate_pairs": 3,
    "boundary_bisections": 5,
    "boundary_vote": "same-pair early and late BIC plateau evidence",
    "effect_curve": "common-zero full H=8 plus hold separation in internally normalized "
                    "anonymous pose/velocity families",
    "passive_null": "cumulative action exposure, squared exposure, passive static component, "
                    "and fixed-grid post-H relaxation",
    "commitment_alternative": "one nonnegative event searched over H=8 and hold 1-5 with step, "
                              "two-step-rise, or five-step-rise shape",
    "search_charge": "2 log(number of commitment candidates) in BIC units",
    "persistence": "late decay-to-zero versus nonzero-asymptote BIC",
    "decision": "alarm when the same at least 2 of 3 pairs pass boundary, commitment, persistence",
    "threshold": "zero penalized evidence; no reference-state calibration",
    "development_result": {"topple_fork": [18, 23], "quiet": [7, 89]},
}

# Hash complete files rather than attempting to infer which lines matter. This intentionally makes
# later refactors fail verification until an explicit new protocol version is created.
FROZEN_FILES = (
    "src/action_branch_monitor.py",
    "src/consequence_monitor.py",
    "eval/jenga_regime_monitor_v0_freeze.py",
    "eval/jenga_action_branch.py",
    "eval/jenga_action_boundary_refine.py",
    "eval/jenga_generic_regime_audit.py",
    "eval/jenga_whole_trajectory_consequence.py",
    "eval/jenga_bench.py",
    "results/jenga/action_boundary_refine_dev.json",
    "results/jenga/whole_trajectory_consequence_dev.json",
    "results/jenga/bench/manifest.json",
    "results/jenga/bench/jenga_bench.npz",
    "results/jenga/holdout_stage0.json",
    "results/jenga/holdout_stage0_cache.npz",
    "vendor/panda_express_sim.tar",
)


def sha256_file(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_digest(protocol, files):
    encoded = json.dumps({"protocol": protocol, "files_sha256": files},
                         sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _dev_summary(path):
    result = json.loads(Path(path).read_text())
    scored = result["summary"]["dev"]["1.0"]
    return {name: [int(scored[name]["alarms"]), int(scored[name]["states"])]
            for name in ("topple_fork", "quiet")}


def _git_state():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def freeze(force=False):
    if MANIFEST_FILE.exists() and not force:
        raise SystemExit(f"{MANIFEST_FILE} already exists; create a new protocol version or use "
                         "--force only to reproduce the identical manifest deliberately")
    missing = [name for name in FROZEN_FILES if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit(f"cannot freeze; missing files: {missing}")
    observed = _dev_summary(ROOT / "results/jenga/whole_trajectory_consequence_dev.json")
    if observed != PROTOCOL["development_result"]:
        raise SystemExit(f"DEV result does not match frozen declaration: {observed}")
    files = {name: sha256_file(ROOT / name) for name in FROZEN_FILES}
    manifest = {
        "protocol": PROTOCOL,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol_sha256": _canonical_digest(PROTOCOL, files),
        "files_sha256": files,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "platform": platform.platform(), "git": _git_state()},
        "usage": {
            "verify": "python eval/jenga_regime_monitor_v0_freeze.py verify",
            "dev_reproduction": "python eval/jenga_whole_trajectory_consequence.py --workers 8",
            "test_policy": "TEST may be run once by a verifier-gated evaluator; never revise v0 "
                           "from its result",
        },
    }
    FREEZE_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify(manifest_path=MANIFEST_FILE):
    path = Path(manifest_path)
    if not path.is_file():
        raise SystemExit(f"Regime Monitor v0 is not frozen: missing {path}")
    manifest = json.loads(path.read_text())
    if manifest.get("protocol") != PROTOCOL:
        raise SystemExit("Regime Monitor v0 protocol constants differ from the frozen manifest")
    expected_files = manifest.get("files_sha256", {})
    if tuple(expected_files) != FROZEN_FILES:
        raise SystemExit("Regime Monitor v0 frozen file list differs from this verifier")
    changed = []
    for name, expected in expected_files.items():
        target = ROOT / name
        actual = sha256_file(target) if target.is_file() else None
        if actual != expected:
            changed.append({"path": name, "expected": expected, "actual": actual})
    if changed:
        lines = [f"  {row['path']}: expected {row['expected']}, now {row['actual']}"
                 for row in changed]
        raise SystemExit("Regime Monitor v0 freeze mismatch:\n" + "\n".join(lines))
    digest = _canonical_digest(PROTOCOL, expected_files)
    if digest != manifest.get("protocol_sha256"):
        raise SystemExit("Regime Monitor v0 protocol digest is invalid")
    observed = _dev_summary(ROOT / "results/jenga/whole_trajectory_consequence_dev.json")
    if observed != PROTOCOL["development_result"]:
        raise SystemExit(f"Regime Monitor v0 DEV result changed: {observed}")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    freeze_parser = sub.add_parser("freeze")
    freeze_parser.add_argument("--force", action="store_true")
    sub.add_parser("verify")
    args = parser.parse_args()
    manifest = freeze(args.force) if args.command == "freeze" else verify()
    print(f"Regime Monitor v0 intact: {manifest['protocol_sha256']}")


if __name__ == "__main__":
    main()
