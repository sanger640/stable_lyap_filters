import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "jenga_regime_monitor_v0_freeze",
    ROOT / "eval/jenga_regime_monitor_v0_freeze.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_canonical_digest_is_order_independent_for_mapping_keys():
    left = MODULE._canonical_digest({"b": 2, "a": 1}, {"y": "2", "x": "1"})
    right = MODULE._canonical_digest({"a": 1, "b": 2}, {"x": "1", "y": "2"})
    assert left == right


def test_verify_detects_changed_frozen_file(tmp_path, monkeypatch):
    root = tmp_path / "repo"; root.mkdir()
    frozen = root / "frozen.txt"; frozen.write_text("original")
    result = root / "result.json"
    result.write_text(json.dumps({"summary": {"dev": {"1.0": {
        "topple_fork": {"alarms": 18, "states": 23},
        "quiet": {"alarms": 7, "states": 89}}}}}))
    monkeypatch.setattr(MODULE, "ROOT", root)
    monkeypatch.setattr(MODULE, "FROZEN_FILES", ("frozen.txt", "result.json"))
    monkeypatch.setitem(MODULE.PROTOCOL, "development_result",
                        {"topple_fork": [18, 23], "quiet": [7, 89]})
    monkeypatch.setattr(MODULE, "_dev_summary", lambda path: MODULE.PROTOCOL["development_result"])
    files = {name: MODULE.sha256_file(root / name) for name in MODULE.FROZEN_FILES}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "protocol": MODULE.PROTOCOL, "files_sha256": files,
        "protocol_sha256": MODULE._canonical_digest(MODULE.PROTOCOL, files)}))
    assert MODULE.verify(manifest)["protocol_sha256"]
    frozen.write_text("changed")
    with pytest.raises(SystemExit, match="freeze mismatch"):
        MODULE.verify(manifest)
