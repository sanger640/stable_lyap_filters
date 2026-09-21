"""The frozen benchmark must refuse to evaluate when anything that decides a number has changed."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
import jenga_bench as bench  # noqa: E402


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    """A tiny frozen benchmark in a temp dir, with the module pointed at it."""
    data = tmp_path / "bench.npz"
    np.savez_compressed(data, x=np.arange(10, dtype=np.float32))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"bench_sha256": bench.sha256_file(data),
                                    "eval_config": json.loads(json.dumps(bench.EVAL_CONFIG)),
                                    "created": "test"}))
    monkeypatch.setattr(bench, "BENCH_FILE", data)
    monkeypatch.setattr(bench, "MANIFEST_FILE", manifest)
    return data, manifest


def test_verify_passes_on_an_untouched_benchmark(frozen):
    assert bench.verify()["created"] == "test"


def test_verify_refuses_a_changed_benchmark_file(frozen):
    data, _ = frozen
    np.savez_compressed(data, x=np.arange(11, dtype=np.float32))
    with pytest.raises(SystemExit, match="changed since it was frozen"):
        bench.verify()


def test_verify_refuses_a_changed_evaluation_constant(frozen, monkeypatch):
    edited = copy.deepcopy(bench.EVAL_CONFIG)
    edited["budget"] = 0.10
    monkeypatch.setattr(bench, "EVAL_CONFIG", edited)
    with pytest.raises(SystemExit, match="budget"):
        bench.verify()


def test_verify_refuses_when_nothing_is_frozen(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "BENCH_FILE", tmp_path / "missing.npz")
    monkeypatch.setattr(bench, "MANIFEST_FILE", tmp_path / "missing.json")
    with pytest.raises(SystemExit, match="freeze"):
        bench.verify()


def test_array_hash_sees_values_shape_and_dtype():
    a = np.arange(6, dtype=np.float32)
    assert bench.sha256_array(a) == bench.sha256_array(a.copy())
    assert bench.sha256_array(a) != bench.sha256_array(a.reshape(2, 3))
    assert bench.sha256_array(a) != bench.sha256_array(a.astype(np.float64))
    b = a.copy(); b[0] += 1e-6
    assert bench.sha256_array(a) != bench.sha256_array(b)


def test_spread_is_rms_distance_from_the_mean():
    endings = np.zeros((4, 9))
    endings[:, 0] = [0.0, 0.0, 2.0, 2.0]          # two probes 1 mm either side of the mean
    assert bench.spread(endings) == pytest.approx(1.0)
    assert bench.spread(np.ones((64, 9))) == 0.0
