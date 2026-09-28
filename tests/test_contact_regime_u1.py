import importlib.util
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "eval")]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "eval" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_u1_noise_maps_have_task_action_dimensions():
    module = load("contact_regime_u1")
    snippets = np.zeros((64, 8, 3), dtype=np.float32)
    assert module.map_noise("pushing", snippets).shape == (64, 8, 2)
    assert module.map_noise("insertion", snippets).shape == (64, 8, 4)


def test_u1_frozen_results_exist_and_protocols_verify():
    detector = load("contact_regime_u1")
    intervention = load("contact_intervention_u1")
    assert detector.verify()["protocol_sha256"] == (
        "3f2386af5537ceddfc70ea6ef9959ac3f049ddda16d8243651b6eb8758df641c")
    assert intervention.verify()["protocol_sha256"] == (
        "44e5e035823fc106647ecc4b872fec063e4c30623177f9432e3f8395a985e3e1")
    assert detector.RESULT.is_file() and intervention.RESULT.is_file()
