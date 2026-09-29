from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from jenga_d13_monitor_evidence import (decode_evidence, evidence_rows, normalize_evidence,
                                        signed_log, target_normalizers)


def test_signed_evidence_transform_is_invertible_and_preserves_decisions():
    values = np.asarray([[-10., -1., 0., 2.], [3., 5., -4., 100.]], np.float32)
    mean, scale = target_normalizers(values)
    restored = decode_evidence(normalize_evidence(values, mean, scale), mean, scale)
    assert np.allclose(restored, values, atol=2e-5)
    assert np.array_equal(np.sign(restored), np.sign(values))
    assert np.all(np.isfinite(signed_log(values)))
    rows = evidence_rows(values)
    assert rows[0] == {"boundary": False, "commitment": False,
                       "persistence": True, "alarm": False}
    assert rows[1] == {"boundary": True, "commitment": False,
                       "persistence": True, "alarm": False}
