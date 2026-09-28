import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "contact_benchmark_previews", ROOT / "eval/contact_benchmark_previews.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_preview_protocol_names_distinct_failure_mechanisms():
    source = (ROOT / "eval/contact_benchmark_previews.py").read_text()
    assert "edge_fall" in source
    assert "rim_jam" in source
    assert "12 mm execution offset" in source
