"""Merge per-episode results from cluster jobs into the usual results/jenga/paired_<arm><tag>.jsonl (sorted by episode).

usage: python3 cluster/merge_paired.py <arm><tag> [...]      e.g. paired_record_dev2x_ak paired_cbf_dev2x_ak
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
for tag in sys.argv[1:]:
    d = ROOT / "results/program_b/policy_rollouts" / tag
    rows = sorted((json.loads(q.read_text()) for q in d.glob("ep*.json")), key=lambda r: r["episode"])
    (ROOT / f"results/jenga/{tag}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(tag, len(rows), "episodes")
