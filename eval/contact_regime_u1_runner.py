"""Memory-efficient runner for frozen contact-regime U1.

This changes no scientific operation: it loads each compressed array once before dispatching the
exact evaluator and summariser bound by the U1 manifest.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import sys

import numpy as np

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

import contact_regime_u1 as u1  # noqa: E402


def main(workers=8):
    manifest = u1.verify()
    if u1.RESULT.exists():
        raise SystemExit("canonical U1 result exists; refusing to overwrite")
    panel = json.loads(u1.PANEL.read_text())
    with np.load(u1.CACHE, allow_pickle=False) as cache:
        snapshots = cache["snapshots"]
        snapshot_sizes = cache["snapshot_sizes"]
        chunks = cache["chunks"]
        action_dims = cache["action_dims"]
        traces = cache["traces"]
        snippets = cache["snippets"]
    jobs = [(i, row, snapshots[i, :snapshot_sizes[i]], chunks[i, :, :action_dims[i]],
             traces[i], snippets) for i, row in enumerate(panel["rows"])]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(u1._evaluate_state, job) for job in jobs]
        for number, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            print(f"v0 state {number}/{len(jobs)}", flush=True)
    rows.sort(key=lambda row: row["index"])
    summary = u1.summarise_results(rows)
    result = {"protocol": u1.PROTOCOL, "protocol_sha256": manifest["protocol_sha256"],
              "runner": "I/O-only cache preload; scientific functions from frozen evaluator",
              "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "summary": summary, "rows": rows}
    u1.RESULT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    main(workers)
