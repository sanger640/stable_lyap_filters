"""Re-label with the gripper-relative target events (regrade_any_outcome.events_three_block_v2): the corrected
three-block H (h_v2), using the stored alarms from threeblock_v2_result.json."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]

from phase2_amend4 import score  # noqa: E402
from regrade_any_outcome import events_three_block_v2, labels  # noqa: E402


def main():
    res = json.loads((ROOT / "results/jenga/threeblock_v2_result.json").read_text())
    rows = [r for r in res["rows"] if r["at_rest"]]
    out = {"label": "any-outcome, target events gripper-relative (dropped > 50 mm when gripper closed)"}
    changed = 0
    for r in rows:
        with np.load(ROOT / "results/program_b/phase1/h_v2" / f"{r['state']}.npz") as f:
            af, asafe = labels(events_three_block_v2(f["traces"]))
        changed += (af != r["any_fork"]); r["fork2"], r["safe2"] = af, asafe
    sub = [r for r in rows if r["fork2"] or r["safe2"]]
    y = np.array([r["fork2"] for r in sub])
    out["H"] = {"forks": int(y.sum()), "safe": int((~y).sum()), "fork_label_changed_vs_any_outcome": int(changed),
                **{m: score(np.array([r["alarms"][m] for r in sub]), y) for m in sub[0]["alarms"]}}
    (ROOT / "results/jenga/target_events_relabel.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
