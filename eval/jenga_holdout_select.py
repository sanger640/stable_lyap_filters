"""Select the fresh holdout states from the physics screen. Rule declared before the screen ran.

From Stage 0 (shared hold) and Stage 1 (physical fork tests) on every chunk of the 57
non-development episodes, with seed 0:
  topple   - mixed at 1x or 2x; all of them, capped at 60 by random draw
  physical - unanimous no-topple at 1x and 2x, Stage 1 all-block physical fork at 1x; up to 20
  quiet    - unanimous no-topple at 1x and 2x, no physical split (neighbor no-floor or
             all-block) at any scale; 60 random
States with a single dissenting topple run are not selected. Writes a Stage 1-format JSON
containing only the selected rows, which Stage 2 consumes.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCALES = ("0.5", "1.0", "2.0")


def pick(rows, count, rng):
    if len(rows) <= count:
        return list(rows)
    return [rows[i] for i in sorted(rng.choice(len(rows), count, replace=False))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", default=str(ROOT / "results/jenga/holdout_stage1.json"))
    ap.add_argument("--output", default=str(ROOT / "results/jenga/holdout_selected_stage1.json"))
    ap.add_argument("--exclude", action="append", default=[],
                    help="a previous selection JSON whose states must not be reused")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--counts", default="60,20,60", help="topple,physical,quiet")
    args = ap.parse_args()
    stage1 = json.loads(Path(args.stage1).read_text())
    used = set()
    for path in args.exclude:
        used |= {(r["episode_id"], r["chunk_start"])
                 for r in json.loads(Path(path).read_text())["rows"]}
    rows = [r for r in stage1["rows"] if (r["episode_id"], r["chunk_start"]) not in used]
    counts = [int(x) for x in args.counts.split(",")]
    rng = np.random.default_rng(args.seed)
    topple = [r for r in rows if "mixed" in (r["grade"]["1.0"], r["grade"]["2.0"])]
    safe = [r for r in rows
            if r["grade"]["1.0"] == "unanimous_safe" and r["grade"]["2.0"] == "unanimous_safe"]
    physical = [r for r in safe if r["scores"]["1.0"]["all_blocks"]]
    quiet = [r for r in safe
             if not any(r["scores"][s]["no_floor"] or r["scores"][s]["all_blocks"]
                        for s in SCALES)]
    chosen = []
    for name, pool, count in zip(("topple", "physical", "quiet"),
                                 (topple, physical, quiet), counts):
        for r in pick(pool, count, rng):
            chosen.append(dict(r, stratum=f"holdout_{name}"))
        print(f"{name}: {len(pool)} available")
    out = dict(stage1, rows=chosen)
    out["selection"] = {"rule": __doc__.strip(), "pool_sizes": {
        "topple": len(topple), "physical": len(physical), "quiet": len(quiet),
        "screened": len(rows)}, "excluded_states": len(used), "seed": args.seed}
    Path(args.output).write_text(json.dumps(out, indent=1) + "\n")
    print(len(chosen), "selected")


if __name__ == "__main__":
    main()
