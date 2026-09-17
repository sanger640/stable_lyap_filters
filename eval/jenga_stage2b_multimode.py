"""Stage 2b: multi-direction, multi-group fork test on cached image latents.

Rule (src/outcome_modes.multi_mode_test, frozen after synthetic tests only): top 5 PCA directions;
diagonal Gaussian mixtures with 1-4 groups, 10 starts, fixed seed; a grouping is valid if every
group has >= 2 runs and every pair passes 1-D BIC (two > one) and Ashman D > 2 along the line
joining them; the count is the valid k with the lowest BIC. Alarm if >= 2 groups at both hold
steps 10 and 30 and the groupings match (<= 1 run misplaced, either direction).

Runs on the shared-hold and own-hold (arm moving) Stage 2 caches; physical classes are taken from
those runs' JSON and used only for grading.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from outcome_modes import coarse_persistent_fork, groupings_persist, multi_mode_test  # noqa: E402

CLASSES = ("topple_fork", "nudge_fork", "small_split", "quiet", "weak")
RUNS = {"shared_hold": ("stage2_visual_forks.json", "stage2_visual_forks_cache.npz"),
        "own_hold": ("stage2_visual_forks_ownhold.json", "stage2_visual_forks_ownhold_cache.npz")}


def score(coords):
    early, late = multi_mode_test(coords[0]), multi_mode_test(coords[1])
    persist = groupings_persist(early["labels"], late["labels"])
    both = early["groups"] >= 2 and late["groups"] >= 2
    return {"alarm": bool(both and persist),
            # Revised persistence, declared after Stage 2b; valid only on fresh data.
            "revised_alarm": bool(both and coarse_persistent_fork(early["labels"],
                                                                  late["labels"])),
            "groups_step10": early["groups"], "groups_step30": late["groups"],
            "sizes_step30": late["sizes"], "persistent": bool(persist)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "results/jenga"))
    ap.add_argument("--output", default=str(ROOT / "results/jenga/stage2b_multimode.json"))
    ap.add_argument("--run", action="append", default=[],
                    help="name=result.json:cache.npz (relative to --dir); replaces defaults")
    args = ap.parse_args()
    out = {"protocol": {"rule": __doc__.strip().split("\n\n")[1],
                        "constants_fixed_before_results": True}, "runs": {}}
    runs = RUNS
    if args.run:
        runs = {}
        for spec in args.run:
            name, files = spec.split("=", 1)
            runs[name] = tuple(files.split(":"))
    for name, (json_name, cache_name) in runs.items():
        previous = json.loads((Path(args.dir) / json_name).read_text())
        coords = np.load(Path(args.dir) / cache_name, allow_pickle=False)["coords"]
        rows = []
        for i, prow in enumerate(previous["rows"]):
            row = {"episode_id": prow["episode_id"], "chunk_start": prow["chunk_start"],
                   "by_scale": {}}
            for s, scale in enumerate(prow["by_scale"]):
                b = prow["by_scale"][scale]
                row["by_scale"][scale] = {"class": b["class"], "topple_count": b["topple_count"],
                                          "stratum": prow.get("stratum"),
                                          "old_pc1_alarm": b["visual"]["alarm"],
                                          "multi": score(coords[i, s])}
            rows.append(row)
        summary = {}
        for scale in rows[0]["by_scale"]:
            summary[scale] = {}
            for c in CLASSES:
                part = [r["by_scale"][scale] for r in rows if r["by_scale"][scale]["class"] == c]
                summary[scale][c] = {"states": len(part),
                                     "multi_alarms": sum(p["multi"]["alarm"] for p in part),
                                     "revised_alarms": sum(p["multi"]["revised_alarm"]
                                                           for p in part),
                                     "old_pc1_alarms": sum(p["old_pc1_alarm"] for p in part)}
        strata = sorted({r["by_scale"][s]["stratum"] for r in rows for s in r["by_scale"]}
                        - {None})
        by_stratum = {}
        for scale in rows[0]["by_scale"]:
            by_stratum[scale] = {}
            for st in strata:
                part = [r["by_scale"][scale] for r in rows if r["by_scale"][scale]["stratum"] == st]
                by_stratum[scale][st] = {
                    "states": len(part),
                    "old_pc1_alarms": sum(p["old_pc1_alarm"] for p in part),
                    "multi_alarms": sum(p["multi"]["alarm"] for p in part),
                    "revised_alarms": sum(p["multi"]["revised_alarm"] for p in part),
                    "states_with_topple_mix_at_this_scale": sum(p["class"] == "topple_fork"
                                                                for p in part)}
            print("  by stratum", scale, by_stratum[scale])
        out["runs"][name] = {"summary": summary, "by_stratum": by_stratum, "rows": rows}
        print(name)
        for scale, entry in summary.items():
            print("  ", scale, {c: f"old {v['old_pc1_alarms']} / multi {v['multi_alarms']} / "
                                   f"revised {v['revised_alarms']} of {v['states']}"
                                for c, v in entry.items() if v["states"]})
    Path(args.output).write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
