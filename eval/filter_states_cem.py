"""Decisive test A (results/jenga/filter_decisive_protocol.json): the CEM filter on the 75 fixed filter-panel states,
graded with 64-future MuJoCo ground truth under the corrected perception error (truth_A seeds, as truth_v3)."""
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]


def main():
    import cem_filter as cf
    import t1_filter as t1
    import uncertainty_monitor_dev as um
    import visual_pipeline_panel3 as p3
    from fresh_fragility_panel import topple_count
    from jenga_short_held_tails import DirectJengaSim, extract_sim
    from scipy.stats import wilcoxon
    truth = json.loads((t1.OUT / "truth_v3.json").read_text())
    mon, rng = cf.Monitor(seed=20261530), np.random.default_rng(20261531)
    errors, snippets = p3.ErrorModel(), np.load(um.BENCH)["snippets"]
    sim = DirectJengaSim(str(extract_sim(um.SIM_ARCHIVE, tempfile.mkdtemp(prefix="fsc_"))))
    import os
    ver = int(os.environ.get("CEM_VERSION", 1))
    out_path = ROOT / ("results/jenga/filter_decisive_A.json" if ver == 1 else f"results/jenga/filter_decisive_A_v{ver}.json")
    rows = json.loads(out_path.read_text())["rows"] if out_path.exists() else []
    done = {r["state"] for r in rows}
    for path, t in sorted(truth.items()):
        name = Path(path).stem
        if not t["at_rest"] or name in done:
            continue
        with np.load(path) as f:
            snap = (f["state"], f["ctrl"], f["target"], float(f["gripper"])); chunk = f["chunk"]; tilt = f["start_tilt"]
        e, s = name[2:].split("_s"); key = (int(e) + t1.CONST["RESET_BASE"], int(s))
        sim.restore(snap)
        out, info = cf.cem_filter(mon, sim, np.asarray(chunk, np.float32), rng=rng)
        nominal = t["0,0:truth_A"] / 64.0
        if out is None:
            filt = 0.0
        elif not info["intervened"]:
            filt = nominal
        else:
            tr = [um.rollout(sim, snap, out, snippets[j], [errors.sample(t1.CONST["SEEDS"]["truth_A"], *key, j)]) for j in range(64)]
            filt = topple_count(tr, tilt) / 64.0
        rows.append({"state": name, "fork": bool(2 <= t["0,0:truth_A"] <= 62), "safe": t["0,0:truth_A"] == 0,
                     "nominal": nominal, "filtered": filt, "intervened": bool(info["intervened"]),
                     "abstained": bool(info.get("abstained", False)), "offset_mm": info.get("offset_mm"), "seconds": round(info["seconds"], 2)})
        print(rows[-1], flush=True)
        out_path.write_text(json.dumps({"rows": rows}, indent=1) + "\n")
    forks = [r for r in rows if r["fork"]]; safe = [r for r in rows if r["safe"]]
    a = np.array([r["nominal"] for r in forks]); b = np.array([r["filtered"] for r in forks])
    p = float(wilcoxon(a, b, alternative="greater").pvalue) if np.any(a != b) else 1.0
    res = {"forks": len(forks), "safe": len(safe),
           "fork_topple_no_filter": round(float(a.mean()), 4), "fork_topple_cem": round(float(b.mean()), 4),
           "wilcoxon_p_one_sided": p, "fork_states_intervened": sum(r["intervened"] for r in forks),
           "fork_states_abstained": sum(r["abstained"] for r in forks),
           "safe_states_changed": sum(r["intervened"] for r in safe), "safe_states_changed_share": round(sum(r["intervened"] for r in safe) / max(len(safe), 1), 3),
           "safe_topple_cem": round(float(np.mean([r["filtered"] for r in safe])), 4)}
    res["filter_works_on_fork_states"] = bool(p < 0.05 and res["safe_states_changed_share"] <= 0.10)
    out_path.write_text(json.dumps({"result": res, "rows": rows}, indent=1) + "\n")
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
