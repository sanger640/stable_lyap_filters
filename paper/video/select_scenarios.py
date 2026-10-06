"""Per-state table for set C (panel 4): truth topple count, MuJoCo-probe and deterministic-WM decisions.

Descriptive (video only; no results depend on it). Reproduces the stored alarm counts of
results/jenga/fork_label_reanalysis.json (probe 62, wm_deterministic 44) as a check.
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
OUT = Path(__file__).with_name("scenarios_table.json")


def main():
    import torch
    import uncertainty_monitor_dev as um
    import visual_v2_panel4 as v4
    import step3_world_model as wm
    from counterfactual_monitor import persistent_mode_alarm
    from fresh_fragility_panel import topple_count
    from monitor_v1_eval import NON_ROBOT, decide, trace_endings
    from step3_data import Latent
    panel, screen, _ = v4.screen_panel(1)
    keys = sorted(panel, key=lambda k: (int(k[0]), k[1]))
    snippets = np.load(um.BENCH)["snippets"]
    latent = Latent()
    model = wm.WorldModel.load(wm.DATA / "wm_deterministic.pt")
    rows = []
    for n, (e, i) in enumerate(keys):
        tilt = np.array(screen[e][str(i)]["start_tilt"])
        with np.load(v4.OUT / f"ep{e}_s{i}.npz") as f:
            truth = int(topple_count(f["truth_A"], tilt))
            probe_top = int(topple_count(f["aware"], tilt))
            probe = bool(decide(*trace_endings(f["aware"], NON_ROBOT["jenga"]))[0])
            z0 = latent(f["close_frame"][None])[0].astype(np.float32)
            early, late = wm.wm_rollouts(model, z0, wm.panel_actions(f["chunk"], snippets), seed=20261047 + n)
            wmd = bool(persistent_mode_alarm(early, late).alarm)
        rows.append({"episode": e, "index": int(i), "n": n, "truth_topples": truth, "fork": 2 <= truth <= 62,
                     "probe_topples": probe_top, "probe_alarm": probe, "wm_alarm": wmd})
        torch.cuda.empty_cache()
    check = {"probe": sum(r["probe_alarm"] for r in rows), "wm": sum(r["wm_alarm"] for r in rows)}
    assert check == {"probe": 62, "wm": 44}, check
    OUT.write_text(json.dumps({"check": check, "rows": rows}, indent=0) + "\n")
    cats = {}
    for r in rows:
        c = ("fork" if r["fork"] else "safe") + f" probe={'A' if r['probe_alarm'] else '-'} wm={'A' if r['wm_alarm'] else '-'}"
        cats[c] = cats.get(c, 0) + 1
    print(json.dumps(cats, indent=1))


if __name__ == "__main__":
    main()
