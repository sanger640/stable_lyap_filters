"""Phase 1e addendum 3: held-out check of readouts P / BOTH / H30 / H30_floor for US and cell E on panels 7 and 6.

Same G starts, seeds, actions and fork labels as eval/phase1e_panel7.py (addendum 1) and eval/phase1e_panel6.py
(addendum 2); only the readout applied to the 64 rolled-out futures differs.
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
RULES = ("P", "BOTH", "H30", "H30_floor")


def main():
    import torch
    import phase1_models as p1
    import phase1e_panel6 as p6
    import step3_world_model as wm
    import step3b_models as sm
    import uncertainty_monitor_dev as um
    import visual_pipeline_panel3 as p3
    import wm_readout_fix_panel7 as w7
    from fresh_fragility_panel import topple_count
    from phase2_amend4 import alarm, score
    from step3_data import Latent
    res1e = json.loads((ROOT / "results/jenga/phase1e_result.json").read_text())
    models = {"US": (wm.WorldModel.load(ROOT / "results/program_b/phase1e/model_US.pt"), res1e["US"]["tau"]),
              "E": (wm.WorldModel.load(p1.OUT / "model_E.pt"), res1e["E_reference"]["tau"])}
    snippets = np.load(um.BENCH)["snippets"]
    latent, errors, g = Latent(), p3.ErrorModel(), sm.G.load(sm.DATA / "G.pt")
    panels = {
        "panel7": (w7.OUT, w7.CONST, [r["state"] for r in json.loads((ROOT / "results/jenga/phase1e_panel7_result.json").read_text())["rows"]]),
        "panel6": (p6.OUT, p6.CONST, [r["state"] for r in json.loads((ROOT / "results/jenga/phase1e_panel6_result.json").read_text())["rows"]]),
    }
    out = {}
    for pname, (folder, const, states) in panels.items():
        screen = json.loads((folder / "screen.json").read_text())
        fork, al = [], {(m, r): [] for m in models for r in RULES}
        with torch.no_grad():
            for st in states:
                e, i = st.split(":")
                key = (int(e) + const["RESET_BASE"], int(i))
                tilt = np.array(screen[e][i]["start_tilt"])
                with np.load(folder / f"ep{e}_s{i}.npz") as f:
                    k = topple_count(f["truth_A"], tilt)
                    z0 = torch.as_tensor(latent(f["close_frame"][None])[0], device=sm.DEVICE).float()
                    d = torch.tensor(np.stack([np.concatenate([p.reshape(9), r.reshape(9)]) for p, r in
                                               (errors.sample(const["SEEDS"]["monitor_d"], *key, j) for j in range(64))]),
                                     dtype=torch.float32)
                    acts = wm.panel_actions(f["chunk"], snippets)
                fork.append(2 <= k <= 62)
                for m, (model, tau) in models.items():
                    e10, e30 = sm.roll(model, g(z0[None].expand(64, -1, -1, -1), d), acts)
                    for r in RULES:
                        al[(m, r)].append(alarm(e10, e30, r, tau))
                torch.cuda.empty_cache()
        fork = np.array(fork)
        out[pname] = {"states": len(fork), "forks": int(fork.sum()),
                      **{f"{m}_{r}": score(np.array(al[(m, r)]), fork) for m in models for r in RULES}}
    rows6 = json.loads((ROOT / "results/jenga/phase1e_panel6_result.json").read_text())
    out["panel6"]["reference_exact_probe"] = rows6["exact"]; out["panel6"]["reference_truth_B"] = rows6["truth_B"]
    out["US_H30_floor_confirmed"] = bool(all(out[p]["US_H30_floor"]["score"] >= out[p]["US_P"]["score"] + 0.03 for p in panels))
    (ROOT / "results/jenga/phase1e_readout_check_result.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
