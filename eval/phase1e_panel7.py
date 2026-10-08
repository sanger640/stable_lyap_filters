"""Phase 1e addendum 1: arm US (and cell E, same code path) on fresh panel 7, R0 and R1, lean/contact split."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]


def main():
    import torch
    import phase1_models as p1
    import step3_world_model as wm
    import step3b_models as sm
    import uncertainty_monitor_dev as um
    import visual_pipeline_panel3 as p3
    import wm_readout_fix_panel7 as w7
    from counterfactual_monitor import persistent_mode_alarm
    from fresh_fragility_panel import topple_count
    from outcome_modes import two_mode_test
    from step3_data import Latent
    res1e = json.loads((ROOT / "results/jenga/phase1e_result.json").read_text())
    models = {"US": (wm.WorldModel.load(ROOT / "results/program_b/phase1e/model_US.pt"), res1e["US"]["tau"]),
              "E": (wm.WorldModel.load(p1.OUT / "model_E.pt"), res1e["E_reference"]["tau"])}
    screen = json.loads((w7.OUT / "screen.json").read_text())
    states = [r["state"] for r in json.loads((ROOT / "results/jenga/wm_readout_fix_panel7_result.json").read_text())["rows"]]
    snippets = np.load(um.BENCH)["snippets"]
    latent, errors, g = Latent(), p3.ErrorModel(), sm.G.load(sm.DATA / "G.pt")
    rows = []
    for st in states:
        e, i = st.split(":")
        key = (int(e) + w7.CONST["RESET_BASE"], int(i))
        tilt = np.array(screen[e][i]["start_tilt"])
        with np.load(w7.OUT / f"ep{e}_s{i}.npz") as f:
            k = topple_count(f["truth_A"], tilt)
            z0 = torch.as_tensor(latent(f["close_frame"][None])[0], device=sm.DEVICE).float()
            d = torch.tensor(np.stack([np.concatenate([p.reshape(9), r.reshape(9)]) for p, r in
                                       (errors.sample(w7.CONST["SEEDS"]["monitor_d"], *key, j) for j in range(64))]),
                             dtype=torch.float32)
            acts = wm.panel_actions(f["chunk"], snippets)
            row = {"state": st, "fork": bool(2 <= k <= 62), "lean": bool(tilt.max() >= 10.0)}
            for name, (model, tau) in models.items():
                e10, e30 = sm.roll(model, g(z0[None].expand(64, -1, -1, -1), d), acts)
                r0 = bool(persistent_mode_alarm(e10, e30).alarm)
                row[f"{name}_R0"] = r0
                row[f"{name}_R1"] = bool(r0 and two_mode_test(e30, 0.0).separation_rms_mm > tau)
            rows.append(row)
        torch.cuda.empty_cache()
    fork = np.array([r["fork"] for r in rows]); lean = np.array([r["lean"] for r in rows])
    out = {"states": len(rows), "forks": int(fork.sum()), "lean_forks": int((fork & lean).sum()),
           "contact_forks": int((fork & ~lean).sum())}
    for col in ("E_R0", "E_R1", "US_R0", "US_R1"):
        a = np.array([r[col] for r in rows])
        out[col] = {"caught": round(float((a & fork).sum() / fork.sum()), 4),
                    "false_alarm": round(float((a & ~fork).sum() / (~fork).sum()), 4),
                    "forks_caught": int((a & fork).sum()), "false_alarms": int((a & ~fork).sum()),
                    "lean": f"{int((a & fork & lean).sum())}/{int((fork & lean).sum())}",
                    "contact": f"{int((a & fork & ~lean).sum())}/{int((fork & ~lean).sum())}"}
    stored = json.loads((ROOT / "results/jenga/wm_readout_fix_panel7_result.json").read_text())["R0"]
    out["E_R0_reproduces_stored"] = bool(out["E_R0"]["forks_caught"] == stored["forks_caught"]
                                         and out["E_R0"]["false_alarms"] == stored["false_alarms"])
    out["rows"] = rows
    (ROOT / "results/jenga/phase1e_panel7_result.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
