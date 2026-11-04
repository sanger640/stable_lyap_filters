"""RCTL1 (results/rctl1/protocol.json): cell E's sampling (uniform over futures) trained with the fast recipe that US used,
then graded on H, panel 7 and panel 6 through the phase-1e code paths, with US re-graded in the same pass.
usage: FAST_TRAIN_Z_ON_CPU=1 python3 eval/recipe_control.py"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
OUT = ROOT / "results/rctl1"
P1E = ROOT / "results/program_b/phase1e"


def train():
    import fast_train as ft
    import phase1b_models as p1b
    import phase1d_models as p1d
    states = p1d.load(p1b.FOLDERS_ENRICHED, proprio=False)
    n = sum(len(s["z"]) for s in states)
    model = ft.train("E_fast", P1E, states, lambda rng, b: rng.integers(n, size=b))
    tau, n_val = p1d.floor(model, False)
    res = p1d.evaluate(model, False, tau) | {"validation_futures": n_val, "tau": tau, "recipe": "fast (phase1e amendment 1)",
                                            "possibly_undertrained": ft.undertrained(json.loads((P1E / "model_E_fast_log.json").read_text()))}
    return model, res


def grade_panel(models, panel_dir, consts, state_ids, screen):
    import torch
    import step3_world_model as wm
    import step3b_models as sm
    import uncertainty_monitor_dev as um
    import visual_pipeline_panel3 as p3
    from counterfactual_monitor import persistent_mode_alarm
    from fresh_fragility_panel import topple_count
    from outcome_modes import two_mode_test
    from step3_data import Latent
    snippets = np.load(um.BENCH)["snippets"]
    latent, errors, g = Latent(), p3.ErrorModel(), sm.G.load(sm.DATA / "G.pt")
    rows = []
    for st in state_ids:
        e, i = st.split(":")
        key = (int(e) + consts["RESET_BASE"], int(i))
        tilt = np.array(screen[e][str(i)]["start_tilt"])
        with np.load(panel_dir / f"ep{e}_s{i}.npz") as f:
            k = topple_count(f["truth_A"], tilt)
            z0 = torch.as_tensor(latent(f["close_frame"][None])[0], device=sm.DEVICE).float()
            d = torch.tensor(np.stack([np.concatenate([p.reshape(9), r.reshape(9)]) for p, r in
                                       (errors.sample(consts["SEEDS"]["monitor_d"], *key, j) for j in range(64))]),
                             dtype=torch.float32)
            acts = wm.panel_actions(f["chunk"], snippets)
            row = {"state": st, "fork": bool(2 <= k <= 62)}
            for name, (model, tau) in models.items():
                e10, e30 = sm.roll(model, g(z0[None].expand(64, -1, -1, -1), d), acts)
                r0 = bool(persistent_mode_alarm(e10, e30).alarm)
                row[f"{name}_R0"] = r0
                row[f"{name}_R1"] = bool(r0 and two_mode_test(e30, 0.0).separation_rms_mm > tau)
            rows.append(row)
        torch.cuda.empty_cache()
    fork = np.array([r["fork"] for r in rows])
    out = {"states": len(rows), "forks": int(fork.sum())}
    for col in [f"{m}_{r}" for m in models for r in ("R0", "R1")]:
        a = np.array([r[col] for r in rows])
        out[col] = {"caught": round(float((a & fork).sum() / fork.sum()), 4), "false_alarm": round(float((a & ~fork).sum() / (~fork).sum()), 4),
                    "forks_caught": int((a & fork).sum()), "false_alarms": int((a & ~fork).sum())}
    return out, rows


def main():
    import step3_world_model as wm
    import phase1e_panel6 as p6
    import wm_readout_fix_panel7 as w7
    OUT.mkdir(parents=True, exist_ok=True)
    model, h = train()
    res1e = json.loads((ROOT / "results/jenga/phase1e_result.json").read_text())
    models = {"E_fast": (model, h["tau"]), "US": (wm.WorldModel.load(P1E / "model_US.pt"), res1e["US"]["tau"])}
    out = {"protocol": "results/rctl1/protocol.json", "H": {"E_fast": h, "US": res1e["US"], "E_old": res1e["E_reference"]}}
    stored = {"panel7": json.loads((ROOT / "results/jenga/phase1e_panel7_result.json").read_text()),
              "panel6": json.loads((ROOT / "results/jenga/phase1e_panel6_result.json").read_text())}
    for name, pdir, consts in (("panel7", w7.OUT, w7.CONST), ("panel6", p6.OUT, p6.CONST)):
        screen = json.loads((pdir / "screen.json").read_text())
        ids = [r["state"] for r in stored[name]["rows"]]
        res, rows = grade_panel(models, pdir, consts, ids, screen)
        s = stored[name]["US_R0"]
        res["US_reproduces_stored"] = bool(res["US_R0"]["forks_caught"] == s["forks_caught"] and res["US_R0"]["false_alarms"] == s["false_alarms"])
        res["E_old_R0_stored"] = stored[name]["E_R0"]
        out[name] = res; (OUT / f"{name}_rows.json").write_text(json.dumps(rows) + "\n")
        print(name, json.dumps(res), flush=True)
    d = {p: 100 * (out[p]["US_R0"]["caught"] - out[p]["E_fast_R0"]["caught"]) for p in ("panel7", "panel6")}
    fa = {p: 100 * (out[p]["US_R0"]["false_alarm"] - out[p]["E_fast_R0"]["false_alarm"]) for p in ("panel7", "panel6")}
    out["US_minus_E_fast_points"] = {"caught": d, "false": fa}
    out["comparable"] = all(out[p]["US_reproduces_stored"] for p in ("panel7", "panel6"))
    out["verdict"] = ("oversampling_helps" if all(d[p] >= 5 and fa[p] <= 5 for p in d) else
                      "recipe_explains" if all(d[p] <= 2 for p in d) else "mixed")
    (OUT / "result.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("US_minus_E_fast_points", "comparable", "verdict")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
