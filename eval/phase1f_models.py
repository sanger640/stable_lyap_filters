"""Branching-WM Phase 1f model L: the US recipe on label-free curated training states.

Frozen protocol: results/jenga/phase1f_protocol.json. Same code path as eval/phase1e_models.py (rare-mode detection,
US sampler, fast recipe, floor, H evaluation) and eval/phase1e_panel7.py (panel 7); only the training folder differs.
Topple labels are loaded for reporting (describe) only.
"""
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]

import phase1d_models as p1d  # noqa: E402
import phase1e_models as p1e  # noqa: E402

PROTOCOL = ROOT / "results/jenga/phase1f_protocol.json"
DATA = ROOT / "results/program_b/phase1f/train"
OUT = ROOT / "results/program_b/phase1f"
RESULT = ROOT / "results/jenga/phase1f_result.json"


def load():
    states, names = [], []
    for path in sorted(DATA.glob("ep*.npz")):
        if path.name.endswith(".tmp.npz"):
            continue
        with np.load(path) as f:
            n = len(f["z"])
            states.append({"z0": torch.from_numpy(f["z0"]), "z": torch.from_numpy(f["z"]), "z0k": torch.from_numpy(f["z0k"]),
                           "actions": torch.from_numpy(p1d.augment(f["actions"], None, False)),
                           "d": torch.from_numpy(np.concatenate([f["d_pos"].reshape(n, 9), f["d_rot"].reshape(n, 9)], 1)).float(),
                           "top": p1d.topples(f["traces"], f["start_tilt"])})
        names.append(str(path))
    return states, names


def panel7(model, tau):
    """Phase-1e addendum-1 grading of one model on panel 7 (same seeds and code path as eval/phase1e_panel7.py)."""
    import step3_world_model as wm
    import step3b_models as sm
    import uncertainty_monitor_dev as um
    import visual_pipeline_panel3 as p3
    import wm_readout_fix_panel7 as w7
    from counterfactual_monitor import persistent_mode_alarm
    from fresh_fragility_panel import topple_count
    from outcome_modes import two_mode_test
    from step3_data import Latent
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
            e10, e30 = sm.roll(model, g(z0[None].expand(64, -1, -1, -1), d), wm.panel_actions(f["chunk"], snippets))
            r0 = bool(persistent_mode_alarm(e10, e30).alarm)
            rows.append({"fork": bool(2 <= k <= 62), "lean": bool(tilt.max() >= 10.0), "R0": r0,
                         "R1": bool(r0 and two_mode_test(e30, 0.0).separation_rms_mm > tau)})
        torch.cuda.empty_cache()
    fork = np.array([r["fork"] for r in rows]); lean = np.array([r["lean"] for r in rows])
    out = {}
    for col in ("R0", "R1"):
        a = np.array([r[col] for r in rows])
        out[col] = {"caught": round(float((a & fork).sum() / fork.sum()), 4),
                    "false_alarm": round(float((a & ~fork).sum() / (~fork).sum()), 4),
                    "forks_caught": int((a & fork).sum()), "false_alarms": int((a & ~fork).sum()),
                    "lean": f"{int((a & fork & lean).sum())}/{int((fork & lean).sum())}",
                    "contact": f"{int((a & fork & ~lean).sum())}/{int((fork & ~lean).sum())}"}
    return out


def main():
    import fast_train as ft
    if not json.loads(PROTOCOL.read_text())["status"].startswith("frozen"):
        raise SystemExit("protocol not frozen")
    states, names = load()
    if len(states) != 4766:
        raise SystemExit(f"expected 4,766 rendered states, found {len(states)}")
    modes_file = OUT / "modes.json"
    if not modes_file.exists():
        modes_file.write_text(json.dumps(p1e.detect_modes(states, names)) + "\n")
    modes = json.loads(modes_file.read_text())
    summary = p1e.describe(states, names, modes)                 # reporting only (uses topple labels)
    rare = np.array([k in set(modes[n]) for n, s in zip(names, states) for k in range(len(s["z"]))])
    has_rare = np.array([bool(modes[n]) for n, s in zip(names, states) for _ in range(len(s["z"]))])
    pools = (np.flatnonzero(rare), np.flatnonzero(~rare & has_rare))
    sampler = lambda rng, b: np.concatenate([rng.choice(pools[0], b // 2), rng.choice(pools[1], b - b // 2)])
    model = ft.train("L", OUT, states, sampler)
    del states
    tau, n_val = p1d.floor(model, False)
    res = p1d.evaluate(model, False, tau) | {"validation_futures": n_val}
    res["possibly_undertrained"] = ft.undertrained(json.loads((OUT / "model_L_log.json").read_text()))
    us = json.loads((ROOT / "results/jenga/phase1e_result.json").read_text())["US"]
    score = lambda r: r["R0"]["caught"] - r["R0"]["false_alarm"]
    result = {"L_on_H": res, "L_score": round(score(res), 4), "US_score": round(score(us), 4),
              "L_within_0.03_of_US": bool(score(res) >= score(us) - 0.03), "modes_summary_reporting_only": summary,
              "selection_summary": json.loads((OUT / "selection.json").read_text())["summary"]}
    RESULT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1), flush=True)
    result["L_on_panel7"] = panel7(model, tau)
    p7 = json.loads((ROOT / "results/jenga/phase1e_panel7_result.json").read_text())
    result["US_on_panel7"] = {"R0": p7["US_R0"], "R1": p7["US_R1"]}
    RESULT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("L_on_panel7", "US_on_panel7")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
