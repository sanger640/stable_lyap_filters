"""Exploratory (protocol addendum 3): the model-error floor (R1) on Phase-1b a1, a2 and Phase-1c conv3, vit3, on H."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
RESULT = ROOT / "results/jenga/wm_readout_fix_more_result.json"


def main():
    import torch
    import phase1_models as p1
    import phase1c_models as pc
    import step3_world_model as wm
    import step3b_models as sm
    from counterfactual_monitor import persistent_mode_alarm
    from outcome_modes import two_mode_test
    gop = sm.G.load(sm.DATA / "G.pt")
    models = {"a1": ("conv", wm.WorldModel.load(ROOT / "results/program_b/phase1b/model_a1.pt")),
              "a2": ("conv", wm.WorldModel.load(ROOT / "results/program_b/phase1b/model_a2.pt")),
              "conv3": ("hist", pc.HistModel.load(ROOT / "results/program_b/phase1c/model_conv3.pt")),
              "vit3": ("hist", pc.HistModel.load(ROOT / "results/program_b/phase1c/model_vit3.pt"))}
    stored = {**json.loads((ROOT / "results/jenga/phase1b_selection.json").read_text()),
              **json.loads((ROOT / "results/jenga/phase1c_selection.json").read_text())}

    @torch.no_grad()
    def roll(kind, model, starts, actions):
        if kind == "conv":
            return sm.roll(model, starts, actions)
        outs = model.rollout(starts, torch.as_tensor(actions, device=wm.DEVICE).reshape(len(starts), wm.STEPS, -1))
        return (outs[wm.HOLD10].reshape(len(starts), -1).double().cpu().numpy(),
                outs[wm.HOLD30].reshape(len(starts), -1).double().cpu().numpy())

    rows = p1.h_states()
    fork = np.array([r["fork"] for r in rows])
    result = {"status": "exploratory (protocol addendum 3); descriptive on H only"}
    for name, (kind, model) in models.items():
        errs = []
        for path in sorted((sm.DATA / "val").glob("ep*.npz")):
            if path.name.endswith(".tmp.npz"):
                continue
            with np.load(path) as f:
                n = len(f["z"])
                d = np.concatenate([f["d_pos"].reshape(n, 9), f["d_rot"].reshape(n, 9)], 1)
                z0 = torch.as_tensor(f["z0"], device=wm.DEVICE).float()
                starts = gop(z0[None].expand(n, -1, -1, -1), torch.as_tensor(d).float())
                pred = roll(kind, model, starts, f["actions"].astype(np.float32))[1]
                real = f["z"][:, wm.HOLD30].reshape(n, -1).astype(np.float64)
                errs.extend(np.linalg.norm(pred - real, axis=1) / np.sqrt(real.shape[1] // 3))
        tau = float(np.mean(errs))
        r0, sep = [], []
        for row in rows:
            z0 = torch.as_tensor(row["z0"], device=wm.DEVICE).float()
            starts = gop(z0[None].expand(64, -1, -1, -1), torch.as_tensor(row["d"]).float())
            e10, e30 = roll(kind, model, starts, row["actions"])
            r0.append(bool(persistent_mode_alarm(e10, e30).alarm))
            sep.append(float(two_mode_test(e30, 0.0).separation_rms_mm))
            torch.cuda.empty_cache()
        r0, sep = np.array(r0), np.array(sep)
        r1 = r0 & (sep > tau)

        def g(a):
            return {"caught": round(float((a & fork).sum() / fork.sum()), 4),
                    "false_alarm": round(float((a & ~fork).sum() / (~fork).sum()), 4),
                    "forks_caught": int((a & fork).sum()), "false_alarms": int((a & ~fork).sum())}
        g0, g1 = g(r0), g(r1)
        result[name] = {"tau": tau, "R0": g0, "R1": g1,
                        "R0_reproduces_stored_alarms": bool(int(r0.sum()) == stored[name]["alarms"]),
                        "would_pass_frozen_rule": bool(g1["false_alarms"] <= 0.5 * g0["false_alarms"]
                                                       and g0["forks_caught"] - g1["forks_caught"] <= 2)}
        print(name, json.dumps(result[name]), flush=True)
    RESULT.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
