"""World-model readout fix R1 (amendment 1): a minimum-gap floor in the split test equal to the model's own hold-30 error.

Frozen protocol: results/jenga/wm_readout_fix_protocol.json. Models: the paper's deterministic token WM
(wm_det_iter1) and Phase-1 cell E. R0 = unchanged split test; R1 = R0 and hold-30 best-split separation > the
model's mean hold-30 prediction error on its own validation set (same units; no outcome labels).
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
PROTOCOL = ROOT / "results/jenga/wm_readout_fix_protocol.json"
RESULT = ROOT / "results/jenga/wm_readout_fix_result.json"
CAL_SEED, N_CAL, FUT_SEED = 20261110, 200, 20261082


def calibration_rows():
    import uncertainty_monitor_dev as um
    import visual_pipeline_panel3 as p3
    import phase1_models as p1
    from fresh_fragility_panel import topple_count
    snippets = np.load(um.BENCH)["snippets"]
    errors = p3.ErrorModel()
    candidates = []
    for path in sorted((p1.OUT / "train").glob("ep*_seed*_s*.npz")):
        if path.name.endswith(".tmp.npz"):
            continue
        with np.load(path) as f:
            if len(f["z"]) == 8 and topple_count(f["traces"], f["start_tilt"]) == 0:
                candidates.append(path)
    pick = np.random.default_rng(CAL_SEED).choice(len(candidates), N_CAL, replace=False)
    rows = []
    for i in sorted(pick):
        path = candidates[i]
        e, rest = path.stem[2:].split("_seed")
        s, idx = rest.split("_s")
        key = (int(e), int(s), int(idx))
        with np.load(path) as f:
            chunks = []
            for k in range(8):                                    # recover the planned chunk exactly
                noise = snippets[np.random.default_rng([FUT_SEED, *key, k]).integers(len(snippets))]
                c = f["actions"][k, :8].astype(np.float64).copy(); c[:, :3] += noise
                chunks.append(c)
            spread = float(np.abs(np.stack(chunks) - chunks[0]).max())
            assert spread < 1e-5, (path.name, spread)
            chunk = chunks[0].astype(np.float32)
            acts = []
            for j in range(64):
                a = chunk.copy(); a[:, :3] -= snippets[j]
                acts.append(np.concatenate([a, np.repeat(chunk[-1:], 30, 0)]))
            draws = [errors.sample(CAL_SEED, *key, j) for j in range(64)]
            rows.append({"name": path.stem, "z0": f["z0"], "actions": np.stack(acts).astype(np.float32),
                         "d": np.stack([np.concatenate([p.reshape(9), r.reshape(9)]) for p, r in draws])})
    return rows, len(candidates)


def model_error(model_name, model):
    """Mean ||predicted - real|| / sqrt(D//3) at hold 30 over the model's validation futures (as deployed)."""
    import torch
    import step3_world_model as wm
    import step3b_models as sm
    folder = (wm.DATA if model_name == "wm_det_iter1" else sm.DATA) / "val"
    gop = sm.G.load(sm.DATA / "G.pt") if model_name == "cell_E" else None
    errs = []
    for path in sorted(folder.glob("ep*.npz")):
        if path.name.endswith(".tmp.npz"):
            continue
        with np.load(path) as f:
            real = f["z"][:, wm.HOLD30].reshape(len(f["z"]), -1).astype(np.float64)
            acts = f["actions"].astype(np.float32)
            z0 = torch.as_tensor(f["z0"], device=wm.DEVICE).float()
            with torch.no_grad():
                if gop is not None:
                    d = np.concatenate([f["d_pos"].reshape(len(acts), 9), f["d_rot"].reshape(len(acts), 9)], 1)
                    starts = gop(z0[None].expand(len(acts), -1, -1, -1), torch.as_tensor(d).float())
                    pred = sm.roll(model, starts, acts)[1]
                else:
                    z = z0[None].expand(len(acts), -1, -1, -1).contiguous()
                    pairs = torch.as_tensor(acts, device=wm.DEVICE).reshape(len(acts), wm.STEPS, -1)
                    for t in range(wm.STEPS):
                        z = model.step(z, pairs[:, t])
                        if t == wm.HOLD30:
                            pred = z.reshape(len(z), -1).double().cpu().numpy()
        d_scale = np.sqrt(real.shape[1] // 3)
        errs.extend(np.linalg.norm(pred - real, axis=1) / d_scale)
    return float(np.mean(errs)), len(errs)


def endings(model_name, model, row, seed):
    import torch
    import phase1_models as p1
    import step3_world_model as wm
    if model_name == "cell_E":
        return p1.model_endings("E", model, row, seed)
    return wm.wm_rollouts(model, np.asarray(row["z0"], np.float32), row["actions"], seed)


def decide(e10, e30):
    from counterfactual_monitor import persistent_mode_alarm
    from outcome_modes import two_mode_test
    return bool(persistent_mode_alarm(e10, e30).alarm), float(two_mode_test(e30, 0.0).separation_rms_mm)


def main():
    import torch
    import phase1_models as p1
    import step3_world_model as wm
    protocol = json.loads(PROTOCOL.read_text())
    if not protocol["status"].startswith("frozen"):
        raise SystemExit("protocol not frozen")
    models = {"wm_det_iter1": wm.WorldModel.load(wm.DATA / "wm_deterministic.pt"),
              "cell_E": wm.WorldModel.load(p1.OUT / "model_E.pt")}
    h = p1.h_states()
    fork = np.array([r["fork"] for r in h])
    result = {"protocol": protocol["id"], "rule": "amendment_1 (model-error floor)"}
    for name, model in models.items():
        tau, n_val = model_error(name, model)
        r0, sep = [], []
        for n, row in enumerate(h):
            a, s = decide(*endings(name, model, row, 20261085 + n))
            r0.append(a); sep.append(s)
            torch.cuda.empty_cache()
        r0, sep = np.array(r0), np.array(sep)
        r1 = r0 & (sep > tau)

        def grade(a):
            return {"caught": float((a & fork).sum() / fork.sum()), "false_alarm": float((a & ~fork).sum() / (~fork).sum()),
                    "alarms": int(a.sum()), "forks_caught": int((a & fork).sum()), "false_alarms": int((a & ~fork).sum())}
        g0, g1 = grade(r0), grade(r1)
        passed = bool(g1["false_alarms"] <= 0.5 * g0["false_alarms"] and g0["forks_caught"] - g1["forks_caught"] <= 2)
        result[name] = {"tau": tau, "validation_futures": n_val,
                        "R0": g0, "R1": g1, "R1_passes": passed,
                        "removed_false_alarms": int(g0["false_alarms"] - g1["false_alarms"]),
                        "removed_forks": int(g0["forks_caught"] - g1["forks_caught"]),
                        "separation_of_R0_alarms": {"forks": sorted(round(float(x), 3) for x in sep[r0 & fork]),
                                                    "safe": sorted(round(float(x), 3) for x in sep[r0 & ~fork])}}
        print(name, json.dumps({k: result[name][k] for k in ("tau", "R0", "R1", "R1_passes")}), flush=True)
        RESULT.write_text(json.dumps(result, indent=1) + "\n")
    stored = json.loads((ROOT / "results/jenga/phase1_selection.json").read_text())["E"]["alarms"]
    result["cell_E_R0_reproduces_stored_alarms"] = bool(result["cell_E"]["R0"]["alarms"] == stored)
    RESULT.write_text(json.dumps(result, indent=1) + "\n")
    print("cell_E R0 reproduces stored:", result["cell_E_R0_reproduces_stored_alarms"])


if __name__ == "__main__":
    main()
