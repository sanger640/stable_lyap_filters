"""3-block hybrid: validation floor (as phase1d_models.floor for the plain model) and re-grade of hold-30 + floor on the
corrected H (results/program_b/phase1/h_v2). Error per future: hybrid sample k vs real future k at hold 30, on
step3b/val (held out from the hybrid's training), exactly recovered states only."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]


def main():
    import torch
    import phase1d_models as p1d
    import phase2_hybrid as hy
    import step3_world_model as wm
    import step3b_models as sm
    from phase2_amend4 import alarm, score
    T1 = ROOT / "results/program_b/t1_hybrid"
    s = torch.load(T1 / "model_HYB_T1.pt", map_location="cuda")
    head = hy.build_head(s["action_dim"]).cuda(); head.load_state_dict(s["state_dict"]); head.eval()
    det = wm.WorldModel.load(T1 / "model_US.pt"); det.net.eval()
    inexact = set(json.loads((sm.DATA / "val_recovery.json").read_text())["inexact"])
    errs = []
    for n, path in enumerate(sorted((sm.DATA / "val").glob("ep*.npz"))):
        if path.name.endswith(".tmp.npz") or path.stem in inexact:
            continue
        with np.load(path) as f:
            k = len(f["z"])
            z0 = torch.as_tensor(f["z0"], device="cuda").float()[None].expand(k, -1, -1, -1).contiguous()
            act = det._act(torch.as_tensor(p1d.augment(f["actions"], None, False), device="cuda"))
            real = f["z"][:, wm.HOLD30].reshape(k, -1).astype(np.float64)
        g = torch.Generator(device="cuda").manual_seed(hy.MONITOR_SEED + 10_000 + n)
        fut = hy.sample(head, det, z0, act, s["residual_scale"], g)
        e30 = fut[:, 1].reshape(k, -1).double().cpu().numpy()
        errs.extend(np.linalg.norm(e30 - real, axis=1) / np.sqrt(real.shape[1] // 3))
        torch.cuda.empty_cache()
    tau = float(np.mean(errs))
    res = json.loads((ROOT / "results/jenga/threeblock_v2_result.json").read_text())
    tau_h = json.loads((ROOT / "results/jenga/hybrid_T1_result.json").read_text())["tau_H"]
    # recompute hold-30 + floor alarms with the validation floor (same seeds/samples as threeblock_v2 grade)
    hp = ROOT / "results/program_b/phase1/h_v2"
    names = [r["state"] for r in res["rows"]]
    alarms = {}
    with torch.no_grad():
        for n, name in enumerate(names):
            with np.load(hp / f"{name}.npz") as f:
                z0 = torch.as_tensor(f["z0"], device="cuda").float()[None].expand(64, -1, -1, -1).contiguous()
                act = det._act(torch.as_tensor(p1d.augment(f["actions"], None, False), device="cuda"))
            g = torch.Generator(device="cuda").manual_seed(hy.MONITOR_SEED + n)
            fut = hy.sample(head, det, z0, act, s["residual_scale"], g)
            alarms[name] = bool(alarm(fut[:, 0].reshape(64, -1).double().cpu().numpy(), fut[:, 1].reshape(64, -1).double().cpu().numpy(), "H30_floor", tau))
            torch.cuda.empty_cache()
    out = {"tau_validation": tau, "validation_futures": len(errs), "tau_H_in_sample_before": tau_h}
    keep = [r for r in res["rows"] if r["at_rest"]]
    for lab, pk, nk in (("any_outcome", "any_fork", "any_safe"), ("harm", "harm", "harm_safe")):
        sub = [r for r in keep if r[pk] or r[nk]]
        y = np.array([r[pk] for r in sub])
        out[lab] = {"HYB_T1_H30_floor_validation": score(np.array([alarms[r["state"]] for r in sub]), y),
                    "HYB_T1_BOTH": res[lab]["HYB_T1_BOTH"], "US_H30_floor": res[lab]["US_H30_floor"], "US_BOTH": res[lab]["US_BOTH"]}
    (ROOT / "results/jenga/hybrid_T1_floor_result.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
