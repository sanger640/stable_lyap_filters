"""NOISE1 (results/noise1/protocol.json): the same trained stochastic world model with its noise drawn three ways --
independent per future (as published), shared by all futures, or zero -- from true and perturbed starts, on the 60 D0
states of eval/step3_diagnostics.py. Only test-time noise changes; the network, data, seeds and readout are unchanged.
usage: python3 eval/noise_placement.py"""
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
import step3_data as sd  # noqa: E402
import step3_diagnostics as dg  # noqa: E402
import step3_world_model as wm  # noqa: E402

OUT = ROOT / "results/noise1"


@torch.no_grad()
def rollout(model, z0, actions, mode, seed):
    """mode: step (independent), shared (one draw per step for all futures), zero; deterministic models ignore it."""
    g = torch.Generator(device=wm.DEVICE).manual_seed(int(seed))
    n = len(actions)
    z = torch.as_tensor(z0, device=wm.DEVICE).float()
    z = z.expand(n, -1, -1, -1).contiguous() if len(z) == 1 else z
    pairs = torch.as_tensor(actions, device=wm.DEVICE).reshape(n, wm.STEPS, -1)
    ends = {}
    for t in range(wm.STEPS):
        a = model._act(pairs[:, t])
        if model.kind == "deterministic":
            r = model.net(z, torch.zeros_like(z), a, torch.zeros(n, device=wm.DEVICE))
        else:
            x = {"step": lambda: torch.randn(z.shape, device=wm.DEVICE, generator=g),
                 "shared": lambda: torch.randn(z.shape[1:], device=wm.DEVICE, generator=g)[None].expand(n, -1, -1, -1).clone(),
                 "zero": lambda: torch.zeros_like(z)}[mode]()
            for k in range(wm.EULER):
                x = x + model.net(z, x, a, torch.full((n,), k / wm.EULER, device=wm.DEVICE)) / wm.EULER
            r = x
        z = z + r * model.stats.res_std
        if t in (wm.HOLD10, wm.HOLD30):
            ends[t] = z.reshape(n, -1).double().cpu().numpy()
    return ends[wm.HOLD10], ends[wm.HOLD30]


def main():
    models = {"stochastic": wm.WorldModel.load(sd.OUT / "wm_stochastic.pt"),
              "deterministic": wm.WorldModel.load(sd.OUT / "wm_deterministic.pt")}
    variants = {f"S_{m}{p}": ("stochastic", m, p) for m in ("step", "shared", "zero") for p in ("", "_pert")}
    variants.update({"D": ("deterministic", "step", ""), "D_pert": ("deterministic", "step", "_pert")})
    rows = []
    for n, path in enumerate(sorted((sd.OUT / "d0").glob("ep*.npz"))):
        with np.load(path) as f, np.load(dg.OUT / f"pert_{path.stem}.npz") as p:
            real = f["z"].astype(np.float64)
            row = {"state": path.stem, "real": dg.describe(real[:, 0].reshape(64, -1), real[:, 1].reshape(64, -1))}
            for name, (kind, mode, pert) in variants.items():
                z0 = p["z0"].astype(np.float32) if pert else f["z0"][None].astype(np.float32)
                row[name] = dg.describe(*rollout(models[kind], z0, f["actions"], mode, seed=20261060 + n))
        rows.append(row); print(n, row["state"], row["real"]["alarm"], {k: row[k]["alarm"] for k in variants}, flush=True)
    fork = np.array([r["real"]["alarm"] for r in rows])
    summary = {}
    for name in variants:
        al = np.array([r[name]["alarm"] for r in rows])
        summary[name] = {"caught": f"{int((al & fork).sum())}/{int(fork.sum())}", "caught_pct": round(100 * (al & fork).sum() / fork.sum(), 1),
                         "false": f"{int((al & ~fork).sum())}/{int((~fork).sum())}", "false_pct": round(100 * (al & ~fork).sum() / (~fork).sum(), 1),
                         "median_spread_ratio_to_real": round(float(np.median([r[name]["spread"] / max(r["real"]["spread"], 1e-9) for r in rows])), 3),
                         "median_pc1_share_on_forks": round(float(np.median([r[name]["pc1_share"] for r in rows if r["real"]["alarm"]])), 3)}
    c = {k: summary[k]["caught_pct"] for k in summary}
    best = max(c["S_shared_pert"], c["S_zero_pert"])
    verdict = ("placement" if best >= 2 * c["S_step_pert"] and best >= c["D_pert"] - 10 else
               "dynamics" if max(abs(c["S_shared_pert"] - c["S_step_pert"]), abs(c["S_zero_pert"] - c["S_step_pert"])) <= 10 else "mixed")
    out = {"protocol": "results/noise1/protocol.json", "states": len(rows), "real_forks": int(fork.sum()), "summary": summary,
           "verdict": verdict}
    (OUT / "result.json").write_text(json.dumps(out, indent=1) + "\n"); (OUT / "rows.json").write_text(json.dumps(rows) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
