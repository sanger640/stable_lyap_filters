"""Exploratory: does a settling readout o (latent change between end-of-chunk and end-of-hold, max over futures)
flag the moments the diagnostic found risky? Uses the diag moments (risk_pol known)."""
import json, pickle, sys
from pathlib import Path
import numpy as np
ROOT = Path("/home/corey/wksp/stable_lyap_filters"); sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src"), str(ROOT / "libero")]
import pillar_filter as F, pillar_task as P, pillar_wm as W, pillar_diag as D
import torch
from wm_libero import auroc
sc = F.Scorer(); rows = {(r["seed"], r["k"]): r for f in sorted((D.OUT / "rows").glob("*.json")) for r in json.loads(f.read_text())}
rec, failed, ctrl, want = D.choose()
out = []
for f in sorted(D.STORE.glob("s*.pkl")):
    Dd = pickle.loads(f.read_bytes()); s = Dd["seed"]
    if s not in failed and s not in ctrl: continue
    sim = P.PillarSim(); P.reset(sim, s); cam = W.Cam(sim)
    for m in Dd["moments"]:
        sim.restore(m["snap"]); z0, p0 = sc.latent(sim, cam), W.proprio(sim)
        chunk = m["options"]["pol"][None]; nz = sc.noise(np.random.default_rng([31, s, m["k"]]))
        nzz = np.concatenate([np.zeros((1, 8, 3)), nz]); N = len(nzz)
        acts = np.concatenate([chunk, np.repeat(chunk[:, -1:], W.HOLD, 1)], 1)
        acts = np.repeat(acts[:, None], N, 1).copy(); acts[:, :, :8, :3] -= nzz[None]; acts[:, :, 8:, :3] -= nzz[None, :, -1:]
        acts = acts.reshape(N, 30, 4)
        inp = torch.as_tensor(W.model_inputs(acts, np.repeat(p0[None], N, 0)), device="cuda")
        z = torch.as_tensor(z0, device="cuda").float()[None].expand(N, -1, -1, -1).contiguous(); keep = {}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for t in range(W.STEPS):
                z = sc.model.step(z, inp[:, t]).float()
                if t in (W.END_CHUNK, W.END_HOLD): keep[t] = z.clone()
        a, b = keep[W.END_CHUNK].reshape(N, -1, 8), keep[W.END_HOLD].reshape(N, -1, 8)
        per = (b - a).pow(2).mean(-1)                                  # (N, patches)
        o_mean = per.mean(1); o_top = per.topk(16, dim=1).values.mean(1)  # whole-image vs most-changed 16 patches
        r = rows[(s, m["k"])]
        out.append({"seed": s, "k": m["k"], "risk": r["risk_pol"], "trig": r["triggered"],
                    "o_mean_max": float(o_mean.max()), "o_top_max": float(o_top.max()), "o_top_med": float(o_top.median())})
json.dump(out, open(ROOT / "results/pillar/diag/o_check_rows.json", "w"))
risk = np.array([r["risk"] for r in out])
for key in ("o_mean_max", "o_top_max", "o_top_med"):
    x = np.array([r[key] for r in out]); print(key, "AUROC risk>=0.25", round(auroc(x, risk >= 0.25), 3), "risk>=0.5", round(auroc(x, risk >= 0.5), 3))
print("moments", len(out), "risky>=0.25", int((risk >= 0.25).sum()))
