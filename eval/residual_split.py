"""Exploratory follow-up to eval/residual_check.py: at the filter arm's decisions whose logged barrier was h = 2 (no split
up to 3x), does the policy's plan split at 4x / 6x / 8x? Pre-topple decisions (the 4 chunks up to each residual topple start,
unfiltered) vs reference decisions with h = 2 from episodes without a topple (every 10th). Same 64 futures and seeds as
eval/margin_check.py. Tells whether a 4x margin would separate the residual topples from ordinary moves.
usage: CL_TAG=scl ARM=cbfg3m python3 eval/residual_split.py
"""
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.environ["MC_RUN"] = f"paired_{os.environ.get('ARM', 'cbfg3m')}_{os.environ.get('CL_TAG', 'scl')}"
sys.path[:0] = [str(ROOT / "eval")]
import margin_check as mc  # noqa: E402

mc.SC = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0)
TAG, ARM = os.environ.get("CL_TAG", "scl"), os.environ.get("ARM", "cbfg3m")


def main():
    import torch
    import cbf_filter as cbff
    import cem_filter as cf
    import inspector_data as insp
    mon = cf.Monitor(); cbf = cbff.CBF(mon); sim = insp._sim()
    R = json.loads((ROOT / f"results/jenga/deep_dive/residual_{TAG}_{ARM}.json").read_text())
    A = json.loads((ROOT / f"results/jenga/outcome_audit_{TAG}.json").read_text())[f"paired_{ARM}_{TAG}"]["rows"]
    safe = sorted(r["ep"] for r in A if not r["logged_neighbour_toppled"])[::10]
    jobs = []
    for t in R["rows_topple"]:
        if t["kt"] is not None:
            jobs += [(t["ep"], k, "pre") for k in range(t["kt"] - 3, t["kt"] + 1)]
    jobs += [(e, None, "ref") for e in safe]
    rows = []
    for ep, kk, kind in jobs:
        L = {c["chunk"]: c for c in json.loads((mc.RUN / f"ep{ep}.json").read_text())["log"]}
        with np.load(mc.RUN / f"ep{ep}.npz") as z:
            S = {k: z[k] for k in z.files}
        for d in range(len(S["chunk_index"])):
            k = int(S["chunk_index"][d]); c = L.get(k, {})
            if c.get("intervened") or c.get("h_nom") != 2.0 or (kk is not None and k != kk):
                continue
            sim.restore((S["state"][d], S["ctrl"][d], S["target"][d], float(S["gripper"][d])))
            mon.rng = np.random.default_rng([20261100, ep, k])
            z0 = torch.as_tensor(mon.latent(mon.frame(sim)[None])[0], device="cuda").float()
            plan = S["plan16"][d].copy(); plan[0, :3] = S["target"][d]
            hold = np.repeat(np.r_[S["target"][d], 0.0][None].astype(np.float32), 16, 0)
            split, o = mc.split_flags(cbf, sim, z0, np.stack([plan, hold]))
            rows.append({"ep": ep, "k": k, "kind": kind, "plan": split[0].tolist(), "hold": split[1].tolist()})
    out = {"scales": mc.SC, "n": {}, "plan_split_share": {}}
    for kind in ("pre", "ref"):
        X = np.array([r["plan"] for r in rows if r["kind"] == kind], bool); H = np.array([r["hold"] for r in rows if r["kind"] == kind], bool)
        out["n"][kind] = len(X)
        if len(X):
            out["plan_split_share"][kind] = {str(s): round(float(X[:, i].mean()), 3) for i, s in enumerate(mc.SC)}
            out["plan_split_not_hold_share_" + kind] = {str(s): round(float((X[:, i] & ~H[:, i]).mean()), 3) for i, s in enumerate(mc.SC)}
    out["rows"] = rows
    (ROOT / f"results/jenga/deep_dive/residual_split_{TAG}_{ARM}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
