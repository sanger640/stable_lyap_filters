"""Exploratory deep dive (docs/FILTER_DEEP_DIVE.md): does the certificate RANK candidate chunks from the same state the way
the real risk does? Uses every chunk of a benchmark-v2 set that already has truth (policy, selection, hold, and all
optimiser outputs of opt2/opt3 runs). Each chunk is scored as chunk + hold tail (the same 8 actions the truth executes),
all chunks of a decision in ONE cbf.score call (common futures, 64), model_OIL.
usage: python3 eval/rank_check.py <dev|holdout|relpass>
"""
import json
import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl"); os.environ["HARD_MASS"] = "0"; os.environ.setdefault("EXEC_NOISE_SCALE", "2")
ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("CBF_WM", str(ROOT / "results/program_b/oil_v1/model/model_OIL.pt"))
os.environ.setdefault("CBF_CALIB", "results/jenga/cbf_calibration_OIL.json")
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src"), str(ROOT / "libero")]
import bench2 as b2  # noqa: E402


def chunks_for(name):
    C, R = b2.load(name); n = len(R); cand = {i: {} for i in range(n)}
    for i in range(n):
        cand[i]["policy"] = C["nominal"][i]; cand[i]["hold"] = b2.hold_chunk(C, i)
        if name != "relpass":
            cand[i]["selection"] = C["executed"][i]
    for f in sorted(b2.OUT.glob(f"opt[23]_{name}_*.json")):
        if "smoke" in f.name:
            continue
        for r in json.loads(f.read_text())["rows"]:
            cand[r["i"]][f.stem.split(f"{name}_", 1)[1]] = np.asarray(r["chunk"], np.float32)
    truth = json.loads((b2.OUT / f"truth_{name}.json").read_text())
    out = {}
    for i, d in cand.items():
        keep = {a: c for a, c in d.items() if b2._hash(i, c) in truth}
        out[i] = {a: (c, truth[b2._hash(i, c)][0]) for a, c in keep.items()}
    return C, out


def main(name):
    import torch
    import cem_filter as cf
    import cbf_filter as cbff
    import inspector_data as insp
    from wm_libero import auroc
    cbff.CFG["H_SAFE"] = float(np.log2(3.0))
    C, cand = chunks_for(name); mon = cf.Monitor(); cbf = cbff.CBF(mon); sim = insp._sim(); rows = []
    for i, d in cand.items():
        if len(d) < 2:
            continue
        sim.restore((C["state"][i], C["ctrl"][i], C["target"][i], float(C["gripper"][i])))
        z0 = torch.as_tensor(C["z0"][i].astype(np.float32), device="cuda")
        names = list(d); tgt = np.r_[C["target"][i], 0.0]
        plans = np.stack([np.concatenate([tgt[None], d[a][0], np.repeat(d[a][0][-1:], 7, 0)]).astype(np.float32) for a in names])
        plans[:, 0, 3] = plans[:, 1, 3]
        mon.rng = np.random.default_rng([20261040, i])
        h, o = cbf.score(sim, z0, plans, 64)
        for a, hh, oo in zip(names, h, o):
            rows.append({"i": i, "arm": a, "risk": d[a][1], "h": float(hh), "o": float(oo)})
    D = ROOT / "results/jenga/deep_dive"; D.mkdir(parents=True, exist_ok=True)
    (D / f"rank_{name}_rows.json").write_text(json.dumps(rows))
    analyse(name, rows, cbf)


def analyse(name, rows, cbf=None):
    from wm_libero import auroc
    import cbf_filter as cbff
    out = {"set": name, "chunks": len(rows), "decisions": len({r["i"] for r in rows})}
    risk = np.array([r["risk"] for r in rows]); hv = np.array([r["h"] for r in rows]); ov = np.array([r["o"] for r in rows])
    y = risk >= 2 / 32
    out["pooled_auroc_minus_h"] = auroc(-hv, y); out["pooled_auroc_o"] = auroc(ov, y)
    # within-decision pairwise concordance, pairs whose real risk differs by >= 2 futures
    conc = {"h": [0, 0, 0], "o": [0, 0, 0]}
    for i in {r["i"] for r in rows}:
        rs = [r for r in rows if r["i"] == i]
        for a in range(len(rs)):
            for b in range(a + 1, len(rs)):
                dr = rs[a]["risk"] - rs[b]["risk"]
                if abs(dr) < 2 / 32 - 1e-9:
                    continue
                for k, sgn in (("h", -1), ("o", 1)):           # risk should rise as h falls / o rises
                    ds = sgn * (rs[a][k] - rs[b][k])
                    conc[k][0 if ds * dr > 0 else (1 if ds * dr < 0 else 2)] += 1
    for k, (c, dcn, t) in conc.items():
        out[f"within_{k}"] = {"concordant": c, "discordant": dcn, "tied": t,
                              "concordance_excl_ties": round(c / max(c + dcn, 1), 3)}
    # does certification (ok) separate risk within decisions?
    O_MAX = json.loads((ROOT / os.environ["CBF_CALIB"]).read_text())["O_MAX"]
    okf = (hv >= np.log2(3.0) - 1e-9) & (ov <= O_MAX)                         # absolute certificate
    out["risk_mean_certified"] = float(risk[okf].mean()) if okf.any() else None
    out["risk_mean_not_certified"] = float(risk[~okf].mean()) if (~okf).any() else None
    by_arm = {}
    for r, k in zip(rows, okf):
        by_arm.setdefault(r["arm"], []).append((r["risk"], r["h"], r["o"], bool(k)))
    out["by_arm"] = {a: {"n": len(v), "risk": round(float(np.mean([x[0] for x in v])), 4),
                         "h": round(float(np.mean([x[1] for x in v])), 3), "o": round(float(np.mean([x[2] for x in v])), 3),
                         "cert_rate": round(float(np.mean([x[3] for x in v])), 3)} for a, v in by_arm.items()}
    (ROOT / "results/jenga/deep_dive" / f"rank_{name}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "analyse":
        analyse(sys.argv[1], json.loads((ROOT / f"results/jenga/deep_dive/rank_{sys.argv[1]}_rows.json").read_text()))
    else:
        main(sys.argv[1])
