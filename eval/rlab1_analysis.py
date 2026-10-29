"""RLAB1 analysis as pre-registered (results/jenga/rlab1_protocol.json), written before the relabelled data, classifier or
values exist and before any episode with them.
  offline  every val-split latent of the four sources: recall of failure on red-fall-only latents (red fall, no neighbour
           topple), on neighbour-topple latents, and false alarms on safe latents, old (lsf_v1) vs new (lsf_v2_red) classifier
  dev      LSF-pub-r eps choice on dev 15000-15099 at 2x (fewest topples + red falls before extraction among eps with v4 >= 59)
  test     the relabelled arms on the H2H5 episodes, paired with the H2H5 arms (eval/h2h5_analysis.py statistics); H_red
usage: python3 eval/rlab1_analysis.py offline | dev | test   -> results/jenga/rlab1_{offline,dev,result}.json"""
import glob
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
OUTJ = ROOT / "results/jenga"
OLD, NEW = ROOT / "results/program_b/lsf_v1", ROOT / "results/program_b/lsf_v2_red"
SOURCES = ("closedloop_v1", "cf_v1", "cf_v2", "oil_v1")
EPS_DEV = ("010", "020", "030", "040", "050")
V4_MIN = 59


def offline():
    import torch
    import lsf as L
    from fresh_fragility_panel import neighbour_tilts, TOPPLE_DEG
    Classifier, _, _ = L.nets()
    clfs = {}
    for name, d in (("old", OLD), ("new", NEW)):
        c = Classifier().to("cuda").eval(); c.load_state_dict(torch.load(d / "classifier.pt", map_location="cuda")); clfs[name] = c
    cls, score = [], {k: [] for k in clfs}            # cls: 0 safe, 1 neighbour topple (with or without red fall), 2 red fall only
    for src in SOURCES:
        for f in sorted(glob.glob(str(ROOT / f"results/program_b/{src}/val/ep*.npz"))):
            if f.endswith(".tmp.npz"):
                continue
            d = np.load(f); tr, fi, acts = d["traces"], d["frame_index"], d["actions"]; elig = d["start_tilt"] < TOPPLE_DEG
            for k in range(len(tr)):
                st = [tr[k][min(i, len(tr[k]) - 1)] for i in fi]
                nb = np.maximum.accumulate([bool(np.any((neighbour_tilts(s) >= TOPPLE_DEG) & elig)) for s in st])
                rd = np.maximum.accumulate([L.red_tilt(s) >= TOPPLE_DEG for s in st])
                cls += list(np.where(nb, 1, np.where(rd, 2, 0)))
                z = torch.as_tensor(d["z"][k].astype(np.float32), device="cuda")
                g = torch.as_tensor(acts[k][np.minimum(fi, len(acts[k]) - 1)].astype(np.float32), device="cuda")
                with torch.no_grad():
                    for n, c in clfs.items():
                        score[n] += list(c(z, g).float().cpu().numpy())
        print(src, len(cls), flush=True)
    cls = np.array(cls); out = {"latents": {"safe": int((cls == 0).sum()), "neighbour_topple": int((cls == 1).sum()), "red_fall_only": int((cls == 2).sum())}}
    for n in clfs:
        s = np.array(score[n]); f = s < 0
        out[n] = {"recall_red_fall_only": round(float(f[cls == 2].mean()), 4), "recall_neighbour_topple": round(float(f[cls == 1].mean()), 4),
                  "false_alarm_safe": round(float(f[cls == 0].mean()), 4)}
    (OUTJ / "rlab1_offline.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


def _rows(path, run):
    return {r["ep"]: r for r in json.loads(Path(path).read_text())[run]["rows"]}


def dev():
    A = OUTJ / "outcome_audit_rl1dev.json"
    out = {}
    for e in EPS_DEV:
        R = _rows(A, f"paired_lsfp{e}_rl1dev").values()
        out[e] = {"topple": sum(r["logged_neighbour_toppled"] for r in R), "red_fall": sum(r["red_fell_before_extraction"] for r in R),
                  "v4": sum(r["v4_success"] for r in R), "v4b": sum(r["v4b_success"] for r in R), "episodes": len(R)}
    R = _rows(A, "paired_lsfa030_rl1dev").values()
    out["unisafe_r_030"] = {"topple": sum(r["logged_neighbour_toppled"] for r in R), "red_fall": sum(r["red_fell_before_extraction"] for r in R),
                            "v4": sum(r["v4_success"] for r in R), "v4b": sum(r["v4b_success"] for r in R), "episodes": len(R)}
    ok = [e for e in EPS_DEV if out[e]["v4"] >= V4_MIN]
    sel = min(ok, key=lambda e: (out[e]["topple"] + out[e]["red_fall"], int(e))) if ok else max(EPS_DEV, key=lambda e: (out[e]["v4"], -int(e)))
    res = {"sweep": out, "eligible": ok, "selected_eps": sel, "rule": "fewest topples + red falls among eps with v4 >= 59, ties -> smaller eps"}
    (OUTJ / "rlab1_dev.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
    (OUTJ / "rlab1_dev_selected.txt").write_text(sel + "\n")


def test():
    import h2h5_analysis as H
    sel = (OUTJ / "rlab1_dev_selected.txt").read_text().strip()
    new = {"unisafe_r": "paired_lsfa030_rl1_{n}", "lsfpub_r": f"paired_lsfp{sel}_rl1_{{n}}"}
    out = {"selected_pub_eps": sel}
    for nz in ("2x", "1x"):
        V, eps = H.load(nz)
        N = json.loads((OUTJ / f"outcome_audit_rl1_{nz}.json").read_text())
        for a, run in new.items():
            V[a] = {r["ep"]: r for r in N[run.format(n=nz)]["rows"]}
        eps = sorted(set(eps).intersection(*(set(V[a]) for a in new)))
        arms = list(H.ARMS) + list(new)
        pairs = [("unisafe_r", "unisafe"), ("lsfpub_r", "lsf_pub"), ("ours90", "unisafe_r"), ("ours90", "lsfpub_r"),
                 ("ours80", "unisafe_r"), ("ours80", "lsfpub_r"), ("unisafe_r", "policy"), ("lsfpub_r", "policy")]
        out[nz] = {"episodes": len(eps), "counts": {a: {k: int(H.vec(V, eps, a, k).sum()) for k in H.KEYS} for a in arms},
                   "pairs": {f"{a}_vs_{b}": {k: H.paired(V, eps, a, b, k) for k in H.KEYS} for a, b in pairs}}
    out["H_red"] = {a: out["2x"]["pairs"][f"{a}_vs_{b}"]["red_fall"]["p_one_sided_a_fewer"] < 0.05 for a, b in (("unisafe_r", "unisafe"), ("lsfpub_r", "lsf_pub"))}
    c = out["2x"]["counts"]
    out["baseline_choice"] = {"unisafe": "unisafe_r" if c["unisafe_r"]["v4_success"] > c["unisafe"]["v4_success"] else "unisafe",
                              "lsf_pub": "lsfpub_r" if c["lsfpub_r"]["v4_success"] > c["lsf_pub"]["v4_success"] else "lsf_pub"}
    (OUTJ / "rlab1_result.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"offline": offline, "dev": dev, "test": test}[sys.argv[1]]()
