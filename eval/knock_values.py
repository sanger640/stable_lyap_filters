"""KNOCK stage 3 (results/jenga/knock_protocol.json stage3): helpers for the values trained on knock-down data.
  plans  the expert's plan at each value-training reset (results/program_b/lsf_knock/train_reset_plan.npy, rows 1-8 of the
         recorded 16-target plan = the executed chunk), in eval/lsf.py data's order, checked against its reset latents
  gate   G3: from the 1-chunk gate runs on the knock val episodes 44130-44149 (paired_lfc000_knock3g_<arm>), the number of
         episodes whose first chunk was edited; an arm runs only if <= 5 of 20 (Jenga references: 0 / 300 for ours90 and
         frag in ABL2; knock stage 2: 138 / 200 and 184 / 200)
  fit    M3 (amendment 1): in-sample recall of a knock margin (share of its own training labels with a failure target that
         it predicts below 0) must be >= half its Jenga counterpart's (undo margin 0.93, fork margin 0.537)
usage: python3 eval/knock_values.py plans | gate <arm> | fit <irr|frag>   (gate / fit exit 1 on a fail)"""
import glob
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LSF = ROOT / "results/program_b/lsf_knock"
REC = ROOT / "results/program_b/policy_rollouts/paired_record_knock_wm"
RUNS = ROOT / "results/program_b/policy_rollouts"
GATE_EPS, GATE_MAX = range(44130, 44150), 5


def plans():
    files = [f for f in sorted(glob.glob(str(ROOT / "results/program_b/knock_v1/train/ep*.npz"))) if not f.endswith(".tmp.npz")]
    Rz = np.load(LSF / "train_reset_z.npy"); assert len(Rz) == len(files), (len(Rz), len(files))
    out, cache = [], {}
    for n, f in enumerate(files):
        e, c = (int(x) for x in Path(f).stem[2:].split("_s"))
        if e not in cache:
            cache[e] = dict(np.load(REC / f"ep{e}.npz"))
        r = cache[e]; i = int(np.flatnonzero(r["chunk_index"] == c)[0])
        out.append(r["plan16"][i, 1:9])
        if n % 100 == 0:                                                   # same reset as lsf.py's (z0 of the future file)
            assert np.array_equal(np.load(f)["z0"].astype(np.float16), Rz[n]), f
    np.save(LSF / "train_reset_plan.npy", np.stack(out).astype(np.float32)); print("plans", len(out), flush=True)


def gate(arm):
    rows = {int(f.stem[2:]): json.loads(f.read_text()) for f in (RUNS / f"paired_lfc000_knock3g_{arm}").glob("ep*.json")}
    missing = [e for e in GATE_EPS if e not in rows]
    edited = sorted(e for e in GATE_EPS if e in rows and rows[e]["log"] and rows[e]["log"][0].get("intervened"))
    out = {"protocol": "results/jenga/knock_protocol.json stage3 G3", "arm": arm, "episodes": len(GATE_EPS) - len(missing),
           "missing": missing, "chunk0_edited": len(edited), "edited_episodes": edited, "max_allowed": GATE_MAX,
           "pass": bool(not missing and len(edited) <= GATE_MAX)}
    path = ROOT / "results/knock/stage3_gate.json"
    allg = json.loads(path.read_text()) if path.exists() else {}
    allg[arm] = out; path.write_text(json.dumps(allg, indent=1) + "\n"); print(json.dumps(out), flush=True)
    sys.exit(0 if out["pass"] else 1)


JENGA_RECALL = {"irr": 0.93, "frag": 0.537}


def fit(kind):
    import torch
    sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
    import lsf as L
    C, Cr, _ = L.nets()
    if kind == "irr":
        T = {k: np.concatenate([np.load(f)[k] for f in sorted(glob.glob(str(ROOT / "results/jenga/irrm2_labels_nm8_knk_*.npz")))]) for k in ("z", "g", "d", "r")}
        w = torch.load(LSF / "margin_irrnm_nl8_knk.pt"); net = Cr().cuda().eval(); net.load_state_dict(w["state"])
        f = T["r"] > w["thr"]
        X = lambda b: (torch.as_tensor(T["z"][b:b + 1024], device="cuda").float(), torch.as_tensor(T["g"][b:b + 1024], device="cuda").float(),
                       torch.cat([torch.as_tensor(T["d"][b:b + 1024], device="cuda").float(), torch.zeros(len(T["d"][b:b + 1024]), 2, device="cuda")], 1))
        pred = lambda b: torch.tanh(net(*X(b))); n = len(f)
    else:
        import lflsf as F
        Z, M, _ = F.load((F.RUN,), lambda e: e < F.SPLIT)
        w = torch.load(LSF / "margin_lf.pt"); net = C().cuda().eval(); net.load_state_dict(w["state"])
        f = np.tanh((M - w["q10"]) / w["s"]) < 0
        pred = lambda b: net(torch.as_tensor(Z[b:b + 1024], device="cuda").float(), torch.zeros(len(Z[b:b + 1024]), 4, device="cuda")); n = len(f)
    with torch.no_grad():
        p = np.concatenate([pred(b).cpu().numpy() for b in range(0, n, 1024)])
    rec = float((p[f] < 0).mean()) if f.any() else 0.0
    out = {"protocol": "results/jenga/knock_protocol.json stage3 amendment_1 M3", "margin": kind, "train_labels": int(n), "failure_targets": int(f.sum()),
           "recall_in_sample": round(rec, 3), "false_flag_in_sample": round(float((p[~f] < 0).mean()), 4), "jenga_recall": JENGA_RECALL[kind],
           "min_required": round(JENGA_RECALL[kind] / 2, 3), "pass": bool(rec >= JENGA_RECALL[kind] / 2)}
    path = ROOT / "results/knock/stage3_margin_fit.json"
    allm = json.loads(path.read_text()) if path.exists() else {}
    allm[kind] = out; path.write_text(json.dumps(allm, indent=1) + "\n"); print(json.dumps(out), flush=True)
    sys.exit(0 if out["pass"] else 1)


if __name__ == "__main__":
    {"plans": plans, "gate": lambda: gate(sys.argv[2]), "fit": lambda: fit(sys.argv[2])}[sys.argv[1]]()
