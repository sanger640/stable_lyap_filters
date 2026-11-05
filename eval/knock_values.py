"""KNOCK stage 3 (results/jenga/knock_protocol.json stage3): helpers for the values trained on knock-down data.
  plans  the expert's plan at each value-training reset (results/program_b/lsf_knock/train_reset_plan.npy, rows 1-8 of the
         recorded 16-target plan = the executed chunk), in eval/lsf.py data's order, checked against its reset latents
  gate   G3: from the 1-chunk gate runs on the knock val episodes 44130-44149 (paired_lfc000_knock3g_<arm>), the number of
         episodes whose first chunk was edited; an arm runs only if <= 5 of 20 (Jenga references: 0 / 300 for ours90 and
         frag in ABL2; knock stage 2: 138 / 200 and 184 / 200)
usage: python3 eval/knock_values.py plans | gate <arm>   (gate exits 1 on a fail)"""
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


if __name__ == "__main__":
    {"plans": plans, "gate": lambda: gate(sys.argv[2])}[sys.argv[1]]()
