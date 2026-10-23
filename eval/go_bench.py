"""GO-CEM offline gate (docs/FILTER_DEEP_DIVE.md 5f, results/jenga/go_protocol.json). Fresh benchmark-v2-format set
"m3int": interventions of the adopted filter (G-CEM, 3x margin) in the margin test (paired_cbfg3m_m3, episodes 19500-19799,
never used offline), one per episode (seeded). Its baseline is exactly the chunk G-CEM-3x executed in closed loop.
usage: python3 eval/go_bench.py prep ; (opt3.py run cem m3int go with OPT3_OBJ=o OPT3_MARGIN=3 ; merge ; truth) ; report
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
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
import bench2 as b2  # noqa: E402

RUN = ROOT / "results/program_b/policy_rollouts/paired_cbfg3m_m3"
SET, SEED = "m3int", 20261125


def prep():
    import cem_filter as cf
    import inspector_data as insp
    rng = np.random.default_rng(SEED); R = []
    for f in sorted(RUN.glob("ep*.json")):
        L = json.loads(f.read_text())["log"]
        kt = next((c["chunk"] for c in L if c.get("topple_started")), None)
        ks = [c["chunk"] for c in L if c.get("intervened") and (kt is None or c["chunk"] <= kt)]
        if ks:
            R.append({"ep": int(f.stem[2:]), "k": int(rng.choice(ks)), "kt": kt})
    mon = cf.Monitor(); sim = insp._sim()
    rec = {k: [] for k in ("state", "ctrl", "target", "gripper", "tilt", "plan16", "nominal", "executed", "site", "z0")}
    for x in R:
        with np.load(RUN / f"ep{x['ep']}.npz") as f:
            S = {k: f[k] for k in f.files}
        d = int(np.flatnonzero(S["chunk_index"] == x["k"])[0]); assert S["intervened"][d]
        sim.restore((S["state"][d], S["ctrl"][d], S["target"][d], float(S["gripper"][d])))
        for k in ("state", "ctrl", "target", "gripper", "plan16", "nominal", "executed"):
            rec[k].append(S[k][d])
        rec["tilt"].append(S["start_tilt"][d]); rec["site"].append(sim.data.site_xpos[sim.site].copy())
        mon.rng = np.random.default_rng([SEED, x["ep"], x["k"]])
        rec["z0"].append(mon.latent(mon.frame(sim)[None])[0].astype(np.float16))
    np.savez_compressed(b2.OUT / f"{SET}.npz", **{k: np.stack(v) for k, v in rec.items()})
    (b2.OUT / f"{SET}.json").write_text(json.dumps(R)); print(SET, len(R), "decisions", flush=True)


def merge(tag="go"):
    parts = sorted(b2.OUT.glob(f"opt3_{SET}_{tag}.part*.json"))
    rows = sorted((r for p in parts for r in json.loads(p.read_text())["rows"]), key=lambda r: r["i"])
    cfg = json.loads(parts[0].read_text())["cfg"]
    (b2.OUT / f"opt3_{SET}_{tag}.json").write_text(json.dumps({"cfg": cfg, "solver": "cem", "rows": rows}))
    print("merged", len(rows), "rows from", len(parts), "parts")


def report(tag="go"):
    """As pre-registered (results/jenga/go_protocol.json): GO vs the executed G-CEM-3x chunk, paired per decision."""
    C, R = b2.load(SET); n = len(R)
    G = json.loads((b2.OUT / f"opt3_{SET}_{tag}.json").read_text())["rows"]; assert len(G) == n
    arms = {"policy": [C["nominal"][i] for i in range(n)], "hold": [b2.hold_chunk(C, i) for i in range(n)],
            "g3_executed": [C["executed"][i] for i in range(n)], "go": [np.asarray(r["chunk"], np.float32) for r in G]}
    res = {a: np.array(b2.truth_chunks(SET, list(enumerate(ch)))) for a, ch in arms.items()}
    closed = np.array(C["gripper"][:n]) < 50
    rng = np.random.default_rng(20261126)
    out = {"set": SET, "decisions": n, "gripper_closed": int(closed.sum()),
           "go_kinds": {k: sum(r["kind"] == k for r in G) for k in ("certified", "certified_hold", "fallback_best")},
           "median_seconds": float(np.median([r["seconds"] for r in G]))}
    for a, v in res.items():
        out[a] = {"neighbour": round(float(v[:, 0].mean()), 4), "red_fall": round(float(v[:, 1].mean()), 4),
                  "red_fall_closed": round(float(v[closed, 1].mean()), 4), "red_fall_open": round(float(v[~closed, 1].mean()), 4)}

    def paired(a, b, col, m=None):
        m = np.ones(n, bool) if m is None else m
        d = res[a][m, col] - res[b][m, col]; bs = np.array([d[rng.integers(len(d), size=len(d))].mean() for _ in range(10000)])
        return {"diff_pp": round(100 * float(d.mean()), 3), "upper95_pp": round(100 * float(np.percentile(bs, 95)), 3),
                "lower95_pp": round(100 * float(np.percentile(bs, 5)), 3), "better": int((d < 0).sum()), "worse": int((d > 0).sum())}
    out["go_minus_g3_red"] = paired("go", "g3_executed", 1)
    out["go_minus_g3_neighbour"] = paired("go", "g3_executed", 0)
    out["go_minus_g3_red_closed"] = paired("go", "g3_executed", 1, closed)
    out["go_minus_g3_red_open"] = paired("go", "g3_executed", 1, ~closed)
    out["PRIMARY_pass"] = bool(out["go_minus_g3_red"]["upper95_pp"] < 0 and out["go_minus_g3_neighbour"]["upper95_pp"] <= 0.25)
    (ROOT / "results/jenga/go_offline_result.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"prep": prep, "merge": merge, "report": report}[sys.argv[1]](*sys.argv[2:])
