"""Relative-pass decisions (docs/JENGA_RESIDUAL.md): G-CEM closed-loop decisions (Stage 4 + 4b, cbfg) that were NOT
intervened although the policy's plan was forked (h_nom < h_safe), because h_nom >= h_hold. Builds a benchmark-v2-format
set "relpass" so opt3.py run / truth work unchanged; report compares the true one-chunk risk of the policy's chunk
(what closed loop executed), hold, and G-CEM with the absolute requirement (OPT3_ABS=1).
usage: python3 eval/relpass.py prep ; (opt3.py run cem relpass abs ; opt3.py truth relpass abs) ; report
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

PR = ROOT / "results/program_b/policy_rollouts"
HS = float(np.log2(3.0))


def select():
    rows = []
    for st in ("s4", "s4b"):
        for f in sorted((PR / f"paired_cbfg_{st}").glob("ep*.json")):
            d = json.loads(f.read_text()); L = d["log"]
            kt = next((c["chunk"] for c in L if c.get("topple_started")), None)
            for c in L:
                if kt is not None and c["chunk"] > kt:
                    continue
                if not c["intervened"] and c["h_nom"] < HS - 1e-6:
                    rows.append({"stage": st, "ep": d["episode"], "k": c["chunk"], "h_now": c["h_now"], "h_nom": c["h_nom"],
                                 "o_nom": c["o_nom"], "kt": kt, "topple_within_3": kt is not None and c["chunk"] >= kt - 3})
    return rows


def prep():
    import cem_filter as cf
    import inspector_data as insp
    R = select(); mon = cf.Monitor(); sim = insp._sim()
    rec = {k: [] for k in ("state", "ctrl", "target", "gripper", "tilt", "plan16", "nominal", "executed", "site", "z0")}
    for x in R:
        with np.load(PR / f"paired_cbfg_{x['stage']}/ep{x['ep']}.npz") as f:
            S = {k: f[k] for k in f.files}
        d = int(np.flatnonzero(S["chunk_index"] == x["k"])[0]); assert not S["intervened"][d]
        assert np.allclose(S["executed"][d], S["nominal"][d])
        sim.restore((S["state"][d], S["ctrl"][d], S["target"][d], float(S["gripper"][d])))
        for k in ("state", "ctrl", "target", "gripper", "plan16", "nominal", "executed"):
            rec[k].append(S[k][d])
        rec["tilt"].append(S["start_tilt"][d]); rec["site"].append(sim.data.site_xpos[sim.site].copy())
        mon.rng = np.random.default_rng([20261030, x["ep"], x["k"]])
        rec["z0"].append(mon.latent(mon.frame(sim)[None])[0].astype(np.float16))
    np.savez_compressed(b2.OUT / "relpass.npz", **{k: np.stack(v) for k, v in rec.items()})
    (b2.OUT / "relpass.json").write_text(json.dumps(R)); print("relpass", len(R), "decisions", flush=True)


def report(tag="abs"):
    C, R = b2.load("relpass"); n = len(R)
    G = json.loads((b2.OUT / f"opt3_relpass_{tag}.json").read_text())["rows"]; assert len(G) == n
    arms = {"policy": [C["nominal"][i] for i in range(n)], "hold": [b2.hold_chunk(C, i) for i in range(n)],
            "g_abs": [np.asarray(r["chunk"], np.float32) for r in G]}
    res = {a: np.array(b2.truth_chunks("relpass", list(enumerate(ch)))) for a, ch in arms.items()}
    out = {"decisions": n, "g_kinds": {k: sum(r["kind"] == k for r in G) for k in ("certified", "fallback_hold", "fallback_best")},
           "progress_kept_median": float(np.median([r["progress"] for r in G]))}
    for a, v in res.items():
        out[a] = {"neighbour_risk": round(float(v[:, 0].mean()), 4), "red_fall_risk": round(float(v[:, 1].mean()), 4)}
    rng = np.random.default_rng(20261031)

    def paired(a, b):
        d = res[a][:, 0] - res[b][:, 0]; bs = np.array([d[rng.integers(n, size=n)].mean() for _ in range(10000)])
        return {"mean": round(float(d.mean()), 4), "one_sided_95_upper": round(float(np.percentile(bs, 95)), 4),
                "p_one_sided_less": float((bs >= 0).mean()), "better": int((d < 0).sum()), "worse": int((d > 0).sum())}
    out["g_abs_minus_policy"] = paired("g_abs", "policy"); out["g_abs_minus_hold"] = paired("g_abs", "hold")
    out["hold_minus_policy"] = paired("hold", "policy")
    out["H1_pass"] = bool(out["g_abs_minus_policy"]["p_one_sided_less"] < 0.05)
    out["H2_pass"] = bool(out["g_abs_minus_hold"]["one_sided_95_upper"] <= 0.005)
    (b2.OUT / f"report_relpass_{tag}.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"prep": prep, "report": report}[sys.argv[1]](*sys.argv[2:])
