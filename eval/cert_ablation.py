"""Offline comparison of G-CEM-3x variants on m3int (interventions of the adopted 3x filter in the margin test; exploratory):
real one-chunk neighbour / red-fall risk (bench2 truth, 32 sim futures) and progress kept, paired per decision vs g3re (the
adopted filter re-run, OIL+ENS certificate). usage: python3 eval/cert_ablation.py g3oil [g3s ...] -> results/jenga/deep_dive/cert_ablation.json
(must run as a file: the truth pool uses multiprocessing spawn)"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval")]
import bench2 as b2  # noqa: E402


def main(tags):
    C, R = b2.load("m3int")
    G = {t: {r["i"]: r for r in json.loads((b2.OUT / f"opt3_m3int_{t}.json").read_text())["rows"]} for t in ["g3re"] + tags}
    I = sorted(set.intersection(*[set(g) for g in G.values()]))
    T = {t: np.array(b2.truth_chunks("m3int", [(i, np.asarray(G[t][i]["chunk"], np.float32)) for i in I])) for t in G}
    pol = np.array(b2.truth_chunks("m3int", [(i, C["nominal"][i]) for i in I]))
    ex = np.array(b2.truth_chunks("m3int", [(i, C["executed"][i]) for i in I]))
    out = {"decisions": len(I), "exploratory": True, "policy": {"neighbour": round(float(pol[:, 0].mean()), 4), "red": round(float(pol[:, 1].mean()), 4)},
           "executed_closed_loop": {"neighbour": round(float(ex[:, 0].mean()), 4), "red": round(float(ex[:, 1].mean()), 4)}}
    A = {r["ep"]: r for r in json.loads((ROOT / "results/jenga/outcome_audit_m3.json").read_text())["paired_cbfg3m_m3"]["rows"]}
    tout = np.array([not A[R[i]["ep"]]["logged_neighbour_toppled"] and not A[R[i]["ep"]]["extracted"]
                     and not A[R[i]["ep"]]["red_fell_before_extraction"] for i in I])     # time-out episodes of the 3x arm (5j)
    out["timeout_decisions"] = int(tout.sum())

    def proj(i, ch):                                                             # progress along the plan's own direction (opt3.progress_proj)
        p0 = C["plan16"][i].copy(); p0[0, :3] = C["target"][i]; a = p0[8, :3] - p0[0, :3]; n2 = float(a @ a)
        return float((np.asarray(ch)[7, :3] - p0[0, :3]) @ a / n2) if n2 > 1e-8 else 1.0
    PJ = {t: np.array([proj(i, G[t][i]["chunk"]) for i in I]) for t in G}
    rng = np.random.default_rng(1)
    for t in G:
        pr = np.array([G[t][i]["progress"] for i in I])
        out[t] = {"neighbour": round(float(T[t][:, 0].mean()), 4), "red": round(float(T[t][:, 1].mean()), 4),
                  "progress_median": float(np.median(pr)), "progress_mean": round(float(pr.mean()), 3),
                  "hold_equivalent": round(float((pr < 0.1).mean()), 3), "seconds_median": float(np.median([G[t][i]["seconds"] for i in I])),
                  "progress_proj_median": round(float(np.median(PJ[t])), 3), "progress_proj_timeout_median": round(float(np.median(PJ[t][tout])), 3),
                  "neighbour_timeout": round(float(T[t][tout, 0].mean()), 4)}
        if t == "g3re":
            continue
        for col, lab in ((0, "neighbour"), (1, "red")):
            x = T[t][:, col] - T["g3re"][:, col]; bs = [x[rng.integers(len(x), size=len(x))].mean() for _ in range(10000)]
            out[t][f"minus_g3re_{lab}_pp"] = [round(100 * x.mean(), 2), round(100 * np.percentile(bs, 5), 2), round(100 * np.percentile(bs, 95), 2)]
        d = np.array([G[t][i]["progress"] - G["g3re"][i]["progress"] for i in I])
        out[t]["progress_minus_g3re"] = {"mean": round(float(d.mean()), 3), "better": int((d > 0.05).sum()), "worse": int((d < -0.05).sum())}
        for ref in [r for r in G if r not in ("g3re", t)]:                        # proj progress vs every other variant (e.g. steer vs g3oil)
            d = PJ[t] - PJ[ref]
            out[t][f"proj_minus_{ref}"] = {"mean": round(float(d.mean()), 3), "timeout_mean": round(float(d[tout].mean()), 3),
                                           "better": int((d > 0.05).sum()), "worse": int((d < -0.05).sum())}
            for col, lab in ((0, "neighbour"), (1, "red")):
                x = T[t][:, col] - T[ref][:, col]; bs = [x[rng.integers(len(x), size=len(x))].mean() for _ in range(10000)]
                out[t][f"minus_{ref}_{lab}_pp"] = [round(100 * x.mean(), 2), round(100 * np.percentile(bs, 5), 2), round(100 * np.percentile(bs, 95), 2)]
    (ROOT / "results/jenga/deep_dive/cert_ablation.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
