"""TO2x analysis as pre-registered (results/jenga/to2x_protocol.json): head-to-head 4's outcomes with every time-out episode
replaced by its 120-chunk re-run (local arms: outcome_audit_to2x_local.json; Fir arms: outcome_audit_to2x_fir.json), the
h2h4 statistics recomputed (eval/h2h4_analysis.py: P1 lfci000 vs psel, P2 lfci000 vs policy, secondary incl. lfc000) next to
the registered 60-chunk results, how the time-outs resolve per arm, and the reproduction check (the re-run's decision
states over the first 60 chunks vs the original run's; local multi-shard runs are not bit-identical, so a re-run can diverge
and be decided within 60 chunks: 'decided_within_60_chunks' separates that re-roll from the effect of the extra time). policy_fir (cross-machine reference) keeps its 60-chunk outcomes.
If the Fir audit is missing, psel / LSF keep their 60-chunk outcomes and the result is marked partial.
usage: python3 eval/to2x_analysis.py  -> results/jenga/to2x_result.json"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval")]
import h2h4_analysis as H                                  # loads the 60-chunk rows (H.V) and writes lsf_h2h4_result.json again (unchanged)

RUNS = ROOT / "results/program_b/policy_rollouts"
TO2X = {"policy": ("local", "record", "paired_record_h2h4"), "lfci000": ("local", "lfci000", "paired_lfci000_h2h4"),
        "lfc000": ("local", "lfc000", "paired_lfc000_h2h4"), "psel": ("fir", "cbfps3m", "paired_cbfps3m_h2h3"),
        "LSF": ("fir", "lsfa030", "paired_lsfa030_h2h3")}


def kind(r):
    if r["logged_neighbour_toppled"]:
        return "topple"
    if r["red_fell_before_extraction"]:
        return "red_fall"
    return "v4_success" if r["v4_success"] else ("timeout" if not r["extracted"] else "extracted_not_v4")


def repro(orig, new, ep):
    """max |state difference| over the decisions of the first 60 chunks present in both runs"""
    with np.load(RUNS / orig / f"ep{ep}.npz") as a, np.load(RUNS / new / f"ep{ep}.npz") as b:
        ka, kb = list(a["chunk_index"]), list(b["chunk_index"])
        common = [k for k in ka if k in kb and k < 60]
        d = [float(np.abs(a["state"][ka.index(k)] - b["state"][kb.index(k)]).max()) for k in common]
    return len(common), max(d) if d else 0.0


def main():
    A = {m: json.loads((ROOT / f"results/jenga/outcome_audit_to2x_{m}.json").read_text())
         for m in ("local", "fir") if (ROOT / f"results/jenga/outcome_audit_to2x_{m}.json").exists()}
    partial = "fir" not in A
    res = {"resolution": {}, "reproduction": {}}
    for arm, (m, run, orig) in TO2X.items():
        eps = json.loads((ROOT / f"results/jenga/to2x/eps_{run}.json").read_text())
        if m not in A:
            res["resolution"][arm] = "not yet re-run (Fir audit missing): 60-chunk outcomes kept"
            continue
        rows = {r["ep"]: r for r in A[m][f"paired_{run}_to2x"]["rows"]}
        miss = [e for e in eps if e not in rows]
        assert not miss, (arm, miss[:5])
        for e in eps:
            H.V[arm][e] = rows[e]                                 # the time-out episode's 120-chunk outcome
        ks = [kind(rows[e]) for e in eps]
        ch = {e: json.loads((RUNS / f"paired_{run}_to2x" / f"ep{e}.json").read_text())["chunks"] for e in eps}
        early = [kind(rows[e]) for e in eps if ch[e] <= 60 and kind(rows[e]) != "timeout"]      # decided within the original budget
        res["resolution"][arm] = {"timeouts_rerun": len(eps), **{q: ks.count(q) for q in ("v4_success", "topple", "red_fall", "timeout", "extracted_not_v4")},
                                  "decided_within_60_chunks": {q: early.count(q) for q in ("v4_success", "topple", "red_fall")}}
        rp = [repro(orig, f"paired_{run}_to2x", e) for e in eps]
        res["reproduction"][arm] = {"episodes": len(rp), "identical_first_60": sum(d == 0.0 for _, d in rp),
                                    "max_state_diff": round(max(d for _, d in rp), 6), "decisions_compared": sum(n for n, _ in rp)}
    out = {"stage": "to2x", "partial": partial, "episodes": len(H.EPS),
           "counts_120": {a: {k: int(H.vec(a, k).sum()) for k in H.KEYS} for a in H.SRC},
           "counts_60": json.loads((ROOT / "results/jenga/lsf_h2h4_result.json").read_text())["counts"],
           "PRIMARY_P1_lfci000_vs_psel": H.noninf("lfci000", "psel"),
           "PRIMARY_P2_lfci000_vs_policy": H.vs_policy("lfci000"),
           "secondary": {"lfci000_vs_LSF": H.noninf("lfci000", "LSF"), "lfc000_vs_psel": H.noninf("lfc000", "psel"),
                         "lfc000_vs_LSF": H.noninf("lfc000", "LSF"), "lfc000_vs_policy": H.vs_policy("lfc000"),
                         "lfci000_vs_lfc000": {k: H.paired("lfci000", "lfc000", k) for k in H.KEYS}},
           **res, "note": "sensitivity analysis; the registered 60-chunk head-to-head 4 verdicts stand as registered"}
    (ROOT / "results/jenga/to2x_result.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("partial", "counts_120", "counts_60", "resolution", "reproduction")}, indent=1))
    for name in ("PRIMARY_P1_lfci000_vs_psel", "PRIMARY_P2_lfci000_vs_policy"):
        print(name, {k: v for k, v in out[name].items() if k in ("NONINF", "PASS")})


if __name__ == "__main__":
    main()
