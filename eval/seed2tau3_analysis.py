"""SEED2 + TAU3 analysis as pre-registered (results/jenga/seed2tau3_protocol.json), written before any episode ran.
Statistics as eval/h2h5_analysis.py (paired by episode, normal approximation, one-sided 95% bounds, exact McNemar).
usage: python3 eval/seed2tau3_analysis.py -> results/jenga/seed2tau3_result.json"""
import json

from abl2_analysis import J, rows, table


def scale(nz):
    arms = ("s2", "s3", "t105") if nz == "2x" else ("s2", "s3")
    runs = {"policy": f"paired_record_h6_{nz}", **{a: f"paired_lfc000_st3_{a}_{nz}" for a in arms}}
    V = {"policy": rows(f"outcome_audit_h6_{nz}.json", runs["policy"])}
    V.update({a: rows(f"outcome_audit_st3_{nz}.json", runs[a]) for a in arms})
    if nz == "2x":
        runs["o90"] = "paired_lfc000_ab2_o90"; V["o90"] = rows("outcome_audit_ab2.json", runs["o90"])
        runs["t085"] = "paired_lfc000_tau2_t085"; V["t085"] = rows("outcome_audit_tau2.json", runs["t085"])
    else:
        runs["o90"] = "paired_lfc000_ab3_o90_1x"; V["o90"] = rows("outcome_audit_ab3_1x.json", runs["o90"])
    V = {a: {e: r for e, r in v.items() if 26800 <= e < 27100} for a, v in V.items()}
    out, _ = table(V, runs, [(a, "o90") for a in arms] + [(a, "policy") for a in arms])
    c = out["counts"]
    out["seed_range"] = {k: max(c[a][k] for a in ("o90", "s2", "s3")) - min(c[a][k] for a in ("o90", "s2", "s3"))
                         for k in ("topple", "v4_success", "v4b_success", "red_fall", "timeout")}
    return out


if __name__ == "__main__":
    res = {"protocol": "results/jenga/seed2tau3_protocol.json", "2x": scale("2x"), "1x": scale("1x")}
    (J / "seed2tau3_result.json").write_text(json.dumps(res, indent=1))
    for nz in ("2x", "1x"):
        r = res[nz]
        print(nz, json.dumps({k: r[k] for k in ("counts", "logs", "seed_range")}, indent=1))
        for name, p in r["pairs"].items():
            print(nz, name, {k: (v["diff_pp"], v["lower95_pp"], v["upper95_pp"], v["mcnemar_p"]) for k, v in p.items()})
