"""GFX1 selection and analysis as pre-registered (results/jenga/gfx1_protocol.json), written before any GFX1 number exists.
  select: part S. Per value (nm = value_irrnm8_alea.pt, ff = value_nsif16_alea.pt, fq = value_nsiq16_alea.pt), its barrier
          among gamma 0.5 / 0.2 / 0.1 on dev 15000-15099 by eval/irrm_select.py's rule (fewest 2x topples among arms with
          v4 >= 80 and topples <= 5 at 1x; ties -> fewer 2x time-outs); one F arm by the same rule over all six F arms.
          -> results/jenga/gfx1_select.json; prints shell lines F_TAG / F_VALUE / F_GAMMA / NM_GAMMA (empty = not run)
  test:   part T on H2H5's 26200-26799 vs H2H5's ours90 / policy / UNISafe and GFCL1's nm90 / ns90; statistics as
          eval/h2h5_analysis.py (paired by episode, normal approximation, one-sided 95% bounds, exact McNemar).
          -> results/jenga/gfx1_result.json
usage: python3 eval/gfx1_analysis.py select|test"""
import json
from pathlib import Path
import sys

import numpy as np

import h2h5_analysis as H
import irrm_select as S

ROOT = Path(__file__).resolve().parent.parent
J = ROOT / "results/jenga"
VALUE = {"nm": "value_irrnm8_alea.pt", "ff": "value_nsif16_alea.pt", "fq": "value_nsiq16_alea.pt"}
GAMMA = {"50": 0.5, "20": 0.2, "10": 0.1}


def table(arms):
    tab = {}
    for a in arms:
        c2, c1 = S.counts(f"{a}dev2x"), S.counts(f"{a}dev1x")
        if c2 and c1:
            tab[a] = {"2x": c2, "1x": c1, "no_cost_1x": c1["v4"] >= 80 and c1["topples"] <= 5}
    return tab


def pick(tab):
    ok = [a for a in tab if tab[a]["no_cost_1x"]]
    return min(ok, key=lambda a: (tab[a]["2x"]["topples"], tab[a]["2x"]["timeouts"])) if ok else None


def select():
    out = {"reference_current_value": table(["i0", "g20", "g10"]), "values": {}}
    for v in VALUE:
        tab = table([f"{v}{g}" for g in GAMMA]); out["values"][v] = {"arms": tab, "selected": pick(tab)}
    ftab = {**out["values"]["ff"]["arms"], **out["values"]["fq"]["arms"]}
    f = pick(ftab); nm = out["values"]["nm"]["selected"]
    out["F_selected"] = f; out["nm_selected"] = nm
    (J / "gfx1_select.json").write_text(json.dumps(out, indent=1))
    print(f"F_TAG={f or ''}"); print(f"F_VALUE={VALUE[f[:2]] if f else ''}"); print(f"F_GAMMA={GAMMA[f[2:]] if f else ''}")
    print(f"NM_GAMMA={GAMMA[nm[2:]] if nm and nm[2:] != '10' else ''}")       # nm at 0.1 = GFCL1's nm90 (not re-run)


def load(nz, extra):
    A = json.loads((J / f"outcome_audit_h5_{nz}.json").read_text())
    for t in ("nm", "ns") + extra:
        A.update(json.loads((J / f"outcome_audit_h5{t}_{nz}.json").read_text()))
    V = {a: {r["ep"]: r for r in A[run.format(n=nz)]["rows"]} for a, run in H.ARMS.items()}
    return V, sorted(set.intersection(*(set(v) for v in V.values())))


def test():
    sel = json.loads((J / "gfx1_select.json").read_text())
    nm_rerun = bool(sel["nm_selected"]) and not sel["nm_selected"].endswith("10")
    H.ARMS = {"policy": "paired_record_h5_{n}", "unisafe": "paired_lsfa030_h5_{n}", "ours90": "paired_lfc000_h5o90_{n}",
              "nm90": "paired_lfc000_h5nm90_{n}", "ns90": "paired_lfc000_h5ns90_{n}", "fx": "paired_lfc000_h5fx_{n}"}
    if nm_rerun:
        H.ARMS["nmx"] = "paired_lfc000_h5nmx_{n}"
    extra = ("fx", "nmx") if nm_rerun else ("fx",)
    res, P = {"selection": {"F": sel["F_selected"], "nm": sel["nm_selected"], "nm_rerun": nm_rerun}}, {}
    for nz in ("2x", "1x"):
        V, eps = load(nz, extra)
        r = {"episodes": len(eps), "counts": {a: {k: int(H.vec(V, eps, a, k).sum()) for k in H.KEYS} for a in H.ARMS}, "edits_per_episode": {}}
        for a in H.ARMS:
            if a not in ("policy", "unisafe"):
                r["edits_per_episode"][a] = round(float(np.mean([sum(c.get("lsf_overrides", 0) or 0 for c in json.loads(
                    (H.RUNS / H.ARMS[a].format(n=nz) / f"ep{e}.json").read_text())["log"]) for e in eps])), 2)
        new = ["fx"] + (["nmx"] if nm_rerun else [])
        r["pairs"] = {f"{a}_vs_{b}": {k: H.paired(V, eps, a, b, k) for k in H.KEYS}
                      for a in new for b in ("ours90", "policy", "unisafe", "ns90", "nm90")}
        res[nz] = r; P[nz] = r["pairs"]
    p = lambda nz, a, b, k: P[nz][f"{a}_vs_{b}"][k]
    noninf = lambda nz, a: bool(p(nz, a, "ours90", "topple")["upper95_pp"] < 3 and p(nz, a, "ours90", "v4_success")["lower95_pp"] > -5)
    hyp = {"T_F_2x": noninf("2x", "fx"), "T_F_1x": noninf("1x", "fx"),
           "T_F_policy": bool(p("2x", "fx", "policy", "topple")["upper95_pp"] < 0)}
    hyp["T_F_better"] = bool(hyp["T_F_2x"] and (p("2x", "fx", "ours90", "topple")["upper95_pp"] < 0 or p("2x", "fx", "ours90", "v4_success")["lower95_pp"] > 0))
    if nm_rerun:
        hyp.update({"T_nm_2x": noninf("2x", "nmx"), "T_nm_1x": noninf("1x", "nmx")})
    hyp["switch_candidate"] = bool(hyp["T_F_2x"] and hyp["T_F_1x"] and hyp["T_F_better"])
    res["hypotheses"] = hyp
    (J / "gfx1_result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"hypotheses": hyp, "selection": res["selection"], "counts_2x": res["2x"]["counts"], "counts_1x": res["1x"]["counts"],
                      "edits_2x": res["2x"]["edits_per_episode"], "edits_1x": res["1x"]["edits_per_episode"],
                      "fx_vs_ours90_2x": P["2x"]["fx_vs_ours90"], "fx_vs_ours90_1x": P["1x"]["fx_vs_ours90"]}, indent=1))


if __name__ == "__main__":
    {"select": select, "test": test}[sys.argv[1]]()
