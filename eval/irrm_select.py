"""IRRM amendments 3-4 selection rule (results/jenga/irrm_protocol.json): among the irreversibility arms, the fewest 2x
topples among arms with 'no cost' at 1x (v4 >= 80 and topples <= 5); ties -> fewer 2x time-outs. Arms whose 4 audits
(2x and 1x) do not all exist yet are skipped.  usage: python3 eval/irrm_select.py [arm ...] -> prints the table and the selection
arms: i0 (value_irr8, gamma 0.5), g20, g10 (value_irr8, gamma 0.2 / 0.1), j0 / jsel (value_irr16, gamma 0.5 / amendment 3's)"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
GAMMA = {"i0": 0.5, "g20": 0.2, "g10": 0.1}


def counts(tag):
    f = ROOT / f"results/jenga/outcome_audit_{tag}.json"
    if not f.exists():
        return None
    R = json.loads(f.read_text())[f"paired_lfc000_{tag}"]["rows"]
    to = sum(not r["extracted"] and not r["logged_neighbour_toppled"] and not r["red_fell_before_extraction"] for r in R)
    return {"topples": sum(r["logged_neighbour_toppled"] for r in R), "v4": sum(r["v4_success"] for r in R),
            "v4b": sum(r["v4b_success"] for r in R), "timeouts": to, "red_falls": sum(r["red_fell_before_extraction"] for r in R)}


def main(arms):
    tab = {}
    for a in arms:
        c2, c1 = counts(f"{a}dev2x"), counts(f"{a}dev1x")
        if c2 and c1:
            tab[a] = {"2x": c2, "1x": c1, "no_cost_1x": c1["v4"] >= 80 and c1["topples"] <= 5}
    ok = [a for a in tab if tab[a]["no_cost_1x"]]
    sel = min(ok, key=lambda a: (tab[a]["2x"]["topples"], tab[a]["2x"]["timeouts"])) if ok else None
    print(json.dumps({"arms": tab, "selected": sel, "selected_gamma": GAMMA.get(sel)}, indent=1))
    return sel


if __name__ == "__main__":
    main(sys.argv[1:] or ["i0", "g20", "g10"])
