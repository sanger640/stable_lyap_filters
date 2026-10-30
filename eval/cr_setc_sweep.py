"""Camera-ready: recall vs false-alarm sweep of the set-C object-position probes (Table 1).

Descriptive, from stored traces only (results/program_b/panel4/ep{e}_s{i}.npz; no new rollouts).
Label = the paper's physical fork label (fork_label_reanalysis.py): 2-62 of 64 truth-A rollouts topple
a neighbour. Probes: 'perception ignored' (naive), 'fork probe' (aware), 'fork probe, 5 hypotheses'
(v2h0..v2h4, >= k of 5 split).

The split test (counterfactual_monitor.persistent_mode_alarm over outcome_modes.two_mode_test,
positions dims 0:9 x1000 at trace indices 17 / 37) is re-implemented here with its constants exposed:
  ashman_d  (default 2)   Ashman's D threshold, both times
  min_side  (default 2)   outcome_modes.MIN_SIDE: the split search keeps >= min_side per side and the
                          smaller group must have >= min_side members (one constant in the library)
  bic_margin (default 0)  require bic1 - bic2 > margin (default: two Gaussians beat one)
  k_votes   (default 3)   5-hypothesis vote count
Step 1 checks that the defaults reproduce the library decision on every trace and the paper's
Table 1 set-C row; the sweep runs only if both hold. One knob varies at a time, others at defaults.
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
RESULT = ROOT / "results/jenga/cr_setc_sweep.json"
FIG = ROOT / "paper/submissions/robots_need_wm_camera_ready/fig_sweep"
EARLY, LATE = 17, 37                      # monitor_v1_eval.TRACE_INDEX (hold steps 10, 30)
DIMS = slice(0, 9)                        # monitor_v1_eval.NON_ROBOT["jenga"]
DEFAULTS = {"ashman_d": 2.0, "min_side": 2, "bic_margin": 0.0, "k_votes": 3}
SWEEPS = {"ashman_d": [0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8],
          "min_side": [1, 2, 3, 4, 6, 8, 12, 16],
          "k_votes": [1, 2, 3, 4, 5],
          "bic_margin": [-20, -10, -5, 0, 5, 10, 20, 50, 100, 200]}
# Paper Table 1, set C (results/jenga/fork_label_reanalysis.json): caught, false (%)
PAPER = {"naive": (67, 24), "probe": (91, 10), "probe_5hyp": (89, 11)}
PROBES = {"naive": ["naive"], "probe": ["aware"], "probe_5hyp": [f"v2h{m}" for m in range(5)]}


def _neg2ll(values):                      # outcome_modes._neg2ll
    var = max(float(np.var(values)), 1e-12)
    return len(values) * (np.log(2 * np.pi * var) + 1)


def split_stats(x, min_side):
    """outcome_modes.two_mode_test with MIN_SIDE = min_side; returns (bic1-bic2, D, minority, labels)."""
    x = np.asarray(x, float)
    n = len(x)
    centered = x - x.mean(0)
    if n < 2 * min_side or not np.any(centered):
        return -np.inf, 0.0, 0, np.zeros(n, int)
    axis = np.linalg.svd(centered, full_matrices=False)[2][0]
    y = centered @ axis
    order = np.argsort(y)
    ys = y[order]
    best, split = -1.0, min_side
    for i in range(min_side, n - min_side + 1):
        a, b = ys[:i], ys[i:]
        between = i * (n - i) / n * (a.mean() - b.mean()) ** 2
        if between > best:
            best, split = between, i
    labels = np.zeros(n, int)
    labels[order[split:]] = 1
    a, b = y[labels == 0], y[labels == 1]
    pooled = np.sqrt(a.var() + b.var())
    d = float(np.inf if pooled == 0 else np.sqrt(2) * abs(a.mean() - b.mean()) / pooled)
    sizes = np.array([len(a), len(b)])
    bic1 = _neg2ll(y) + 2 * np.log(n)
    bic2 = _neg2ll(a) + _neg2ll(b) - 2 * np.sum(sizes * np.log(sizes / n)) + 5 * np.log(n)
    return float(bic1 - bic2), d, int(sizes.min()), labels


def same_partition(a, b, tolerance=1):    # outcome_modes.same_partition
    return int(min(np.sum(a != b), np.sum(a != 1 - b))) <= tolerance


def alarm(stats_early, stats_late, ashman_d, min_side, bic_margin, **_):
    """persistent_mode_alarm with exposed constants (stats from split_stats at this min_side)."""
    def supported(s):
        return s[0] > bic_margin and s[1] > ashman_d and s[2] >= min_side
    return bool(supported(stats_early) and supported(stats_late)
                and same_partition(stats_early[3], stats_late[3]))


def load():
    import visual_v2_panel4 as v4
    from fork_label_reanalysis import fork, state_rule
    panel, screen, _ = v4.screen_panel(1)
    keys = sorted(panel, key=lambda k: (int(k[0]), k[1]))
    label, traces, library = [], {n: [] for n in ("naive", "aware", *PROBES["probe_5hyp"])}, {}
    for e, i in keys:
        tilt = np.array(screen[e][str(i)]["start_tilt"])
        with np.load(v4.OUT / f"ep{e}_s{i}.npz") as f:
            label.append(fork(f["truth_A"], tilt))
            for name in traces:
                t = f[name]
                traces[name].append((t[:, EARLY, DIMS] * 1000.0, t[:, LATE, DIMS] * 1000.0))
                library.setdefault(name, []).append(state_rule(t))   # the paper's own decision
    return keys, np.array(label, bool), traces, {k: np.array(v, bool) for k, v in library.items()}


def grade(dec, y):
    dec = np.asarray(dec, bool)
    tp, fp = int((dec & y).sum()), int((dec & ~y).sum())
    return {"caught": tp, "forks": int(y.sum()), "false": fp, "safe": int((~y).sum()),
            "caught_pct": 100 * tp / y.sum(), "false_pct": 100 * fp / (~y).sum()}


def main():
    keys, y, traces, library = load()
    cache = {}

    def stats(name, min_side):
        if (name, min_side) not in cache:
            cache[(name, min_side)] = [(split_stats(e, min_side), split_stats(l, min_side))
                                       for e, l in traces[name]]
        return cache[(name, min_side)]

    def decisions(name, s):
        return np.array([alarm(a, b, **s) for a, b in stats(name, s["min_side"])])

    def probe(p, s):
        votes = np.sum([decisions(n, s) for n in PROBES[p]], 0)
        return votes >= (s["k_votes"] if p == "probe_5hyp" else 1)

    # Step 1: reproduction (per-trace vs library, then vs paper Table 1)
    per_trace = {n: int(np.sum(decisions(n, DEFAULTS) != library[n])) for n in traces}
    repro = {p: grade(probe(p, DEFAULTS), y) for p in PROBES}
    lib5 = np.sum([library[n] for n in PROBES["probe_5hyp"]], 0) >= 3
    lib = {"naive": library["naive"], "probe": library["aware"], "probe_5hyp": lib5}
    paper_ok = all(round(repro[p]["caught_pct"]) == PAPER[p][0] and round(repro[p]["false_pct"]) == PAPER[p][1]
                   for p in PROBES)
    stored = json.loads((ROOT / "results/jenga/fork_label_reanalysis.json").read_text())["set_C_panel4"]
    stored_ok = (int(y.sum()) == stored["forks_A"] and len(y) == stored["states"]
                 and all(int(lib[p].sum()) == stored[p]["alarms"] for p in PROBES))
    check = {"states": len(y), "forks": int(y.sum()), "safe": int((~y).sum()),
             "trace_decision_mismatches_vs_library": per_trace,
             "local_equals_library_probe_decisions": all(bool(np.array_equal(probe(p, DEFAULTS), lib[p]))
                                                         for p in PROBES),
             "matches_fork_label_reanalysis_json": stored_ok,
             "paper_table1_setC": {p: {"paper": PAPER[p], "reproduced": [round(repro[p]["caught_pct"], 1),
                                                                          round(repro[p]["false_pct"], 1)],
                                       **repro[p]} for p in PROBES},
             "matches_paper": paper_ok}
    print(json.dumps(check, indent=1))
    if not (paper_ok and stored_ok and check["local_equals_library_probe_decisions"]):
        RESULT.write_text(json.dumps({"reproduction": check, "sweep": "not run"}, indent=1) + "\n")
        raise SystemExit("reproduction failed; sweep not run")

    # Step 2: one knob at a time
    sweep = {}
    for knob, values in SWEEPS.items():
        rows = []
        for v in values:
            s = dict(DEFAULTS, **{knob: v})
            probes = ["probe_5hyp"] if knob == "k_votes" else list(PROBES)
            rows.append({"value": v, **{p: grade(probe(p, s), y) for p in probes}})
        sweep[knob] = rows
    # Secondary min-side variant: split search fixed at MIN_SIDE=2, only the minority check varies
    check_only = []
    for m in SWEEPS["min_side"]:
        row = {"value": m}
        for p in PROBES:
            votes = np.sum([[alarm(a, b, DEFAULTS["ashman_d"], m, 0.0) for a, b in stats(n, 2)]
                            for n in PROBES[p]], 0)
            row[p] = grade(votes >= (3 if p == "probe_5hyp" else 1), y)
        check_only.append(row)
    # BIC-margin distribution (defaults; min of the two times) for context
    margins = {n: np.array([min(a[0], b[0]) for a, b in stats(n, 2)]) for n in ("naive", "aware")}
    out = {"note": __doc__.strip().splitlines()[0],
           "source": "results/program_b/panel4/ep*_s*.npz (stored traces), labels via fork_label_reanalysis.fork",
           "defaults": DEFAULTS, "reproduction": check, "sweep": sweep,
           "min_side_check_only": check_only,
           "bic_gain_quantiles_min_over_times": {n: np.percentile(m[np.isfinite(m)], [10, 25, 50, 75, 90]).round(1).tolist()
                                                 for n, m in margins.items()}}
    RESULT.write_text(json.dumps(out, indent=1) + "\n")
    for knob, rows in sweep.items():
        print(knob)
        for r in rows:
            print(f"  {r['value']:>6}", "  ".join(f"{p}: {r[p]['caught_pct']:5.1f}/{r[p]['false_pct']:5.1f}"
                                                for p in PROBES if p in r))
    print("min_side (check only)")
    for r in check_only:
        print(f"  {r['value']:>6}", "  ".join(f"{p}: {r[p]['caught_pct']:5.1f}/{r[p]['false_pct']:5.1f}"
                                            for p in PROBES))
    figure(sweep)


def figure(sweep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif"], "font.size": 8,
                         "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    style = {"probe": ("#0e7a86", "fork probe"), "probe_5hyp": ("#c96a14", "fork probe, 5 hyp."),
             "naive": ("#5d6873", "perception ignored")}
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.0), sharey=True, sharex=True)
    for ax, knob, title in ((axes[0], "ashman_d", "Ashman's D threshold"),
                            (axes[1], "min_side", "min. smaller-group size")):
        rows = sweep[knob]
        for p in ("naive", "probe_5hyp", "probe"):
            c, lab = style[p]
            fa = [r[p]["false_pct"] for r in rows]
            rc = [r[p]["caught_pct"] for r in rows]
            ax.plot(fa, rc, "-o", color=c, ms=2.5, lw=1.1, label=lab)
            d = next(r for r in rows if r["value"] == DEFAULTS[knob])
            ax.plot(d[p]["false_pct"], d[p]["caught_pct"], "*", color=c, ms=8, mec="k", mew=0.4, zorder=5)
        ax.set_title(title, fontsize=8, pad=3)
        ax.set_xlabel("false alarms (%)")
        ax.set_xlim(0, 29)
        ax.grid(alpha=0.25, lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("forks caught (%)")
    axes[0].set_ylim(0, 100)
    axes[1].legend(loc="lower right", frameon=False, handlelength=1.4, borderaxespad=0.2)
    fig.tight_layout(pad=0.3, w_pad=0.8)
    fig.savefig(f"{FIG}.pdf")
    fig.savefig(f"{FIG}.png", dpi=300)


if __name__ == "__main__":
    main()
