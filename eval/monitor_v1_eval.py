"""Monitor v1 (persistent two-mode split) against frozen Regime Monitor v0.

Frozen protocol: results/jenga/monitor_v1_protocol.json. v1 is
``counterfactual_monitor.persistent_mode_alarm`` UNCHANGED, applied to the positions (mm) of every
non-robot rigid body at hold steps 10 and 30 of the same 64 probes v0 used. Labels grade only.

E1 Jenga DEV and E2 robustness on R's traces run first; E3 (Jenga TEST, upright-push TEST, U1)
runs once, only if gates 1 and 2 pass.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from counterfactual_monitor import persistent_mode_alarm  # noqa: E402

PROTOCOL = ROOT / "results/jenga/monitor_v1_protocol.json"
OUTPUT = ROOT / "results/jenga/monitor_v1_result.json"
BENCH = ROOT / "results/jenga/bench/jenga_bench.npz"
CACHES = {"dev": ("results/jenga/holdout_stage0_cache.npz", "results/jenga/holdout_stage0.json"),
          "test": ("results/jenga/holdout3_stage0_cache.npz", "results/jenga/holdout3_stage0.json")}
CACHE_HOLD_INDEX = {10: 1, 30: 4}           # recorded hold steps (5, 10, 20, 29, 30)
TRACE_INDEX = {10: 17, 30: 37}              # 38-step traces: index 7 + hold step
SCALE_1X = 1
V0 = {"dev": (18, 23, 7, 89), "jenga_test": (55, 84, 4, 113), "push_test": (14, 20, 0, 15),
      "u1_pushing": (0, 12, 3, 108), "u1_insertion": (0, 11, 0, 109)}
# "Every non-robot rigid body": slot position dims per benchmark layout (metres).
NON_ROBOT = {"jenga": slice(0, 9), "push": slice(0, 9), "u1_pushing": slice(3, 9),
             "u1_insertion": slice(0, 9)}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n; den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(centre - half), float(centre + half)]


def decide(early_mm, late_mm):
    """v1 on one state: (alarm, minority at hold 30, Ashman D at hold 30)."""
    result = persistent_mode_alarm(early_mm, late_mm)
    d = result.late["ashman_d"]
    return bool(result.alarm), int(result.late["minority"]), float("inf") if d is None else float(d)


def trace_endings(traces, dims):
    """(probes, 38, features) metres -> (early, late) positions in mm."""
    return (traces[:, TRACE_INDEX[10], dims] * 1000.0, traces[:, TRACE_INDEX[30], dims] * 1000.0)


def jenga_split(split):
    bench = np.load(BENCH)
    cache_path, rows_path = CACHES[split]
    cache = np.load(ROOT / cache_path)
    rows = json.loads((ROOT / rows_path).read_text())["rows"]
    index = {(r["episode_id"], int(r["chunk_start"])): i for i, r in enumerate(rows)}
    idx = [index[(str(e), int(c))] for e, c in zip(bench[f"{split}_episode"], bench[f"{split}_chunk"])]
    pose = cache["pose"][idx][:, SCALE_1X]              # (states, 64, 5 times, 27)
    alignment = float(np.abs(cache["pose"][idx][:, :, :, CACHE_HOLD_INDEX[30], :9] * 1000.0
                             - bench[f"{split}_real_mm"]).max())
    if alignment != 0.0:
        raise SystemExit(f"{split} endings do not match the frozen bench oracle endings: {alignment}")
    classes = bench[f"{split}_classes"][:, SCALE_1X]
    decisions = [decide(p[:, CACHE_HOLD_INDEX[10], :9] * 1000.0, p[:, CACHE_HOLD_INDEX[30], :9] * 1000.0)
                 for p in pose]
    return classes, decisions, alignment


def summarise(classes, decisions, positive, negative):
    alarms = np.array([d[0] for d in decisions])
    classes = np.asarray(classes)
    out = {}
    for name in sorted(set(classes.tolist())):
        mask = classes == name
        k, n = int(alarms[mask].sum()), int(mask.sum())
        out[name] = {"states": n, "alarms": k, "rate": k / n if n else None, "wilson": wilson(k, n)}
    out["_recall"] = out[positive]["rate"]
    out["_quiet_rate"] = out[negative]["rate"]
    out["_positive_minorities"] = [d[1] for d, c in zip(decisions, classes) if c == positive]
    return out


def robustness():
    import program_b_reference as pr
    data = np.load(pr.NEIGHBOURHOODS, allow_pickle=False)
    stored = {}
    for path in Path(pr.OUTPUT_DIR).glob("ep*_seed*.npz"):
        with np.load(path, allow_pickle=False) as f:
            stored[path.stem] = {k: f[k] for k in f.files if k.startswith("n")}
    axes = {"episode": lambda e, s: str(e) in pr.EPISODE_AXIS,
            "configuration": lambda e, s: int(s) in pr.CONFIG_SEEDS}
    names = [c[0] for c in pr.conditions()]
    out = {}
    for axis, keep in axes.items():
        rows = [r for r, (e, s) in enumerate(zip(data["episode_id"], data["seed"])) if keep(e, s)]
        source = lambda r: f"ep{data['episode_id'][r]}_seed{int(data['seed'][r])}"
        decisions = {name: np.array([decide(*trace_endings(stored[source(r)][f"n{r}_{name}"],
                                                           NON_ROBOT["jenga"]))[0] for r in rows])
                     for name in names}
        recorded = np.array([decide(*trace_endings(data["traces"][r], NON_ROBOT["jenga"]))[0]
                             for r in rows])
        if not np.array_equal(recorded, decisions["sigma_0"]):
            raise SystemExit(f"{axis}: sigma_0 decisions differ from the recorded traces")
        base = decisions["sigma_0"]
        row = {"states": len(rows), "sigma0_alarms": int(base.sum())}
        for name in names[1:]:
            d = decisions[name]
            row[name] = {
                "retention": float((d & base).sum() / base.sum()) if base.sum() else None,
                "added": float((d & ~base).sum() / (~base).sum()) if (~base).sum() else None,
                "alarms": int(d.sum())}
        cousins = np.array([decisions[n] for n in ["sigma_2.65"] + [f"cousin_{k}" for k in range(1, 5)]])
        row["cousin_disagreement"] = float(np.mean(cousins.min(0) != cousins.max(0)))
        out[axis] = row
    return out


def push_test():
    panel = json.loads((ROOT / "results/panda_block_push/fresh_test/panel.json").read_text())["rows"]
    traces = np.load(ROOT / "results/panda_block_push/fresh_test/panel_cache.npz")["traces"]
    classes = [r["cohort"] for r in panel]
    decisions = [decide(*trace_endings(traces[r["panel_index"]], NON_ROBOT["push"])) for r in panel]
    return summarise(classes, decisions, "topple_fork", "safe_centered")


def u1_test():
    detector = json.loads((ROOT / "results/contact_benchmarks/u1/ground_truth_detector.json").read_text())
    traces = np.load(ROOT / "results/contact_benchmarks/u1/mechanism_panel.npz")["traces"]
    out = {}
    for task in ("pushing", "insertion"):
        rows = [r for r in detector["rows"] if r["task"] == task and r["split"] == "test"]
        decisions = [decide(*trace_endings(traces[r["index"]], NON_ROBOT[f"u1_{task}"])) for r in rows]
        out[task] = summarise([r["cohort"] for r in rows], decisions, "consequential_fork",
                              "unanimous_quiet")
    return out


def main():
    protocol = json.loads(PROTOCOL.read_text())
    if not protocol["status"].startswith("frozen"):
        raise SystemExit("protocol not frozen")
    classes, decisions, dev_alignment = jenga_split("dev")
    e1 = summarise(classes, decisions, "topple_fork", "quiet")
    gate1 = bool(e1["_recall"] >= 0.73 and e1["_quiet_rate"] <= 0.10)
    e2 = robustness()
    gate2 = all(e2[a]["sigma_2.65"]["retention"] is not None and e2[a]["sigma_2.65"]["retention"] >= 0.70
                and e2[a]["sigma_2.65"]["added"] <= 0.10 for a in ("episode", "configuration"))
    result = {
        "protocol_sha256": sha256(PROTOCOL),
        "code_sha256": {name: sha256(ROOT / name) for name in
                        ("src/counterfactual_monitor.py", "src/outcome_modes.py",
                         "eval/monitor_v1_eval.py")},
        "data_alignment": {"dev_max_abs_mm_vs_bench": dev_alignment},
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "E1_dev": e1, "gate_1": gate1, "E2_robustness": e2, "gate_2": gate2,
        "v0_reference": V0,
    }
    if gate1 and gate2:
        test_classes, test_decisions, test_alignment = jenga_split("test")
        result["data_alignment"]["test_max_abs_mm_vs_bench"] = test_alignment
        jenga = summarise(test_classes, test_decisions, "topple_fork", "quiet")
        push = push_test()
        u1 = u1_test()
        gate3 = bool(jenga["_recall"] >= 0.55 and jenga["_quiet_rate"] <= 0.05
                     and push["_recall"] >= 0.60 and push["safe_centered"]["alarms"] <= 2)
        result.update({"E3_test": {"jenga": jenga, "upright_push": push, "u1": u1},
                       "gate_3": gate3})
    else:
        result["E3_test"] = "not run: an earlier gate failed"
        gate3 = False
    result["adopted"] = bool(gate1 and gate2 and gate3)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    brief = {"E1": {k: e1[k] for k in ("topple_fork", "quiet", "nudge_fork", "small_split") if k in e1},
             "gate_1": gate1, "E2": e2, "gate_2": gate2, "adopted": result["adopted"]}
    if isinstance(result["E3_test"], dict):
        brief["E3"] = {"jenga": {k: result["E3_test"]["jenga"][k] for k in ("topple_fork", "quiet")},
                       "push": {k: result["E3_test"]["upright_push"][k]
                                for k in ("topple_fork", "safe_centered")},
                       "u1": {t: {k: v for k, v in r.items() if not k.startswith("_")}
                              for t, r in result["E3_test"]["u1"].items()},
                       "gate_3": result["gate_3"]}
    print(json.dumps(brief, indent=1))


if __name__ == "__main__":
    main()
