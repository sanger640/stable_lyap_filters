"""Re-analysis (descriptive) of the frozen test-set runs against ONE physical answer key.

A state is a FORK if 2-62 of the 64 oracle (truth-A) rollouts topple a neighbouring block (the protocols'
"harm fork"). The ceiling compares oracle run A's fork labels with oracle run B's. Every monitor's
decision is recomputed with its frozen rule from stored rollouts (no new simulation, no tuning); the
world models' set-C decisions are reproduced with the evaluation's own seeds and checked against the
stored totals. Original results (graded against split-test fragility truth) are unchanged on disk.

Test set A = fresh panel 2 (visual pipeline v1), B = panel 3, C = panel 4.
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
RESULT = ROOT / "results/jenga/fork_label_reanalysis.json"


def fork(traces, start_tilt):
    from fresh_fragility_panel import topple_count
    return 2 <= topple_count(traces, start_tilt) <= 62


def grade(decisions, label):
    import fragility_grading_dev as fg
    d, y = np.asarray(decisions, bool), np.asarray(label, bool)
    return {"kappa": fg.kappa(d, y), "recall": float((d & y).sum() / max(y.sum(), 1)),
            "false_alarm": float((d & ~y).sum() / max((~y).sum(), 1)), "alarms": int(d.sum())}


def state_rule(traces):
    from monitor_v1_eval import NON_ROBOT, decide, trace_endings
    return bool(decide(*trace_endings(traces, NON_ROBOT["jenga"]))[0])


def set_a():
    import visual_pipeline_panel2 as p2
    panel, screen = p2.panel2_selection()
    keys = sorted(panel, key=lambda k: (int(k[0]), k[1]))
    rows = {k: [] for k in ("fork_A", "fork_B", "naive", "probe", "probe_5hyp", "probe_images")}
    for e, i in keys:
        tilt = np.array(screen[e][str(i)]["start_tilt"])
        with np.load(p2.OUT / f"ep{e}_s{i}.npz") as f, \
                np.load(ROOT / f"results/program_b/compression/panel2_tokens/ep{e}_s{i}.npz") as t:
            rows["fork_A"].append(fork(f["truth_A"], tilt)); rows["fork_B"].append(fork(f["truth_B"], tilt))
            rows["naive"].append(state_rule(f["naive"])); rows["probe"].append(state_rule(f["aware"]))
            rows["probe_5hyp"].append(sum(state_rule(f[f"v2h{m}"]) for m in range(5)) >= 3)
            rows["probe_images"].append(bool(t["aware_full"]))
    return keys, rows


def set_b():
    import visual_pipeline_panel3 as p3
    from jenga_runtime import DEFAULT_CHECKPOINT, load_world_model
    panel, screen, _ = p3.screen_panel(1)
    keys = sorted(panel, key=lambda k: (int(k[0]), k[1]))
    encoder = load_world_model(DEFAULT_CHECKPOINT, "cuda").encoder
    rows = {k: [] for k in ("fork_A", "fork_B", "naive", "probe", "probe_5hyp", "probe_images")}
    for e, i in keys:
        tilt = np.array(screen[e][str(i)]["start_tilt"])
        with np.load(p3.OUT / f"ep{e}_s{i}.npz") as f:
            rows["fork_A"].append(fork(f["truth_A"], tilt)); rows["fork_B"].append(fork(f["truth_B"], tilt))
            rows["naive"].append(state_rule(f["naive"])); rows["probe"].append(state_rule(f["aware"]))
            rows["probe_5hyp"].append(sum(state_rule(f[f"v2h{m}"]) for m in range(5)) >= 3)
        rows["probe_images"].append(p3.token_decisions(p3.OUT / f"ep{e}_s{i}_frames.npz", encoder)["aware"])
    return keys, rows


def set_c():
    import torch
    import uncertainty_monitor_dev as um
    import visual_v2_panel4 as v4
    import step3_world_model as wm
    from counterfactual_monitor import persistent_mode_alarm
    from step3_data import Latent
    panel, screen, _ = v4.screen_panel(1)
    keys = sorted(panel, key=lambda k: (int(k[0]), k[1]))
    snippets = np.load(um.BENCH)["snippets"]
    latent = Latent()
    models = {"wm_stochastic": wm.WorldModel.load(wm.DATA / "wm_stochastic.pt"),
              "wm_deterministic": wm.WorldModel.load(wm.DATA / "wm_deterministic.pt")}
    names = ("fork_A", "fork_B", "naive", "probe", "probe_5hyp", "probe_images", "probe_5hyp_images",
             "wm_stochastic", "wm_deterministic")
    rows = {k: [] for k in names}
    for n, (e, i) in enumerate(keys):
        tilt = np.array(screen[e][str(i)]["start_tilt"])
        with np.load(v4.OUT / f"ep{e}_s{i}.npz") as f:
            rows["fork_A"].append(fork(f["truth_A"], tilt)); rows["fork_B"].append(fork(f["truth_B"], tilt))
            rows["naive"].append(state_rule(f["naive"])); rows["probe"].append(state_rule(f["aware"]))
            rows["probe_5hyp"].append(sum(state_rule(f[f"v2h{m}"]) for m in range(5)) >= 3)
            rows["probe_images"].append(bool(f["tok_aware"]))
            rows["probe_5hyp_images"].append(sum(bool(f[f"tok_v2h{m}"]) for m in range(5)) >= 3)
            z0 = latent(f["close_frame"][None])[0].astype(np.float32)        # as step3_world_model.evaluate
            actions = wm.panel_actions(f["chunk"], snippets)
            for name, model in models.items():
                early, late = wm.wm_rollouts(model, z0, actions, seed=20261047 + n)
                rows[name].append(bool(persistent_mode_alarm(early, late).alarm))
        torch.cuda.empty_cache()
    stored = json.loads((ROOT / "results/jenga/step3_token_world_model_result.json").read_text())
    for name, key in (("wm_stochastic", "stochastic"), ("wm_deterministic", "deterministic")):
        if sum(rows[name]) != stored["graded_vs_truth_A"][key]["alarms"]:
            raise SystemExit(f"{name}: reproduced {sum(rows[name])} alarms, stored "
                             f"{stored['graded_vs_truth_A'][key]['alarms']}")
    return keys, rows


def summarise(keys, rows):
    label = rows["fork_A"]
    out = {"states": len(keys), "forks_A": int(np.sum(label)), "forks_B": int(np.sum(rows["fork_B"])),
           "ceiling_B_vs_A": grade(rows["fork_B"], label)}
    for name in rows:
        if not name.startswith("fork_"):
            out[name] = grade(rows[name], label)
    return out


def main():
    result = {"note": __doc__.strip().splitlines()[0],
              "fork_definition": "2-62 of 64 oracle (truth-A) rollouts topple a neighbouring block (tilt >= 45 deg)",
              "set_A_panel2": summarise(*set_a()), "set_B_panel3": summarise(*set_b()),
              "set_C_panel4": summarise(*set_c())}
    RESULT.write_text(json.dumps(result, indent=1) + "\n")
    for s in ("set_A_panel2", "set_B_panel3", "set_C_panel4"):
        r = result[s]
        print(s, r["states"], "forks", r["forks_A"], {k: round(v["kappa"], 3) for k, v in r.items()
                                                       if isinstance(v, dict) and "kappa" in v})


if __name__ == "__main__":
    main()
