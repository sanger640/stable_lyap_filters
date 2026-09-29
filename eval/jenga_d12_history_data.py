"""Reconstruct exact causal histories for the D12 response groups."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "eval"))

from jenga_d6_set_response import sha256
from jenga_d9b_history_data import history_tail


PROTOCOL = ROOT / "results/jenga/d12_ladder_protocol.json"
OUTPUT = ROOT / "results/jenga/d12_history_data.npz"
REPORT = ROOT / "results/jenga/d12_history_alignment_result.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nested", default=str(ROOT / "results/jenga/d12_coverage_data"))
    parser.add_argument("--trace", default=str(ROOT / "results/jenga/trace_data"))
    parser.add_argument("--protocol", default=str(PROTOCOL)); parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--report", default=str(REPORT)); parser.add_argument("--force", action="store_true")
    args = parser.parse_args(); output = Path(args.output); report = Path(args.report)
    if (output.exists() or report.exists()) and not args.force:
        raise SystemExit("refusing to overwrite D12 history output")
    protocol = json.loads(Path(args.protocol).read_text()); config = protocol["history_replay"]
    if sha256(Path(args.nested) / "meta.json") != protocol["data"]["d12_meta_sha256"]:
        raise SystemExit("D12 metadata differs from frozen protocol")

    from jenga_runtime import DEFAULT_LMDB, JengaReplay
    from jenga_short_held_tails import DirectJengaSim, extract_sim
    from jenga_state_data import step_state

    state_histories, action_histories, starts = [], [], []
    sources, source_rows, max_errors, rms_errors, contact_exact = [], [], [], [], []
    paths = sorted(Path(args.nested).glob("ep*_seed*.npz")); replay = JengaReplay(DEFAULT_LMDB)
    with tempfile.TemporaryDirectory(prefix="jenga_d12_history_") as temporary:
        xml = str(extract_sim(ROOT / "vendor/panda_express_sim.tar", temporary)); sim = DirectJengaSim(xml)
        try:
            for number, path in enumerate(paths, 1):
                nested = np.load(path, allow_pickle=False)
                trace = np.load(Path(args.trace) / path.name, allow_pickle=False)
                episode_id, reset_seed = str(trace["episode_id"]), int(trace["seed"])
                episode_actions = np.asarray(replay.episode(episode_id).actions, np.float32)
                needed = {int(trace["starts"][index]) for index in nested["state_index"]}
                clean = {}; sim.reset(reset_seed)
                for step, action in enumerate(episode_actions):
                    if step in needed:
                        clean[step] = (sim.snapshot(), sim.data.qacc_warmstart.copy())
                        if len(clean) == len(needed): break
                    sim.execute(action)
                for row, (state_index, probe_index, contact_step) in enumerate(zip(
                        nested["state_index"], nested["probe_index"], nested["contact_step"])):
                    snapshot, warm = clean[int(trace["starts"][state_index])]
                    sim.restore(snapshot); sim.data.qacc_warmstart[:] = warm
                    states = [step_state(sim)]; controls = trace["actions"][
                        state_index, probe_index, 2:2 + int(contact_step)]
                    for action in controls:
                        sim.execute(action); states.append(step_state(sim))
                    state_history, action_history = history_tail(states, controls, config["states"])
                    expected = nested["start"][row]; difference = state_history[-1] - expected
                    state_histories.append(state_history); action_histories.append(action_history)
                    starts.append(expected); sources.append(path.stem); source_rows.append(row)
                    max_errors.append(float(np.max(np.abs(difference))))
                    rms_errors.append(float(np.sqrt(np.mean(difference ** 2))))
                    contact_exact.append(bool(np.array_equal(state_history[-1, 45:57], expected[45:57])))
                if number % 20 == 0 or number == len(paths):
                    print(f"D12 history files {number}/{len(paths)}, groups {len(starts)}", flush=True)
        finally:
            sim.close(); replay.close()
    arrays = {"state_history": np.asarray(state_histories, np.float32),
              "action_history": np.asarray(action_histories, np.float32),
              "start": np.asarray(starts, np.float32), "source": np.asarray(sources),
              "source_row": np.asarray(source_rows, np.int64)}
    checks = {"group_count": len(starts) == config["d12_groups"],
              "max_abs": max(max_errors) <= config["maximum_absolute_current_state_error"],
              "max_rms": max(rms_errors) <= config["maximum_rms_current_state_error"],
              "contacts": all(contact_exact),
              "last_state_exactly_cached": bool(np.array_equal(arrays["state_history"][:, -1], arrays["start"])),
              "shapes": (arrays["state_history"].shape == (config["d12_groups"], config["states"], 61)
                         and arrays["action_history"].shape == (config["d12_groups"], config["actions"], 4))}
    gate = {"checks": checks, "passes": all(checks.values()),
            "maximum_absolute_error": max(max_errors), "maximum_rms_error": max(rms_errors),
            "contact_mismatch_groups": sum(not value for value in contact_exact)}
    result = {"protocol_sha256": sha256(args.protocol), "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "gate": gate}
    if not gate["passes"]:
        report.write_text(json.dumps(result, indent=2) + "\n"); raise SystemExit(f"history alignment failed: {gate}")
    np.savez_compressed(output, **arrays); result.update({"output": str(output.relative_to(ROOT)),
                                                          "output_sha256": sha256(output)})
    report.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
