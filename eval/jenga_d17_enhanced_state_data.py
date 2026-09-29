"""Prospectively replay D3+D12 groups and record richer generic physical state."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "eval"))

from jenga_d6_set_response import sha256
from jenga_state_data import step_state


PROTOCOL = ROOT / "results/jenga/d17_enhanced_state_protocol.json"
OUTPUT = ROOT / "results/jenga/d17_enhanced_state_data.npz"
REPORT = ROOT / "results/jenga/d17_enhanced_state_integrity.json"
JOINT_NAMES = tuple([f"joint{index}" for index in range(1, 8)] +
                    ["finger_joint1", "finger_joint2"])


def contact_category(sim, contact):
    bodies = (int(sim.model.geom_bodyid[contact.geom1]),
              int(sim.model.geom_bodyid[contact.geom2]))
    block_index = {body: index for index, body in enumerate(sim.tracked_block_ids)}
    blocks = [block_index[body] for body in bodies if body in block_index]
    if len(blocks) == 2:
        return {(0, 1): 0, (0, 2): 1, (1, 2): 2}[tuple(sorted(blocks))]
    if len(blocks) != 1:
        return None
    block = blocks[0]; other = bodies[0] if bodies[1] in block_index else bodies[1]
    if other == sim.table_id:
        return 3 + block
    if other == 0:
        return 6 + block
    return 9 + block


def instantaneous_contacts(sim):
    """Twelve generic contact categories x force, penetration, and count features."""
    values = np.zeros((12, 5), np.float64)
    for index, contact in enumerate(sim.data.contact[:sim.data.ncon]):
        category = contact_category(sim, contact)
        if category is None:
            continue
        wrench = np.zeros(6, np.float64)
        sim.mj.mj_contactForce(sim.model, sim.data, index, wrench)
        values[category, 0] += max(float(wrench[0]), 0.)
        values[category, 1] += float(np.linalg.norm(wrench[1:3]))
        values[category, 2] += float(np.linalg.norm(wrench[3:6]))
        values[category, 3] += max(-float(contact.dist), 0.)
        values[category, 4] += 1.
    return values


def robot_proprioception(sim):
    positions, velocities = [], []
    for name in JOINT_NAMES:
        joint = sim.model.joint(name).id
        positions.append(sim.data.qpos[sim.model.jnt_qposadr[joint]])
        velocities.append(sim.data.qvel[sim.model.jnt_dofadr[joint]])
    jacobian = np.zeros((6, sim.model.nv), np.float64)
    sim.mj.mj_jacSite(sim.model, sim.data, jacobian[:3], jacobian[3:], sim.site)
    twist = jacobian @ sim.data.qvel
    return np.asarray(positions + velocities + twist.tolist(), np.float32)


def execute_with_impulse(sim, action):
    """Mirror DirectJengaSim.execute while integrating contact wrench over one control step."""
    action = np.asarray(action, np.float32)
    sim.target = action[:3].astype(float)
    if action[3] > .9:
        sim.gripper = 0.
    elif action[3] < -.9:
        sim.gripper = 110.
    impulse = np.zeros((12, 3), np.float64)
    for _ in range(sim.steps_per_action):
        sim._control(); sim.mj.mj_step(sim.model, sim.data)
        impulse += instantaneous_contacts(sim)[:, :3] * sim.model.opt.timestep
    return impulse.astype(np.float32)


def collect(protocol):
    from jenga_runtime import DEFAULT_LMDB, JengaReplay
    from jenga_short_held_tails import DirectJengaSim, extract_sim

    manifest = json.loads((ROOT / protocol["sources"]["manifest"]).read_text())
    grouped = defaultdict(list)
    for global_index, row in enumerate(manifest["records"]):
        grouped[(row["dataset"], row["file"])].append((global_index, row))
    count = len(manifest["records"]); base = [None] * count; proprio = [None] * count
    instant = [None] * count; recent = [None] * count
    source = [None] * count; source_row = np.zeros(count, np.int64); dataset = [None] * count
    max_error = np.zeros(count); contact_exact = np.zeros(count, bool)
    replay = JengaReplay(DEFAULT_LMDB)
    with tempfile.TemporaryDirectory(prefix="jenga_d17_") as temporary:
        xml = str(extract_sim(ROOT / protocol["sources"]["simulator"], temporary))
        sim = DirectJengaSim(xml)
        try:
            for file_number, ((dataset_name, filename), records) in enumerate(
                    sorted(grouped.items()), 1):
                directory = ROOT / protocol["sources"][f"{dataset_name}_directory"]
                with np.load(directory / filename, allow_pickle=False) as nested, np.load(
                        ROOT / protocol["sources"]["trace_directory"] / filename,
                        allow_pickle=False) as trace:
                    episode_id, reset_seed = str(trace["episode_id"]), int(trace["seed"])
                    episode_actions = np.asarray(replay.episode(episode_id).actions, np.float32)
                    needed = {int(trace["starts"][int(row["state_index"])]) for _, row in records}
                    clean = {}; sim.reset(reset_seed)
                    for step, action in enumerate(episode_actions):
                        if step in needed:
                            clean[step] = (sim.snapshot(), sim.data.qacc_warmstart.copy())
                            if len(clean) == len(needed):
                                break
                        sim.execute(action)
                    for global_index, row in records:
                        state_index = int(row["state_index"]); probe_index = int(row["probe_index"])
                        contact_step = int(row["contact_step"]); start_step = int(trace["starts"][state_index])
                        snapshot, warmstart = clean[start_step]
                        sim.restore(snapshot); sim.data.qacc_warmstart[:] = warmstart
                        controls = trace["actions"][state_index, probe_index, 2:2 + contact_step]
                        for action in controls[:-1]:
                            sim.execute(action)
                        recent[global_index] = execute_with_impulse(sim, controls[-1])
                        observed = step_state(sim); expected = nested["start"][int(row["index"])]
                        difference = observed - expected
                        base[global_index] = observed; proprio[global_index] = robot_proprioception(sim)
                        instant[global_index] = instantaneous_contacts(sim).astype(np.float32)
                        max_error[global_index] = float(np.max(np.abs(difference)))
                        contact_exact[global_index] = np.array_equal(observed[45:57], expected[45:57])
                        source[global_index] = Path(filename).stem
                        source_row[global_index] = int(row["index"]); dataset[global_index] = dataset_name
                if file_number % 20 == 0 or file_number == len(grouped):
                    done = sum(value is not None for value in base)
                    print(f"D17 files {file_number}/{len(grouped)}, groups {done}/{count}", flush=True)
        finally:
            sim.close(); replay.close()
    return {"base_state": np.asarray(base, np.float32),
            "proprioception": np.asarray(proprio, np.float32),
            "instant_contact": np.asarray(instant, np.float32),
            "recent_contact_impulse": np.asarray(recent, np.float32),
            "dataset": np.asarray(dataset), "source": np.asarray(source),
            "source_row": source_row, "max_state_error": max_error,
            "contact_exact": contact_exact}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=str(PROTOCOL))
    parser.add_argument("--output", default=str(OUTPUT)); parser.add_argument("--report", default=str(REPORT))
    parser.add_argument("--workers", type=int, default=1); parser.add_argument("--force", action="store_true")
    args = parser.parse_args(); output = Path(args.output); report = Path(args.report)
    if args.workers != 1:
        raise SystemExit("D17 is frozen to one simulator worker")
    if (output.exists() or report.exists()) and not args.force:
        raise SystemExit("refusing to overwrite D17 enhanced-state artifacts")
    protocol_path = Path(args.protocol); protocol = json.loads(protocol_path.read_text()); sources = protocol["sources"]
    checks = (("manifest", "manifest_sha256"), ("trace_directory", "trace_meta_sha256"),
              ("d3_directory", "d3_meta_sha256"), ("d12_directory", "d12_meta_sha256"),
              ("simulator", "simulator_sha256"), ("d16_result", "d16_result_sha256"),
              ("targets", "targets_sha256"))
    for path_key, hash_key in checks:
        path = ROOT / sources[path_key]
        if path.is_dir(): path = path / "meta.json"
        if sha256(path) != sources[hash_key]:
            raise SystemExit(f"frozen input hash differs: {path}")
    arrays = collect(protocol); expected = protocol["collection"]["groups"]
    integrity = {"group_count": len(arrays["base_state"]) == expected,
                 "base_shape": arrays["base_state"].shape == (expected, 61),
                 "proprioception_shape": arrays["proprioception"].shape == (expected, 24),
                 "instant_contact_shape": arrays["instant_contact"].shape == (expected, 12, 5),
                 "recent_impulse_shape": arrays["recent_contact_impulse"].shape == (expected, 12, 3),
                 "finite": all(np.all(np.isfinite(value)) for value in (
                     arrays["base_state"], arrays["proprioception"], arrays["instant_contact"],
                     arrays["recent_contact_impulse"])),
                 "state_exact": float(np.max(arrays["max_state_error"])) == 0.,
                 "contacts_exact": bool(np.all(arrays["contact_exact"]))}
    result = {"protocol": protocol,
              "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "protocol_sha256": sha256(protocol_path), "checks": integrity,
              "passes": all(integrity.values()),
              "maximum_state_error": float(np.max(arrays["max_state_error"])),
              "contact_mismatch_groups": int(np.sum(~arrays["contact_exact"]))}
    if not result["passes"]:
        report.write_text(json.dumps(result, indent=2) + "\n")
        raise SystemExit(f"D17 integrity failed: {integrity}")
    output.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(output, **arrays)
    result.update({"output": str(output.relative_to(ROOT)), "output_sha256": sha256(output)})
    report.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
