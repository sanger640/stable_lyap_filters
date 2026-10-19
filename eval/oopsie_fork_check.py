"""OopsieBench due diligence (pick_egg): do damage outcomes FORK under small execution noise?

From recorded teleop demos (states + actions): at decision points every 20 steps, restore the exact state (simulator
state, solver warm start, gripper's internal command, damage evaluators' velocities), reset egg health to 100, then
replay the next H recorded actions K times: k = 0 exact, k >= 1 with smoothed Gaussian noise on the arm deltas
(dims 0-5). Damage in the window = egg health drop >= DROP. A decision point FORKS if 2..K-2 of the K noisy futures
damage the egg; ALWAYS if all do; NEVER if none.

usage: python dd/fork_check.py <with|without> [noise_xyz]
"""
import json, os, sys
import h5py
import numpy as np

ROOT = "/home/corey/wksp/oopsie/oopsieverse"
sys.path[:0] = [ROOT, os.path.join(ROOT, "oopsiebench")]
from robosuite.controllers import load_composite_controller_config  # noqa: E402
from envs.registry import EnvironmentRegistry  # noqa: E402

H, K, EVERY, DROP = 20, 9, 20, 10.0


def sync(env):
    for obj in env.get_damageable_objects():
        for ev in obj.damage_evaluators:
            if hasattr(ev, "prev_linear_velocities") and hasattr(ev, "_get_part_linear_velocity"):
                for p in ev._get_damageable_part_names():
                    ev.prev_linear_velocities[p] = ev._get_part_linear_velocity(p)


def egg_health(env):
    return min(o.health for o in env.get_damageable_objects() if "egg" in o.name.lower())


def grippers(env):
    return [g for r in env.robots for g in r.gripper.values()] if isinstance(env.robots[0].gripper, dict) else [r.gripper for r in env.robots]


def main():
    split = sys.argv[1]; nxyz = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    c = EnvironmentRegistry.get("pick_egg")
    env = c.damageable_class(robots=c.robot, controller_configs=load_composite_controller_config(robot=c.robot),
                             translucent_robot=False, has_renderer=False, has_offscreen_renderer=True,
                             render_camera=c.camera_name, camera_names=[c.camera_name], camera_widths=64, camera_heights=64,
                             camera_depths=False, ignore_done=True, use_camera_obs=False, control_freq=c.control_freq)
    f = h5py.File(f"{ROOT}/oopsiebench/demos/paper_demos/teleop_data/pick_egg/{split}_live_feedback.hdf5", "r")
    rng = np.random.default_rng(20261019); rows = []
    for dn in sorted(f["data"], key=lambda s: int(s.split("_")[1])):
        g = f["data"][dn]; states = g["states"][:]; acts = g["actions"][1:]
        names = [x.decode() if isinstance(x, bytes) else str(x) for x in g.attrs["health_list_link_names"]]
        egg_cols = [i for i, nm in enumerate(names) if nm.lower().startswith("egg")]
        rec = np.array(g["obs/health"])[:, egg_cols].min(1)                 # recorded egg health per step
        meta = json.loads(g.attrs.get("ep_meta", "{}"))
        env.set_ep_meta(meta) if meta else env.unset_ep_meta()
        env.reset()
        if "model_file" in g.attrs:
            env.reset_from_xml_string(g.attrs["model_file"]); env.sim.reset()
        for t in range(0, len(acts) - H, EVERY):
            env.sim.set_state_from_flattened(states[t]); env.sim.forward()
            ws = env.sim.data.qacc_warmstart.copy()
            gstate = [np.array(gr.current_action, copy=True) if hasattr(gr, "current_action") else None for gr in grippers(env)]
            out = []
            for k in range(K):
                env.sim.set_state_from_flattened(states[t]); env.sim.forward(); env.sim.data.qacc_warmstart[:] = ws
                for gr, gs in zip(grippers(env), gstate):
                    if gs is not None:
                        gr.current_action = gs.copy()
                sync(env)
                for o in env.get_damageable_objects():
                    o.initialize_health()
                a = acts[t:t + H].copy()
                if k > 0:
                    n = rng.normal(size=(H + 2, 6)) * np.r_[[nxyz] * 3, [nxyz / 3] * 3]
                    n = (n[:-2] + n[1:-1] + n[2:]) / 3
                    a[:, :6] = np.clip(a[:, :6] + n, -1, 1)
                hmin = 100.0
                for at in a:
                    env.step(at); hmin = min(hmin, egg_health(env))
                out.append(100.0 - hmin)
            out = np.array(out); dmg = out[1:] >= DROP
            rec_drop = float(rec[t] - rec[t:t + H + 1].min())
            rows.append({"demo": dn, "t": t, "drop_exact": float(out[0]), "drop_recorded": rec_drop, "drops": out[1:].round(1).tolist(),
                         "n_damaged": int(dmg.sum())})
        print(dn, [r["n_damaged"] for r in rows if r["demo"] == dn], flush=True)
    nd = np.array([r["n_damaged"] for r in rows]); M = K - 1
    res = {"split": split, "noise_xyz": nxyz, "decision_points": len(rows), "never": int((nd == 0).sum()),
           "always": int((nd == M).sum()), "fork_2_to_M-2": int(((nd >= 2) & (nd <= M - 2)).sum()),
           "any_split": int(((nd > 0) & (nd < M)).sum())}
    ex = np.array([r["drop_exact"] for r in rows]); rc = np.array([r["drop_recorded"] for r in rows])
    res["exactness_check"] = {"damage_agree_exact_vs_recorded": float(((ex >= DROP) == (rc >= DROP)).mean()),
                              "recorded_damage_points": int((rc >= DROP).sum()), "exact_replay_damage_points": int((ex >= DROP).sum())}
    res["fork_share_of_damage_points"] = res["any_split"] / max(1, res["any_split"] + res["always"])
    json.dump({"summary": res, "rows": rows}, open(f"/home/corey/wksp/oopsie/dd/fork_{split}_{nxyz}.json", "w"))
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
