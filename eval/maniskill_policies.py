"""LiftPegUpright policies (docs/BENCHMARK_DD.md "LiftPegUpright policies"): the shipped PPO checkpoint (per-step) and our
ManiSkill Diffusion Policy (chunked). Both act on the flat state observation of LiftPegUpright-v1 (pd_ee_delta_pose).
Runs in ~/wksp/maniskill_bench/.venv.
usage: python eval/maniskill_policies.py eval ppo|dp [CKPT] [--n 100] [--steps 250]
       python eval/maniskill_policies.py video ppo|dp [CKPT] --episodes 3 0 5 --out f.mp4 [--steps 250]
"""
import argparse
import pathlib

import numpy as np

ROOT = pathlib.Path.home() / "wksp/maniskill_bench"
PPO_CKPT = ROOT / "demos/LiftPegUpright-v1/rl/ppo_pd_ee_delta_pose_ckpt.pt"
ENV, MODE, MAX_STEPS = "LiftPegUpright-v1", "pd_ee_delta_pose", 50   # task default; success = upright at the END


def make_env(n, render=False, cam=None):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    kw = {}
    if render:
        from mani_skill.utils import sapien_utils
        e, tg = cam or ([0.0, -0.5, 0.15], [0.0, 0.0, 0.05])
        kw = dict(render_mode="rgb_array", human_render_camera_configs=dict(width=384, height=384, pose=sapien_utils.look_at(e, tg), fov=1.0))
    return gym.make(ENV, num_envs=n, obs_mode="state", control_mode=MODE, sim_backend="physx_cuda", max_episode_steps=MAX_STEPS,
                    reconfiguration_freq=None, **kw)


class PPO:
    """Per-step policy: deterministic mean of the shipped PPO actor."""
    chunked = False

    def __init__(self, dev="cuda"):
        import torch
        import torch.nn as nn
        w = torch.load(PPO_CKPT, map_location=dev, weights_only=True)
        d_in, d_out = w["actor_mean.0.weight"].shape[1], w["actor_mean.6.weight"].shape[0]
        self.net = nn.Sequential(nn.Linear(d_in, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(),
                                 nn.Linear(256, d_out)).to(dev)
        self.net.load_state_dict({k.replace("actor_mean.", ""): v for k, v in w.items() if k.startswith("actor_mean.")})
        self.net.eval()

    def reset(self, n):
        pass

    def act(self, obs):
        import torch
        with torch.no_grad():
            return torch.clamp(self.net(obs), -1, 1)


class PPOChunk(PPO):
    """PPO made chunked by action repeat: query once, the chunk is that action repeated k times (exploratory)."""
    chunked = True

    def __init__(self, k=4, dev="cuda"):
        super().__init__(dev); self.k = k

    def chunk(self, obs):
        return self.act(obs)[:, None].repeat(1, self.k, 1)

    def push(self, obs):
        pass


class DP:
    """Chunked policy: ManiSkill's state Diffusion Policy; returns the next act_horizon actions for each env."""
    chunked = True

    def __init__(self, ckpt, dev="cuda", act_h=8):
        import torch
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        from diffusion_policy.conditional_unet1d import ConditionalUnet1D
        w = torch.load(ckpt, map_location=dev, weights_only=False)
        self.obs_h, self.act_h, self.pred_h = 2, act_h, 16
        sd = w["ema_agent"] if "ema_agent" in w else w["agent"]
        sd = {k.replace("noise_pred_net.", ""): v for k, v in sd.items() if k.startswith("noise_pred_net.")}
        self.act_dim = sd["final_conv.1.weight"].shape[0]
        gdim = sd["down_modules.0.0.cond_encoder.1.weight"].shape[1] - 64
        self.net = ConditionalUnet1D(input_dim=self.act_dim, global_cond_dim=gdim, diffusion_step_embed_dim=64,
                                     down_dims=[64, 128, 256], n_groups=8).to(dev)
        self.net.load_state_dict(sd); self.net.eval()
        self.sched = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2", clip_sample=True, prediction_type="epsilon")
        self.dev, self.hist = dev, None

    def reset(self, n):
        self.hist = None

    def chunk(self, obs):
        """obs (N, d) current state -> (N, act_h, act_dim), using the last obs_h observations."""
        import torch
        if self.hist is None:
            self.hist = obs[:, None].repeat(1, self.obs_h, 1)
        else:
            self.hist = torch.cat([self.hist[:, 1:], obs[:, None]], 1)
        n = len(obs)
        with torch.no_grad():
            g = getattr(self, "group", 1)                                     # copies of one moment share the sampling noise
            a = torch.randn(n // g, self.pred_h, self.act_dim, device=self.dev).repeat_interleave(g, 0)
            self.sched.set_timesteps(100)
            cond = self.hist.flatten(1)
            for k in self.sched.timesteps:
                eps = self.net(sample=a, timestep=k, global_cond=cond)
                a = self.sched.step(model_output=eps, timestep=k, sample=a).prev_sample
        s = self.obs_h - 1
        return a[:, s:s + self.act_h].clamp(-1, 1)

    def push(self, obs):
        """Record an observation seen mid-chunk (keeps the obs history right for the next chunk)."""
        import torch
        self.hist = torch.cat([self.hist[:, 1:], obs[:, None]], 1)


def rollout(policy, n=100, seed=0):
    """Closed-loop success on n parallel GPU episodes: (upright at the end of 50 steps, upright at any step)."""
    import torch
    env = make_env(n); obs, _ = env.reset(seed=seed); policy.reset(n)
    succ = torch.zeros(n, dtype=torch.bool, device=obs.device); t = 0
    while t < MAX_STEPS:
        if policy.chunked:
            ch = policy.chunk(obs)
            for k in range(ch.shape[1]):
                obs, _, _, _, info = env.step(ch[:, k]); succ |= info["success"]; t += 1
                if k < ch.shape[1] - 1:
                    policy.push(obs)
        else:
            obs, _, _, _, info = env.step(policy.act(obs)); succ |= info["success"]; t += 1
    env.close()
    return float(info["success"].float().mean()), float(succ.float().mean())


def video(policy, out, episodes, label, n=32, seed=1, steps=None):
    """Render chosen episodes of a closed-loop batch (same seeding as the fork check) from the world-model camera."""
    import cv2
    import imageio
    import torch
    steps = steps or MAX_STEPS
    env = make_env(n, render=True); obs, _ = env.reset(seed=seed); policy.reset(n); frames = []; t = 0
    while t < steps:
        acts = policy.chunk(obs) if policy.chunked else policy.act(obs)[:, None]
        for k in range(acts.shape[1]):
            obs, _, _, _, info = env.step(acts[:, k]); t += 1
            if policy.chunked and k < acts.shape[1] - 1:
                policy.push(obs)
            frames.append(env.render().cpu().numpy()[list(episodes)])
    ok = info["success"].cpu().numpy()[list(episodes)]
    env.close()
    vid = []
    for i, f in enumerate(frames):
        tiles = []
        for j, e in enumerate(episodes):
            bar = np.full((34, f.shape[2], 3), (40, 150, 60) if ok[j] else (190, 50, 40), np.uint8)
            cv2.putText(bar, f"{label} ep {e}  step {i + 1}/{steps}  {'UPRIGHT AT END' if ok[j] else 'FAILS'}", (8, 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(np.concatenate([bar, f[j]], 0))
        vid.append(np.concatenate(tiles, 1))
    imageio.mimwrite(out, vid + [vid[-1]] * 30, fps=20 if steps > 100 else 10, quality=6, macro_block_size=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("policy"); ap.add_argument("ckpt", nargs="?")
    ap.add_argument("--n", type=int, default=100); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=MAX_STEPS, help="episode length (planner-demo DP: 250)")
    ap.add_argument("--episodes", type=int, nargs="*", default=[0, 1]); ap.add_argument("--out", default="video.mp4")
    ap.add_argument("--act-h", type=int, default=8, help="DP chunk length (must match training act_horizon)")
    a = ap.parse_args(); MAX_STEPS = a.steps
    pol = PPO() if a.policy == "ppo" else PPOChunk(a.act_h) if a.policy == "ppochunk" else DP(a.ckpt, act_h=a.act_h)
    if a.cmd == "video":                                                     # video ppo|dp [CKPT] --episodes 3 0 5 --out f.mp4
        video(pol, a.out, a.episodes, a.policy.upper(), steps=MAX_STEPS)
    else:
        print(a.policy, "success at end / once", rollout(pol, a.n, a.seed))

