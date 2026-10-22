"""LiftPegUpright policies (docs/BENCHMARK_DD.md "LiftPegUpright policies"): the shipped PPO checkpoint (per-step) and our
ManiSkill Diffusion Policy (chunked). Both act on the flat state observation of LiftPegUpright-v1 (pd_ee_delta_pose).
Runs in ~/wksp/maniskill_bench/.venv.
usage: python eval/maniskill_policies.py eval ppo|dp [CKPT] [--n 100]
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


class DP:
    """Chunked policy: ManiSkill's state Diffusion Policy; returns the next act_horizon actions for each env."""
    chunked = True

    def __init__(self, ckpt, dev="cuda"):
        import torch
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        from diffusion_policy.conditional_unet1d import ConditionalUnet1D
        w = torch.load(ckpt, map_location=dev, weights_only=False)
        self.obs_h, self.act_h, self.pred_h = 2, 8, 16
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
            a = torch.randn(n, self.pred_h, self.act_dim, device=self.dev)
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("policy"); ap.add_argument("ckpt", nargs="?")
    ap.add_argument("--n", type=int, default=100); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=MAX_STEPS, help="episode length (planner-demo DP: 250)")
    a = ap.parse_args(); MAX_STEPS = a.steps
    pol = PPO() if a.policy == "ppo" else DP(a.ckpt)
    print(a.policy, "success at end / once", rollout(pol, a.n, a.seed))
