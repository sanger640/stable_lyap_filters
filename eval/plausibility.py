"""The diffusion policy's own plausibility of an action plan (task-agnostic, label-free; uses only the policy being
filtered). error(plan) = the policy's denoising loss (epsilon MSE on normalised actions) at fixed timesteps with fixed
noise draws (common random numbers), conditioned on the decision's 2-frame observation. Lower = more like what the
policy itself would do. Differentiable in the plan (used as a cost by eval/cbf_opt.py).
"""
import numpy as np

TS, N_EPS = (2, 5, 10, 20, 35), 8


class Plausibility:
    def __init__(self, pol):
        import torch
        self.torch, self.pol = torch, pol
        for p_ in pol.parameters():
            p_.requires_grad_(False)
        g = torch.Generator(device="cuda").manual_seed(0)
        self.eps = torch.randn(len(TS) * N_EPS, pol.horizon, pol.action_dim, device="cuda", generator=g)
        self.t = torch.tensor([t for t in TS for _ in range(N_EPS)], device="cuda").long()

    def cond(self, hist):
        """hist: [(wrist, fixed, agent_pos)] x 2 (previous, current) -> global condition (1, D)."""
        torch, pol = self.torch, self.pol
        from diffusion_policy.common.pytorch_util import dict_apply
        w = np.stack([h[0] for h in hist]); f = np.stack([h[1] for h in hist]); s = np.stack([h[2] for h in hist])
        t = lambda x: torch.from_numpy(x).permute(0, 3, 1, 2)[None].float().cuda() / 255.0
        obs = {"camera_1": t(w), "camera_2": t(f), "agent_pos": torch.from_numpy(s)[None].cuda()}
        with torch.no_grad():
            nobs = pol.normalizer.normalize(obs)
            this = dict_apply(nobs, lambda x: x[:, :pol.n_obs_steps].reshape(-1, *x.shape[2:]))
            return pol.obs_encoder(this).reshape(1, -1)

    def error(self, gc, plans):
        """plans (C, 16, 4) torch (differentiable) or numpy -> (C,) torch mean denoising error."""
        torch, pol = self.torch, self.pol
        plans = torch.as_tensor(plans, device="cuda").float()
        na = pol.normalizer["action"].normalize(plans)
        C, K = len(na), len(self.t)
        x0 = na[:, None].expand(C, K, -1, -1).reshape(C * K, *na.shape[1:])
        eps = self.eps[None].expand(C, -1, -1, -1).reshape(C * K, *na.shape[1:]); t = self.t.repeat(C)
        noisy = pol.noise_scheduler.add_noise(x0, eps, t)
        pred = pol.model(noisy, t, local_cond=None, global_cond=gc.expand(C * K, -1))
        return ((pred - eps) ** 2).mean((1, 2)).reshape(C, K).mean(1)


def as_plan(plan16, executed):
    """An executed 8-step chunk as a 16-step plan: steps 1-8 replaced, 9-15 shifted by the step-8 change."""
    p = np.asarray(plan16, np.float32).copy(); e = np.asarray(executed, np.float32)
    shift = e[-1, :3] - p[8, :3]
    p[1:9] = e; p[9:, :3] += shift
    return p


def obs_hist(sim, cams, S, d, noises):
    """Re-render the policy's 2-frame observation at saved decision d of an episode (S = its npz dict): the frame after
    step 7 of the previous chunk (replayed with the run's noise) and the current frame."""
    snap = lambda i: (S["state"][i], S["ctrl"][i], S["target"][i], float(S["gripper"][i]))
    sim.restore(snap(d)); cur = cams.obs()
    if d > 0 and S["chunk_index"][d - 1] == S["chunk_index"][d] - 1:
        sim.restore(snap(d - 1)); k = int(S["chunk_index"][d - 1])
        for j, a in enumerate(S["executed"][d - 1][:7]):
            a = np.asarray(a, np.float32).copy(); a[:3] -= noises[k][j]; sim.execute(a)
        prev = cams.obs()
        sim.restore(snap(d))
    else:
        prev = cur
    return [prev, cur]


def episode_noises(ep, n_chunks, scale=2.0):
    import uncertainty_monitor_dev as um
    snippets = np.load(um.BENCH)["snippets"]; nrng = np.random.default_rng([20261502, ep])
    return [snippets[nrng.integers(len(snippets))] * scale for _ in range(n_chunks)]
