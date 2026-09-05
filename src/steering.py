"""
Diffusion steering (section 8.2): classifier-guidance-style correction of the diffusion
policy's OWN denoising sampler using the frozen dino_wm world model's divergence gradient.
Directly implements CLAUDE.md's "active safety filter" roadmap item -- no new world model,
just a guidance hook into the existing DDIM loop (diffusion_unet_image_policy.py's
`conditional_sample`).

Mechanism, standard classifier guidance adapted to a value/energy function instead of a
class label: on the later (cleaner) denoising steps, compute the closed-form predicted clean
sample x0 from the current noisy trajectory and the UNet's epsilon prediction, decode it to
real action units, feed it through the frozen world model for a single (differentiable,
"nominal" -- section 7.3, no perturbation needed) rollout, and take the divergence there as
an energy to descend: nudge the noisy trajectory opposite the gradient of that energy before
continuing the scheduler step. Only guiding the later steps (cleaner x0 estimate) avoids
wasting compute/gradient-noise on the early, still-mostly-random steps where x0's estimate is
unreliable anyway.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DINO_WM_DIR, DINO_WM_CKPT, add_external_paths  # noqa: E402

add_external_paths(dino_wm=True)
from server_single_max import load_model as load_world_model, build_patch_keep_mask  # noqa: E402

DINO_IMG_SIZE = 224
ACTION_MEAN = torch.tensor([0.45678952, 0.00051019, 0.50954217, 0.21926114])
ACTION_STD = torch.tensor([0.03182372, 0.01151787, 0.03419121, 0.41397065])
PROPRIO_MEAN = torch.tensor([0.4564166, 0.00056233, 0.50817657, 0.21921302])
PROPRIO_STD = torch.tensor([0.03217997, 0.01056713, 0.0327194, 0.4139551])


def load_frozen_world_model(device):
    import hydra
    with hydra.initialize_config_dir(config_dir=str(DINO_WM_DIR / "conf"), version_base=None):
        cfg = hydra.compose(config_name="train")
    model = load_world_model(DINO_WM_CKPT, cfg, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)  # frozen -- guidance only ever needs grad w.r.t. the ACTION input
    return model


def dino_preprocess(imgs_uint8_or_float01, device):
    """imgs: (B, T, C, H, W) in [0,1] or [0,255], any H/W -> dino_wm's 224x224, normalized
    to [-1,1]-ish via the same (0.5,0.5,0.5) transform server_single_max.py uses."""
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize(DINO_IMG_SIZE),
        transforms.CenterCrop(DINO_IMG_SIZE),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    x = imgs_uint8_or_float01.to(device).float()
    if x.max() > 1.5:
        x = x / 255.0
    b, t, c, h, w = x.shape
    x = tf(x.reshape(b * t, c, h, w)).reshape(b, t, c, DINO_IMG_SIZE, DINO_IMG_SIZE)
    return x


def nominal_divergence(world_model, visual_hist, proprio_hist, actions_real, device):
    """Single (unperturbed) rollout -- the project's own cheapest validated metric
    (section 7.3: nominal N=1 matches the 50-rollout Deviator Agent on cost-benefit, and is
    the only metric cheap enough to compute at every guided denoising step). actions_real:
    (B, T_pred, 4) in REAL units (not yet dino_wm-normalized) -- MUST be differentiable
    (no .detach()) for gradients to reach the diffusion trajectory.
    """
    am, asd = ACTION_MEAN.to(device), ACTION_STD.to(device)
    pm, psd = PROPRIO_MEAN.to(device), PROPRIO_STD.to(device)
    proprio_n = (proprio_hist.to(device) - pm) / psd
    actions_n = (actions_real - am) / asd
    obs0 = {"visual": visual_hist, "proprio": proprio_n}
    z_obses, _ = world_model.rollout(obs0, actions_n)
    z_visual = z_obses["visual"]  # (B, T_hist+T_pred, 196, 384)
    n_hist = visual_hist.shape[1]
    cos = torch.nn.functional.cosine_similarity
    d_end = 1 - cos(z_visual[:, n_hist], z_visual[:, -1], dim=-1)  # (B, 196) -- "nominal" per patch
    keep = build_patch_keep_mask(d_end.shape[-1], device)
    return d_end[:, keep].mean(dim=-1)  # (B,) -- differentiable scalar per batch item


def steered_conditional_sample(policy, condition_data, condition_mask, world_model,
                               visual_hist_dino, proprio_hist_dino,
                               guidance_scale=0.0, guidance_frac=0.3, skip_last_n=2,
                               grad_clip_norm=None,
                               local_cond=None, global_cond=None, generator=None, **kwargs):
    """Drop-in replacement for DiffusionUnetImagePolicy.conditional_sample with guidance
    added on the last `guidance_frac` fraction of denoising steps, EXCEPT the final
    `skip_last_n` steps. guidance_scale=0 recovers the exact unsteered baseline (useful for a
    paired before/after comparison with identical randomness).

    skip_last_n=2 default is empirical, not arbitrary: guiding all the way to t=0 caused a
    real overshoot on one of the first 5 test observations (divergence 6x WORSE than
    unsteered at guidance_scale=5) -- diagnosed via debug_ep1.py by logging per-step gradient
    norms (never large, ruling out gradient explosion / clipping as a fix) and the final
    post-guidance score at skip_last_n in {0,1,2}. The last denoising step has no further
    refinement left to absorb a guided nudge, so it edits the final output directly with no
    correction opportunity -- skipping the last 2 steps fixed it (0.0442 -> 0.0063, actually
    better than the 0.0073 unsteered baseline on that same observation)."""
    model = policy.model
    scheduler = policy.noise_scheduler
    device = condition_data.device

    trajectory = torch.randn(size=condition_data.shape, dtype=condition_data.dtype,
                             device=device, generator=generator)
    scheduler.set_timesteps(policy.num_inference_steps)
    timesteps = list(scheduler.timesteps)
    n_guided = int(len(timesteps) * guidance_frac) if guidance_scale > 0 else 0
    guided_from = len(timesteps) - n_guided
    guided_to = len(timesteps) - skip_last_n if guidance_scale > 0 else 0

    divergences = []
    for i, t in enumerate(timesteps):
        trajectory[condition_mask] = condition_data[condition_mask]

        if guided_from <= i < guided_to:
            trajectory = trajectory.detach().clone().requires_grad_(True)
            model_output = model(trajectory, t, local_cond=local_cond, global_cond=global_cond)
            alpha_prod_t = scheduler.alphas_cumprod.to(device)[t]
            pred_x0 = (trajectory - (1 - alpha_prod_t).sqrt() * model_output) / alpha_prod_t.sqrt()
            real_actions = policy.normalizer["action"].unnormalize(pred_x0)
            div = nominal_divergence(world_model, visual_hist_dino, proprio_hist_dino,
                                     real_actions, device)
            divergences.append(div.mean().item())
            grad = torch.autograd.grad(div.mean(), trajectory)[0]
            if grad_clip_norm is not None:
                gnorm = grad.norm()
                if gnorm > grad_clip_norm:
                    grad = grad * (grad_clip_norm / (gnorm + 1e-8))
            with torch.no_grad():
                trajectory = trajectory - guidance_scale * grad
            model_output = model_output.detach()
        else:
            with torch.no_grad():
                model_output = model(trajectory, t, local_cond=local_cond, global_cond=global_cond)

        with torch.no_grad():
            trajectory = scheduler.step(model_output, t, trajectory, generator=generator,
                                        **kwargs).prev_sample

    trajectory[condition_mask] = condition_data[condition_mask]
    return trajectory.detach(), divergences


def steered_predict_action(policy, obs_dict, world_model, visual_hist_dino, proprio_hist_dino,
                           guidance_scale=0.0, guidance_frac=0.3, skip_last_n=2, generator=None):
    """Mirrors DiffusionUnetImagePolicy.predict_action but routes through
    steered_conditional_sample. Assumes obs_as_global_cond=True (true for the jenga config)."""
    nobs = policy.normalizer.normalize(obs_dict)
    value = next(iter(nobs.values()))
    B, To = value.shape[:2]
    T = policy.horizon
    Da = policy.action_dim
    device = policy.device
    dtype = policy.dtype

    this_nobs = {k: v[:, :To, ...].reshape(-1, *v.shape[2:]) for k, v in nobs.items()}
    nobs_features = policy.obs_encoder(this_nobs)
    global_cond = nobs_features.reshape(B, -1)
    cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
    cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

    nsample, divergences = steered_conditional_sample(
        policy, cond_data, cond_mask, world_model, visual_hist_dino, proprio_hist_dino,
        guidance_scale=guidance_scale, guidance_frac=guidance_frac, skip_last_n=skip_last_n,
        local_cond=None, global_cond=global_cond, generator=generator)

    action_pred = policy.normalizer["action"].unnormalize(nsample[..., :Da])
    start = To - 1
    end = start + policy.n_action_steps
    return {"action": action_pred[:, start:end], "action_pred": action_pred,
            "guidance_divergences": divergences}
