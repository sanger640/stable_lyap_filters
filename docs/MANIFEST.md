# Manifest: models and data not in git (all on this machine, 2026-10-27)

Paths are relative to the repo root unless absolute. sha256 = first 16 hex characters. Regenerating any of these is
expensive (hours to days); do not delete. `*.pt`, `*.ckpt`, `*.lmdb` are git-ignored by design.

| File | Size | sha256 (16) | Role |
|---|---|---|---|
| `results/program_b/oil_v1/model/model_OIL.pt` | 9.1 MB | 87be24b02635426e | world model used by every filter (DINOv2 patch latents 28x28x8, deterministic) |
| `results/program_b/ens_v1/model_ENS0.pt` | 9.1 MB | 9f0aff91eed3ab74 | ensemble members 0-3 (uncertainty margin for UNISafe-style values, psel certificate) |
| `results/program_b/ens_v1/model_ENS1.pt` | 9.1 MB | 7612f92f1219a90f | ensemble members 0-3 (uncertainty margin for UNISafe-style values, psel certificate) |
| `results/program_b/ens_v1/model_ENS2.pt` | 9.1 MB | 0dd681d35af6cf1a | ensemble members 0-3 (uncertainty margin for UNISafe-style values, psel certificate) |
| `results/program_b/ens_v1/model_ENS3.pt` | 9.1 MB | 1ad06dc7846e4422 | ensemble members 0-3 (uncertainty margin for UNISafe-style values, psel certificate) |
| `results/program_b/threeblock_v3/G.pt` | 3.2 MB | 38df941b88e8da6d | start operator: perception-noise perturbation of a latent |
| `data/jenga/policy/latest.ckpt` | 855.0 MB | ad3cdd090e9276f4 | diffusion policy (Jenga); code at ~/wksp/diffusion_policy (DIFFUSION_POLICY_DIR) |
| `vendor/panda_express_sim.tar` | 36.7 MB | 46c6b30dac82b267 | MuJoCo scene / robot (extracted to a temp dir per process) |
| `results/jenga/bench/jenga_bench.npz` | 31.8 MB | 8bf358f2b6ab56f5 | measured execution-noise snippets + error model inputs (the 1x noise) |
| `data/jenga/jenga_single_100.lmdb` | 21.5 GB (dir) | - | recorded dataset (world-model / encoder data) |
| `results/program_b/lsf_v1/classifier.pt` | 3.4 MB | b491b808729a6793 | LABELLED failure classifier (labelled baselines only) |
| `results/program_b/lsf_v1/margin_lf.pt` | 3.4 MB | ee15cb71a3fbfd7d | label-free fragility margin (old failure set) |
| `results/program_b/lsf_v1/margin_irr_nl8.pt` | 5.5 MB | 3f4e041b6c79e205 | label-free irreversibility margin, 0.8 s (MAIN method's failure set) |
| `results/program_b/lsf_v1/margin_irr_nl16.pt` | 5.5 MB | c9b388118ee2eccb | same, 1.6 s (ablation) |
| `results/program_b/lsf_v1/value_irr8_alea.pt` | 13.4 MB | d2197a2a03739e84 | MAIN METHOD's value (fragile irreversibility, 0.8 s) |
| `results/program_b/lsf_v1/value_irr16_alea.pt` | 13.4 MB | a77c0d966cc2213a | 1.6 s value (ablation) |
| `results/program_b/lsf_v1/value_lf_alea.pt` | 13.4 MB | 9131ea351bfca36e | fragility value (ablation: old always-on / gated edits) |
| `results/program_b/lsf_v1/value_lf_alea_n1.pt` | 13.4 MB | 78c422cc5a3756e9 | fragility value trained at 1x (IRRM-B) |
| `results/program_b/lsf_v1/value_unisafe_alea.pt` | 13.4 MB | 2ac6aa9dbf61b7c7 | UNISafe + noise-aware pessimism (baseline lsfa030) |
| `results/program_b/lsf_v1/value_unisafe.pt` | 13.4 MB | 830c098d33ca4745 | UNISafe re-implementation without noise pessimism (dev reference) |
| `results/program_b/lsf_v1/value_pub_code.pt` | 11.0 MB | 2213580d8869be44 | LSF as published, authors' schedule (baseline lsfp020) |
| `results/program_b/lsf_v1/value_pub_paper.pt` | 11.0 MB | 0630c2dca52b2c85 | LSF as published, paper gamma (diverged; not used) |
| `results/program_b/lsf_v1/train_reset_z.npy` | 74.9 MB | ae53978dee5b0ede | imagination reset latents for all value training |
| `results/program_b/lsf_v1/train_reset_prop.npy` | 0.1 MB | 457314e26979a650 | their proprioception |
| `results/program_b/lsf_v1/train_reset_plan.npy` | 0.8 MB | ffc47dae9283cb73 | the policy's plans at those resets |
| `results/jenga/cbf_calibration_OIL.json` | 0.0 MB | b99a7b1aef14c376 | label-free calibration used by every filter run (CBF_CALIB); committed |

## Large data directories (untracked, on disk)

| Path | Size | Content |
|---|---|---|
| `results/program_b/policy_rollouts/` | ~2.0 GB | raw per-episode rollouts (`paired_<mode><tag>/ep*.json` + `.npz` decision states); the replay audits and videos read these |
| `results/jenga/hj2/data/` | ~0.97 GB | per-decision latents + fork statistics of the training/held-out runs (margin training, CF1-3, IRRM stage 1) |
| `results/jenga/irrm2_labels_nl{2,8,16}_{0,1}.npz`, `irrm2_held_nl*.npz` | ~0.5 GB | undo-test labels for the irreversibility margins (eval/irrm2.py) |
| `results/program_b/lsf_v1/` | ~3.4 GB | all value / margin / classifier networks + labelled LSF data (train/val latents) |
| `results/program_b/filter_videos/` | ~60 MB | annotated four-way videos (eval/filter_videos.py) used by the dashboard |
| `~/.cache/torch/hub/facebookresearch_dinov2_main` | - | DINOv2 code + weights (loaded via torch.hub, offline cache) |
| `/home/corey/wksp/diffusion_policy` | - | diffusion-policy code the policy checkpoint needs |

Verify: `sha256sum <file> | cut -c1-16` and compare.
