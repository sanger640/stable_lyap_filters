# Environment (the one every Jenga result since October 2026 was produced with)

Machine: this workstation, NVIDIA RTX 3090 (24 GB), driver 580.178.04. Do NOT use the machine `gpu5090` (user rule).

- **Interpreter:** `/home/corey/wksp/dp_venv/bin/python` (Python 3.10.12). Always call it explicitly.
- **The venv has `include-system-site-packages = true`.** torch 2.9.0+cu128, torchvision 0.24.0, triton 3.5.0 and the
  nvidia-*-cu12 12.8 libraries are imported from the USER site (`~/.local/lib/python3.10/site-packages`), not from the
  venv. Other key versions: numpy 2.2.6, mujoco 3.13.0, dill 0.4.1, hydra 1.3.7, scikit-learn 1.7.2, scipy 1.15.3,
  imageio-ffmpeg 0.6.0, lmdb 2.3.0. Full list: `docs/env_dp_venv_freeze.txt` (pip freeze of the venv as seen by it).
- **Fragility:** a third-party installer run with user-site enabled uninstalled torch from `~/.local` once (2026-10-26).
  Run any other project's pip / conda / setup script with `PYTHONNOUSERSITE=1` and `PYTHONPATH=` unset. Health check
  after any install: `/home/corey/wksp/dp_venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
  (expect 2.9.0+cu128 True). Restore recipe: `PYTHONPATH= /usr/bin/python3 -m pip install --user --no-deps <exact versions>`.
- **Rendering:** `MUJOCO_GL=egl` for every simulator / video job.
- **External code:** diffusion-policy code at `/home/corey/wksp/diffusion_policy` (`DIFFUSION_POLICY_DIR`); DINOv2
  through `torch.hub` from the local cache `~/.cache/torch/hub/facebookresearch_dinov2_main` (works offline).
- `requirements.txt` in the repo root documents an OLDER conda environment (`dino_wm`, Python 3.11, torch 2.11) from a
  different machine (Sept 2026); it is not the environment above.
- Other environments on this machine, unrelated to the current work: `~/miniforge3` (conda env `behavior`, ManiGuard,
  on hold), `~/wksp/safelibero`, `~/wksp/maniskill_bench` (closed benchmarks).
- Cluster (Alliance: Fir / Nibi / Rorqual / Narval): container + pyenv per cluster, see `cluster/README.md`; needs the
  user's Duo login; repos there are rsynced copies, not git.
