# Running paired closed-loop tests on Alliance clusters

Layout per cluster (`$SLF` = `~/projects/def-soojeon/ali313/slf` on Narval/Nibi, `/scratch/ali313/slf` on Rorqual):
- `torch290.sif`: apptainer image of `pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime` (same torch as the local machines)
- `pyenv/`: venv inside the image (`--system-site-packages`), mujoco 3.13, diffusers 0.41, ... (see build_sif.sh)
- `glx/osmesa/`: OSMesa + LLVM 15 + libxml2/ICU from Ubuntu jammy. MIG GPU slices have no OpenGL/EGL, so MuJoCo
  renders on the CPU (about 40 ms per 384 px frame); the world model and policy still run on the GPU slice.
- `repo/`: eval/ src/ vendor/ data/ cluster/ + the model files the runs open (policy ckpt, model_US.pt, G.pt, ...)
- `torch_home/hub`: DINOv2 code + dinov2_vits14 weights (compute nodes have no internet)
- `diffusion_policy/`: the policy code (sanger640 fork)

Gotchas: Alliance Python refuses PyPI manylinux wheels (hence the container); run the container with `--cleanenv`
(host cert/library paths leak in); use `pip --isolated` only outside it. Uploads from the lab machine to Narval and
Rorqual are slow (~0.5 MB/s); to Nibi fast.

Check (2026-10-15, Narval A100 2g.10gb slice): episode 9000 record/cbf reproduced the local run (60 chunks / 25
chunks, max lift 48.1 vs 47.9 mm, 51.5 vs 51.7 mm); one paired episode = 9 min.

Submit (Narval example; Rorqual: add SLF_ROOT=/scratch/ali313/slf/repo and default H100 slice; Nibi: default):
  sbatch --gpus=a100_2g.10gb:1 --array=0-$((NT-1)) --export=ALL,NT=$NT,N=$N,OFF=$OFF,TAG=$TAG,ARMS="record cbf" cluster/paired_array.sbatch
Merge: python3 cluster/merge_paired.py paired_record$TAG paired_cbf$TAG  (after copying policy_rollouts/<tag>/ back)
