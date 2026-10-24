#!/bin/bash
# Paired closed-loop episodes on an Alliance cluster: one array task = a set of episodes, BOTH arms in the same job
# (same GPU, same software), so each pair differs only by the filter.
#   sbatch --array=0-$((NT-1)) --export=ALL,NT=<tasks>,N=<episodes>,OFF=<first episode>,TAG=<tag>,ARMS="record cbf" cluster/paired_array.sbatch
# Results: results/program_b/policy_rollouts/paired_<arm><TAG>/ep<i>.{json,npz}; merge with cluster/merge_paired.py
#SBATCH --exclude=fc10713   # Fir node with cudaErrorDevicesUnavailable (2026-10-23)
#SBATCH --account=def-soojeon
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out
# EXTRA_ENV: separate settings with ';' (sbatch --export splits on commas, so commas would be lost)
set -euo pipefail
cd "${SLF_ROOT:-$HOME/projects/def-soojeon/ali313/slf/repo}"
module load apptainer
export CBF_WM=${CBF_WM:-$PWD/results/program_b/closedloop_v1/model/model_US.pt}
# clean container environment (host cert/library paths break it); pass only what the run needs
ENV="MUJOCO_GL=osmesa,PYOPENGL_PLATFORM=osmesa,TORCH_HOME=$PWD/../torch_home,DIFFUSION_POLICY_DIR=$PWD/../diffusion_policy"
ENV="$ENV,PER_EPISODE_SEEDS=1,PAIRED_PER_EP=1,PAIRED_TAG=$TAG,EP_OFFSET=$OFF,SHARD=$SLURM_ARRAY_TASK_ID/$NT"
ENV="$ENV,LD_LIBRARY_PATH=$PWD/../glx/osmesa:/.singularity.d/libs"   # MIG GPU slices have no OpenGL/EGL: MuJoCo renders on the CPU (OSMesa, libs from Ubuntu jammy)
ENV="$ENV,EXEC_NOISE_SCALE=${EXEC_NOISE_SCALE:-2},HARD_OFF=${HARD_OFF:-1},CBF_WM=$CBF_WM,PYTHONUNBUFFERED=1${EXTRA_ENV:+,${EXTRA_ENV//;/,}}"
PY="apptainer exec --nv --cleanenv --env $ENV -B /project -B /scratch -B $HOME ../torch290.sif ../pyenv/bin/python"   # torch 2.9.0+cu128, mujoco 3.13 (= local)
for arm in $ARMS; do
  $PY eval/policy_rollout.py paired $arm $N
done
