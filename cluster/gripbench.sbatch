#!/bin/bash
# Offline gripper bench (results/jenga/gripbench_protocol.json): NSH=10 sbatch --array=0-9 --export=ALL cluster/gripbench.sbatch
#SBATCH --exclude=fc10713
#SBATCH --account=def-soojeon
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
cd "${SLF_ROOT:-/scratch/ali313/slf/repo}"
module load apptainer
ENV="MUJOCO_GL=osmesa,PYOPENGL_PLATFORM=osmesa,TORCH_HOME=$PWD/../torch_home,DIFFUSION_POLICY_DIR=$PWD/../diffusion_policy"
ENV="$ENV,LD_LIBRARY_PATH=$PWD/../glx/osmesa:/.singularity.d/libs,EXEC_NOISE_SCALE=2,SHARD=$SLURM_ARRAY_TASK_ID/${NSH:-10},LIMIT=${LIMIT:-1000000000},PYTHONUNBUFFERED=1"
apptainer exec --nv --cleanenv --env $ENV -B /project -B /scratch -B $HOME ../torch290.sif ../pyenv/bin/python eval/grip_bench.py run
