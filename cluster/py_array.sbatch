#!/bin/bash
# Generic array job: runs `python $SCRIPT $ARGS` in the container with TASK=<array index>/<NT> and EXTRA_ENV.
#   sbatch --array=0-$((NT-1)) --export=ALL,NT=<n>,SCRIPT=eval/cf_data.py,ARGS="",EXTRA_ENV="CF_RUN=...;CF_OUT=..." cluster/py_array.sbatch
#SBATCH --exclude=fc10713   # Fir node with cudaErrorDevicesUnavailable (2026-10-23)
#SBATCH --account=def-soojeon
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
# EXTRA_ENV: separate settings with ';' (sbatch --export splits on commas, so commas would be lost)
set -euo pipefail
cd "${SLF_ROOT:-$HOME/projects/def-soojeon/ali313/slf/repo}"
module load apptainer
TASK=${SLURM_ARRAY_TASK_ID:-0}/${NT:-1}
ENV="MUJOCO_GL=osmesa,PYOPENGL_PLATFORM=osmesa,TORCH_HOME=$PWD/../torch_home,DIFFUSION_POLICY_DIR=$PWD/../diffusion_policy"
ENV="$ENV,LD_LIBRARY_PATH=$PWD/../glx/osmesa:/.singularity.d/libs,PYTHONUNBUFFERED=1,CF_TASK=$TASK,TASK=$TASK${EXTRA_ENV:+,${EXTRA_ENV//;/,}}"
apptainer exec --nv --cleanenv --env "$ENV" -B /project -B /scratch -B $HOME ../torch290.sif ../pyenv/bin/python $SCRIPT ${ARGS:-}
