#!/bin/bash
# Generic CPU array job (no GPU): python $SCRIPT $ARGS with TASK=<index>/<NT>, EXTRA_ENV ';'-separated.
#   sbatch --array=0-$((NT-1)) --export=ALL,NT=<n>,SCRIPT=eval/value_oracle.py,ARGS=run,EXTRA_ENV="VO_SET=scl" cluster/cpu_array.sbatch
#SBATCH --account=def-soojeon
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
cd "${SLF_ROOT:-/scratch/ali313/slf/repo}"
module load apptainer
TASK=${SLURM_ARRAY_TASK_ID:-0}/${NT:-1}
ENV="MUJOCO_GL=osmesa,PYOPENGL_PLATFORM=osmesa,LD_LIBRARY_PATH=$PWD/../glx/osmesa:/.singularity.d/libs,PYTHONUNBUFFERED=1,OMP_NUM_THREADS=1,TASK=$TASK${EXTRA_ENV:+,${EXTRA_ENV//;/,}}"
apptainer exec --cleanenv --env "$ENV" -B /project -B /scratch -B $HOME ../torch290.sif ../pyenv/bin/python $SCRIPT ${ARGS:-}
