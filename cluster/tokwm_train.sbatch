#!/bin/bash
# Full-token DINO-WM training (eval/tokwm.py train) on one H100 slice, data in $SLF_ROOT/results/program_b/tok_v1.
#   sbatch --export=ALL,STEPS=30000,OUT=results/program_b/tok_v1/tokwm_v1.pt cluster/tokwm_train.sbatch
#SBATCH --account=def-soojeon
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "${SLF_ROOT:-/scratch/ali313/slf/repo}"
module load apptainer
ENV="TORCH_HOME=$PWD/../torch_home,PYTHONUNBUFFERED=1"
apptainer exec --nv --cleanenv --env "$ENV" -B /project -B /scratch -B $HOME ../torch290.sif ../pyenv/bin/python \
  eval/tokwm.py train results/program_b/tok_v1 $OUT ${STEPS:-30000}
