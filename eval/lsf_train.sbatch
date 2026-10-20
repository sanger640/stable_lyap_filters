#!/bin/bash
# LSF baseline training on Nibi: merge data parts (cf built on Nibi + clo uploaded), classifier, DDPG value function.
#SBATCH --account=def-soojeon
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd $HOME/projects/def-soojeon/ali313/slf/repo
module load apptainer
RUN() { apptainer exec --nv --cleanenv --env "TORCH_HOME=$PWD/../torch_home,PYTHONUNBUFFERED=1,LSF_TAGS=cf+clo,LSF_EPOCHS=20,LSF_ITERS=300" -B /project -B /scratch -B $HOME ../torch290.sif ../pyenv/bin/python eval/lsf.py "$1"; }
RUN merge; echo "merge done $(date)"
RUN classifier; echo "classifier done $(date)"
RUN value; echo "value done $(date)"
