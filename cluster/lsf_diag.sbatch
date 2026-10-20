#!/bin/bash
# LSF selection check on Nibi: eval/lsf_diag.py for every trained value checkpoint named in NAMES (space-separated).
#SBATCH --account=def-soojeon
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
cd ~/projects/def-soojeon/ali313/slf/repo
module load apptainer
for n in $NAMES; do
  apptainer exec --nv --cleanenv --env "TORCH_HOME=$PWD/../torch_home,PYTHONUNBUFFERED=1,LSF_VALUE_NAME=$n" -B /project -B /scratch -B $HOME \
    ../torch290.sif ../pyenv/bin/python eval/lsf_diag.py 2>&1 | grep -E "AUROC|share_V_below_0\"|decision" | sed "s/^/$n /"
done
