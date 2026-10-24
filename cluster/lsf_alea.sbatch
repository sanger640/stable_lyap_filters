#!/bin/bash
# LSF audit 3 stage 2 (docs/LATENT_SAFETY_FILTER.md): UNISafe value retrained with the original LSF's pessimistic sampling
# under our noise model (LSF_ALEA=8 noisy transitions per imagined step, perceived starts). Nibi: sbatch cluster/lsf_alea.sbatch
#SBATCH --account=def-soojeon
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd $HOME/projects/def-soojeon/ali313/slf/repo
module load apptainer
apptainer exec --nv --cleanenv --env "TORCH_HOME=$PWD/../torch_home,PYTHONUNBUFFERED=1,LSF_OUT=$PWD/results/program_b/lsf_v1_alea,LSF_ALEA=8,LSF_ALEA_SCALE=2,LSF_VALUE_NAME=${VNAME:-value_unisafe_alea.pt},LSF_UPDATES=${UPD:-60000}" -B /project -B /scratch -B $HOME ../torch290.sif ../pyenv/bin/python eval/lsf.py sac
echo "sac done $(date)"
