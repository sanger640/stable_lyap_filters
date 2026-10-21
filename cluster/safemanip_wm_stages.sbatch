#!/bin/bash
# SafeManip world-model stages on Fir (eval/safemanip_wm.py) in the Jenga container: STAGES="pca encode train ens0 ... check"
# for each view in VIEWS (default "monitor wrist").   sbatch --export=ALL,STAGES="pca encode" cluster/safemanip_wm_stages.sbatch
#SBATCH --account=def-soojeon
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
set -uo pipefail
cd /scratch/ali313/slf/repo
module load apptainer
for v in ${VIEWS:-monitor wrist}; do
  for st in $STAGES; do
    a=$st; [[ $st == ens* ]] && a="ens ${st#ens}"
    apptainer exec --nv --cleanenv --env "TORCH_HOME=$PWD/../torch_home,PYTHONUNBUFFERED=1,PYTHONUSERBASE=/scratch/ali313/slf/userbase,SM_VIEW=$v,SM_DATA=/scratch/ali313/safemanip/wm_data" \
      -B /scratch -B $HOME ../torch290.sif ../pyenv/bin/python eval/safemanip_wm.py $a 2>&1 | grep --line-buffered -v -iE "warn|emb_dim"
    echo "== $v $st done $(date)"
  done
done
