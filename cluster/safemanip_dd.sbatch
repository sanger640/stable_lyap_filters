#!/bin/bash
# SafeManip due-diligence runs on Rorqual: one H100 slice runs the pi0.5 server (openpi venv, on the GPU) and up to NCLI
# eval/safemanip_fork_check.py clients (same venv, inside the Jenga container for its OSMesa runtime; CPU rendering,
# since EGL does not work on MIG slices). The script lives in $SM/scripts: next to the robosuite checkout in $SM it
# would import that folder as a namespace package instead of the installed robosuite.
#   sbatch --export=ALL,UNITS=<file>,OUT=<dir>,NFUT=33 cluster/safemanip_dd.sbatch
# UNITS: one line per client: "<task> <seed>" (one episode each).
#SBATCH --account=def-soojeon
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
set -uo pipefail
SM=/scratch/ali313/safemanip; OPI=$SM/SafeManip/eval/models/openpi
cd $SM; mkdir -p logs $OUT
NCLI=${NCLI:-8}; PP=$((8100 + SLURM_JOB_ID % 500)); L=$SM/logs/dd_$SLURM_JOB_ID
( cd $OPI && XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 OPENPI_ROOT=$OPI .venv/bin/python ../../single_task/serve_policy_wrapper.py \
  --port $PP policy:checkpoint --policy.config=pi05_pretrain_human300 \
  --policy.dir=$SM/ckpt/pi05_pretrain_human300/multitask_learning/75000 > $L.policy.log 2>&1 ) &
module load apptainer
CENV="LD_LIBRARY_PATH=/scratch/ali313/libero/osmesa,MUJOCO_GL=osmesa,PYOPENGL_PLATFORM=osmesa,LP_NUM_THREADS=2,NUMBA_DISABLE_JIT=1,PYTHONUNBUFFERED=1"
while read -r task seed; do
  echo "cd $OPI && apptainer exec --cleanenv --env $CENV -B /scratch -B $HOME /scratch/ali313/slf/torch290.sif .venv/bin/python $SM/scripts/safemanip_fork_check.py --tasks $task --episodes 1 --seed $seed --masked --n-fut ${NFUT:-33} --port $PP --out $OUT/s$seed > $L.${task}_$seed.log 2>&1"
done < $UNITS | xargs -P $NCLI -I{} bash -c '{}'
kill %1 2>/dev/null; echo "done $(date)"
