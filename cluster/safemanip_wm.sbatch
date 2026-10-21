#!/bin/bash
# SafeManip world-model data on Fir (docs/BENCHMARK_DD.md 2b-WM): one H100 slice runs the pi0.5 server (openpi venv, GPU)
# and up to NCLI eval/safemanip_wm_data.py clients (same venv inside the Jenga container for OSMesa; EGL does not work on
# MIG slices). Scripts live in $SM/scripts (next to the robosuite checkout they would import it as a namespace package).
# The tokenizer is read from /scratch (OPENPI_DATA_HOME): compute nodes have no internet, Fir's home gave I/O errors.
#   sbatch --array=0-<n-1> --export=ALL,UNITS=<file>,OUT=<dir> cluster/safemanip_wm.sbatch
# UNITS: one line per client: "<task> <seed> <mode> [max_points]"; array task i runs lines with (line_no % NT) == i.
#SBATCH --account=def-soojeon
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -uo pipefail
SM=/scratch/ali313/safemanip; OPI=$SM/SafeManip/eval/models/openpi
cd $SM; mkdir -p logs $OUT
NCLI=${NCLI:-8}; NT=${NT:-1}; TASK=${SLURM_ARRAY_TASK_ID:-0}
PP=$((8100 + (SLURM_JOB_ID + TASK) % 500)); L=$SM/logs/wm_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_$TASK
( cd $OPI && XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 OPENPI_ROOT=$OPI OPENPI_DATA_HOME=/scratch/ali313/openpi_cache .venv/bin/python \
  ../../single_task/serve_policy_wrapper.py --port $PP policy:checkpoint --policy.config=pi05_pretrain_human300 \
  --policy.dir=$SM/ckpt/pi05_pretrain_human300/multitask_learning/75000 > $L.policy.log 2>&1 ) &
module load apptainer
CENV="LD_LIBRARY_PATH=/scratch/ali313/libero/osmesa,MUJOCO_GL=osmesa,PYOPENGL_PLATFORM=osmesa,LP_NUM_THREADS=2,NUMBA_DISABLE_JIT=1,PYTHONUNBUFFERED=1"
awk -v n=$NT -v t=$TASK '(NR - 1) % n == t' $UNITS | while read -r task seed mode maxp; do
  echo "cd $OPI && apptainer exec --cleanenv --env $CENV -B /scratch -B $HOME /scratch/ali313/slf/torch290.sif .venv/bin/python $SM/scripts/safemanip_wm_data.py --tasks $task --episodes 1 --seed $seed --mode $mode --max-points ${maxp:-0} --port $PP --out $OUT/$mode/s$seed > $L.${task}_$seed.log 2>&1"
done | xargs -P $NCLI -I{} bash -c '{}'
kill %1 2>/dev/null; echo "done $(date)"
