#!/bin/bash
# SafeLIBERO closed loop on Rorqual: one H100 slice per array task runs the pi0.5 server (openpi venv), the filter
# server (Jenga container) and up to NCLI simulator clients (SafeLIBERO client venv).
#   sbatch --array=0-$((NT-1)) --export=ALL,NT=<n>,UNITS=<file>,OUT=<dir> libero/cluster_l4.sbatch
# UNITS: one line per client run: "<arm> <suite> <level> <task> <ep> [<ep> ...]"; task i runs lines with index % NT == i.
#SBATCH --account=def-soojeon
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -uo pipefail
SCR=/scratch/ali313/libero
cd $SCR/repo
TASK=${SLURM_ARRAY_TASK_ID:-0}; NCLI=${NCLI:-4}
PP=$((8100 + (SLURM_JOB_ID + TASK) % 500)); MP=$((PP + 600))           # ports unique per job on a shared node
L=$PWD/logs/l4_${SLURM_ARRAY_JOB_ID:-0}_$TASK; mkdir -p logs $OUT

( cd $SCR/vlsa-aegis && XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 OPENPI_DATA_HOME=$SCR/openpi_cache \
  openpi/.venv/bin/python scripts/serve_policy.py --env LIBERO --port $PP > $L.policy.log 2>&1 ) &
module load apptainer
apptainer exec --nv --cleanenv --env "TORCH_HOME=/scratch/ali313/slf/torch_home,PYTHONUNBUFFERED=1${EXTRA_ENV:+,${EXTRA_ENV//;/,}}" \
  -B /scratch -B $HOME /scratch/ali313/slf/torch290.sif /scratch/ali313/slf/pyenv/bin/python libero/monitor_server.py --port $MP > $L.monitor.log 2>&1 &
for i in $(seq 120); do grep -q ready $L.monitor.log 2>/dev/null && break; sleep 10; done      # the client itself waits for pi0.5
grep -q ready $L.monitor.log || { echo "monitor server failed"; tail -20 $L.monitor.log; exit 1; }

export LIBERO_CONFIG_PATH=$SCR/libero_config PYTHONPATH=$SCR/vlsa-aegis/safelibero:$PWD/libero MUJOCO_GL=${MUJOCO_GL:-egl} PYOPENGL_PLATFORM=${MUJOCO_GL:-egl}
awk -v n=$NT -v t=$TASK 'NR % n == t' $UNITS | while read -r arm suite level task eps; do
  echo "$SCR/clientenv/bin/python libero/filter_eval.py --arm $arm --suite $suite --level $level --tasks $task --episodes $eps --port $PP --mport $MP --out $OUT/$arm --videos ${VIDEOS:-0}"
done | xargs -P $NCLI -I{} bash -c '{} 2>&1 | grep -E "^\{|Traceback|Error"' >> $L.clients.log
pkill -P $$ ; echo "task $TASK done $(date)" >> $L.clients.log
