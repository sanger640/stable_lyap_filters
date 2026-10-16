#!/bin/bash
# Packed paired episodes: one job on a FULL GPU runs GROUP shard-workers in parallel (each = what one paired_array task
# would do: SHARD=t/NT, all ARMS sequentially for its episodes). Array index g covers tasks T0+g*GROUP .. +GROUP-1.
#   sbatch --array=0-<ngroups-1> --export=ALL,NT=150,T0=75,TMAX=149,GROUP=8,N=600,OFF=12000,TAG=_dev8,ARMS="record cbfm cbfmo" cluster/paired_group.sbatch
#SBATCH --account=def-soojeon
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -uo pipefail
cd "${SLF_ROOT:-$HOME/projects/def-soojeon/ali313/slf/repo}"
module load apptainer
export CBF_WM=${CBF_WM:-$PWD/results/program_b/closedloop_v1/model/model_US.pt}
BASE="MUJOCO_GL=osmesa,PYOPENGL_PLATFORM=osmesa,TORCH_HOME=$PWD/../torch_home,DIFFUSION_POLICY_DIR=$PWD/../diffusion_policy"
BASE="$BASE,LD_LIBRARY_PATH=$PWD/../glx/osmesa:/.singularity.d/libs,PER_EPISODE_SEEDS=1,PAIRED_PER_EP=1,PAIRED_TAG=$TAG,EP_OFFSET=$OFF"
BASE="$BASE,EXEC_NOISE_SCALE=${EXEC_NOISE_SCALE:-2},HARD_OFF=${HARD_OFF:-1},CBF_WM=$CBF_WM,PYTHONUNBUFFERED=1,OMP_NUM_THREADS=2"
for w in $(seq 0 $((GROUP - 1))); do
  t=$((T0 + SLURM_ARRAY_TASK_ID * GROUP + w))
  [ $t -gt $TMAX ] && break
  (for arm in $ARMS; do
     apptainer exec --nv --cleanenv --env "$BASE,SHARD=$t/$NT" -B /project -B /scratch -B $HOME ../torch290.sif ../pyenv/bin/python eval/policy_rollout.py paired $arm $N
   done) > logs/${SLURM_JOB_NAME}_t${t}.log 2>&1 &
done
wait
