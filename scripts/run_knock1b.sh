#!/usr/bin/env bash
# KNOCK stage 1b (results/jenga/knock_protocol.json): label-free knock-down data -> futures -> model_KNK -> stage-1 gates on
# fresh seeds 42100-42139. Single GPU processes (no pool) alongside the SEED2/TAU3 chain.
# usage (repo root): nohup bash scripts/run_knock1b.sh > results/knock/knock1b_chain.log 2>&1 &
cd "$(dirname "$0")/.."
PY=/home/corey/wksp/dp_venv/bin/python
echo "start $(date)"
env MUJOCO_GL=egl PER_EPISODE_SEEDS=1 PAIRED_TAG=_knock_wm EP_OFFSET=44000 EXEC_NOISE_SCALE=2 HARD_OFF=1 PAIRED_MAX_CHUNKS=15 \
  $PY eval/knock_rollout.py record 150 > logs/knock_record_wm.out 2>&1
echo "record done $(ls results/program_b/policy_rollouts/paired_record_knock_wm/ep*.npz | wc -l) episodes $(date)"
env MUJOCO_GL=egl CL_TAG=paired_record_knock_wm CL_OUT=results/program_b/knock_v1 CL_S2S=44130 CL_N_TRAIN=800 CL_N_VAL=100 CL_WORKERS=1 \
  PYTHONUNBUFFERED=1 $PY eval/closedloop_data.py > logs/knock_futures.out 2>&1
echo "futures $(ls results/program_b/knock_v1/train/*.npz | grep -v start | wc -l) train / $(ls results/program_b/knock_v1/val/*.npz | grep -v start | wc -l) val $(date)"
PYTHONUNBUFFERED=1 $PY eval/knock_wm.py train > results/knock/knk_train.log 2>&1
ls results/program_b/knock_v1/model/model_KNK.pt || { echo "model missing, stop $(date)"; exit 1; }
echo "model_KNK trained $(date)"
$PY eval/knock_wm.py error > results/knock/knk_error.log 2>&1
CBF_WM=$PWD/results/program_b/knock_v1/model/model_KNK.pt KNOCK_WM_SUFFIX=_knk $PY eval/knock_wm.py run 40 42100 > results/knock/wm_run_knk.log 2>&1
CBF_WM=$PWD/results/program_b/knock_v1/model/model_KNK.pt KNOCK_WM_SUFFIX=_knk $PY eval/knock_wm.py grade > results/knock/wm_grade_knk.log 2>&1
echo "KNOCK1B graded $(date)"; grep -E '"(W1|W2|pass|auroc_D30_vs_fork)"' results/knock/wm_result_knk.json
