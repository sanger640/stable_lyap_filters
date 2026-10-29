#!/usr/bin/env bash
# RLAB1 (results/jenga/rlab1_protocol.json): labelled baselines with red-block falls in the failure set.
# S0 (CPU data) runs separately; this chain waits for it and for XFER1 (one GPU job at a time), then:
# classifier -> offline report -> UNISafe value -> LSF-pub value -> dev (2x, 15000-15099) -> test on the H2H5 episodes (2x, 1x).
# usage: nohup bash scripts/run_rlab1.sh <S0 pid> <XFER1 pid> > results/jenga/rlab1_chain.log 2>&1 &
cd "$(dirname "$0")/.."
PY=/home/corey/wksp/dp_venv/bin/python
export LSF_OUT=$PWD/results/program_b/lsf_v2_red
while kill -0 $1 2>/dev/null || kill -0 $2 2>/dev/null; do sleep 300; done
echo "start $(date)"
$PY - <<'P' || exit 1
import numpy as np
for k in ("train_reset_z", "train_reset_prop", "val_reset_z", "val_reset_prop"):
    a, b = np.load(f"results/program_b/lsf_v1/{k}.npy"), np.load(f"results/program_b/lsf_v2_red/{k}.npy")
    assert a.shape == b.shape and np.array_equal(a, b), k
print("reset arrays identical")
P
cp results/program_b/lsf_v1/train_reset_plan.npy $LSF_OUT/
$PY eval/lsf.py classifier > $LSF_OUT/classifier_train.log 2>&1; echo "classifier $(date)"; tail -2 $LSF_OUT/classifier_train.log
$PY eval/rlab1_analysis.py offline > results/jenga/rlab1_offline.log 2>&1; echo "offline $(date)"; tail -20 results/jenga/rlab1_offline.log
LSF_ALEA=8 LSF_ALEA_SCALE=2 LSF_VALUE_NAME=value_unisafe_alea.pt LSF_UPDATES=60000 $PY eval/lsf.py sac > $LSF_OUT/value_unisafe_alea_train.log 2>&1
echo "unisafe value $(date)"
LSF_PUB_MODE=code LSF_VALUE_NAME=value_pub_code.pt $PY eval/lsf.py ddpg_pub > $LSF_OUT/value_pub_code_train.log 2>&1
echo "pub value $(date)"
dev() {  # mode extra-env...
  local mode=$1; shift
  for i in 0 1 2 3 4 5; do
    env MUJOCO_GL=egl PER_EPISODE_SEEDS=1 PAIRED_PER_EP=1 PAIRED_TAG=_rl1dev EP_OFFSET=15000 SHARD=$i/6 EXEC_NOISE_SCALE=2 HARD_OFF=1 \
      CBF_CALIB=results/jenga/cbf_calibration_OIL.json PYTHONUNBUFFERED=1 "$@" $PY eval/policy_rollout.py paired $mode 100 >> logs/rl1dev_$i.out 2>&1 &
  done
  wait; echo "dev $mode done $(date)"
}
dev lsfa030 LSF_ALEA_SCALE=2
for e in 010 020 030 040 050; do dev lsfp$e LSF_VALUE_NAME=value_pub_code.pt; done
RUNS="paired_lsfa030_rl1dev $(for e in 010 020 030 040 050; do echo -n "paired_lsfp${e}_rl1dev "; done)"
$PY cluster/merge_paired.py $RUNS
AUDIT_OUT=results/jenga/outcome_audit_rl1dev.json MUJOCO_GL=egl $PY eval/outcome_audit.py runs $RUNS > results/jenga/outcome_audit_rl1dev.log 2>&1
$PY eval/rlab1_analysis.py dev > results/jenga/rlab1_dev.log 2>&1 || exit 1
SEL=$(cat results/jenga/rlab1_dev_selected.txt); echo "dev selected lsfp$SEL $(date)"
arm() {  # name noise mode tag extra-env...
  local name=$1 nz=$2 mode=$3 tag=$4; shift 4
  echo "$name ${nz}x start $(date)"
  for i in 0 1 2 3 4 5; do
    env MUJOCO_GL=egl PER_EPISODE_SEEDS=1 PAIRED_PER_EP=1 PAIRED_TAG=$tag EP_OFFSET=26200 SHARD=$i/6 EXEC_NOISE_SCALE=$nz HARD_OFF=1 \
      PAIRED_MAX_CHUNKS=120 CBF_CALIB=results/jenga/cbf_calibration_OIL.json PYTHONUNBUFFERED=1 "$@" \
      $PY eval/policy_rollout.py paired $mode 600 >> logs/rl1${tag}_$i.out 2>&1 &
  done
  wait
  echo "$name ${nz}x done $(ls results/program_b/policy_rollouts/paired_$mode$tag/ep*.json | wc -l) episodes $(date)"
}
for nz in 2 1; do
  arm unisafe_r $nz lsfa030    _rl1_${nz}x LSF_ALEA_SCALE=$nz
  arm lsfpub_r  $nz lsfp$SEL   _rl1_${nz}x LSF_VALUE_NAME=value_pub_code.pt
  RUNS="paired_lsfa030_rl1_${nz}x paired_lsfp${SEL}_rl1_${nz}x"
  $PY cluster/merge_paired.py $RUNS
  AUDIT_OUT=results/jenga/outcome_audit_rl1_${nz}x.json MUJOCO_GL=egl AUDIT_NOISE_SCALE=$nz $PY eval/outcome_audit.py runs $RUNS \
    > results/jenga/outcome_audit_rl1_${nz}x.log 2>&1
  echo "${nz}x audited $(date)"
done
$PY eval/rlab1_analysis.py test > results/jenga/rlab1_analysis.log 2>&1
echo "RLAB1 analysed $(date)"
