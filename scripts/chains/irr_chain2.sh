# CF3 grade, then IRR dev (results/jenga/irr_protocol.json) iff CF3 passes, after the LF5 + LF6 chains (one GPU pool at a time)
cd /home/corey/wksp/stable_lyap_filters
PY=/home/corey/wksp/dp_venv/bin/python


P=$($PY -c "import json;print(json.load(open('results/jenga/cf3_result.json'))['GATE_CF3_pass'])")
echo "CF3 graded GATE_CF3_pass=$P $(date)"
[ "$P" = "True" ] || { echo "CF3 failed: IRR cancelled per protocol $(date)"; exit 0; }
while kill -0 2152046 2>/dev/null || kill -0 3648391 2>/dev/null; do sleep 120; done
echo "IRR sweep start $(date)"
for i in 0 1 2 3 4 5; do
  ( for arm in lfci000 irg irs; do
      MUJOCO_GL=egl PER_EPISODE_SEEDS=1 PAIRED_PER_EP=1 PAIRED_TAG=_irrdev EP_OFFSET=15000 SHARD=$i/6 EXEC_NOISE_SCALE=2 HARD_OFF=1 \
      CBF_CALIB=results/jenga/cbf_calibration_OIL.json PYTHONUNBUFFERED=1 $PY eval/irr_rollout.py $arm 100
    done ) >> logs/irrdev_$i.out 2>&1 &
done
wait
echo "IRR sweep done $(date)"
$PY cluster/merge_paired.py paired_irg_irrdev paired_irs_irrdev paired_lfci000_irrdev
while pgrep -f "[o]utcome_audit.py runs" >/dev/null; do sleep 120; done
AUDIT_OUT=results/jenga/outcome_audit_irrdev.json MUJOCO_GL=egl $PY eval/outcome_audit.py runs paired_irg_irrdev paired_irs_irrdev paired_lfci000_irrdev > results/jenga/outcome_audit_irrdev.log 2>&1
$PY eval/irr_dev.py > results/jenga/irr_dev.log 2>&1
echo "IRR analysed $(date)"
