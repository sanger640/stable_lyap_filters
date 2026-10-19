#!/bin/bash
# SafeLIBERO closed loop. usage: libero/run_l4.sh pilot|test   (policy server on :8000 must be running)
# pilot: G-CEM on dev states 5, 15 of all 32 scenarios -> results/libero/l4a_pilot/gcem
# test:  G-CEM and hold on test states 25-34 -> results/libero/l4b/{gcem,hold}; policy arm = L0 (libero/l4_analysis.py)
set -u
cd "$(dirname "$0")/.."
PHASE=$1
nohup python3 libero/monitor_server.py --port 6100 > results/libero/monitor_server_$PHASE.log 2>&1 &
SRV=$!
until grep -q "ready" results/libero/monitor_server_$PHASE.log; do sleep 5; kill -0 $SRV || exit 1; done
source ~/wksp/safelibero/vlsa-aegis/main/.venv/bin/activate
export LIBERO_CONFIG_PATH=$HOME/wksp/safelibero/libero_config PYTHONPATH=$HOME/wksp/safelibero/vlsa-aegis/safelibero:$PWD/libero MUJOCO_GL=egl
PAIRS=(safelibero_spatial:I safelibero_spatial:II safelibero_object:I safelibero_object:II safelibero_goal:I safelibero_goal:II safelibero_long:I safelibero_long:II)
run() {  # worker arm out episodes...
  local w=$1 arm=$2 out=$3; shift 3
  for i in "${!PAIRS[@]}"; do
    [ $((i % 3)) -eq $w ] || continue
    s=${PAIRS[$i]%:*}; l=${PAIRS[$i]#*:}
    python libero/filter_eval.py --arm $arm --suite $s --level $l --tasks 0 1 2 3 --episodes "$@" --out $out --videos 1 2>&1 \
      | grep -E '^\{|Traceback|Error' >> $out.w$w.log
  done
}
if [ "$PHASE" = pilot ]; then
  for w in 0 1 2; do run $w gcem results/libero/l4a_pilot/gcem 5 15 & done; wait
else
  for arm in gcem hold; do
    for w in 0 1 2; do run $w $arm results/libero/l4b/$arm $(seq 25 34) & done; wait
  done
fi
kill $SRV; echo "$PHASE done $(date)" >> results/libero/l4_runs.log
