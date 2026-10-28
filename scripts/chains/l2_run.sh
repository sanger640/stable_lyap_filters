#!/bin/bash
# L2 collection: 32 scenarios x dev states 0-24, 3 workers splitting the 8 (suite, level) pairs.
cd /home/corey/wksp/stable_lyap_filters
source ~/wksp/safelibero/vlsa-aegis/main/.venv/bin/activate
export LIBERO_CONFIG_PATH=$HOME/wksp/safelibero/libero_config PYTHONPATH=$HOME/wksp/safelibero/vlsa-aegis/safelibero:$PWD/libero MUJOCO_GL=egl
EPS=$(seq 0 24)
work() { for pair in "$@"; do s=${pair%:*}; l=${pair#*:}; python libero/collect_l2.py --suite $s --level $l --tasks 0 1 2 3 --episodes $EPS --out results/libero/l2_data 2>&1 | grep -E '^\{|Traceback|Error' >> results/libero/l2_data_w$W.log; done; }
W=0 work safelibero_spatial:I safelibero_object:II safelibero_long:II &
W=1 work safelibero_spatial:II safelibero_goal:I &
W=2 work safelibero_object:I safelibero_goal:II safelibero_long:I &
wait; echo "L2 done $(date)" >> results/libero/l2_data_w0.log
