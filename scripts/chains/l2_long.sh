#!/bin/bash
cd /home/corey/wksp/stable_lyap_filters
source ~/wksp/safelibero/vlsa-aegis/main/.venv/bin/activate
export LIBERO_CONFIG_PATH=$HOME/wksp/safelibero/libero_config PYTHONPATH=$HOME/wksp/safelibero/vlsa-aegis/safelibero:$PWD/libero MUJOCO_GL=egl
EPS=$(seq 0 24)
c() { python libero/collect_l2.py --suite safelibero_long --level $1 --tasks ${@:2} --episodes $EPS --out results/libero/l2_data 2>&1 | grep -E '^\{|Traceback|Error' >> results/libero/l2_long_w$W.log; }
(W=0; c I 0 1; c II 2) &
(W=1; c I 2 3; c II 3) &
(W=2; c II 0 1) &
wait; echo "L2 long done $(date)" >> results/libero/l2_long_w0.log
