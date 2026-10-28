#!/bin/bash
cd /home/corey/wksp/stable_lyap_filters
source ~/wksp/safelibero/vlsa-aegis/main/.venv/bin/activate
export LIBERO_CONFIG_PATH=$HOME/wksp/safelibero/libero_config PYTHONPATH=$HOME/wksp/safelibero/vlsa-aegis/safelibero MUJOCO_GL=egl
for suite in safelibero_spatial safelibero_object safelibero_goal safelibero_long; do
  for level in I II; do
    python libero/policy_eval.py --suite $suite --level $level --tasks 0 1 2 3 --out results/libero/l0_policy --videos 1 2>&1 | grep -E '^\{|Traceback|Error' >> results/libero/l0_policy.log
    echo "$suite $level done $(date)" >> results/libero/l0_policy.log
  done
done
