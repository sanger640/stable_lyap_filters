cd /home/corey/wksp/stable_lyap_filters
export MC_RUN=paired_record_scl MC_TAG=_scl WM_COMPILE=1 MUJOCO_GL=egl
/home/corey/wksp/dp_venv/bin/python eval/margin_check.py run 0 1 > results/jenga/deep_dive/margin_scl.log 2>&1 || exit 1
/home/corey/wksp/dp_venv/bin/python eval/spread_check.py run > results/jenga/deep_dive/spread_scl.log 2>&1 || exit 1
python3 eval/spread_gate.py > results/jenga/deep_dive/spread_gate_scl.log 2>&1
python3 -c "import json;d=json.load(open('results/jenga/deep_dive/spread_gate_scl.json'));print(json.dumps({k:d[k] for k in ('decisions','topple_episodes','fork_2x_rate','fork_3x_rate','fork_4x_rate')},indent=1))"
