cd /home/corey/wksp/stable_lyap_filters
L=results/program_b/cbf_final.log
until grep -aq "chain done" results/program_b/closedloop_train.log; do sleep 60; done
grep -aq "Traceback" <(awk '/^restart-remote/{f=1;next} f' results/program_b/closedloop_train.log) && { echo "training chain failed" >> $L; exit 1; }
python3 eval/cbf_final_test.py decide >> $L 2>&1 || exit 1
WM=$(python3 -c "import json;print(json.load(open('results/jenga/cbf_final_choice.json'))['world_model'])")
NL=$(python3 -c "import json;print(json.load(open('results/jenga/cbf_final_choice.json'))['n_look'])")
if [ "$WM" = "threeblock_v3" ]; then unset CBF_WM; else export CBF_WM=$WM; fi
sed -i "s/\"N_LOOK\": [0-9]*/\"N_LOOK\": $NL/" eval/cbf_filter.py
python3 eval/cbf_calibrate.py >> $L 2>&1 || exit 1
rsync -a --relative eval results/jenga/cbf_calibration.json results/jenga/cbf_final_choice.json gpu5090:wksp/stable_lyap_filters/ >> $L 2>&1
RWM=""; [ "$WM" != "threeblock_v3" ] && RWM="CBF_WM=\$HOME/wksp/stable_lyap_filters/results/program_b/closedloop_v1/model/model_US.pt"
echo "test start $(date) WM=$WM N_LOOK=$NL" >> $L
ssh gpu5090 "cd ~/wksp/stable_lyap_filters && (setsid nohup bash -c 'export DIFFUSION_POLICY_DIR=\$HOME/wksp/diffusion_policy PY=.venv/bin/python MUJOCO_GL=egl HARD_OFF=1 EXEC_NOISE_SCALE=2 EP_OFFSET=1000 PAIRED_TAG=_test2x $RWM && bash eval/run_paired_parallel.sh cbf 300 4 6 7' > /dev/null 2>&1 < /dev/null &)" < /dev/null
HARD_OFF=1 EXEC_NOISE_SCALE=2 EP_OFFSET=1000 PAIRED_TAG=_test2x bash eval/run_paired_parallel.sh cbf 300 0 3 7
while ssh gpu5090 'pgrep -f "eval/policy_rollout.py paired cbf" >/dev/null && echo busy' < /dev/null | grep -q busy; do sleep 60; done
scp -q gpu5090:wksp/stable_lyap_filters/results/jenga/paired_cbf_test2x.jsonl /tmp/remote_cbf_test2x.jsonl
cat /tmp/remote_cbf_test2x.jsonl >> results/jenga/paired_cbf_test2x.jsonl
rsync -a gpu5090:wksp/stable_lyap_filters/results/program_b/policy_rollouts/paired_cbf_test2x/ results/program_b/policy_rollouts/paired_cbf_test2x/ >> $L 2>&1
python3 eval/cbf_final_test.py analyse >> $L 2>&1
echo "final done $(date)" >> $L
