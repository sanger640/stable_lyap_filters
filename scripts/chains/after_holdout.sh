cd /home/corey/wksp/stable_lyap_filters
R=results/program_b/bench2/report_holdout_g_g_cem.json
until [ -f $R ] || ! pgrep -f g_holdout.sh >/dev/null; do sleep 60; done
sleep 5
if [ -f $R ] && python3 -c "import json,sys; sys.exit(0 if json.load(open('$R'))['gate_A2']['PASS'] else 1)"; then
  echo "holdout PASS -> launching Stage 4 $(date)"; bash eval/stage4_local.sh 5
  echo "Stage 4 finished $(date)"
else
  echo "holdout did not pass (or failed to run) -> Stage 4 NOT launched $(date)"
fi
