cd /home/corey/wksp/stable_lyap_filters
SP=scripts/chains
R=results/program_b/bench2/report_holdout_g_g_cem.json
until [ -f $R ] || ! pgrep -f g_holdout.sh >/dev/null; do sleep 60; done
sleep 5
if [ -f $R ] && python3 -c "import json,sys; sys.exit(0 if json.load(open('$R'))['gate_A2']['PASS'] else 1)"; then
  if [ -f $SP/smoke_ok ]; then
    echo "holdout PASS -> submitting Stage 4 on Nibi $(date)"
    ssh nibi 'cd ~/projects/def-soojeon/ali313/slf/repo && sbatch --job-name=s4 --array=0-37 --export=ALL,NT=300,T0=0,TMAX=299,GROUP=8,N=300,OFF=14000,TAG=_s4,ARMS="record cbfmsoil cbfh cbfg" cluster/paired_group.sbatch'
  else
    echo "holdout PASS but Nibi smoke not confirmed -> NOT submitted $(date)"
  fi
else
  echo "holdout did not pass (or failed to run) -> Stage 4 NOT launched $(date)"
fi
