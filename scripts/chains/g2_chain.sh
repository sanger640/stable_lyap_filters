cd /home/corey/wksp/stable_lyap_filters
export OPT2_CERT=OIL+ENS OPT3_GRIP=warp CBF_WM=$PWD/results/program_b/oil_v1/model/model_OIL.pt CBF_CALIB=results/jenga/cbf_calibration_OIL.json
pass() { python3 -c "import json,sys; sys.exit(0 if json.load(open('results/program_b/bench2/report_$1_g_g2_cem.json'))['gate_A2']['PASS'] else 1)"; }
for set in dev holdout; do
  python3 eval/opt3.py run cem $set g2_cem && python3 eval/opt3.py truth $set g2_cem && python3 eval/opt3.py report $set g2_cem || { echo "G2 $set run failed"; exit 1; }
  pass $set || { echo "G2 $set gate FAILED -> stop $(date)"; exit 0; }
  echo "G2 $set gate PASS $(date)"
done
R=nibi:projects/def-soojeon/ali313/slf/repo
for f in eval/cbf_filter.py eval/policy_rollout.py eval/opt3.py eval/opt2.py eval/bench2.py; do rsync -a $f $R/$f || { echo "rsync failed"; exit 1; }; done
ssh nibi 'cd ~/projects/def-soojeon/ali313/slf/repo && sbatch --job-name=s4b --array=0-37 --export=ALL,NT=300,T0=0,TMAX=299,GROUP=8,N=300,OFF=14300,TAG=_s4b,ARMS="record cbfmsoil cbfh2 cbfg cbfg2" cluster/paired_group.sbatch' && echo "Stage 4b submitted $(date)"
