cd /home/corey/wksp/stable_lyap_filters
R=results/program_b/cbf_qp/cbfopt_report_sep50.json
until [ -f $R ]; do sleep 30; done
ok=$(python3 -c "import json;d=json.load(open('$R'));print(int(d['real_risk']['cbf_opt']<=0.1242))")
echo "offline gate: $ok"; [ "$ok" = 1 ] || exit 0
ssh nibi "cd ~/projects/def-soojeon/ali313/slf/repo && sbatch --job-name=dev9 --array=0-9 --export=ALL,NT=150,T0=75,TMAX=149,GROUP=8,N=150,OFF=12000,TAG=_dev9,ARMS='record cbfmos' cluster/paired_group.sbatch" < /dev/null
sleep 60
until timeout 60 ssh nibi 'squeue -u ali313 -h -n dev9 | wc -l' < /dev/null | grep -qx 0; do sleep 60; done
for arm in record cbfmos; do
  d=results/program_b/policy_rollouts/paired_${arm}_dev9; mkdir -p $d
  rsync -a --include='ep*.json' --exclude='*' nibi:projects/def-soojeon/ali313/slf/repo/$d/ $d/
done
python3 cluster/merge_paired.py paired_record_dev9 paired_cbfmos_dev9
python3 eval/cbf_dev9.py
echo GATE_DONE
