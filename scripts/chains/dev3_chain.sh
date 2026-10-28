cd /home/corey/wksp/stable_lyap_filters
L=results/program_b/cbf_dev3.log
echo "dev3 start $(date)" >> $L
rsync -a --relative eval results/jenga/cbf_calibration.json gpu5090:wksp/stable_lyap_filters/ >> $L 2>&1
COMMON="HARD_OFF=1 EXEC_NOISE_SCALE=2 EP_OFFSET=2000 PER_EPISODE_SEEDS=1"
LWM=$PWD/results/program_b/closedloop_v1/model/model_US.pt
RWM='$HOME/wksp/stable_lyap_filters/results/program_b/closedloop_v1/model/model_US.pt'
run_arm() {   # mode tag trigger select
  local mode=$1 tag=$2 trig=$3 sel=$4 mk=results/program_b/dev3$2.done
  echo "arm $mode$tag start $(date)" >> $L
  ssh gpu5090 "cd ~/wksp/stable_lyap_filters && rm -f $mk && (setsid nohup bash -c 'export DIFFUSION_POLICY_DIR=\$HOME/wksp/diffusion_policy PY=.venv/bin/python MUJOCO_GL=egl $COMMON PAIRED_TAG=$tag CBF_WM=$RWM CBF_TRIGGER=$trig CBF_SELECT=$sel && bash eval/run_paired_parallel.sh $mode 150 4 6 7; touch $mk' > /dev/null 2>&1 < /dev/null &)" < /dev/null
  env $COMMON PAIRED_TAG=$tag CBF_WM=$LWM CBF_TRIGGER=$trig CBF_SELECT=$sel bash eval/run_paired_parallel.sh $mode 150 0 3 7
  until ssh gpu5090 "test -f ~/wksp/stable_lyap_filters/$mk" < /dev/null; do sleep 60; done
  scp -q gpu5090:wksp/stable_lyap_filters/results/jenga/paired_$mode$tag.jsonl scripts/chains/remote_$mode$tag.jsonl
  python3 - "$mode$tag" <<'EOF'
import json, sys
t = sys.argv[1]; p = f"results/jenga/paired_{t}.jsonl"
rows = {}
for f in (p, f"scripts/chains/remote_{t}.jsonl"):
    try:
        for l in open(f):
            r = json.loads(l); rows.setdefault(r["episode"], r)
    except FileNotFoundError:
        pass
open(p, "w").write("".join(json.dumps(rows[e]) + "\n" for e in sorted(rows)))
print("merged", t, len(rows))
EOF
  echo "arm $mode$tag done $(date) $(wc -l < results/jenga/paired_$mode$tag.jsonl)" >> $L
  python3 eval/cbf_dev3.py > /dev/null 2>&1
}
run_arm record _dev3 both closest
run_arm cbf _dev3A both closest
run_arm cbf _dev3B1 barrier closest
run_arm cbf _dev3B2 barrier margin
python3 eval/cbf_dev3.py >> $L 2>&1
echo "dev3 done $(date)" >> $L
