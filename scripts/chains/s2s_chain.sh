set -x
cd /home/corey/wksp/stable_lyap_filters
PY=/home/corey/wksp/dp_venv/bin/python; P=results/program_b/policy_rollouts
n() { timeout 30 ssh -o BatchMode=yes nibi "squeue -u ali313 -h -n s2s | wc -l" 2>/dev/null || echo 1; }
until grep -q "trial ok" results/jenga/s2s_launch.log && [ "$(n)" = "0" ]; do grep -q "trial failed" results/jenga/s2s_launch.log && exit 1; sleep 300; done
mkdir -p $P/paired_record_s2s
rsync -a --exclude '*.tmp' nibi:projects/def-soojeon/ali313/slf/repo/$P/paired_record_s2s/ $P/paired_record_s2s/
echo episodes $(ls $P/paired_record_s2s/ep*.json | wc -l)
python3 cluster/merge_paired.py paired_record_s2s
# topple labels from the run logs only (the audit's replay forces normal physics); evaluation episodes only
python3 - <<'PY'
import json
rows=[json.loads(l) for l in open("results/jenga/paired_record_s2s.jsonl")]
ev=[{"ep":r["episode"],"logged_neighbour_toppled":r["neighbour_toppled"]} for r in rows if r["episode"]>=30100]
json.dump({"paired_record_s2s":{"source":"run logs (no replay)","rows":ev}},open("results/jenga/outcome_audit_s2s.json","w"),indent=1)
print("eval episodes",len(ev),"topples",sum(r["logged_neighbour_toppled"] for r in ev))
PY
# GPU: wait for the monitor replication (single DINOv2 process) before the data pool
until [ -s results/jenga/deep_dive/spread_gate_scl.json ]; do sleep 120; done
CL_TAG=paired_record_s2s CL_OUT=results/program_b/s2s_v1 CL_S2S=30100 CL_N_TRAIN=400 CL_N_VAL=60 MASS_ONLY=0.3 CL_WORKERS=4 \
  $PY eval/closedloop_data.py > results/jenga/deep_dive/s2s_data.log 2>&1 || exit 1
$PY eval/s2s.py train > results/jenga/deep_dive/s2s_train.log 2>&1 || exit 1
CBF_WM=$PWD/results/program_b/s2s_v1/model/model_S2S.pt CBF_CALIB_OUT=results/jenga/cbf_calibration_S2S.json $PY eval/cbf_calibrate.py > results/jenga/deep_dive/s2s_calib.log 2>&1 || exit 1
$PY eval/s2s.py error > results/jenga/deep_dive/s2s_error.log 2>&1 || exit 1
MC_RUN=paired_record_s2s MC_TAG=_s2s WM_COMPILE=1 $PY eval/margin_check.py run 0 1 > results/jenga/deep_dive/margin_s2s.log 2>&1 || exit 1
MC_RUN=paired_record_s2s MC_TAG=_s2s MC_OUT=s2s_adapt WM_COMPILE=1 CBF_WM=$PWD/results/program_b/s2s_v1/model/model_S2S.pt CBF_CALIB=results/jenga/cbf_calibration_S2S.json \
  $PY eval/margin_check.py run 0 1 > results/jenga/deep_dive/margin_s2s_adapt.log 2>&1 || exit 1
python3 eval/s2s_gate.py
echo done $(date)
