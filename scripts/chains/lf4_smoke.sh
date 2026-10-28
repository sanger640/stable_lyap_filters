cd /home/corey/wksp/stable_lyap_filters
for arm in lfg030 lfm030 lfc030; do
  MUJOCO_GL=egl PER_EPISODE_SEEDS=1 PAIRED_PER_EP=1 PAIRED_TAG=_lfsmoke4 EP_OFFSET=99000 EXEC_NOISE_SCALE=2 HARD_OFF=1 \
  CBF_WM=$PWD/results/program_b/oil_v1/model/model_OIL.pt CBF_CALIB=results/jenga/cbf_calibration_OIL.json PYTHONUNBUFFERED=1 \
  /home/corey/wksp/dp_venv/bin/python eval/policy_rollout.py paired $arm 2 2>&1 | grep -vE "^\s*$" | tail -4
  /home/corey/wksp/dp_venv/bin/python - $arm <<'PY'
import json,glob,sys
for f in sorted(glob.glob(f"results/program_b/policy_rollouts/paired_{sys.argv[1]}_lfsmoke4/ep*.json")):
    L=json.load(open(f))["log"]; g=lambda k: sum(c.get(k,0) for c in L)
    print(f.split("/")[-1], "chunks",len(L),"steps",g("lsf_steps"),"ovr",g("lsf_overrides"),"edit_mm",round(g("lsf_edit_mm"),1),"iters",g("lsf_edit_iters"),"sec",round(g("seconds"),1),
          "vmin",round(min(c.get("lsf_vmin",9) for c in L),3),"vafter",round(min(c.get("lsf_vafter_min",9) for c in L),3))
PY
done
