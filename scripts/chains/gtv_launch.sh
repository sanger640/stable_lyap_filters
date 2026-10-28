cd /home/corey/wksp/stable_lyap_filters
rsync -a eval/cbf_filter.py eval/policy_rollout.py eval/steer_analysis.py fir:/scratch/ali313/slf/repo/eval/ && rsync -a results/jenga/gtv_protocol.json fir:/scratch/ali313/slf/repo/results/jenga/ || exit 1
SUB() { echo "cd /scratch/ali313/slf/repo && md5sum eval/step3_world_model.py && sbatch --parsable -J gtv --array=$1 --export=ALL,SLF_ROOT=/scratch/ali313/slf/repo,NT=200,N=600,OFF=23000,TAG=_gtv,ARMS='cbfg3m cbfgtv3m',CBF_WM=/scratch/ali313/slf/repo/results/program_b/oil_v1/model/model_OIL.pt,EXTRA_ENV='CBF_CALIB=results/jenga/cbf_calibration_OIL.json;WM_COMPILE=1;CC=/scratch/ali313/slf/ccwrap/cc' cluster/paired_array.sbatch"; }
J=$(timeout 60 ssh -o BatchMode=yes fir "$(SUB 0)" | tail -1); echo trial $J
until timeout 30 ssh -o BatchMode=yes fir "ls /scratch/ali313/slf/repo/results/program_b/policy_rollouts/paired_cbfgtv3m_gtv/*.json" >/dev/null 2>&1; do
  st=$(timeout 30 ssh -o BatchMode=yes fir "sacct -j $J -X -n -o State | head -1" 2>/dev/null)
  case "$st" in *FAIL*|*CANCEL*|*TIMEOUT*) echo "trial failed: $st"; exit 1;; esac
  sleep 120; done
echo trial ok $(date); timeout 60 ssh -o BatchMode=yes fir "$(SUB 1-199)" | tail -1
