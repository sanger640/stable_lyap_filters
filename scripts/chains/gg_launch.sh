cd /home/corey/wksp/stable_lyap_filters
until [ "$(timeout 30 ssh -o BatchMode=yes fir "squeue -u ali313 -h -n tr | wc -l" 2>/dev/null || echo 1)" = "0" ]; do sleep 120; done
rsync -a -e "ssh -o BatchMode=yes" eval/policy_rollout.py eval/opt3.py fir:/scratch/ali313/slf/repo/eval/
SUB="cd /scratch/ali313/slf/repo && sbatch --parsable -J gg --export=ALL,SLF_ROOT=/scratch/ali313/slf/repo,NT=150,N=300,OFF=20100,TAG=_gg,ARMS='record cbfg3m cbfgg3m',CBF_WM=/scratch/ali313/slf/repo/results/program_b/oil_v1/model/model_OIL.pt,EXTRA_ENV='CBF_CALIB=results/jenga/cbf_calibration_OIL.json' cluster/paired_array.sbatch"
J=$(timeout 60 ssh -o BatchMode=yes fir "$SUB --array=0"); echo trial $J
until timeout 30 ssh -o BatchMode=yes fir "ls /scratch/ali313/slf/repo/results/program_b/policy_rollouts/paired_cbfgg3m_gg/*.json" >/dev/null 2>&1; do
  st=$(timeout 30 ssh -o BatchMode=yes fir "sacct -j $J -X -n -o State | head -1" 2>/dev/null)
  case "$st" in *FAIL*|*CANCEL*|*TIMEOUT*) echo "trial failed: $st"; exit 1;; esac
  sleep 120; done
echo trial ok; timeout 60 ssh -o BatchMode=yes fir "$SUB --array=1-149"
