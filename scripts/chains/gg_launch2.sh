cd /home/corey/wksp/stable_lyap_filters
SUB() { echo "cd /scratch/ali313/slf/repo && sbatch --parsable -J gg --array=$1 --export=ALL,SLF_ROOT=/scratch/ali313/slf/repo,NT=150,N=300,OFF=20100,TAG=_gg,ARMS='record cbfg3m cbfgg3m',CBF_WM=/scratch/ali313/slf/repo/results/program_b/oil_v1/model/model_OIL.pt,EXTRA_ENV='CBF_CALIB=results/jenga/cbf_calibration_OIL.json' cluster/paired_array.sbatch"; }
J=$(timeout 60 ssh -o BatchMode=yes fir "$(SUB 0)"); echo trial $J
until timeout 30 ssh -o BatchMode=yes fir "ls /scratch/ali313/slf/repo/results/program_b/policy_rollouts/paired_cbfgg3m_gg/*.json" >/dev/null 2>&1; do
  st=$(timeout 30 ssh -o BatchMode=yes fir "sacct -j $J -X -n -o State | head -1" 2>/dev/null)
  case "$st" in *FAIL*|*CANCEL*|*TIMEOUT*) echo "trial failed: $st"; exit 1;; esac
  sleep 120; done
echo trial ok; timeout 60 ssh -o BatchMode=yes fir "$(SUB 1-149)"
