cd /home/corey/wksp/stable_lyap_filters
SUB() { echo "cd projects/def-soojeon/ali313/slf/repo && sbatch --parsable -J s2s --array=$1 --export=ALL,NT=100,N=400,OFF=30000,TAG=_s2s,ARMS=record,EXTRA_ENV='MASS_ONLY=0.3' cluster/paired_array.sbatch"; }
J=$(timeout 60 ssh -o BatchMode=yes nibi "$(SUB 0)" | tail -1); echo trial $J
until timeout 30 ssh -o BatchMode=yes nibi "ls projects/def-soojeon/ali313/slf/repo/results/program_b/policy_rollouts/paired_record_s2s/*.json" >/dev/null 2>&1; do
  st=$(timeout 30 ssh -o BatchMode=yes nibi "sacct -j $J -X -n -o State | head -1" 2>/dev/null)
  case "$st" in *FAIL*|*CANCEL*|*TIMEOUT*) echo "trial failed: $st"; exit 1;; esac
  sleep 120; done
echo trial ok $(date); timeout 60 ssh -o BatchMode=yes nibi "$(SUB 1-99)" | tail -1
