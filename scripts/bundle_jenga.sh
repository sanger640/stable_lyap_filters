#!/usr/bin/env bash
# Bundle everything the Jenga work needs that is NOT in git, for transfer to another machine.
#
#   ./scripts/bundle_jenga.sh            # core bundle (~1.3 GB): offline monitor work
#   ./scripts/bundle_jenga.sh --live     # adds the diffusion policy (~2.1 GB): live rollout
#
# On the far side: git clone the repo, then untar this over it.
set -euo pipefail
W=${WKSP:-$HOME/wksp}
OUT=${1:-jenga_bundle.tar}
LIVE=0; [[ "${1:-}" == "--live" ]] && { LIVE=1; OUT=jenga_bundle_live.tar; }

STAGE=$(mktemp -d); trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/data/jenga" "$STAGE/vendor"

echo "staging world model (stripped, 136 MB) ..."
cp data/jenga/world_model.pt "$STAGE/data/jenga/"
cp data/jenga/labels_noise*.json "$STAGE/data/jenga/"

echo "staging eval LMDB (1.1 GB) ..."
cp -r "$W/panda_express/tasks/jenga_noise_50/jenga_single_100.lmdb" "$STAGE/data/jenga/"

echo "staging panda_express sim (for rollouts) ..."
tar -C "$W" -cf "$STAGE/vendor/panda_express_sim.tar" \
    panda_express/sim.py panda_express/tasks/jenga_mujoco/ 2>/dev/null || \
    echo "  (sim assets not found -- skipping, only needed for live rollout)"

if [[ $LIVE == 1 ]]; then
  echo "staging diffusion policy (815 MB) ..."
  P="$W/diffusion_policy/data/outputs/2026.08.11/21.59.04_train_franka_dual_jenga_jenga_image_dual"
  mkdir -p "$STAGE/data/jenga/policy"
  cp "$P/checkpoints/latest.ckpt" "$STAGE/data/jenga/policy/"
  cp -r "$P/.hydra" "$STAGE/data/jenga/policy/" 2>/dev/null || true
fi

tar -C "$STAGE" -cf "$OUT" .
echo
echo "-> $OUT  ($(du -h "$OUT" | cut -f1))"
echo
echo "On the target machine:"
echo "   git clone git@github.com:sanger640/stable_lyap_filters.git && cd stable_lyap_filters"
echo "   tar -xf /path/to/$OUT"
echo "   python -m pytest tests/ -q          # 23 tests"
echo "   python eval/jenga_basins.py --lmdb data/jenga/jenga_single_100.lmdb \\"
echo "                               --labels data/jenga/labels_noise100.json"
