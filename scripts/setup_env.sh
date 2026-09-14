#!/usr/bin/env bash
# Create the local development environment. Reusing system packages avoids downloading another
# multi-gigabyte CUDA PyTorch wheel when the host already has a working cu128 installation.
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
LYAP_UV_CACHE=${UV_CACHE_DIR:-/tmp/stable-lyap-uv-cache}

command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }
if [[ ! -x .venv/bin/python ]]; then
  UV_CACHE_DIR="$LYAP_UV_CACHE" uv venv --python "$PYTHON_BIN" --system-site-packages .venv
fi
if ! .venv/bin/python -c 'import lmdb, einops, hydra, omegaconf, imageio, imageio_ffmpeg, scipy, sklearn' \
     >/dev/null 2>&1; then
  UV_CACHE_DIR="$LYAP_UV_CACHE" uv pip install --python .venv/bin/python \
    lmdb einops hydra-core omegaconf imageio imageio-ffmpeg scipy 'scikit-learn>=1.3'
fi

.venv/bin/python -c 'import torch, torchvision; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'

echo "Environment ready. Activate with: source .venv/bin/activate"
echo "Then verify with: python -m pytest tests/ -q"
