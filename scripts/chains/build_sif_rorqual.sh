set -e
module load apptainer
export APPTAINER_CACHEDIR=$SCRATCH/.apptainer_cache APPTAINER_TMPDIR=$SCRATCH/.apptainer_tmp
mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR
apptainer build --force torch290.sif docker://pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime
apptainer exec --cleanenv torch290.sif python -m venv --system-site-packages pyenv
apptainer exec --cleanenv torch290.sif pyenv/bin/pip install -q --no-cache-dir numpy==2.2.6 scipy==1.15.3 mujoco==3.13.0 diffusers==0.41.0 dill==0.4.1 einops==0.8.2 hydra-core==1.3.7 omegaconf==2.3.1 zarr==2.18.3 "numcodecs<0.16" imageio-ffmpeg==0.6.0 imageio pillow==12.3.0 scikit-learn==1.7.2 timm==1.0.30 numba==0.68.0 opencv-python-headless==4.12.0.88 torchvision==0.24.0
apptainer exec --cleanenv torch290.sif pyenv/bin/python -c "import torch, mujoco, diffusers, torchvision; print(\"ENV OK\", torch.__version__, mujoco.__version__, torchvision.__version__)"
