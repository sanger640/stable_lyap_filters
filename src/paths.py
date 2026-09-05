"""
Single place where the external repo locations live.

This repo holds the safety-filter logic and evaluation; the world model and the simulator
live in two sibling repos. Override any of these with environment variables rather than
editing the file, e.g.

    export DINO_WM_DIR=/path/to/dino_wm
    export PANDA_EXPRESS_DIR=/path/to/panda_express
    export DIFFUSION_POLICY_DIR=/path/to/diffusion_policy
    export DINO_WM_CKPT=/path/to/model_latest_single.pth

Call `add_external_paths()` before importing anything from those repos.
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DINO_WM_DIR = Path(os.environ.get("DINO_WM_DIR", "/home/sanger/wksp/dino_wm"))
PANDA_EXPRESS_DIR = Path(os.environ.get("PANDA_EXPRESS_DIR", "/home/sanger/wksp/panda_express"))
DIFFUSION_POLICY_DIR = Path(os.environ.get("DIFFUSION_POLICY_DIR",
                                           "/home/sanger/wksp/diffusion_policy"))
DINO_WM_CKPT = Path(os.environ.get(
    "DINO_WM_CKPT", str(DINO_WM_DIR / "outputs" / "model_latest_single.pth")))

# expert teleop episodes used by the replay experiment
EPISODES_DIR = Path(os.environ.get(
    "JENGA_EPISODES_DIR", str(PANDA_EXPRESS_DIR / "tasks/jenga_mujoco/episodes")))


def add_external_paths(*, dino_wm=True, panda_express=False, diffusion_policy=False):
    """Put the requested sibling repos on sys.path, plus this repo's own src/."""
    for p in [REPO_ROOT / "src",
              DINO_WM_DIR if dino_wm else None,
              PANDA_EXPRESS_DIR if panda_express else None,
              DIFFUSION_POLICY_DIR if diffusion_policy else None]:
        if p is not None and str(p) not in sys.path:
            sys.path.insert(0, str(p))


def enter_sim_dir():
    """chdir into panda_express before importing `sim`.

    Not optional: sim.py loads its MuJoCo model with a path relative to the repo root
    ("franka_emika_panda/panda_jenga_setup.xml"), so importing it from anywhere else raises
    "ParseXML: Error opening file". Resolve any output paths to absolute BEFORE calling this."""
    os.chdir(PANDA_EXPRESS_DIR)


def check():
    """Fail loudly and early if an external dependency is missing, rather than deep inside a
    run. This project has repeatedly lost time to failures that surface late."""
    missing = [str(p) for p in (DINO_WM_DIR, PANDA_EXPRESS_DIR, DINO_WM_CKPT) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "missing external dependencies:\n  " + "\n  ".join(missing) +
            "\nSet DINO_WM_DIR / PANDA_EXPRESS_DIR / DINO_WM_CKPT (see src/paths.py).")
