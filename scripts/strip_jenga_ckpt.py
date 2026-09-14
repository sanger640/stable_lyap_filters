"""Strip the Jenga world-model checkpoint to what inference needs.

The original is 360 MB and holds two optimizer states plus whole nn.Module objects. The optimizers
are useless for a monitor and drag in an `accelerate` dependency at unpickle time. This writes a
smaller checkpoint holding only the four modules, loadable with the vendored code in
src/models/dinowm and nothing else.
"""
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "models"))
from dinowm import install_import_shim                             # noqa: E402

SRC = Path(sys.argv[1] if len(sys.argv) > 1
           else "/home/sanger/wksp/dino_wm/outputs/model_latest_single.pth")
DST = ROOT / "data" / "jenga" / "world_model.pt"

install_import_shim()
c = torch.load(SRC, map_location="cpu", weights_only=False)
keep = {k: c[k] for k in ("predictor", "action_encoder", "proprio_encoder", "decoder")}
keep["epoch"] = c["epoch"]
keep["_provenance"] = {"source": str(SRC), "epoch": c["epoch"],
                       "note": "optimizers stripped; load via src/models/dinowm install_import_shim()"}
DST.parent.mkdir(parents=True, exist_ok=True)
torch.save(keep, DST)
print(f"{SRC.stat().st_size/1e6:.0f} MB -> {DST.stat().st_size/1e6:.0f} MB   {DST.relative_to(ROOT)}")
for k in ("predictor", "decoder"):
    print(f"   {k:<16} {sum(p.numel() for p in keep[k].parameters())/1e6:6.2f}M params")
