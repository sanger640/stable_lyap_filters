"""dino_wm's model code, vendored so this repo is self-contained for the Jenga work.

WHY THIS EXISTS. The Jenga world-model checkpoint (`model_latest_single.pth`) was saved with whole
`nn.Module` objects rather than a state_dict, so unpickling it REQUIRES these classes to be
importable at the module paths they were pickled under:

    models.vit.ViTPredictor / Transformer / Attention / FeedForward
    models.proprio.ProprioceptiveEmbedding
    models.vqvae.VQVAE / Decoder / Quantize / ResBlock

`install_import_shim()` registers this package under those names so the checkpoint loads without a
dino_wm checkout on the path.

These files are copied VERBATIM and must not be edited -- they are the same code the real world
model was trained with, and the point of using them is that it IS the same code. To re-sync:

    for f in vit proprio vqvae visual_world_model dino; do
        cp ~/wksp/dino_wm/models/$f.py src/models/dinowm/$f.py   # then restore the header line
    done
"""
import sys
import types


def install_import_shim():
    """Make `import models.vit` (etc.) resolve here, so dino_wm checkpoints unpickle."""
    from . import vit, proprio, vqvae                              # noqa: F401
    pkg = sys.modules.setdefault("models", types.ModuleType("models"))
    pkg.__path__ = []                                              # mark as a package
    for name in ("vit", "proprio", "vqvae"):
        mod = sys.modules[f"{__name__}.{name}"]
        sys.modules[f"models.{name}"] = mod
        setattr(pkg, name, mod)
    return pkg
