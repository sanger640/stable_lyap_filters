from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/"eval")); sys.path.insert(0, str(ROOT/"src"))

import panda_push_visual_regions as regions


class FakeEncoder:
    def forward(self, image):
        batch = len(image)
        return torch.ones((batch, 256, 384), device=image.device)


def test_protocol_separates_deployable_and_oracle_readouts():
    assert regions.protocol("motion_pca45")["object_identity_or_segmentation"] is False
    assert regions.protocol("block_pca45")["object_identity_or_segmentation"] is True


def test_weighted_feature_shapes_for_both_modes():
    frames = np.zeros((2, 3, 224, 224, 3), np.uint8)
    frames[:, :, 50:80, 70:100] = 255
    start = np.zeros((224, 224, 3), np.uint8)
    masks = np.zeros((2, 3, 224, 224), bool); masks[:, :, 50:80, 70:100] = True
    encoder = FakeEncoder()
    motion = regions.weighted_dino_features(encoder, frames, "motion_pca45", "cpu", 4,
                                            start=start)
    block = regions.weighted_dino_features(encoder, frames, "block_pca45", "cpu", 4,
                                           block_masks=masks)
    assert motion.shape == block.shape == (2, 3, 16*384)
    assert np.isfinite(motion).all() and np.isfinite(block).all()
    assert np.any(motion) and np.any(block)
