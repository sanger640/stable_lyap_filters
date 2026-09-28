from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval")); sys.path.insert(0, str(ROOT / "src"))

from panda_push_optical_flow import detect_points, track_geometry


def corner_frame(shift_x=0):
    frame = np.zeros((80, 80, 3), np.uint8)
    cv2.rectangle(frame, (20 + shift_x, 20), (50 + shift_x, 50), (255, 255, 255), -1)
    return frame


def test_detector_finds_generic_corners_without_a_mask():
    points = detect_points(corner_frame(), max_corners=16)
    assert len(points) == 4


def test_static_frames_have_zero_velocity_and_visible_tracks():
    start = corner_frame()
    points = detect_points(start, max_corners=16)
    geometry = track_geometry(start, np.stack([start, start]), points).reshape(2, len(points), 5)
    assert np.allclose(geometry[..., 2:4], 0., atol=1e-5)
    assert np.all(geometry[..., 4] == 1.)


def test_translation_is_tracked_at_pixel_resolution():
    start = corner_frame()
    points = detect_points(start, max_corners=16)
    frames = np.stack([corner_frame(1), corner_frame(2)])
    geometry = track_geometry(start, frames, points).reshape(2, len(points), 5)
    expected = 2. / 79.
    assert np.allclose(geometry[..., 2], expected, atol=2e-3)
    assert np.allclose(geometry[..., 3], 0., atol=2e-3)
    assert np.all(geometry[..., 4] == 1.)


def test_lost_tracks_carry_position_and_stay_invisible():
    start = corner_frame()
    points = detect_points(start, max_corners=16)
    blank = np.zeros_like(start)
    geometry = track_geometry(start, np.stack([blank, start]), points).reshape(2, len(points), 5)
    assert np.all(geometry[..., 4] == 0.)
    assert np.allclose(geometry[1, :, :2], geometry[0, :, :2])
    assert np.allclose(geometry[..., 2:4], 0.)
