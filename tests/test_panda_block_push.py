import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from systems.panda_block_push import PUSH_XML


def test_push_scene_uses_panda_one_upright_free_block_and_visual_goal():
    assert '<include file="panda.xml"/>' in PUSH_XML
    assert PUSH_XML.count('type="free"') == 1
    assert 'name="push_block"' in PUSH_XML
    assert 'name="goal_geom"' in PUSH_XML
    assert 'contype="0" conaffinity="0"' in PUSH_XML
    assert '0.535 0 0.463 1 0 0 0' in PUSH_XML


def test_push_scene_preserves_existing_panda_camera_convention():
    assert 'name="attachment_site"' not in PUSH_XML  # supplied by panda.xml
    assert 'name="cam_fixed"' in PUSH_XML
