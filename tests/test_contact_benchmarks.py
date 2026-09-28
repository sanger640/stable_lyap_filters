import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from systems.contact_benchmarks import INSERTION_XML, PUSHING_XML


def test_pushing_scene_has_free_object_goal_and_edge():
    assert '<freejoint name="object_free"' in PUSHING_XML
    assert 'name="goal"' in PUSHING_XML
    assert 'name="table"' in PUSHING_XML


def test_insertion_scene_has_tight_socket_and_four_dof_tool():
    assert all(name in INSERTION_XML for name in (
        "socket_left", "socket_right", "socket_front", "socket_back"))
    assert all(name in INSERTION_XML for name in ("peg_x", "peg_y", "peg_z", "peg_yaw"))
