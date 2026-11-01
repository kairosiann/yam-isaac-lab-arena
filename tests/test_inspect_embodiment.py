import json
import socket
import subprocess
import sys
import tempfile
from importlib.metadata import distribution
from pathlib import Path

import numpy as np
import pytest
from inspect_robots.compat import check_compatibility
from inspect_robots.conformance import assert_embodiment_conformant
from inspect_robots.embodiment import PRIVILEGED_SUCCESS, RESETTABLE, SEEDABLE
from inspect_robots_yam.config import DEFAULT_CAMERAS
from inspect_robots_yam.embodiment import YAMEmbodiment

from yam_arena import YAM_URDF, inspect_embodiment
from yam_arena.constants import ARM_JOINTS, CONTROL_HZ
from yam_arena.inspect_embodiment import YamArenaEmbodiment, window_display, yam_arena_embodiment
from yam_arena.urdf import parse_joints

SIM_MODULES = {"isaacsim", "isaaclab", "isaaclab_arena", "isaaclab_physx", "omni", "carb", "pxr", "torch"}
RIG_POLICIES = [ep for ep in distribution("inspect-robots-yam").entry_points if ep.group == "inspect_robots.policies"]


def test_the_entry_point_constructs_it_without_isaac():
    script = (
        "import json, sys\n"
        "from importlib.metadata import entry_points\n"
        "(ep,) = entry_points(group='inspect_robots.embodiments', name='yam_arena')\n"
        "ep.load()()\n"
        "print(json.dumps(sorted({m.split('.')[0] for m in sys.modules})))\n"
    )
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120, check=True)
    assert SIM_MODULES.isdisjoint(json.loads(proc.stdout))


def test_info_declares_a_seedable_resettable_simulator():
    info = YamArenaEmbodiment().info
    assert (info.name, info.is_simulated, info.control_hz) == ("yam_arena", True, CONTROL_HZ)
    assert info.capabilities == {SEEDABLE, RESETTABLE}
    assert_embodiment_conformant(info)


def test_actions_are_bounded_by_the_urdf_limits_and_normalized_grippers():
    joints = parse_joints(YAM_URDF)
    low = [joints[n].lower for n in ARM_JOINTS] + [0.0]
    high = [joints[n].upper for n in ARM_JOINTS] + [1.0]
    space = YamArenaEmbodiment().info.action_space
    np.testing.assert_array_equal(space.low, low * 2)
    np.testing.assert_array_equal(space.high, high * 2)


def test_it_has_the_real_rigs_contract():
    sim, real = YamArenaEmbodiment().info, YAMEmbodiment().info
    assert sim.observation_space == real.observation_space
    assert sim.action_space.shape == real.action_space.shape
    for field in ("control_mode", "rotation_repr", "gripper", "frame", "dim_labels"):
        assert getattr(sim.action_space.semantics, field) == getattr(real.action_space.semantics, field)
    assert sim.control_hz == real.control_hz


@pytest.mark.parametrize("entry_point", RIG_POLICIES, ids=lambda ep: ep.name)
def test_the_real_rigs_policies_are_compatible_unchanged(entry_point):
    assert check_compatibility(entry_point.load()(), YamArenaEmbodiment()).issues == []


def test_options_set_the_camera_size_and_control_rate():
    info = yam_arena_embodiment(cam_height=176, cam_width=320, control_hz=20).info
    assert [(c.name, c.height, c.width, c.channels) for c in info.observation_space.cameras] == [
        (name, 176, 320, 3) for name in DEFAULT_CAMERAS
    ]
    assert info.control_hz == 20.0


EXAMPLE_TASK = str(Path(__file__).resolve().parents[1] / "examples" / "stack_cubes.py")


def test_a_task_file_sets_the_scene_docs_success_and_environment():
    from yam_arena.task_file import lookup

    emb = YamArenaEmbodiment(task=EXAMPLE_TASK)
    info = emb.info
    assert emb.scene == "station" and emb.rig.name == "station-calibrated"
    assert info.environment_id == "isaaclab-arena-0.3.1-yam-bimanual-station-calibrated-task-stack_cubes"
    assert info.environment_revision == emb.task.sha256[:12]
    assert PRIVILEGED_SUCCESS in info.capabilities and info.supported_target_kinds == frozenset()
    assert "On the station table: a red 4 cm cube" in info.docs and "apple" not in info.docs
    assert info.docs.startswith("Two identical 6-DoF arms")
    assert lookup(info.environment_id) is emb.task
    assert_embodiment_conformant(info)


def test_a_task_file_and_a_different_scene_conflict():
    with pytest.raises(ValueError, match="sets SCENE='station'"):
        YamArenaEmbodiment(task=EXAMPLE_TASK, scene="tabletop")
    assert YamArenaEmbodiment(task=EXAMPLE_TASK, scene="station").scene == "station"


def test_object_jitter_defaults_to_3_cm_and_cannot_be_negative():
    assert YamArenaEmbodiment().object_jitter == 0.03
    assert yam_arena_embodiment(object_jitter="0").object_jitter == 0.0
    with pytest.raises(ValueError, match="object_jitter"):
        YamArenaEmbodiment(object_jitter=-0.01)


def test_a_window_goes_to_display_when_it_is_set():
    assert window_display({"DISPLAY": ":7"}) == ":7"


@pytest.fixture
def x_socket(monkeypatch):
    """A Unix socket path standing in for the desktop's (short: macOS caps socket paths at 104 bytes)."""
    with tempfile.TemporaryDirectory(dir="/tmp") as d:
        path = Path(d) / "X1"
        monkeypatch.setattr(inspect_embodiment, "DESKTOP_SOCKET", path)
        yield path


def test_a_window_falls_back_to_the_browser_desktop(x_socket):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(x_socket))
        server.listen()
        assert window_display({}) == ":1"


def test_a_window_without_a_display_says_how_to_get_one(x_socket):
    with pytest.raises(RuntimeError, match=r"setup\.sh --desktop"):
        window_display({"DISPLAY": ""})


def test_a_stale_desktop_socket_does_not_count_as_a_display(x_socket):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(x_socket))  # the file exists, but nothing listens on it
    assert x_socket.exists()
    with pytest.raises(RuntimeError, match="needs a display"):
        window_display({})
