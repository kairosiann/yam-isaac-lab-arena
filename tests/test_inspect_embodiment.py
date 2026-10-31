import json
import subprocess
import sys
from importlib.metadata import distribution

import numpy as np
import pytest
from inspect_robots.compat import check_compatibility
from inspect_robots.conformance import assert_embodiment_conformant
from inspect_robots.embodiment import RESETTABLE, SEEDABLE
from inspect_robots_yam.config import DEFAULT_CAMERAS
from inspect_robots_yam.embodiment import YAMEmbodiment

from yam_arena import YAM_URDF
from yam_arena.constants import ARM_JOINTS, CONTROL_HZ
from yam_arena.inspect_embodiment import YamArenaEmbodiment, yam_arena_embodiment
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
