"""The yam_scripted policy: its plans, its actions, and how it finds them (no Isaac Sim)."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from inspect_robots import Scene
from inspect_robots.compat import check_compatibility
from inspect_robots.types import Observation

from yam_arena import ik, station
from yam_arena.constants import READY
from yam_arena.demo import ArmPlan
from yam_arena.inspect_embodiment import YamArenaEmbodiment
from yam_arena.scripted import ARMS, YamScriptedPolicy, station_plan

EXAMPLE = str(Path(__file__).resolve().parents[1] / "examples" / "stack_cubes.py")
SIM_MODULES = {"isaacsim", "isaaclab", "isaaclab_arena", "isaaclab_physx", "omni", "carb", "pxr", "torch"}


def bound(embodiment: YamArenaEmbodiment, **kwargs) -> YamScriptedPolicy:
    policy = YamScriptedPolicy(**kwargs)
    policy.bind(embodiment.info)
    return policy


def run(policy: YamScriptedPolicy, instruction: str, hz: float, seconds: float):
    policy.reset(Scene(id="s", instruction=instruction))
    chunks = [policy.act(Observation(state_time=k / hz)) for k in range(round(seconds * hz))]
    return np.array([c.actions[0].data for c in chunks]), [bool(c.actions[0].meta.get("request_stop")) for c in chunks]


@pytest.mark.parametrize("goal", station.DREAM_TASKS, ids=lambda g: g.object)
def test_each_station_task_is_reachable_by_its_fruits_arm(goal):
    plan = station_plan(goal)
    assert list(plan) == [station.DEMO_ARM[goal.object]]
    arm = ArmPlan(plan[station.DEMO_ARM[goal.object]], station.BASES[station.DEMO_ARM[goal.object]])
    actions = np.array([arm.action7((k + 1) / 20)[0] for k in range(round(arm.duration * 20) + 5)])
    assert arm.worst_ik_mm < 0.5 and arm.worst_ik_deg < 0.1
    assert np.all(actions[:, :6] >= ik.LOWER) and np.all(actions[:, :6] <= ik.UPPER)


def test_the_plan_grips_the_fruit_and_lets_go_above_its_target():
    waypoints = station_plan(station.Goal("apple", "bowl"))["left"]
    by_label = {w.label: w for w in waypoints}
    assert by_label["descend"].pos[:2] == station.ITEMS["apple"].xy
    assert by_label["release"].pos[:2] == station.ITEMS["bowl"].xy
    release_bottom = by_label["release"].pos[2] - station.GRIP_HEIGHT["apple"] - station.TABLE_TOP
    assert release_bottom == pytest.approx(station.TARGET_TOP["bowl"] + 0.015)


def test_on_the_station_it_moves_one_arm_and_holds_the_other_ready():
    policy = bound(YamArenaEmbodiment(scene="station", control_hz=20))
    actions, stops = run(policy, "put banana on the plate", 20, 14)
    np.testing.assert_allclose(actions[:, :7], np.tile([*READY, 1.0], (len(actions), 1)))
    assert np.ptp(actions[:, 7:13], axis=0).max() > 0.3  # the right arm moves
    assert set(np.round(actions[:, 13], 6)) >= {0.0, 1.0}  # and closes and opens its gripper
    space = policy.info.action_space
    assert np.all(actions >= space.low - 1e-9) and np.all(actions <= space.high + 1e-9)
    first_stop = stops.index(True)
    assert not any(stops[:first_stop]) and all(stops[first_stop:])
    assert first_stop / 20 == pytest.approx(policy._end + policy.hold_s, abs=0.1)


def test_it_plays_the_task_files_demo_for_the_embodiment_that_loaded_it():
    policy = bound(YamArenaEmbodiment(task=EXAMPLE))
    actions, _ = run(policy, "put the red cube on the blue block", 10, 13)
    assert np.ptp(actions[:, :6], axis=0).max() > 0.3
    np.testing.assert_allclose(actions[:, 7:], np.tile([*READY, 1.0], (len(actions), 1)))


def test_it_is_compatible_with_yam_arena():
    for embodiment in (YamArenaEmbodiment(scene="station"), YamArenaEmbodiment(task=EXAMPLE)):
        assert check_compatibility(YamScriptedPolicy(), embodiment).issues == []


@pytest.mark.parametrize(
    "embodiment, instruction, message",
    [
        (lambda: YamArenaEmbodiment(scene="station"), "wave hello", "knows the station's tasks"),
        (lambda: YamArenaEmbodiment(), "put the red cube on the plate", "plays the station's tasks"),
    ],
)
def test_it_says_what_it_cannot_play(embodiment, instruction, message):
    with pytest.raises(ValueError, match=message):
        bound(embodiment()).reset(Scene(id="s", instruction=instruction))


def test_arms_are_left_then_right():
    assert ARMS == ("left", "right")


def test_the_entry_point_constructs_it_without_isaac():
    script = (
        "import json, sys\n"
        "from importlib.metadata import entry_points\n"
        "(ep,) = entry_points(group='inspect_robots.policies', name='yam_scripted')\n"
        "ep.load()(hold_s='2')\n"
        "print(json.dumps(sorted({m.split('.')[0] for m in sys.modules})))\n"
    )
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120, check=True)
    assert SIM_MODULES.isdisjoint(json.loads(proc.stdout))
