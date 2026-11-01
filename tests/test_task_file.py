"""Task files: loading, validation, and the example task's scoring and demo (no Isaac Sim)."""

from pathlib import Path

import numpy as np
import pytest

from yam_arena import ik, station
from yam_arena.constants import CONTROL_HZ
from yam_arena.demo import ArmPlan
from yam_arena.task_file import TABLE_TOP, Box, ObjectState, load

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "stack_cubes.py"


def state(pos, size=(0.04, 0.04, 0.04), vel=(0.0, 0.0, 0.0)):
    pos, half = np.asarray(pos, dtype=float), np.asarray(size, dtype=float) / 2
    return ObjectState(pos, np.array([0.0, 0.0, 0.0, 1.0]), np.asarray(vel, float), np.zeros(3), pos - half, pos + half)


def write(tmp_path, body: str) -> Path:
    path = tmp_path / "task.py"
    path.write_text("from yam_arena.task_file import Box, Cylinder, Sphere, Asset\n" + body)
    return path


def test_the_example_loads_with_its_objects_hooks_and_docs():
    task = load(EXAMPLE)
    assert (task.name, task.scene, task.table_top) == ("stack_cubes", "station", TABLE_TOP["station"])
    assert task.names == ("red_cube", "blue_block") == task.free_objects
    assert all(callable(f) for f in (task.success, task.distance, task.metrics, task.demo))
    assert task.docs.startswith("On the station table")
    assert len(task.sha256) == 64 and task.success_hold_s == 1.0


def test_the_example_scores_a_stacked_cube_and_not_a_loose_one():
    task = load(EXAMPLE)
    top = TABLE_TOP["station"]
    blue = state((0.25, -0.05, top + 0.02), (0.06, 0.06, 0.04))
    stacked = {"red_cube": state((0.255, -0.05, top + 0.06)), "blue_block": blue}
    loose = {"red_cube": state((0.17, 0.12, top + 0.02)), "blue_block": blue}
    falling = {"red_cube": state((0.255, -0.05, top + 0.06), vel=(0, 0, -0.2)), "blue_block": blue}
    assert task.success(stacked) and not task.success(loose) and not task.success(falling)
    assert task.distance(stacked) == pytest.approx(0.005)
    assert task.distance(loose) == pytest.approx(np.linalg.norm([0.08, 0.17, 0.04]))
    assert task.metrics(stacked)["red_lift"] == pytest.approx(0.04)


def test_the_examples_demo_is_reachable_within_the_joint_limits():
    plan = load(EXAMPLE).demo()
    assert set(plan) == {"left"}
    arm = ArmPlan(plan["left"], station.BASES["left"])
    actions = np.array([arm.action7((k + 1) / CONTROL_HZ)[0] for k in range(round(arm.duration * CONTROL_HZ) + 5)])
    assert arm.worst_ik_mm < 0.5 and arm.worst_ik_deg < 0.1
    assert np.all(actions[:, :6] >= ik.LOWER) and np.all(actions[:, :6] <= ik.UPPER)


def test_resting_means_slower_than_a_centimetre_per_second():
    assert state((0, 0, 0)).resting() and not state((0, 0, 0), vel=(0.02, 0, 0)).resting()


def test_the_scene_defaults_to_the_station_and_fixed_objects_are_not_free(tmp_path):
    objects = "OBJECTS = [Box('tray', pos=(0.3, 0, 0.77)), Sphere('ball', pos=(0.2, 0, 0.8), mass=0.1)]"
    task = load(write(tmp_path, objects))
    assert task.scene == "station" and task.free_objects == ("ball",)
    assert task.success is task.distance is task.demo is None and task.docs is None


@pytest.mark.parametrize(
    "body, message",
    [
        ("SCENE = 'kitchen'\nOBJECTS = [Box('a', pos=(0, 0, 1))]", "SCENE must be one of"),
        ("OBJECTS = []", "at least one object"),
        ("OBJECTS = [Box('a', pos=(0, 0, 1)), Box('a', pos=(0, 0, 1))]", "repeated"),
        ("OBJECTS = [Box('table', pos=(0, 0, 1))]", "already name part of the room"),
        ("OBJECTS = [Box('red cube', pos=(0, 0, 1))]", "identifier"),
        ("OBJECTS = [Box('a', pos=(0, 0, 1), mass=-1)]", "mass must be positive"),
        ("OBJECTS = [Box('a', pos=(0, 0, 1), color=(255, 0, 0))]", "sRGB in 0..1"),
        ("OBJECTS = [Box('a', pos=(0, 0))]", "3 finite numbers"),
        ("OBJECTS = [Box('a', pos=(0, 0, 1), size=(0.1, 0, 0.1))]", "size must be positive"),
        ("OBJECTS = [Cylinder('a', pos=(0, 0, 1), radius=0)]", "radius and height"),
        ("OBJECTS = [Asset('a', 'bowl_ycb_robolab', xy=(0.3, 0), extent=0)]", "extent must be positive"),
        ("OBJECTS = [Box('a', pos=(0, 0, 1))]\nSUCCESS_HOLD_S = -1", "SUCCESS_HOLD_S"),
    ],
)
def test_bad_task_files_are_reported(tmp_path, body, message):
    with pytest.raises(ValueError, match=message):
        load(write(tmp_path, body))


def test_objects_must_be_object_specs_and_hooks_functions(tmp_path):
    with pytest.raises(TypeError, match="use Box, Cylinder, Sphere or Asset"):
        load(write(tmp_path, "OBJECTS = [{'name': 'a'}]"))
    with pytest.raises(TypeError, match="success must be a function"):
        load(write(tmp_path, "OBJECTS = [Box('a', pos=(0, 0, 1))]\nsuccess = True"))


def test_a_missing_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        load(tmp_path / "nope.py")


def test_box_takes_keywords():
    box = Box("cube", pos=(0.1, 0.2, 0.8), size=(0.05, 0.05, 0.05), mass=0.1, color=(1, 0, 0), yaw=0.3)
    assert box.free and box.size == (0.05, 0.05, 0.05)
