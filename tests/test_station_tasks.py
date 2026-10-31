"""Which station task a scene asks for: Dream's instructions, phrasing variants, Inspect targets."""

from types import SimpleNamespace

import pytest

from yam_arena.station import DREAM_TASKS, TARGET_KIND, Goal, goal_for


@pytest.mark.parametrize(
    "instruction, goal",
    [
        ("put apple into the bowl", Goal("apple", "bowl")),
        ("put orange into the bowl", Goal("orange", "bowl")),
        ("put banana on the plate", Goal("banana", "plate")),
        ("Put the apple in the bowl.", Goal("apple", "bowl")),
        ("  place an orange inside the bowl ", Goal("orange", "bowl")),
        ("put the banana onto the plate!", Goal("banana", "plate")),
        ("move the banana on top of the plate", Goal("banana", "plate")),
        ("put banana into the bowl", Goal("banana", "bowl")),
    ],
)
def test_instructions_name_their_goal(instruction, goal):
    assert goal_for(instruction) == goal


@pytest.mark.parametrize(
    "instruction",
    [
        "put the red cube on the plate",
        "wave hello",
        "",
        None,
        "put apple into the bowl and orange on the plate",
        "eat the apple",
    ],
)
def test_other_instructions_have_no_goal(instruction):
    assert goal_for(instruction) is None


def test_dreams_tasks_round_trip_through_their_instructions():
    assert [g.instruction for g in DREAM_TASKS] == [
        "put apple into the bowl",
        "put orange into the bowl",
        "put banana on the plate",
    ]
    assert all(goal_for(g.instruction) == g for g in DREAM_TASKS)
    assert [g.relation for g in DREAM_TASKS] == ["in", "in", "on"]


def test_a_put_target_wins_over_the_instruction():
    target = SimpleNamespace(kind=TARGET_KIND, spec={"object": "orange", "target": "plate"})
    assert goal_for("put apple into the bowl", target) == Goal("orange", "plate")


def test_bad_targets_are_errors():
    with pytest.raises(ValueError):
        goal_for("x", SimpleNamespace(kind="reach_object", spec={}))
    with pytest.raises(ValueError):
        goal_for("x", SimpleNamespace(kind=TARGET_KIND, spec={"object": "bottle", "target": "bowl"}))
