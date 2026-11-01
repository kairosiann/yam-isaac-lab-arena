"""The task_score scorer, on hand-made trial records."""

from importlib.metadata import entry_points
from types import SimpleNamespace

import pytest

from yam_arena.scoring import task_score


def record(*steps):
    return SimpleNamespace(
        steps=[
            SimpleNamespace(result=SimpleNamespace(terminated=t, termination_reason=r, info=info))
            for t, r, info in steps
        ]
    )


@pytest.mark.parametrize(
    "trial, value",
    [
        (record((False, None, {"score": 0.5}), (True, "success", {"score": 0.5})), 1.0),
        (record((False, None, {"score": 0.5}), (False, None, {"score": 0.25})), 0.25),
        (record((False, None, {})), 0.0),
        (record((True, "operator_end", {"score": 0.5})), 0.5),
        (record(), 0.0),
    ],
)
def test_success_scores_one_and_otherwise_the_last_steps_partial_credit(trial, value):
    assert task_score()(trial, None).value == value


def test_it_is_registered_for_the_cli():
    (ep,) = entry_points(group="inspect_robots.scorers", name="task_score")
    assert ep.load()().name == "task_score"
