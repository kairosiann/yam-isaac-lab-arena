"""Inspect Robots scorer ``task_score``: 1.0 for success, otherwise a task file's partial credit. Pure Python.

    inspect-robots run --embodiment yam_arena -E task=my_task.py --scorer task_score ...

A trial that ends with ``termination_reason="success"`` scores 1.0. Any other trial (time limit, a stop, a
failure) scores the last step's ``info["score"]``, which a task file's ``score(objects)`` sets (the score at the
end of the episode, or the best it reached with ``SCORE_OVER = "best"``), and 0.0 when there is none.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from inspect_robots.scorer import Score


@dataclass(frozen=True)
class TaskScore:
    name: str = "task_score"

    def __call__(self, record: Any, target: Any) -> Score:
        last = record.steps[-1] if record.steps else None
        if last is None:
            return Score(value=0.0, explanation="no steps")
        if last.result.terminated and last.result.termination_reason == "success":
            return Score(value=1.0, explanation="success")
        partial = last.result.info.get("score")
        if partial is None:
            return Score(value=0.0, explanation="no success and no partial credit")
        return Score(value=float(partial), explanation=f"no success; partial credit {float(partial):g}")


def task_score(*, name: str = "task_score") -> TaskScore:
    """Scorer ``task_score``: 1.0 for success, else the last step's ``info["score"]``, else 0.0."""
    return TaskScore(name=name)
