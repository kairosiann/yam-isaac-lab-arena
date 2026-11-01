"""Inspect Robots policy ``yam_scripted``: scripted demos that reach full score on ``yam_arena``. NumPy only.

It plays top-down pick-and-place plans (:mod:`yam_arena.demo`) open loop, with IK on the URDF, from the
scene's starting layout. Run it with ``-E object_jitter=0``, so every object starts where the scene puts it:

- ``-E scene=station``: Dream's three tasks, chosen by the instruction ("put apple into the bowl", ...).
- ``-E task=my_task.py``: the task file's ``demo()``.

When the plan ends it holds the last pose for ``hold_s`` seconds (``-P hold_s=3``), then asks the rollout to
stop, unless the embodiment has already ended the episode with success.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from inspect_robots.policy import PolicyBase, PolicyInfo
from inspect_robots.types import Action, ActionChunk, Observation
from inspect_robots_yam.config import action_box
from inspect_robots_yam.packing import pack

from yam_arena import ik, station, task_file
from yam_arena.constants import CONTROL_HZ, LEFT_BASE_POS, READY, RIGHT_BASE_POS
from yam_arena.demo import ArmPlan, Waypoint, pick_place_waypoints

ARMS = ("left", "right")
TRAVEL_HEIGHT = 0.16  # above the table: a straight-down grip reaches every station object and goal at this height
RELEASE_GAP = 0.015  # the object's bottom is let go this far above the bowl's rim or the plate


def station_plan(goal: station.Goal) -> dict[str, list[Waypoint]]:
    """Dream's task on the station: the fruit's own arm picks it and lets go just above its target."""
    top = station.TABLE_TOP
    grip = station.GRIP_HEIGHT[goal.object]
    pick = (*station.ITEMS[goal.object].xy, top + grip)
    place = (*station.ITEMS[goal.target].xy, top + station.TARGET_TOP[goal.target] + grip + RELEASE_GAP)
    return {station.DEMO_ARM[goal.object]: pick_place_waypoints(pick, place, travel_z=top + TRAVEL_HEIGHT)}


def _on_station(environment_id: str | None) -> bool:
    return "-station-" in (environment_id or "")


class YamScriptedPolicy(PolicyBase):
    """Plays a scripted plan per arm; an arm without one holds the ready pose with its gripper open."""

    def __init__(self, hold_s: float = 3.0) -> None:
        low, high = np.append(ik.LOWER, 0.0), np.append(ik.UPPER, 1.0)
        self.info = PolicyInfo(name="yam_scripted", action_space=action_box(pack(low, low), pack(high, high)))
        self.hold_s = float(hold_s)
        self._dt = 1.0 / CONTROL_HZ
        self._environment_id: str | None = None
        self._bases = {"left": LEFT_BASE_POS, "right": RIGHT_BASE_POS}
        self._task: task_file.TaskFile | None = None
        self._plans: dict[str, ArmPlan] = {}
        self._end = 0.0

    def bind(self, embodiment_info: Any) -> None:
        """Adopt the embodiment's control rate, rig (arm bases) and task file."""
        if embodiment_info.control_hz:
            self._dt = 1.0 / float(embodiment_info.control_hz)
        self._environment_id = embodiment_info.environment_id
        if _on_station(self._environment_id):
            self._bases = dict(station.BASES)
        self._task = task_file.lookup(self._environment_id)

    def plan_for(self, instruction: str | None, target: Any = None) -> dict[str, list[Waypoint]]:
        if self._task is not None:
            if self._task.demo is None:
                raise ValueError(f"{self._task.path.name} has no demo() for yam_scripted to play")
            plan = dict(self._task.demo())
        elif _on_station(self._environment_id):
            goal = station.goal_for(instruction, target)
            if goal is None:
                tasks = ", ".join(repr(g.instruction) for g in station.DREAM_TASKS)
                raise ValueError(f"yam_scripted knows the station's tasks ({tasks}), not {instruction!r}")
            plan = station_plan(goal)
        else:
            raise ValueError("yam_scripted plays the station's tasks (-E scene=station) or a task file's demo()")
        unknown = sorted(set(plan) - set(ARMS))
        if unknown:
            raise ValueError(f"a demo plan's keys are arms ({ARMS}), got {unknown}")
        return plan

    def reset(self, scene: Any) -> None:
        plan = self.plan_for(scene.instruction, scene.target)
        self._plans = {arm: ArmPlan(list(waypoints), self._bases[arm]) for arm, waypoints in plan.items()}
        self._end = max((p.duration for p in self._plans.values()), default=0.0)

    def act(self, observation: Observation) -> ActionChunk:
        t = float(observation.state_time) + self._dt  # where the arm should be at the end of this control step
        per_arm = [self._plans[arm].action7(t)[0] if arm in self._plans else np.r_[READY, 1.0] for arm in ARMS]
        meta = {"request_stop": True} if t > self._end + self.hold_s else {}
        return ActionChunk(actions=[Action(data=pack(*per_arm), meta=meta)], control_hz=1.0 / self._dt)


def yam_scripted_policy(**kwargs: Any) -> YamScriptedPolicy:
    """Policy ``yam_scripted``; ``-P hold_s=3`` sets how long it holds the last pose before stopping."""
    return YamScriptedPolicy(**kwargs)
