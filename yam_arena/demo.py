"""Scripted bimanual contact demo (port of the 2.3 ``yam_sim/demo.py``). NumPy only.

The left arm picks the cube and sets it down elsewhere; the right arm presses its closed
fingers onto the plate. Each arm follows Cartesian TCP waypoints; every control step the target
is interpolated (smoothstep) and turned into joint targets with DLS IK on the URDF kinematics,
then sent through the embodiment's 14-D joint action (the 2.3 demo ran closed-loop IK at
240 Hz on the simulator's Jacobian instead).

Two deliberate differences from 2.3, both forced by the arm's reach (see ``ik.py`` tests):
- gripper orientation ``DOWN_GRIP``: pointing straight down with the fingers closing along world
  Y, but rotated 180 deg about the approach axis relative to the 2.3 ``DOWN``. The 2.3 ``DOWN``
  needs joint6 beyond its +-120 deg limit at these targets; a parallel gripper grasps the same
  either way.
- hover height 0.16 m above the table instead of 0.20 m: straight-down at 0.20 m is past joint4's
  limit over the cube and plate (the 2.3 IK silently settled for a tilted, offset pose there).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from yam_arena import ik
from yam_arena.constants import (
    CUBE_POS,
    CUBE_SIZE,
    DOWN_XYZW,
    FINGER_CLOSED,
    FINGER_OPEN,
    LEFT_BASE_POS,
    PAD_TIP_PAST_TCP,
    PLATE_HEIGHT,
    PLATE_POS,
    READY,
    RIGHT_BASE_POS,
    TABLE_HEIGHT,
)
from yam_arena.urdf import quat_xyzw_to_matrix

DOWN_GRIP = ik.matrix_to_quat_xyzw(quat_xyzw_to_matrix(np.array(DOWN_XYZW)) @ np.diag([-1.0, -1.0, 1.0]))
TIPS_ON_TABLE = TABLE_HEIGHT + PAD_TIP_PAST_TCP  # TCP height at which the pad tips touch the table
HOVER = TABLE_HEIGHT + 0.16
GRASP_Z = TIPS_ON_TABLE + CUBE_SIZE / 2
DROP_XY = (CUBE_POS[0] - 0.1, CUBE_POS[1] + 0.08)
PRESS_Z = TIPS_ON_TABLE + PLATE_HEIGHT - 0.01  # commanded 10 mm *into* the plate


@dataclass
class Waypoint:
    duration: float
    pos: tuple[float, float, float]
    finger: float  # finger joint target (m): FINGER_OPEN .. FINGER_CLOSED
    label: str = ""


def pick_and_place(hold_s: float = 1.5) -> list[Waypoint]:
    x, y, _ = CUBE_POS
    return [
        Waypoint(1.0, (x, y, HOVER), FINGER_OPEN, "to hover"),
        Waypoint(2.5, (x, y, HOVER), FINGER_OPEN, "wait"),
        Waypoint(2.0, (x, y, GRASP_Z), FINGER_OPEN, "descend"),
        Waypoint(1.0, (x, y, GRASP_Z), FINGER_CLOSED, "close"),
        Waypoint(2.0, (x, y, HOVER), FINGER_CLOSED, "lift"),
        Waypoint(hold_s, (x, y, HOVER), FINGER_CLOSED, "hold lifted"),  # new: jitter window
        Waypoint(2.0, (*DROP_XY, HOVER), FINGER_CLOSED, "carry"),
        Waypoint(2.0, (*DROP_XY, GRASP_Z + 0.005), FINGER_CLOSED, "lower"),
        Waypoint(1.0, (*DROP_XY, GRASP_Z + 0.005), FINGER_OPEN, "release"),
        Waypoint(2.0, (*DROP_XY, HOVER), FINGER_OPEN, "retreat"),
    ]


def press_plate() -> list[Waypoint]:
    x, y, _ = PLATE_POS
    return [
        Waypoint(1.0, (x, y, HOVER), FINGER_CLOSED, "to hover"),
        Waypoint(2.5, (x, y, HOVER), FINGER_CLOSED, "wait"),
        Waypoint(2.5, (x, y, PRESS_Z), FINGER_CLOSED, "press"),
        Waypoint(2.0, (x, y, PRESS_Z), FINGER_CLOSED, "hold press"),
        Waypoint(2.0, (x, y, HOVER), FINGER_CLOSED, "up"),
    ]


class ArmPlan:
    """Smoothly interpolated TCP waypoints -> joint targets for one arm."""

    def __init__(self, waypoints: list[Waypoint], base_pos, q_start=READY, finger_start=FINGER_OPEN):
        self.waypoints = waypoints
        self.base_pos = np.asarray(base_pos, dtype=float)
        self.q = np.array(q_start, dtype=float)
        self.start_pos, self.start_rot = ik.tcp_pose(self.q, self.base_pos)
        self.down = quat_xyzw_to_matrix(np.asarray(DOWN_GRIP))
        self.finger_start = finger_start
        self.worst_ik_mm = 0.0
        self.worst_ik_deg = 0.0

    @property
    def duration(self) -> float:
        return sum(w.duration for w in self.waypoints)

    def target(self, t: float):
        """(TCP position, rotation matrix, finger, label) at time t."""
        prev_pos, prev_rot, prev_finger = self.start_pos, self.start_rot, self.finger_start
        for w in self.waypoints:
            goal = np.asarray(w.pos, dtype=float)
            if t < w.duration:
                s = ik.smoothstep(t / w.duration)
                return (prev_pos + s * (goal - prev_pos), ik.slerp_matrix(prev_rot, self.down, s),
                        prev_finger + s * (w.finger - prev_finger), w.label)
            t -= w.duration
            prev_pos, prev_rot, prev_finger = goal, self.down, w.finger
        return prev_pos, prev_rot, prev_finger, "done"

    def action7(self, t: float) -> tuple[np.ndarray, str]:
        """[j1..j6, normalized grip] at time t (IK warm-started from the previous step)."""
        pos, rot, finger, label = self.target(t)
        q, pe, re = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), self.q, base_pos=self.base_pos, iters=60)
        self.q = q
        self.worst_ik_mm = max(self.worst_ik_mm, pe * 1000)
        self.worst_ik_deg = max(self.worst_ik_deg, float(np.degrees(re)))
        grip = float(np.clip(finger / FINGER_OPEN, 0.0, 1.0)) if FINGER_OPEN != FINGER_CLOSED else 1.0
        return np.r_[q, grip], label


def make_plans(hold_s: float = 1.5) -> tuple[ArmPlan, ArmPlan]:
    left = ArmPlan(pick_and_place(hold_s), LEFT_BASE_POS, finger_start=FINGER_OPEN)
    right = ArmPlan(press_plate(), RIGHT_BASE_POS, finger_start=FINGER_OPEN)
    return left, right
