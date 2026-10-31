import numpy as np
import pytest

from yam_arena import ik
from yam_arena.constants import (
    CONTROL_HZ,
    CUBE_POS,
    DOWN_XYZW,
    FINGER_OPEN,
    LEFT_BASE_POS,
    PLATE_POS,
    READY,
    RIGHT_BASE_POS,
    TABLE_HEIGHT,
)
from yam_arena.demo import DOWN_GRIP, HOVER, make_plans
from yam_arena.urdf import quat_xyzw_to_matrix

TARGETS = {"cube": (CUBE_POS, LEFT_BASE_POS), "plate": (PLATE_POS, RIGHT_BASE_POS)}


def at_limit(q, joint):
    return min(q[joint] - ik.LOWER[joint], ik.UPPER[joint] - q[joint]) < 1e-6


def solve_over(target, height, orientation):
    (x, y, _), base = TARGETS[target]
    return ik.solve((x, y, height), orientation, READY, base_pos=base, iters=300)


def test_grip_orientation_points_down_with_the_fingers_closing_along_y():
    down, grip = quat_xyzw_to_matrix(np.array(DOWN_XYZW)), quat_xyzw_to_matrix(DOWN_GRIP)
    np.testing.assert_allclose(down[:, 2], [0, 0, -1], atol=1e-4)
    np.testing.assert_allclose(down[:, 0], [0, 1, 0], atol=1e-4)
    np.testing.assert_allclose(grip[:, 2], down[:, 2], atol=1e-4)
    np.testing.assert_allclose(grip[:, 0], -down[:, 0], atol=1e-4)


def test_targets_start_at_ready_and_pass_through_every_waypoint():
    plan = make_plans()[0]
    start = ik.tcp_pose(READY, LEFT_BASE_POS)
    pos, rot, finger, label = plan.target(0.0)
    np.testing.assert_allclose(pos, start[0])
    np.testing.assert_allclose(rot, start[1])
    assert (finger, label) == (FINGER_OPEN, plan.waypoints[0].label)
    t = 0.0
    for waypoint, following in zip(plan.waypoints, [*plan.waypoints[1:], None], strict=True):
        t += waypoint.duration
        pos, rot, finger, label = plan.target(t)
        np.testing.assert_allclose(pos, waypoint.pos)
        assert finger == waypoint.finger
        assert label == (following.label if following else "done")
    np.testing.assert_allclose(rot, quat_xyzw_to_matrix(DOWN_GRIP), atol=1e-12)


def test_both_arms_track_their_plans_within_the_joint_limits():
    dt = 1 / CONTROL_HZ
    for plan in make_plans():
        steps = [plan.action7((k + 1) * dt) for k in range(round(plan.duration / dt) + 5)]
        actions = np.array([a for a, _ in steps])
        assert plan.worst_ik_mm < 0.5 and plan.worst_ik_deg < 0.1
        assert np.all(actions[:, :6] >= ik.LOWER) and np.all(actions[:, :6] <= ik.UPPER)
        assert np.all((actions[:, 6] >= 0) & (actions[:, 6] <= 1))
        assert list(dict.fromkeys(label for _, label in steps)) == [w.label for w in plan.waypoints] + ["done"]


@pytest.mark.parametrize("target", TARGETS)
def test_straight_down_grip_reaches_the_hover_point(target):
    q, pos_err, rot_err = solve_over(target, HOVER, DOWN_GRIP)
    assert pos_err < 1e-3 and np.degrees(rot_err) < 0.1
    assert not any(at_limit(q, j) for j in range(6))


@pytest.mark.parametrize("target", TARGETS)
def test_the_2_3_orientation_would_need_joint6_past_its_limit(target):
    q, _, rot_err = solve_over(target, HOVER, DOWN_XYZW)
    assert np.degrees(rot_err) > 10 and at_limit(q, 5)


@pytest.mark.parametrize("target", TARGETS)
def test_the_2_3_hover_height_would_need_joint4_past_its_limit(target):
    q, pos_err, _ = solve_over(target, TABLE_HEIGHT + 0.20, DOWN_GRIP)
    assert pos_err > 5e-3 and at_limit(q, 3)
