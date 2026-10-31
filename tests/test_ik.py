import numpy as np
import pytest

from yam_arena import YAM_URDF, ik
from yam_arena.constants import ARM_JOINTS, LEFT_BASE_POS, READY
from yam_arena.urdf import axis_angle_matrix, parse_joints, quat_xyzw_to_matrix, rotation_angle_deg

ROTATIONS = [
    ([0.3, -0.2, 0.9], 0.4),
    ([1, 0, 0], np.pi - 0.1),
    ([0, 1, 0], np.pi - 0.1),
    ([0, 0, 1], np.pi - 0.1),
    ([1, 1, 1], np.pi),
]
REACHABLE = np.clip(READY + np.random.default_rng(0).uniform(-0.5, 0.5, size=(25, 6)), ik.LOWER, ik.UPPER)


def within_limits(q):
    return bool(np.all(q >= ik.LOWER) and np.all(q <= ik.UPPER))


def test_joint_bounds_are_the_urdf_limits():
    joints = parse_joints(YAM_URDF)
    np.testing.assert_array_equal(ik.LOWER, [joints[n].lower for n in ARM_JOINTS])
    np.testing.assert_array_equal(ik.UPPER, [joints[n].upper for n in ARM_JOINTS])


@pytest.mark.parametrize(("axis", "angle"), ROTATIONS)
def test_matrix_to_quat_round_trip(axis, angle):
    r = axis_angle_matrix(np.array(axis, dtype=float), angle)
    q = ik.matrix_to_quat_xyzw(r)
    assert np.linalg.norm(q) == pytest.approx(1.0)
    np.testing.assert_allclose(quat_xyzw_to_matrix(q), r, atol=1e-12)


def test_slerp_matrix_turns_about_one_axis_at_a_steady_rate():
    r0 = axis_angle_matrix(np.array([0.2, 1.0, -0.4]), 0.3)
    r1 = axis_angle_matrix(np.array([1.0, -0.5, 0.3]), 1.4) @ r0
    np.testing.assert_allclose(ik.slerp_matrix(r0, r1, 0.0), r0, atol=1e-12)
    np.testing.assert_allclose(ik.slerp_matrix(r0, r1, 1.0), r1, atol=1e-12)
    for s in (0.25, 0.5, 0.75):
        mid = ik.slerp_matrix(r0, r1, s)
        assert rotation_angle_deg(r0, mid) == pytest.approx(s * np.degrees(1.4))
        assert rotation_angle_deg(mid, r1) == pytest.approx((1 - s) * np.degrees(1.4))


def test_smoothstep_eases_in_and_out():
    assert [ik.smoothstep(s) for s in (-1.0, 0.0, 0.5, 1.0, 2.0)] == [0.0, 0.0, 0.5, 1.0, 1.0]
    assert np.all(np.diff([ik.smoothstep(s) for s in np.linspace(0, 1, 101)]) > 0)
    h = 1e-6
    assert ik.smoothstep(h) / h < 1e-5 and (1 - ik.smoothstep(1 - h)) / h < 1e-5


def test_tcp_pose_moves_with_the_base():
    p, r = ik.tcp_pose(READY)
    p2, r2 = ik.tcp_pose(READY, base_pos=LEFT_BASE_POS)
    np.testing.assert_allclose(p2 - p, LEFT_BASE_POS, atol=1e-12)
    np.testing.assert_array_equal(r2, r)


@pytest.mark.parametrize("q", REACHABLE, ids=range(len(REACHABLE)))
def test_solve_recovers_reachable_poses(q):
    pos, rot = ik.tcp_pose(q, LEFT_BASE_POS)
    q_ik, pos_err, rot_err = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), READY, base_pos=LEFT_BASE_POS, iters=1000)
    reached, reached_rot = ik.tcp_pose(q_ik, LEFT_BASE_POS)
    assert pos_err == pytest.approx(np.linalg.norm(reached - pos))
    assert pos_err < 5e-4 and np.degrees(rot_err) < 0.05
    assert rotation_angle_deg(reached_rot, rot) < 0.05
    assert within_limits(q_ik)


def test_solve_without_an_orientation_matches_position_only():
    target = ik.tcp_pose(REACHABLE[0])[0]
    q, pos_err, rot_err = ik.solve(target)
    assert pos_err < 1e-4 and rot_err == 0.0
    np.testing.assert_allclose(ik.tcp_pose(q)[0], target, atol=1e-4)


def test_solve_reports_unreachable_targets_and_stays_in_limits():
    q, pos_err, _ = ik.solve((2.0, 0.0, 0.3))
    assert pos_err > 1.0
    assert within_limits(q)
