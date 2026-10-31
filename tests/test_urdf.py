import numpy as np
import pytest

from yam_arena import ASSET_DIR, YAM_URDF
from yam_arena.constants import ARM_JOINTS, LINKS, READY
from yam_arena.ik import tcp_pose
from yam_arena.urdf import (
    axis_angle_matrix,
    collision_counts,
    forward_kinematics,
    parse_joints,
    quat_xyzw_to_matrix,
    rotation_angle_deg,
    rpy_matrix,
)

JOINTS = parse_joints(YAM_URDF)
I2RT_URDF = ASSET_DIR / "yam.urdf"


def fk(q):
    return forward_kinematics(YAM_URDF, dict(zip(ARM_JOINTS, map(float, q), strict=True)))


def rot(axis, angle):
    return axis_angle_matrix(np.array(axis, dtype=float), angle)


def test_one_serial_arm_with_two_prismatic_fingers():
    assert [(j.name, j.type, j.parent, j.child) for j in JOINTS.values()] == [
        ("joint1", "revolute", "base", "link1"),
        ("joint2", "revolute", "link1", "link2"),
        ("joint3", "revolute", "link2", "link3"),
        ("joint4", "revolute", "link3", "link4"),
        ("joint5", "revolute", "link4", "link5"),
        ("joint6", "revolute", "link5", "gripper"),
        ("joint7", "prismatic", "gripper", "tip_left"),
        ("joint8", "prismatic", "gripper", "tip_right"),
    ]
    assert all(j.lower < j.upper for j in JOINTS.values())


def test_sim_urdf_keeps_the_i2rt_kinematics_and_adds_hulls():
    i2rt = parse_joints(I2RT_URDF)
    assert i2rt.keys() == JOINTS.keys()
    for name, joint in JOINTS.items():
        for field in ("type", "parent", "child", "lower", "upper", "xyz", "rpy", "axis"):
            np.testing.assert_array_equal(getattr(joint, field), getattr(i2rt[name], field), err_msg=f"{name}.{field}")
    assert set(collision_counts(I2RT_URDF).values()) == {0}


def test_every_link_lists_each_of_its_hull_files():
    counts = collision_counts(YAM_URDF)
    assert set(counts) == set(LINKS)
    for link, n in counts.items():
        assert n == len(list((ASSET_DIR / "collision" / link).glob("*.stl"))) > 0


def test_rpy_is_fixed_axis_roll_then_pitch_then_yaw():
    r, p, y = 0.3, -0.7, 1.9
    expected = rot([0, 0, 1], y) @ rot([0, 1, 0], p) @ rot([1, 0, 0], r)
    np.testing.assert_allclose(rpy_matrix(np.array([r, p, y])), expected, atol=1e-12)


def test_axis_angle_matrix_is_a_rotation_about_the_axis():
    np.testing.assert_allclose(rot([0, 0, 2], np.pi / 2) @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    r = rot([1, 2, 3], 0.8)
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(r) == pytest.approx(1.0)
    np.testing.assert_allclose(r @ [1, 2, 3], [1, 2, 3], atol=1e-12)


def test_quaternions_are_xyzw_and_normalized():
    s = np.sin(np.pi / 4)
    np.testing.assert_allclose(quat_xyzw_to_matrix(np.array([0, 0, s, s])), rot([0, 0, 1], np.pi / 2), atol=1e-12)
    np.testing.assert_allclose(quat_xyzw_to_matrix(np.array([0, 0, 0, 3.0])), np.eye(3))


def test_rotation_angle_between_rotations():
    assert rotation_angle_deg(rot([1, 1, 0], 0.2), rot([1, 1, 0], 0.5)) == pytest.approx(np.degrees(0.3))
    r = rot([0.3, -1, 2], 1.1)
    assert rotation_angle_deg(r, r @ r.T @ r) == pytest.approx(0.0, abs=1e-5)


def test_fk_gives_a_rigid_pose_per_link_in_the_base_frame():
    poses = fk(READY)
    assert set(poses) == set(LINKS)
    np.testing.assert_array_equal(poses["base"], np.eye(4))
    for pose in poses.values():
        np.testing.assert_allclose(pose[:3, :3] @ pose[:3, :3].T, np.eye(3), atol=1e-12)
        np.testing.assert_array_equal(pose[3], [0, 0, 0, 1])


def test_all_zero_joints_fold_the_arm_low_with_the_gripper_forward():
    pos, r = tcp_pose(np.zeros(6))
    np.testing.assert_allclose(r[:, 2], [1, 0, 0], atol=1e-5)
    assert pos[0] > 0.2 and pos[1] == pytest.approx(0.0, abs=1e-5) and pos[2] < 0.2


def test_the_straight_arm_reaches_076_m_from_the_shoulder():
    shoulder = fk(np.zeros(6))["link2"][:3, 3]
    up, up_rot = tcp_pose([0, np.pi / 2, np.pi, 0, 0, 0])
    forward, forward_rot = tcp_pose([0, np.pi, np.pi, 0, 0, 0])
    np.testing.assert_allclose(up_rot[:, 2], [0, 0, 1], atol=1e-5)
    np.testing.assert_allclose(forward_rot[:, 2], [1, 0, 0], atol=1e-5)
    assert np.linalg.norm(up - shoulder) == pytest.approx(0.76, abs=0.01)
    assert np.linalg.norm(forward - shoulder) == pytest.approx(0.76, abs=0.01)


def test_segment_lengths_match_the_real_arm():
    q = [0.4, 1.2, 0.9, -0.3, 0.0, 0.5]
    poses = fk(q)

    def axis(joint):
        link = JOINTS[joint].child
        return poses[link][:3, 3], poses[link][:3, :3] @ JOINTS[joint].axis

    def distance(point, line):
        origin, direction = line
        d = point - origin
        return np.linalg.norm(d - (d @ direction) * direction)

    shoulder, elbow, wrist = axis("joint2"), axis("joint3"), axis("joint4")
    assert abs(shoulder[1] @ elbow[1]) == pytest.approx(1.0) and abs(elbow[1] @ wrist[1]) == pytest.approx(1.0)
    assert distance(elbow[0], shoulder) == pytest.approx(0.26, abs=0.01)
    assert distance(wrist[0], elbow) == pytest.approx(0.25, abs=0.01)
    assert distance(tcp_pose(q)[0], wrist) == pytest.approx(0.25, abs=0.01)


def test_j0_turns_the_whole_arm_about_the_base_vertical():
    q = np.array([0.0, 1.2, 0.9, -0.3, 0.4, 0.5])
    turned = fk([q[0] + 0.7, *q[1:]])
    lift = np.eye(4)
    lift[:3, :3] = rot([0, 0, 1], 0.7)
    for link, pose in fk(q).items():
        np.testing.assert_allclose(turned[link], pose if link == "base" else lift @ pose, atol=1e-5)


def test_j1_and_j2_zero_references():
    def upper_arm(j1):
        poses = fk([0, j1, 0, 0, 0, 0])
        d = (poses["link3"][:3, 3] - poses["link2"][:3, 3])[[0, 2]]
        return d / np.linalg.norm(d)

    np.testing.assert_allclose(upper_arm(0.0), [-1, 0], atol=1e-5)
    np.testing.assert_allclose(upper_arm(np.pi / 2), [0, 1], atol=1e-5)
    np.testing.assert_allclose(upper_arm(np.pi), [1, 0], atol=1e-5)
    assert JOINTS["joint2"].lower == pytest.approx(0.0) and JOINTS["joint3"].lower == pytest.approx(0.0)

    def elbow_angle(j2):
        poses = fk([0, 1.0, j2, 0, 0, 0])
        back = (poses["link2"][:3, 3] - poses["link3"][:3, 3])[[0, 2]]
        fore = (poses["link4"][:3, 3] - poses["link3"][:3, 3])[[0, 2]]
        return np.arccos(back @ fore / np.linalg.norm(back) / np.linalg.norm(fore))

    angles = [elbow_angle(j2) for j2 in np.linspace(0.0, 3.0, 7)]
    assert angles[0] < np.radians(20) and np.all(np.diff(angles) > 0)


def test_positive_directions_match_the_joint_guide():
    q0 = np.array(READY, dtype=float)
    p0, r0 = tcp_pose(q0)

    def nudged(i):
        q = q0.copy()
        q[i] += 0.05
        return tcp_pose(q)

    assert nudged(0)[0][1] > p0[1]
    assert nudged(3)[1][2, 2] > r0[2, 2]
    assert nudged(4)[0][1] < p0[1]
    p5, r5 = nudged(5)
    np.testing.assert_allclose(p5, p0, atol=1e-12)
    assert np.cross(r0[:, 0], r5[:, 0]) @ r0[:, 2] > 0
