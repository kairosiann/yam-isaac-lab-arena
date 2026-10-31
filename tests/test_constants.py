import math

import numpy as np
import pytest

from yam_arena import YAM_URDF, constants
from yam_arena.analysis import world_hulls
from yam_arena.constants import (
    ALL_JOINTS,
    ARM_JOINTS,
    CONTROL_HZ,
    CUBE_POS,
    D405_FOCAL,
    D435_BOX_POS,
    D435_BOX_ROT_XYZW,
    D435_FOCAL,
    D435_SIZE_XYZ,
    DECIMATION,
    FINGER_CLOSED,
    FINGER_JOINTS,
    FINGER_OPEN,
    LEFT_BASE_POS,
    LINKS,
    PAD_TIP_PAST_TCP,
    PHYSICS_DT,
    PLATE_POS,
    READY,
    RENDER_INTERVAL,
    RIGHT_BASE_POS,
    TABLE_CENTER_X,
    TABLE_SIZE,
    TCP_OFFSET,
    TOP_CAM_FORWARD,
    TOP_CAM_POS,
    TOP_CAM_ROT_XYZW,
)
from yam_arena.urdf import forward_kinematics, parse_joints, quat_xyzw_to_matrix

JOINTS = parse_joints(YAM_URDF)


def finger_pads(opening):
    """Left and right pad vertices in the gripper frame: from 5 mm short of the TCP to the tips."""
    poses = forward_kinematics(YAM_URDF, {"joint7": opening, "joint8": opening}, root="gripper")
    pads = []
    for link in ("tip_left", "tip_right"):
        vertices = np.concatenate([hull.vertices for _, hull in world_hulls(poses, [link])])
        pads.append(vertices[vertices[:, 2] > TCP_OFFSET[2] - 0.005])
    return pads


def test_control_period_is_a_whole_number_of_physics_steps():
    period = 1 / CONTROL_HZ
    assert period == pytest.approx(DECIMATION * PHYSICS_DT, rel=1e-12)
    assert RENDER_INTERVAL == DECIMATION


def test_joint_and_link_names_are_the_urdf_names():
    assert ARM_JOINTS + FINGER_JOINTS == ALL_JOINTS == list(JOINTS)
    assert set(LINKS) == {"base"} | {j.child for j in JOINTS.values()}


def test_ready_and_finger_range_are_inside_the_joint_limits():
    for name, q in zip(ARM_JOINTS, READY, strict=True):
        assert JOINTS[name].lower < q < JOINTS[name].upper
    for name in FINGER_JOINTS:
        assert (JOINTS[name].lower, JOINTS[name].upper) == pytest.approx((FINGER_OPEN, FINGER_CLOSED))


def test_quaternions_are_unit():
    quaternions = {name: q for name, q in vars(constants).items() if name.endswith("_XYZW")}
    assert quaternions
    for name, q in quaternions.items():
        assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-4), name


@pytest.mark.parametrize(("opening", "gap"), [(FINGER_CLOSED, 0.0), (FINGER_OPEN, 0.094)])
def test_finger_range_runs_from_pads_touching_to_94_mm_apart(opening, gap):
    left, right = finger_pads(opening)
    assert right[:, 0].min() - left[:, 0].max() == pytest.approx(gap, abs=5e-4)


def test_tcp_is_between_the_pads_short_of_their_tips():
    for pad in finger_pads(FINGER_CLOSED):
        assert pad[:, 2].max() - TCP_OFFSET[2] == pytest.approx(PAD_TIP_PAST_TCP, abs=5e-4)
        assert pad[:, 1].min() < TCP_OFFSET[1] < pad[:, 1].max()


def test_top_camera_body_sits_behind_its_lens():
    forward = quat_xyzw_to_matrix(np.array(TOP_CAM_ROT_XYZW))[:, 0]
    np.testing.assert_allclose(TOP_CAM_FORWARD, forward, atol=1e-4)
    np.testing.assert_allclose(D435_BOX_POS, np.array(TOP_CAM_POS) - D435_SIZE_XYZ[0] / 2 * forward, atol=1e-4)
    assert D435_BOX_ROT_XYZW == TOP_CAM_ROT_XYZW


@pytest.mark.parametrize(("focal", "hfov"), [(D435_FOCAL, 69.0), (D405_FOCAL, 87.0)])
def test_focal_lengths_give_the_realsense_fields_of_view(focal, hfov):
    horizontal_aperture = 20.955
    assert math.degrees(2 * math.atan(horizontal_aperture / (2 * focal))) == pytest.approx(hfov, abs=0.5)


def test_arms_and_objects_stand_on_the_table():
    for x, y, _ in (LEFT_BASE_POS, RIGHT_BASE_POS, CUBE_POS, PLATE_POS):
        assert abs(x - TABLE_CENTER_X) < TABLE_SIZE[0] / 2 and abs(y) < TABLE_SIZE[1] / 2
    assert LEFT_BASE_POS[1] > 0 > RIGHT_BASE_POS[1]
