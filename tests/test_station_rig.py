"""Dream's station rig: arm mounts, cameras and their lenses, against Dream's own kinematics."""

import math

import numpy as np
import pytest

from yam_arena import YAM_URDF, station
from yam_arena.constants import D405_FOCAL, D435_FOCAL
from yam_arena.ik import matrix_to_quat_xyzw
from yam_arena.rig import TABLETOP, Lens
from yam_arena.urdf import axis_angle_matrix, forward_kinematics, quat_xyzw_to_matrix, rotation_angle_deg, rpy_matrix

# Dream's yam.urdf (dreamlabs 1.0.23), base_link -> link_6: joint origin (xyz, rpy); every axis is +z.
DREAM_CHAIN = [
    ((-3.55271e-18, -1.05495e-18, 0.0631), (0.0, 0.0, 1.5708)),
    ((2.5e-05, -0.02, 0.0409), (1.5708, -1.98004e-28, 1.5708)),
    ((0.264, 0.0, 6.77626e-21), (-3.14159, 1.26218e-29, 7.88861e-31)),
    ((-0.245, -0.0600003, -1.9984e-15), (-5.65471e-31, -1.1507e-13, 2.94252e-43)),
    ((-0.0739968, -0.0395003, 2.44738e-05), (-1.5708, 1.5708, 0.0)),
    ((-4.0742e-16, 0.0353, 0.0395), (-1.5708, 3.81639e-16, 4.68582e-15)),
]
POSES = [
    np.zeros(6),
    np.array([0.0, 1.0, 1.0, -0.6, 0.0, 0.0]),
    np.array([0.3, 1.2, 0.8, -0.4, 0.2, 0.5]),
    np.array([-1.2, 2.0, 0.5, 1.0, -1.0, 2.0]),
]


def transform(pos, rot) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = rot, pos
    return t


def dream_link6(q) -> np.ndarray:
    t = np.eye(4)
    for (xyz, rpy), angle in zip(DREAM_CHAIN, q, strict=True):
        t = (
            t
            @ transform(xyz, rpy_matrix(np.array(rpy)))
            @ transform((0, 0, 0), axis_angle_matrix(np.array([0, 0, 1.0]), angle))
        )
    return t


def camera_rot(cam) -> np.ndarray:
    return quat_xyzw_to_matrix(np.array(cam.rot_xyzw))


def test_bases_keep_the_tabletop_height_and_dreams_spacing():
    assert station.BASES == {"left": pytest.approx((0.0, 0.305, 0.75)), "right": pytest.approx((0.0, -0.305, 0.75))}


def test_frames_agree_on_dreams_arm_mounts():
    # The calibration's mounts (0.2525, +-0.305, 0.76) and Dream's (+-0.305, 0.24, 0.772) are the same arms.
    for side, y in (("left", 0.305), ("right", -0.305)):
        assert station.from_dream(station.DREAM_BASE_LINKS[side])[:2] == pytest.approx(
            station.from_calibration((0.2525, y, 0.76))[:2]
        )


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("q", POSES)
def test_wrist_camera_matches_dreams_link6_mount(side, q):
    """Our base + URDF FK + mount on `gripper` == Dream's base_link + Dream's FK + the calibration's link_6 mount."""
    w, x, y, z = station.WRIST_CAM_WXYZ
    link6_cam = transform(station.WRIST_CAM_POS[side], quat_xyzw_to_matrix(np.array([x, y, z, w])))
    dream = transform(station.from_dream(station.DREAM_BASE_LINKS[side]), np.eye(3)) @ dream_link6(q) @ link6_cam
    cam = station.wrist_camera(side)
    fk = forward_kinematics(YAM_URDF, {f"joint{i + 1}": v for i, v in enumerate(q)})["gripper"]
    ours = transform(station.BASES[side], np.eye(3)) @ fk @ transform(cam.pos, camera_rot(cam))
    assert np.linalg.norm(ours[:3, 3] - dream[:3, 3]) < 5e-5
    assert rotation_angle_deg(ours[:3, :3], dream[:3, :3]) < 0.01


def test_calibrated_top_camera_looks_60_degrees_down_over_the_arms():
    cam = station.top_camera("calibrated")
    assert cam.convention == "opengl"
    assert cam.pos == pytest.approx((0.08600512 - 0.2525, -0.009, 1.70432055 - 0.0125), abs=1e-7)
    r = camera_rot(cam)
    view, right = r @ (0, 0, -1), r @ (1, 0, 0)
    assert math.degrees(math.asin(-view[2])) == pytest.approx(60.0, abs=1e-3)
    assert view[1] == pytest.approx(0.0, abs=1e-6) and view[0] > 0
    assert right == pytest.approx((0.0, -1.0, 0.0), abs=1e-5)
    assert cam.pos[2] - station.TABLE_TOP == pytest.approx(0.944, abs=1e-3)


def test_demo_top_camera_is_lower_and_pitched_up():
    calibrated, demo = station.top_camera("calibrated"), station.top_camera("demo")
    assert np.subtract(calibrated.pos, demo.pos) == pytest.approx((0.0, 0.0, 0.15))
    view = camera_rot(demo) @ (0, 0, -1)
    assert math.degrees(math.asin(-view[2])) == pytest.approx(55.0, abs=1e-3)
    with pytest.raises(ValueError):
        station.top_camera("closer")


def test_lenses_are_the_calibration_after_the_640x360_crop():
    top = station.top_camera().lens
    assert (top.width, top.height) == (640, 360)
    assert (top.fx, top.fy, top.cx) == station.TOP_CAM_K[:3]
    assert top.cy == pytest.approx(station.TOP_CAM_K[3] - 60)
    assert math.degrees(2 * math.atan(320 / top.fx)) == pytest.approx(78.42, abs=0.01)
    assert not top.is_ideal
    wrist = station.wrist_camera("left").lens
    assert wrist.cy == pytest.approx(176.146) and wrist.fx == pytest.approx(390.666)


def test_wrist_camera_faces_along_the_fingers_tilted_25_degrees():
    r = camera_rot(station.wrist_camera("left"))
    view = r @ (0, 0, -1)  # in the gripper frame, whose +z is the approach axis
    assert math.degrees(math.acos(view[2])) == pytest.approx(25.35, abs=0.05)


def test_lens_resizing_and_focal_length():
    lens = Lens(392.0, 391.0, 318.0, 178.0, 640, 360)
    half = lens.resized(320, 180)
    assert (half.fx, half.fy, half.cx, half.cy) == pytest.approx((196.0, 195.5, 159.0, 89.0))
    assert lens.focal_mm == pytest.approx(392.0 * 20.955 / 640)


def test_tabletop_rig_keeps_its_lenses():
    assert TABLETOP.top_cam.lens.focal_mm == pytest.approx(D435_FOCAL)
    assert TABLETOP.wrist_cams["left"].lens.focal_mm == pytest.approx(D405_FOCAL)
    assert TABLETOP.top_cam.lens.is_ideal and not TABLETOP.dream_look
    assert station.rig("demo").dream_look and station.rig("demo").name == "station-demo"


def test_link6_is_our_gripper_turned_half_a_turn():
    q = matrix_to_quat_xyzw(station.LINK6_IN_GRIPPER_ROT)
    assert abs(q[2]) == pytest.approx(1.0)
