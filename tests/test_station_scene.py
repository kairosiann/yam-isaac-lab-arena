"""Dream's station scene: object placement, the camera stand, the arms' look and colours."""

import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from yam_arena import ASSET_DIR, YAM_URDF, station
from yam_arena.analysis import read_stl
from yam_arena.urdf import quat_xyzw_to_matrix, rpy_matrix


def test_table_and_floor_follow_dreams_scene():
    assert station.TABLE_TOP == pytest.approx(0.7475)
    assert station.TABLE_CENTER == pytest.approx((0.36, 0.0, 0.7175))
    assert station.TABLE_SIZE == (1.10, 1.30, 0.06)
    assert station.TABLE_TOP - station.FLOOR_Z == pytest.approx(0.76)


def test_items_sit_where_dream_puts_them():
    want = {
        "apple": (0.135, 0.15),
        "orange": (0.11, 0.0),
        "banana": (0.155, -0.115),
        "bowl": (0.28, 0.0),
        "plate": (0.285, -0.22),
    }
    for name, xy in want.items():
        assert station.ITEMS[name].xy == pytest.approx(xy)
    assert station.BOTTLE_XY == pytest.approx((0.35, 0.22))
    assert set(station.FREE_OBJECTS) == {"apple", "orange", "banana", "bottle"}
    assert station.ITEMS["bowl"].mass is None and station.ITEMS["plate"].mass is None


def test_placement_scales_and_rests_an_asset_on_the_table():
    item = station.Item("some_asset", (0.3, -0.1), extent=0.07, mass=0.1)
    lo, hi = np.array([-2.0, -3.0, 1.0]), np.array([4.0, 1.0, 5.0])  # off-centre, z up from 1
    scale, pos, rot = station.placement(item, lo, hi)
    assert scale == pytest.approx(0.07 / 6.0)
    r = quat_xyzw_to_matrix(np.array(rot))
    assert r @ (1, 0, 0) == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)  # Dream's frame, turned into ours
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]) * scale
    world = corners @ r.T + pos
    assert world[:, 2].min() == pytest.approx(station.TABLE_TOP + station.REST_GAP)
    assert world.mean(axis=0)[:2] == pytest.approx(item.xy)
    assert max(np.ptp(world[:, 0]), np.ptp(world[:, 1])) == pytest.approx(0.07)


@pytest.mark.parametrize("direction", [(0, 0, 1), (0, 0, -1), (1, 0, 0), (0.1, 0.3, -1.0), (-0.4, 0.2, 0.3)])
def test_aim_turns_z_onto_a_direction(direction):
    d = np.array(direction, dtype=float) / np.linalg.norm(direction)
    assert quat_xyzw_to_matrix(np.array(station.aim(direction))) @ (0, 0, 1) == pytest.approx(d, abs=1e-9)


def test_lamp_hangs_where_dreams_lamp_does():
    assert station.LAMP_POS == pytest.approx((0.04, -0.8, 2.6875))


@pytest.mark.parametrize("variant", station.TOP_CAMS)
def test_camera_stand_follows_the_top_camera(variant):
    top = station.top_camera(variant)
    parts = station.camera_stand(top)
    foot, mast, arm, housing = (parts[k] for k in ("stand_foot", "stand_mast", "stand_arm", "camera_housing"))
    assert foot[2][2] - foot[1][2] / 2 == pytest.approx(station.TABLE_TOP)
    radius, length = mast[1]
    assert quat_xyzw_to_matrix(np.array(mast[3])) @ (0, 0, 1) == pytest.approx((0, 0, 1))
    assert mast[2][2] + length / 2 == pytest.approx(top.pos[2])
    assert mast[2][0] == pytest.approx(top.pos[0] - 0.07) and radius == 0.012
    tip = np.array(arm[2]) + quat_xyzw_to_matrix(np.array(arm[3])) @ (0, 0, arm[1][1] / 2)
    assert tip == pytest.approx(np.array(top.pos) + (-0.02, 0.0, 0.02))
    behind = np.array(housing[2]) - top.pos
    view = quat_xyzw_to_matrix(np.array(top.rot_xyzw)) @ (0, 0, -1)
    assert behind == pytest.approx(-0.014 * view)


def _our_visual_vertices(link: str) -> np.ndarray:
    visual = ET.parse(YAM_URDF).getroot().find(f"link[@name='{link}']/visual")
    origin = visual.find("origin")
    rot = rpy_matrix(np.array([float(v) for v in origin.get("rpy").split()]))
    pos = np.array([float(v) for v in origin.get("xyz").split()])
    return read_stl(ASSET_DIR / visual.find("geometry/mesh").get("filename")).reshape(-1, 3) @ rot.T + pos


@pytest.mark.parametrize("link", ["link2", "link3"])
def test_dreams_black_and_white_parts_cover_our_link_mesh(link):
    parts = np.concatenate(
        [
            read_stl(path).reshape(-1, 3) @ station.ARM_PART_ROT.T + station.ARM_PART_POS[link]
            for path in station.ARM_PARTS[link].values()
        ]
    )
    ours = _our_visual_vertices(link)
    assert parts.min(axis=0) == pytest.approx(ours.min(axis=0), abs=5e-4)
    assert parts.max(axis=0) == pytest.approx(ours.max(axis=0), abs=5e-4)


def test_colours_convert_to_linear():
    assert station.linear_rgb((0.0, 1.0, 0.5)) == pytest.approx((0.0, 1.0, 0.21404), abs=1e-5)
    assert math.isclose(sum(station.ARM_WHITE) / 3, 0.945)
