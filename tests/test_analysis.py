import struct

import numpy as np
import pytest

from yam_arena import YAM_URDF
from yam_arena.analysis import (
    Hull,
    box_signed_distance,
    fk_errors,
    fk_link_poses_world,
    hull_set_distance,
    hulls_to_box,
    link_hulls,
    pose7_to_matrix,
    read_stl,
    world_hulls,
)
from yam_arena.constants import ARM_JOINTS, LINKS, READY
from yam_arena.ik import matrix_to_quat_xyzw
from yam_arena.urdf import axis_angle_matrix, collision_counts, forward_kinematics

Q_READY = dict(zip(ARM_JOINTS, READY, strict=True))
TETRAHEDRON = np.array(
    [
        [[0, 0, 0], [0, 1, 0], [1, 0, 0]],
        [[0, 0, 0], [1, 0, 0], [0, 0, 1]],
        [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    ],
    dtype=float,
)


def box_hull(center, size):
    c, h = np.asarray(center, dtype=float), np.asarray(size, dtype=float) / 2
    corners = c + h * np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    normals = np.vstack([np.eye(3), -np.eye(3)])
    return Hull(vertices=corners, planes=np.c_[normals, normals @ c + np.r_[h, h]])


def signed(points, hull):
    return (points @ hull.planes[:, :3].T - hull.planes[:, 3]).max(axis=1)


def transform(axis, angle, translation):
    t = np.eye(4)
    t[:3, :3] = axis_angle_matrix(np.array(axis, dtype=float), angle)
    t[:3, 3] = translation
    return t


def pose7(t):
    return np.r_[t[:3, 3], matrix_to_quat_xyzw(t[:3, :3])]


def test_pose7_is_position_then_xyzw_quaternion():
    s = np.sin(np.pi / 4)
    t = pose7_to_matrix(np.array([0.1, 0.2, 0.3, 0, 0, s, s]))
    np.testing.assert_allclose(t[:3, 3], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(t[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    np.testing.assert_array_equal(t[3], [0, 0, 0, 1])


def test_fk_errors_compare_link_poses_in_the_base_frame():
    root = transform([0, 0, 1], 0.7, [1.0, -2.0, 0.75])
    fk = forward_kinematics(YAM_URDF, Q_READY)
    world = {name: root @ fk[name] for name in LINKS}
    world["gripper"] = world["gripper"].copy()
    world["gripper"][:3, 3] += root[:3, :3] @ [0.001, 0.0, 0.0]
    world["link4"] = world["link4"] @ transform([1, 0, 0], np.radians(2.0), [0, 0, 0])
    names = [*LINKS, "camera"]
    poses = np.array([pose7(world[n]) for n in LINKS] + [pose7(np.eye(4))])
    errors = fk_errors(names, poses, pose7(root), Q_READY)
    assert set(errors) == set(LINKS)
    assert errors["gripper"] == pytest.approx((1.0, 0.0), abs=1e-4)
    assert errors["link4"] == pytest.approx((0.0, 2.0), abs=1e-4)
    assert all(mm < 1e-6 and deg < 1e-4 for name, (mm, deg) in errors.items() if name not in ("gripper", "link4"))


def test_read_stl_reads_ascii_and_binary(tmp_path):
    lines = ["solid tetrahedron"]
    for tri in TETRAHEDRON:
        lines += [
            "facet normal 0 0 0",
            "outer loop",
            *(f"vertex {x} {y} {z}" for x, y, z in tri),
            "endloop",
            "endfacet",
        ]
    (tmp_path / "ascii.stl").write_text("\n".join([*lines, "endsolid tetrahedron"]))
    records = b"".join(struct.pack("<12fH", 0, 0, 0, *tri.ravel(), 0) for tri in TETRAHEDRON)
    header = b"solid, but binary".ljust(80, b" ")
    (tmp_path / "binary.stl").write_bytes(header + struct.pack("<I", len(TETRAHEDRON)) + records)
    np.testing.assert_array_equal(read_stl(tmp_path / "ascii.stl"), TETRAHEDRON)
    np.testing.assert_array_equal(read_stl(tmp_path / "binary.stl"), TETRAHEDRON)


def test_collision_hulls_are_convex_with_outward_faces():
    hulls = link_hulls()
    assert {link: len(h) for link, h in hulls.items()} == collision_counts(YAM_URDF)
    for link_hulls_ in hulls.values():
        for hull in link_hulls_:
            np.testing.assert_allclose(np.linalg.norm(hull.planes[:, :3], axis=1), 1.0)
            assert signed(hull.vertices, hull).max() < 1e-4
            assert signed(hull.vertices.mean(axis=0, keepdims=True), hull)[0] < 0


def test_world_hulls_move_vertices_and_faces_together():
    pose = transform([1, 2, 3], 0.9, [0.4, -0.2, 1.0])
    moved = world_hulls({"gripper": pose, "link5": pose}, links=["gripper"])
    assert [name for name, _ in moved] == ["gripper"] * len(link_hulls()["gripper"])
    for (_, hull), original in zip(moved, link_hulls()["gripper"], strict=True):
        np.testing.assert_allclose(hull.vertices, original.vertices @ pose[:3, :3].T + pose[:3, 3])
        assert signed(hull.vertices, hull).max() < 1e-4
        assert signed(hull.vertices.mean(axis=0, keepdims=True), hull)[0] < 0


def test_hull_set_distance_is_the_gap_or_the_overlap_depth():
    a = [("a", box_hull((0, 0, 0), (0.1, 0.1, 0.1)))]
    small = (0.1, 0.06, 0.06)
    assert hull_set_distance(a, [("b", box_hull((0.12, 0, 0), small))]) == (pytest.approx(0.02), ("a", "b"))
    assert hull_set_distance(a, [("b", box_hull((0.09, 0, 0), small))]) == (pytest.approx(-0.01), ("a", "b"))
    assert hull_set_distance(a, [("b", box_hull((0.3, 0, 0), small))]) == (np.inf, None)


def test_box_signed_distance_is_exact_for_a_rotated_box():
    center, size = np.array([1.0, 2.0, 0.5]), np.array([0.4, 0.2, 0.1])
    rot = axis_angle_matrix(np.array([0, 0, 1.0]), np.pi / 2)
    offsets = np.array([[0, 0, 0], [0, 0.3, 0], [0.2, 0, 0], [0.13, 0.24, 0]])
    np.testing.assert_allclose(box_signed_distance(center + offsets, center, rot, size), [-0.05, 0.1, 0.1, 0.05])


def test_the_base_frame_origin_is_the_underside_of_the_base():
    hulls = world_hulls(fk_link_poses_world(Q_READY, (0.0, 0.0, 0.0)))
    mount = {"center": (0.0, 0.0, -0.02), "rot": np.eye(3), "size": (1.0, 1.0, 0.04)}
    distance, link = hulls_to_box(hulls, **mount)
    assert link == "base" and distance == pytest.approx(0.0, abs=0.002)
    assert hulls_to_box([h for h in hulls if h[0] != "base"], **mount)[0] > 0.05


def test_fk_link_poses_world_translates_without_rotating():
    local = forward_kinematics(YAM_URDF, Q_READY)
    world = fk_link_poses_world(Q_READY, (0.0, 0.3, 0.75))
    for name, pose in local.items():
        np.testing.assert_allclose(world[name][:3, 3] - pose[:3, 3], [0.0, 0.3, 0.75], atol=1e-12)
        np.testing.assert_array_equal(world[name][:3, :3], pose[:3, :3])
