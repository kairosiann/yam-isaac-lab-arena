"""Measurement helpers for the stage tests. NumPy only (no Isaac).

- FK comparison: simulated link poses vs the URDF's forward kinematics at the measured joints.
- Collision geometry: the CoACD hulls the URDF lists, placed in the world from simulated link
  poses, and signed distances between them and other hulls or boxes. This is how stages 3-4
  report *how close* bodies got (and how deep any overlap went), not just "looks fine".

Signed distances use each convex hull's face planes: for a point p and a hull with outward
planes n_i . x = d_i, s(p) = max_i (n_i . p - d_i). s < 0 means p is inside, and -s is then its
depth below the nearest face. For p outside, s is a lower bound on the true distance (exact
when the nearest feature is a face). Penetration between two hull sets is reported as the
deepest vertex of either set inside the other.
"""

from __future__ import annotations

import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from yam_arena import ASSET_DIR, YAM_URDF
from yam_arena.urdf import forward_kinematics, quat_xyzw_to_matrix, rotation_angle_deg, rpy_matrix


def pose7_to_matrix(pose: np.ndarray) -> np.ndarray:
    """(x, y, z, qx, qy, qz, qw) -> 4x4."""
    t = np.eye(4)
    t[:3, :3] = quat_xyzw_to_matrix(np.asarray(pose[3:7], dtype=float))
    t[:3, 3] = pose[:3]
    return t


def fk_errors(body_names: list[str], body_poses_w: np.ndarray, root_pose_w: np.ndarray, q: dict[str, float]):
    """Per-link (position error [mm], rotation error [deg]) of simulated vs URDF FK, in the base frame."""
    fk = forward_kinematics(YAM_URDF, q)
    t_root_inv = np.linalg.inv(pose7_to_matrix(root_pose_w))
    errors = {}
    for name, pose in zip(body_names, body_poses_w, strict=True):
        if name not in fk:
            continue
        sim = t_root_inv @ pose7_to_matrix(pose)
        ref = fk[name]
        errors[name] = (
            float(np.linalg.norm(sim[:3, 3] - ref[:3, 3]) * 1000.0),
            rotation_angle_deg(sim[:3, :3], ref[:3, :3]),
        )
    return errors


# -- collision hulls ---------------------------------------------------------------------


def read_stl(path: Path) -> np.ndarray:
    """Triangles (n, 3, 3) from a binary or ASCII STL."""
    data = path.read_bytes()
    if data[:5] == b"solid" and b"facet" in data[:400]:
        lines = data.decode().splitlines()
        verts = [list(map(float, line.split()[1:4])) for line in lines if line.strip().startswith("vertex")]
        return np.array(verts).reshape(-1, 3, 3)
    (n,) = struct.unpack("<I", data[80:84])
    rec = np.frombuffer(data[84 : 84 + n * 50], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    return rec["v"].astype(float)


@dataclass
class Hull:
    vertices: np.ndarray  # (n, 3) in the link frame
    planes: np.ndarray  # (f, 4): outward unit normal, offset (n . x = d) in the link frame


def _hull_from_triangles(tris: np.ndarray) -> Hull:
    verts = np.unique(tris.reshape(-1, 3).round(9), axis=0)
    centroid = verts.mean(axis=0)
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    norms = np.linalg.norm(normals, axis=1)
    keep = norms > 1e-12
    normals = normals[keep] / norms[keep, None]
    d = np.einsum("ij,ij->i", normals, tris[keep, 0])
    # Make every plane face away from the centroid (robust to inconsistent STL winding).
    flip = (normals @ centroid - d) > 0
    normals[flip] *= -1
    d[flip] *= -1
    return Hull(vertices=verts, planes=np.c_[normals, d])


@lru_cache(maxsize=1)
def link_hulls() -> dict[str, list[Hull]]:
    """Every <collision> hull of the URDF, in its link's frame."""
    hulls: dict[str, list[Hull]] = {}
    for link in ET.parse(YAM_URDF).getroot().iter("link"):
        for col in link.findall("collision"):
            origin = col.find("origin")
            t = np.eye(4)
            if origin is not None:
                t[:3, :3] = rpy_matrix(np.array([float(v) for v in origin.get("rpy", "0 0 0").split()]))
                t[:3, 3] = [float(v) for v in origin.get("xyz", "0 0 0").split()]
            tris = read_stl(ASSET_DIR / col.find("geometry/mesh").get("filename"))
            tris = tris @ t[:3, :3].T + t[:3, 3]
            hulls.setdefault(link.get("name"), []).append(_hull_from_triangles(tris))
    return hulls


def world_hulls(link_poses: dict[str, np.ndarray], links: list[str] | None = None) -> list[tuple[str, Hull]]:
    """Hulls of ``links`` moved into the world by 4x4 ``link_poses``."""
    out = []
    for name, pose in link_poses.items():
        if links is not None and name not in links:
            continue
        r, p = pose[:3, :3], pose[:3, 3]
        for h in link_hulls().get(name, []):
            n_w = h.planes[:, :3] @ r.T
            d_w = h.planes[:, 3] + n_w @ p
            out.append((name, Hull(vertices=h.vertices @ r.T + p, planes=np.c_[n_w, d_w])))
    return out


def _signed(points: np.ndarray, hull: Hull) -> np.ndarray:
    return (points @ hull.planes[:, :3].T - hull.planes[:, 3]).max(axis=1)


def hull_set_distance(a: list[tuple[str, Hull]], b: list[tuple[str, Hull]]):
    """(min signed distance [m], (link_a, link_b)) between two hull sets; negative = overlap depth."""
    best, pair = np.inf, None
    for name_a, ha in a:
        lo_a, hi_a = ha.vertices.min(0), ha.vertices.max(0)
        for name_b, hb in b:
            lo_b, hi_b = hb.vertices.min(0), hb.vertices.max(0)
            if np.maximum(lo_b - hi_a, lo_a - hi_b).max() > 0.05:
                continue  # AABBs more than 5 cm apart: not a contact candidate
            s = min(_signed(ha.vertices, hb).min(), _signed(hb.vertices, ha).min())
            if s < best:
                best, pair = float(s), (name_a, name_b)
    return best, pair


def box_signed_distance(points: np.ndarray, center: np.ndarray, rot: np.ndarray, size: np.ndarray) -> np.ndarray:
    """Exact signed distance of points to an oriented box (negative inside)."""
    local = (points - center) @ rot  # into box frame
    q = np.abs(local) - np.asarray(size) / 2
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(q.max(axis=1), 0.0)
    return outside + inside


def hulls_to_box(hulls: list[tuple[str, Hull]], center, rot, size):
    """(min signed distance of any hull vertex to the box [m], link) -- negative = inside the box."""
    best, link = np.inf, None
    for name, h in hulls:
        s = box_signed_distance(h.vertices, np.asarray(center), np.asarray(rot), np.asarray(size)).min()
        if s < best:
            best, link = float(s), name
    return best, link


def fk_link_poses_world(q: dict[str, float], base_pos) -> dict[str, np.ndarray]:
    """URDF FK link poses in the world for an arm whose base sits at ``base_pos`` (no rotation)."""
    poses = forward_kinematics(YAM_URDF, q)
    shift = np.eye(4)
    shift[:3, 3] = base_pos
    return {k: shift @ v for k, v in poses.items()}
