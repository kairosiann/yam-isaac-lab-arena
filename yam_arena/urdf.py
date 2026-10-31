"""URDF parsing and forward kinematics in plain NumPy (no Isaac).

The stage tests compare what the converted USD and the simulator do against the URDF, which
stays the source of truth: joint types and limits (stage 1) and link poses at non-zero joint
angles (stage 2), which catches axis or sign flips made by the URDF importer.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class UrdfJoint:
    name: str
    type: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray
    lower: float | None
    upper: float | None


def _vec(text: str | None, default: tuple[float, float, float]) -> np.ndarray:
    return np.array([float(v) for v in text.split()]) if text else np.array(default, dtype=float)


def parse_joints(urdf: str | Path) -> dict[str, UrdfJoint]:
    return dict(_parse_joints(str(urdf)))


@lru_cache(maxsize=8)
def _parse_joints(urdf: str) -> tuple[tuple[str, UrdfJoint], ...]:
    joints = {}
    for j in ET.parse(urdf).getroot().iter("joint"):
        origin = j.find("origin")
        axis = j.find("axis")
        limit = j.find("limit")
        joints[j.get("name")] = UrdfJoint(
            name=j.get("name"),
            type=j.get("type"),
            parent=j.find("parent").get("link"),
            child=j.find("child").get("link"),
            xyz=_vec(origin.get("xyz") if origin is not None else None, (0, 0, 0)),
            rpy=_vec(origin.get("rpy") if origin is not None else None, (0, 0, 0)),
            axis=_vec(axis.get("xyz") if axis is not None else None, (1, 0, 0)),
            lower=float(limit.get("lower")) if limit is not None and limit.get("lower") else None,
            upper=float(limit.get("upper")) if limit is not None and limit.get("upper") else None,
        )
    return tuple(joints.items())


def collision_counts(urdf: str | Path) -> dict[str, int]:
    """Number of <collision> elements per link."""
    return {link.get("name"): len(link.findall("collision")) for link in ET.parse(urdf).getroot().iter("link")}


def rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def forward_kinematics(urdf: str | Path, q: dict[str, float], root: str = "base") -> dict[str, np.ndarray]:
    """4x4 pose of every link in the root link's frame for joint positions ``q`` (rad / m)."""
    joints = parse_joints(urdf)
    by_parent: dict[str, list[UrdfJoint]] = {}
    for j in joints.values():
        by_parent.setdefault(j.parent, []).append(j)
    poses = {root: np.eye(4)}
    stack = [root]
    while stack:
        link = stack.pop()
        for j in by_parent.get(link, []):
            t_origin = np.eye(4)
            t_origin[:3, :3] = rpy_matrix(j.rpy)
            t_origin[:3, 3] = j.xyz
            t_joint = np.eye(4)
            value = q.get(j.name, 0.0)
            if j.type in ("revolute", "continuous"):
                t_joint[:3, :3] = axis_angle_matrix(j.axis, value)
            elif j.type == "prismatic":
                t_joint[:3, 3] = j.axis / np.linalg.norm(j.axis) * value
            poses[j.child] = poses[link] @ t_origin @ t_joint
            stack.append(j.child)
    return poses


def quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def rotation_angle_deg(r_a: np.ndarray, r_b: np.ndarray) -> float:
    """Angle of the relative rotation between two rotation matrices, in degrees."""
    c = (np.trace(r_a.T @ r_b) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
