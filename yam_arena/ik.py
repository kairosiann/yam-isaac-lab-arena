"""Damped-least-squares IK for one YAM arm on the URDF's kinematics. NumPy only.

The stage tests use it to plan scripted motions (the 2.3 demo's pick-and-place and plate press,
reaching into the camera body) as joint targets, which the embodiment's 14-D joint action then
executes. Stage 2 checks that the simulator's link poses match this URDF FK, so the plan is valid
in the sim. The TCP is ``TCP_OFFSET`` in the `gripper` link frame (approach axis +Z).
"""

from __future__ import annotations

import numpy as np

from yam_arena import YAM_URDF
from yam_arena.constants import ARM_JOINTS, TCP_OFFSET
from yam_arena.urdf import forward_kinematics, parse_joints, quat_xyzw_to_matrix

_J = parse_joints(YAM_URDF)
LOWER = np.array([_J[n].lower for n in ARM_JOINTS])
UPPER = np.array([_J[n].upper for n in ARM_JOINTS])


def tcp_pose(q: np.ndarray, base_pos=(0.0, 0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """(position, rotation matrix) of the TCP in the world for arm joints ``q``."""
    g = forward_kinematics(YAM_URDF, {n: float(v) for n, v in zip(ARM_JOINTS, q, strict=True)})["gripper"]
    return g[:3, :3] @ np.array(TCP_OFFSET) + g[:3, 3] + np.asarray(base_pos), g[:3, :3]


def _rot_error(r_target: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Rotation vector taking r to r_target, in the world frame (robust up to 180 deg)."""
    q = matrix_to_quat_xyzw(r_target @ r.T)
    if q[3] < 0:
        q = -q
    v = q[:3]
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.zeros(3)
    return v / n * (2.0 * np.arctan2(n, q[3]))


def solve(
    target_pos,
    target_rot_xyzw=None,
    q_init=None,
    base_pos=(0.0, 0.0, 0.0),
    iters: int = 200,
    damping: float = 0.05,
    rot_weight: float = 0.3,
    tol: float = 1e-4,
) -> tuple[np.ndarray, float, float]:
    """Joint angles reaching ``target_pos`` (and ``target_rot_xyzw`` if given).

    Returns (q, position error [m], rotation error [rad]).
    """
    q = np.array(q_init if q_init is not None else [0.0, 1.0, 1.0, -0.6, 0.0, 0.0], dtype=float)
    r_target = quat_xyzw_to_matrix(np.asarray(target_rot_xyzw, dtype=float)) if target_rot_xyzw is not None else None
    target_pos = np.asarray(target_pos, dtype=float)
    eps = 1e-6
    for _ in range(iters):
        p, r = tcp_pose(q, base_pos)
        err = [target_pos - p]
        if r_target is not None:
            err.append(rot_weight * _rot_error(r_target, r))
        err = np.concatenate(err)
        if np.linalg.norm(err) < tol:
            break
        jac = np.zeros((len(err), 6))
        for i in range(6):
            dq = q.copy()
            dq[i] += eps
            p2, r2 = tcp_pose(dq, base_pos)
            col = [(p2 - p) / eps]
            if r_target is not None:
                col.append(rot_weight * (_rot_error(r2, r) / eps))
            jac[:, i] = np.concatenate(col)
        step = jac.T @ np.linalg.solve(jac @ jac.T + damping**2 * np.eye(len(err)), err)
        q = np.clip(q + step, LOWER, UPPER)
    p, r = tcp_pose(q, base_pos)
    rot_err = float(np.linalg.norm(_rot_error(r_target, r))) if r_target is not None else 0.0
    return q, float(np.linalg.norm(target_pos - p)), rot_err


def smoothstep(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * (3 - 2 * s)


def slerp_matrix(r0: np.ndarray, r1: np.ndarray, s: float) -> np.ndarray:
    """Rotation interpolated from r0 to r1."""
    from yam_arena.urdf import axis_angle_matrix

    w = _rot_error(r1, r0)
    angle = np.linalg.norm(w)
    if angle < 1e-9:
        return r0
    return axis_angle_matrix(w / angle, s * angle) @ r0


def matrix_to_quat_xyzw(r: np.ndarray) -> np.ndarray:
    tr = np.trace(r)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        w, x, y, z = (r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        w, x, y, z = (r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        w, x, y, z = (r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s
    return np.array([x, y, z, w])
