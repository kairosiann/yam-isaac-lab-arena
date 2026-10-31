"""Small helpers that read simulator state into NumPy. Import after the app has started."""

from __future__ import annotations

import numpy as np

from yam_arena.analysis import pose7_to_matrix
from yam_arena.constants import ARM_JOINTS, PAD_TIP_PAST_TCP, TCP_OFFSET


def link_poses(robot, env_index: int = 0) -> dict[str, np.ndarray]:
    """4x4 world pose of every link of an articulation."""
    poses = robot.data.body_link_pose_w.torch[env_index].cpu().numpy()
    return {name: pose7_to_matrix(p) for name, p in zip(robot.body_names, poses)}


def tcp_world(robot, env_index: int = 0, past_tcp: float = 0.0) -> np.ndarray:
    """World position of the TCP (or a point ``past_tcp`` further along the approach axis)."""
    g = link_poses(robot, env_index)["gripper"]
    return g[:3, :3] @ (np.array(TCP_OFFSET) + np.array([0.0, 0.0, past_tcp])) + g[:3, 3]


def pad_tips_world(robot, env_index: int = 0) -> np.ndarray:
    return tcp_world(robot, env_index, past_tcp=PAD_TIP_PAST_TCP)


def arm_q(robot, env_index: int = 0) -> np.ndarray:
    ids = robot.find_joints(ARM_JOINTS, preserve_order=True)[0]
    return robot.data.joint_pos.torch[env_index, ids].cpu().numpy()


def contact_forces(sensor, env_index: int = 0) -> dict[str, float]:
    """|net normal contact force| [N] per sensed body (PhysX reports normal forces only)."""
    forces = sensor.data.net_normal_forces_w.torch[env_index].cpu().numpy()
    return {name: float(np.linalg.norm(f)) for name, f in zip(sensor.body_names, forces)}
