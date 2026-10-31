"""Action terms for one YAM arm, and the observation/reset helpers that go with them.

Import only after the app has started.

- :class:`YamArmAction` (6-D): absolute joint1..joint6 targets in rad, clamped to the joint
  limits. ``apply_actions`` runs every physics substep, so it also feeds the arm's gravity
  compensation torques as a feed-forward effort, the way i2rt drives the real arm (joint PD
  plus gravity compensation). In 2.3 this was ``hold_arms()`` in the stepping loop.
- :class:`YamGripperAction` (1-D): normalized opening g in [0, 1] (0 closed, 1 open); both
  finger joints track ``g * FINGER_OPEN``.

The bimanual action vector is these four terms in order, 14-D:
``[left j1..j6, left grip, right j1..j6, right grip]`` (inspect-robots-yam's packing).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS, FINGER_OPEN, WRIST_CAM_POS, WRIST_CAM_ROT_XYZW


class YamArmAction(ActionTerm):
    cfg: YamArmActionCfg
    _asset: Articulation

    def __init__(self, cfg: YamArmActionCfg, env):
        super().__init__(cfg, env)
        ids, names = self._asset.find_joints(list(cfg.joint_names), preserve_order=True)
        assert list(names) == list(cfg.joint_names), f"{cfg.asset_name}: expected {cfg.joint_names}, got {names}"
        self._joint_ids = list(ids)
        default = self._asset.data.default_joint_pos.torch[:, self._joint_ids].clone()
        self._default = default
        self._raw = default.clone()
        self._processed = default.clone()
        limits = self._asset.data.joint_pos_limits.torch[:, self._joint_ids]
        self._low, self._high = limits[..., 0].clone(), limits[..., 1].clone()
        # Optional wrist camera that is moved onto its gripper mount every substep ("follow" mode).
        self._camera = None
        if cfg.wrist_camera is not None and cfg.wrist_camera in env.scene.sensors:
            self._camera = env.scene[cfg.wrist_camera]
            self._gripper_body = self._asset.find_bodies("gripper")[0][0]
            n = self.num_envs
            self._mount_pos = torch.tensor([WRIST_CAM_POS], device=self.device).repeat(n, 1)
            self._mount_quat = torch.tensor([WRIST_CAM_ROT_XYZW], device=self.device).repeat(n, 1)

    @property
    def action_dim(self) -> int:
        return len(self._joint_ids)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed

    def process_actions(self, actions: torch.Tensor):
        self._raw[:] = actions
        self._processed[:] = torch.clamp(actions, self._low, self._high)

    def apply_actions(self):
        command = self._asset.actuators.target_command
        command.set_position_index(value=self._processed, joint_ids=self._joint_ids)
        if self.cfg.gravity_compensation:
            gravity = self._asset.data.gravity_compensation_forces.torch
            if gravity.shape[1] == self._asset.num_joints + 6:  # floating base: 6 root DOFs come first
                gravity = gravity[:, 6:]
            gravity = gravity[:, self._joint_ids]
            command.set_effort_index(value=gravity.contiguous(), joint_ids=self._joint_ids)
        if self._camera is not None:
            self.update_wrist_camera()

    def update_wrist_camera(self) -> None:
        """Put the wrist camera at its mount on the current `gripper` link pose (ROS convention)."""
        from isaaclab.utils.math import quat_apply, quat_mul

        pose = self._asset.data.body_link_pose_w.torch[:, self._gripper_body]
        pos, quat = pose[:, :3], pose[:, 3:7]
        self._camera.set_world_poses(
            pos + quat_apply(quat, self._mount_pos), quat_mul(quat, self._mount_quat), convention="ros"
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._raw[ids] = self._default[ids]
        self._processed[ids] = self._default[ids]


@configclass
class YamArmActionCfg(ActionTermCfg):
    class_type: type = YamArmAction
    joint_names: list[str] = list(ARM_JOINTS)
    gravity_compensation: bool = True
    wrist_camera: str | None = None
    """Scene name of a wrist camera to move onto this arm's gripper mount each substep, or None."""


class YamGripperAction(ActionTerm):
    cfg: YamGripperActionCfg
    _asset: Articulation

    def __init__(self, cfg: YamGripperActionCfg, env):
        super().__init__(cfg, env)
        ids, names = self._asset.find_joints(list(FINGER_JOINTS), preserve_order=True)
        assert list(names) == list(FINGER_JOINTS), f"{cfg.asset_name}: expected {FINGER_JOINTS}, got {names}"
        self._finger_ids = list(ids)
        default = self._asset.data.default_joint_pos.torch[:, self._finger_ids].mean(dim=1, keepdim=True)
        self._default = torch.clamp(default / FINGER_OPEN, 0.0, 1.0)
        self._raw = self._default.clone()
        self._processed = self._default.clone()

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed

    def process_actions(self, actions: torch.Tensor):
        self._raw[:] = actions
        self._processed[:] = torch.clamp(actions, 0.0, 1.0)

    def apply_actions(self):
        target = (self._processed * FINGER_OPEN).expand(-1, len(self._finger_ids)).contiguous()
        self._asset.actuators.target_command.set_position_index(value=target, joint_ids=self._finger_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._raw[ids] = self._default[ids]
        self._processed[ids] = self._default[ids]


@configclass
class YamGripperActionCfg(ActionTermCfg):
    class_type: type = YamGripperAction


# -- observations ------------------------------------------------------------------------


def arm_state_7(env, asset_cfg: SceneEntityCfg = MISSING) -> torch.Tensor:
    """One arm's ``[j1..j6, gripper]``: joints in rad, gripper normalized (0 closed, 1 open).

    Both fingers mirror one rack, so the gripper is their mean opening, as in the 2.3 wrapper.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    q = robot.data.joint_pos.torch
    arm_ids = robot.find_joints(list(ARM_JOINTS), preserve_order=True)[0]
    finger_ids = robot.find_joints(list(FINGER_JOINTS), preserve_order=True)[0]
    grip = torch.clamp(q[:, finger_ids].mean(dim=1, keepdim=True) / FINGER_OPEN, 0.0, 1.0)
    return torch.cat([q[:, arm_ids], grip], dim=1)


def bimanual_joint_pos_14(env, left: str = "left_robot", right: str = "right_robot") -> torch.Tensor:
    """The 14-D ``joint_pos`` state of inspect-robots-yam: left 7-D then right 7-D."""
    return torch.cat([arm_state_7(env, SceneEntityCfg(left)), arm_state_7(env, SceneEntityCfg(right))], dim=1)


# -- events ------------------------------------------------------------------------------


def reset_arms_to_default(env, env_ids: torch.Tensor, asset_names: tuple[str, ...] = ("left_robot", "right_robot")):
    """Put each arm's joints (and joint targets) back at the spawn pose with zero velocity."""
    for name in asset_names:
        robot: Articulation = env.scene[name]
        q = robot.data.default_joint_pos.torch[env_ids].clone()
        v = robot.data.default_joint_vel.torch[env_ids].clone()
        robot.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
        robot.write_joint_velocity_to_sim_index(velocity=v, env_ids=env_ids)
        robot.actuators.target_command.set_position_index(value=q, env_ids=env_ids)
        robot.actuators.target_command.set_velocity_index(value=v, env_ids=env_ids)
