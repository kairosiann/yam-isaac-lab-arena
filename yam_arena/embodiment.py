"""Arena embodiment: one or two I2RT YAM arms with joint-position control. Import after app start.

Scene keys are ``left_robot`` / ``right_robot`` (Arena's two-root convention, see
docs/pages/concepts/embodiment/index.rst). Prims are ``{ENV_REGEX_NS}/LeftArm`` and
``{ENV_REGEX_NS}/RightArm``.

Action (14-D for two arms): ``[left j1..j6, left grip, right j1..j6, right grip]``, joints as
absolute targets in rad, grippers normalized 0 (closed) .. 1 (open). Gravity compensation is
fed every physics substep by the arm action term.

Cameras (``enable_cameras=True``): ``top_cam`` (D435, fixed in the world) and ``left_cam`` /
``right_cam`` (D405 on each wrist). With ``wrist_cam_mount="link"`` the wrist cameras are USD
children of the ``gripper`` link; with ``"follow"`` they live at the env root and are moved
onto the mount every physics substep (the 2.3 port's workaround).
"""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.utils import configclass

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.utils.cameras import ArenaCameraCfg
from isaaclab_arena.utils.configclass import make_configclass
from isaaclab_arena.utils.pose import Pose

from yam_arena.actions import (
    YamArmActionCfg,
    YamGripperActionCfg,
    arm_state_7,
    bimanual_joint_pos_14,
    reset_arms_to_default,
)
from yam_arena.constants import (
    CAM_RENDER_HEIGHT,
    CAM_RENDER_WIDTH,
    D405_FOCAL,
    D435_FOCAL,
    LEFT_BASE_POS,
    OVERVIEW_CAM_POS,
    OVERVIEW_CAM_ROT_XYZW,
    RIGHT_BASE_POS,
    TOP_CAM_POS,
    TOP_CAM_ROT_XYZW,
    WRIST_CAM_POS,
    WRIST_CAM_ROT_XYZW,
)
from yam_arena.robot import asset_link_paths, yam_arm_cfg

SIDES = {
    "left": dict(key="left_robot", prim="LeftArm", base=LEFT_BASE_POS, cam="left_cam"),
    "right": dict(key="right_robot", prim="RightArm", base=RIGHT_BASE_POS, cam="right_cam"),
}


@configclass
class _PolicyObs(ObservationGroupCfg):
    joint_pos: ObservationTermCfg = MISSING

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class YamObservationsCfg:
    policy: _PolicyObs = MISSING


@register_asset
class BimanualYamEmbodiment(EmbodimentBase):
    """Two (or, for testing, one) YAM arms mounted on the near edge of the table."""

    name = "yam_bimanual"
    tags = ["embodiment"]

    def __init__(
        self,
        enable_cameras: bool = False,
        arms: tuple[str, ...] = ("left", "right"),
        gravity_compensation: bool = True,
        cam_height: int = CAM_RENDER_HEIGHT,
        cam_width: int = CAM_RENDER_WIDTH,
        wrist_cam_mount: str = "link",
        overview_cam: bool = False,
        contact_sensors: bool = True,
        robot_contact_offset: float | None = None,
    ):
        super().__init__(enable_cameras=enable_cameras, initial_pose=None)
        assert set(arms) <= set(SIDES) and arms, f"arms must be a subset of {tuple(SIDES)}"
        assert wrist_cam_mount in ("link", "follow"), wrist_cam_mount
        self.arms = tuple(a for a in SIDES if a in arms)  # always left before right
        self.wrist_cam_mount = wrist_cam_mount
        self.link_paths = asset_link_paths()
        self.scene_keys = tuple(SIDES[a]["key"] for a in self.arms)

        # -- scene: articulations (+ one contact sensor per arm covering every link) ----------
        fields = []
        for side in self.arms:
            s = SIDES[side]
            cfg = yam_arm_cfg("{ENV_REGEX_NS}/" + s["prim"], s["base"], contact_offset=robot_contact_offset)
            fields.append((s["key"], ArticulationCfg, cfg))
        if contact_sensors:
            for side in self.arms:
                s = SIDES[side]
                fields.append((f"{side}_contacts", ContactSensorCfg, self._contact_sensor_cfg(s["prim"])))
        self.scene_config = make_configclass("YamSceneCfg", fields)()

        # -- actions: field order is the action layout -----------------------------------
        action_fields = []
        for side in self.arms:
            key = SIDES[side]["key"]
            follow_cam = SIDES[side]["cam"] if (enable_cameras and wrist_cam_mount == "follow") else None
            arm = YamArmActionCfg(asset_name=key, gravity_compensation=gravity_compensation, wrist_camera=follow_cam)
            action_fields.append((f"{side}_arm", YamArmActionCfg, arm))
            action_fields.append((f"{side}_gripper", YamGripperActionCfg, YamGripperActionCfg(asset_name=key)))
        self.action_config = make_configclass("YamActionsCfg", action_fields)()

        # -- observations: the inspect-robots-yam state layout ---------------------------
        if self.arms == ("left", "right"):
            term = ObservationTermCfg(func=bimanual_joint_pos_14)
        else:
            term = ObservationTermCfg(func=arm_state_7, params={"asset_cfg": SceneEntityCfg(self.scene_keys[0])})
        self.observation_config = YamObservationsCfg(policy=_PolicyObs(joint_pos=term))

        # -- events: arms back to the ready pose on reset --------------------------------
        reset = EventTermCfg(func=reset_arms_to_default, mode="reset", params={"asset_names": self.scene_keys})
        self.event_config = make_configclass("YamEventsCfg", [("reset_arms", EventTermCfg, reset)])()

        # -- cameras ---------------------------------------------------------------------
        self.camera_config = self._camera_cfg(cam_height, cam_width, overview_cam)

    # -- builders ----------------------------------------------------------------------------

    def _contact_sensor_cfg(self, prim: str) -> ContactSensorCfg:
        # Isaac Lab 3.0 splits the path into a parent expression and a leaf-name regex and collects
        # every contact-reporting descendant, so this covers the importer's nested links.
        return ContactSensorCfg(prim_path="{ENV_REGEX_NS}/" + prim + "/.*", history_length=0, track_air_time=False)

    def _wrist_cam_cfg(self, side: str, height: int, width: int) -> CameraCfg:
        s = SIDES[side]
        spawn = sim_utils.PinholeCameraCfg(focal_length=D405_FOCAL, clipping_range=(0.01, 20.0))
        if self.wrist_cam_mount == "link":
            prim = "{ENV_REGEX_NS}/" + s["prim"] + self.link_paths["gripper"] + f"/{side}_wrist_cam"
            offset = CameraCfg.OffsetCfg(pos=WRIST_CAM_POS, rot=WRIST_CAM_ROT_XYZW, convention="ros")
        else:
            prim = "{ENV_REGEX_NS}/" + s["prim"].replace("Arm", "WristCam")
            offset = CameraCfg.OffsetCfg()
        return CameraCfg(
            prim_path=prim,
            height=height,
            width=width,
            data_types=["rgb"],
            spawn=spawn,
            offset=offset,
            update_latest_camera_pose=True,
        )

    def _camera_cfg(self, height: int, width: int, overview: bool) -> ArenaCameraCfg:
        fields = [
            (
                "top_cam",
                CameraCfg,
                CameraCfg(
                    prim_path="{ENV_REGEX_NS}/TopCam",
                    height=height,
                    width=width,
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(focal_length=D435_FOCAL, clipping_range=(0.05, 20.0)),
                    offset=CameraCfg.OffsetCfg(pos=TOP_CAM_POS, rot=TOP_CAM_ROT_XYZW, convention="world"),
                ),
            )
        ]
        for side in self.arms:
            fields.append((SIDES[side]["cam"], CameraCfg, self._wrist_cam_cfg(side, height, width)))
        if overview:
            fields.append(
                (
                    "overview_cam",
                    CameraCfg,
                    CameraCfg(
                        prim_path="{ENV_REGEX_NS}/OverviewCam",
                        height=height,
                        width=width,
                        data_types=["rgb"],
                        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 20.0)),
                        offset=CameraCfg.OffsetCfg(
                            pos=OVERVIEW_CAM_POS, rot=OVERVIEW_CAM_ROT_XYZW, convention="world"
                        ),
                    ),
                )
            )
        return make_configclass("YamCameraCfg", fields, bases=(ArenaCameraCfg,))()

    # -- EmbodimentBase overrides: the base class assumes one root called `robot` ----------

    def get_scene_key(self) -> str:
        return self.scene_keys[0]

    def get_initial_pose(self) -> Pose:
        cfg = getattr(self.scene_config, self.scene_keys[0])
        return Pose(position_xyz=tuple(cfg.init_state.pos), rotation_xyzw=tuple(cfg.init_state.rot))

    def set_joint_initial_pos(self, joint_pos) -> None:
        for key in self.scene_keys:
            getattr(self.scene_config, key).init_state.joint_pos.update(joint_pos)

    def get_placement_geometry_source(self):
        raise NotImplementedError(
            "BimanualYamEmbodiment has two independent roots and no single placement geometry; "
            "build it with ArenaEnvBuilderCfg(solve_relations=False)."
        )

    def get_ee_frame_transformer_names(self) -> list[str]:
        return []
