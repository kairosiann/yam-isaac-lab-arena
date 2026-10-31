"""The built-in YAM tabletop as an Arena environment. Import after app start.

Two arms on the near edge of a table, a red 4 cm cube in front of the left arm, a white plate
in front of the right, a dome light, a ground plane and, new in this port, a static collider
the size of the top camera's D435 body at the camera's pose (the 2.3 scene had none, so an
arm could pass through the camera). NoTask: no success criteria, no time-out.

Physics: 240 Hz PhysX, 24 substeps per action (10 Hz control), one render per action.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab_physx.sim.schemas import PhysxCollisionPropertiesCfg, PhysxRigidBodyPropertiesCfg

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_library import DomeLight, GroundPlane
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.utils.pose import Pose

from yam_arena.constants import (
    CAMERA_MAST_POS,
    CAMERA_MAST_SIZE,
    CUBE_MASS,
    CUBE_POS,
    CUBE_SIZE,
    D435_BOX_POS,
    D435_BOX_ROT_XYZW,
    D435_SIZE_XYZ,
    DECIMATION,
    DOME_LIGHT_INTENSITY,
    GRIPPY_FRICTION,
    PHYSICS_DT,
    PLATE_HEIGHT,
    PLATE_MASS,
    PLATE_POS,
    PLATE_RADIUS,
    RENDER_INTERVAL,
    TABLE_CENTER_X,
    TABLE_HEIGHT,
    TABLE_SIZE,
)

ENV_NAME = "yam_bimanual_tabletop"
OBJECT_NAMES = ("cube", "plate")


def _box(name, size, pos, color, rot_xyzw=(0.0, 0.0, 0.0, 1.0)) -> Object:
    """A static box collider (no rigid body): table, camera body, mast."""
    return Object(
        name=name,
        object_type=ObjectType.BASE,
        spawner_cfg=sim_utils.CuboidCfg(
            size=tuple(size),
            collision_props=PhysxCollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.7),
        ),
        initial_pose=Pose(position_xyz=tuple(pos), rotation_xyzw=tuple(rot_xyzw)),
    )


def scene_assets(objects: bool = True, d435_collider: bool = True, camera_mast: bool = False) -> list:
    grippy = sim_utils.RigidBodyMaterialCfg(static_friction=GRIPPY_FRICTION[0], dynamic_friction=GRIPPY_FRICTION[1])
    assets = [
        GroundPlane(),
        DomeLight(spawner_cfg=sim_utils.DomeLightCfg(intensity=DOME_LIGHT_INTENSITY, color=(0.9, 0.9, 0.9))),
        _box("table", TABLE_SIZE, (TABLE_CENTER_X, 0.0, TABLE_HEIGHT - TABLE_SIZE[2] / 2), (0.55, 0.42, 0.30)),
    ]
    if objects:
        assets.append(
            Object(
                name="cube",
                object_type=ObjectType.RIGID,
                spawner_cfg=sim_utils.CuboidCfg(
                    size=(CUBE_SIZE,) * 3,
                    rigid_props=PhysxRigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=CUBE_MASS),
                    # Contacts start 5 mm out and rest at zero separation, so thin finger pads
                    # register contact before they penetrate the cube.
                    collision_props=PhysxCollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
                    physics_material=grippy,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.15, 0.15)),
                ),
                initial_pose=Pose(position_xyz=CUBE_POS),
            )
        )
        assets.append(
            Object(
                name="plate",
                object_type=ObjectType.RIGID,
                spawner_cfg=sim_utils.CylinderCfg(
                    radius=PLATE_RADIUS,
                    height=PLATE_HEIGHT,
                    rigid_props=PhysxRigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=PLATE_MASS),
                    collision_props=PhysxCollisionPropertiesCfg(),
                    physics_material=grippy,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.92, 0.90)),
                ),
                initial_pose=Pose(position_xyz=PLATE_POS),
            )
        )
    if d435_collider:
        # Static box with the D435's 90 x 25 x 25 mm body, lens on its front face.
        assets.append(_box("d435", D435_SIZE_XYZ, D435_BOX_POS, (0.08, 0.08, 0.09), D435_BOX_ROT_XYZW))
    if camera_mast:
        assets.append(_box("camera_mast", CAMERA_MAST_SIZE, CAMERA_MAST_POS, (0.18, 0.20, 0.23)))
    return assets


def env_cfg_callback(control_hz: float | None = None):
    """Arena env_cfg_callback: 240 Hz physics, ``control_hz`` control (10 Hz), one render per action."""
    decimation = DECIMATION if control_hz is None else max(1, round(1.0 / (control_hz * PHYSICS_DT)))

    def configure(env_cfg):
        env_cfg.sim.dt = PHYSICS_DT
        env_cfg.decimation = decimation
        env_cfg.sim.render_interval = decimation if control_hz is not None else RENDER_INTERVAL
        return env_cfg

    return configure


def build_environment(
    embodiment,
    objects: bool = True,
    d435_collider: bool = True,
    camera_mast: bool = False,
    table: bool = True,
    name: str = ENV_NAME,
    control_hz: float | None = None,
) -> IsaacLabArenaEnvironment:
    assets = scene_assets(objects=objects, d435_collider=d435_collider, camera_mast=camera_mast)
    if not table:
        assets = [a for a in assets if a.name != "table"]
    return IsaacLabArenaEnvironment(
        name=name,
        scene=Scene(assets),
        embodiment=embodiment,
        env_cfg_callback=env_cfg_callback(control_hz),
    )


def make_env(embodiment, *, device: str = "cuda:0", **scene_kwargs):
    """Build, register and make the Arena env (one env, no relation solving)."""
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

    arena_env = build_environment(embodiment, **scene_kwargs)
    builder = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device=device))
    return builder.make_registered()
