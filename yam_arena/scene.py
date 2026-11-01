"""The YAM scenes as Arena environments: the built-in tabletop and Dream's station. Import after app start.

``tabletop``: two arms on the near edge of a table, a red 4 cm cube in front of the left arm, a
white plate in front of the right, a dome light, a ground plane and, new in this port, a static
collider the size of the top camera's D435 body at the camera's pose (the 2.3 scene had none, so an
arm could pass through the camera). NoTask: no success criteria.

``station``: Dream's calibrated station (``yam_arena.station``): its table, apple, orange, banana,
water bottle, bowl and plate at Dream's poses and sizes, and the top camera's stand. Given a goal
(Dream's "put apple into the bowl", ...), its Arena task: ``ObjectInTask`` for the bowl,
``PickAndPlaceTask`` for the plate.

A task file (``yam_arena.task_file``): the station's or the tabletop's room (table, lights, camera body or
stand) with the file's objects on the table. Its success, distance and metrics are plain Python over
:func:`object_states`.

Inspect Robots owns the episode, so the env never ends or resets one by itself (no termination terms)
and keeps no Arena metrics: a task's success is read from Arena's progress tracker, see :func:`goal_status`.

Physics: 240 Hz PhysX, 24 substeps per action (10 Hz control), one render per action.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.managers.recorder_manager import DatasetExportMode, RecorderManagerBaseCfg
from isaaclab.sim.spawners.from_files import spawn_from_usd
from isaaclab.sim.spawners.shapes import spawn_cylinder
from isaaclab.sim.utils import clone
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_library import DomeLight, GroundPlane
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.registries import AssetRegistry
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.tasks.object_in_task import ObjectInTask
from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
from isaaclab_arena.utils.configclass import make_configclass
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.usd.helpers import compute_local_bounding_box_from_usd
from isaaclab_physx.sim.schemas import PhysxCollisionPropertiesCfg, PhysxRigidBodyPropertiesCfg

from yam_arena import station, task_file
from yam_arena.constants import (
    CAMERA_MAST_POS,
    CAMERA_MAST_SIZE,
    CONTROL_HZ,
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
from yam_arena.station import linear_rgb

ENV_NAME = "yam_bimanual_tabletop"
STATION_ENV_NAME = "yam_bimanual_station"
OBJECT_NAMES = ("cube", "plate")
FREE_OBJECTS = {"tabletop": OBJECT_NAMES, "station": station.FREE_OBJECTS}
# Station look, matched by eye to Dream's renders: a sky dome, and a lamp where Dream's shadow-casting lamp
# hangs, whose fall-off also darkens the distant floor as in Dream's views. RTX tone-maps where Dream's MuJoCo
# clips, so the table is Dream's oak brightened and saturated, and the floor a darker grey (linear colours).
STATION_DOME_INTENSITY = 250.0
STATION_LAMP_INTENSITY = 6.0e5
STATION_LAMP_RADIUS = 0.1
STATION_TABLE_COLOR = (0.792, 0.524, 0.273)
STATION_FLOOR_COLOR = (0.45, 0.45, 0.46)
# Success holds once the goal has held for this long; an "into the bowl" object must have at least this much of
# its bounding box inside the bowl's (Dream's bowl is shallower than the fruit is tall), and may still spin this
# fast (rad/s): fruit resting in the bowl keeps a residual ~0.1 rad/s from PhysX's mesh contacts, which Arena's
# 0.05 default would never call settled. Its speed limit stays Arena's 1 cm/s.
SETTLED_S = 1.0
MIN_CONTAINED_FRACTION = 0.5
MAX_SETTLED_SPIN = 0.5


def _box(name, size, pos, color, rot_xyzw=(0.0, 0.0, 0.0, 1.0), cls=Object) -> Object:
    """A static box collider (no rigid body): table, camera body, mast."""
    return cls(
        name=name,
        object_type=ObjectType.BASE,
        spawner_cfg=sim_utils.CuboidCfg(
            size=tuple(size),
            collision_props=PhysxCollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.7),
        ),
        initial_pose=Pose(position_xyz=tuple(pos), rotation_xyzw=tuple(rot_xyzw)),
    )


def _grippy() -> sim_utils.RigidBodyMaterialCfg:
    return sim_utils.RigidBodyMaterialCfg(static_friction=GRIPPY_FRICTION[0], dynamic_friction=GRIPPY_FRICTION[1])


def _contact_props() -> PhysxCollisionPropertiesCfg:
    # Contacts start 5 mm out and rest at zero separation, so thin finger pads register contact before they
    # penetrate the object.
    return PhysxCollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0)


def scene_assets(objects: bool = True, d435_collider: bool = True, camera_mast: bool = False) -> list:
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
                    collision_props=_contact_props(),
                    physics_material=_grippy(),
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
                    physics_material=_grippy(),
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


# -- station -----------------------------------------------------------------------------


class _Table(Object):
    """The station table, also PickAndPlaceTask's background: an object below ``object_min_z`` has dropped."""

    object_min_z = station.TABLE_TOP - 0.1


@clone
def _spawn_fixture(prim_path: str, cfg: sim_utils.UsdFileCfg, *args, **kwargs):
    """A kinematic library asset, its CCD flag cleared (PhysX ignores CCD on kinematic bodies, with an error)."""
    prim = spawn_from_usd.__wrapped__(prim_path, cfg, *args, **kwargs)
    ccd = prim.GetAttribute("physxRigidBody:enableCCD")
    if ccd:
        ccd.Set(False)
    return prim


@clone
def _spawn_convex(prim_path: str, cfg: sim_utils.UsdFileCfg, *args, **kwargs):
    """A library asset whose colliders are each one convex hull instead of a convex decomposition."""
    from pxr import Usd, UsdPhysics

    prim = spawn_from_usd.__wrapped__(prim_path, cfg, *args, **kwargs)
    for part in Usd.PrimRange(prim):
        if part.HasAPI(UsdPhysics.MeshCollisionAPI):
            UsdPhysics.MeshCollisionAPI(part).GetApproximationAttr().Set(UsdPhysics.Tokens.convexHull)
    return prim


def _library_item(name: str, item: station.Item) -> Object:
    """One of Dream's objects from Arena's object library, scaled and posed by ``station.placement``."""
    usd_path = AssetRegistry().get_asset_by_name(item.asset).usd_path
    bounds = compute_local_bounding_box_from_usd(usd_path)
    scale, pos, rot = station.placement(item, bounds.min_point[0].tolist(), bounds.max_point[0].tolist())
    addon = {"physics_material": _grippy()}
    if item.mass is None:
        addon.update(func=_spawn_fixture, rigid_props=PhysxRigidBodyPropertiesCfg(kinematic_enabled=True))
    else:
        mass = {"": [sim_utils.MassCfg(mass=item.mass)]}
        addon.update(mass_props=mass, mass_props_create_if_missing=True, collision_props=_contact_props())
    if item.convex:
        addon["func"] = _spawn_convex
    if item.color is not None:
        addon["visual_material"] = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(item.color), roughness=0.3)
    return Object(
        name=name,
        usd_path=usd_path,
        object_type=ObjectType.RIGID,
        scale=(scale,) * 3,
        initial_pose=Pose(position_xyz=pos, rotation_xyzw=rot),
        spawn_cfg_addon=addon,
    )


@clone
def _spawn_dream_bottle(prim_path: str, cfg: sim_utils.CylinderCfg, *args, **kwargs):
    """Dream's water bottle: the cylinder body, then its visual trim (``station.BOTTLE_TRIM``) riding on it."""
    from pxr import Gf, UsdGeom

    prim = spawn_cylinder.__wrapped__(prim_path, cfg, *args, **kwargs)
    stage = prim.GetStage()
    for name, (shape, (radius, half), z, color) in station.BOTTLE_TRIM.items():
        path = f"{prim_path}/trim/{name}"
        if shape == "cylinder":
            geom = UsdGeom.Cylinder.Define(stage, path)
            geom.CreateRadiusAttr(radius)
            geom.CreateHeightAttr(2 * half)
            geom.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, z))
        else:
            geom = UsdGeom.Sphere.Define(stage, path)
            geom.CreateRadiusAttr(1.0)
            geom.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, z))
            geom.AddScaleOp().Set(Gf.Vec3f(radius, radius, half))
        look = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(color), roughness=0.3)
        look.func(f"{path}_material", look)
        sim_utils.bind_visual_material(path, f"{path}_material", stage=stage)
    return prim


def _stand_part(name: str, kind: str, size, pos, rot_xyzw) -> Object:
    color = linear_rgb(station.HOUSING if name == "camera_housing" else station.STAND)
    if kind == "box":
        return _box(name, size, pos, color, rot_xyzw)
    return Object(
        name=name,
        object_type=ObjectType.BASE,
        spawner_cfg=sim_utils.CapsuleCfg(
            radius=size[0],
            height=size[1],
            collision_props=PhysxCollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.5),
        ),
        initial_pose=Pose(position_xyz=pos, rotation_xyzw=rot_xyzw),
    )


def _station_room() -> list[Object]:
    """The station's floor, lights and table."""
    return [
        _box("floor", (200.0, 200.0, 0.02), (0.0, 0.0, station.FLOOR_Z - 0.01), STATION_FLOOR_COLOR),
        DomeLight(spawner_cfg=sim_utils.DomeLightCfg(intensity=STATION_DOME_INTENSITY, color=linear_rgb(station.SKY))),
        Object(
            name="lamp",
            object_type=ObjectType.BASE,
            spawner_cfg=sim_utils.SphereLightCfg(intensity=STATION_LAMP_INTENSITY, radius=STATION_LAMP_RADIUS),
            initial_pose=Pose(position_xyz=station.LAMP_POS),
        ),
        _box("table", station.TABLE_SIZE, station.TABLE_CENTER, STATION_TABLE_COLOR, cls=_Table),
    ]


def station_assets(top_cam) -> dict[str, Object]:
    """Dream's station around the arms; the camera stand follows ``top_cam`` (a ``rig.Camera``)."""
    bottle_z = station.TABLE_TOP + station.REST_GAP + station.BOTTLE_HEIGHT / 2
    assets = [
        *_station_room(),
        *(_library_item(name, item) for name, item in station.ITEMS.items()),
        Object(
            name="bottle",
            object_type=ObjectType.RIGID,
            spawner_cfg=sim_utils.CylinderCfg(
                func=_spawn_dream_bottle,
                radius=station.BOTTLE_RADIUS,
                height=station.BOTTLE_HEIGHT,
                rigid_props=PhysxRigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=station.BOTTLE_MASS),
                collision_props=_contact_props(),
                physics_material=_grippy(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=linear_rgb(station.BOTTLE_COLOR), roughness=0.3
                ),
            ),
            initial_pose=Pose(position_xyz=(*station.BOTTLE_XY, bottle_z)),
        ),
        *(_stand_part(name, *part) for name, part in station.camera_stand(top_cam).items()),
    ]
    return {asset.name: asset for asset in assets}


def station_task(goal: station.Goal, assets: dict[str, Object], control_hz: float):
    """Dream's "put <fruit> into the bowl / on the plate" as an Arena task with success criteria."""
    steps = max(1, round(SETTLED_S * control_hz))
    obj, target = assets[goal.object], assets[goal.target]
    if goal.relation == "in":
        return ObjectInTask(
            obj,
            target,
            angular_velocity_threshold=MAX_SETTLED_SPIN,
            consecutive_success_steps=steps,
            minimum_contained_fraction=MIN_CONTAINED_FRACTION,
            task_description=goal.instruction,
        )
    return PickAndPlaceTask(
        obj, target, assets["table"], placement_consecutive_steps=steps, task_description=goal.instruction
    )


# -- task files --------------------------------------------------------------------------


def _yaw_xyzw(yaw: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


def task_object(spec: task_file.ObjectSpec) -> Object:
    """A task file's object as an Arena object: free (a rigid body with its mass) or fixed (a static collider)."""
    if isinstance(spec, task_file.Asset):
        item = station.Item(spec.asset, spec.xy, spec.extent, spec.mass, spec.color, spec.convex, spec.yaw)
        return _library_item(spec.name, item)
    look = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(spec.color), roughness=0.4)
    if spec.free:
        body = dict(
            rigid_props=PhysxRigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=spec.mass),
            collision_props=_contact_props(),
        )
    else:
        body = dict(collision_props=PhysxCollisionPropertiesCfg())
    body.update(physics_material=_grippy(), visual_material=look)
    if isinstance(spec, task_file.Box):
        cfg = sim_utils.CuboidCfg(size=tuple(spec.size), **body)
    elif isinstance(spec, task_file.Cylinder):
        cfg = sim_utils.CylinderCfg(radius=spec.radius, height=spec.height, **body)
    else:
        cfg = sim_utils.SphereCfg(radius=spec.radius, **body)
    return Object(
        name=spec.name,
        object_type=ObjectType.RIGID if spec.free else ObjectType.BASE,
        spawner_cfg=cfg,
        initial_pose=Pose(position_xyz=tuple(spec.pos), rotation_xyzw=_yaw_xyzw(spec.yaw)),
    )


def task_room(scene: str, top_cam) -> list[Object]:
    """What stands around a task file's objects: the station's room and camera stand, or the tabletop's."""
    if scene == "station":
        return [*_station_room(), *(_stand_part(name, *part) for name, part in station.camera_stand(top_cam).items())]
    return scene_assets(objects=False)


def build_task_environment(embodiment, task: task_file.TaskFile, control_hz: float | None = None):
    """A task file's objects in its scene's room, for an embodiment built with that scene's rig."""
    return IsaacLabArenaEnvironment(
        name=f"yam_bimanual_task_{task.name}",
        scene=Scene([*task_room(task.scene, embodiment.rig.top_cam), *(task_object(o) for o in task.objects)]),
        embodiment=embodiment,
        env_cfg_callback=env_cfg_callback(control_hz),
    )


def object_states(env, names) -> dict[str, task_file.ObjectState]:
    """Each named object's bounding-box centre, orientation, velocities and bounds (world frame, env 0)."""
    import numpy as np

    base = env.unwrapped
    world = base.arena_world
    origin = base.scene.env_origins[0]
    rigid = base.scene.rigid_objects

    def arr(t) -> np.ndarray:
        return t.detach().cpu().numpy().astype(np.float64)

    out = {}
    for name in names:
        box = world.get_aabb_w(name)
        moving = name in rigid
        out[name] = task_file.ObjectState(
            pos=arr(world.get_centroid_w(name)[0] - origin),
            quat_xyzw=arr(world.get_pose_w(name)[0, 3:]),
            lin_vel=arr(world.get_root_linear_velocity_w(name)[0]) if moving else np.zeros(3),
            ang_vel=arr(world.get_root_angular_velocity_w(name)[0]) if moving else np.zeros(3),
            low=arr(box.min_point[0] - origin),
            high=arr(box.max_point[0] - origin),
        )
    return out


# -- environments ------------------------------------------------------------------------


def env_cfg_callback(control_hz: float | None = None):
    """Arena env_cfg_callback: 240 Hz physics, ``control_hz`` control (10 Hz), one render per action."""
    decimation = DECIMATION if control_hz is None else max(1, round(1.0 / (control_hz * PHYSICS_DT)))

    def configure(env_cfg):
        env_cfg.sim.dt = PHYSICS_DT
        env_cfg.decimation = decimation
        env_cfg.sim.render_interval = decimation if control_hz is not None else RENDER_INTERVAL
        # Arena's metric recorders read the success termination and export every episode to disk.
        env_cfg.terminations = make_configclass("TerminationsCfg", [])()
        env_cfg.recorders = RecorderManagerBaseCfg(dataset_export_mode=DatasetExportMode.EXPORT_NONE)
        env_cfg.metrics = None
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


def build_station_environment(
    embodiment, goal: station.Goal | None = None, control_hz: float | None = None
) -> IsaacLabArenaEnvironment:
    """Dream's station for an embodiment built with ``station.rig()``; ``goal`` adds its Arena task."""
    assets = station_assets(embodiment.rig.top_cam)
    return IsaacLabArenaEnvironment(
        name=STATION_ENV_NAME,
        scene=Scene(list(assets.values())),
        embodiment=embodiment,
        task=None if goal is None else station_task(goal, assets, control_hz or CONTROL_HZ),
        env_cfg_callback=env_cfg_callback(control_hz),
    )


def make_env(embodiment, *, scene: str = "tabletop", device: str = "cuda:0", **scene_kwargs):
    """Build, register and make the Arena env for ``scene`` (one env, no relation solving)."""
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

    build = {"tabletop": build_environment, "station": build_station_environment, "task": build_task_environment}[scene]
    cfg = ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device=device)
    return ArenaEnvBuilder(build(embodiment, **scene_kwargs), cfg).make_registered()


def goal_status(env, goal: station.Goal) -> tuple[bool, float]:
    """(success, metres between the object's and the target's bounding-box centres) after a control step.

    Success is the task's: its Arena progress tracker, advanced here by this control step.
    """
    base = env.unwrapped
    base.update_task_progress()
    world = base.arena_world
    gap = world.get_centroid_w(goal.object) - world.get_centroid_w(goal.target)
    return bool(base.progress_tracker.is_complete()[0]), float(gap[0].norm())
