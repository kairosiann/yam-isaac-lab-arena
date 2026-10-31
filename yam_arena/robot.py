"""Isaac Lab 3.0 articulation config for one I2RT YAM arm (6 DoF + linear gripper).

Import only after the app has started. Drive gains and joint friction come from i2rt's own
controller config (yam_v1.yml, linear_4310.yml), as in the 2.3 port, so a joint target moves
the simulated arm the way it moves the real one.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
import numpy as np
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.spawners.from_files import spawn_from_usd
from isaaclab.sim.utils import clone
from isaaclab_physx.sim.schemas import (
    PhysxArticulationRootPropertiesCfg,
    PhysxCollisionPropertiesCfg,
    PhysxRigidBodyPropertiesCfg,
)

from yam_arena import require_usd, station
from yam_arena.analysis import read_stl
from yam_arena.constants import FINGER_JOINTS, FINGER_OPEN, GRIPPER, READY, SHOULDER, WRIST


def yam_arm_cfg(
    prim_path: str,
    base_pos: tuple[float, float, float],
    base_rot_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    contact_offset: float | None = None,
    dream_look: bool = False,
) -> ArticulationCfg:
    """One YAM arm spawned from the pre-converted USD at ``prim_path``, base at ``base_pos``.

    ``contact_offset`` (m), if set, is authored on every arm collider with a zero rest offset. The
    2.3 port asked for 5 mm but Isaac Lab 2.3 could not author it on the instanced colliders and
    warned instead, so its measured behaviour used PhysX defaults; the default here (None) matches
    that behaviour. Setting it un-instances the arm's colliders (more stage memory), as does
    ``dream_look`` (colour the arm as Dream's station renders it, see :func:`spawn_in_dream_look`).
    """
    spawn_kwargs = {}
    if contact_offset is not None:
        spawn_kwargs["collision_props"] = PhysxCollisionPropertiesCfg(contact_offset=contact_offset, rest_offset=0.0)
    if dream_look:
        spawn_kwargs["func"] = spawn_in_dream_look
    if spawn_kwargs:
        spawn_kwargs["make_uninstanceable"] = True
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(require_usd()),
            activate_contact_sensors=True,
            rigid_props=PhysxRigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=1.0),
            articulation_props=PhysxArticulationRootPropertiesCfg(
                # The finger carriages slide inside the gripper housing and past each other, so their
                # convex hulls overlap by design; self-collision would push the fingers apart.
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
            **spawn_kwargs,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(base_pos),
            rot=tuple(base_rot_xyzw),
            joint_pos={**{f"joint{i + 1}": q for i, q in enumerate(READY)}, "joint[7-8]": FINGER_OPEN},
            joint_vel={".*": 0.0},
        ),
        actuators={
            # Joints 1-3: Damiao DM4340 (27 Nm peak, ~5.5 rad/s no-load at 24 V).
            "shoulder": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-3]"],
                joint_effort_limit=SHOULDER["effort"],
                joint_velocity_limit=SHOULDER["velocity"],
                stiffness=SHOULDER["stiffness"],
                damping=SHOULDER["damping"],
                friction=SHOULDER["friction"],
            ),
            # Joints 4-6: DM4310 (7 Nm peak, ~20 rad/s no-load at 24 V).
            "wrist": ImplicitActuatorCfg(
                joint_names_expr=["joint[4-6]"],
                joint_effort_limit=WRIST["effort"],
                joint_velocity_limit=WRIST["velocity"],
                stiffness=WRIST["stiffness"],
                damping=WRIST["damping"],
                friction=WRIST["friction"],
            ),
            # One DM4310 drives both fingers through a rack; i2rt's driver caps grip force at 50 N.
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=list(FINGER_JOINTS),
                joint_effort_limit=GRIPPER["effort"],
                joint_velocity_limit=GRIPPER["velocity"],
                stiffness=GRIPPER["stiffness"],
                damping=GRIPPER["damping"],
            ),
        },
    )


def asset_link_paths() -> dict[str, str]:
    """Each rigid-body link's prim path relative to the asset's default prim, read from the USD.

    The 3.0 importer's layered output decides how deep the links sit, so camera mounts and
    contact sensors look the paths up instead of assuming ``<robot>/<link>``.
    """
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(require_usd()))
    root = str(stage.GetDefaultPrim().GetPath())
    paths = {}
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            paths[prim.GetName()] = str(prim.GetPath())[len(root) :]
    return paths


@clone
def spawn_in_dream_look(prim_path: str, cfg: sim_utils.UsdFileCfg, *args, **kwargs):
    """Spawn an (un-instanced) arm and colour it as Dream's station renders it: black, with white covers on
    link2 and link3, whose visuals are replaced by Dream's black and white parts of the same meshes."""
    from pxr import UsdGeom, Vt

    prim = spawn_from_usd.__wrapped__(prim_path, cfg, *args, **kwargs)
    stage = prim.GetStage()
    materials = {}
    for color, srgb in (("black", station.ARM_BLACK), ("white", station.ARM_WHITE)):
        materials[color] = f"{prim_path}/DreamLook/{color}"
        look = sim_utils.PreviewSurfaceCfg(diffuse_color=station.linear_rgb(srgb), roughness=0.5)
        look.func(materials[color], look)
    for link, path in asset_link_paths().items():
        link_path = prim_path + path
        if link not in station.ARM_PARTS:
            sim_utils.bind_visual_material(f"{link_path}/{link}", materials["black"], stage=stage)
            continue
        UsdGeom.Imageable(stage.GetPrimAtPath(f"{link_path}/{link}")).MakeInvisible()
        for color, stl in station.ARM_PARTS[link].items():
            corners = read_stl(stl).reshape(-1, 3) @ station.ARM_PART_ROT.T + station.ARM_PART_POS[link]
            points, corner_ids = np.unique(corners.round(7), axis=0, return_inverse=True)
            mesh = UsdGeom.Mesh.Define(stage, f"{link_path}/dream_{color}")
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * (len(corners) // 3)))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(corner_ids.reshape(-1).astype(np.int32)))
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            mesh.CreateDoubleSidedAttr(True)
            sim_utils.bind_visual_material(str(mesh.GetPath()), materials[color], stage=stage)
    return prim
