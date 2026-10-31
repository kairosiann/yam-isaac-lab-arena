"""Stage 4: full tabletop scene, top-camera collider, and a cube grasp.

    python scripts/stage4_scene.py [--out DIR] [--camera-mast]

Part A, scene: table, cube, plate, dome light, ground, and the new static D435 collider
(90 x 25 x 25 mm) at the top camera. Objects must settle on the table.

Part B, camera collider: the left arm reaches for the centre of the D435 box (joint-space move
to an IK solution on the URDF). Reported: the commanded penetration (URDF FK + collision hulls
vs the box), the achieved minimum signed distance of the arm's hulls to the box, the TCP
distance to the box centre, and the contact force. The box must not move.

Part C, grasp (port of the 2.3 diagnose_physics.py demo): the left arm picks the cube, holds it
lifted for 1.5 s, carries and places it; the right arm presses closed fingers 10 mm into the
plate. Same metrics as diagnose_physics.py (joint-limit overshoot, peak joint speed, contact
force per link, cube lift, fingertip penetration into the plate, cube final position), plus
jitter while lifted (cube motion relative to the gripper), fall-through, and blow-up checks.
State is sampled at the 10 Hz control rate (2.3 sampled every 240 Hz physics step).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from yam_arena.checks import Checks

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--out", type=Path, default=None)
parser.add_argument("--camera-mast", action="store_true", help="Also add the (guessed) camera mast collider.")
args = parser.parse_args()
checks = Checks("stage4_scene", args.out)

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import numpy as np
import torch

from yam_arena import ik
from yam_arena.analysis import fk_link_poses_world, hulls_to_box, pose7_to_matrix, world_hulls
from yam_arena.constants import (
    CUBE_POS,
    CUBE_SIZE,
    D435_BOX_POS,
    D435_BOX_ROT_XYZW,
    D435_SIZE_XYZ,
    LEFT_BASE_POS,
    PLATE_HEIGHT,
    PLATE_POS,
    READY,
    TABLE_HEIGHT,
)
from yam_arena.demo import make_plans
from yam_arena.ik import smoothstep
from yam_arena.runtime import arm_q, contact_forces, link_poses, pad_tips_world, tcp_world
from yam_arena.urdf import quat_xyzw_to_matrix

BOX_ROT = quat_xyzw_to_matrix(np.array(D435_BOX_ROT_XYZW))
# Seed for the reach into the camera body: joint1 turned back toward the camera.
D435_IK_SEED = [-2.0, 1.4, 1.0, -0.3, 0.0, 0.0]


def act(env, left7, right7, n=1):
    a = torch.tensor([[*left7, *right7]], dtype=torch.float32, device=env.unwrapped.device)
    for _ in range(n):
        env.step(a)


def obj_pos(obj, origin):
    return obj.data.root_link_pose_w.torch[0, :3].cpu().numpy() - origin


def obj_speed(obj):
    return float(torch.linalg.vector_norm(obj.data.root_com_vel_w.torch[0, :3]))


def part_a(env, origin):
    base = env.unwrapped
    from pxr import UsdPhysics

    stage = base.sim.stage if hasattr(base.sim, "stage") else None
    if stage is None:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
    checks.check(
        "rigid objects == {cube, plate}",
        set(base.scene.rigid_objects) == {"cube", "plate"},
        str(sorted(base.scene.rigid_objects)),
    )
    prim = stage.GetPrimAtPath("/World/envs/env_0/d435")
    from pxr import Usd, UsdGeom

    if prim.IsValid():
        sub = list(Usd.PrimRange(prim))
        has_col = any(p.HasAPI(UsdPhysics.CollisionAPI) for p in sub)
        has_rb = any(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in sub)
        xf = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)).T
        pos_err = np.linalg.norm(xf[:3, 3] - origin - np.array(D435_BOX_POS))
        # Edge lengths as PhysX sees them: UsdGeom.Cube size x the cube's scale relative to the d435 Xform
        # (Isaac Lab spawns a cuboid as a Cube of size min(dims), scaled per axis).
        cubes = [p for p in sub if p.IsA(UsdGeom.Cube)]
        ext, extent_attr = None, None
        if cubes:
            c = cubes[0]
            rel = np.array(UsdGeom.Xformable(c).ComputeLocalToWorldTransform(0)).T @ np.linalg.inv(xf)
            ext = UsdGeom.Cube(c).GetSizeAttr().Get() * np.linalg.norm(rel[:3, :3], axis=0)
            extent_attr = UsdGeom.Boundable(c).GetExtentAttr().Get()
        ok_size = ext is not None and np.allclose(ext, D435_SIZE_XYZ, atol=1e-5)
        checks.check(
            "D435 collider: static collision prim at the top camera, 25x90x25 mm (box axes)",
            has_col and not has_rb and pos_err < 1e-4 and ok_size,
            f"collision={has_col} rigid_body={has_rb} pos err {pos_err * 1000:.3f} mm;"
            f" edges {None if ext is None else np.round(ext * 1000, 2).tolist()} mm (cube extent attr {extent_attr})",
        )
    else:
        checks.check("D435 collider prim exists", False, "no prim at /World/envs/env_0/d435")
    act(env, [*READY, 1.0], [*READY, 1.0], n=10)
    cube, plate = base.scene["cube"], base.scene["plate"]
    cz, pz = obj_pos(cube, origin)[2], obj_pos(plate, origin)[2]
    checks.check(
        "cube and plate settle on the table (1 s)",
        abs(cz - CUBE_POS[2]) < 0.002 and abs(pz - PLATE_POS[2]) < 0.002 and obj_speed(cube) < 0.005,
        f"cube z {cz:.4f} (expect {CUBE_POS[2]:.4f}), plate z {pz:.4f} (expect {PLATE_POS[2]:.4f}),"
        f" cube speed {obj_speed(cube) * 1000:.2f} mm/s",
    )


def part_b(env, origin):
    base = env.unwrapped
    left = base.scene["left_robot"]
    lc = base.scene["left_contacts"]
    box_c = np.array(D435_BOX_POS) + origin
    q_goal, pe, _ = ik.solve(D435_BOX_POS, None, D435_IK_SEED, base_pos=LEFT_BASE_POS, iters=300)
    # Closed fingers: open ones (94 mm apart) would straddle the 90 mm box at this target.
    q_d = {f"joint{i + 1}": float(v) for i, v in enumerate(q_goal)} | {"joint7": 0.0, "joint8": 0.0}
    cmd_hulls = world_hulls(fk_link_poses_world(q_d, LEFT_BASE_POS))
    cmd_dist, cmd_link = hulls_to_box(cmd_hulls, D435_BOX_POS, BOX_ROT, D435_SIZE_XYZ)
    checks.note(
        "D435 reach target",
        {
            "q_goal": np.round(q_goal, 3).tolist(),
            "ik_pos_err_mm": round(pe * 1000, 3),
            "commanded_min_signed_dist_to_box_mm": round(cmd_dist * 1000, 1),
            "link": cmd_link,
        },
    )
    act(env, [*READY, 0.0], [*READY, 1.0], n=10)  # close the left gripper first
    q0 = arm_q(left)
    n_move, n_hold = 40, 20
    min_d, min_link, peak_f, peak_link, first_contact = np.inf, None, 0.0, None, None
    for k in range(n_move + n_hold):
        s = smoothstep((k + 1) / n_move)
        q = q0 + s * (q_goal - q0)
        act(env, [*q, 0.0], [*READY, 1.0])
        poses = {n: p.copy() for n, p in link_poses(left).items()}
        for p in poses.values():
            p[:3, 3] -= origin
        d, link = hulls_to_box(world_hulls(poses), D435_BOX_POS, BOX_ROT, D435_SIZE_XYZ)
        if d < min_d:
            min_d, min_link = d, link
        f = contact_forces(lc)
        name, val = max(f.items(), key=lambda kv: kv[1])
        if val > peak_f:
            peak_f, peak_link = val, name
        if first_contact is None and val > 0.05:
            first_contact = (k + 1) * base.step_dt
    tcp_d = float(np.linalg.norm(tcp_world(left) - box_c))
    q_err = np.abs(arm_q(left) - q_goal)
    checks.note(
        "D435 reach achieved",
        {
            "min_signed_dist_to_box_mm (negative = inside)": round(min_d * 1000, 2),
            "closest_link": min_link,
            "tcp_to_box_centre_mm": round(tcp_d * 1000, 1),
            "peak_contact_N": round(peak_f, 1),
            "peak_contact_link": peak_link,
            "first_contact_s": first_contact,
            "final_joint_error_rad": np.round(q_err, 3).tolist(),
        },
    )
    checks.check(
        "reach would pass through the camera body without the collider",
        cmd_dist < -0.01,
        f"commanded min signed distance {cmd_dist * 1000:.1f} mm (negative = inside; the box is 25 mm thick)",
    )
    checks.check(
        "D435 collider blocks the arm (max penetration <= 3 mm)",
        min_d > -0.003,
        f"min signed distance {min_d * 1000:.2f} mm ({min_link}); TCP stopped {tcp_d * 1000:.1f} mm from box centre",
    )
    checks.check("arm-camera contact force > 1 N", peak_f > 1.0, f"peak {peak_f:.1f} N on {peak_link}")
    import omni.usd
    from pxr import UsdGeom

    prim = omni.usd.get_context().get_stage().GetPrimAtPath("/World/envs/env_0/d435")
    xf = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)).T
    moved = np.linalg.norm(xf[:3, 3] - box_c)
    checks.check("camera body did not move", moved < 1e-6, f"moved {moved * 1000:.4f} mm")
    # back out the way we came
    q1 = arm_q(left)
    for k in range(40):
        s = smoothstep((k + 1) / 40)
        act(env, [*(q1 + s * (np.array(READY) - q1)), 0.0], [*READY, 1.0])
    act(env, [*READY, 1.0], [*READY, 1.0], n=10)
    err = np.abs(arm_q(left) - READY).max()
    checks.check("left arm returns to READY", err < 0.02, f"max error {err:.4f} rad")


def part_c(env, origin):
    base = env.unwrapped
    env.reset()
    left, right = base.scene["left_robot"], base.scene["right_robot"]
    lc, rc = base.scene["left_contacts"], base.scene["right_contacts"]
    cube, plate = base.scene["cube"], base.scene["plate"]
    act(env, [*READY, 1.0], [*READY, 1.0], n=5)  # settle 0.5 s
    cube0, plate0 = obj_pos(cube, origin), obj_pos(plate, origin)
    names = list(left.joint_names)
    hard = left.data.joint_pos_limits.torch[0].cpu().numpy()
    vel_lim = left.data.joint_vel_limits.torch[0].cpu().numpy()
    over = np.zeros(len(names))
    peak_vel = np.zeros(len(names))
    touched: dict[str, float] = {}
    lowest_tip = 9.0
    cube_peak, cube_min_z, cube_vmax, nan = -9.0, 9.0, 0.0, False
    hold_rel = []
    lplan, rplan = make_plans(hold_s=1.5)
    steps = round((max(lplan.duration, rplan.duration) + 1.0) / base.step_dt)
    for k in range(steps):
        t = (k + 1) * base.step_dt
        la, llabel = lplan.action7(t)
        ra, _ = rplan.action7(t)
        act(env, la, ra)
        for arm in (left, right):
            q = arm.data.joint_pos.torch[0].cpu().numpy()
            nan |= not np.isfinite(q).all()
            over = np.maximum(over, np.maximum(hard[:, 0] - q, q - hard[:, 1]))
            peak_vel = np.maximum(peak_vel, np.abs(arm.data.joint_vel.torch[0].cpu().numpy()))
        lowest_tip = min(lowest_tip, float(pad_tips_world(right)[2] - origin[2]))
        cp = obj_pos(cube, origin)
        nan |= not np.isfinite(cp).all()
        cube_peak, cube_min_z = max(cube_peak, cp[2]), min(cube_min_z, cp[2])
        cube_vmax = max(cube_vmax, obj_speed(cube))
        for side, sensor in (("left", lc), ("right", rc)):
            for name, f in contact_forces(sensor).items():
                if f > 0.05:
                    key = f"{side}/{name}"
                    touched[key] = max(touched.get(key, 0.0), f)
        if llabel == "hold lifted":
            g = link_poses(left)["gripper"]
            c = pose7_to_matrix(cube.data.root_link_pose_w.torch[0].cpu().numpy())
            hold_rel.append((np.linalg.inv(g) @ c)[:3, 3])
    cube_final = obj_pos(cube, origin)
    plate_final = obj_pos(plate, origin)

    print("[stage4_scene] joint      worst limit overshoot   peak |vel| / sim limit", flush=True)
    for i, n in enumerate(names):
        print(
            f"[stage4_scene] {n:8s}   {max(0.0, float(over[i])):8.4f}            {peak_vel[i]:6.3f} / {vel_lim[i]:.3f}",
            flush=True,
        )
    checks.note("peak contact force per link (N)", {k: round(v, 1) for k, v in sorted(touched.items())})
    checks.note(
        "ik_worst (left mm/deg, right mm/deg)",
        [
            round(lplan.worst_ik_mm, 3),
            round(lplan.worst_ik_deg, 3),
            round(rplan.worst_ik_mm, 3),
            round(rplan.worst_ik_deg, 3),
        ],
    )
    lift = cube_peak - cube0[2]
    checks.check("cube lifted > 10 cm (fingers grip the cube)", lift > 0.1, f"lifted {lift * 100:.1f} cm")
    plate_top = TABLE_HEIGHT + PLATE_HEIGHT
    sink = plate_top - lowest_tip
    checks.check(
        "right fingertips stop on the plate (penetration < 4 mm)",
        sink < 0.004,
        f"lowest tip z {lowest_tip:.4f}, plate top {plate_top:.4f}, commanded 10 mm below;"
        f" penetration {max(0.0, sink) * 1000:.1f} mm",
    )
    rel = np.array(hold_rel)
    if len(rel) >= 3:
        jitter = rel.std(axis=0).max()
        slip = np.linalg.norm(rel[-1] - rel[0])
        checks.check(
            "cube steady in the gripper while lifted (std < 1 mm, slip < 3 mm over 1.5 s)",
            jitter < 0.001 and slip < 0.003,
            f"position std {jitter * 1000:.3f} mm, slip {slip * 1000:.3f} mm over {len(rel)} samples",
        )
    else:
        checks.check("cube hold window sampled", False, f"{len(rel)} samples")
    on_table = abs(cube_final[2] - CUBE_POS[2]) < 0.003
    checks.check(
        "cube never falls through the table",
        cube_min_z > TABLE_HEIGHT + CUBE_SIZE / 2 - 0.003,
        f"min cube centre z {cube_min_z:.4f} (resting {CUBE_POS[2]:.4f})",
    )
    checks.check(
        "cube placed and resting on the table",
        on_table and obj_speed(cube) < 0.01,
        f"final {np.round(cube_final, 4).tolist()}, speed {obj_speed(cube) * 1000:.2f} mm/s",
    )
    checks.check(
        "no blow-up: no NaN, cube speed < 1.5 m/s, joint speeds within sim limits",
        not nan and cube_vmax < 1.5 and bool((peak_vel <= vel_lim * 1.05).all()),
        f"nan={nan} cube peak speed {cube_vmax:.3f} m/s; peak joint speed {peak_vel.max():.3f} rad/s",
    )
    checks.check(
        "joint limits respected (overshoot < 0.01)",
        float(over.max()) < 0.01,
        f"worst overshoot {max(0.0, float(over.max())):.5f}",
    )
    checks.note(
        "cube final / plate displacement",
        {
            "cube_final": np.round(cube_final, 4).tolist(),
            "plate_moved_mm": round(float(np.linalg.norm(plate_final - plate0)) * 1000, 1),
        },
    )


def main():
    from yam_arena.embodiment import BimanualYamEmbodiment
    from yam_arena.scene import make_env

    emb = BimanualYamEmbodiment(arms=("left", "right"), contact_sensors=True)
    env = make_env(emb, objects=True, d435_collider=True, camera_mast=args.camera_mast)
    env.reset()
    origin = env.unwrapped.scene.env_origins[0].cpu().numpy()
    checks.note("env_origin", origin.tolist())
    for part in (part_a, part_b, part_c):
        try:
            part(env, origin)
        except Exception as exc:
            checks.error(part.__name__, exc)


try:
    main()
except Exception as exc:
    checks.error("stage 4", exc)
checks.finish()
