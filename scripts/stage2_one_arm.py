"""Stage 2: one YAM arm loads in Arena (NoTask), steps, and holds the READY pose.

    python scripts/stage2_one_arm.py [--out DIR]

Builds the yam_bimanual embodiment with only the left arm (scene key `left_robot`) on the
table, no objects, through ArenaEnvBuilder(solve_relations=False). Checks:
  - the articulation is fixed-base, has joint1..joint8 and the URDF's joint limits
  - the action term layout is 7-D [j1..j6, grip] and the env runs at 240 Hz / 10 Hz
  - holding READY: max arm joint error and joint speed after settling, base drift
  - link poses match the URDF's forward kinematics at the measured joints (READY and a
    second, asymmetric pose), which catches axis/sign flips in the URDF import
  - gravity compensation matters: the same hold without it (informational)
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
parser.add_argument("--hold-s", type=float, default=3.0)
args = parser.parse_args()
checks = Checks("stage2_one_arm", args.out)

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import numpy as np
import torch

from yam_arena import YAM_URDF
from yam_arena.analysis import fk_errors
from yam_arena.constants import (
    ALL_JOINTS,
    ARM_JOINTS,
    DECIMATION,
    FINGER_OPEN,
    LEFT_BASE_POS,
    LINKS,
    PHYSICS_DT,
    READY,
)
from yam_arena.urdf import parse_joints

TEST_Q = [0.6, 1.4, 0.9, -0.4, 0.5, 0.8]  # asymmetric, every joint non-zero, clear of the table
HOLD_TOL_RAD = 0.01  # holding READY from an exact start: gravity compensation leaves no error
# After a move the joints settle inside the joint-friction deadband (i2rt friction; kp 10 on the
# wrist), so tracking a new target is checked to 0.02 rad (1.1 deg).
TRACK_TOL_RAD = 0.02
FK_TOL_MM, FK_TOL_DEG = 1.0, 0.5


def hold(env, robot, q_arm, grip, seconds):
    """Step with a constant action; return (arm joint trace [n,6], finger trace [n,2], qd trace [n,8])."""
    action = torch.tensor([[*q_arm, grip]], dtype=torch.float32, device=env.unwrapped.device)
    arm_ids = robot.find_joints(ARM_JOINTS, preserve_order=True)[0]
    qs, fs, vs = [], [], []
    for _ in range(round(seconds / env.unwrapped.step_dt)):
        env.step(action)
        q = robot.data.joint_pos.torch[0]
        qs.append(q[arm_ids].cpu().numpy())
        fs.append(q[6:8].cpu().numpy())
        vs.append(robot.data.joint_vel.torch[0].cpu().numpy())
    return np.array(qs), np.array(fs), np.array(vs)


def fk_check(label, robot):
    q = robot.data.joint_pos.torch[0].cpu().numpy()
    qd = dict(zip(robot.joint_names, q.tolist(), strict=True))
    errs = fk_errors(
        robot.body_names,
        robot.data.body_link_pose_w.torch[0].cpu().numpy(),
        robot.data.root_link_pose_w.torch[0].cpu().numpy(),
        qd,
    )
    worst_mm = max(e[0] for e in errs.values())
    worst_deg = max(e[1] for e in errs.values())
    checks.check(
        f"link poses match URDF FK at {label}",
        worst_mm < FK_TOL_MM and worst_deg < FK_TOL_DEG and len(errs) == len(LINKS),
        f"worst {worst_mm:.3f} mm / {worst_deg:.3f} deg over {len(errs)} links",
        per_link={k: [round(v[0], 3), round(v[1], 3)] for k, v in errs.items()},
    )


def main():
    from yam_arena.embodiment import BimanualYamEmbodiment
    from yam_arena.scene import make_env

    emb = BimanualYamEmbodiment(arms=("left",), contact_sensors=False)
    checks.note("link_paths", emb.link_paths)
    env = make_env(emb, objects=False, d435_collider=False)
    base = env.unwrapped
    robot = base.scene["left_robot"]

    checks.check(
        "articulations == {left_robot}",
        set(base.scene.articulations) == {"left_robot"},
        str(sorted(base.scene.articulations)),
    )
    checks.check("is_fixed_base", bool(robot.is_fixed_base), f"is_fixed_base={robot.is_fixed_base}")
    checks.check("joint names joint1..joint8", list(robot.joint_names) == ALL_JOINTS, str(robot.joint_names))
    checks.note("body_names", list(robot.body_names))
    checks.check("9 bodies", sorted(robot.body_names) == sorted(LINKS), str(robot.body_names))
    dim = base.action_manager.total_action_dim
    checks.check("action dim == 7", dim == 7, f"dim={dim} terms={base.action_manager.active_terms}")
    checks.check(
        "240 Hz physics, 24 substeps (10 Hz control)",
        abs(base.physics_dt - PHYSICS_DT) < 1e-9
        and base.cfg.decimation == DECIMATION
        and abs(base.step_dt - 0.1) < 1e-9,
        f"physics_dt={base.physics_dt:.6f} decimation={base.cfg.decimation} step_dt={base.step_dt:.4f}"
        f" physics_handles_decimation={getattr(base, '_physics_handles_decimation', None)}",
    )
    limits = robot.data.joint_pos_limits.torch[0].cpu().numpy()
    urdf = parse_joints(YAM_URDF)
    bad = {
        n: (limits[i].round(5).tolist(), [urdf[n].lower, urdf[n].upper])
        for i, n in enumerate(robot.joint_names)
        if abs(limits[i][0] - urdf[n].lower) > 1e-4 or abs(limits[i][1] - urdf[n].upper) > 1e-4
    }
    checks.check("runtime joint limits match URDF", not bad, f"mismatch={bad}" if bad else "8 joints")

    env.reset()
    q0 = robot.data.joint_pos.torch[0].cpu().numpy()
    checks.check(
        "reset puts the arm at READY, fingers open",
        np.abs(q0[:6] - READY).max() < 1e-4 and np.abs(q0[6:] - FINGER_OPEN).max() < 1e-4,
        f"q0={np.round(q0, 4).tolist()}",
    )

    # -- hold READY --------------------------------------------------------------------
    qs, fs, vs = hold(env, robot, READY, 1.0, args.hold_s)
    settled = qs[-10:]  # last 1 s
    err = np.abs(settled - np.array(READY)).max(axis=0)
    checks.check(
        f"hold READY: max arm joint error < {HOLD_TOL_RAD} rad (last 1 s of {args.hold_s:.0f} s)",
        err.max() < HOLD_TOL_RAD,
        f"max {err.max():.5f} rad ({np.degrees(err.max()):.3f} deg); per joint {np.round(err, 5).tolist()};"
        f" peak during settle {np.abs(qs - np.array(READY)).max():.5f} rad",
    )
    checks.check(
        "hold READY: fingers stay open",
        np.abs(fs[-10:] - FINGER_OPEN).max() < 1e-3,
        f"finger error {np.abs(fs[-10:] - FINGER_OPEN).max() * 1000:.3f} mm",
    )
    vmax = np.abs(vs[-10:]).max()
    checks.check("hold READY: at rest (|qd| < 0.01 rad/s)", vmax < 0.01, f"max |qd| {vmax:.5f}")
    root = robot.data.root_link_pose_w.torch[0, :3].cpu().numpy() - base.scene.env_origins[0].cpu().numpy()
    drift = np.linalg.norm(root - np.array(LEFT_BASE_POS))
    checks.check(
        "base stays at its mount (< 0.1 mm)",
        drift < 1e-4,
        f"base at {np.round(root, 5).tolist()}, drift {drift * 1000:.4f} mm",
    )
    fk_check("READY", robot)

    # -- second pose: tracking and FK -----------------------------------------------------
    qs, fs, vs = hold(env, robot, TEST_Q, 0.5, args.hold_s)
    err = np.abs(qs[-10:] - np.array(TEST_Q)).max(axis=0)
    checks.check(
        f"track TEST_Q {TEST_Q}: max error < {TRACK_TOL_RAD} rad",
        err.max() < TRACK_TOL_RAD,
        f"max {err.max():.5f} rad; per joint {np.round(err, 5).tolist()}",
    )
    grip = float(np.clip(fs[-1].mean() / FINGER_OPEN, 0, 1))
    checks.check("gripper 0.5 command -> half open", abs(grip - 0.5) < 0.02, f"measured normalized grip {grip:.4f}")
    fk_check("TEST_Q", robot)

    # -- gravity compensation off (informational) ------------------------------------------
    term = base.action_manager.get_term("left_arm")
    term.cfg.gravity_compensation = False
    qs, _, _ = hold(env, robot, READY, 1.0, args.hold_s)
    sag = np.abs(qs[-10:] - np.array(READY)).max(axis=0)
    checks.note("hold READY WITHOUT gravity compensation: per-joint error (rad)", np.round(sag, 5).tolist())
    term.cfg.gravity_compensation = True
    qs, _, _ = hold(env, robot, READY, 1.0, args.hold_s)
    err = np.abs(qs[-10:] - np.array(READY)).max()
    checks.check("gravity compensation back on: READY error recovers", err < TRACK_TOL_RAD, f"{err:.5f} rad")


try:
    main()
except Exception as exc:
    checks.error("stage 2", exc)
checks.finish()
