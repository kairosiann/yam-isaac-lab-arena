"""Stage 3: both YAM arms (`left_robot` / `right_robot`), 14-D action, arm-arm contact.

    python scripts/stage3_bimanual.py [--out DIR]

Checks:
  - two fixed-base articulations; action manager terms [left_arm, left_gripper, right_arm,
    right_gripper] with dims [6, 1, 6, 1] = 14; the policy observation is the 14-D joint_pos
  - the layout really is [L j1..j6, L grip, R j1..j6, R grip]: distinct per-slot targets land on
    the right joints of the right arm, and the observation packs them back in the same order
  - both arms hold READY
  - arm-arm contact: the arms face each other (joint1 -/+90 deg) with closed grippers, posed so
    the commanded TCPs would cross by ~20 cm. Reported: the commanded overlap (URDF FK + hulls),
    the achieved minimum hull distance between the arms, the TCP gap, joint1 tracking error, and
    the contact forces on both arms. PASS needs the arms held apart (overlap <= 5 mm, grippers
    stopped >= 100 mm short of the commanded crossing, joints > 0.1 rad off their command), no
    blow-up, and both arms back at READY afterwards.
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
args = parser.parse_args()
checks = Checks("stage3_bimanual", args.out)

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import numpy as np
import torch

from yam_arena.analysis import fk_link_poses_world, hull_set_distance, world_hulls
from yam_arena.constants import ARM_JOINTS, FINGER_CLOSED, FINGER_OPEN, LEFT_BASE_POS, READY, RIGHT_BASE_POS
from yam_arena.ik import smoothstep
from yam_arena.runtime import arm_q, contact_forces, link_poses, tcp_world

# After a move, joints settle inside the joint-friction deadband (i2rt's friction, kp 10 on the
# wrist), which stage 2 measured at up to ~0.009 rad; from an exact start the hold error is 0.
HOLD_TOL_RAD = 0.02
LAYOUT_TOL_RAD = 0.03  # slots differ by >= 0.1 rad, so this still pins the mapping
# Arm pose with a horizontal approach axis and the TCP 0.40 m from the base (found offline with
# URDF FK); joint1 -/+90 deg turns the left/right arm toward the other.
PUSH_POSE = [0.0, 1.15, 0.65, 0.5, 0.0, 0.0]
NEAR_LINKS = ["link4", "link5", "gripper", "tip_left", "tip_right"]


def step(env, left7, right7, n=1):
    a = torch.tensor([[*left7, *right7]], dtype=torch.float32, device=env.unwrapped.device)
    for _ in range(n):
        env.step(a)


def main():
    from yam_arena.embodiment import BimanualYamEmbodiment
    from yam_arena.scene import make_env

    emb = BimanualYamEmbodiment(arms=("left", "right"), contact_sensors=True)
    env = make_env(emb, objects=False, d435_collider=False)
    base = env.unwrapped
    left, right = base.scene["left_robot"], base.scene["right_robot"]
    lc, rc = base.scene["left_contacts"], base.scene["right_contacts"]
    dt = base.step_dt

    checks.check(
        "articulations == {left_robot, right_robot}",
        set(base.scene.articulations) == {"left_robot", "right_robot"},
        str(sorted(base.scene.articulations)),
    )
    checks.check(
        "both fixed-base",
        left.is_fixed_base and right.is_fixed_base,
        f"left={left.is_fixed_base} right={right.is_fixed_base}",
    )
    am = base.action_manager
    terms = list(am.active_terms)
    dims = [am.get_term(t).action_dim for t in terms]
    checks.check("action dim == 14", am.total_action_dim == 14, f"total={am.total_action_dim}")
    checks.check(
        "term order [left_arm, left_gripper, right_arm, right_gripper] dims [6,1,6,1]",
        terms == ["left_arm", "left_gripper", "right_arm", "right_gripper"] and dims == [6, 1, 6, 1],
        f"terms={terms} dims={dims}",
    )
    checks.note("contact sensor bodies", {"left": list(lc.body_names), "right": list(rc.body_names)})

    obs, _ = env.reset()
    jp = obs["policy"]["joint_pos"]
    checks.check("policy obs joint_pos is (1, 14)", tuple(jp.shape) == (1, 14), f"shape={tuple(jp.shape)}")

    # -- layout: distinct targets per slot --------------------------------------------------
    lq = [0.25, 1.1, 0.9, -0.5, 0.3, -0.4]
    rq = [-0.3, 0.9, 1.1, -0.7, -0.2, 0.5]
    step(env, [*lq, 0.0], [*rq, 1.0], n=30)
    got_l, got_r = arm_q(left), arm_q(right)
    gl = float(np.clip(left.data.joint_pos.torch[0, 6:8].mean().item() / FINGER_OPEN, 0, 1))
    gr = float(np.clip(right.data.joint_pos.torch[0, 6:8].mean().item() / FINGER_OPEN, 0, 1))
    ok = (
        np.abs(got_l - lq).max() < LAYOUT_TOL_RAD
        and np.abs(got_r - rq).max() < LAYOUT_TOL_RAD
        and gl < 0.05
        and gr > 0.95
    )
    checks.check(
        "slots map to [L j1..j6, L grip, R j1..j6, R grip]",
        ok,
        f"left err {np.abs(got_l - lq).max():.4f} rad, right err {np.abs(got_r - rq).max():.4f} rad,"
        f" left grip {gl:.3f} (cmd 0), right grip {gr:.3f} (cmd 1)",
    )
    obs14 = base.observation_manager.compute()["policy"]["joint_pos"][0].cpu().numpy()
    expect = np.r_[got_l, gl, got_r, gr]
    checks.check(
        "obs joint_pos packs [L 7, R 7] in the same order",
        np.abs(obs14 - expect).max() < 1e-3,
        f"max diff {np.abs(obs14 - expect).max():.2e}",
    )

    # -- hold READY ---------------------------------------------------------------------------
    trace = []
    for _ in range(30):
        step(env, [*READY, 1.0], [*READY, 1.0])
        trace.append(np.r_[arm_q(left), arm_q(right)])
    err = np.abs(np.array(trace[-10:]) - np.r_[READY, READY]).max()
    checks.check(f"both arms hold READY (< {HOLD_TOL_RAD} rad)", err < HOLD_TOL_RAD, f"max error {err:.5f} rad")

    # -- arm-arm contact -----------------------------------------------------------------
    closed = FINGER_CLOSED / FINGER_OPEN  # 0.0
    # Commanded end pose: what the geometry would do with no collision.
    q_l = dict(zip(ARM_JOINTS, [-np.pi / 2, *PUSH_POSE[1:]], strict=True)) | {"joint7": 0.0, "joint8": 0.0}
    q_r = dict(zip(ARM_JOINTS, [np.pi / 2, *PUSH_POSE[1:]], strict=True)) | {"joint7": 0.0, "joint8": 0.0}
    cmd_l, cmd_r = fk_link_poses_world(q_l, LEFT_BASE_POS), fk_link_poses_world(q_r, RIGHT_BASE_POS)
    cmd_dist, cmd_pair = hull_set_distance(world_hulls(cmd_l, NEAR_LINKS), world_hulls(cmd_r, NEAR_LINKS))
    from yam_arena.constants import TCP_OFFSET

    def tcp_from(poses):
        g = poses["gripper"]
        return g[:3, :3] @ np.array(TCP_OFFSET) + g[:3, 3]

    cmd_gap = tcp_from(cmd_l)[1] - tcp_from(cmd_r)[1]
    checks.note(
        "commanded (no-collision) end pose",
        {
            "min_hull_distance_mm": round(cmd_dist * 1000, 2),
            "pair": cmd_pair,
            "tcp_y_gap_mm (left minus right; negative = crossed)": round(cmd_gap * 1000, 1),
        },
    )

    # Move to the facing-forward push pose with closed grippers, then turn both joint1 inward.
    step(env, [*PUSH_POSE, closed], [*PUSH_POSE, closed], n=20)
    ramp_s, hold_s = 4.0, 2.0
    n_ramp, n_hold = round(ramp_s / dt), round(hold_s / dt)
    min_dist, min_pair, peak_l, peak_r, peak_vel, nan = np.inf, None, {}, {}, 0.0, False
    first_contact_t = None
    gaps = []
    for k in range(n_ramp + n_hold):
        s = smoothstep((k + 1) / n_ramp)
        a = s * np.pi / 2
        step(env, [-a, *PUSH_POSE[1:], closed], [a, *PUSH_POSE[1:], closed])
        q = torch.cat([left.data.joint_pos.torch, right.data.joint_pos.torch], dim=1)
        nan |= not bool(torch.isfinite(q).all())
        peak_vel = max(
            peak_vel, float(torch.cat([left.data.joint_vel.torch, right.data.joint_vel.torch], 1).abs().max())
        )
        fl, fr = contact_forces(lc), contact_forces(rc)
        for d, f in ((peak_l, fl), (peak_r, fr)):
            for name, v in f.items():
                d[name] = max(d.get(name, 0.0), v)
        if first_contact_t is None and max(fl.values()) > 0.05:
            first_contact_t = (k + 1) * dt
        dist, pair = hull_set_distance(
            world_hulls(link_poses(left), NEAR_LINKS), world_hulls(link_poses(right), NEAR_LINKS)
        )
        if dist < min_dist:
            min_dist, min_pair = dist, pair
        gaps.append(tcp_world(left)[1] - tcp_world(right)[1])
    j1_err = (abs(arm_q(left)[0] + np.pi / 2), abs(arm_q(right)[0] - np.pi / 2))
    deflect_l = arm_q(left) - np.array([-np.pi / 2, *PUSH_POSE[1:]])
    deflect_r = arm_q(right) - np.array([np.pi / 2, *PUSH_POSE[1:]])
    final_dist, _ = hull_set_distance(
        world_hulls(link_poses(left), NEAR_LINKS), world_hulls(link_poses(right), NEAR_LINKS)
    )

    checks.note(
        "achieved",
        {
            "min_hull_distance_mm (negative = overlap)": round(min_dist * 1000, 2),
            "pair": min_pair,
            "final_hull_distance_mm": round(final_dist * 1000, 2),
            "min_tcp_y_gap_mm": round(min(gaps) * 1000, 1),
            "final_tcp_y_gap_mm": round(gaps[-1] * 1000, 1),
            "joint1_final_error_rad (left, right)": [round(float(v), 3) for v in j1_err],
            "final joint deflection from command, rad (left j1..j6)": np.round(deflect_l, 3).tolist(),
            "final joint deflection from command, rad (right j1..j6)": np.round(deflect_r, 3).tolist(),
            "first_contact_s_into_ramp": first_contact_t,
            "peak_contact_N_left": {k: round(v, 1) for k, v in peak_l.items() if v > 0.05},
            "peak_contact_N_right": {k: round(v, 1) for k, v in peak_r.items() if v > 0.05},
            "peak_joint_speed_rad_s": round(peak_vel, 3),
        },
    )
    checks.check(
        "commanded pose would overlap (test is meaningful)",
        cmd_dist < -0.005 and cmd_gap < -0.1,
        f"commanded hull distance {cmd_dist * 1000:.1f} mm, TCP gap {cmd_gap * 1000:.0f} mm",
    )
    checks.check(
        "no interpenetration: max hull overlap <= 5 mm (sampled at 10 Hz)",
        min_dist > -0.005,
        f"min hull distance {min_dist * 1000:.2f} mm ({min_pair}); commanded {cmd_dist * 1000:.1f} mm",
    )
    short = gaps[-1] - cmd_gap
    checks.check(
        "contact stops the grippers >= 100 mm short of the commanded crossing",
        short > 0.1,
        f"final TCP y gap {gaps[-1] * 1000:.1f} mm vs commanded {cmd_gap * 1000:.0f} mm ({short * 1000:.0f} mm short);"
        f" min gap during push {min(gaps) * 1000:.1f} mm",
    )
    worst = max(np.abs(deflect_l).max(), np.abs(deflect_r).max())
    checks.check(
        "the arms are held off their command (max joint deflection > 0.1 rad)",
        worst > 0.1,
        f"max deflection {worst:.3f} rad; joint1 only {j1_err[0]:.3f}/{j1_err[1]:.3f} rad short --"
        " the compliant wrists (kp 10) give way and the grippers slide partly past each other",
    )
    checks.check(
        "no NaN, peak joint speed < 10 rad/s", not nan and peak_vel < 10.0, f"nan={nan} peak {peak_vel:.2f} rad/s"
    )

    # -- back to READY -------------------------------------------------------------------------
    step(env, [*PUSH_POSE, closed], [*PUSH_POSE, closed], n=20)
    step(env, [*READY, 1.0], [*READY, 1.0], n=30)
    err = max(np.abs(arm_q(left) - READY).max(), np.abs(arm_q(right) - READY).max())
    checks.check("arms recover to READY after contact", err < HOLD_TOL_RAD, f"max error {err:.5f} rad")


try:
    main()
except Exception as exc:
    checks.error("stage 3", exc)
checks.finish()
