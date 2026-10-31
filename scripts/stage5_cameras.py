"""Stage 5: the three policy cameras render RGB, and the wrist cameras follow the grippers.

    python scripts/stage5_cameras.py --out /data/arena031/out/yam-arena [--wrist-mount link|follow]

Replays the start of the 2.3 contact demo (both arms from READY to hover over the cube and the
plate) and, at the times the 2.3 screenshots were taken (view_scene.py saved every 120 physics
steps at 240 Hz: t = 0.5, 1.0, 1.5, 2.0, 2.5 s), saves:
  - native renders (640x360) of top_cam, left_cam, right_cam and an overview camera
  - the 224x224 observation images (cv2.resize, as the inspect-robots wrapper does)
Checks: shapes and dtype, frames are not blank, cameras differ from each other, the wrist
cameras' reported poses match their gripper mounts, and the wrist images change as the arm moves.
Whether the *rendered* wrist view follows the gripper is judged by looking at the PNGs.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from yam_arena.checks import Checks

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--out", type=Path, default=Path("/data/arena031/out/yam-arena"))
parser.add_argument("--wrist-mount", choices=("link", "follow"), default="link")
parser.add_argument("--obs-size", type=int, default=224)
args = parser.parse_args()
stage_name = f"stage5_cameras_{args.wrist_mount}"
checks = Checks(stage_name, args.out)
img_dir = args.out / stage_name
img_dir.mkdir(parents=True, exist_ok=True)

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True, enable_cameras=True).app

import cv2
import numpy as np
import torch
from PIL import Image

from yam_arena.analysis import pose7_to_matrix
from yam_arena.constants import READY, WRIST_CAM_POS, WRIST_CAM_ROT_XYZW
from yam_arena.demo import make_plans
from yam_arena.urdf import quat_xyzw_to_matrix, rotation_angle_deg

POLICY_CAMS = ("top_cam", "left_cam", "right_cam")
CAPTURE_T = (0.5, 1.0, 1.5, 2.0, 2.5)


def rgb(cam) -> np.ndarray:
    x = cam.data.output["rgb"]
    x = x.torch if hasattr(x, "torch") else x
    return x[0, ..., :3].cpu().numpy().astype(np.uint8)


def t_np(x):
    x = x.torch if hasattr(x, "torch") else x
    return x[0].cpu().numpy()


def main():
    from yam_arena.embodiment import BimanualYamEmbodiment
    from yam_arena.scene import make_env

    emb = BimanualYamEmbodiment(
        enable_cameras=True, overview_cam=True, wrist_cam_mount=args.wrist_mount, contact_sensors=False
    )
    t0 = time.monotonic()
    env = make_env(emb, objects=True, d435_collider=True)
    base = env.unwrapped
    checks.note("env build s", round(time.monotonic() - t0, 1))
    cams = {n: base.scene[n] for n in (*POLICY_CAMS, "overview_cam")}
    checks.note("camera classes", {n: type(c).__name__ for n, c in cams.items()})
    checks.note("camera prim paths", {n: c.cfg.prim_path for n, c in cams.items()})
    env.reset()
    dev = base.device
    ready = torch.tensor([[*READY, 1.0, *READY, 1.0]], dtype=torch.float32, device=dev)
    for _ in range(5):  # 0.5 s settle, like the wrapper's reset
        env.step(ready)
    for _ in range(4):  # extra renders so RTX converges, like the 2.3 wrapper
        base.sim.render()

    lplan, rplan = make_plans()
    frames: dict[float, dict[str, np.ndarray]] = {}
    step_times = []
    k = 0
    while k * base.step_dt < max(CAPTURE_T) - 1e-9:
        t = (k + 1) * base.step_dt
        la, _ = lplan.action7(t)
        ra, _ = rplan.action7(t)
        s0 = time.monotonic()
        env.step(torch.tensor([[*la, *ra]], dtype=torch.float32, device=dev))
        step_times.append(time.monotonic() - s0)
        k += 1
        tt = round(k * base.step_dt, 3)
        if any(abs(tt - c) < 1e-6 for c in CAPTURE_T):
            frames[tt] = {}
            for name, cam in cams.items():
                img = rgb(cam)
                frames[tt][name] = img
                Image.fromarray(img).save(img_dir / f"{name}_t{tt:.1f}.png")
                if name in POLICY_CAMS:
                    obs = cv2.resize(img, (args.obs_size, args.obs_size))
                    Image.fromarray(obs).save(img_dir / f"{name}_t{tt:.1f}_{args.obs_size}.png")
            # wrist camera pose vs its gripper mount
            for side in ("left", "right"):
                robot, cam = base.scene[f"{side}_robot"], cams[f"{side}_cam"]
                g = pose7_to_matrix(t_np(robot.data.body_link_pose_w)[robot.body_names.index("gripper")])
                exp_p = g[:3, :3] @ np.array(WRIST_CAM_POS) + g[:3, 3]
                exp_r = g[:3, :3] @ quat_xyzw_to_matrix(np.array(WRIST_CAM_ROT_XYZW))
                got_p = t_np(cam.data.pos_w)
                got_r = quat_xyzw_to_matrix(t_np(cam.data.quat_w_ros))
                frames[tt][f"{side}_pose_err"] = (
                    float(np.linalg.norm(got_p - exp_p) * 1000),
                    rotation_angle_deg(got_r, exp_r),
                )
    checks.note("mean env.step with 4 cameras (s)", round(float(np.mean(step_times[2:])), 3))

    last = frames[max(frames)]
    for name in POLICY_CAMS:
        img = last[name]
        obs = cv2.resize(img, (args.obs_size, args.obs_size))
        checks.check(
            f"{name}: {args.obs_size}x{args.obs_size} uint8 RGB, not blank",
            obs.shape == (args.obs_size, args.obs_size, 3) and obs.dtype == np.uint8 and img.std() > 5,
            f"render {img.shape}, obs {obs.shape} {obs.dtype}, std {img.std():.1f}, mean {img.mean():.1f}",
        )
    diffs = {
        f"{a}-{b}": float(np.abs(last[a].astype(int) - last[b].astype(int)).mean())
        for i, a in enumerate(POLICY_CAMS)
        for b in POLICY_CAMS[i + 1 :]
    }
    checks.check(
        "the three cameras see different views",
        min(diffs.values()) > 5,
        str({k: round(v, 1) for k, v in diffs.items()}),
    )
    for side in ("left", "right"):
        errs = [frames[t][f"{side}_pose_err"] for t in sorted(frames)]
        worst = (max(e[0] for e in errs), max(e[1] for e in errs))
        checks.check(
            f"{side}_cam reported pose sits on its gripper mount (all {len(errs)} captures)",
            worst[0] < 1.0 and worst[1] < 0.5,
            f"worst {worst[0]:.3f} mm / {worst[1]:.3f} deg",
        )
        change = float(np.abs(frames[min(frames)][f"{side}_cam"].astype(int) - last[f"{side}_cam"].astype(int)).mean())
        checks.check(
            f"{side}_cam image changes as the arm moves (t=0.5 vs 2.5 s)", change > 5, f"mean abs diff {change:.1f}"
        )
    checks.note("images", str(img_dir))


try:
    main()
except Exception as exc:
    checks.error("stage 5", exc)
checks.finish()
