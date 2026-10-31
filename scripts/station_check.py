"""Station check: Dream's calibrated station scene (``-E scene=station``) and its success detection.

    python scripts/station_check.py --out /data/yam-dream/station-work/out/station

1. Renders, for both top-camera variants: the three cameras at Dream's reference poses (home: all
   joints 0; ready: READY; mixed: left READY, right [0.3, 1.2, 0.8, -0.4, 0.2, 0.5], grippers half
   open), after the objects have settled for 1.8 s, as 640x360 renders and as the 320x176 policy view
   the Inspect wrapper sends (cv2 resize). Checks each camera's reported pose and intrinsics.
2. Objects: Arena library assets at Dream's sizes and masses; every object rests stably for 5 s.
3. Success, through the Inspect Robots wrapper, for each of Dream's three tasks: no success while the
   object rests where it starts or sits right beside its goal; success (terminated, reason "success")
   once it is dropped into the bowl / onto the plate; ``info["distance"]`` every step.
4. End to end through ``inspect_robots.eval`` at 20 Hz: a scripted (privileged) pick-and-place of the
   apple into the bowl by the left arm scores ``success_at_end`` 1; holding still scores 0.
5. Timings: env build, reset, and the wrapper's 20 Hz step on the tabletop and the station.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from yam_arena.checks import Checks  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--out", type=Path, default=Path("/data/yam-dream/station-work/out/station"))
parser.add_argument("--parts", default="renders,objects,success,inspect,timing", help="comma-separated subset to run")
args = parser.parse_args()
checks = Checks("station_check", args.out)
img_dir = args.out / "renders"
img_dir.mkdir(parents=True, exist_ok=True)

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True, enable_cameras=True).app

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from yam_arena import station  # noqa: E402
from yam_arena.analysis import pose7_to_matrix  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, READY  # noqa: E402
from yam_arena.urdf import quat_xyzw_to_matrix, rotation_angle_deg  # noqa: E402

CAMS = ("top_cam", "left_cam", "right_cam")
POSES = {
    "home": ([0.0] * 6 + [1.0], [0.0] * 6 + [1.0]),
    "ready": (READY + [1.0], READY + [1.0]),
    "mixed": (READY + [0.5], [0.3, 1.2, 0.8, -0.4, 0.2, 0.5, 0.5]),
}
SETTLE_S = 1.8  # Dream's reference renders let the objects settle this long first
EXPECTED = {"apple": 0.066, "orange": 0.064, "banana": 0.140, "bowl": 0.166, "plate": 0.146}


def t_np(x):
    x = x.torch if hasattr(x, "torch") else x
    return x.cpu().numpy() if hasattr(x, "cpu") else x.numpy()


def rgb(cam) -> np.ndarray:
    return t_np(cam.data.output["rgb"])[0, ..., :3].astype(np.uint8)


def act(env, left7, right7, n=1):
    a = torch.tensor([[*left7, *right7]], dtype=torch.float32, device=env.unwrapped.device)
    for _ in range(n):
        env.step(a)


def put_arms(env, left7, right7):
    """Teleport both arms to a pose and hold it there (the reference renders set the joints kinematically)."""
    base = env.unwrapped
    for name, q7 in (("left_robot", left7), ("right_robot", right7)):
        robot = base.scene[name]
        q = robot.data.default_joint_pos.torch.clone()
        arm = robot.find_joints([f"joint{i}" for i in range(1, 7)], preserve_order=True)[0]
        fingers = robot.find_joints(["joint7", "joint8"], preserve_order=True)[0]
        q[0, arm] = torch.tensor(q7[:6], dtype=q.dtype, device=q.device)
        q[0, fingers] = q7[6] * -0.04695
        robot.write_joint_position_to_sim_index(position=q)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
    act(env, left7, right7, n=3)


def make_station(variant: str, goal=None):
    from yam_arena.embodiment import BimanualYamEmbodiment
    from yam_arena.scene import make_env

    emb = BimanualYamEmbodiment(enable_cameras=True, contact_sensors=False, rig=station.rig(variant))
    t0 = time.monotonic()
    env = make_env(emb, scene="station", goal=goal)
    env.reset()
    return env, emb, time.monotonic() - t0


def renders(variant: str):
    env, emb, build_s = make_station(variant)
    base = env.unwrapped
    checks.note(f"{variant}: env build + reset (s)", round(build_s, 1))
    put_arms(env, *POSES["home"])
    act(env, *POSES["home"], n=round(SETTLE_S / base.step_dt))
    for pose, (left7, right7) in POSES.items():
        put_arms(env, left7, right7)
        for name in CAMS:
            img = rgb(base.scene[name])
            stem = f"ours_{variant}_{pose}_{name.removesuffix('_cam')}"
            Image.fromarray(img).save(img_dir / f"{stem}_640x360.png")
            Image.fromarray(cv2.resize(img, (320, 176))).save(img_dir / f"{stem}_320x176.png")
        if pose == "mixed":
            check_camera_poses(base, emb, variant)
    for name in CAMS:
        cam = base.scene[name]
        k = t_np(cam.data.intrinsic_matrices)[0]
        lens = emb.rig.top_cam.lens if name == "top_cam" else emb.rig.wrist_cams["left"].lens
        want = np.array([lens.fx, lens.fy, lens.cx, lens.cy])
        got = np.array([k[0, 0], k[1, 1], k[0, 2], k[1, 2]])
        checks.check(f"{variant}: {name} renders with the calibrated intrinsics (640x360 crop)",
                     np.abs(got - want).max() < 0.01, f"fx fy cx cy = {np.round(got, 3).tolist()}")
    env.close()


def check_camera_poses(base, emb, variant):
    top = base.scene["top_cam"]
    want_r = quat_xyzw_to_matrix(np.array(emb.rig.top_cam.rot_xyzw))
    got_r = quat_xyzw_to_matrix(t_np(top.data.quat_w_opengl)[0])
    pos_err = np.linalg.norm(t_np(top.data.pos_w)[0] - t_np(base.scene.env_origins)[0] - emb.rig.top_cam.pos)
    angle = rotation_angle_deg(got_r, want_r)
    checks.check(f"{variant}: top_cam at the calibrated pose", pos_err < 1e-4 and angle < 0.05,
                 f"{pos_err * 1000:.3f} mm, {angle:.4f} deg")
    for side in ("left", "right"):
        robot, cam, mount = base.scene[f"{side}_robot"], base.scene[f"{side}_cam"], emb.rig.wrist_cams[side]
        g = pose7_to_matrix(t_np(robot.data.body_link_pose_w)[0][robot.body_names.index("gripper")])
        exp_p = g[:3, :3] @ np.array(mount.pos) + g[:3, 3]
        exp_r = g[:3, :3] @ quat_xyzw_to_matrix(np.array(mount.rot_xyzw))
        dp = np.linalg.norm(t_np(cam.data.pos_w)[0] - exp_p) * 1000
        dr = rotation_angle_deg(quat_xyzw_to_matrix(t_np(cam.data.quat_w_opengl)[0]), exp_r)
        checks.check(f"{variant}: {side}_cam on Dream's link_6 mount", dp < 1.0 and dr < 0.5,
                     f"{dp:.3f} mm, {dr:.4f} deg")


def objects():
    env, _, _ = make_station("calibrated")
    base = env.unwrapped
    world = base.arena_world
    origin = t_np(base.scene.env_origins)[0]
    for name, extent in EXPECTED.items():
        box = world.get_aabb_w(name)
        size = t_np(box.max_point)[0] - t_np(box.min_point)[0]
        checks.check(f"{name}: Dream's size ({extent * 1000:.0f} mm across)", abs(max(size[:2]) - extent) < 0.002,
                     f"aabb {np.round(size * 1000, 1).tolist()} mm")
    for name in station.FREE_OBJECTS:
        mass = float(t_np(base.scene[name].root_physx_view.get_masses()).sum())
        want = station.BOTTLE_MASS if name == "bottle" else station.ITEMS[name].mass
        checks.check(f"{name}: Dream's mass", abs(mass - want) < 1e-3, f"{mass:.4f} kg (Dream {want} kg)")
    names = (*station.FREE_OBJECTS, "bowl", "plate")
    traj = {n: [] for n in names}
    t0 = time.monotonic()
    steps = round(5.0 / base.step_dt)
    for _ in range(steps):
        act(env, READY + [1.0], READY + [1.0])
        for n in names:
            traj[n].append(t_np(base.scene[n].data.root_link_pose_w)[0].copy())
    checks.note("mean env.step, 3 cameras 640x360 (s)", round((time.monotonic() - t0) / steps, 4))
    first = round(0.5 / base.step_dt)
    for n in names:
        p = np.array(traj[n])
        drift = np.linalg.norm(p[first:, :3] - p[first, :3], axis=1).max()
        tilt = max(rotation_angle_deg(quat_xyzw_to_matrix(q[3:]), quat_xyzw_to_matrix(p[first, 3:])) for q in p[first:])
        speed = float(np.linalg.norm(t_np(base.scene[n].data.root_com_vel_w)[0, :3]))
        drop = (p[0, 2] - p[-1, 2]) * 1000
        checks.check(f"{n}: rests stably for 5 s (after 0.5 s: drift < 2 mm, tilt < 2 deg; final speed < 2 mm/s)",
                     drift < 0.002 and tilt < 2.0 and speed < 0.002,
                     f"drift {drift * 1000:.2f} mm, tilt {tilt:.2f} deg, speed {speed * 1000:.2f} mm/s, settled "
                     f"{drop:.1f} mm in the first step; rest pos {np.round(p[-1, :3] - origin, 4).tolist()}")
    env.close()


def success():
    import inspect_robots.types as irt
    from inspect_robots import Scene

    from yam_arena import inspect_embodiment

    inspect_embodiment._APP = app  # reuse this process's Kit app
    emb = inspect_embodiment.YamArenaEmbodiment(scene="station", cam_height=176, cam_width=320)
    hold = irt.Action(data=np.r_[READY, 1.0, READY, 1.0])

    def run(seconds):
        results = []
        for _ in range(round(seconds * emb.info.control_hz)):
            results.append(emb.step(hold))
            if results[-1].terminated:
                break
        return results

    def move(name, centre_xyz):
        """Move an object so its bounding-box centre is at centre_xyz (env frame), at rest."""
        base = emb._base
        obj = base.scene[name]
        world = base.arena_world
        pose = obj.data.root_link_pose_w.torch.clone()
        centre = torch.tensor(centre_xyz, dtype=pose.dtype, device=pose.device)
        pose[:, :3] += centre + base.scene.env_origins - world.get_centroid_w(name)
        obj.write_root_pose_to_sim_index(root_pose=pose)
        obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=pose.device))

    def half_height(name):
        box = emb._base.arena_world.get_aabb_w(name)
        return float((box.max_point - box.min_point)[0, 2]) / 2

    for goal in station.DREAM_TASKS:
        t0 = time.monotonic()
        obs = emb.reset(Scene(id=goal.object, instruction=goal.instruction))
        checks.note(f"{goal.instruction}: reset incl. env build (s)", round(time.monotonic() - t0, 1))
        shapes = {k: v.shape for k, v in obs.images.items()}
        checks.check(f"{goal.instruction}: observation is 3 x 320x176",
                     all(shape == (176, 320, 3) for shape in shapes.values()), str(shapes))
        at_rest = run(2.0)
        d0 = at_rest[-1].info["distance"]
        checks.check(f"{goal.instruction}: no success with the {goal.object} where it starts (2 s)",
                     not any(r.terminated for r in at_rest) and all("distance" in r.info for r in at_rest),
                     f"distance {d0:.3f} m")
        world = emb._base.arena_world
        target = t_np(world.get_centroid_w(goal.target))[0] - t_np(emb._base.scene.env_origins)[0]
        beside = (target + (0.0, 0.13, 0.0)) if goal.relation == "in" else (target + (-0.15, 0.0, 0.0))
        beside[2] = station.TABLE_TOP + half_height(goal.object) + 0.002
        move(goal.object, beside)
        near = run(3.0)
        checks.check(f"{goal.instruction}: no success with the {goal.object} right beside the {goal.target} (3 s)",
                     not any(r.terminated for r in near), f"distance {near[-1].info['distance']:.3f} m")
        box = world.get_aabb_w(goal.target)
        top = float(box.max_point[0, 2]) - float(emb._base.scene.env_origins[0, 2])
        move(goal.object, (target[0], target[1], top + half_height(goal.object) + 0.01))
        placed = run(6.0)
        last = placed[-1]
        steps = len(placed)
        checks.check(f"{goal.instruction}: success once the {goal.object} is dropped {goal.relation} the {goal.target}",
                     last.terminated and last.termination_reason == "success" and last.info.get("success") is True,
                     f"terminated={last.terminated} reason={last.termination_reason!r} after {steps} steps "
                     f"({steps / emb.info.control_hz:.1f} s), distance {last.info['distance']:.3f} m")
    t0 = time.monotonic()
    emb.reset(Scene(id="free", instruction="wave hello"))
    checks.check("an instruction outside Dream's tasks runs without success detection",
                 emb._goal is None and not emb.step(hold).info, f"rebuilt in {time.monotonic() - t0:.1f} s")
    emb._env.close()


def wrapper(scene: str):
    from yam_arena import inspect_embodiment

    inspect_embodiment._APP = app  # reuse this process's Kit app
    return inspect_embodiment.YamArenaEmbodiment(scene=scene, cam_height=176, cam_width=320, control_hz=20)


def inspect_run():
    import inspect_robots
    from inspect_robots import Scene
    from inspect_robots.policy import PolicyConfig, PolicyInfo
    from inspect_robots.scorer import min_distance_to_goal, success_at_end
    from inspect_robots.spaces import ObservationSpace
    from inspect_robots.task import Task
    from inspect_robots.types import Action, ActionChunk

    from yam_arena.demo import ArmPlan, Waypoint

    emb = wrapper("station")

    class ScriptedPick:
        """Left arm: pick the apple where the reset put it and drop it into the bowl (privileged state)."""

        def __init__(self, hold: bool = False):
            self.hold = hold
            self.info = PolicyInfo(name="scripted-pick", action_space=emb.info.action_space,
                                   observation_space=ObservationSpace(state_keys=frozenset({"joint_pos"})))
            self.config = PolicyConfig(action_horizon=1)

        def reset(self, scene):
            self.plan, self.t = None, 0.0

        def act(self, observation):
            if self.plan is None:
                world, origin = emb._base.arena_world, emb._base.scene.env_origins
                (ax, ay, az), (bx, by, _) = ((world.get_centroid_w(n) - origin)[0].tolist() for n in ("apple", "bowl"))
                top, release = station.TABLE_TOP + 0.16, station.TABLE_TOP + 0.115
                self.plan = ArmPlan([
                    Waypoint(1.5, (ax, ay, top), FINGER_OPEN, "hover"),
                    Waypoint(2.0, (ax, ay, az), FINGER_OPEN, "descend"),
                    Waypoint(1.0, (ax, ay, az), FINGER_CLOSED, "grip"),
                    Waypoint(2.0, (ax, ay, top), FINGER_CLOSED, "lift"),
                    Waypoint(2.0, (bx, by, top), FINGER_CLOSED, "carry"),
                    Waypoint(1.0, (bx, by, release), FINGER_CLOSED, "lower"),
                    Waypoint(1.0, (bx, by, release), FINGER_OPEN, "release"),
                    Waypoint(1.5, (bx, by, top + 0.04), FINGER_OPEN, "retreat"),
                ], station.BASES["left"])
            self.t += 1 / emb.info.control_hz
            left = np.r_[READY, 1.0] if self.hold else self.plan.action7(self.t)[0]
            return ActionChunk(actions=[Action(data=np.r_[left, READY, 1.0])])

    for label, hold in (("scripted pick-and-place", False), ("holding still", True)):
        task = Task(name="station-apple", scenes=[Scene(id="apple", instruction="put apple into the bowl")],
                    scorer=[success_at_end(), min_distance_to_goal()], max_steps=400)
        t0 = time.monotonic()
        logs = inspect_robots.eval(task, ScriptedPick(hold), emb, log_dir=str(args.out / "ir-logs"), grader=None, seed=1)
        log = logs[0]
        sample = log.samples[0]
        scores = dict(sample.reduced)
        want = 0.0 if hold else 1.0
        checks.check(f"inspect_robots.eval, {label}: success_at_end = {want:.0f}",
                     log.status == "success" and scores.get("success_at_end") == want,
                     f"scores {scores}, reasons {sample.termination_reasons}, {time.monotonic() - t0:.0f} s")
    emb._env.close()


def timing():
    import inspect_robots.types as irt
    from inspect_robots import Scene

    for scene in ("tabletop", "station"):
        emb = wrapper(scene)
        t0 = time.monotonic()
        emb.reset(Scene(id="t", instruction="put apple into the bowl"), seed=1)
        reset_s = time.monotonic() - t0
        hold = irt.Action(data=np.r_[READY, 1.0, READY, 1.0])
        emb.step(hold)
        t0 = time.monotonic()
        for _ in range(40):
            emb.step(hold)
        checks.note(f"{scene}: first reset incl. env build (s) / wrapper step at 20 Hz, 320x176 (s)",
                    [round(reset_s, 1), round((time.monotonic() - t0) / 40, 4)])
        emb._env.close()


for part in args.parts.split(","):
    try:
        if part == "renders":
            for variant in station.TOP_CAMS:
                renders(variant)
        else:
            {"objects": objects, "success": success, "inspect": inspect_run, "timing": timing}[part]()
    except Exception as exc:  # noqa: BLE001
        checks.error(part, exc)
checks.note("images", str(img_dir))
checks.finish()
