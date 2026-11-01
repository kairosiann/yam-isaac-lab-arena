"""Inspect Robots embodiment ``yam_arena``: the bimanual YAM in Isaac Lab-Arena 0.3.1.

Same contract as the real rig's ``yam_arms`` (inspect-robots-yam 0.37.0) and the 2.3 port's
``yam_sim``, so policies built for the real arms run against it unchanged:

- images ``top_cam`` (the overhead D435) and ``left_cam`` / ``right_cam`` (the D405 on each
  wrist), RGB uint8, rendered at 640x360 and resized (cv2.resize) to ``cam_height`` x ``cam_width``;
- state ``joint_pos``: 14-D ``[left j0..j5, left gripper, right j0..j5, right gripper]``, joints
  in rad, grippers normalized (0 closed, 1 open);
- action: the same 14-D layout as absolute joint targets, clipped to the URDF joint limits.

Each step holds the target for one control period (10 Hz: 24 PhysX substeps at 240 Hz) of joint
PD plus gravity compensation, the way i2rt drives the real arms.

``-E scene=tabletop`` (default) is the built-in table with a red cube and a plate; ``-E scene=station``
is Dream's calibrated station (``yam_arena.station``), its top camera as calibrated or, with
``-E top_cam=demo``, as Dream's demo moves it. On the station, a scene that asks for one of Dream's
tasks ("put apple into the bowl", "put orange into the bowl", "put banana on the plate"; or
``Target(kind="put", spec={"object": ..., "target": ...})``) gets Arena success detection: every step
reports ``info["distance"]`` (object to goal, m) and ``info["success"]``, and the step on which the
task's success criteria hold returns ``terminated=True, termination_reason="success"``.

``-E task=my_task.py`` loads a task file (``yam_arena.task_file``): its objects in the station's or the
tabletop's room, scored by the file's own ``success`` and ``distance``. Seeded resets shift every free
object by up to ``-E object_jitter`` metres (default 0.03); ``-E object_jitter=0`` keeps the layout fixed.

Importing this module is cheap: Isaac Sim starts on the first ``reset()``. One Kit app is
started per process and reused by every instance.
"""

from __future__ import annotations

import atexit
import os
import sys
import threading
from typing import Any

import numpy as np
from inspect_robots import Scene
from inspect_robots.embodiment import PRIVILEGED_SUCCESS, RESETTABLE, SEEDABLE, EmbodimentBase, EmbodimentInfo
from inspect_robots.types import Observation, StepResult
from inspect_robots_yam.config import DEFAULT_CAMERAS, action_box, observation_space
from inspect_robots_yam.packing import ARM_DOF, STATE_KEY, pack, split

from yam_arena import YAM_URDF, station, task_file
from yam_arena.constants import ARM_JOINTS, CONTROL_HZ, FINGER_JOINTS, FINGER_OPEN, READY
from yam_arena.rig import TABLETOP
from yam_arena.urdf import parse_joints

ARMS = ("left_robot", "right_robot")
# Settle time after a reset, so the first observation shows the arms at rest.
SETTLE_S = 0.5
# Extra renders after a reset: the RTX renderer needs a few frames to converge.
RESET_RENDERS = 4
# Seeded resets shift every free object (tabletop: the cube and plate; station: the fruit and the bottle,
# not the fixed bowl and plate) by up to this much in x and y.
OBJECT_JITTER = 0.03
# Seconds Kit gets to shut down at interpreter exit before the process is ended anyway.
CLOSE_TIMEOUT_S = 30.0

JOINT_GUIDE = """Two identical 6-DoF arms, prefixed left_ and right_, each with a parallel-jaw gripper.
Each arm has its own base frame: +x forward (where the gripper points at all-zero joints), +y left,
+z up. Joints and their positive direction, the same on both arms:
- j0: base yaw about vertical; positive turns the arm counterclockwise seen from above.
- j1: shoulder pitch; 0 holds the upper arm level and pointing back (its lower limit), about 1.57
  points it straight up and about 3.14 level and forward.
- j2: elbow; 0 folds the forearm back against the upper arm (its lower limit), positive opens it.
- j3: wrist pitch, parallel to the elbow; positive tilts the gripper up.
- j4: wrist yaw; positive swings the gripper to the arm's right seen from above, the opposite
  sense to j0.
- j5: wrist roll about the gripper's pointing axis; positive turns it clockwise seen from behind.
- gripper: 0 is closed, 1 is open (about 9.4 cm between the jaws).
Upper arm 0.26 m, forearm 0.25 m, wrist to grasp point 0.25 m. At all-zero joints the arm is
folded low with the gripper pointing forward. While it is folded, one joint can move the gripper
in unexpected ways: move in small steps and check the observation after each."""

DOCS = {
    "tabletop": JOINT_GUIDE
    + """

Simulated rig (Isaac Lab-Arena 0.3.1, PhysX): both arms are mounted on the near edge of a table
0.6 m apart, left arm at +y, both facing +x across it; each arm's base frame is its own mount.
On the table: a red 4 cm cube in front of the left arm and a white plate in front of the right.
Cameras: top_cam looks down over the arms from behind (its D435 body is a solid obstacle);
left_cam and right_cam are on the wrists, looking out along the fingers.""",
    "station": JOINT_GUIDE
    + """

Simulated rig (Isaac Lab-Arena 0.3.1, PhysX): Dream's calibrated YAM station. Both arms are mounted
0.61 m apart at the near edge of an oak table, left arm at +y, both facing +x across it; each arm's
base frame is its own mount. On the table, as seen from behind the arms: an apple in front of the
left arm, an orange between the arms, a banana in front of the right arm, a white bowl further out
in the middle, a white plate further out on the right, and a teal water bottle at the far left.
The bowl and plate are fixed. Cameras: top_cam looks down over the arms from a stand behind them;
left_cam and right_cam are on the wrists, looking out along the fingers.""",
}
# The rig without its objects, for task files (whose docstring describes what is on the table).
ROOM_DOCS = {
    "tabletop": """Simulated rig (Isaac Lab-Arena 0.3.1, PhysX): both arms are mounted on the near edge of a table
0.6 m apart, left arm at +y, both facing +x across it; each arm's base frame is its own mount.
Cameras: top_cam looks down over the arms from behind (its D435 body is a solid obstacle);
left_cam and right_cam are on the wrists, looking out along the fingers.""",
    "station": """Simulated rig (Isaac Lab-Arena 0.3.1, PhysX): Dream's calibrated YAM station. Both arms are mounted
0.61 m apart at the near edge of an oak table, left arm at +y, both facing +x across it; each arm's
base frame is its own mount. Cameras: top_cam looks down over the arms from a stand behind them;
left_cam and right_cam are on the wrists, looking out along the fingers.""",
}


def environment_id(rig_name: str, task: task_file.TaskFile | None = None) -> str:
    """The id an embodiment reports for a rig (and task file); the yam_scripted policy reads the rig from it."""
    base = f"isaaclab-arena-0.3.1-yam-bimanual-{rig_name}"
    return base if task is None else f"{base}-task-{task.name}"


_APP: Any = None  # one SimulationApp per process


def _urdf_arm_limits() -> tuple[np.ndarray, np.ndarray]:
    """Joint limits of joint1..joint6 straight from the URDF, so ``info`` needs no Isaac."""
    joints = parse_joints(YAM_URDF)
    return np.array([joints[n].lower for n in ARM_JOINTS]), np.array([joints[n].upper for n in ARM_JOINTS])


def _close_app_at_exit() -> None:
    """Close Kit at interpreter exit, but never hang there: Isaac Sim can stall in close()."""
    if _APP is None:
        return
    sys.stdout.flush()
    sys.stderr.flush()

    def _force_exit() -> None:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    timer = threading.Timer(CLOSE_TIMEOUT_S, _force_exit)
    timer.daemon = True
    timer.start()
    try:
        _APP.close()
    finally:
        timer.cancel()


class YamArenaEmbodiment(EmbodimentBase):
    """Bimanual YAM in Isaac Lab-Arena, observed through the rig's overhead and wrist cameras."""

    def __init__(
        self,
        *,
        headless: bool = True,
        device: str = "cuda:0",
        cam_height: int = 224,
        cam_width: int = 224,
        control_hz: float = CONTROL_HZ,
        wrist_cam_mount: str = "link",
        scene: str | None = None,
        top_cam: str | None = None,
        task: str | None = None,
        object_jitter: float = OBJECT_JITTER,
    ) -> None:
        self.task = task_file.load(task) if task else None
        if self.task is not None and scene not in (None, self.task.scene):
            raise ValueError(f"{self.task.path.name} sets SCENE={self.task.scene!r}; drop -E scene={scene}")
        scene = self.task.scene if self.task is not None else (scene or "tabletop")
        if scene not in DOCS:
            raise ValueError(f"scene must be one of {tuple(DOCS)}, got {scene!r}")
        if top_cam is not None and scene != "station":
            raise ValueError("top_cam applies to scene=station")
        self.object_jitter = float(object_jitter)
        if not self.object_jitter >= 0:
            raise ValueError(f"object_jitter must be >= 0 metres, got {object_jitter!r}")
        self.headless = headless
        self.device = device
        self.cam_height, self.cam_width = int(cam_height), int(cam_width)
        self.wrist_cam_mount = wrist_cam_mount
        self.scene = scene
        self.rig = station.rig(top_cam or "calibrated") if scene == "station" else TABLETOP
        lower, upper = _urdf_arm_limits()
        per_arm_low, per_arm_high = np.append(lower, 0.0), np.append(upper, 1.0)
        self._low, self._high = pack(per_arm_low, per_arm_low), pack(per_arm_high, per_arm_high)
        if self.task is None:
            scored, targets, docs = scene == "station", scene == "station", DOCS[scene]
        else:
            scored, targets = self.task.success is not None, False
            docs = "\n\n".join(d for d in (JOINT_GUIDE, ROOM_DOCS[scene], self.task.docs) if d)
        self.info = EmbodimentInfo(
            name="yam_arena",
            action_space=action_box(self._low, self._high),
            observation_space=observation_space(self.cam_height, self.cam_width, DEFAULT_CAMERAS),
            control_hz=float(control_hz),
            is_simulated=True,
            capabilities=frozenset({SEEDABLE, RESETTABLE} | ({PRIVILEGED_SUCCESS} if scored else set())),
            supported_target_kinds=frozenset({station.TARGET_KIND}) if targets else frozenset(),
            docs=docs,
            environment_id=environment_id(self.rig.name, self.task),
            environment_revision=None if self.task is None else self.task.sha256[:12],
        )
        if self.task is not None:
            task_file.register(self.info.environment_id, self.task)
        self._env: Any = None
        self._base: Any = None
        self._goal: station.Goal | None = None
        self._instruction: str | None = None
        self._t = 0.0
        self._held = 0  # consecutive control steps a task file's success() has held
        self._best_score = 0.0  # the best a task file's score() has reached this episode

    # -- Isaac lifecycle ---------------------------------------------------------------

    def _ensure_sim(self, goal: station.Goal | None) -> None:
        """Start Kit and build the env, rebuilding it when an episode asks for a different task."""
        global _APP
        if self._env is not None and goal == self._goal:
            return
        if self._env is not None:
            self._env.close()
            self._env = self._base = None
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        if _APP is None:
            from isaaclab.app import AppLauncher

            kwargs: dict[str, Any] = {"headless": self.headless, "device": self.device, "enable_cameras": True}
            if not self.headless:
                kwargs["visualizer"] = ["kit"]  # Isaac Lab 3.0: a window needs a Kit visualizer (untested)
            _APP = AppLauncher(**kwargs).app
            atexit.register(_close_app_at_exit)

        import torch

        from yam_arena.embodiment import BimanualYamEmbodiment
        from yam_arena.scene import FREE_OBJECTS, make_env

        emb = BimanualYamEmbodiment(
            enable_cameras=True, wrist_cam_mount=self.wrist_cam_mount, contact_sensors=False, rig=self.rig
        )
        if self.task is not None:
            scene, build = "task", {"task": self.task}
            self._free_objects = self.task.free_objects
        else:
            scene, build = self.scene, ({"goal": goal} if self.scene == "station" else {})
            self._free_objects = FREE_OBJECTS[self.scene]
        self._env = make_env(emb, scene=scene, device=self.device, control_hz=self.info.control_hz, **build)
        self._base = self._env.unwrapped
        self._goal = goal
        self._torch = torch
        self._ready = torch.tensor([[*READY, 1.0, *READY, 1.0]], dtype=torch.float32, device=self._base.device)
        self._ids = {
            name: (
                self._base.scene[name].find_joints(ARM_JOINTS, preserve_order=True)[0],
                self._base.scene[name].find_joints(FINGER_JOINTS, preserve_order=True)[0],
            )
            for name in ARMS
        }

    def close(self) -> None:
        """Drop the env; Kit itself closes at interpreter exit.

        ``SimulationApp.close()`` ends the process, which would swallow whatever the caller prints
        afterwards (the inspect-robots CLI prints its results after ``close()``).
        """
        self._env = self._base = None

    # -- Embodiment contract -----------------------------------------------------------

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        """Return both arms to READY, put every object back (free ones jittered if seeded), and settle."""
        on_station = self.scene == "station" and self.task is None
        self._ensure_sim(station.goal_for(scene.instruction, scene.target) if on_station else None)
        torch = self._torch
        base = self._base
        self._instruction = scene.instruction
        self._held, self._best_score = 0, 0.0
        self._env.reset()
        if seed is not None and self.object_jitter > 0:
            rng = np.random.default_rng(seed)
            for name in self._free_objects:
                obj = base.scene[name]
                pose = obj.data.default_root_pose.torch.clone()
                pose[:, :3] += base.scene.env_origins
                jitter = rng.uniform(-self.object_jitter, self.object_jitter, size=2)
                pose[:, :2] += torch.tensor(jitter, dtype=pose.dtype, device=pose.device)
                obj.write_root_pose_to_sim_index(root_pose=pose)
                obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros_like(obj.data.default_root_vel.torch))
        for _ in range(max(1, round(SETTLE_S / base.step_dt))):
            self._env.step(self._ready)
        for _ in range(RESET_RENDERS):
            base.sim.render()
        self._t = 0.0
        return self._observation()

    def step(self, action: Any) -> StepResult:
        """Hold the 14-D absolute joint target for one control period."""
        target = np.clip(np.asarray(action.data, dtype=np.float64).reshape(-1), self._low, self._high)
        left, right = split(target)
        packed = np.concatenate([left, right])  # validates the 14-D layout
        a = self._torch.tensor(packed, dtype=self._torch.float32, device=self._base.device).unsqueeze(0)
        self._env.step(a)
        self._t += self._base.step_dt
        if self.task is not None:
            return self._task_step()
        if self._goal is None:
            return StepResult(observation=self._observation())
        from yam_arena.scene import goal_status

        success, distance = goal_status(self._env, self._goal)
        return StepResult(
            observation=self._observation(),
            terminated=success,
            termination_reason="success" if success else None,
            info={"success": success, "distance": distance},
        )

    # -- internals ---------------------------------------------------------------------

    def _task_step(self) -> StepResult:
        """Score a task file's step: its success() must hold for success_hold_s to end the episode; its distance()
        and score() go in ``info``."""
        from yam_arena.scene import object_states

        task = self.task
        states = object_states(self._env, task.names)
        info: dict[str, Any] = {}
        success = False
        if task.success is not None:
            self._held = self._held + 1 if task.success(states) else 0
            success = self._held >= max(1, round(task.success_hold_s * self.info.control_hz))
            info["success"] = success
        if task.distance is not None:
            info["distance"] = float(task.distance(states))
        if task.score is not None:
            now = min(max(float(task.score(states)), 0.0), 1.0)
            self._best_score = max(self._best_score, now)
            info["score"] = self._best_score if task.score_over == "best" else now
        return StepResult(
            observation=self._observation(),
            terminated=success,
            termination_reason="success" if success else None,
            info=info,
        )

    def _observation(self) -> Observation:
        import cv2

        base = self._base
        images = {}
        for name in DEFAULT_CAMERAS:
            out = base.scene[name].data.output["rgb"]
            out = out.torch if hasattr(out, "torch") else out
            rgb = out[0, ..., :3].cpu().numpy()
            images[name] = np.ascontiguousarray(cv2.resize(rgb, (self.cam_width, self.cam_height)), dtype=np.uint8)
        per_arm = []
        for name in ARMS:
            arm_ids, finger_ids = self._ids[name]
            q = base.scene[name].data.joint_pos.torch[0]
            joints = q[arm_ids].cpu().numpy().astype(np.float64)
            # Both fingers mirror one rack; report their mean opening as the 0..1 gripper.
            grip = float(np.clip(q[finger_ids].mean().item() / FINGER_OPEN, 0.0, 1.0))
            per_arm.append(np.append(joints, grip))
        assert len(per_arm[0]) == ARM_DOF + 1
        return Observation(
            images=images,
            state={STATE_KEY: pack(*per_arm)},
            instruction=self._instruction,
            image_times=dict.fromkeys(images, self._t),
            state_time=self._t,
        )


def yam_arena_embodiment(**kwargs: Any) -> YamArenaEmbodiment:
    """Embodiment ``yam_arena``; ``-E scene=station [-E top_cam=demo]`` selects Dream's station, ``-E task=file.py``
    a task file, and ``-E object_jitter=0`` a fixed layout."""
    return YamArenaEmbodiment(**kwargs)
