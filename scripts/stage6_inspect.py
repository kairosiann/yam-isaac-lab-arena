"""Stage 6: the ``yam_arena`` Inspect Robots embodiment.

    python scripts/stage6_inspect.py --out /data/arena031/out/yam-arena

1. Contract, without starting Isaac: the entry point resolves; 14-D action box with the URDF
   joint limits and 0..1 grippers; 14-D ``joint_pos`` state; three 224x224 RGB cameras; the same
   spaces inspect-robots-yam 0.37.0 builds for the real rig; ``assert_embodiment_conformant``.
2. Direct use: reset (unseeded and seeded), step, observation shapes/dtypes, state layout,
   seeded object jitter <= 3 cm.
3. A short Inspect Robots episode through ``inspect_robots.eval()`` (guardrails on) with a
   non-LLM hold policy defined here. Inspect Robots' own non-LLM policies (noop, random,
   scripted) are written for the 2-D CubePick mock and are rejected by the compatibility check
   for a 14-D joint embodiment; scripts/stage6_cli.sh records that.
"""

from __future__ import annotations

import argparse
import sys
import time
from importlib.metadata import entry_points
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from yam_arena.checks import Checks

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--out", type=Path, default=Path("/data/arena031/out/yam-arena"))
parser.add_argument("--steps", type=int, default=10)
args = parser.parse_args()
checks = Checks("stage6_inspect", args.out)

import numpy as np


def contract():
    eps = [ep for ep in entry_points(group="inspect_robots.embodiments") if ep.name == "yam_arena"]
    checks.check("entry point inspect_robots.embodiments:yam_arena", len(eps) == 1, eps[0].value if eps else "missing")
    factory = eps[0].load()
    t0 = time.monotonic()
    emb = factory()
    checks.check(
        "constructing the embodiment does not start Isaac",
        "isaacsim" not in sys.modules and time.monotonic() - t0 < 5,
        f"{time.monotonic() - t0:.2f} s, isaacsim imported={'isaacsim' in sys.modules}",
    )
    info = emb.info
    from inspect_robots.conformance import check_embodiment
    from inspect_robots_yam.config import DEFAULT_CAMERAS, action_box, observation_space
    from inspect_robots_yam.packing import DIM_LABELS, STATE_KEY

    from yam_arena import YAM_URDF
    from yam_arena.urdf import parse_joints

    a = info.action_space
    j = parse_joints(YAM_URDF)
    lo = np.array([j[f"joint{i}"].lower for i in range(1, 7)] + [0.0])
    hi = np.array([j[f"joint{i}"].upper for i in range(1, 7)] + [1.0])
    checks.check(
        "action: 14-D box, URDF limits + 0..1 grippers, joint_pos semantics",
        a.shape == (14,)
        and np.allclose(a.low, np.r_[lo, lo])
        and np.allclose(a.high, np.r_[hi, hi])
        and a.semantics.control_mode == "joint_pos"
        and tuple(a.semantics.dim_labels) == DIM_LABELS,
        f"shape={a.shape} mode={a.semantics.control_mode} labels={a.semantics.dim_labels[:2]}...",
    )
    ref_a = action_box(emb._low, emb._high)
    checks.check("action box identical to inspect-robots-yam's action_box()", a == ref_a or repr(a) == repr(ref_a), "")
    o = info.observation_space
    cams = {c.name: (c.height, c.width, c.channels) for c in o.cameras}
    checks.check(
        "cameras top_cam/left_cam/right_cam at 224x224x3",
        cams == dict.fromkeys(DEFAULT_CAMERAS, (224, 224, 3)),
        str(cams),
    )
    fields = {f.key: f.shape for f in o.state.fields}
    checks.check(
        "state: joint_pos (14,)", fields == {STATE_KEY: (14,)} and set(o.state_keys) == {STATE_KEY}, str(fields)
    )
    ref_o = observation_space(224, 224, DEFAULT_CAMERAS)
    checks.check(
        "observation space identical to inspect-robots-yam's observation_space()",
        o == ref_o or repr(o) == repr(ref_o),
        "",
    )
    rep = check_embodiment(info)
    checks.check(
        "inspect_robots.conformance.check_embodiment",
        rep.ok if hasattr(rep, "ok") else not rep.errors,
        rep.summary()[:400],
    )
    checks.note(
        "info",
        {
            "name": info.name,
            "control_hz": info.control_hz,
            "capabilities": sorted(info.capabilities),
            "environment_id": info.environment_id,
        },
    )
    return emb


def direct(emb):
    from inspect_robots import Scene

    t0 = time.monotonic()
    obs = emb.reset(Scene(id="direct", instruction="put the red cube on the plate"))
    checks.note("first reset incl. Isaac start (s)", round(time.monotonic() - t0, 1))
    shapes = {k: (v.shape, str(v.dtype)) for k, v in obs.images.items()}
    checks.check(
        "reset obs: 3 images 224x224x3 uint8",
        shapes == dict.fromkeys(("top_cam", "left_cam", "right_cam"), ((224, 224, 3), "uint8")),
        str(shapes),
    )
    st = obs.state["joint_pos"]
    from yam_arena.constants import READY

    want = np.r_[READY, 1.0, READY, 1.0]
    checks.check(
        "reset state: 14-D, both arms at READY with open grippers",
        st.shape == (14,) and np.abs(st - want).max() < 0.02,
        f"max |state - READY| {np.abs(st - want).max():.4f}",
    )
    checks.check(
        "images not blank",
        all(v.std() > 5 for v in obs.images.values()),
        str({k: round(float(v.std()), 1) for k, v in obs.images.items()}),
    )
    from inspect_robots.types import Action

    target = want.copy()
    target[0] += 0.3  # left j1
    target[13] = 0.0  # right gripper closed
    t0 = time.monotonic()
    for _ in range(10):
        res = emb.step(Action(data=target))
    dt = (time.monotonic() - t0) / 10
    st = res.observation.state["joint_pos"]
    checks.check(
        "step: left j1 +0.3 and right gripper closed show up in state slots 0 and 13",
        abs(st[0] - target[0]) < 0.02 and st[13] < 0.05 and abs(st[6] - 1.0) < 0.02,
        f"state[0]={st[0]:.4f} (cmd {target[0]:.2f}), state[13]={st[13]:.3f}, state[6]={st[6]:.3f}; {dt:.3f} s/step",
    )
    checks.check(
        "observation time advances 0.1 s per step",
        abs(res.observation.state_time - 1.0) < 1e-6,
        f"state_time={res.observation.state_time}",
    )
    base = emb._base
    origin = base.scene.env_origins[0].cpu().numpy()
    default = {n: o.data.default_root_pose.torch[0, :3].cpu().numpy() for n, o in base.scene.rigid_objects.items()}
    emb.reset(Scene(id="seeded", instruction="x"), seed=7)
    moved = {
        n: (o.data.root_link_pose_w.torch[0, :3].cpu().numpy() - origin - default[n])
        for n, o in base.scene.rigid_objects.items()
    }
    xy = {n: np.abs(v[:2]).max() for n, v in moved.items()}
    checks.check(
        "seeded reset jitters cube and plate by <= 3 cm in xy, objects still on the table",
        all(0 < v <= 0.0301 for v in xy.values()) and all(abs(v[2]) < 0.003 for v in moved.values()),
        str({n: np.round(v * 1000, 1).tolist() for n, v in moved.items()}) + " mm",
    )
    emb.reset(Scene(id="unseeded", instruction="x"))
    back = {
        n: float(np.linalg.norm(o.data.root_link_pose_w.torch[0, :3].cpu().numpy() - origin - default[n]))
        for n, o in base.scene.rigid_objects.items()
    }
    checks.check(
        "unseeded reset restores the default object poses",
        max(back.values()) < 0.002,
        str({n: round(v * 1000, 2) for n, v in back.items()}) + " mm",
    )


class HoldPolicy:
    """Non-LLM test policy: hold the observed joints, sweeping left j1 by +-0.1 rad."""

    def __init__(self, action_space):
        from inspect_robots.policy import PolicyConfig, PolicyInfo
        from inspect_robots.spaces import ObservationSpace

        self.info = PolicyInfo(
            name="hold-sweep",
            action_space=action_space,
            observation_space=ObservationSpace(state_keys=frozenset({"joint_pos"})),
        )
        self.config = PolicyConfig(action_horizon=1)
        self._q0 = None
        self._t = 0

    def reset(self, scene):
        self._q0, self._t = None, 0

    def act(self, observation):
        from inspect_robots.types import Action, ActionChunk

        q = np.asarray(observation.state["joint_pos"], dtype=np.float64)
        if self._q0 is None:
            self._q0 = q.copy()
        a = self._q0.copy()
        a[0] += 0.1 * np.sin(self._t * np.pi / 5)
        self._t += 1
        return ActionChunk(actions=[Action(data=a)])


def ir_episode(emb):
    import inspect_robots
    from inspect_robots import Scene
    from inspect_robots.scorer import episode_length
    from inspect_robots.task import Task

    task = Task(
        name="yam-arena-hold",
        scenes=[Scene(id="hold", instruction="hold still", init_seed=0)],
        scorer=episode_length(),
        max_steps=args.steps,
    )
    t0 = time.monotonic()
    logs = inspect_robots.eval(
        task, HoldPolicy(emb.info.action_space), emb, log_dir=str(args.out / "ir-logs"), grader=None, seed=0
    )
    log = logs[0]
    sample = log.samples[0]
    checks.note(
        "eval log",
        {
            "status": log.status,
            "scene_status": sample.status,
            "reduced": dict(sample.reduced),
            "error": sample.error,
            "seconds": round(time.monotonic() - t0, 1),
        },
    )
    checks.check(
        f"inspect_robots.eval: {args.steps}-step episode with a non-LLM policy completes",
        log.status == "success"
        and sample.status == "success"
        and abs(sample.reduced.get("episode_length", -1) - args.steps) < 1e-9,
        f"log={log.status} scene={sample.status} reduced={dict(sample.reduced)} error={sample.error}",
    )


emb = None
try:
    emb = contract()
except Exception as exc:
    checks.error("contract", exc)
if emb is not None:
    for part in (direct, ir_episode):
        try:
            part(emb)
        except Exception as exc:
            checks.error(part.__name__, exc)
checks.finish()
