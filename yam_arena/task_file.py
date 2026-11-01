"""Task files: your own ``yam_arena`` task in one Python file. Pure Python: loading one needs no Isaac Sim.

    inspect-robots run --embodiment yam_arena -E task=my_task.py --instruction "..." ...

A task file is an ordinary Python module that imports nothing from Isaac Sim (the embodiment reads it before
the simulator starts). The embodiment reads these names from it:

- the module docstring: what is on the table. Agents read it after the robot's own docs.
- ``SCENE``: the room and rig around your objects. ``"station"`` (the default) is Dream's calibrated station
  with its table, lights, camera stand and cameras; ``"tabletop"`` is the built-in table.
- ``OBJECTS``: a list of :class:`Box`, :class:`Cylinder`, :class:`Sphere` and :class:`Asset` (a model from
  Arena's object library). An object with a ``mass`` is free: it falls, can be grasped, and seeded resets shift
  it by up to ``-E object_jitter``. One without a mass is fixed where you put it.
- ``success(objects) -> bool`` (optional): called after every control step with an :class:`ObjectState` per
  object. The episode ends with ``termination_reason="success"`` once it has held for ``SUCCESS_HOLD_S``
  seconds in a row (default 1.0), so ``--scorer success_at_end`` scores it.
- ``distance(objects) -> float`` (optional): metres from the goal, reported every step as ``info["distance"]``
  (``--scorer min_distance_to_goal`` scores the closest approach).
- ``score(objects) -> float`` (optional): partial credit from 0 to 1, reported every step as ``info["score"]``.
  ``--scorer task_score`` gives a trial 1.0 for success and otherwise this score at the end of the episode, or
  the best it reached with ``SCORE_OVER = "best"``.
- ``demo() -> dict[str, list[Waypoint]]`` (optional): a scripted plan reaching full score, per arm
  (``"left"``, ``"right"``), which ``--policy yam_scripted`` plays. Build it with
  :func:`yam_arena.demo.pick_place_waypoints`.

World frame: x forward (away from the arms), y left, z up, metres. The arm bases stand at x 0, y +-0.305
(station; +-0.3 on the tabletop), z 0.75. :data:`TABLE_TOP` gives each scene's table height.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SCENES = ("station", "tabletop")
TABLE_TOP = {"station": 0.7475, "tabletop": 0.75}  # station.TABLE_TOP and constants.TABLE_HEIGHT
SUCCESS_HOLD_S = 1.0
SCORE_OVER = ("end", "best")
GREY = (0.7, 0.7, 0.7)
# Scene keys the room, robots and cameras already use.
RESERVED_NAMES = frozenset(
    {
        "table",
        "floor",
        "ground_plane",
        "dome_light",
        "lamp",
        "d435",
        "camera_mast",
        "stand_foot",
        "stand_mast",
        "stand_arm",
        "camera_housing",
        "left_robot",
        "right_robot",
        "top_cam",
        "left_cam",
        "right_cam",
    }
)


def _vec(value, n: int, what: str) -> tuple[float, ...]:
    out = tuple(float(v) for v in value)
    if len(out) != n or not all(math.isfinite(v) for v in out):
        raise ValueError(f"{what} must be {n} finite numbers, got {value!r}")
    return out


@dataclass(frozen=True)
class _Shape:
    name: str
    pos: tuple[float, float, float]  # centre, world frame (m)
    color: tuple[float, float, float] = GREY  # sRGB, 0..1
    mass: float | None = None  # kg; None: fixed in place
    yaw: float = 0.0  # rad about the vertical

    def _check(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"object name {self.name!r} must be a Python identifier")
        _vec(self.pos, 3, f"{self.name}.pos")
        if any(not 0.0 <= c <= 1.0 for c in _vec(self.color, 3, f"{self.name}.color")):
            raise ValueError(f"{self.name}.color must be sRGB in 0..1, got {self.color!r}")
        if self.mass is not None and not self.mass > 0:
            raise ValueError(f"{self.name}.mass must be positive or None, got {self.mass!r}")

    @property
    def free(self) -> bool:
        return self.mass is not None


@dataclass(frozen=True)
class Box(_Shape):
    """A box: ``size`` is its x, y, z edge lengths (m) before ``yaw``."""

    size: tuple[float, float, float] = (0.04, 0.04, 0.04)

    def _check(self) -> None:
        super()._check()
        if any(s <= 0 for s in _vec(self.size, 3, f"{self.name}.size")):
            raise ValueError(f"{self.name}.size must be positive, got {self.size!r}")


@dataclass(frozen=True)
class Cylinder(_Shape):
    """An upright cylinder."""

    radius: float = 0.03
    height: float = 0.05

    def _check(self) -> None:
        super()._check()
        if not (self.radius > 0 and self.height > 0):
            raise ValueError(f"{self.name}: radius and height must be positive")


@dataclass(frozen=True)
class Sphere(_Shape):
    radius: float = 0.03

    def _check(self) -> None:
        super()._check()
        if not self.radius > 0:
            raise ValueError(f"{self.name}.radius must be positive")


@dataclass(frozen=True)
class Asset:
    """A model from Arena's object library (``asset`` is its registry name, e.g. ``"bowl_ycb_robolab"``),
    scaled so its widest horizontal extent is ``extent`` and resting on the table with its bounding-box centre
    over ``xy``. ``color`` (sRGB) replaces its look; ``convex`` collides it as one convex hull."""

    name: str
    asset: str
    xy: tuple[float, float]
    extent: float
    mass: float | None = None
    color: tuple[float, float, float] | None = None
    yaw: float = 0.0
    convex: bool = False

    def _check(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"object name {self.name!r} must be a Python identifier")
        _vec(self.xy, 2, f"{self.name}.xy")
        if not self.extent > 0:
            raise ValueError(f"{self.name}.extent must be positive")
        if self.mass is not None and not self.mass > 0:
            raise ValueError(f"{self.name}.mass must be positive or None, got {self.mass!r}")

    @property
    def free(self) -> bool:
        return self.mass is not None


ObjectSpec = Box | Cylinder | Sphere | Asset


@dataclass(frozen=True)
class ObjectState:
    """One object as the simulator has it after a control step (world frame, m, m/s, rad/s)."""

    pos: np.ndarray  # bounding-box centre
    quat_xyzw: np.ndarray
    lin_vel: np.ndarray  # zero for fixed objects
    ang_vel: np.ndarray
    low: np.ndarray  # world-aligned bounding box
    high: np.ndarray

    def resting(self, max_speed: float = 0.01, max_spin: float = 0.5) -> bool:
        """Still: under ``max_speed`` m/s and ``max_spin`` rad/s."""
        return bool(np.linalg.norm(self.lin_vel) < max_speed and np.linalg.norm(self.ang_vel) < max_spin)


@dataclass(frozen=True)
class TaskFile:
    path: Path
    name: str
    sha256: str
    scene: str
    objects: tuple[ObjectSpec, ...]
    docs: str | None = None
    success: Callable[[dict[str, ObjectState]], bool] | None = None
    distance: Callable[[dict[str, ObjectState]], float] | None = None
    score: Callable[[dict[str, ObjectState]], float] | None = None
    demo: Callable[[], dict[str, Any]] | None = None
    success_hold_s: float = SUCCESS_HOLD_S
    score_over: str = "end"
    module: Any = field(default=None, repr=False, compare=False)

    @property
    def table_top(self) -> float:
        return TABLE_TOP[self.scene]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(o.name for o in self.objects)

    @property
    def free_objects(self) -> tuple[str, ...]:
        return tuple(o.name for o in self.objects if o.free)


def load(path: str | Path) -> TaskFile:
    """Import a task file and check what it declares."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"task file not found: {path}")
    source = path.read_bytes()
    spec = importlib.util.spec_from_file_location(f"yam_task_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import {path} as a Python module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    scene = getattr(module, "SCENE", "station")
    if scene not in SCENES:
        raise ValueError(f"{path.name}: SCENE must be one of {SCENES}, got {scene!r}")
    objects = tuple(getattr(module, "OBJECTS", ()))
    if not objects:
        raise ValueError(f"{path.name}: OBJECTS must list at least one object")
    for obj in objects:
        if not isinstance(obj, Box | Cylinder | Sphere | Asset):
            raise TypeError(f"{path.name}: OBJECTS holds {obj!r}; use Box, Cylinder, Sphere or Asset")
        obj._check()
    names = [o.name for o in objects]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"{path.name}: object names must be unique, repeated: {duplicates}")
    clashes = sorted(set(names) & RESERVED_NAMES)
    if clashes:
        raise ValueError(f"{path.name}: {clashes} already name part of the room or robot; rename them")
    hooks = {}
    if getattr(module, "metrics", None) is not None:
        raise ValueError(f"{path.name}: metrics() is not read; return partial credit from score() instead")
    for hook in ("success", "distance", "score", "demo"):
        fn = getattr(module, hook, None)
        if fn is not None and not callable(fn):
            raise TypeError(f"{path.name}: {hook} must be a function")
        hooks[hook] = fn
    hold = float(getattr(module, "SUCCESS_HOLD_S", SUCCESS_HOLD_S))
    if not hold >= 0:
        raise ValueError(f"{path.name}: SUCCESS_HOLD_S must be >= 0")
    score_over = getattr(module, "SCORE_OVER", "end")
    if score_over not in SCORE_OVER:
        raise ValueError(f"{path.name}: SCORE_OVER must be one of {SCORE_OVER}, got {score_over!r}")
    docs = (module.__doc__ or "").strip() or None
    return TaskFile(
        path=path,
        name=path.stem,
        sha256=hashlib.sha256(source).hexdigest(),
        scene=scene,
        objects=objects,
        docs=docs,
        success_hold_s=hold,
        score_over=score_over,
        module=module,
        **hooks,
    )


# The task file each yam_arena embodiment loaded, by its environment_id, so the yam_scripted policy (which
# Inspect Robots binds to that embodiment's info) can play the same file's demo.
_LOADED: dict[str, TaskFile] = {}


def register(environment_id: str, task: TaskFile) -> None:
    _LOADED[environment_id] = task


def lookup(environment_id: str | None) -> TaskFile | None:
    return _LOADED.get(environment_id or "")
