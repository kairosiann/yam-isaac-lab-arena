"""Dream's calibrated YAM station: rig, look, table, objects and tasks. Pure Python.

Numbers come from Dream Labs' MuJoCo replica of its station (dreamlabs 1.0.23,
``dream_examples/yam_mujoco``: ``yam_env.py``, ``assets/sim-eval-yam.json``, ``assets/model.json``),
whose cameras and arm mounts are the calibration of ARISE-Initiative/yamlab@ec0455d2
(``configs/robot/yam.yaml``).

World frame: the tabletop scene's. Our arm bases sit at (0, +-0.305, 0.75), facing +x, left arm at
+y, z up. :func:`from_dream` and :func:`from_calibration` move Dream's example scene and the
calibration into it. Cameras use the OpenGL convention (look along -z, +y up), as the calibration
does. Colours are sRGB in 0..1, as Dream renders them; :func:`linear_rgb` converts for USD materials.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from yam_arena import ASSET_DIR
from yam_arena.ik import matrix_to_quat_xyzw
from yam_arena.rig import Camera, Lens, Rig
from yam_arena.urdf import axis_angle_matrix, quat_xyzw_to_matrix

DREAM_YAW = math.pi / 2  # Dream's example frame turned into ours
BASE_Z = 0.75
# Our URDF's chain is Dream's, fitted to 3 um over random joint configurations: our `base` frame sits 9.5 mm
# below Dream's base_link, and Dream's link_6 is our `gripper` link turned 180 deg about z, its origin 0.3 mm
# back along the approach axis.
BASE_BELOW_BASE_LINK = 0.0095
DREAM_BASE_LINKS = {"left": (0.305, 0.24, 0.772), "right": (-0.305, 0.24, 0.772)}  # Dream's example frame
DREAM_DZ = BASE_Z + BASE_BELOW_BASE_LINK - DREAM_BASE_LINKS["left"][2]


def from_dream(p: tuple[float, ...]) -> tuple[float, ...]:
    """A position (x, y[, z]) in Dream's example frame, in ours: (0.24 - y, x[, z - 0.0125])."""
    return (0.24 - p[1], p[0], *(z + DREAM_DZ for z in p[2:]))


def from_calibration(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """A world-fixed position in the calibration's frame (bases at x 0.2525), where Dream's example puts it."""
    return (p[0] - 0.2525, p[1], p[2] + DREAM_DZ)


def _rz(angle: float) -> np.ndarray:
    return axis_angle_matrix(np.array([0.0, 0.0, 1.0]), angle)


def _xyzw(wxyz: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = np.asarray(wxyz, dtype=float) / np.linalg.norm(wxyz)
    return np.array([x, y, z, w])


def _floats(v) -> tuple[float, ...]:
    return tuple(float(x) for x in v)


def aim(direction) -> tuple[float, float, float, float]:
    """The rotation (XYZW) that turns +z onto ``direction``."""
    z = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
    axis = np.cross([0.0, 0.0, 1.0], z)
    s = np.linalg.norm(axis)
    if s < 1e-12:
        return (0.0, 0.0, 0.0, 1.0) if z[2] > 0 else (1.0, 0.0, 0.0, 0.0)
    return _floats(matrix_to_quat_xyzw(axis_angle_matrix(axis / s, math.atan2(s, z[2]))))


BASES = {
    side: _floats(np.array(from_dream(p)) - (0.0, 0.0, BASE_BELOW_BASE_LINK)) for side, p in DREAM_BASE_LINKS.items()
}
LINK6_IN_GRIPPER_ROT = _rz(math.pi)
LINK6_IN_GRIPPER_POS = np.array([0.0, 0.0, -0.000301])

# -- look --------------------------------------------------------------------------------

TABLE_TOP = 0.76 + DREAM_DZ
TABLE_SIZE = (1.10, 1.30, 0.06)
TABLE_CENTER = from_dream((0.0, -0.12, 0.73))
FLOOR_Z = DREAM_DZ
SKY = (0.86, 0.90, 0.915)  # middle of Dream's gradient skybox; its oak.png table and floor are matched in scene.py
LAMP_POS = from_dream((-0.8, 0.2, 2.7))  # Dream's shadow-casting lamp, high on the right of the arms
# Dream renders the arms black with white covers on link2 and link3, whose I2RT meshes it splits into black
# and white parts (assets/yam/dream/, MIT like the other meshes). The parts are in Dream's link frames, which
# are ours with the axes permuted and shifted along our y (fitted like the wrist mounts).
ARM_BLACK = (0.065, 0.07, 0.08)
ARM_WHITE = (0.94, 0.945, 0.95)
ARM_PARTS_DIR = ASSET_DIR / "dream"
ARM_PART_ROT = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
ARM_PART_POS = {"link2": np.array([0.0, -0.033875, 0.0]), "link3": np.array([0.0, -0.034925, 0.0])}
ARM_PARTS = {
    link: {color: ARM_PARTS_DIR / f"{link}_{color}.stl" for color in ("black", "white")} for link in ARM_PART_POS
}

# -- cameras -----------------------------------------------------------------------------

CROP_TOP = 60  # Dream and yamlab render 640x480 and keep rows 60:420 (640x360)
TOP_CAM_POS = (0.08600512146949768, -0.008999999612569809, 1.7043205499649048)  # calibration frame
TOP_CAM_WXYZ = (0.68301, 0.18301, -0.18301, -0.68301)
TOP_CAM_K = (392.195617675781, 391.722351074219, 318.389434814453, 237.876312255859)  # fx fy cx cy at 640x480
WRIST_CAM_POS = {"left": (-0.0004, 0.069638, 0.073063), "right": (0.0, 0.069638, 0.072)}  # in link_6
WRIST_CAM_WXYZ = (-0.003227, 0.002817, 0.975619, 0.219430)
WRIST_CAM_K = (390.666, 390.162, 317.526, 236.146)
# Dream's demo view (yam_env.py TOP_CAMERA_LOWER_M, TOP_CAMERA_TILT_UP_DEG): the top camera 15 cm lower and
# pitched 5 deg up about its own x axis, for a closer view of the table.
DEMO_LOWER = 0.15
DEMO_TILT_UP_DEG = 5.0
TOP_CAMS = ("calibrated", "demo")


def lens(k: tuple[float, float, float, float]) -> Lens:
    """A 640x480 calibration's intrinsics, after the 640x360 crop."""
    fx, fy, cx, cy = k
    return Lens(fx, fy, cx, cy - CROP_TOP, 640, 360)


def top_camera(variant: str = "calibrated") -> Camera:
    if variant not in TOP_CAMS:
        raise ValueError(f"top_cam must be one of {TOP_CAMS}, got {variant!r}")
    pos = np.array(from_calibration(TOP_CAM_POS))
    rot = quat_xyzw_to_matrix(_xyzw(TOP_CAM_WXYZ))
    if variant == "demo":
        pos[2] -= DEMO_LOWER
        rot = rot @ axis_angle_matrix(np.array([1.0, 0.0, 0.0]), math.radians(DEMO_TILT_UP_DEG))
    return Camera(_floats(pos), _floats(matrix_to_quat_xyzw(rot)), "opengl", lens(TOP_CAM_K))


def wrist_camera(side: str) -> Camera:
    """The wrist camera in our `gripper` link frame (the calibration gives it in Dream's link_6)."""
    rot = LINK6_IN_GRIPPER_ROT @ quat_xyzw_to_matrix(_xyzw(WRIST_CAM_WXYZ))
    pos = LINK6_IN_GRIPPER_ROT @ np.array(WRIST_CAM_POS[side]) + LINK6_IN_GRIPPER_POS
    return Camera(_floats(pos), _floats(matrix_to_quat_xyzw(rot)), "opengl", lens(WRIST_CAM_K))


def rig(top_cam: str = "calibrated") -> Rig:
    return Rig(
        name=f"station-{top_cam}",
        bases=BASES,
        top_cam=top_camera(top_cam),
        wrist_cams={side: wrist_camera(side) for side in ("left", "right")},
        dream_look=True,
    )


def _segment(a, b) -> tuple[tuple[float, ...], tuple[float, ...], float]:
    """(centre, rot_xyzw, length) of a capsule spine from a to b, the spine along the part's z."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return _floats((a + b) / 2), aim(b - a), float(np.linalg.norm(b - a))


def camera_stand(top: Camera) -> dict[str, tuple]:
    """Dream's mock stand for the top camera: a foot on the table, a mast behind the camera, a short arm,
    and a housing box behind the optical centre. It follows the top camera, so it moves with ``top_cam``.

    Part name -> ("box", edge lengths, pos, rot_xyzw) or ("capsule", (radius, spine length), pos, rot_xyzw).
    """
    x, y, z = top.pos
    mast = _segment((x - 0.07, y, TABLE_TOP + 0.016), (x - 0.07, y, z))
    arm = _segment((x - 0.07, y, z), (x - 0.02, y, z + 0.02))
    housing = np.array(top.pos) + quat_xyzw_to_matrix(np.array(top.rot_xyzw)) @ (0.0, 0.0, 0.014)
    return {
        "stand_foot": ("box", (0.11, 0.11, 0.016), (x - 0.07, y, TABLE_TOP + 0.008), (0.0, 0.0, 0.0, 1.0)),
        "stand_mast": ("capsule", (0.012, mast[2]), mast[0], mast[1]),
        "stand_arm": ("capsule", (0.009, arm[2]), arm[0], arm[1]),
        "camera_housing": ("box", (0.05, 0.042, 0.022), _floats(housing), top.rot_xyzw),
    }


STAND = (0.09, 0.095, 0.105)
HOUSING = (0.78, 0.80, 0.82)

# -- objects -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Item:
    """One of Dream's objects as an Arena object-library asset, scaled so its widest horizontal extent is
    ``extent`` and resting on the table with its bounding-box centre over ``xy`` (Dream's pose)."""

    asset: str
    xy: tuple[float, float]
    extent: float
    mass: float | None = None  # None: a fixture, static as in Dream's scene
    color: tuple[float, float, float] | None = None  # replaces the asset's look
    convex: bool = False  # collide as one convex hull, as Dream models it, instead of the asset's decomposition
    yaw: float | None = None  # turn about the vertical (rad); None: Dream's (DREAM_YAW)


# Dream's colours: the mean of its apple scan's texture, its orange's flat colour, and the means of its near-white
# brown_glaze.png and blue_glaze.png.
APPLE_RED = (0.629, 0.104, 0.138)
ORANGE = (0.92, 0.33, 0.025)
BOWL_GLAZE = tuple(c / 255 for c in (219.5, 214.5, 203.5))
PLATE_GLAZE = tuple(c / 255 for c in (224.5, 223.5, 214.5))
ITEMS = {
    # Dream collides the apple as its scan's convex hull.
    "apple": Item("apple_01_objaverse_robolab", from_dream((0.15, 0.105)), 0.066, 0.12, APPLE_RED, convex=True),
    "orange": Item("orange_02_fruits_veggies_robolab", from_dream((0.0, 0.13)), 0.064, 0.11, ORANGE),
    # The same YCB banana scan Dream uses, at Dream's scale (140 mm long).
    "banana": Item("banana_ycb_robolab", from_dream((-0.115, 0.085)), 0.140, 0.045),
    "bowl": Item("bowl_ycb_robolab", from_dream((0.0, -0.04)), 0.166, color=BOWL_GLAZE),
    "plate": Item("plate_small_vomp_robolab", from_dream((-0.22, -0.045)), 0.146, color=PLATE_GLAZE),
}
# Dream's water bottle, a distractor: a 58 mm cylinder body that collides, and its shoulder, cap, label band
# and ribs as visual trim, each (shape, (radius, half height), height above the body's centre, colour).
BOTTLE_XY = from_dream((0.22, -0.11))
BOTTLE_RADIUS = 0.029
BOTTLE_HEIGHT = 0.15
BOTTLE_MASS = 0.276
BOTTLE_COLOR = (0.36, 0.58, 0.65)
BOTTLE_TRIM = {
    "shoulder": ("ellipsoid", (0.029, 0.020), 0.074, BOTTLE_COLOR),
    "cap": ("cylinder", (0.013, 0.012), 0.097, (0.12, 0.24, 0.30)),
    "band": ("cylinder", (0.0295, 0.025), -0.005, (0.90, 0.95, 0.97)),
    **{
        f"rib{i}": ("cylinder", (0.0298, 0.0012), z, (0.43, 0.64, 0.70))
        for i, z in enumerate((-0.063, -0.053, 0.043, 0.053))
    },
    **{f"label{i}": ("cylinder", (0.0298, 0.0012), z, (0.17, 0.35, 0.40)) for i, z in enumerate((-0.028, 0.018))},
}
FREE_OBJECTS = (*(name for name, item in ITEMS.items() if item.mass is not None), "bottle")
REST_GAP = 0.001  # objects spawn this far above the table and settle onto it


def placement(item: Item, lo, hi) -> tuple[float, tuple[float, ...], tuple[float, ...]]:
    """(uniform scale, position, rot_xyzw) of an asset whose unscaled local bounds are lo..hi (z up)."""
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    scale = item.extent / float(np.max(hi[:2] - lo[:2]))
    rot = _rz(DREAM_YAW if item.yaw is None else item.yaw)
    xy = np.asarray(item.xy) - (rot @ ((lo + hi) / 2 * scale))[:2]
    return (
        scale,
        (float(xy[0]), float(xy[1]), TABLE_TOP + REST_GAP - float(lo[2]) * scale),
        _floats(matrix_to_quat_xyzw(rot)),
    )


def linear_rgb(srgb: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb)


# -- tasks -------------------------------------------------------------------------------

FRUITS = ("apple", "orange", "banana")
RELATIONS = {"bowl": "in", "plate": "on"}
TARGET_KIND = "put"  # Scene.target = Target(kind="put", spec={"object": "apple", "target": "bowl"})


@dataclass(frozen=True)
class Goal:
    object: str
    target: str

    def __post_init__(self):
        if self.object not in FRUITS or self.target not in RELATIONS:
            raise ValueError(
                f"no goal 'put {self.object} -> {self.target}': objects {FRUITS}, targets {tuple(RELATIONS)}"
            )

    @property
    def relation(self) -> str:
        return RELATIONS[self.target]

    @property
    def instruction(self) -> str:
        return f"put {self.object} {'into' if self.relation == 'in' else 'on'} the {self.target}"


DREAM_TASKS = (Goal("apple", "bowl"), Goal("orange", "bowl"), Goal("banana", "plate"))

# yam_scripted's plans (heights above the table top, m): where the TCP grips each fruit (its bounding-box centre)
# and where each target's top is, measured on the settled scene; and which arm takes each fruit.
GRIP_HEIGHT = {"apple": 0.032, "orange": 0.031, "banana": 0.018}
TARGET_TOP = {"bowl": 0.055, "plate": 0.02}
DEMO_ARM = {"apple": "left", "orange": "left", "banana": "right"}

_PUT = re.compile(
    rf"(?:put|place|move|drop) (?:the |an? )?({'|'.join(FRUITS)}) "
    rf"(?:in|into|inside|in to|on|onto|on to|on top of) (?:the |a )?({'|'.join(RELATIONS)})"
)


def goal_for(instruction: str | None, target=None) -> Goal | None:
    """The goal a scene asks for: its ``put`` target, else its instruction (e.g. Dream's "put apple into
    the bowl"); None when it asks for neither, and the episode then runs without success detection."""
    if target is not None:
        if target.kind != TARGET_KIND:
            raise ValueError(f"the station realizes Target(kind={TARGET_KIND!r}), got {target.kind!r}")
        return Goal(target.spec["object"], target.spec["target"])
    words = " ".join(re.sub(r"[^\w\s]", " ", (instruction or "").lower()).split())
    match = _PUT.fullmatch(words)
    return Goal(*match.groups()) if match else None
