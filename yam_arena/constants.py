"""YAM robot, rig and scene constants. Pure Python: safe to import before Isaac Sim starts.

Values come from YAM-inspect-Isaac. Every quaternion here is **XYZW** (Isaac Lab 3.0).
"""

from __future__ import annotations

import math

# -- robot -------------------------------------------------------------------------------

ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]
FINGER_JOINTS = ["joint7", "joint8"]
ALL_JOINTS = ARM_JOINTS + FINGER_JOINTS
LINKS = ["base", "link1", "link2", "link3", "link4", "link5", "gripper", "tip_left", "tip_right"]

# Finger joints run from 0 (closed, pads touching) to -0.04695 m (open, pads 94 mm apart).
FINGER_OPEN = -0.04695
FINGER_CLOSED = 0.0

# Raised "ready" pose: shoulder up, elbow bent, wrist pitched down toward the table.
READY = [0.0, 1.0, 1.0, -0.6, 0.0, 0.0]

# Tool centre point in the `gripper` link frame, between the finger pads, 13 mm short of the
# pad tips. The gripper's approach axis is +Z of that frame.
TCP_OFFSET = (0.0, 0.0, 0.134)
PAD_TIP_PAST_TCP = 0.013

# Gripper transmission (i2rt linear_4310.yml): 6.57 rad of motor travel moves the fingers
# 0.096 m apart in total, so each finger travels 0.0073 m per motor radian.
FINGER_M_PER_RAD = 0.096 / 2 / 6.57

# Actuator gains and limits from i2rt's yam_v1.yml / linear_4310.yml.
SHOULDER = dict(effort=27.0, velocity=5.5, stiffness=80.0, damping=5.0, friction=0.3)  # DM4340, joints 1-3
WRIST = dict(effort=7.0, velocity=20.0, stiffness=10.0, damping=1.5, friction=0.06)  # DM4310, joints 4-6
GRIPPER = dict(effort=50.0, velocity=20.0 * FINGER_M_PER_RAD, stiffness=5000.0, damping=100.0)

# -- physics and control -----------------------------------------------------------------

PHYSICS_DT = 1 / 240  # 240 Hz keeps finger/object contacts stable
CONTROL_HZ = 10.0  # the real rig's control rate
DECIMATION = round(1.0 / (CONTROL_HZ * PHYSICS_DT))  # 24 physics steps per action
RENDER_INTERVAL = DECIMATION  # render once per control step, after the last physics substep

# -- scene (built-in tabletop) -----------------------------------------------------------

TABLE_HEIGHT = 0.75
TABLE_SIZE = (0.8, 1.2, 0.04)
TABLE_CENTER_X = 0.35
# Arms sit 0.6 m apart on the near table edge, both reaching along +X. Left arm at +Y.
ARM_SPACING = 0.6
LEFT_BASE_POS = (0.0, ARM_SPACING / 2, TABLE_HEIGHT)
RIGHT_BASE_POS = (0.0, -ARM_SPACING / 2, TABLE_HEIGHT)
CUBE_SIZE = 0.04
CUBE_MASS = 0.05
CUBE_POS = (0.35, 0.12, TABLE_HEIGHT + CUBE_SIZE / 2)
PLATE_RADIUS = 0.09
PLATE_HEIGHT = 0.015
PLATE_MASS = 0.2
PLATE_POS = (0.38, -0.12, TABLE_HEIGHT + PLATE_HEIGHT / 2)
GRIPPY_FRICTION = (1.2, 1.0)  # static, dynamic: rubber-like, for the graspable objects
DOME_LIGHT_INTENSITY = 2500.0

# -- cameras -----------------------------------------------------------------------------

# Rendered resolution; observations are resized to the inspect-robots contract (224x224 by
# default), like the real rig's cv2.resize.
CAM_RENDER_HEIGHT, CAM_RENDER_WIDTH = 360, 640
# Focal lengths reproduce each sensor's colour HFOV on the default 20.955 mm aperture
# (D435 69 deg, D405 87 deg).
D435_FOCAL = 15.24
D405_FOCAL = 11.04
# Top camera (D435): between the arms, 0.19 m behind and 0.43 m above their bases, pitched
# 45 deg down; "world" convention (forward +X, up +Z).
TOP_CAM_POS = (-0.19, 0.0, TABLE_HEIGHT + 0.43)
TOP_CAM_ROT_XYZW = (0.0, 0.3827, 0.0, 0.9239)
# Wrist camera (D405) in the `gripper` link frame, ROS optical convention (+Z forward, +Y down):
# 8 cm off the link's -Y face, looking along the approach axis, tilted 8 deg toward the fingers.
WRIST_CAM_POS = (0.0, -0.08, 0.0)
WRIST_CAM_ROT_XYZW = (-0.0698, 0.0, 0.0, 0.9976)
# Overview (debug) camera: behind and above the arms, pitched 30 deg down, "world" convention.
OVERVIEW_CAM_POS = (-1.0, 0.0, 1.65)
OVERVIEW_CAM_ROT_XYZW = (0.0, 0.2588, 0.0, 0.9659)

# Static collider for the top camera body. A RealSense D435 is 90 x 25 x 25 mm (W x H x D).
# In the camera's "world"-convention frame: X = optical axis (depth), Y = image left (width),
# Z = image up (height). The lens sits on the front face, so the box centre is half a depth
# behind the optical centre.
D435_SIZE_XYZ = (0.025, 0.090, 0.025)


def _quat_rotate_xyzw(q: tuple[float, float, float, float], v: tuple[float, float, float]) -> tuple[float, ...]:
    x, y, z, w = q
    # v' = v + 2w (u x v) + 2 u x (u x v), u = (x, y, z)
    ux = (y * v[2] - z * v[1], z * v[0] - x * v[2], x * v[1] - y * v[0])
    uux = (y * ux[2] - z * ux[1], z * ux[0] - x * ux[2], x * ux[1] - y * ux[0])
    return tuple(v[i] + 2 * w * ux[i] + 2 * uux[i] for i in range(3))


TOP_CAM_FORWARD = _quat_rotate_xyzw(TOP_CAM_ROT_XYZW, (1.0, 0.0, 0.0))
D435_BOX_POS = tuple(TOP_CAM_POS[i] - 0.5 * D435_SIZE_XYZ[0] * TOP_CAM_FORWARD[i] for i in range(3))
D435_BOX_ROT_XYZW = TOP_CAM_ROT_XYZW

# Optional camera mast (off by default): the real mount is not modelled, so its shape is a guess.
# A 20 mm square post from the floor to just under the camera body, behind it.
CAMERA_MAST_SIZE = (0.02, 0.02, D435_BOX_POS[2] - 0.0125 - 0.005)
CAMERA_MAST_POS = (D435_BOX_POS[0] - 0.03, 0.0, CAMERA_MAST_SIZE[2] / 2)

# Gripper pointing straight down, fingers closing along world Y (gripper X -> world Y,
# gripper Z -> world -Z). Used by the scripted IK demo.
DOWN_XYZW = (0.70711, 0.70711, 0.0, 0.0)


def deg(rad: float) -> float:
    return math.degrees(rad)
