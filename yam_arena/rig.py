"""Where the arms and cameras sit: the built-in tabletop rig. Pure Python.

A :class:`Rig` is everything the embodiment needs to place and dress its hardware: the two arm bases
(world frame), three cameras (the top camera fixed in the world, one on each wrist in the `gripper`
link frame) and the arms' look. Dream's calibrated station is the other rig (``yam_arena.station.rig``).
"""

from __future__ import annotations

from dataclasses import dataclass

from yam_arena.constants import (
    CAM_RENDER_HEIGHT,
    CAM_RENDER_WIDTH,
    D405_FOCAL,
    D435_FOCAL,
    LEFT_BASE_POS,
    RIGHT_BASE_POS,
    TOP_CAM_POS,
    TOP_CAM_ROT_XYZW,
    WRIST_CAM_POS,
    WRIST_CAM_ROT_XYZW,
)

APERTURE_MM = 20.955  # Isaac's default horizontal aperture, which the focal lengths are given on


@dataclass(frozen=True)
class Lens:
    """Pinhole intrinsics in pixels of a ``width`` x ``height`` image."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_focal(cls, focal_mm: float, width: int = CAM_RENDER_WIDTH, height: int = CAM_RENDER_HEIGHT) -> Lens:
        f = focal_mm * width / APERTURE_MM
        return cls(f, f, width / 2, height / 2, width, height)

    def resized(self, width: int, height: int) -> Lens:
        sx, sy = width / self.width, height / self.height
        return Lens(self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy, width, height)

    @property
    def focal_mm(self) -> float:
        return self.fx * APERTURE_MM / self.width

    @property
    def is_ideal(self) -> bool:
        """Square pixels and a centred principal point: a plain USD pinhole renders it exactly."""
        return self.fx == self.fy and self.cx == self.width / 2 and self.cy == self.height / 2


@dataclass(frozen=True)
class Camera:
    """A camera's pose in its parent frame (Isaac Lab ``convention``: world, ros or opengl) and lens."""

    pos: tuple[float, float, float]
    rot_xyzw: tuple[float, float, float, float]
    convention: str
    lens: Lens


@dataclass(frozen=True)
class Rig:
    name: str
    bases: dict[str, tuple[float, float, float]]
    top_cam: Camera
    wrist_cams: dict[str, Camera]
    dream_look: bool = False  # arms coloured as Dream renders them (station.ARM_*) instead of the URDF's colours


TABLETOP = Rig(
    name="tabletop",
    bases={"left": LEFT_BASE_POS, "right": RIGHT_BASE_POS},
    top_cam=Camera(TOP_CAM_POS, TOP_CAM_ROT_XYZW, "world", Lens.from_focal(D435_FOCAL)),
    wrist_cams=dict.fromkeys(
        ("left", "right"), Camera(WRIST_CAM_POS, WRIST_CAM_ROT_XYZW, "ros", Lens.from_focal(D405_FOCAL))
    ),
)
