"""On the station table: a red 4 cm cube in front of the left arm, and a blue 6 x 6 x 4 cm block between
the arms, a little further out. Both are loose.

Run it:  inspect-robots run --embodiment yam_arena -E task=examples/stack_cubes.py \\
           --instruction "put the red cube on the blue block" --scorer task_score ...
"""

import numpy as np

from yam_arena.demo import pick_place_waypoints
from yam_arena.task_file import TABLE_TOP, Box

SCENE = "station"
TOP = TABLE_TOP[SCENE]

RED = Box("red_cube", pos=(0.17, 0.12, TOP + 0.02), size=(0.04, 0.04, 0.04), mass=0.05, color=(0.85, 0.12, 0.12))
BLUE = Box("blue_block", pos=(0.25, -0.05, TOP + 0.02), size=(0.06, 0.06, 0.04), mass=0.2, color=(0.15, 0.30, 0.85))
OBJECTS = [RED, BLUE]


def _offset(objects):
    """The red cube's centre relative to where it sits when stacked squarely on the blue block (m)."""
    red, blue = objects["red_cube"], objects["blue_block"]
    return red.pos - blue.pos - np.array([0.0, 0.0, (BLUE.size[2] + RED.size[2]) / 2])


def success(objects):
    """Stacked: within 2.5 cm of square on the block, resting on it, and both cubes still."""
    off = _offset(objects)
    return bool(
        np.hypot(off[0], off[1]) < 0.025
        and abs(off[2]) < 0.01
        and objects["red_cube"].resting()
        and objects["blue_block"].resting()
    )


def distance(objects):
    return float(np.linalg.norm(_offset(objects)))


def score(objects):
    """Partial credit (success scores 1.0): 0.5 if the red cube ends up on the block but not squarely on it."""
    return 0.5 if objects["red_cube"].low[2] > TOP + BLUE.size[2] - 0.01 else 0.0


def demo():
    """Left arm: grasp the red cube at its centre, set it down on the block, let go. It travels 14 cm above the
    table: higher up, a straight-down grip over the block would need joint4 past its limit."""
    place = (BLUE.pos[0], BLUE.pos[1], TOP + BLUE.size[2] + RED.size[2] / 2 + 0.005)
    return {"left": pick_place_waypoints(RED.pos, place, travel_z=TOP + 0.14)}
