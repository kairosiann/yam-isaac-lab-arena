# yam-arena

Two [I2RT YAM](https://i2rt.com/products/yam-6-dof-arm) arms in
[Isaac Lab-Arena](https://github.com/isaac-sim/IsaacLab-Arena) 0.3.1 (Isaac Lab 3.0, Isaac Sim 6.1),
and the [Inspect Robots](https://github.com/robocurve/inspect-robots) embodiment `yam_arena` with the
same action layout, state and cameras as the real rig's `yam_arms`. Use it to evaluate Dream Labs'
Dream-1 ([`inspect-robots-dream`](https://github.com/kairosiann/inspect-robots-dream)) or an LLM agent
on the YAM in simulation. It is a port of
[YAM-inspect-Isaac](https://github.com/ttpierson/YAM-inspect-Isaac).
Not affiliated with I2RT, NVIDIA or Dream Labs.

## Quickstart

You need a Linux machine with an NVIDIA RTX GPU (an L40S, for example) and its driver. Isaac Sim
renders with RT cores, so an A100 or H100 will not do.

Install Arena, this package and the policies:

```bash
git clone https://github.com/kairosiann/yam-isaac-lab-arena && cd yam-isaac-lab-arena
./setup.sh
source ~/IsaacLab-Arena/yam.env
```

`setup.sh` clones Arena into `ARENA_DIR` (default `~/IsaacLab-Arena`), installs everything into its
venv, and is safe to re-run. Start every new shell with `source <ARENA_DIR>/yam.env`.

Save your key from [platform.dreamlabs.ai](https://platform.dreamlabs.ai/api-keys):

```bash
mkdir -p ~/.dreamlabs && echo 'DREAM_API_KEY=dl_live_...' > ~/.dreamlabs/credentials
chmod 600 ~/.dreamlabs/credentials
```

Run Dream-1 in the `station` scene, a replica of Dream's calibrated YAM station that scores the task
automatically. The other `-E` options give Dream the control rate and frame size it expects:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E scene=station -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "put apple into the bowl" --scorer success_at_end \
  --max-steps 2400 --store-frames --log-dir ~/yam-eval/dream
```

The Dream client installs in Arena's environment, so the policy runs it in-process, without
`inspect-robots-dream`'s bridge.

Or run a model through OpenRouter:

```bash
export OPENROUTER_API_KEY=sk-or-...
inspect-robots run --embodiment yam_arena --policy agent -P model=anthropic/claude-haiku-5.5 \
  -E scene=station --instruction "put apple into the bowl" --scorer success_at_end \
  --max-steps 50 --store-frames --log-dir ~/yam-eval/agent
```

Or play the scripted demo, which reaches full score. It is open loop, so `-E object_jitter=0` keeps every
object where the scene puts it:

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E scene=station -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "put apple into the bowl" --scorer success_at_end \
  --max-steps 600 --store-frames --log-dir ~/yam-eval/scripted
```

Render each camera of each run to MP4 (one file per camera, in the log's `frames/` folder;
`inspect-robots video` takes one log at a time):

```bash
for log in ~/yam-eval/dream/*.json; do inspect-robots video "$log"; done
```

Serve the trial reports from the VM:

```bash
inspect-robots view ~/yam-eval/dream --serve
```

Forward port 8300 to your laptop (`brev port-forward <instance> -p 8300:8300` on Brev, or
`ssh -L 8300:127.0.0.1:8300 <vm>`) and open http://127.0.0.1:8300. Add `--force` to re-render
pages written while a run was live. `inspect-robots view <log>.json -o report.html` renders one
log to a file.

## Embodiment

| | |
|---|---|
| Action | 14 absolute joint targets, `[left j0..j5, left gripper, right j0..j5, right gripper]`; joints in rad, clipped to the URDF limits; grippers 0 closed to 1 open |
| State | `joint_pos`, the same 14 values as measured |
| Cameras | `top_cam` (the overhead D435), `left_cam` and `right_cam` (a D405 on each wrist); RGB, `cam_height` x `cam_width` |
| Control | each step holds the target for one control period: joint PD with i2rt's gains plus gravity compensation, at 240 Hz PhysX |
| Resets | both arms to the ready pose; a seeded reset shifts each loose object by up to `object_jitter` (3 cm) |

Policies written for `yam_arms` are compatible with `yam_arena` as they are. The docs it hands an
agent describe each joint's direction, checked against the URDF in `tests/test_urdf.py`.

| `-E` option | Default | Meaning |
|---|---|---|
| `scene` | `tabletop` | `tabletop` or `station` (see [Scenes](#scenes)); a task file sets its own |
| `task` | none | A task file (see [Your own task](#your-own-task)) |
| `object_jitter` | `0.03` | How far (m) a seeded reset shifts each loose object in x and y; `0` keeps the layout fixed |
| `top_cam` | `calibrated` | Station only: the calibrated top camera, or `demo`, Dream's example view (0.15 m lower, pitched 5 degrees up) |
| `control_hz` | `10` | Control rate; physics stays at 240 Hz |
| `cam_height`, `cam_width` | `224` | Observed image size |
| `wrist_cam_mount` | `link` | `link` parents the wrist cameras to the gripper; `follow` moves them there every physics step |
| `device` | `cuda:0` | Simulation device |
| `headless` | `true` | `false` asks for a Kit window (untested) |

From Arena directly, the embodiment is `yam_bimanual`, with scene keys `left_robot` and `right_robot`:

```python
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True, enable_cameras=True).app

from yam_arena.embodiment import BimanualYamEmbodiment
from yam_arena.scene import make_env

env = make_env(BimanualYamEmbodiment(enable_cameras=True))
```

## Scenes

| `-E scene=` | What |
|---|---|
| `tabletop` | A plain table: arms 0.6 m apart, a red 4 cm cube, a white plate, the D435 0.43 m above the arms |
| `station` | Dream Labs' calibrated YAM station, from Dream's MuJoCo replica (dreamlabs 1.0.23, `yam_mujoco`) and the calibration it follows ([yamlab](https://github.com/ARISE-Initiative/yamlab) at `ec0455d`): arms 0.61 m apart in Dream's colours, an oak-coloured table, apple, orange, banana, water bottle, fixed bowl and plate, the top camera 0.94 m above the table on a stand, and the wrist cameras on Dream's mount |

The station's cameras render the calibrated intrinsics (640x480 cropped to rows 60:420, as the station
does), so a policy trained on Dream's station sees the same views. `scripts/station_check.py` compares
them with renders from Dream's replica.

On the station, Arena's success checks score Dream's three tasks. The instruction picks the task:
`put apple into the bowl`, `put orange into the bowl` or
`put banana on the plate` (a scene's `Target(kind="put", spec={"object": ..., "target": ...})` takes
precedence). Every step reports `info["distance"]` (object to goal, in metres) and `info["success"]`,
and the episode ends with `termination_reason="success"` once the goal has held for a second, so
`success_at_end` and `min_distance_to_goal` score it. Other instructions run without success checks.

## Your own task

A task file is one Python file that puts your objects in the station (or on the tabletop) and scores them.
It imports nothing from Isaac Sim. [`examples/stack_cubes.py`](examples/stack_cubes.py) is a complete one;
its outline:

```python
"""On the station table: a red 4 cm cube in front of the left arm, and a blue 6 x 6 x 4 cm block..."""

from yam_arena.demo import pick_place_waypoints
from yam_arena.task_file import TABLE_TOP, Box

SCENE = "station"
TOP = TABLE_TOP[SCENE]
RED = Box("red_cube", pos=(0.17, 0.12, TOP + 0.02), size=(0.04, 0.04, 0.04), mass=0.05, color=(0.85, 0.12, 0.12))
BLUE = Box("blue_block", pos=(0.25, -0.05, TOP + 0.02), size=(0.06, 0.06, 0.04), mass=0.2, color=(0.15, 0.30, 0.85))
OBJECTS = [RED, BLUE]


def success(objects):
    """Stacked squarely and still; must hold for 1 s."""


def distance(objects):
    """Metres from the red cube to its stacked position."""


def score(objects):
    """Partial credit: 0.5 if the red cube ends up on the block but not squarely on it."""


def demo():
    place = (BLUE.pos[0], BLUE.pos[1], TOP + BLUE.size[2] + RED.size[2] / 2 + 0.005)
    return {"left": pick_place_waypoints(RED.pos, place, travel_z=TOP + 0.14)}
```

Run Dream-1 on it, and play its demo:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=examples/stack_cubes.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "put the red cube on the blue block" --scorer task_score \
  --max-steps 2400 --store-frames --log-dir ~/yam-eval/stack-dream
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=examples/stack_cubes.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "put the red cube on the blue block" --scorer task_score \
  --max-steps 600 --store-frames --log-dir ~/yam-eval/stack-demo
```

What the embodiment reads from the file:

| Name | Meaning |
|---|---|
| docstring | What is on the table; agents read it after the robot's docs |
| `SCENE` | `"station"` (default): Dream's room, table, lights and cameras; `"tabletop"`: the built-in table |
| `OBJECTS` | `Box`, `Cylinder`, `Sphere` (sizes in m, `pos` the centre, sRGB `color`) and `Asset` (a model from Arena's object library, e.g. `"bowl_ycb_robolab"`, scaled to `extent` and set on the table at `xy`). With a `mass` (kg) an object is loose; without one it is fixed |
| `success(objects)` | Optional. Gets an `ObjectState` per object (`pos`, `quat_xyzw`, `lin_vel`, `ang_vel`, bounding box `low`/`high`, `resting()`). Once it has returned True for `SUCCESS_HOLD_S` (1.0 s) in a row, the episode ends with `termination_reason="success"` |
| `distance(objects)` | Optional. Metres to the goal, in each step's `info["distance"]`; `--scorer min_distance_to_goal` reports the closest approach |
| `score(objects)` | Optional. Partial credit from 0 to 1, in each step's `info["score"]`: at the end of the episode, or the best it reached with `SCORE_OVER = "best"` |
| `demo()` | Optional. Waypoints per arm (`"left"`, `"right"`) for `--policy yam_scripted`; `pick_place_waypoints(pick, place, travel_z=...)` builds a top-down pick and place |

`--scorer task_score` scores a trial 1.0 when it ends in success, and otherwise its `score()` (0 without one).
The run prints it at the end (`task_score: 1`), and the log keeps it. Per-step `info` is not saved, so put
anything you want scored into `success`, `distance` or `score`.

World frame: x forward from the arms, y left, z up, in metres; the arm bases stand at x 0, y +-0.305, z 0.75
on the station. Pointing straight down, an arm reaches the table from 0.10 to 0.50 m in front of its own base,
and up to 0.40 m out on the midline between the arms; 16 cm above the table, both ranges end 0.10 m sooner. It
cannot reach in front of the other arm. Check a demo offline with `yam_arena.demo.ArmPlan` (as
`tests/test_task_file.py` does) before spending GPU time. An `Asset` names a model in Arena's object library
(`isaaclab_arena.assets.registries.AssetRegistry`); the station's are `apple_01_objaverse_robolab`,
`orange_02_fruits_veggies_robolab`, `banana_ycb_robolab`, `bowl_ycb_robolab` and `plate_small_vomp_robolab`.

## Status

Dream-1 beta (`dreamlabs/dream-1-beta`) ran on an L40S on 2026-10-31 with the commands above: 20 Hz,
320x176 frames, up to 120 s per trial, the Dream client in-process.

| Scene | Task | Successes |
|---|---|---|
| `station` | put apple into the bowl | 5 of 6 |
| `station` | put orange into the bowl | 6 of 6 |
| `station` | put banana on the plate | 1 of 6 |
| `station`, `top_cam=demo` | the same three tasks | 2, 3 and 0 of 3 |
| `tabletop` | put the red cube on the plate | 0 of 5 (judged from the frames) |

Successful trials took 12 to 90 s (median 24 s). The median round trip per trial was 0.5 to 0.7 s,
and one session that Dream expired mid-trial was reopened without losing the trial. On the tabletop,
Dream reaches the cube and closes the gripper beside it but never lifts it; that scene's top camera
sits 0.5 m lower than the station's, and its arms and floor look nothing like Dream's station. In the
station Dream grasps the fruit and carries it to the goal; the banana, the thinnest object, is the one
it misses. An OpenRouter agent (`anthropic/claude-haiku-5.5`) also ran on the station:
17 steps, then it gave up.

On 2026-11-01 a fresh `l40s-48gb.1x` (128 GB disk) ran the Quickstart: `setup.sh` took 3 minutes and the
install about 40 GB; the first trial, with Isaac Sim's first start, took 6 minutes, later ones under 1 minute
to start. With `-E object_jitter=0`:

| Policy | Task | Result |
|---|---|---|
| `dream` | station: put apple into the bowl | success |
| `yam_scripted` | station: Dream's three tasks | success on each |
| `yam_scripted` | `examples/stack_cubes.py` (station), and the same file on the tabletop | success on each |
| `yam_scripted` | a task file with Arena library assets (an apple into a bowl) | success |
| `yam_scripted` | `stack_cubes.py` with the cube set down 2.8 cm off square | `task_score` 0.5 |
| `dream` | `examples/stack_cubes.py` | no success in 120 s |

## Layout

| Path | What |
|---|---|
| `yam_arena/assets/yam/` | URDF, meshes and convex collision hulls from YAM-inspect-Isaac, which vendors the i2rt model |
| `yam_arena/assets/yam/usd/` | the robot converted to USD by `scripts/convert_usd.py`; the URDF stays the source of truth |
| `yam_arena/assets/yam/dream/` | Dream's split link meshes (I2RT's, MIT) that colour the station's arms |
| `yam_arena/constants.py` | joints, ready pose, gains, timing, and the tabletop's scene and camera poses (quaternions are XYZW) |
| `yam_arena/rig.py`, `station.py` | where the arms and cameras stand in each scene; the station's numbers, objects and tasks, converted from Dream's frame |
| `yam_arena/robot.py`, `actions.py`, `embodiment.py`, `scene.py`, `runtime.py` | the Arena embodiment and environment, and readers of simulator state (import these after Isaac Sim starts) |
| `yam_arena/inspect_embodiment.py` | the Inspect Robots embodiment `yam_arena` |
| `yam_arena/task_file.py`, `examples/` | task files: loading and checking one, and an example |
| `yam_arena/scripted.py`, `scoring.py` | the Inspect Robots policy `yam_scripted` (scripted demos that reach full score) and scorer `task_score` |
| `yam_arena/urdf.py`, `ik.py`, `analysis.py`, `demo.py`, `checks.py` | URDF kinematics, IK, collision-hull distances, the scripted demo and the stage tests' PASS/FAIL records; no Isaac Sim needed |
| `scripts/` | the stage tests, which run in Isaac Sim |
| `setup.sh`, `overrides.txt` | the installer, and the override that keeps Isaac Sim's own `cv2` |

## Stage tests

Run them from the checkout in Arena's venv, for example `python scripts/stage2_one_arm.py`. Each runs
headless and prints PASS or FAIL per check, then a summary; the Python ones write it as JSON with
`--out DIR`.

| Script | Verifies |
|---|---|
| `convert_usd.py` | converts the URDF (`--force`) or checks the shipped USD (`--check-only`): joints, limits, a fixed base, every collision hull, link masses |
| `stage2_one_arm.py` | one arm: fixed base, 7-D action, 240 Hz physics at 10 Hz control, holding the ready pose, link poses against URDF kinematics |
| `stage3_bimanual.py` | both arms: each action slot drives its joint, the 14-D state, arm-on-arm contact without interpenetration |
| `stage4_scene.py` | the tabletop: objects settle, the camera body stops the arm, the scripted pick-and-place and plate press |
| `stage5_cameras.py` | the three cameras render, and the wrist cameras follow the grippers; saves frames to `out/` |
| `stage6_inspect.py`, `stage6_cli.sh` | the Inspect Robots embodiment: its contract, reset and step, an `inspect_robots.eval()` episode, the CLI |
| `station_check.py` | the station: camera poses and intrinsics, object sizes and masses, objects at rest, each task's success check, and side-by-side renders against Dream's replica |
| `check_collision.py` | how closely each link's simulated colliders follow its visual mesh |

## Development

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"
ruff check . && ruff format --check . && pytest
```

The tests need neither a GPU nor Isaac Sim. They check the URDF kinematics, IK, collision hulls,
scripted demo and constants, and that `yam_arena` has the real rig's contract.

## License

**Placeholder.** This repository has no license yet: most of it is ported from
[YAM-inspect-Isaac](https://github.com/ttpierson/YAM-inspect-Isaac) by Thomas Pierson, which has none.

The YAM model in `yam_arena/assets/yam/` comes from [i2rt](https://github.com/i2rt-robotics/i2rt)
under the MIT license: see `LICENSE.i2rt` and `SOURCE_COMMIT` there.
