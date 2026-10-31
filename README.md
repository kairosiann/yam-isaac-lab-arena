# yam-arena

Two [I2RT YAM](https://i2rt.com/products/yam-6-dof-arm) arms in
[Isaac Lab-Arena](https://github.com/isaac-sim/IsaacLab-Arena) 0.3.1 (Isaac Lab 3.0, Isaac Sim 6.1),
and the [Inspect Robots](https://github.com/robocurve/inspect-robots) embodiment `yam_arena` with the
same action layout, state and cameras as the real rig's `yam_arms`. Use it to evaluate Dream Labs'
Dream-1 ([`inspect-robots-dream`](https://github.com/kairosiann/inspect-robots-dream)) or an LLM agent
on the YAM in simulation. It is a port of
[YAM-inspect-Isaac](https://github.com/ttpierson/YAM-inspect-Isaac) (`yam_sim`, Isaac Lab 2.3).
Not affiliated with I2RT, NVIDIA or Dream Labs.

## Quickstart

You need a Linux machine with an NVIDIA RTX GPU (an L40S, for example) and its driver. Isaac Sim
renders with RT cores, so an A100 or H100 will not do.

Install Arena, this package and the policies:

```bash
git clone https://github.com/kairosiann/yam-arena && cd yam-arena
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
| Resets | both arms to the ready pose; a seeded reset shifts each loose object by up to 3 cm |

Policies written for `yam_arms` are compatible with `yam_arena` as they are. The docs it hands an
agent describe each joint's direction, checked against the URDF in `tests/test_urdf.py`.

| `-E` option | Default | Meaning |
|---|---|---|
| `scene` | `tabletop` | `tabletop` or `station` (see [Scenes](#scenes)) |
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
| `tabletop` | The table from `yam_sim`: arms 0.6 m apart, a red 4 cm cube, a white plate, the D435 0.43 m above the arms |
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

## Differences from yam_sim

- **Fixed base.** Isaac Lab 3.0's URDF importer leaves the articulation root on the `base` rigid body,
  which PhysX simulates as a floating base. `convert_usd.py` moves the root to the asset's default prim.
- **Camera body.** A static 90 x 25 x 25 mm box stands at the top camera, so an arm cannot pass
  through it.
- **Contact offsets.** The arms use PhysX's default contact offsets. `yam_sim` asked for 5 mm, but
  Isaac Lab 2.3 could not set it on instanced colliders. `yam_arm_cfg(contact_offset=0.005)` sets it.
- **Scripted demo.** The grippers point straight down from 0.16 m, turned 180 degrees about the
  approach axis. `yam_sim`'s orientation needs joint6 past its limit there, and its 0.20 m hover needs
  joint4 past its limit; `tests/test_demo.py` checks both.
- The scene editor (`yam-sim edit`) is not ported.

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
