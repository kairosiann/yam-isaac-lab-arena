# yam-arena

Bimanual [I2RT YAM](https://i2rt.com/products/yam-6-dof-arm) for **Isaac Lab-Arena 0.3.1**
(Isaac Lab 3.0 EA, Isaac Sim 6.1.0, Python 3.12, PhysX), plus its
[Inspect Robots](https://github.com/robocurve/inspect-robots) embodiment `yam_arena`.
It's a port of the Isaac Lab 2.3 package YAM-inspect-Isaac (`yam_sim`), and has the same
observation and action contract as the real rig's `yam_arms` (inspect-robots-yam 0.37.0).

| | |
|---|---|
| Action | 14-D absolute joint targets `[L j1..j6, L grip, R j1..j6, R grip]`, rad; grippers 0 closed .. 1 open; 10 Hz |
| State `joint_pos` | the same 14-D layout |
| Cameras | `top_cam` (D435, overhead), `left_cam` / `right_cam` (D405, wrists); rendered 640x360, observed 224x224 RGB |
| Control | joint PD (i2rt gains) + gravity-compensation feed-forward every 240 Hz substep, 24 substeps per action |
| Scene | table, red 4 cm cube, white plate, dome light, ground, and a static 90x25x25 mm collider for the top camera's body |

## Layout

| Path | What |
|---|---|
| `yam_arena/assets/yam/` | URDF, meshes and CoACD collision hulls from the 2.3 repo (vendored from i2rt, MIT: `LICENSE.i2rt`, `SOURCE_COMMIT`) |
| `yam_arena/assets/yam/usd/yam_sim/yam_sim.usda` | the converted robot, made by `scripts/convert_usd.py` (shipped; the URDF stays the source of truth) |
| `yam_arena/constants.py` | joints, READY pose, gains, scene and camera poses (quaternions are XYZW) |
| `yam_arena/robot.py`, `actions.py`, `embodiment.py`, `scene.py` | Arena embodiment `yam_bimanual` (scene keys `left_robot` / `right_robot`) and the tabletop environment |
| `yam_arena/inspect_embodiment.py` | Inspect Robots embodiment `yam_arena` |
| `yam_arena/urdf.py`, `ik.py`, `analysis.py`, `demo.py` | NumPy-only URDF FK/IK, collision-hull distances, the scripted contact demo (used by the tests) |
| `scripts/convert_usd.py` | stage 1: URDF -> USD and its checks |
| `scripts/stage{2..6}_*.py`, `stage6_cli.sh`, `check_collision.py` | stage tests; each prints PASS/FAIL per check and a summary |

## Install (inside the Arena 0.3.1 venv)

```bash
source IsaacLab-Arena/.venv/bin/activate        # Arena 0.3.1 checkout, `uv sync` done
uv pip install --override ir-overrides.txt -e ./yam-arena
```

`ir-overrides.txt` holds `opencv-python-headless ; sys_platform == "never"`, which stops
inspect-robots-yam's opencv from replacing Isaac Sim's own `cv2`. Install editable: the assets
are read from the checkout. In an existing venv, never run plain `uv sync` afterwards, because it
uninstalls the Inspect Robots packages (use `uv sync --inexact`).

## Use

```bash
python scripts/convert_usd.py --force          # only when the URDF changes (the USD is shipped)
python scripts/convert_usd.py --check-only     # check the shipped USD
python scripts/stage2_one_arm.py               # each stage ends with a PASS/FAIL summary
...
inspect-robots list embodiments                # includes yam_arena
inspect-robots run --embodiment yam_arena --policy agent -P model=... --instruction "put the red cube on the plate"
```

Arena:

```python
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True, enable_cameras=True).app
from yam_arena.embodiment import BimanualYamEmbodiment
from yam_arena.scene import make_env
env = make_env(BimanualYamEmbodiment(enable_cameras=True))   # ArenaEnvBuilder, NoTask, 1 env
```

`-E` options for `yam_arena`: `cam_height`, `cam_width`, `control_hz`, `device`,
`wrist_cam_mount` (`link`, the default, parents the wrist cameras to the gripper link;
`follow` moves them every substep), and `headless` (`false` asks for a Kit window, untested).

## What changed from the 2.3 port

- **Fixed base.** The 3.0 URDF importer leaves `ArticulationRootAPI` on the `base` rigid body,
  which PhysX simulates as a floating base pinned by a maximal joint (`is_fixed_base=False`, and
  the arm flails). `convert_usd.py` moves the root schemas to the asset's default prim, as
  overrides in the interface layer.
- **Robot colliders** use PhysX default contact offsets. 2.3 asked for 5 mm but could not author
  it on instanced colliders (it logged a warning), so 2.3's measured behaviour had defaults too.
  `yam_arm_cfg(contact_offset=0.005)` applies it if you want it.
- **New:** a static collider for the top camera's D435 body. An optional camera mast exists
  (`camera_mast=True`); its shape is a guess, because 2.3 models no mount.
- The URDF's link colours now render: black arms with silver tips, where 2.3 rendered them white.
- The scripted demo grips straight down at 0.16 m hover, rotated 180 deg about the approach
  axis. The 2.3 orientation is outside joint6's range there, and 0.20 m is outside joint4's.
- The `edit` GUI and scene files from 2.3 are not ported.
