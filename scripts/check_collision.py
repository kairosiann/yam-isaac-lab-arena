"""Check that each YAM link's simulated collision geometry matches its visual mesh (Isaac Lab 3.0).

Port of the 2.3 scripts/check_collision.py. Spawns one arm at zero joint angles from the
converted USD and probes the colliders PhysX actually simulates with raycasts. For each link it
samples points on the visual surface and casts a ray inward along the surface normal, starting
``--standoff`` outside it, keeping only hits on that link's own colliders. The signed error is how
much sooner the ray meets the collider than the link's visual surface:

- positive: collider bulges out past the mesh (e.g. convex hulls bridging a concavity)
- negative: collider is inset from the mesh
- miss:     no collider of this link anywhere along the ray

What changed for 3.0: the importer nests links (base/link1/link2/...) and stores each hull as an
instanced ``<link>_NNN`` Xform next to the instanced visual ``<link>``; links and colliders are
found by schema (RigidBodyAPI, CollisionAPI) instead of by the 2.3 ``/visuals/`` and
``/collisions/`` path names, and a hit belongs to the link with the longest matching path.

    python scripts/check_collision.py [--out DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from yam_arena.checks import Checks  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--samples", type=int, default=3000, help="Visual surface points probed per link.")
parser.add_argument("--standoff", type=float, default=0.05, help="Ray start distance outside the surface (m).")
parser.add_argument("--tol", type=float, default=0.002, help="Error within this many metres counts as matching.")
parser.add_argument("--out", type=Path, default=None)
args = parser.parse_args()
checks = Checks("check_collision", args.out)

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True).app

import numpy as np  # noqa: E402

ROBOT = "/World/Yam"


def world_mesh(prim):
    import trimesh
    from pxr import Usd, UsdGeom

    mesh = UsdGeom.Mesh(prim)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
    faces, start = [], 0
    for c in counts:  # fan-triangulate polygons
        faces += [[idx[start], idx[start + k], idx[start + k + 1]] for k in range(1, c - 1)]
        start += c
    xf = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    pts = np.c_[pts, np.ones(len(pts))] @ xf  # USD: row vectors
    return trimesh.Trimesh(pts[:, :3], np.array(faces), process=True)


def main():
    import carb
    import isaaclab.sim as sim_utils
    import omni.physx
    import omni.usd
    import trimesh
    from isaaclab.assets import Articulation
    from pxr import Usd, UsdGeom, UsdPhysics

    from yam_arena.robot import yam_arm_cfg

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 240, enable_scene_query_support=True))
    cfg = yam_arm_cfg(ROBOT, (0.0, 0.0, 0.0))
    cfg.init_state = cfg.init_state.replace(joint_pos={".*": 0.0})
    robot = Articulation(cfg)
    sim.reset()
    sim.step()
    checks.check("arm spawned fixed-base", bool(robot.is_fixed_base), f"is_fixed_base={robot.is_fixed_base}")

    stage = omni.usd.get_context().get_stage()
    prims = [p for p in stage.Traverse(Usd.TraverseInstanceProxies()) if str(p.GetPath()).startswith(ROBOT + "/")]
    links = {str(p.GetPath()): p.GetName() for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    by_len = sorted(links, key=len, reverse=True)

    def link_of(path: str) -> str | None:
        for lp in by_len:
            if path == lp or path.startswith(lp + "/"):
                return links[lp]
        return None

    visuals: dict[str, list] = {}
    for p in prims:
        if p.IsA(UsdGeom.Mesh) and not p.HasAPI(UsdPhysics.CollisionAPI):
            if UsdGeom.Imageable(p).ComputePurpose() in ("guide", "proxy"):
                continue
            visuals.setdefault(link_of(str(p.GetPath())), []).append(world_mesh(p))
    checks.note("visual meshes per link", {k: len(v) for k, v in visuals.items()})

    sq = omni.physx.get_physx_scene_query_interface()
    rng = np.random.default_rng(0)
    print(f"\n{'link':10s} {'match':>6s} {'bulge':>6s} {'inset':>6s} {'miss':>6s}   err p5/p50/p95 mm   other-link hits", flush=True)
    table = {}
    for link in sorted(k for k in visuals if k):
        vis = trimesh.util.concatenate(visuals[link])
        pts, face = trimesh.sample.sample_surface(vis, args.samples, seed=int(rng.integers(1 << 31)))
        normals = vis.face_normals[face]
        errs, others, origins = [], {}, []
        for p, n in zip(pts, normals):
            hits = []

            def report(h, hits=hits):
                hits.append((h.distance, h.collision))
                return True

            origin = p + n * args.standoff
            sq.raycast_all(carb.Float3(*origin), carb.Float3(*(-n)), 2 * args.standoff, report)
            mine = [d for d, c in hits if link_of(str(c)) == link]
            for d, c in hits:
                other = link_of(str(c))
                if other != link and d < args.standoff:
                    others[other] = others.get(other, 0) + 1
            origins.append(origin)
            errs.append(min(mine) if mine else np.nan)
        origins = np.array(origins)
        dirs = -normals
        vis_hit = np.full(len(origins), args.standoff)
        locs, ray_idx, _ = vis.ray.intersects_location(origins, dirs, multiple_hits=True)
        if len(ray_idx):
            d = np.einsum("ij,ij->i", locs - origins[ray_idx], dirs[ray_idx])
            np.minimum.at(vis_hit, ray_idx, np.where(d > 1e-6, d, np.inf))
        inside = vis.contains(origins)
        errs = (vis_hit - np.array(errs))[~inside]
        hit = errs[~np.isnan(errs)]
        frac = lambda m: 100 * m.sum() / max(1, len(errs))  # noqa: E731
        pct = np.percentile(hit * 1000, [5, 50, 95]) if len(hit) else [np.nan] * 3
        row = dict(match=frac(np.abs(hit) <= args.tol), bulge=frac(hit > args.tol), inset=frac(hit < -args.tol),
                   miss=frac(np.isnan(errs)), p5=pct[0], p50=pct[1], p95=pct[2], skipped=int(inside.sum()))
        table[link] = {k: round(float(v), 2) for k, v in row.items()}
        print(f"{link:10s} {row['match']:5.1f}% {row['bulge']:5.1f}% {row['inset']:5.1f}% {row['miss']:5.1f}%"
              f"   {pct[0]:6.1f}/{pct[1]:5.1f}/{pct[2]:5.1f}     {others or ''}"
              + (f"  ({inside.sum()} rays skipped: start inside link)" if inside.any() else ""), flush=True)
    checks.note("fit table", table)
    checks.check("all 9 links probed", len(table) == 9, str(sorted(table)))
    worst_miss = max(r["miss"] for r in table.values())
    checks.check("every link's colliders are hit (miss < 5% per link)", worst_miss < 5.0, f"worst miss {worst_miss:.1f}%")


try:
    main()
except Exception as exc:  # noqa: BLE001
    checks.error("check_collision", exc)
checks.finish()
