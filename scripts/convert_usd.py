"""Stage 1: convert yam_sim.urdf to USD with Isaac Lab 3.0's URDF importer, then check it.

    python scripts/convert_usd.py            # convert if missing or stale, then check
    python scripts/convert_usd.py --force    # always re-convert
    python scripts/convert_usd.py --check-only

Writes yam_arena/assets/yam/usd/yam_sim/yam_sim.usda (a layered asset: interface layer plus
payloads/). Conversion settings: fixed base, no fixed-joint merging, zero drive gains (the
Arena embodiment's actuator config sets the real gains), the "physx" physics variant.

Each URDF <collision> is already one convex CoACD hull (the coworker's
make_collision_meshes.py), and the importer marks every URDF collision mesh with the
convexHull approximation, so PhysX keeps the hulls as-is. Note that the cfg field
``collision_type`` only applies to collision_from_visuals, so it is not what makes them hulls.

Post-processing (``fix_articulation_root``): the importer leaves ``ArticulationRootAPI`` on the
``base`` rigid body and anchors ``base`` with ``/yam/Physics/root_joint``. PhysX treats an
articulation rooted on a rigid body as *floating*-base (the fixed joint becomes a maximal
constraint), which stage 2 measured (``is_fixed_base == False``, the arm flailing). Isaac Lab
3.0's PhysX backend fixes the same topology by moving the root off the rigid body
(``PhysxManager.fix_articulation_root``); here the root schemas move to the asset's default
prim ``/yam``, authored as overrides in the interface layer (yam_sim.usda).

Checks (from the URDF, which stays the source of truth):
  - joint1..joint8 exist with the URDF's types and limits
  - the base is fixed (a world-anchored FixedJoint to `base`, articulation root found)
  - every link has collider prims, with the URDF's hull count, all convexHull
  - link masses match the URDF
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from yam_arena import YAM_URDF, YAM_USD, YAM_USD_DIR  # noqa: E402
from yam_arena.checks import Checks  # noqa: E402
from yam_arena.constants import ALL_JOINTS, LINKS  # noqa: E402
from yam_arena.urdf import collision_counts, parse_joints  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--force", action="store_true", help="Re-convert even if the USD exists.")
parser.add_argument("--check-only", action="store_true", help="Only check the existing USD.")
parser.add_argument("--out", type=Path, default=None, help="Directory for the JSON summary.")
args = parser.parse_args()

checks = Checks("stage1_convert_usd", args.out)

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True).app


def convert() -> None:
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    # The importer never overwrites: a non-empty yam_sim/ makes it write yam_sim_1/, so start clean.
    if YAM_USD_DIR.exists():
        shutil.rmtree(YAM_USD_DIR)
    cfg = UrdfConverterCfg(
        asset_path=str(YAM_URDF),
        usd_dir=str(YAM_USD_DIR),
        fix_base=True,
        merge_fixed_joints=False,
        self_collision=False,
        collision_from_visuals=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
        physics_variant="physx",
        force_usd_conversion=True,
    )
    converter = UrdfConverter(cfg)
    checks.note("converter_usd_path", converter.usd_path)
    if Path(converter.usd_path).resolve() != YAM_USD.resolve():
        raise RuntimeError(f"importer wrote {converter.usd_path}, expected {YAM_USD}")


ROOT_SCHEMAS = ("PhysicsArticulationRootAPI", "PhysxArticulationAPI", "NewtonArticulationRootAPI")
ROOT_PROPERTY_PREFIXES = ("physxArticulation:", "newton:selfCollisionEnabled")


def fix_articulation_root() -> None:
    """Move the articulation-root schemas from the `base` rigid body to the default prim."""
    from pxr import Sdf, Usd, UsdPhysics

    stage = Usd.Stage.Open(str(YAM_USD))
    root_prim = stage.GetDefaultPrim()
    roots = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if len(roots) != 1:
        raise RuntimeError(f"expected one articulation root, found {[str(r.GetPath()) for r in roots]}")
    old = roots[0]
    if not old.HasAPI(UsdPhysics.RigidBodyAPI):
        checks.note("fix_articulation_root", f"root already on non-rigid prim {old.GetPath()}, nothing to do")
        return
    stage.SetEditTarget(Usd.EditTarget(stage.GetRootLayer()))
    listed = []
    for spec in old.GetPrimStack():
        if spec.HasInfo("apiSchemas"):
            listed += [t for t in spec.GetInfo("apiSchemas").GetAddedOrExplicitItems() if t in ROOT_SCHEMAS]
    listed = list(dict.fromkeys(listed))
    moved_props = []
    for prop in old.GetAuthoredProperties():
        if prop.GetName().startswith(ROOT_PROPERTY_PREFIXES):
            if not prop.FlattenTo(root_prim):
                raise RuntimeError(f"could not copy {prop.GetPath()} to {root_prim.GetPath()}")
            moved_props.append(prop.GetName())
    for name in listed:
        if not root_prim.AddAppliedSchema(name):
            raise RuntimeError(f"could not apply {name} to {root_prim.GetPath()}")
    for name in listed:
        if not old.RemoveAppliedSchema(name):
            raise RuntimeError(f"could not remove {name} from {old.GetPath()}")
    for name in moved_props:
        old.RemoveProperty(name)
    stage.GetRootLayer().Save()
    checks.note("fix_articulation_root", {"from": str(old.GetPath()), "to": str(root_prim.GetPath()),
                                          "schemas": listed, "properties": moved_props})


def link_of(prim, rigid_api) -> str | None:
    """Name of the nearest ancestor (or self) with RigidBodyAPI."""
    p = prim
    while p and p.IsValid() and str(p.GetPath()) != "/":
        if p.HasAPI(rigid_api):
            return p.GetName()
        p = p.GetParent()
    return None


def check_usd() -> None:
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(YAM_USD))
    checks.check("usd opens", bool(stage), str(YAM_USD))
    root = stage.GetDefaultPrim()
    checks.note("default_prim", str(root.GetPath()))
    layers = sorted(str(p.relative_to(YAM_USD.parent)) for p in YAM_USD.parent.rglob("*") if p.is_file())
    checks.note("usd_files", layers)
    variants = root.GetVariantSets()
    if "Physics" in variants.GetNames():
        checks.note("physics_variant", variants.GetVariantSet("Physics").GetVariantSelection())

    prims = list(stage.Traverse(Usd.TraverseInstanceProxies()))
    # A short tree (no meshes) so later stages know the link and camera-mount paths.
    tree = [str(p.GetPath()) for p in prims if str(p.GetPath()).count("/") <= 3]
    checks.note("prim_tree_depth3", tree)

    # -- joints ----------------------------------------------------------------------
    urdf_joints = parse_joints(YAM_URDF)
    usd_joints = {}
    for p in prims:
        if p.IsA(UsdPhysics.Joint):
            usd_joints[p.GetName()] = p
    checks.note("usd_joint_names", sorted(usd_joints))
    missing = [j for j in ALL_JOINTS if j not in usd_joints]
    checks.check("joint1..joint8 present", not missing, f"missing={missing}" if missing else "all 8 found")
    for name in ALL_JOINTS:
        if name not in usd_joints:
            continue
        prim, ref = usd_joints[name], urdf_joints[name]
        if ref.type == "revolute":
            joint = UsdPhysics.RevoluteJoint(prim)
            ok_type = prim.IsA(UsdPhysics.RevoluteJoint)
            lo, hi = (math.radians(v) for v in (joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()))
            unit = "rad"
        else:
            joint = UsdPhysics.PrismaticJoint(prim)
            ok_type = prim.IsA(UsdPhysics.PrismaticJoint)
            lo, hi = joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()
            unit = "m"
        axis = joint.GetAxisAttr().Get()
        b0 = [str(t).split("/")[-1] for t in joint.GetBody0Rel().GetTargets()]
        b1 = [str(t).split("/")[-1] for t in joint.GetBody1Rel().GetTargets()]
        tol = 1e-4
        ok_lim = abs(lo - ref.lower) < tol and abs(hi - ref.upper) < tol
        flipped = abs(lo + ref.upper) < tol and abs(hi + ref.lower) < tol
        detail = (
            f"type={prim.GetTypeName()} axis={axis} body0={b0} body1={b1} "
            f"usd=[{lo:+.5f}, {hi:+.5f}] urdf=[{ref.lower:+.5f}, {ref.upper:+.5f}] {unit}"
            + (" (limits negated: the importer flipped the axis; stage 2 FK decides)" if flipped and not ok_lim else "")
        )
        checks.check(f"{name} type {ref.type}", ok_type, detail)
        checks.check(f"{name} limits match URDF", ok_lim, detail)
        ok_bodies = b0 == [ref.parent] and b1 == [ref.child]
        checks.check(f"{name} connects {ref.parent} -> {ref.child}", ok_bodies, f"body0={b0} body1={b1}")

    # -- fixed base ------------------------------------------------------------------
    # A fixed joint whose body0 is empty or a prim without RigidBodyAPI anchors body1 to the world.
    fixed = []
    for p in prims:
        if p.IsA(UsdPhysics.FixedJoint):
            j = UsdPhysics.FixedJoint(p)
            b0 = [stage.GetPrimAtPath(t) for t in j.GetBody0Rel().GetTargets()]
            b1 = [str(t) for t in j.GetBody1Rel().GetTargets()]
            world = all(not b.IsValid() or not b.HasAPI(UsdPhysics.RigidBodyAPI) for b in b0)
            fixed.append((str(p.GetPath()), [str(b.GetPath()) for b in b0], b1, world))
    anchors = [f for f in fixed if f[3] and len(f[2]) == 1 and f[2][0].endswith("/base")]
    checks.check("a world-anchored FixedJoint holds `base` (USD)", len(anchors) == 1, f"fixed joints={fixed}")
    roots = [p for p in prims if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    checks.check("exactly one ArticulationRootAPI", len(roots) == 1, f"roots={[str(r.GetPath()) for r in roots]}")
    if roots:
        r = roots[0]
        checks.note("articulation_root", {"path": str(r.GetPath()), "applied_schemas": list(r.GetAppliedSchemas())})
        checks.check(
            "articulation root is not a rigid body (PhysX fixed-base topology)",
            not r.HasAPI(UsdPhysics.RigidBodyAPI),
            f"root at {r.GetPath()}; is_fixed_base itself is measured at runtime in stage 2",
        )

    # -- colliders -------------------------------------------------------------------
    expected = collision_counts(YAM_URDF)
    counts: dict[str, int] = {}
    approx: dict[str, set] = {}
    for p in prims:
        if not p.HasAPI(UsdPhysics.CollisionAPI):
            continue
        link = link_of(p, UsdPhysics.RigidBodyAPI) or "<no rigid body>"
        counts[link] = counts.get(link, 0) + 1
        if p.HasAPI(UsdPhysics.MeshCollisionAPI):
            approx.setdefault(link, set()).add(str(UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get()))
        else:
            approx.setdefault(link, set()).add(f"non-mesh:{p.GetTypeName()}")
    checks.note("collider_counts_per_link", counts)
    checks.note("collider_approximation_per_link", {k: sorted(v) for k, v in approx.items()})
    for link in LINKS:
        n, want = counts.get(link, 0), expected.get(link, 0)
        checks.check(f"{link} colliders", n == want and n > 0, f"{n} collider prims (URDF has {want} hulls)")
    stray = {k: v for k, v in counts.items() if k not in LINKS}
    checks.check("no colliders outside the 9 links", not stray, f"stray={stray}")
    non_hull = {k: sorted(v) for k, v in approx.items() if v != {"convexHull"}}
    checks.check("every collider is a convexHull mesh", not non_hull, f"other={non_hull}" if non_hull else "")

    # -- masses ----------------------------------------------------------------------
    import xml.etree.ElementTree as ET

    urdf_mass = {
        link.get("name"): float(link.find("inertial/mass").get("value"))
        for link in ET.parse(YAM_URDF).getroot().iter("link")
        if link.find("inertial/mass") is not None
    }
    usd_mass = {}
    for p in prims:
        if p.HasAPI(UsdPhysics.RigidBodyAPI) and p.HasAPI(UsdPhysics.MassAPI):
            usd_mass[p.GetName()] = UsdPhysics.MassAPI(p).GetMassAttr().Get()
    bad = {k: (usd_mass.get(k), v) for k, v in urdf_mass.items() if usd_mass.get(k) is None or abs(usd_mass[k] - v) > 1e-4}
    checks.check("link masses match URDF", not bad, f"mismatch={bad}" if bad else f"{len(usd_mass)} links")


try:
    if not args.check_only:
        if args.force or not YAM_USD.is_file():
            convert()
            fix_articulation_root()
        else:
            checks.note("conversion", "skipped: USD exists (pass --force to re-convert)")
    check_usd()
except Exception as exc:  # noqa: BLE001 -- report and still print the summary
    checks.error("stage 1", exc)
checks.finish()
