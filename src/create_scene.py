"""
create_scene_v2.py
==================
Builds scene_v2.usd from scratch matching the physical lab setup described
in the supervisor email:

  Coordinate system (Z-up, metres):
    X  =  right from camera perspective
    Y  =  depth (positive = back / away from camera)
    Z  =  up

  Two 100 cm × 70 cm tables side by side:
    Left table  : X ∈ [-0.70,  0.00],  Y ∈ [0.00, 1.00]
    Right table : X ∈ [ 0.00,  0.70],  Y ∈ [0.00, 1.00]
    Table top surface at Z = 0.74 m
    Front (camera side) = Y = 0.00
    Back  (wall  side)  = Y = 1.00

  Franka mounting (rear-left screwhole + 35 mm to base centre):
    Left  Franka base: X = -0.355,  Y = 0.865,  Z = 0.74
    Right Franka base: X =  0.345,  Y = 0.865,  Z = 0.74
    Both robots face -Y (towards camera / workspace front)

Run with:
    ~/isaac-sim/python.sh create_scene_v2.py
"""

import math
import os

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni.usd                                                       # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, PhysxSchema       # noqa: E402
from isaacsim.core.utils.stage import (                               # noqa: E402
    add_reference_to_stage,
    create_new_stage,
)
from isaacsim.storage.native import get_assets_root_path              # noqa: E402

# ── Output path ───────────────────────────────────────────────────────────────
SCENE_OUT = r"/home/teamb/Desktop/Sim2RealB/scenes/scene_v2.usd"

# ── Asset paths ───────────────────────────────────────────────────────────────
ASSETS_ROOT = get_assets_root_path()
FRANKA_USD  = ASSETS_ROOT + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"

# Set these to the correct Linux paths for your Orca hand USDs.
# If the files don't exist yet, set to None — placeholder prims will be created
# and you can add the references later in USD Composer.
ORCA_LEFT_USD  = "/home/teamb/Desktop/Sim2RealB/assets/orca/scene_left/scene_left.usd"
ORCA_RIGHT_USD = "/home/teamb/Desktop/Sim2RealB/assets/orca/scene_right/scene_right.usd"

# ── Physical constants (all metres) ───────────────────────────────────────────
TABLE_WIDTH   = 0.70   # X extent of each table
TABLE_DEPTH   = 1.00   # Y extent (100 cm, faces camera)
TABLE_HEIGHT  = 0.74   # Z of table top surface
TABLE_THICK   = 0.04   # thickness of table top slab
LEG_SIZE      = 0.05   # cross-section of each table leg
LEG_HEIGHT    = TABLE_HEIGHT - TABLE_THICK   # leg height below table top

# Distance from table edges to rear-left screwhole (from supervisor email)
FROM_BACK     = 0.10   # 10 cm from back edge
FROM_LEFT     = 0.31   # 31 cm from left edge

# Screwhole world positions
# Left table left edge  = X = -TABLE_WIDTH = -0.70
# Right table left edge = X =  0.00
# Back of both tables   = Y =  TABLE_DEPTH  = 1.00
SCREWHOLE_LEFT_X  = -TABLE_WIDTH + FROM_LEFT   # -0.39
SCREWHOLE_RIGHT_X =  0.00        + FROM_LEFT   #  0.31
SCREWHOLE_Y       =  TABLE_DEPTH - FROM_BACK   #  0.90
SCREWHOLE_Z       =  TABLE_HEIGHT              #  0.74

# Link-origin to screwhole offset (Blender measurement by supervisor)
# In robot LOCAL frame: (-0.138, 0.04, 0)
# Robot is rotated -90 deg around Z, so rotate into world frame:
#   local (x, y) -> world: x'= y,  y' = -x   (for -90 deg Z rotation)
#   local (-0.138, 0.04, 0)  ->  world (0.04, 0.138, 0)
# link_origin_world = screwhole_world - world_offset
LINK_TO_HOLE_WORLD_X =  0.04    # metres
LINK_TO_HOLE_WORLD_Y =  0.138   # metres

# Final robot base (link origin) world positions
LEFT_BASE_X  = SCREWHOLE_LEFT_X  - LINK_TO_HOLE_WORLD_X   # -0.43
LEFT_BASE_Y  = SCREWHOLE_Y       - LINK_TO_HOLE_WORLD_Y   #  0.762
RIGHT_BASE_X = SCREWHOLE_RIGHT_X - LINK_TO_HOLE_WORLD_X   #  0.27
RIGHT_BASE_Y = LEFT_BASE_Y                                 #  0.762
BASE_Z       = TABLE_HEIGHT                                #  0.74

# ── Robot orientation: facing -Y (towards camera / workspace front) ────────────
#
# Franka default "reach" direction = +X  (standard URDF convention, Z-up)
# We want reach direction = -Y  (towards camera)
# Rotation: -90° around Z
#   w = cos(-π/4) ≈  0.7071
#   z = sin(-π/4) ≈ -0.7071
#
ROBOT_ORIENT = Gf.Quatd(math.cos(-math.pi / 4),
                         Gf.Vec3d(0.0, 0.0, math.sin(-math.pi / 4)))

# ── Offset from panda_link7 origin to flange (end-of-wrist) ───────────────────
# In the standard Franka URDF, the flange is ~107 mm along panda_link7's Z axis.
ORCA_ATTACH_OFFSET = Gf.Vec3d(0.0, 0.0, 0.107)

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def set_xform(prim: Usd.Prim,
              translate: Gf.Vec3d,
              orient: Gf.Quatd = Gf.Quatd(1, 0, 0, 0),
              scale: Gf.Vec3f  = Gf.Vec3f(1, 1, 1)):
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(translate)
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(orient)
    if scale != Gf.Vec3f(1, 1, 1):
        xf.AddScaleOp().Set(scale)


def add_box(stage: Usd.Stage,
            path: str,
            half_extents: Gf.Vec3f,
            translate: Gf.Vec3d,
            color: Gf.Vec3f = Gf.Vec3f(0.8, 0.75, 0.6)):
    """Create a simple box mesh prim (visual only, no physics)."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.GetSizeAttr().Set(1.0)
    cube.GetDisplayColorAttr().Set([color])
    set_xform(cube.GetPrim(),
              translate=translate,
              scale=Gf.Vec3f(*half_extents) * 2)  # cube size=1 → scale=full size
    return cube.GetPrim()


def strip_physics_from_subtree(stage: Usd.Stage, root_path: str):
    """Remove all physics APIs from a prim subtree (makes it purely visual)."""
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return
    apis_to_remove = [
        UsdPhysics.ArticulationRootAPI,
        UsdPhysics.RigidBodyAPI,
        UsdPhysics.CollisionAPI,
        UsdPhysics.MassAPI,
    ]
    removed_counts = {a.__name__: 0 for a in apis_to_remove}
    joint_prims_removed = 0

    for prim in Usd.PrimRange(root):
        for api in apis_to_remove:
            if prim.HasAPI(api):
                prim.RemoveAPI(api)
                removed_counts[api.__name__] += 1
        if prim.IsA(UsdPhysics.Joint):
            stage.RemovePrim(prim.GetPath())
            joint_prims_removed += 1

    print(f"  Stripped physics from {root_path}:")
    for name, count in removed_counts.items():
        if count:
            print(f"    {name}: {count} prim(s)")
    if joint_prims_removed:
        print(f"    Joints removed: {joint_prims_removed}")


# ─────────────────────────────────────────────────────────────────────────────
# Build scene
# ─────────────────────────────────────────────────────────────────────────────

def build_scene():
    print("\n══════════════════════════════════════════════")
    print(" Building scene_v2.usd")
    print("══════════════════════════════════════════════\n")

    create_new_stage()
    stage = omni.usd.get_context().get_stage()

    # ── Stage metadata ────────────────────────────────────────────────────────
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)  # 1 USD unit = 1 metre

    # ── World Xform (top-level group) ─────────────────────────────────────────
    world = UsdGeom.Xform.Define(stage, "/World")

    # ── Physics scene ─────────────────────────────────────────────────────────
    phys_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    phys_scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    phys_scene.CreateGravityMagnitudeAttr(9.81)

    # ── Ground plane ──────────────────────────────────────────────────────────
    ground = UsdGeom.Mesh.Define(stage, "/World/GroundPlane/CollisionMesh")
    ground_plane_col = UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    ground_xform = UsdGeom.Xform.Define(stage, "/World/GroundPlane")
    set_xform(ground_xform.GetPrim(), Gf.Vec3d(0, 0, 0))
    # Visual ground quad
    ground_vis = UsdGeom.Mesh.Define(stage, "/World/GroundPlane/Visual")
    ground_vis.CreatePointsAttr([(-5,-5,0),( 5,-5,0),( 5,5,0),(-5,5,0)])
    ground_vis.CreateFaceVertexCountsAttr([4])
    ground_vis.CreateFaceVertexIndicesAttr([0,1,2,3])
    ground_vis.GetDisplayColorAttr().Set([Gf.Vec3f(0.35, 0.35, 0.35)])
    # Ground collision plane
    ground_phys = UsdGeom.Plane.Define(stage, "/World/GroundPlane/Collision")
    ground_phys.CreateAxisAttr("Z")
    UsdPhysics.CollisionAPI.Apply(ground_phys.GetPrim())

    print("✓ Ground plane")

    # ── Dome light ────────────────────────────────────────────────────────────
    # A single large dome light gives even ambient illumination across the
    # whole workspace, which is ideal for sim-to-real work (no harsh shadows
    # that differ from the real lab).  Intensity is in nits (cd/m²).
    dome_light = UsdLux.DomeLight.Define(stage, "/World/Lights/DomeLight")
    dome_light.CreateIntensityAttr(1000.0)   # adjust if scene looks too bright/dark
    dome_light.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.95))  # slightly warm white
    # Enable in both real-time (RTX) and path-traced renders
    dome_light.GetPrim().SetCustomDataByKey("isaacsim:visible", True)
    set_xform(dome_light.GetPrim(), translate=Gf.Vec3d(0.0, 0.5, 3.0))
    print("✓ Dome light")

    # ── Tables ────────────────────────────────────────────────────────────────
    # Each table: 4 thin legs + 1 top slab
    for side, sign in [("left", -1), ("right", +1)]:
        table_root = UsdGeom.Xform.Define(stage, f"/World/Table_{side}")

        # Table top slab
        slab_x = sign * TABLE_WIDTH / 2  # centre X of this table
        slab_y = TABLE_DEPTH / 2         # centre Y
        slab_z = TABLE_HEIGHT - TABLE_THICK / 2  # centre Z of slab

        add_box(stage,
                f"/World/Table_{side}/TopSlab",
                half_extents=Gf.Vec3f(TABLE_WIDTH/2, TABLE_DEPTH/2, TABLE_THICK/2),
                translate=Gf.Vec3d(slab_x, slab_y, slab_z),
                color=Gf.Vec3f(0.82, 0.72, 0.50))

            # Add thin collision surface on the slab so objects can rest on it
        slab_prim = stage.GetPrimAtPath(f"/World/Table_{side}/TopSlab")
        UsdPhysics.CollisionAPI.Apply(slab_prim)
        UsdPhysics.RigidBodyAPI.Apply(slab_prim)
        UsdPhysics.RigidBodyAPI(slab_prim).GetKinematicEnabledAttr().Set(True)

        # 4 Legs
        # Leg positions relative to table top corners (inset slightly)
        inset = 0.08
        x_left  = sign * TABLE_WIDTH - sign * inset   # outer edge inset
        x_right = sign * inset                         # inner edge
        leg_positions = [
            (x_left,  inset,             LEG_HEIGHT / 2),
            (x_left,  TABLE_DEPTH-inset, LEG_HEIGHT / 2),
            (x_right, inset,             LEG_HEIGHT / 2),
            (x_right, TABLE_DEPTH-inset, LEG_HEIGHT / 2),
        ]
        for i, (lx, ly, lz) in enumerate(leg_positions):
            leg = add_box(stage,
                          f"/World/Table_{side}/Leg{i}",
                          half_extents=Gf.Vec3f(LEG_SIZE/2, LEG_SIZE/2, LEG_HEIGHT/2),
                          translate=Gf.Vec3d(lx, ly, lz),
                          color=Gf.Vec3f(0.9, 0.9, 0.9))
            UsdPhysics.CollisionAPI.Apply(leg)
            UsdPhysics.RigidBodyAPI.Apply(leg)
            UsdPhysics.RigidBodyAPI(leg).GetKinematicEnabledAttr().Set(True)

    print("✓ Tables (left + right)")

    # ── Franka arms ───────────────────────────────────────────────────────────
    print(f"\nFranka USD: {FRANKA_USD}")
    for side, base_x, base_y in [
        ("left",  LEFT_BASE_X,  LEFT_BASE_Y),
        ("right", RIGHT_BASE_X, RIGHT_BASE_Y),
    ]:
        prim_path = f"/World/Franka_{side}"
        add_reference_to_stage(FRANKA_USD, prim_path)
        franka_prim = stage.GetPrimAtPath(prim_path)

        set_xform(franka_prim,
                  translate=Gf.Vec3d(base_x, base_y, BASE_Z),
                  orient=ROBOT_ORIENT)

        print(f"✓ Franka_{side}  position=({base_x:.3f}, {base_y:.3f}, {BASE_Z:.3f})")
        print(f"           orientation=-90° around Z (facing -Y / camera)")

    # ── Orca hands (visual only) ──────────────────────────────────────────────
    print("\n── Orca hands ──────────────────────────────────────────────────")
    orca_configs = [
        ("left",  ORCA_LEFT_USD,  "/World/Franka_left/panda_link7/orca_hand_left",
                                    "/World/Franka_left/panda_hand",
                                    "/World/Franka_left/panda_rightfinger",
                                    "/World/Franka_left/panda_leftfinger"),
        ("right", ORCA_RIGHT_USD, "/World/Franka_right/panda_link7/orca_hand_right",
                                    "/World/Franka_right/panda_hand",
                                    "/World/Franka_right/panda_rightfinger",
                                    "/World/Franka_right/panda_leftfinger"),
    ]

    for side, orca_usd, orca_path, gripper_hand, gripper_r, gripper_l in orca_configs:
        orca_usd_exists = orca_usd and os.path.exists(orca_usd)

        if orca_usd_exists:
            # Use raw USD API to add the reference as an over on the root layer.
            # add_reference_to_stage() fails here because panda_link7 lives inside
            # the Franka USD reference and may not be fully resolved yet when called.
            # DefinePrim creates the over correctly even on an unresolved parent.
            orca_prim = stage.DefinePrim(orca_path, "Xform")
            orca_prim.GetReferences().AddReference(orca_usd)

            # Position at Franka flange (end of panda_link7, ~107 mm along its Z)
            set_xform(orca_prim, translate=ORCA_ATTACH_OFFSET)

            # Strip ALL physics — make it purely visual
            strip_physics_from_subtree(stage, orca_path)

            # Remove the standard Franka gripper entirely and replace with Orca hand.
            # Also remove panda_hand_joint which connects the gripper to link7 —
            # without removing it PhysX will log errors about a joint with no body.
            gripper_joint = gripper_hand.replace("panda_hand", "panda_link7/panda_hand_joint")
            for remove_path in [
                f"/World/Franka_{side}/panda_hand",
                f"/World/Franka_{side}/panda_rightfinger",
                f"/World/Franka_{side}/panda_leftfinger",
                f"/World/Franka_{side}/panda_link7/panda_hand_joint",
            ]:
                prim = stage.GetPrimAtPath(remove_path)
                if prim.IsValid():
                    prim.SetActive(False)
                    print(f"  Deactivated: {remove_path}")
                else:
                    print(f"  [skip — not found]: {remove_path}")

            print(f"✓ Orca hand {side} added as visual at {orca_path}")
        else:
            placeholder = UsdGeom.Xform.Define(stage, orca_path)
            set_xform(placeholder.GetPrim(), translate=ORCA_ATTACH_OFFSET)
            placeholder.GetPrim().SetCustomDataByKey(
                "TODO",
                f"Add reference to Orca {side} hand USD here. "
                f"Expected path: {orca_usd}"
            )
            print(f"  [placeholder] Orca hand {side} — USD not found at:")
            print(f"    {orca_usd}")
            print(f"    Open scene_v2.usd in USD Composer and add the reference manually.")

    # ── Right arm: set kinematic (no controller yet, prevent drift) ───────────
    print("\n── Freezing right arm (kinematic) ──────────────────────────────")
    right_root = stage.GetPrimAtPath("/World/Franka_right")
    count = 0
    for prim in Usd.PrimRange(right_root):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Set(True)
            count += 1
    print(f"  Set {count} rigid bodies to kinematic under /World/Franka_right")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── Scene summary ───────────────────────────────────────────────")
    print(f"  Stage up-axis   : Z")
    print(f"  Metres per unit : 1.0")
    print(f"  Table surface   : Z = {TABLE_HEIGHT} m")
    print(f"  Franka_left     : ({LEFT_BASE_X:.3f}, {LEFT_BASE_Y:.3f}, {BASE_Z})")
    print(f"  Franka_right    : ({RIGHT_BASE_X:.3f}, {RIGHT_BASE_Y:.3f}, {BASE_Z})")
    print(f"  Robot facing    : -Y  (towards camera)")

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(SCENE_OUT), exist_ok=True)
    stage.Export(SCENE_OUT)
    print(f"\n✓ Saved → {SCENE_OUT}")
    print("  Use scene_v2.usd in all future scripts (update SCENE_PATH).\n")


if __name__ == "__main__":
    build_scene()
    simulation_app.close()


# ─────────────────────────────────────────────────────────────────────────────
# NOTES FOR NEXT STEPS
# ─────────────────────────────────────────────────────────────────────────────
#
# 1. Robot orientation check
#    After opening scene_v2.usd, verify that both Franka arms face the front
#    (towards the camera / -Y direction).  If they face the wrong way, adjust
#    ROBOT_ORIENT by changing the rotation angle:
#      Facing -Y (towards camera) = -90° around Z  ← current setting
#      Facing +Y (away from camera) = +90° around Z
#      Facing +X                    =   0° (no rotation)
#      Facing -X                    = 180° around Z
#
# 2. Orca hand attachment offset
#    ORCA_ATTACH_OFFSET = (0, 0, 0.107) places the hand at the standard
#    Franka flange position.  If the hand appears misaligned, adjust this
#    value.  The correct value depends on how the Orca USD defines its root
#    frame relative to the mounting interface.
#
# 3. Using this scene with franka_ik_replay_final.py
#    Update SCENE_PATH in that script to point to scene_v2.usd.
#    The Franka base transforms will be read automatically from the stage —
#    no other changes are needed.
#
# 4. Phase 2: Both arms simultaneously
#    Remove the kinematic flag from Franka_right and add a second
#    FrankaIKReplay controller using observations/qpos_arm_right.
#
# 5. Phase 3: Orca hand motion
#    Read observations/qpos_hand_left and observations/qpos_hand_right
#    and drive the hand joints directly (no IK needed — these are joint
#    angles).  The hands are already in the scene as visual prims; you
#    just need to promote them to a driven SingleArticulation at that point
#    and re-enable their physics (keeping them separate from the Franka
#    articulation).
# ─────────────────────────────────────────────────────────────────────────────