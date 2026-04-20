"""Build the dual-arm scene from scratch.

Creates a new USD stage, adds two independent tables each with an overhead
camera, references one robot on each table, then saves to ./scenes/scene.usd.
"""

from isaacsim import SimulationApp
VISUALIZE = True
simulation_app = SimulationApp({"headless": not VISUALIZE})

from pathlib import Path
import os
import omni.usd
from pxr import UsdGeom, UsdLux, UsdPhysics, Gf

BASE_DIR = Path(__file__).resolve().parent.parent
DESCRIPTION_ROOT = Path(
    os.environ.get(
        "PANDAORCA_ROOT",
        str(BASE_DIR / "assets" / "pandaorca_description-main"),
    )
)

LEFT_ROBOT_USD  = DESCRIPTION_ROOT / "usd" / "fer_orcahand_left_extended"  / "fer_orcahand_left_extended.usd"
RIGHT_ROBOT_USD = DESCRIPTION_ROOT / "usd" / "fer_orcahand_right_extended" / "fer_orcahand_right_extended.usd"
OUTPUT_SCENE    = BASE_DIR / "scenes" / "scene.usd"

# Cameras 
T1 = (-0.045965, 0.095137, 1.203792)
R1 = (36.244522, 2.155732, -91.261346)
T2 = (-0.074849, -0.003505, 1.238954)
R2 = (33.763158, 4.545887, -88.313427)

def add_reference(stage, prim_path: str, asset_path: Path):
    prim = stage.DefinePrim(prim_path, "Xform")
    prim.GetReferences().AddReference(str(asset_path))
    print(f"Added reference: {prim_path} -> {asset_path}")
    return prim


def set_xform(prim, translate=(0, 0, 0), rotate_xyz=(0, 0, 0), scale=(1, 1, 1)):
    xform = UsdGeom.Xformable(prim)
    xform.AddTranslateOp(opSuffix="scene").Set(Gf.Vec3d(*translate))
    xform.AddRotateXYZOp(opSuffix="scene").Set(Gf.Vec3f(*rotate_xyz))
    xform.AddScaleOp(opSuffix="scene").Set(Gf.Vec3f(*scale))
    print(f"  translate={translate}  rotateXYZ={rotate_xyz}  scale={scale}")


def create_table(stage, prim_path: str, translate, scale):
    prim = stage.DefinePrim(prim_path, "Cube")
    UsdPhysics.CollisionAPI.Apply(prim)
    set_xform(prim, translate=translate, scale=scale)
    print(f"Table created: {prim_path}")
    return prim


def add_camera(stage, prim_path: str, translate, rotate_xyz=(0, 0, 0)):
    """Overhead camera — with Z-up, identity orientation looks straight down."""
    cam = UsdGeom.Camera.Define(stage, prim_path)
    xform = UsdGeom.Xformable(cam)
    xform.AddTranslateOp(opSuffix="scene").Set(Gf.Vec3d(*translate))
    xform.AddRotateXYZOp(opSuffix="scene").Set(Gf.Vec3f(*rotate_xyz))
    cam.CreateFocalLengthAttr(24.0)
    cam.CreateHorizontalApertureAttr(36.0)
    print(f"Camera created: {prim_path}  translate={translate}")
    return cam


def main():
    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    if stage is None:
        raise RuntimeError("Could not create a new USD stage.")

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    stage.DefinePrim("/World", "Xform")

    # Physics scene
    physics = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    physics.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    physics.CreateGravityMagnitudeAttr(9.81)

    # Ground plane
    ground = UsdGeom.Mesh.Define(stage, "/World/GroundPlane/CollisionMesh")
    ground.CreatePointsAttr([(-25, -25, 0), (25, -25, 0), (25, 25, 0), (-25, 25, 0)])
    ground.CreateFaceVertexCountsAttr([4])
    ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    ground.CreateDisplayColorAttr([(0.5, 0.5, 0.5)])
    UsdPhysics.CollisionAPI.Apply(
        stage.DefinePrim("/World/GroundPlane/CollisionPlane", "Plane")
    )

    # Lighting
    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(500.0)

    # ── LEFT SIDE ──────────────────────────────────────────────────────────
    # Table: 1 m × 0.7 m × 1 m box, top surface at z = 1.0 m
    create_table(
        stage, "/World/table_left",
        translate=(0.0, 0.35, 0.75),
        scale=(0.5, 0.35, 0.02),
    )
    # Overhead camera above the left table, looking straight down
    add_camera(
        stage, "/World/camera_left",
        translate=T1,
        rotate_xyz=R1
    )


    # Robot
    if LEFT_ROBOT_USD.exists():
        left = add_reference(stage, "/World/fer_orcahand_left_extended", LEFT_ROBOT_USD)
        left.SetInstanceable(False)
        set_xform(left, translate=(-0.255, 0.35, 0.77))
    else:
        print(f"WARNING: left robot USD not found: {LEFT_ROBOT_USD}")

    # ── RIGHT SIDE ─────────────────────────────────────────────────────────
    create_table(
        stage, "/World/table_right",
        translate=(0.0, -0.35, 0.75),
        scale=(0.5, 0.35, 0.02),
    )
    add_camera(
        stage, "/World/camera_right",
        translate=T2,
        rotate_xyz=R2
    )

    if RIGHT_ROBOT_USD.exists():
        right = add_reference(stage, "/World/fer_orcahand_right_extended", RIGHT_ROBOT_USD)
        right.SetInstanceable(False)
        set_xform(right, translate=(-0.255, -0.35, 0.77))
    else:
        print(f"WARNING: right robot USD not found: {RIGHT_ROBOT_USD}")

    # Visualization Loop
    if VISUALIZE:
        print("Visualization mode active. Close the window or press Ctrl+C to exit.")
        try:
            while simulation_app.is_running():
                simulation_app.update()
        except KeyboardInterrupt:
            pass

    # Save
    OUTPUT_SCENE.parent.mkdir(parents=True, exist_ok=True)
    if not ctx.save_as_stage(str(OUTPUT_SCENE)):
        raise RuntimeError(f"Could not save scene to: {OUTPUT_SCENE}")
    print(f"\nScene saved to: {OUTPUT_SCENE}")
    simulation_app.close()


if __name__ == "__main__":
    main()
