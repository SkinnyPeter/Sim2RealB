from isaacsim import SimulationApp
VISUALIZE = True
simulation_app = SimulationApp({"headless": not VISUALIZE})

from pathlib import Path
import time
import omni.usd
from pxr import UsdGeom, Gf, UsdLux

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_SCENE = BASE_DIR / "scenes" / "scene.usd"
FRANKA_USD = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"

ORCA_LEFT_USD = BASE_DIR / "assets" / "hands_usd" /"scene_left.usd"
ORCA_RIGHT_USD = BASE_DIR / "assets" / "hands_usd" / "scene_right.usd"

OBJECT_USD = BASE_DIR / "assets" / "obj" / "rubber_duck.usd"

from pxr import Usd

stage = Usd.Stage.Open(str(ORCA_LEFT_USD))
print("Default prim:", stage.GetDefaultPrim())
for prim in stage.Traverse():
    print(prim.GetPath(), prim.GetTypeName())


def add_reference(stage, prim_path: str, asset_path: str, prim_type: str = "Xform"):
    prim = stage.DefinePrim(prim_path, prim_type)
    prim.GetReferences().AddReference(asset_path)
    prim.SetInstanceable(False)
    print(f"Added reference: {prim_path} -> {asset_path}")
    return prim

def set_xform(prim, translate=(0, 0, 0), rotate_xyz=(0, 0, 0), scale=(1, 1, 1)):
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()

    t_op = xform.AddTranslateOp(opSuffix="scene")
    r_op = xform.AddRotateXYZOp(opSuffix="scene")
    s_op = xform.AddScaleOp(opSuffix="scene")

    t_op.Set(Gf.Vec3d(*translate))
    r_op.Set(Gf.Vec3f(*rotate_xyz))
    s_op.Set(Gf.Vec3f(*scale))

    print(f"Set transform for {prim.GetPath()}")
    print(f"  translate = {translate}")
    print(f"  rotateXYZ = {rotate_xyz}")
    print(f"  scale     = {scale}")

def deactivate_prim(stage, prim_path):
    """
    Deactivate a prim in the current edit target.
    """

    print("Had many bugs with this one, use hide_prim_visual if possible")

    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid():
        prim.SetActive(False)
        print(f"Deactivated prim: {prim_path}")
    else:
        print(f"WARNING: prim not found for deactivation: {prim_path}")

def hide_prim_visual(stage, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    if prim and prim.IsValid():
        imageable = UsdGeom.Imageable(prim)
        if imageable:
            imageable.MakeInvisible()
            print(f"Hidden prim: {prim_path}")

def create_table(stage, prim_path: str, translate, rotate_xyz, scale):
    prim = stage.DefinePrim(prim_path, "Cube")
    set_xform(prim, translate=translate, rotate_xyz=rotate_xyz, scale=scale)
    return prim

def add_orca_under_hand(
    stage,
    hand_path: str,
    orca_name: str,
    orca_usd: str,
    translate=(0, 0, 0),
    rotate_xyz=(0, 0, 0),
    scale=(1, 1, 1),
):
    parent = stage.GetPrimAtPath(hand_path)
    if not parent or not parent.IsValid():
        print(f"WARNING: hand path not found: {hand_path}")
        return None

    prim_path = f"{hand_path}/{orca_name}"
    prim = stage.DefinePrim(prim_path, "Xform")
    prim.GetReferences().AddReference(orca_usd)

    set_xform(prim, translate=translate, rotate_xyz=rotate_xyz, scale=scale)
    print(f"Added ORCA under hand: {prim_path}")
    return prim

def main():
    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()

    if stage is None:
        raise RuntimeError("Could not create a new USD stage.")

    stage.DefinePrim("/World", "Xform")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(500.0)

    # Load robots
    franka_left = add_reference(stage, "/World/Franka_left", FRANKA_USD)
    set_xform(
        franka_left,
        translate=(-0.25, 0.35, 0.77),
        rotate_xyz=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
    )

    franka_right = add_reference(stage, "/World/Franka_right", FRANKA_USD)
    set_xform(
        franka_right,
        translate=(-0.25, -0.35, 0.77),
        rotate_xyz=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
    )

    # Wait for loading so the sub-hierarchy exists
    print("Loading robot components...")
    for _ in range(60): simulation_app.update()

    # Hide the original hands for both robots
    for side in ["left", "right"]:
            root = f"/World/Franka_{side}"
            # These are not used in the manipulation
            # deactivate_prim(stage, f"{root}/panda_hand/geometry") # TODO: panda_hand is not the actual EEF - Once the actual EEF is hide panda_hand
            hide_prim_visual(stage, f"{root}/panda_leftfinger")
            hide_prim_visual(stage, f"{root}/panda_rightfinger")

    # Attach ORCA
    if ORCA_LEFT_USD.exists():
            path = "/World/ORCA_left" # TODO: Change this when we will have the connector
            prim = stage.DefinePrim(path, "Xform")
            prim.GetReferences().AddReference(str(ORCA_LEFT_USD))
            # Adjust transform if ORCA is misaligned with the wrist
            set_xform(prim, translate=(0, 0, 0), rotate_xyz=(0, 0, 0), scale=(1,1,1))
            print(f"ORCA RIGHT added successfully from: {ORCA_LEFT_USD}")
    else: print(f"WARNING: ORCA left USD not found: {ORCA_LEFT_USD}")

    if ORCA_RIGHT_USD.exists():
        path = "/World/ORCA_right" # TODO: Change this when we will have the connector
        prim = stage.DefinePrim(path, "Xform")
        prim.GetReferences().AddReference(str(ORCA_RIGHT_USD))
        set_xform(prim, translate=(0, 0, 0), rotate_xyz=(0, 0, 0), scale=(1,1,1))
        print(f"ORCA RIGHT added successfully from: {ORCA_RIGHT_USD}")
    else: print(f"WARNING: ORCA right USD not found: {ORCA_RIGHT_USD}")

    # Load tables
    create_table(
        stage,
        "/World/table_left",
        translate=(0.0, 0.35, 0.75),
        rotate_xyz=(0.0, 0.0, 0.0),
        scale=(0.5, 0.35, 0.02),
    )

    create_table(
        stage,
        "/World/table_right",
        translate=(0.0, -0.35, 0.75),
        rotate_xyz=(0.0, 0.0, 0.0),
        scale=(0.5, 0.35, 0.02),
    )

    # Load object (rubber duck)
    if OBJECT_USD.exists():
        path = "/World/object"
        prim = stage.DefinePrim(path, "Xform")
        prim.GetReferences().AddReference(str(OBJECT_USD))
        set_xform(prim, translate=(0, 0, 0), rotate_xyz=(0, 0, 0), scale=(0.001,0.001,0.001))
        print(f"OBJECT added successfully from: {OBJECT_USD}")
    else: print(f"WARNING: OBJECT USD not found: {OBJECT_USD}")

    # Let Franka references resolve before authoring under panda_hand
    for _ in range(30):
        simulation_app.update()
        time.sleep(0.1)

    out_path = str(OUTPUT_SCENE)
    print(f"Saving stage to: {out_path}")
    ctx.save_as_stage(out_path)
    print("Scene saved successfully.")

    # --- Visualization Loop ---
    if VISUALIZE:
        print("Visualization mode active. Close the window or press Ctrl+C to exit.")
        try:
            while simulation_app.is_running():
                simulation_app.update()
        except KeyboardInterrupt:
            pass

    simulation_app.close()


if __name__ == "__main__":
    main()