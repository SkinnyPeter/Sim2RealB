from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from pathlib import Path
import time
import omni.usd

from pxr import UsdGeom, Gf, UsdLux, Usd


BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_SCENE = BASE_DIR / "scenes" / "scene.usd"

FRANKA_USD = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"

ORCA_LEFT_USD = BASE_DIR / "assets" / "orca_flat" / "scene_left_flat.usd"
ORCA_RIGHT_USD = BASE_DIR / "assets" / "orca_flat" / "scene_right_flat.usd"
CONNECTOR_LEFT_USD = BASE_DIR / "assets" / "connectors" / "connector_left_flat.usd"
CONNECTOR_RIGHT_USD = BASE_DIR / "assets" / "connectors" / "connector_right_flat.usd"


def add_reference(stage, prim_path: str, asset_path: str, prim_type: str = "Xform"):
    prim = stage.DefinePrim(prim_path, prim_type)
    prim.GetReferences().AddReference(str(asset_path))
    print(f"Added reference: {prim_path} -> {asset_path}")
    return prim


def set_xform(prim, translate=(0, 0, 0), rotate_xyz=(0, 0, 0), scale=(1, 1, 1)):
    xform = UsdGeom.Xformable(prim)

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


def create_table(stage, prim_path: str, translate, rotate_xyz, scale):
    prim = stage.DefinePrim(prim_path, "Cube")
    set_xform(prim, translate=translate, rotate_xyz=rotate_xyz, scale=scale)
    return prim


def add_child_reference(
    stage,
    parent_path: str,
    child_name: str,
    asset_path: str,
    translate=(0, 0, 0),
    rotate_xyz=(0, 0, 0),
    scale=(1, 1, 1),
):
    parent = stage.GetPrimAtPath(parent_path)
    if not parent or not parent.IsValid():
        print(f"WARNING: parent path not found: {parent_path}")
        return None

    child_path = f"{parent_path}/{child_name}"
    prim = stage.DefinePrim(child_path, "Xform")
    prim.GetReferences().AddReference(str(asset_path))
    set_xform(prim, translate=translate, rotate_xyz=rotate_xyz, scale=scale)
    print(f"Added child reference: {child_path}")
    return prim


def hide_prim(stage, prim_path: str):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        print(f"WARNING: prim not found for hiding: {prim_path}")
        return

    imageable = UsdGeom.Imageable(prim)
    if imageable:
        imageable.MakeInvisible()
        print(f"Hid prim: {prim_path}")


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

    # Franka robots
    franka_left = add_reference(stage, "/World/Franka_left", FRANKA_USD)
    franka_left.SetInstanceable(False)
    set_xform(
        franka_left,
        translate=(-0.255, 0.35, 0.77),
        rotate_xyz=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
    )

    franka_right = add_reference(stage, "/World/Franka_right", FRANKA_USD)
    franka_right.SetInstanceable(False)
    set_xform(
        franka_right,
        translate=(-0.255, -0.35, 0.77),
        rotate_xyz=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
    )

    # Tables
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

    # Let Franka references resolve before authoring inside them
    for _ in range(30):
        simulation_app.update()
        time.sleep(0.1)

    # Hide default Franka hand parts visually only
    hide_prim(stage, "/World/Franka_left/panda_hand")
    hide_prim(stage, "/World/Franka_right/panda_hand")
    hide_prim(stage, "/World/Franka_left/panda_leftfinger")
    hide_prim(stage, "/World/Franka_left/panda_rightfinger")
    hide_prim(stage, "/World/Franka_right/panda_leftfinger")
    hide_prim(stage, "/World/Franka_right/panda_rightfinger")

    # Connectors stay under panda_link7 (visual only)
    left_connector_parent = "/World/Franka_left/panda_link7"
    if CONNECTOR_LEFT_USD.exists():
        add_child_reference(
            stage,
            left_connector_parent,
            "connector_left",
            CONNECTOR_LEFT_USD,
            translate=(0.033, 0.08, 0.11),
            rotate_xyz=(137.0, 0.0, -90.0),
            scale=(0.001, 0.001, 0.001),
        )
    else:
        print(f"WARNING: connector left USD not found: {CONNECTOR_LEFT_USD}")

    right_connector_parent = "/World/Franka_right/panda_link7"
    if CONNECTOR_RIGHT_USD.exists():
        add_child_reference(
            stage,
            right_connector_parent,
            "connector_right",
            CONNECTOR_RIGHT_USD,
            translate=(0.033, 0.08, 0.11),
            rotate_xyz=(137.0, 0.0, -90.0),
            scale=(0.001, 0.001, 0.001),
        )
    else:
        print(f"WARNING: connector right USD not found: {CONNECTOR_RIGHT_USD}")

    # ORCA hands as separate articulations under /World
    # Temporary placement only — simulator.py will make them follow panda_link7
    if ORCA_LEFT_USD.exists():
        add_reference(stage, "/World/ORCA_left", ORCA_LEFT_USD)
        set_xform(
            stage.GetPrimAtPath("/World/ORCA_left"),
            translate=(0.2, 0.6, 1.0),
            rotate_xyz=(0.0, 0.0, 0.0),
            scale=(1.0, 1.0, 1.0),
        )
    else:
        print(f"WARNING: ORCA left USD not found: {ORCA_LEFT_USD}")

    if ORCA_RIGHT_USD.exists():
        add_reference(stage, "/World/ORCA_right", ORCA_RIGHT_USD)
        set_xform(
            stage.GetPrimAtPath("/World/ORCA_right"),
            translate=(0.2, -0.6, 1.0),
            rotate_xyz=(0.0, 0.0, 0.0),
            scale=(1.0, 1.0, 1.0),
        )
    else:
        print(f"WARNING: ORCA right USD not found: {ORCA_RIGHT_USD}")

    for _ in range(20):
        simulation_app.update()
        time.sleep(0.1)

    out_path = str(OUTPUT_SCENE)
    print(f"Saving stage to: {out_path}")
    ctx.save_as_stage(out_path)
    print("Scene saved successfully.")

    simulation_app.close()


if __name__ == "__main__":
    main()