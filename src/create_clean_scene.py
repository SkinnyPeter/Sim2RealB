from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from pathlib import Path
import time
import omni.usd

from pxr import UsdGeom, Gf, UsdLux


BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_SCENE = BASE_DIR / "scenes" / "scene.usd"

FRANKA_USD = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"

ORCA_LEFT_USD = BASE_DIR / "assets" / "orca" / "scene_left" / "scene_left.usd"
ORCA_RIGHT_USD = BASE_DIR / "assets" / "orca" / "scene_right" / "scene_right.usd"


def add_reference(stage, prim_path: str, asset_path: str, prim_type: str = "Xform"):
    prim = stage.DefinePrim(prim_path, prim_type)
    prim.GetReferences().AddReference(asset_path)
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

    franka_left = add_reference(stage, "/World/Franka_left", FRANKA_USD)
    franka_left.SetInstanceable(False)
    set_xform(
        franka_left,
        translate=(-0.25, 0.35, 0.77),
        rotate_xyz=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
    )

    franka_right = add_reference(stage, "/World/Franka_right", FRANKA_USD)
    franka_right.SetInstanceable(False)
    set_xform(
        franka_right,
        translate=(-0.25, -0.35, 0.77),
        rotate_xyz=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
    )

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

    # Let Franka references resolve before authoring under panda_hand
    for _ in range(30):
        simulation_app.update()
        time.sleep(0.1)

    if ORCA_LEFT_USD.exists():
        add_orca_under_hand(
            stage,
            "/World/Franka_left/panda_hand",
            "ORCA_left",
            str(ORCA_LEFT_USD),
            translate=(0.0, 0.0, 0.0),
            rotate_xyz=(0.0, 0.0, 0.0),
            scale=(1.0, 1.0, 1.0),
        )
    else:
        print(f"WARNING: ORCA left USD not found: {ORCA_LEFT_USD}")

    if ORCA_RIGHT_USD.exists():
        add_orca_under_hand(
            stage,
            "/World/Franka_right/panda_hand",
            "ORCA_right",
            str(ORCA_RIGHT_USD),
            translate=(0.0, 0.0, 0.0),
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