"""
TODO:
extrinsics are labeled cam -> base. That means this matrix can be used directly only if the camera prim is defined relative to that same base frame. If /World is not that calibration base frame, you need:

T_cam_world = T_base_world @ T_cam_base

or the equivalent depending on how you define your frames.
"""

from isaacsim import SimulationApp
VISUALIZE = True
simulation_app = SimulationApp({"headless": not VISUALIZE})

from pathlib import Path
import time
import omni.usd
from pxr import UsdGeom, Gf, UsdLux, Sdf
import numpy as np

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_SCENE = BASE_DIR / "scenes" / "scene.usd"
FRANKA_USD = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"

ORCA_LEFT_USD = BASE_DIR / "assets" / "hands_usd" /"scene_left.usd"
ORCA_RIGHT_USD = BASE_DIR / "assets" / "hands_usd" / "scene_right.usd"

OBJECT_USD = BASE_DIR / "assets" / "obj" / "rubber_duck.usd"

ARIA_INTRINSICS = np.array([
    [133.25430222 * 2, 0.0, 320, 0],
    [0.0, 133.25430222 * 2, 240, 0],
    [0.0, 0.0, 1.0, 0]
])

ARIA_INTRINSICS_HALF = np.array([
    [133.25430222, 0.0, 320 / 2, 0],
    [0.0, 133.25430222, 240 / 2, 0],
    [0.0, 0.0, 1.0, 0],
])

best_calib = {
    "left_cam": np.array([
        [-0.02199727, -0.80581615,  0.59175708,  0.20403467],
        [-0.99905014,  0.03998766,  0.01731508, -0.25486327],
        [-0.03761575, -0.59081411, -0.80593036,  0.43379187],
        [ 0.        ,  0.        ,  0.        ,  1.        ]
    ]),
    "right_cam": np.array([
        [ 0.02933941, -0.83227828,  0.55358113,  0.17515134],
        [-0.99642232,  0.01956109,  0.08221870,  0.34649483],
        [-0.07925749, -0.55401284, -0.82872675,  0.46895363],
        [ 0.        ,  0.        ,  0.        ,  1.        ]
    ])
}

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

def set_xform_from_matrix(prim, T):
    """
    Set a prim transform from a 4x4 homogeneous matrix.
    T must be the transform of the prim with respect to its parent frame.
    """
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform, got {T.shape}")

    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()

    mat_op = xform.AddTransformOp(opSuffix="scene")
    mat_op.Set(
        Gf.Matrix4d(
            T[0, 0], T[0, 1], T[0, 2], T[0, 3],
            T[1, 0], T[1, 1], T[1, 2], T[1, 3],
            T[2, 0], T[2, 1], T[2, 2], T[2, 3],
            T[3, 0], T[3, 1], T[3, 2], T[3, 3],
        )
    )

def add_camera(
    stage,
    prim_path: str,
    intrinsics: np.ndarray,
    extrinsic_cam_to_parent: np.ndarray,
    image_size=(640, 480),
    clipping_range=(0.01, 100.0),
    horizontal_aperture_mm=20.955,
):
    """
    Add a pinhole camera to the USD stage.

    Parameters
    ----------
    stage : Usd.Stage
    prim_path : str
        Example: "/World/left_cam"
    intrinsics : np.ndarray
        3x4 or 3x3 intrinsic matrix.
        Expected:
            [[fx, 0, cx, ...],
             [0, fy, cy, ...],
             [0,  0,  1, ...]]
    extrinsic_cam_to_parent : np.ndarray
        4x4 transform from camera frame to parent/base frame.
        Since your calibration is noted as (cam -> base), this can be used
        directly if the parent frame in Isaac Sim is that same base frame.
    image_size : tuple[int, int]
        (width, height)
    clipping_range : tuple[float, float]
    horizontal_aperture_mm : float
        Chosen sensor width in mm. 20.955 is a common USD default.
        Focal length and vertical aperture are derived consistently from this.
    """
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if intrinsics.shape not in [(3, 3), (3, 4)]:
        raise ValueError(f"Expected intrinsics shape (3,3) or (3,4), got {intrinsics.shape}")

    T = np.asarray(extrinsic_cam_to_parent, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"Expected extrinsic shape (4,4), got {T.shape}")

    width, height = image_size
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])

    # Create USD camera prim
    camera = UsdGeom.Camera.Define(stage, prim_path)
    prim = camera.GetPrim()

    # Pose
    set_xform_from_matrix(prim, T)

    # Convert pixel intrinsics -> USD camera parameters
    #
    # USD relation:
    #   fx_pixels = focalLength_mm / horizontalAperture_mm * image_width_pixels
    # so:
    #   focalLength_mm = fx_pixels * horizontalAperture_mm / image_width_pixels
    focal_length_mm = fx * horizontal_aperture_mm / width
    vertical_aperture_mm = horizontal_aperture_mm * height / width

    camera.CreateHorizontalApertureAttr(horizontal_aperture_mm)
    camera.CreateVerticalApertureAttr(vertical_aperture_mm)
    camera.CreateFocalLengthAttr(focal_length_mm)
    camera.CreateClippingRangeAttr(Gf.Vec2f(*clipping_range))

    # Principal point offsets from image center
    #
    # USD offsets are in mm on the sensor plane.
    # Positive sign conventions can differ depending on the rendering pipeline.
    # This formulation is the usual one to start with.
    horiz_offset_mm = (cx - width / 2.0) * horizontal_aperture_mm / width
    vert_offset_mm = (cy - height / 2.0) * vertical_aperture_mm / height

    camera.CreateHorizontalApertureOffsetAttr(horiz_offset_mm)
    camera.CreateVerticalApertureOffsetAttr(vert_offset_mm)

    # Optional metadata to keep the intended raster size nearby
    prim.CreateAttribute("camera:imageWidth", Sdf.ValueTypeNames.Int).Set(width)
    prim.CreateAttribute("camera:imageHeight", Sdf.ValueTypeNames.Int).Set(height)

    print(f"Added camera: {prim_path}")
    print(f"  fx, fy = {fx:.4f}, {fy:.4f}")
    print(f"  cx, cy = {cx:.4f}, {cy:.4f}")
    print(f"  focalLength = {focal_length_mm:.6f} mm")
    print(f"  hAperture   = {horizontal_aperture_mm:.6f} mm")
    print(f"  vAperture   = {vertical_aperture_mm:.6f} mm")
    print(f"  hOffset     = {horiz_offset_mm:.6f} mm")
    print(f"  vOffset     = {vert_offset_mm:.6f} mm")

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

    # TODO: put the right image_size
    # For the half-resolution intrinsics replace with ARIA_INTRINSICS_HALF
    add_camera(
        stage,
        "/World/left_cam",
        intrinsics=ARIA_INTRINSICS,
        extrinsic_cam_to_parent=best_calib["left_cam"],
        image_size=(640, 480),
    )

    # TODO: put the right image_size
    add_camera(
        stage,
        "/World/right_cam",
        intrinsics=ARIA_INTRINSICS,
        extrinsic_cam_to_parent=best_calib["right_cam"],
        image_size=(640, 480),
    )
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