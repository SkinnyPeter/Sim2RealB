"""
Steps:
  1. Open Isaac Sim and load (or create) a stage.
  2. Make sure the duck mesh is already on the stage as a Xform prim.
     Set PRIM_PATH below to match its path.
  3. Set TRAJECTORY_NPY to the path of the .npy file exported by export_trajectory.py.
     Shape: (N, 4, 4) — ob_in_cam poses, OpenCV convention (X right, Y down, Z forward).
  4. Run the script — it bakes the poses as USD keyframes so you can scrub
     the timeline or press Play to watch the animation.
"""

import numpy as np
from pxr import Gf, UsdGeom
import omni.usd
import omni.timeline

# ── Configuration ────────────────────────────────────────────────────────────
TRAJECTORY_NPY = "/home/teamb/Desktop/Sim2RealB/data/trajectory.npy"
PRIM_PATH      = "/World/duck"   # USD path of the mesh prim to animate
TIME_CODES_PER_FRAME = 1         # increase if your stage runs at e.g. 60 fps
# ─────────────────────────────────────────────────────────────────────────────

# OpenCV (Y-down, Z-forward) -> Isaac Sim / USD (Y-up, Z-backward)
CV2USD = np.array([
    [1,  0,  0,  0],
    [0, -1,  0,  0],
    [0,  0, -1,  0],
    [0,  0,  0,  1],
], dtype=np.float64)


def bake_trajectory(prim_path, poses):
    """
    poses: (N, 4, 4) numpy array of ob_in_cam matrices
    """
    stage = omni.usd.get_context().get_stage()

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        UsdGeom.Xform.Define(stage, prim_path)
        prim = stage.GetPrimAtPath(prim_path)
        print(f"Created Xform prim at {prim_path}. "
              "Attach your mesh as a child or replace the path.")

    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    translate_op = xform.AddTranslateOp()
    orient_op    = xform.AddOrientOp()

    for frame_idx, ob_in_cam in enumerate(poses):
        ob_in_usd = CV2USD @ ob_in_cam
        t  = ob_in_usd[:3, 3]
        R  = ob_in_usd[:3, :3]

        # Rotation matrix -> quaternion (w, x, y, z)
        from scipy.spatial.transform import Rotation
        q = Rotation.from_matrix(R).as_quat()  # scipy: [qx, qy, qz, qw]
        qw, qx, qy, qz = q[3], q[0], q[1], q[2]

        tc = frame_idx * TIME_CODES_PER_FRAME
        translate_op.Set(Gf.Vec3d(float(t[0]), float(t[1]), float(t[2])), tc)
        orient_op.Set(Gf.Quatd(float(qw), float(qx), float(qy), float(qz)), tc)

    timeline = omni.timeline.get_timeline_interface()
    last_tc  = (len(poses) - 1) * TIME_CODES_PER_FRAME
    timeline.set_end_time(last_tc / timeline.get_time_codes_per_second())

    print(f"Baked {len(poses)} keyframes onto {prim_path}")


poses = np.load(TRAJECTORY_NPY)   # (N, 4, 4)
print(f"Loaded trajectory: {poses.shape}")
bake_trajectory(PRIM_PATH, poses)
