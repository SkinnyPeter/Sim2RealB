"""
Standalone Isaac Sim script — replays a (N, 4, 4) trajectory on the rubber duck.

Run with:
    python replay_trajectories_standalone.py

The poses in trajectory.npy are ob_in_cam (object pose expressed in the camera
frame), using the OpenCV convention (X right, Y down, Z forward).
We convert them to the USD / Isaac Sim world convention (Y-up, Z-backward)
and bake them as USD keyframes so you can scrub the timeline or press Play.
"""

# ── Launch Isaac Sim first — must happen before any omni.* imports ────────────
from isaacsim import SimulationApp
HEADLESS = False                            # set True for no GUI
simulation_app = SimulationApp({"headless": HEADLESS})
# ─────────────────────────────────────────────────────────────────────────────

from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

import omni.usd
import omni.timeline
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent.parent
TRAJECTORY_NPY = BASE_DIR / "data" / "trajectory.npy"
DUCK_USD       = BASE_DIR / "assets" / "obj" / "rubber_duck.usd"
DUCK_PRIM_PATH = "/World/RubberDuck"

# 1 time-code per frame; increase if your stage fps > 1
TIME_CODES_PER_FRAME = 20
STAGE_FPS = 20                        # keyframe playback fps — lower = slower (0.5 = 1 frame every 2 seconds)

# Smoothing — set SMOOTH = False to replay raw FoundationPose output
SMOOTH               = True
JUMP_THRESH_DEG      = 30.0   # frames with a larger rotation jump are treated as outliers
TRANSLATION_SIGMA    = 5.0    # Gaussian sigma (in frames) applied to XYZ
ROTATION_WINDOW      = 21     # rolling-average window for quaternions (frames, odd)
# ─────────────────────────────────────────────────────────────────────────────

# OpenCV (X right, Y down, Z forward) → USD/Isaac (X right, Y up, Z backward)
CV2USD = np.array([
    [1,  0,  0,  0],
    [0, -1,  0,  0],
    [0,  0, -1,  0],
    [0,  0,  0,  1],
], dtype=np.float64)


def smooth_poses(poses,
                 jump_thresh_deg=JUMP_THRESH_DEG,
                 translation_sigma=TRANSLATION_SIGMA,
                 rotation_window=ROTATION_WINDOW):
    """
    Clean up noisy FoundationPose ob_in_cam trajectories.

    Two-pass approach:
      1. Detect frames where the rotation jumps by more than *jump_thresh_deg*
         (FoundationPose re-initialisations / 180° symmetry flips) and replace
         them by SLERP interpolation from the last good frame to the next good
         frame.
      2. Apply a Gaussian filter to the translations and a rolling quaternion
         average to the rotations to reduce residual jitter.
    """
    from scipy.spatial.transform import Rotation, Slerp
    from scipy.ndimage import gaussian_filter1d

    N = len(poses)
    poses = poses.copy()

    # ── Pass 1: interpolate over large-jump (outlier) frames ─────────────────
    quats = Rotation.from_matrix(poses[:, :3, :3]).as_quat()   # (N,4) xyzw

    # Ensure sign consistency so angle differences are meaningful
    for i in range(1, N):
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]

    # Compute per-frame rotation magnitude (degrees)
    rot_deltas = np.zeros(N)
    for i in range(1, N):
        dR = poses[i - 1, :3, :3].T @ poses[i, :3, :3]
        rot_deltas[i] = np.degrees(Rotation.from_matrix(dR).magnitude())

    bad = rot_deltas > jump_thresh_deg
    bad_indices = np.where(bad)[0]
    print(f"Outlier frames (>{jump_thresh_deg:.0f}° jump): {len(bad_indices)} "
          f"— {bad_indices.tolist()[:10]}{'...' if len(bad_indices) > 10 else ''}")

    if bad_indices.size:
        # Build good-frame time array for Slerp
        good_mask = ~bad
        good_mask[0] = True   # always keep first frame
        good_t   = np.where(good_mask)[0].astype(float)
        good_q   = quats[good_mask]
        good_xyz = poses[good_mask, :3, 3]

        slerp    = Slerp(good_t, Rotation.from_quat(good_q))
        all_t    = np.arange(N, dtype=float)

        # Replace bad frames
        quats[bad_indices] = slerp(all_t[bad_indices]).as_quat()
        # Linearly interpolate translations for bad frames
        for ax in range(3):
            poses[bad_indices, ax, 3] = np.interp(
                all_t[bad_indices], good_t, good_xyz[:, ax])

    # ── Pass 2: temporal smoothing ────────────────────────────────────────────
    # Translations — Gaussian filter
    poses[:, :3, 3] = gaussian_filter1d(poses[:, :3, 3],
                                        sigma=translation_sigma, axis=0)

    # Rotations — rolling quaternion average (naïve but works for small windows)
    half_w = rotation_window // 2
    quats_smooth = quats.copy()
    for i in range(N):
        start = max(0, i - half_w)
        end   = min(N, i + half_w + 1)
        avg   = quats[start:end].mean(axis=0)
        quats_smooth[i] = avg / np.linalg.norm(avg)

    poses[:, :3, :3] = Rotation.from_quat(quats_smooth).as_matrix()

    print(f"Smoothing done  (σ_t={translation_sigma} frames, "
          f"rot window={rotation_window} frames)")
    return poses


def setup_stage():
    """Create a minimal stage: World root + dome light."""
    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()

    if stage is None:
        raise RuntimeError("Failed to create a new USD stage.")

    stage.DefinePrim("/World", "Xform")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetFramesPerSecond(STAGE_FPS)

    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(500.0)

    print("Stage created.")
    return stage


def add_duck(stage):
    """Reference the rubber duck USD onto the stage (no transform set here)."""
    duck_path = str(DUCK_USD)
    prim = stage.DefinePrim(DUCK_PRIM_PATH, "Xform")

    if not DUCK_USD.exists():
        print(f"WARNING: Duck USD not found at {duck_path}. Using placeholder Xform.")
    else:
        prim.GetReferences().AddReference(duck_path)
        print(f"Duck added: {DUCK_PRIM_PATH}")

    return prim


def bake_trajectory(stage, prim, poses):
    """
    Bake (N, 4, 4) ob_in_cam poses as USD keyframes on *prim*.

    USD xform ops are composed left-to-right and applied right-to-left to
    vertices.  Correct order for this stack:
        [translate, orient, scale]
    → scale is applied first (mm mesh → metres), then orient, then translate.
    This keeps the translation in metres and unaffected by the scale.
    """
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()

    translate_op = xform.AddTranslateOp(opSuffix="anim")
    orient_op    = xform.AddOrientOp(opSuffix="anim")
    # Scale last in the stack → applied first to vertices (mm → m).
    xform.AddScaleOp(opSuffix="init").Set(Gf.Vec3f(0.001, 0.001, 0.001))

    for frame_idx, ob_in_cam in enumerate(poses):
        ob_in_usd = CV2USD @ ob_in_cam

        t = ob_in_usd[:3, 3]                    # translation (metres)
        R = ob_in_usd[:3, :3]                    # rotation matrix

        q = Rotation.from_matrix(R).as_quat()    # [qx, qy, qz, qw] (scipy)
        qw, qx, qy, qz = float(q[3]), float(q[0]), float(q[1]), float(q[2])

        tc = frame_idx * TIME_CODES_PER_FRAME
        translate_op.Set(Gf.Vec3d(float(t[0]), float(t[1]), float(t[2])), tc)
        orient_op.Set(Gf.Quatf(qw, qx, qy, qz), tc)

    print(f"Baked {len(poses)} keyframes → {prim.GetPath()}")


def configure_timeline(poses):
    """Set the timeline end time to match the number of keyframes."""
    timeline = omni.timeline.get_timeline_interface()
    tps      = timeline.get_time_codes_per_second()          # usually == STAGE_FPS
    last_tc  = (len(poses) - 1) * TIME_CODES_PER_FRAME
    timeline.set_end_time(last_tc / tps)
    timeline.set_current_time(0.0)
    print(f"Timeline: 0 – {last_tc / tps:.2f} s  "
          f"({len(poses)} frames @ {tps} tcs/s)")


def main():
    # 1. Load trajectory
    poses = np.load(str(TRAJECTORY_NPY))        # (N, 4, 4)
    print(f"Loaded trajectory: {poses.shape}")
    if SMOOTH:
        poses = smooth_poses(poses)

    # 2. Build stage
    stage = setup_stage()

    # Let Omniverse resolve the new stage before authoring prims
    for _ in range(10):
        simulation_app.update()

    # 3. Add duck and bake keyframes
    duck_prim = add_duck(stage)
    bake_trajectory(stage, duck_prim, poses)

    # 4. Configure timeline
    configure_timeline(poses)

    # 5. Press Play and run the visualisation loop
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    print("Playing trajectory… close the window or press Ctrl+C to exit.")

    try:
        while simulation_app.is_running():
            simulation_app.update()
    except KeyboardInterrupt:
        pass

    timeline.stop()
    simulation_app.close()


if __name__ == "__main__":
    main()
