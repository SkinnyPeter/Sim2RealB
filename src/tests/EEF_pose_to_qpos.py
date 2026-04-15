"""
Franka Left Arm – IK Replay from Orca Hand Poses
=================================================
Builds on the NVIDIA FrankaKinematicsExample pattern.
 
Assumptions about the HDF5 data (adjust if needed):
  - observations/qpos_arm_left  shape (N, 7)  → [x, y, z, qw, qx, qy, qz]
    i.e. EEF position (metres, world frame) + quaternion (w-first)
"""
 
import os
import time
 
import carb
import h5py
import numpy as np
 
from isaacsim import SimulationApp
 
# ──────────────────────────────────────────────
# Launch the simulator FIRST – before any omni imports
# ──────────────────────────────────────────────
simulation_app = SimulationApp({"headless": False})
 
from isaacsim.core.api import World                                         # noqa: E402
from isaacsim.core.prims import SingleArticulation                          # noqa: E402
from isaacsim.core.utils.extensions import get_extension_path_from_name    # noqa: E402
from isaacsim.core.utils.stage import open_stage                            # noqa: E402
from isaacsim.robot_motion.motion_generation import (                       # noqa: E402
    ArticulationKinematicsSolver,
    LulaKinematicsSolver,
)
 
# ──────────────────────────────────────────────
# Paths  –  edit these
# ──────────────────────────────────────────────
SCENE_PATH = r"/home/teamb/Desktop/Sim2RealB/scenes/scene.usd"
H5_PATH    = r"/home/teamb/Desktop/Sim2RealB/data/20250827_151212.h5"
 
# Prim path of the left Franka inside your USD scene
ROBOT_PRIM_PATH = "/World/Franka_left"
 
# End-effector frame name — confirm with:
#   print(kinematics_solver.get_all_frame_names())
END_EFFECTOR_FRAME = "right_gripper"   # or "panda_hand" — see printout below
 
# Replay speed
CONTROL_HZ = 30
 
# ──────────────────────────────────────────────
# Optional: rigid transform to align the Orca
# coordinate frame with Isaac world frame.
#
# If the replayed motion looks mirrored / rotated,
# apply a rotation here.  Identity = no correction.
# Example 90° rotation about Z:
#   FRAME_CORRECTION_QUAT_XYZW = [0, 0, sin(pi/4), cos(pi/4)]
# ──────────────────────────────────────────────
FRAME_CORRECTION_QUAT_XYZW = None   # set to np.array([x,y,z,w]) if needed
POSITION_OFFSET = np.zeros(3)       # add a translation offset if needed
 
 
# ══════════════════════════════════════════════
class FrankaIKReplay:
    """Drives the left Franka arm by replaying recorded EEF poses via IK."""
 
    def __init__(self, robot: SingleArticulation):
        self._robot = robot
        self._kinematics_solver: LulaKinematicsSolver | None = None
        self._articulation_solver: ArticulationKinematicsSolver | None = None
 
        # trajectory data
        self._positions: np.ndarray | None = None
        self._quaternions_xyzw: np.ndarray | None = None
        self._n_frames = 0
 
        self._frame = 0
        self._ik_fails = 0
 
    # ------------------------------------------------------------------
    def setup(self):
        mg_ext = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")
        cfg_dir = os.path.join(mg_ext, "motion_policy_configs")
 
        self._kinematics_solver = LulaKinematicsSolver(
            robot_description_path=os.path.join(cfg_dir, "franka/rmpflow/robot_descriptor.yaml"),
            urdf_path=os.path.join(cfg_dir, "franka/lula_franka_gen.urdf"),
        )
 
        print("\nValid EEF frame names:")
        print(self._kinematics_solver.get_all_frame_names())
 
        self._articulation_solver = ArticulationKinematicsSolver(
            self._robot, self._kinematics_solver, END_EFFECTOR_FRAME
        )
        print(f"\nUsing EEF frame: {self._articulation_solver.get_end_effector_frame()}\n")
 
    # ------------------------------------------------------------------
    def load_trajectory(self, h5_path: str):
        with h5py.File(h5_path, "r") as f:
            raw = np.array(f["observations/qpos_arm_left"])  # (N, 7)
 
        print(f"Loaded trajectory  shape={raw.shape}  dtype={raw.dtype}")
        print(f"Position range  x=[{raw[:,0].min():.3f}, {raw[:,0].max():.3f}]  "
              f"y=[{raw[:,1].min():.3f}, {raw[:,1].max():.3f}]  "
              f"z=[{raw[:,2].min():.3f}, {raw[:,2].max():.3f}]")
 
        # ── Interpret columns 0:3 as XYZ, 3:7 as w,x,y,z quaternion ──
        positions   = raw[:, 0:3] + POSITION_OFFSET   # (N, 3)
        quat_wxyz   = raw[:, 3:7]                     # (N, 4)  w-first
 
        # Convert to xyzw (what Lula expects)
        quat_xyzw = np.column_stack([quat_wxyz[:, 1],   # x
                                     quat_wxyz[:, 2],   # y
                                     quat_wxyz[:, 3],   # z
                                     quat_wxyz[:, 0]])  # w
 
        # ── Optional frame correction ──────────────────────────────────
        if FRAME_CORRECTION_QUAT_XYZW is not None:
            quat_xyzw = _batch_quat_multiply(
                np.broadcast_to(FRAME_CORRECTION_QUAT_XYZW, (len(quat_xyzw), 4)),
                quat_xyzw,
            )
 
        self._positions      = positions
        self._quaternions_xyzw = quat_xyzw
        self._n_frames       = len(positions)
 
        print(f"Replay duration @ {CONTROL_HZ} Hz: {self._n_frames / CONTROL_HZ:.1f} s\n")
 
    # ------------------------------------------------------------------
    def update(self, _step: float):
        """Called once per simulation step."""
        if self._frame >= self._n_frames:
            return False   # signal: done
 
        # Keep IK aware of any movement of the robot base
        base_t, base_r = self._robot.get_world_pose()
        self._kinematics_solver.set_robot_base_pose(base_t, base_r)
 
        pos       = self._positions[self._frame]
        quat_xyzw = self._quaternions_xyzw[self._frame]
 
        action, success = self._articulation_solver.compute_inverse_kinematics(
            target_position=pos,
            target_orientation=quat_xyzw,
        )
 
        if success:
            self._robot.apply_action(action)
        else:
            self._ik_fails += 1
            carb.log_warn(f"[frame {self._frame}] IK did not converge — holding pose")
 
        self._frame += 1
        return True   # still running
 
    # ------------------------------------------------------------------
    def reset(self):
        self._frame    = 0
        self._ik_fails = 0
 
 
# ══════════════════════════════════════════════
# Quaternion helpers
# ══════════════════════════════════════════════
def _batch_quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product  q1 ⊗ q2  for arrays of xyzw quaternions."""
    x1, y1, z1, w1 = q1[...,0], q1[...,1], q1[...,2], q1[...,3]
    x2, y2, z2, w2 = q2[...,0], q2[...,1], q2[...,2], q2[...,3]
    return np.stack([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], axis=-1)
 
 
# ══════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════
def main():
    # 1. Open your existing USD scene
    open_stage(SCENE_PATH)
 
    world = World()
 
    # 2. Register the robot with the World so world.reset() initialises it
    robot = world.scene.add(
        SingleArticulation(ROBOT_PRIM_PATH, name="franka_left")
    )
 
    world.reset()
    print("DOF names:", robot.dof_names)
 
    # 3. Build the IK replay controller
    controller = FrankaIKReplay(robot)
    controller.setup()
    controller.load_trajectory(H5_PATH)
 
    # 4. Simulation loop  –  NO time.sleep()!
    #    The physics timestep controls timing; sleeping here causes drift.
    steps_per_frame = max(1, round((1.0 / CONTROL_HZ) / world.get_physics_dt()))
    print(f"Physics dt={world.get_physics_dt():.4f}s  →  "
          f"advancing {steps_per_frame} physics step(s) per IK frame\n")
 
    world.play()
    step_count = 0
 
    while simulation_app.is_running():
        world.step(render=True)
 
        if not world.is_playing():
            continue
 
        # Only call the IK controller every N physics steps
        if step_count % steps_per_frame == 0:
            still_running = controller.update(world.get_physics_dt())
            if not still_running:
                print(f"\nReplay finished. "
                      f"IK failures: {controller._ik_fails}/{controller._n_frames}")
                break
 
        step_count += 1
 
    simulation_app.close()
 
 
if __name__ == "__main__":
    main()