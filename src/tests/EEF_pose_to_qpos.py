import time
import h5py
import numpy as np
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.utils.stage import open_stage
from isaacsim.core.prims import SingleArticulation
from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver

SCENE_PATH             = r"/home/teamb/Desktop/Sim2RealB/scenes/scene.usd"
H5_PATH                = r"/home/teamb/Desktop/Sim2RealB/data/20250827_151212.h5"
ROBOT_DESCRIPTION_PATH = r"/home/teamb/isaac-sim/exts/isaacsim.robot_motion.motion_generation/motion_policy_configs/franka/rmpflow/robot_descriptor.yaml"
URDF_PATH              = r"/home/teamb/isaac-sim/exts/isaacsim.robot_motion.motion_generation/motion_policy_configs/franka/lula_franka_gen.urdf"

CONTROL_HZ = 30  # adjust to 31 if replay feels slightly fast
dt         = 1.0 / CONTROL_HZ

# -------------------
# setup
# -------------------
open_stage(SCENE_PATH)
world = World()

# Add robot through the scene so world.reset() manages its lifecycle
robot = world.scene.add(
    SingleArticulation("/World/Franka", name="franka")
)

world.reset()  # robot.initialize() is called automatically
print("DOF names:", robot.dof_names)

# -------------------
# setup IK solver
# -------------------
kinematics_solver = LulaKinematicsSolver(
    robot_description_path=ROBOT_DESCRIPTION_PATH,
    urdf_path=URDF_PATH,
)
articulation_solver = ArticulationKinematicsSolver(
    robot,
    kinematics_solver,
    end_effector_frame_name="panda_hand",
)
print("EEF frame:", articulation_solver.get_end_effector_frame())

# -------------------
# load dataset
# -------------------
with h5py.File(H5_PATH, "r") as f:
    eef_data = np.array(f["observations/qpos_arm_left"])  # (N, 7)

print("Trajectory shape:", eef_data.shape)
print(f"Replay duration at {CONTROL_HZ} Hz: {len(eef_data) / CONTROL_HZ:.1f} s")

positions   = eef_data[:, 0:3]  # xyz
quaternions = eef_data[:, 3:7]  # wxyz (confirmed) — conversion to xyzw applied per frame below

# -------------------
# replay
# -------------------
frame    = 0
ik_fails = 0
world.play()

while simulation_app.is_running():
    world.step(render=True)

    if not world.is_playing():
        continue

    if frame >= len(eef_data):
        print(f"Replay finished. IK failures: {ik_fails}/{len(eef_data)}")
        break

    pos       = positions[frame]                                                     # (3,) xyz
    quat_wxyz = quaternions[frame]                                                   # (4,) wxyz (confirmed)
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]) # → xyzw for Lula

    joint_action, ik_success = articulation_solver.compute_inverse_kinematics(
        target_position=pos,
        target_orientation=quat_xyzw,
    )

    if ik_success:
        q = robot.get_joint_positions()      # (9,) — preserves current finger state
        q[:7] = joint_action.joint_positions # overwrite arm joints from IK
        q[7:] = 0.04                         # both fingers open
        robot.set_joint_positions(q)
    else:
        ik_fails += 1
        print(f"[frame {frame}] IK failed — holding previous pose")

    frame += 1
    time.sleep(dt)  # throttle to CONTROL_HZ

simulation_app.close()