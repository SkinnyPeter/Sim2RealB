from isaacsim.core.utils.stage import open_stage
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver

import omni.usd
import h5py
import numpy as np
import time

PANDA_ARM_DESCRIPTION_PATH = r"/home/teamb/isaac-sim/exts/isaacsim.robot_motion.motion_generation/motion_policy_configs/franka/rmpflow/robot_descriptor.yaml"
PANDA_ARM_URDF_PATH              = r"/home/teamb/isaac-sim/exts/isaacsim.robot_motion.motion_generation/motion_policy_configs/franka/lula_franka_gen.urdf"

class Simulator:
    """
    """
    def __init__(self, app, stage_path, h5_path):
        self.app = app
        self.stage_path = stage_path
        self.h5_path = h5_path
    
    def inspect(self):
        """
        Inspect the scene structure

        Print the stage content
        """
        open_stage(self.stage_path)

        stage = omni.usd.get_context().get_stage()

        if stage is None:
            print("ERROR: Could not access the USD stage.")
            return

        print("===== STAGE CONTENT =====")
        for prim in stage.Traverse():
            print(prim.GetPath())

    def play(self):
        """
        Replay the whole HDF5 trajectory using:
        - EEF position
        - EEF orientation
        - IK + set_joint_positions (stable mode)
        """

        open_stage(self.stage_path)

        world = World()

        # Add robots from stage
        arm_right = world.scene.add(
            SingleArticulation("/World/Franka_right", name="franka_right")
        )
        arm_left = world.scene.add(
            SingleArticulation("/World/Franka_left", name="franka_left")
        )

        world.reset()

        # Kinematics solvers
        kin_solver_r = LulaKinematicsSolver(
            robot_description_path=PANDA_ARM_DESCRIPTION_PATH,
            urdf_path=PANDA_ARM_URDF_PATH,
        )
        kin_solver_l = LulaKinematicsSolver(
            robot_description_path=PANDA_ARM_DESCRIPTION_PATH,
            urdf_path=PANDA_ARM_URDF_PATH,
        )

        articulation_solver_r = ArticulationKinematicsSolver(
            arm_right, kin_solver_r, end_effector_frame_name="panda_hand"
        )
        articulation_solver_l = ArticulationKinematicsSolver(
            arm_left, kin_solver_l, end_effector_frame_name="panda_hand"
        )

        print("EEF frame right:", articulation_solver_r.get_end_effector_frame())
        print("EEF frame left :", articulation_solver_l.get_end_effector_frame())

        # Load dataset
        with h5py.File(self.h5_path, "r") as f:
            right_eef_trajectory = np.array(f["observations/qpos_arm_right"])
            left_eef_trajectory = np.array(f["observations/qpos_arm_left"])

        # Positions
        right_positions = right_eef_trajectory[:, 0:3]
        left_positions = left_eef_trajectory[:, 0:3]

        # Quaternions
        right_quaternions = right_eef_trajectory[:, 3:7]
        left_quaternions = left_eef_trajectory[:, 3:7]

        n_frames = min(len(right_positions), len(left_positions))
        frame = 0

        ik_fail_r = 0
        ik_fail_l = 0

        CONTROL_HZ = 30
        dt = 1.0 / CONTROL_HZ

        world.play()

        # Let Isaac stabilize
        for _ in range(10):
            world.step(render=True)

        while self.app.is_running() and frame < n_frames:

            pos_r = right_positions[frame]
            pos_l = left_positions[frame]

            quat_r_h5 = right_quaternions[frame]
            quat_l_h5 = left_quaternions[frame]

            # Convert quaternion [w,x,y,z] → [x,y,z,w]
            quat_r = np.array(quat_r_h5, dtype=np.float32)
            quat_l = np.array(quat_l_h5, dtype=np.float32)

            joint_action_r, ik_success_r = articulation_solver_r.compute_inverse_kinematics(
                target_position=pos_r,
                target_orientation=quat_r,
            )

            joint_action_l, ik_success_l = articulation_solver_l.compute_inverse_kinematics(
                target_position=pos_l,
                target_orientation=quat_l,
            )

            if ik_success_r:
                q_r = np.array(joint_action_r.joint_positions, dtype=np.float32)
                arm_right.set_joint_positions(q_r)
            else:
                ik_fail_r += 1
                print(f"[frame {frame}] IK failed RIGHT")

            if ik_success_l:
                q_l = np.array(joint_action_l.joint_positions, dtype=np.float32)
                arm_left.set_joint_positions(q_l)
            else:
                ik_fail_l += 1
                print(f"[frame {frame}] IK failed LEFT")

            if frame % 100 == 0:
                print(f"frame {frame}/{n_frames}")
                print("  pos_r :", pos_r)
                print("  quat_r:", quat_r)
                print("  pos_l :", pos_l)
                print("  quat_l:", quat_l)
                print(f"  IK fails right={ik_fail_r}, left={ik_fail_l}")

            world.step(render=True)
            time.sleep(dt)

            frame += 1

        print("Replay finished.")
        print(f"IK failures right: {ik_fail_r}/{n_frames}")
        print(f"IK failures left : {ik_fail_l}/{n_frames}")

        while self.app.is_running():
            world.step(render=True)