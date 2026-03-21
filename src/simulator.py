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
        Play the sequence from a h5 file
        """
        CONTROL_HZ = 30  # adjust to 31 if replay feels slightly fast
        dt = 1.0 / CONTROL_HZ

        open_stage(self.stage_path)

        world = World()

        # Localizes the elements in the stage
        arm_right = world.scene.add(
            SingleArticulation("/World/Franka_right", name="franka_right")
        )
        arm_left = world.scene.add(
            SingleArticulation("/World/Franka_left", name="franka_left")
        )

        world.reset()  # robot.initialize() is called automatically

        # -------------------
        # Setup Kynematics Solver
        # -------------------
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

        print("EEF frame:", articulation_solver_r.get_end_effector_frame())
        print("EEF frame:", articulation_solver_l.get_end_effector_frame())

        # -------------------
        # Load dataset
        # -------------------
        with h5py.File(self.h5_path, "r") as f:
            right_eef_trajectory = np.array(f["observations/qpos_arm_right"])  # (N, 7)
            left_eef_trajectory = np.array(f["observations/qpos_arm_left"])  # (N, 7)

        right_positions   = right_eef_trajectory[:, 0:3]  # xyz
        right_quaternions = right_eef_trajectory[:, 3:7]  # wxyz (confirmed) — conversion to xyzw applied per frame below

        left_positions   = left_eef_trajectory[:, 0:3]  # xyz
        left_quaternions = left_eef_trajectory[:, 3:7]  # wxyz (confirmed) — conversion to xyzw applied per frame below

        # -------------------
        # Replay
        # -------------------
        frame    = 0
        ik_fails_r = 0
        ik_fails_l = 0
        world.play()

        while self.app.is_running():
            world.step(render=True)

            if not world.is_playing():
                continue

            if frame >= len(right_eef_trajectory):
                print(f"Replay finished. IK failures on right arm trajectory: {ik_fails_r}/{len(right_eef_trajectory)}")
                break

            if frame >= len(left_eef_trajectory):
                print(f"Replay finished. IK failures on left arm trajectory: {ik_fails_l}/{len(left_eef_trajectory)}")
                break

            pos_r       = right_positions[frame]   # (3,) xyz
            quat_wxyz_r = right_quaternions[frame]     # (4,) wxyz (confirmed)
            quat_xyzw_r = np.array([quat_wxyz_r[1], quat_wxyz_r[2], quat_wxyz_r[3], quat_wxyz_r[0]]) # → xyzw for Lula
            
            pos_l       = left_positions[frame]   # (3,) xyz
            quat_wxyz_l = left_quaternions[frame]                                                   # (4,) wxyz (confirmed)
            quat_xyzw_l = np.array([quat_wxyz_l[1], quat_wxyz_l[2], quat_wxyz_l[3], quat_wxyz_l[0]]) # → xyzw for Lula

            joint_action_r, ik_success_r = articulation_solver_r.compute_inverse_kinematics(
                target_position=pos_r,
                target_orientation=quat_xyzw_r,
            )

            joint_action_l, ik_success_l = articulation_solver_l.compute_inverse_kinematics(
                target_position=pos_l,
                target_orientation=quat_xyzw_l,
            )

            if ik_success_r:
                q_r = arm_right.get_joint_positions()      # (9,) — preserves current finger state
                q_r[:7] = joint_action_r.joint_positions # overwrite arm joints from IK
                q_r[7:] = 0.04                         # both fingers open
                arm_right.set_joint_positions(q_r)
            else:
                ik_fails_r += 1
                print(f"[frame {frame}] IK failed — holding previous pose")

            if ik_success_l:
                q_l = arm_left.get_joint_positions()      # (9,) — preserves current finger state
                q_l[:7] = joint_action_l.joint_positions # overwrite arm joints from IK
                q_l[7:] = 0.04                         # both fingers open
                arm_left.set_joint_positions(q_l)
            else:
                ik_fails_l += 1
                print(f"[frame {frame}] IK failed — holding previous pose")

            frame += 1
            time.sleep(dt)  # throttle to CONTROL_HZ

        self.app.close()