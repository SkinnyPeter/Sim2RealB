from isaacsim.core.utils.stage import open_stage
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver,
    LulaKinematicsSolver,
)

import omni.usd
import h5py
import numpy as np
import time
from pathlib import Path

from pxr import UsdPhysics

ISAACSIM_ROOT = Path.home() / "isaac-sim"

PANDA_ARM_DESCRIPTION_PATH = str(
    ISAACSIM_ROOT
    / "exts"
    / "isaacsim.robot_motion.motion_generation"
    / "motion_policy_configs"
    / "franka"
    / "rmpflow"
    / "robot_descriptor.yaml"
)

PANDA_ARM_URDF_PATH = str(
    ISAACSIM_ROOT
    / "exts"
    / "isaacsim.robot_motion.motion_generation"
    / "motion_policy_configs"
    / "franka"
    / "lula_franka_gen.urdf"
)


class Simulator:
    def __init__(self, app, stage_path, h5_path):
        self.app = app
        self.stage_path = str(stage_path)
        self.h5_path = str(h5_path)

    def inspect(self):
        open_stage(self.stage_path)
        stage = omni.usd.get_context().get_stage()

        if stage is None:
            print("ERROR: Could not access the USD stage.")
            return

        print("===== STAGE CONTENT =====")
        for prim in stage.Traverse():
            print(prim.GetPath())

    def _print_articulation_info(self, articulation, label):
        try:
            dof_names = articulation.dof_names
        except Exception:
            dof_names = articulation.get_dof_names()

        print(f"\n===== {label} =====")
        print(f"Number of DOFs: {len(dof_names)}")
        print("DOF names:")
        for i, name in enumerate(dof_names):
            print(f"  [{i:02d}] {name}")

    def _safe_set_joints(self, articulation, q_target, label):
        q_target = np.asarray(q_target, dtype=np.float32).reshape(-1)

        try:
            n_dofs = articulation.num_dof
        except Exception:
            n_dofs = len(articulation.dof_names)

        if q_target.shape[0] != n_dofs:
            print(
                f"[WARN] {label}: target size {q_target.shape[0]} does not match articulation DOFs {n_dofs}"
            )
            return False

        articulation.set_joint_positions(q_target)
        return True

    def play(self):
        open_stage(self.stage_path)
        world = World()

        ####
        stage = omni.usd.get_context().get_stage()

        print("=== Articulation roots in stage ===")
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                print(prim.GetPath())
        ####

        # ===== Robots from stage =====
        arm_right = world.scene.add(
            SingleArticulation("/World/Franka_right", name="franka_right")
        )
        arm_left = world.scene.add(
            SingleArticulation("/World/Franka_left", name="franka_left")
        )

        hand_right = world.scene.add(
            SingleArticulation("/World/scene_combined/right_tower/right_tower", name="orca_right")
        )

        hand_left = world.scene.add(
            SingleArticulation("/World/scene_combined/left_tower/left_tower", name="orca_left")
        )

        world.reset()

        # ===== Print DOF order once =====
        self._print_articulation_info(arm_right, "RIGHT ARM")
        self._print_articulation_info(arm_left, "LEFT ARM")
        self._print_articulation_info(hand_right, "RIGHT HAND")
        self._print_articulation_info(hand_left, "LEFT HAND")

        # ===== IK solvers for arms =====
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

        # ===== Load dataset =====
        with h5py.File(self.h5_path, "r") as f:
            right_arm_data = np.array(f["observations/qpos_arm_right"])
            left_arm_data = np.array(f["observations/qpos_arm_left"])

            right_hand_q = np.array(f["observations/qpos_hand_right"], dtype=np.float32)
            left_hand_q  = np.array(f["observations/qpos_hand_left"], dtype=np.float32)

        print("\n===== H5 DATA =====")
        print("right_arm_data shape :", right_arm_data.shape)
        print("left_arm_data shape  :", left_arm_data.shape)
        print("right_hand_q shape   :", right_hand_q.shape)
        print("left_hand_q shape    :", left_hand_q.shape)

        # Assuming arm data is [px, py, pz, qx, qy, qz, qw] or similar pose representation.
        # Verify this. If this is actually joint-space, do NOT use IK.
        right_positions = right_arm_data[:, 0:3]
        left_positions = left_arm_data[:, 0:3]

        right_quaternions = right_arm_data[:, 3:7]
        left_quaternions = left_arm_data[:, 3:7]

        n_frames = min(
            len(right_positions),
            len(left_positions),
            len(right_hand_q),
            len(left_hand_q),
        )

        frame = 0
        ik_fail_r = 0
        ik_fail_l = 0

        CONTROL_HZ = 30
        dt = 1.0 / CONTROL_HZ

        world.play()

        for _ in range(10):
            world.step(render=True)

        while self.app.is_running() and frame < n_frames:
            # ===== Arm targets =====
            pos_r = np.asarray(right_positions[frame], dtype=np.float32)
            pos_l = np.asarray(left_positions[frame], dtype=np.float32)

            quat_r = np.asarray(right_quaternions[frame], dtype=np.float32)
            quat_l = np.asarray(left_quaternions[frame], dtype=np.float32)

            # If your H5 quaternions are stored as [w, x, y, z], convert them:
            # quat_r = np.array([quat_r[1], quat_r[2], quat_r[3], quat_r[0]], dtype=np.float32)
            # quat_l = np.array([quat_l[1], quat_l[2], quat_l[3], quat_l[0]], dtype=np.float32)

            # Normalize for safety
            if np.linalg.norm(quat_r) > 1e-8:
                quat_r = quat_r / np.linalg.norm(quat_r)
            if np.linalg.norm(quat_l) > 1e-8:
                quat_l = quat_l / np.linalg.norm(quat_l)

            # ===== Hand targets =====
            q_hand_r = right_hand_q[frame]
            q_hand_l = left_hand_q[frame]

            if not np.isfinite(q_hand_r).all():
                print(f"[frame {frame}] right hand has NaN/inf:", q_hand_r)
                break

            if not np.isfinite(q_hand_l).all():
                print(f"[frame {frame}] left hand has NaN/inf:", q_hand_l)
                break

            hand_right.set_joint_positions(q_hand_r)
            hand_left.set_joint_positions(q_hand_l)

            # ===== Compute arm IK =====
            joint_action_r, ik_success_r = articulation_solver_r.compute_inverse_kinematics(
                target_position=pos_r,
                target_orientation=quat_r,
            )
            joint_action_l, ik_success_l = articulation_solver_l.compute_inverse_kinematics(
                target_position=pos_l,
                target_orientation=quat_l,
            )

            # ===== Apply arm joints =====
            if ik_success_r:
                q_r = np.asarray(joint_action_r.joint_positions, dtype=np.float32)
                arm_right.set_joint_positions(q_r)
            else:
                ik_fail_r += 1
                print(f"[frame {frame}] IK failed RIGHT")

            if ik_success_l:
                q_l = np.asarray(joint_action_l.joint_positions, dtype=np.float32)
                arm_left.set_joint_positions(q_l)
            else:
                ik_fail_l += 1
                print(f"[frame {frame}] IK failed LEFT")

            # ===== Apply hand joints directly =====
            ok_r = self._safe_set_joints(hand_right, q_hand_r, "RIGHT HAND")
            ok_l = self._safe_set_joints(hand_left, q_hand_l, "LEFT HAND")

            if frame % 100 == 0:
                print(f"\nframe {frame}/{n_frames}")
                print("  pos_r       :", pos_r)
                print("  quat_r      :", quat_r)
                print("  pos_l       :", pos_l)
                print("  quat_l      :", quat_l)
                print("  hand_q_r[0:5]:", q_hand_r[:5])
                print("  hand_q_l[0:5]:", q_hand_l[:5])
                print(f"  hand set ok : right={ok_r}, left={ok_l}")
                print(f"  IK fails    : right={ik_fail_r}, left={ik_fail_l}")

            world.step(render=True)
            time.sleep(dt)
            frame += 1

        print("\nReplay finished.")
        print(f"IK failures right: {ik_fail_r}/{n_frames}")
        print(f"IK failures left : {ik_fail_l}/{n_frames}")

        while self.app.is_running():
            world.step(render=True)