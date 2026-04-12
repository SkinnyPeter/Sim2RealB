from isaacsim.core.utils.stage import open_stage
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation, XFormPrim
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

from src.visualization import (
    EEFVisualizer,
    COLOR_AXIS_X, COLOR_AXIS_Y, COLOR_AXIS_Z,
    COLOR_AXIS_X_FADED, COLOR_AXIS_Y_FADED, COLOR_AXIS_Z_FADED,
    ORIENT_LENGTH, FRAME_LINE_SIZE,
)

import isaacsim.robot_motion.motion_generation as _mg_pkg
_MOTION_GEN_EXT = Path(_mg_pkg.__file__).parents[3]  # .../isaacsim.robot_motion.motion_generation/

PANDA_ARM_DESCRIPTION_PATH = str(
    _MOTION_GEN_EXT / "motion_policy_configs" / "franka" / "rmpflow" / "robot_descriptor.yaml"
)

PANDA_ARM_URDF_PATH = str(
    _MOTION_GEN_EXT / "motion_policy_configs" / "franka" / "lula_franka_gen.urdf"
)


class Simulator:
    def __init__(self, app, stage_path, h5_path):
        self.app = app
        self.stage_path = str(stage_path)
        self.h5_path = str(h5_path)

        missing = []
        for label, path in [
            ("Stage (USD scene)",       self.stage_path),
            ("H5 dataset",              self.h5_path),
            ("Franka robot descriptor", PANDA_ARM_DESCRIPTION_PATH),
            ("Franka LULA URDF",        PANDA_ARM_URDF_PATH),
        ]:
            if not Path(path).exists():
                missing.append(f"  {label}: {path}")
        if missing:
            raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

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

    def play(self, visualize_eef=False, set_joints=True, enable_right=True, enable_left=True):
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

        world.reset()

        # ===== Arm base positions in world frame =====
        base_pos_r, _ = arm_right.get_world_pose()
        base_pos_l, _ = arm_left.get_world_pose()

        # ===== EEF prim handles for actual pose readback =====
        eef_prim_r = XFormPrim("/World/Franka_right/panda_hand")
        eef_prim_l = XFormPrim("/World/Franka_left/panda_hand")

        # ===== Print DOF order once =====
        self._print_articulation_info(arm_right, "RIGHT ARM")
        self._print_articulation_info(arm_left, "LEFT ARM")

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

        print("\n===== H5 DATA =====")
        print("right_arm_data shape :", right_arm_data.shape)
        print("left_arm_data shape  :", left_arm_data.shape)

        # Arm data is EEF pose: [px, py, pz, qx, qy, qz, qw]
        right_positions = right_arm_data[:, 0:3]
        left_positions = left_arm_data[:, 0:3]

        right_quaternions = right_arm_data[:, 3:7]
        left_quaternions = left_arm_data[:, 3:7]

        n_frames = min(len(right_positions), len(left_positions))

        visualizer = EEFVisualizer() if visualize_eef else None

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

            # H5 stores quaternions as wxyz; Lula/Isaac Sim expects xyzw — normalize on read
            q_raw_r  = np.asarray(right_quaternions[frame], dtype=np.float32)
            q_raw_l  = np.asarray(left_quaternions[frame],  dtype=np.float32)
            quat_r   = np.array([q_raw_r[1], q_raw_r[2], q_raw_r[3], q_raw_r[0]], dtype=np.float32)
            quat_r  /= np.linalg.norm(quat_r)
            quat_l   = np.array([q_raw_l[1], q_raw_l[2], q_raw_l[3], q_raw_l[0]], dtype=np.float32)
            quat_l  /= np.linalg.norm(quat_l)

            # Other variants (uncomment and assign to cur_quat to use):
            quat_r_flip = np.array([ quat_r[3],  quat_r[2], -quat_r[1], -quat_r[0]], dtype=np.float32)  # 180° flip around local X
            quat_l_flip = np.array([ quat_l[3],  quat_l[2], -quat_l[1], -quat_l[0]], dtype=np.float32)

            # Active quaternion — change to any variant above
            cur_quat_r = quat_r_flip
            cur_quat_l = quat_l_flip

            # ===== Compute arm IK =====
            # ArticulationKinematicsSolver expects world-frame positions and handles
            # the robot base transform internally — pass raw positions only.
            ARM_JOINT_INDICES = list(range(7))

            if enable_right:
                joint_action_r, ik_success_r = articulation_solver_r.compute_inverse_kinematics(
                    target_position=pos_r,
                    target_orientation=cur_quat_r,
                )
                if set_joints:
                    if ik_success_r:
                        arm_right.set_joint_positions(np.asarray(joint_action_r.joint_positions, dtype=np.float32), joint_indices=ARM_JOINT_INDICES)
                    else:
                        ik_fail_r += 1
                        print(f"[frame {frame}] IK failed RIGHT")

            if enable_left:
                joint_action_l, ik_success_l = articulation_solver_l.compute_inverse_kinematics(
                    target_position=pos_l,
                    target_orientation=cur_quat_l,
                )
                if set_joints:
                    if ik_success_l:
                        arm_left.set_joint_positions(np.asarray(joint_action_l.joint_positions, dtype=np.float32), joint_indices=ARM_JOINT_INDICES)
                    else:
                        ik_fail_l += 1
                        print(f"[frame {frame}] IK failed LEFT")

            # ===== EEF visualization =====
            VIZ_OFFSET = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # 1 m up for unobstructed inspection

            if visualizer is not None:
                if enable_right:
                    # Ground truth target frame — vivid
                    visualizer.draw_frame(pos_r + base_pos_r, cur_quat_r, COLOR_AXIS_X, COLOR_AXIS_Y, COLOR_AXIS_Z)
                    # Actual robot EEF frame — faded
                    act_pos_r_b, act_quat_wxyz_r_b = eef_prim_r.get_world_poses()
                    act_quat_r = np.array([act_quat_wxyz_r_b[0,1], act_quat_wxyz_r_b[0,2], act_quat_wxyz_r_b[0,3], act_quat_wxyz_r_b[0,0]], dtype=np.float32)
                    visualizer.draw_frame(act_pos_r_b[0], act_quat_r, COLOR_AXIS_X_FADED, COLOR_AXIS_Y_FADED, COLOR_AXIS_Z_FADED)
                    # Offset copies — same orientations, lifted 1 m; ground truth: half length, double width
                    visualizer.draw_frame(pos_r + base_pos_r + VIZ_OFFSET, cur_quat_r, COLOR_AXIS_X, COLOR_AXIS_Y, COLOR_AXIS_Z, length=ORIENT_LENGTH * 0.5, width=FRAME_LINE_SIZE * 2)
                    visualizer.draw_frame(act_pos_r_b[0] + VIZ_OFFSET, act_quat_r, COLOR_AXIS_X_FADED, COLOR_AXIS_Y_FADED, COLOR_AXIS_Z_FADED)
                if enable_left:
                    # Ground truth target frame — vivid
                    visualizer.draw_frame(pos_l + base_pos_l, cur_quat_l, COLOR_AXIS_X, COLOR_AXIS_Y, COLOR_AXIS_Z)
                    # Actual robot EEF frame — faded
                    act_pos_l_b, act_quat_wxyz_l_b = eef_prim_l.get_world_poses()
                    act_quat_l = np.array([act_quat_wxyz_l_b[0,1], act_quat_wxyz_l_b[0,2], act_quat_wxyz_l_b[0,3], act_quat_wxyz_l_b[0,0]], dtype=np.float32)
                    visualizer.draw_frame(act_pos_l_b[0], act_quat_l, COLOR_AXIS_X_FADED, COLOR_AXIS_Y_FADED, COLOR_AXIS_Z_FADED)
                    # Offset copies — same orientations, lifted 1 m; ground truth: half length, double width
                    visualizer.draw_frame(pos_l + base_pos_l + VIZ_OFFSET, cur_quat_l, COLOR_AXIS_X, COLOR_AXIS_Y, COLOR_AXIS_Z, length=ORIENT_LENGTH * 0.5, width=FRAME_LINE_SIZE * 2)
                    visualizer.draw_frame(act_pos_l_b[0] + VIZ_OFFSET, act_quat_l, COLOR_AXIS_X_FADED, COLOR_AXIS_Y_FADED, COLOR_AXIS_Z_FADED)

            if frame % 100 == 0:
                print(f"\nframe {frame}/{n_frames}")
                if enable_right:
                    print("  pos_r         :", pos_r)
                    print("  cur_quat_r    :", cur_quat_r)
                if enable_left:
                    print("  pos_l         :", pos_l)
                    print("  cur_quat_l    :", cur_quat_l)
                print(f"  IK fails    : right={ik_fail_r}, left={ik_fail_l}")
                if visualizer is not None:
                    # |dot| = 1.0 means perfect alignment, 0.0 means 90° off
                    if enable_right:
                        align_r = abs(float(np.dot(act_quat_r, cur_quat_r)))
                        print(f"  EEF align r : {align_r:.4f}  (1.0=perfect)")
                    if enable_left:
                        align_l = abs(float(np.dot(act_quat_l, cur_quat_l)))
                        print(f"  EEF align l : {align_l:.4f}  (1.0=perfect)")

            world.step(render=True)
            time.sleep(dt)
            frame += 1

        print("\nReplay finished.")
        print(f"IK failures right: {ik_fail_r}/{n_frames}")
        print(f"IK failures left : {ik_fail_l}/{n_frames}")

        while self.app.is_running():
            world.step(render=True)