from isaacsim.core.utils.stage import open_stage
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation, XFormPrim
from isaacsim.robot_motion.motion_generation import (
    LulaKinematicsSolver,
)
import isaacsim.robot_motion.motion_generation as _mg_pkg

import omni.usd
import h5py
import numpy as np
import time
from pathlib import Path

from pxr import UsdPhysics

# For debugging purposes
DEBUG = True
PLAY_HAND = False
PLAY_OBJECT = False

# Rx(180°) quaternion in wxyz — converts tool convention (Z-down) to URDF convention (Z-forward)
_Q_RX180 = np.array([0.0, 1.0, 0.0, 0.0])


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float64)


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)  ],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)  ],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
    ])


from src.visualization import (
    EEFVisualizer,
    VisConfig,
    COLOR_AXIS_X,
    COLOR_AXIS_Y,
    COLOR_AXIS_Z,
    COLOR_AXIS_X_FADED,
    COLOR_AXIS_Y_FADED,
    COLOR_AXIS_Z_FADED,
    ORIENT_LENGTH,
    FRAME_LINE_SIZE,
)

_MOTION_GEN_EXT = Path(_mg_pkg.__file__).parents[3]

PANDA_ARM_DESCRIPTION_PATH = str(
    _MOTION_GEN_EXT / "motion_policy_configs" / "franka" / "rmpflow" / "robot_descriptor.yaml"
)

PANDA_ARM_URDF_PATH = str(
    _MOTION_GEN_EXT / "motion_policy_configs" / "franka" / "lula_franka_gen.urdf"
)

# Translation vector flange to EEF
# The flange is the ring at the end of panda_link7 where the mount is attached using the 4 screw holes
T_flange_EEF = np.array([0.13,0,0.07])


class Simulator:
    def __init__(self, app, stage_path, h5_path):
        self.app = app
        self.stage_path = str(stage_path)
        self.h5_path = str(h5_path)

        missing = []
        for label, path in [
            ("Stage (USD scene)", self.stage_path),
            ("H5 dataset", self.h5_path),
            ("Franka robot descriptor", PANDA_ARM_DESCRIPTION_PATH),
            ("Franka LULA URDF", PANDA_ARM_URDF_PATH),
        ]:
            if not Path(path).exists():
                missing.append(f"  {label}: {path}")

        if missing:
            raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    def inspect(self):
        """
        Output the stage content of the scene
        """
        open_stage(self.stage_path)
        stage = omni.usd.get_context().get_stage()

        if stage is None:
            print("ERROR: Could not access the USD stage.")
            return

        print("===== STAGE CONTENT =====")
        for prim in stage.Traverse():
            print(prim.GetPath())

    def _print_articulation_info(self, articulation, label):
        """
        Prints DOF and their names
        """
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
        """
        Set joints to q_target
        """
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
    
    def play_one_hand(self):
        """
        Minimal play method for debug purposes
        """
        # ===== Load dataset =====
        with h5py.File(self.h5_path, "r") as f:
            left_hand_q  = np.array(f["observations/qpos_hand_left"], dtype=np.float32)

        dt = 1.0 # Set at your preference

        open_stage(self.stage_path)
        world = World()
        stage = omni.usd.get_context().get_stage()

        print("=== Articulation roots in stage ===")
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                print(prim.GetPath())

        # Deactivate everything except one ORCA candidate
        for path in [
            "/World/Franka_left",
            "/World/Franka_right",
            "/World/ORCA_right",
            "/World/object"
        ]:
            prim = stage.GetPrimAtPath(path)
            if prim and prim.IsValid():
                prim.SetActive(False)
                print(f"Deactivated {path}")

        hand_left_path = "/World/ORCA_left/scene_left/left_root/left_root"

        hand_left = world.scene.add(
            SingleArticulation(hand_left_path, name="test_art")
        )

        world.reset()
        world.play()

        for _ in range(5):
            world.step(render=True)

        self._print_articulation_info(hand_left, "TEST ARTICULATION")

        frame = 0
        while self.app.is_running() and frame < len(left_hand_q):
            q_hand_l = left_hand_q[frame] # Hand target
            if not np.isfinite(q_hand_l).all():
                print(f"[frame {frame}] left hand has NaN/inf:", q_hand_l)
                break
            self._safe_set_joints(hand_left, q_hand_l, "LEFT HAND")
            world.step(render=True)
            time.sleep(dt)
            frame += 1

    def play(self, sim_config=None, vis_config=None):
        """
        Play the entire simulation
        """
        set_joints = getattr(sim_config, "set_joints", True)
        enable_right = getattr(sim_config, "enable_right", True)
        enable_left = getattr(sim_config, "enable_left", True)

        open_stage(self.stage_path)
        world = World()

        stage = omni.usd.get_context().get_stage()

        if DEBUG:
            print("=== Articulation roots in stage ===")
            for prim in stage.Traverse():
                if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                    print(prim.GetPath())

        # ===== Franka robots from stage =====
        arm_right = world.scene.add(
            SingleArticulation("/World/fer_orcahand_right_extended", name="franka_right")
        )
        arm_left = world.scene.add(
            SingleArticulation("/World/fer_orcahand_left_extended", name="franka_left")
        )

        if PLAY_HAND:
            hand_right = world.scene.add(
                SingleArticulation("/World/ORCA_right/scene_right/right_root/right_root", name="orca_right")
            )

            hand_left = world.scene.add(
                SingleArticulation("/World/ORCA_left/scene_left/left_root/left_root", name="orca_left")
            )

        world.reset()

        if DEBUG:
            self._print_articulation_info(arm_right, "RIGHT ARM")
            self._print_articulation_info(arm_left, "LEFT ARM")
            if PLAY_HAND:
                self._print_articulation_info(hand_right, "RIGHT HAND")
                self._print_articulation_info(hand_left, "LEFT HAND")

        # ===== Arm base positions in world frame =====
        base_pos_r, _ = arm_right.get_world_pose()
        base_pos_l, _ = arm_left.get_world_pose()
        base_pos_r = np.asarray(base_pos_r, dtype=np.float64).flatten()
        base_pos_l = np.asarray(base_pos_l, dtype=np.float64).flatten()

        # TODO: find the correct prim path for fer_link8 (flange) in the combined USD.
        # fer_link8 is an empty link in the URDF so Isaac Sim may not create a USD prim for it.
        # Disabled for now — only needed when visualization is enabled.
        # eef_prim_r = XFormPrim("/World/fer_orcahand_right_extended/fer_link8")
        # eef_prim_l = XFormPrim("/World/fer_orcahand_left_extended/fer_link8")
        eef_prim_r = None
        eef_prim_l = None
        
        # ===== Lula IK solver setup =====
        kin_solver_r = LulaKinematicsSolver(
            robot_description_path=PANDA_ARM_DESCRIPTION_PATH,
            urdf_path=PANDA_ARM_URDF_PATH,
        )
        kin_solver_l = LulaKinematicsSolver(
            robot_description_path=PANDA_ARM_DESCRIPTION_PATH,
            urdf_path=PANDA_ARM_URDF_PATH,
        )

        _home_arm = np.array([0.0, -1.3, 0.0, -2.5, 0.0, 1.0, 0.0], dtype=np.float64)
        prev_arm_joints_r = _home_arm.copy()
        prev_arm_joints_l = _home_arm.copy()

        # ===== Load dataset =====
        with h5py.File(self.h5_path, "r") as f:
            right_arm_data = np.array(f["observations/qpos_arm_right"])
            left_arm_data = np.array(f["observations/qpos_arm_left"])

            right_hand_q = np.array(f["observations/qpos_hand_right"], dtype=np.float32)
            left_hand_q  = np.array(f["observations/qpos_hand_left"], dtype=np.float32)

        if DEBUG:
            print("\n===== H5 DATA =====")
            print("right_arm_data shape :", right_arm_data.shape)
            print("left_arm_data shape  :", left_arm_data.shape)
            print("right_hand_q shape   :", right_hand_q.shape)
            print("left_hand_q shape    :", left_hand_q.shape)

        # Arm data is EEF pose
        right_positions = right_arm_data[:, 0:3]
        left_positions = left_arm_data[:, 0:3]

        right_quaternions = right_arm_data[:, 3:7]
        left_quaternions = left_arm_data[:, 3:7]

        n_frames = min(len(right_positions), len(left_positions))

        if vis_config is None:
            vis_config = VisConfig(enabled=False)
        visualizer = EEFVisualizer() if vis_config.enabled else None

        if vis_config.eef_alpha is not None:
            _a = vis_config.eef_alpha
        elif vis_config.video_mode:
            _a = 0.15
        else:
            _a = COLOR_AXIS_X_FADED[3]

        cx_f = (COLOR_AXIS_X[0], COLOR_AXIS_X[1], COLOR_AXIS_X[2], _a)
        cy_f = (COLOR_AXIS_Y[0], COLOR_AXIS_Y[1], COLOR_AXIS_Y[2], _a)
        cz_f = (COLOR_AXIS_Z[0], COLOR_AXIS_Z[1], COLOR_AXIS_Z[2], _a)

        frame = 0
        ik_fail_r = 0
        ik_fail_l = 0

        CONTROL_HZ = 30
        dt = 1.0 / CONTROL_HZ

        world.play()

        for _ in range(10):
            world.step(render=True)

        _cam_eye = getattr(sim_config, "camera_eye", None)
        _cam_target = getattr(sim_config, "camera_target", None)

        if _cam_eye is not None and _cam_target is not None:
            from isaacsim.core.utils.viewports import set_camera_view

            set_camera_view(eye=np.array(_cam_eye), target=np.array(_cam_target))

        # Simulation loop
        while self.app.is_running() and frame < n_frames:
            # EEF position
            pos_r = np.asarray(right_positions[frame], dtype=np.float32)
            pos_l = np.asarray(left_positions[frame], dtype=np.float32)

            # H5 stores quaternions as wxyz; apply Rx(180°) to convert tool→URDF convention
            q_raw_r = np.asarray(right_quaternions[frame], dtype=np.float64)
            q_raw_l = np.asarray(left_quaternions[frame], dtype=np.float64)
            q_raw_r /= np.linalg.norm(q_raw_r)
            q_raw_l /= np.linalg.norm(q_raw_l)
            # World-frame wxyz after tool→URDF rotation (used for IK and visualization)
            quat_world_wxyz_r = _quat_mul(_Q_RX180, q_raw_r)
            quat_world_wxyz_l = _quat_mul(_Q_RX180, q_raw_l)
            # xyzw for visualization helpers that expect xyzw
            cur_quat_r = np.array([quat_world_wxyz_r[1], quat_world_wxyz_r[2], quat_world_wxyz_r[3], quat_world_wxyz_r[0]], dtype=np.float32)
            cur_quat_l = np.array([quat_world_wxyz_l[1], quat_world_wxyz_l[2], quat_world_wxyz_l[3], quat_world_wxyz_l[0]], dtype=np.float32)

            ARM_JOINT_INDICES = list(range(7))

            # ===== Hand targets =====
            if PLAY_HAND:
                q_hand_r = right_hand_q[frame]
                q_hand_l = left_hand_q[frame]

                if not np.isfinite(q_hand_r).all():
                    print(f"[frame {frame}] right hand has NaN/inf:", q_hand_r)
                    break

                if not np.isfinite(q_hand_l).all():
                    print(f"[frame {frame}] left hand has NaN/inf:", q_hand_l)
                    break

                ok_r = self._safe_set_joints(hand_right, q_hand_r, "RIGHT HAND")
                ok_l = self._safe_set_joints(hand_left, q_hand_l, "LEFT HAND")

            # Back-project EEF → flange: no rotation offset between flange and ORCA hand,
            # so flange orientation = EEF orientation; only position needs adjustment.
            R_r = _quat_to_rot(quat_world_wxyz_r)
            R_l = _quat_to_rot(quat_world_wxyz_l)
            flange_pos_r = pos_r.astype(np.float64) - R_r @ T_flange_EEF
            flange_pos_l = pos_l.astype(np.float64) - R_l @ T_flange_EEF

            # ===== Lula IK Solver =====
            if enable_right:
                arm_joints_r, ik_success_r = kin_solver_r.compute_inverse_kinematics(
                    frame_name="panda_hand",
                    target_position=flange_pos_r,
                    target_orientation=quat_world_wxyz_r,
                    warm_start=prev_arm_joints_r,
                )
                if set_joints:
                    if ik_success_r:
                        prev_arm_joints_r = arm_joints_r.copy()
                        arm_right.set_joint_positions(arm_joints_r.astype(np.float32), joint_indices=ARM_JOINT_INDICES)
                    else:
                        ik_fail_r += 1
                        arm_right.set_joint_positions(prev_arm_joints_r.astype(np.float32), joint_indices=ARM_JOINT_INDICES)
                        print(f"[frame {frame}] IK failed RIGHT")

            if enable_left:
                arm_joints_l, ik_success_l = kin_solver_l.compute_inverse_kinematics(
                    frame_name="panda_hand",
                    target_position=flange_pos_l,
                    target_orientation=quat_world_wxyz_l,
                    warm_start=prev_arm_joints_l,
                )
                if set_joints:
                    if ik_success_l:
                        prev_arm_joints_l = arm_joints_l.copy()
                        arm_left.set_joint_positions(arm_joints_l.astype(np.float32), joint_indices=ARM_JOINT_INDICES)
                    else:
                        ik_fail_l += 1
                        arm_left.set_joint_positions(prev_arm_joints_l.astype(np.float32), joint_indices=ARM_JOINT_INDICES)
                        print(f"[frame {frame}] IK failed LEFT")

            VIZ_OFFSET = np.array([0.0, 0.0, 1.0], dtype=np.float32)

            if visualizer is not None:
                if enable_right:
                    act_pos_r_b, act_quat_wxyz_r_b = eef_prim_r.get_world_poses()
                    act_quat_r = np.array(
                        [
                            act_quat_wxyz_r_b[0, 1],
                            act_quat_wxyz_r_b[0, 2],
                            act_quat_wxyz_r_b[0, 3],
                            act_quat_wxyz_r_b[0, 0],
                        ],
                        dtype=np.float32,
                    )
                    if vis_config.show_eef:
                        visualizer.draw_frame(
                            pos_r + base_pos_r,
                            cur_quat_r,
                            COLOR_AXIS_X,
                            COLOR_AXIS_Y,
                            COLOR_AXIS_Z,
                        )
                        visualizer.draw_frame(act_pos_r_b[0], act_quat_r, cx_f, cy_f, cz_f)

                    if vis_config.show_offset:
                        visualizer.draw_frame(
                            pos_r + base_pos_r + VIZ_OFFSET,
                            cur_quat_r,
                            COLOR_AXIS_X,
                            COLOR_AXIS_Y,
                            COLOR_AXIS_Z,
                            length=ORIENT_LENGTH * 0.5,
                            width=FRAME_LINE_SIZE * 2,
                        )
                        visualizer.draw_frame(act_pos_r_b[0] + VIZ_OFFSET, act_quat_r, cx_f, cy_f, cz_f)

                if enable_left:
                    act_pos_l_b, act_quat_wxyz_l_b = eef_prim_l.get_world_poses()
                    act_quat_l = np.array(
                        [
                            act_quat_wxyz_l_b[0, 1],
                            act_quat_wxyz_l_b[0, 2],
                            act_quat_wxyz_l_b[0, 3],
                            act_quat_wxyz_l_b[0, 0],
                        ],
                        dtype=np.float32,
                    )
                    if vis_config.show_eef:
                        visualizer.draw_frame(
                            pos_l + base_pos_l,
                            cur_quat_l,
                            COLOR_AXIS_X,
                            COLOR_AXIS_Y,
                            COLOR_AXIS_Z,
                        )
                        visualizer.draw_frame(act_pos_l_b[0], act_quat_l, cx_f, cy_f, cz_f)

                    if vis_config.show_offset:
                        visualizer.draw_frame(
                            pos_l + base_pos_l + VIZ_OFFSET,
                            cur_quat_l,
                            COLOR_AXIS_X,
                            COLOR_AXIS_Y,
                            COLOR_AXIS_Z,
                            length=ORIENT_LENGTH * 0.5,
                            width=FRAME_LINE_SIZE * 2,
                        )
                        visualizer.draw_frame(act_pos_l_b[0] + VIZ_OFFSET, act_quat_l, cx_f, cy_f, cz_f)

            if DEBUG and frame % 100 == 0:
                print(f"\nframe {frame}/{n_frames}")
                if enable_right:
                    print("  pos_r         :", pos_r)
                    print("  cur_quat_r    :", cur_quat_r)
                if enable_left:
                    print("  pos_l         :", pos_l)
                    print("  cur_quat_l    :", cur_quat_l)
                print(f"  IK fails      : right={ik_fail_r}, left={ik_fail_l}")

                if visualizer is not None:
                    if enable_right:
                        align_r = abs(float(np.dot(act_quat_r, cur_quat_r)))
                        print(f"  EEF align r   : {align_r:.4f}  (1.0=perfect)")
                    if enable_left:
                        align_l = abs(float(np.dot(act_quat_l, cur_quat_l)))
                        print(f"  EEF align l   : {align_l:.4f}  (1.0=perfect)")

            world.step(render=True)
            time.sleep(dt)
            frame += 1

        print("\nReplay finished.")
        print(f"IK failures right: {ik_fail_r}/{n_frames}")
        print(f"IK failures left : {ik_fail_l}/{n_frames}")

        while self.app.is_running():
            world.step(render=True)
