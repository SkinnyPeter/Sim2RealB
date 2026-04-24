from isaacsim.core.utils.stage import open_stage
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation, XFormPrim

import omni.usd
import h5py
import numpy as np
import time
from pathlib import Path
from scipy.spatial.transform import Rotation

from src.visualization import (
    EEFVisualizer,
    VisConfig,
    COLOR_AXIS_X,
    COLOR_AXIS_Y,
    COLOR_AXIS_Z,
    COLOR_AXIS_X_FADED,
    ORIENT_LENGTH,
    FRAME_LINE_SIZE,
)

# Import quaternions utility functions
from src.simulator.quat_utils import (
    normalize_quat_wxyz,
    tool_quat_to_urdf,
    detect_quaternion_order
)
from src.simulator.IK_solver import FrankaIKController
from src.data.h5_loader import load_replay_h5

# Camera-to-world transforms (T_base_in_world @ best_calib["cam"]) from compute_camera_transform.py.
# Each matrix maps a point in that camera's OpenCV frame to the USD world frame.
T_CAM_LEFT_WORLD = np.array([
    [-0.02199727, -0.80581615,  0.59175708, -0.04596533],
    [-0.99905014,  0.03998766,  0.01731508,  0.09513673],
    [-0.03761575, -0.59081411, -0.80593036,  1.20379187],
    [ 0.,          0.,          0.,          1.        ],
], dtype=np.float64)

T_CAM_RIGHT_WORLD = np.array([
    [ 0.02933941, -0.83227828,  0.55358113, -0.07484866],
    [-0.99642232,  0.01956109,  0.08221870, -0.00350517],
    [-0.07925749, -0.55401284, -0.82872675,  1.23895363],
    [ 0.,          0.,          0.,          1.        ],
], dtype=np.float64)

import isaacsim.robot_motion.motion_generation as _mg_pkg

_MOTION_GEN_EXT = Path(_mg_pkg.__file__).parents[3]

PANDA_ARM_DESCRIPTION_PATH = str(
    _MOTION_GEN_EXT / "motion_policy_configs" / "franka" / "rmpflow" / "robot_descriptor.yaml"
)
PANDA_ARM_URDF_PATH = str(
    _MOTION_GEN_EXT / "motion_policy_configs" / "franka" / "lula_franka_gen.urdf"
)

FRANKA_RIGHT_PATH = "/World/fer_orcahand_right_extended"
FRANKA_LEFT_PATH = "/World/fer_orcahand_left_extended"
EE_FRAME_NAME = "panda_hand"   # frame name in the plain Franka URDF, used by LulaKinematicsSolver
EE_USD_PRIM_NAME = "fer_link8" # prim name in the USD scene (orca-hand robot uses fer_ prefix)
# TODO: update to exact measured flange→orca-wrist offset vector
EE_FLANGE_TO_EEF_OFFSET = np.array([0.13, 0.0, 0.07], dtype=np.float32)

ARM_JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
HAND_LEFT_JOINT_NAMES = [
    "left_wrist",
    "left_thumb_mcp", "left_thumb_abd", "left_thumb_pip", "left_thumb_dip",
    "left_index_abd", "left_index_mcp", "left_index_pip",
    "left_middle_abd", "left_middle_mcp", "left_middle_pip",
    "left_ring_abd", "left_ring_mcp", "left_ring_pip",
    "left_pinky_abd", "left_pinky_mcp", "left_pinky_pip",
]
HAND_RIGHT_JOINT_NAMES = [
    "right_wrist",
    "right_thumb_mcp", "right_thumb_abd", "right_thumb_pip", "right_thumb_dip",
    "right_index_abd", "right_index_mcp", "right_index_pip",
    "right_middle_abd", "right_middle_mcp", "right_middle_pip",
    "right_ring_abd", "right_ring_mcp", "right_ring_pip",
    "right_pinky_abd", "right_pinky_mcp", "right_pinky_pip",
]

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
            ("Franka Lula URDF",        PANDA_ARM_URDF_PATH),
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
            dof_names = list(articulation.dof_names)
        except Exception:
            dof_names = list(articulation.get_dof_names())
        print(f"\n===== {label} =====")
        print(f"Number of DOFs: {len(dof_names)}")
        print("DOF names:")
        for i, name in enumerate(dof_names):
            print(f"  [{i:02d}] {name}")

    def _resolve_dof_indices(self, articulation, names, label):
        try:
            dof_names = list(articulation.dof_names)
        except Exception:
            dof_names = list(articulation.get_dof_names())
        name_to_idx = {n: i for i, n in enumerate(dof_names)}

        def candidates(name):
            if name.startswith("panda_joint"):
                return [name, name.replace("panda_joint", "fer_joint", 1)]
            if name.startswith("fer_joint"):
                return [name, name.replace("fer_joint", "panda_joint", 1)]
            return [name]

        indices = []
        for name in names:
            resolved = False
            for cand in candidates(name):
                if cand in name_to_idx:
                    if cand != name:
                        print(f"[DOF] '{name}' matched via alias '{cand}'")
                    indices.append(name_to_idx[cand])
                    resolved = True
                    break
            if resolved:
                continue

            matches = [dof for dof in dof_names if dof.endswith(name)]
            if len(matches) == 1:
                print(f"[DOF] '{name}' matched via suffix to '{matches[0]}'")
                indices.append(name_to_idx[matches[0]])
            elif len(matches) > 1:
                raise RuntimeError(f"Ambiguous suffix match for '{name}' in {label}: {matches}")
            else:
                raise RuntimeError(f"Cannot find '{name}' in {label} DOFs: {dof_names}")

        return np.array(indices, dtype=int)

    @staticmethod
    def _resolve_descendant_prim_path(stage, subtree_root, prim_name):
        direct_path = f"{subtree_root}/{prim_name}"
        if stage.GetPrimAtPath(direct_path).IsValid():
            return direct_path

        matches = [
            str(prim.GetPath())
            for prim in stage.Traverse()
            if str(prim.GetPath()).startswith(f"{subtree_root}/") and prim.GetName() == prim_name
        ]

        if len(matches) == 1:
            print(f"[stage] Resolved '{prim_name}' under '{subtree_root}' -> {matches[0]}")
            return matches[0]

        if not matches:
            raise RuntimeError(
                f"Could not find prim '{prim_name}' under '{subtree_root}' in the loaded USD stage"
            )

        raise RuntimeError(
            f"Ambiguous prim '{prim_name}' under '{subtree_root}': {matches}"
        )


    def play(self, sim_config=None, vis_config=None):
        set_joints = getattr(sim_config, "set_joints", True)
        enable_right = getattr(sim_config, "enable_right", True)
        enable_left = getattr(sim_config, "enable_left", True)

        open_stage(self.stage_path)
        world = World()
        stage = omni.usd.get_context().get_stage()

        print("=== Articulation roots in stage ===")
        from pxr import UsdPhysics

        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                print(prim.GetPath())

        arm_right = world.scene.add(SingleArticulation(FRANKA_RIGHT_PATH, name="fer_right"))
        arm_left = world.scene.add(SingleArticulation(FRANKA_LEFT_PATH, name="fer_left"))
        world.reset()

        self._print_articulation_info(arm_right, "RIGHT COMBINED ROBOT")
        self._print_articulation_info(arm_left, "LEFT COMBINED ROBOT")

        arm_idx_r = self._resolve_dof_indices(arm_right, ARM_JOINT_NAMES, "RIGHT COMBINED ROBOT")
        hand_idx_r = self._resolve_dof_indices(arm_right, HAND_RIGHT_JOINT_NAMES, "RIGHT COMBINED ROBOT")
        arm_idx_l = self._resolve_dof_indices(arm_left, ARM_JOINT_NAMES, "LEFT COMBINED ROBOT")
        hand_idx_l = self._resolve_dof_indices(arm_left, HAND_LEFT_JOINT_NAMES, "LEFT COMBINED ROBOT")

        solver_r = FrankaIKController(
            label="right",
            robot_description_path=PANDA_ARM_DESCRIPTION_PATH,
            urdf_path=PANDA_ARM_URDF_PATH,
            ee_frame_name=EE_FRAME_NAME,
            flange_to_eef_offset=EE_FLANGE_TO_EEF_OFFSET,
        )

        solver_l = FrankaIKController(
            label="left",
            robot_description_path=PANDA_ARM_DESCRIPTION_PATH,
            urdf_path=PANDA_ARM_URDF_PATH,
            ee_frame_name=EE_FRAME_NAME,
            flange_to_eef_offset=EE_FLANGE_TO_EEF_OFFSET,
        )

        # Load the robot joints data inside h5 file
        replay = load_replay_h5(self.h5_path)

        right_arm_data = replay.right_arm
        left_arm_data = replay.left_arm
        right_hand_data = replay.right_hand
        left_hand_data = replay.left_hand
        n_frames = replay.n_frames
        structure = replay.structure

        # Assuming its always the right arm playing when there is only data for one arm enables/disables the arm
        # If one of them is already set to False it is a user simulation preference, does change it
        if structure == "structure_1":
            print("[H5] Following h5 structure, only one arm playing - assume its the right arm")
            if enable_left:
                print("[H5] Left arm enabled but one arm detected in h5. Disabling left.")
                enable_left = False
            if not enable_right:
                print("[H5] Right arm disabled but one arm detected in h5. Enabling right.")
                enable_right = True
        else:
            print("[H5] Following h5 structure, two arms playing")

        # Object trajectory replay: ob_in_cam (OpenCV) -> world frame
        object_cam = getattr(sim_config, "object_cam", "right")
        object_scale = getattr(sim_config, "object_scale", 0.001)
        T_cam_world = T_CAM_LEFT_WORLD if object_cam == "left" else T_CAM_RIGHT_WORLD
        scale_3 = (object_scale, object_scale, object_scale)
        h5_n_frames = n_frames

        active_objects = []  # list of (XFormPrim, traj ndarray)
        for obj_cfg in getattr(sim_config, "objects", []):
            usd_path = Path(obj_cfg.usd_path)
            traj_path = Path(obj_cfg.trajectory_npy)
            prim_path = obj_cfg.prim_path or f"/World/{usd_path.stem}"
            if not usd_path.exists():
                print(f"WARNING: object USD not found: {usd_path} — skipping")
                continue
            if not traj_path.exists():
                print(f"WARNING: trajectory not found: {traj_path} — skipping")
                continue
            traj = np.load(traj_path)  # (N, 4, 4)
            if len(traj) != h5_n_frames:
                print(f"WARNING: {usd_path.name} trajectory length {len(traj)} != H5 length {h5_n_frames}")
            n_frames = min(n_frames, len(traj))
            stage.DefinePrim(prim_path, "Xform").GetReferences().AddReference(str(usd_path))
            prim = XFormPrim(prim_path)
            prim.set_local_scales(np.array([scale_3], dtype=np.float32))
            active_objects.append((prim, traj))
            print(f"[object] {prim_path}  usd={usd_path.name}  traj_len={len(traj)}  scale={object_scale}  cam={object_cam}")

        if vis_config is None:
            vis_config = VisConfig(enabled=False)
        visualizer = EEFVisualizer() if vis_config.enabled else None

        alpha = vis_config.eef_alpha if vis_config.eef_alpha is not None else (
            0.15 if vis_config.video_mode else COLOR_AXIS_X_FADED[3]
        )
        cx_f = (COLOR_AXIS_X[0], COLOR_AXIS_X[1], COLOR_AXIS_X[2], alpha)
        cy_f = (COLOR_AXIS_Y[0], COLOR_AXIS_Y[1], COLOR_AXIS_Y[2], alpha)
        cz_f = (COLOR_AXIS_Z[0], COLOR_AXIS_Z[1], COLOR_AXIS_Z[2], alpha)

        eef_prim_r = None
        eef_prim_l = None
        if visualizer is not None:
            eef_prim_r = XFormPrim(
                self._resolve_descendant_prim_path(stage, FRANKA_RIGHT_PATH, EE_USD_PRIM_NAME)
            )
            eef_prim_l = XFormPrim(
                self._resolve_descendant_prim_path(stage, FRANKA_LEFT_PATH, EE_USD_PRIM_NAME)
            )

        prev_arm_r = arm_right.get_joint_positions()[arm_idx_r].copy()
        prev_arm_l = arm_left.get_joint_positions()[arm_idx_l].copy()

        frame = 0
        ik_fail_r = 0
        ik_fail_l = 0
        control_hz = 30
        dt = 1.0 / control_hz

        world.play()
        for _ in range(10):
            world.step(render=True)

        cam_eye = getattr(sim_config, "camera_eye", None)
        cam_target = getattr(sim_config, "camera_target", None)
        if cam_eye is not None and cam_target is not None:
            from isaacsim.core.utils.viewports import set_camera_view

            set_camera_view(eye=np.array(cam_eye), target=np.array(cam_target))

        # Simulation loop
        while self.app.is_running() and frame < n_frames:
            if enable_right:
                wrist_pose_r = np.asarray(right_arm_data[frame], dtype=np.float32)
                pos_r = wrist_pose_r[:3]
                quat_tool_r = normalize_quat_wxyz(wrist_pose_r[3:7])
                quat_urdf_r = tool_quat_to_urdf(quat_tool_r)
                q_full_r = arm_right.get_joint_positions().copy()
                q_arm_r, ok_r = solver_r.compute(
                    target_wrist_pos=pos_r,
                    target_quat_wxyz=quat_urdf_r,
                    warm_start=prev_arm_r,
                )
                if ok_r:
                    q_full_r[arm_idx_r] = q_arm_r
                    prev_arm_r = q_arm_r.copy()
                else:
                    ik_fail_r += 1
                    q_full_r[arm_idx_r] = prev_arm_r
                    print(f"[frame {frame}] IK failed RIGHT")
                if right_hand_data is not None:
                    q_hand_r = np.asarray(right_hand_data[frame], dtype=np.float32).reshape(-1)
                    if q_hand_r.shape[0] == hand_idx_r.shape[0]:
                        q_full_r[hand_idx_r] = q_hand_r
                    else:
                        print(
                            f"[WARN] RIGHT hand qpos size {q_hand_r.shape[0]} != expected {hand_idx_r.shape[0]}"
                        )
                if set_joints:
                    arm_right.set_joint_positions(q_full_r)

            if enable_left:
                wrist_pose_l = np.asarray(left_arm_data[frame], dtype=np.float32)
                pos_l = wrist_pose_l[:3]
                quat_tool_l = normalize_quat_wxyz(wrist_pose_l[3:7])
                quat_urdf_l = tool_quat_to_urdf(quat_tool_l)
                q_full_l = arm_left.get_joint_positions().copy()
                q_arm_l, ok_l = solver_l.compute(
                    target_wrist_pos=pos_l,
                    target_quat_wxyz=quat_urdf_l,
                    warm_start=prev_arm_l,
                )
                if ok_l:
                    q_full_l[arm_idx_l] = q_arm_l
                    prev_arm_l = q_arm_l.copy()
                else:
                    ik_fail_l += 1
                    q_full_l[arm_idx_l] = prev_arm_l
                    print(f"[frame {frame}] IK failed LEFT")
                if left_hand_data is not None:
                    q_hand_l = np.asarray(left_hand_data[frame], dtype=np.float32).reshape(-1)
                    if q_hand_l.shape[0] == hand_idx_l.shape[0]:
                        q_full_l[hand_idx_l] = q_hand_l
                    else:
                        print(
                            f"[WARN] LEFT hand qpos size {q_hand_l.shape[0]} != expected {hand_idx_l.shape[0]}"
                        )
                if set_joints:
                    arm_left.set_joint_positions(q_full_l)

            for obj_prim, object_traj in active_objects:
                ob_in_world = T_cam_world @ object_traj[frame]
                t = ob_in_world[:3, 3].astype(np.float32)
                q_xyzw = Rotation.from_matrix(ob_in_world[:3, :3]).as_quat()
                q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)
                obj_prim.set_world_poses(positions=t.reshape(1, 3), orientations=q_wxyz.reshape(1, 4))

            viz_offset = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            if visualizer is not None:
                if enable_right:
                    act_pos_r_b, act_quat_wxyz_r_b = eef_prim_r.get_world_poses()
                    act_quat_r_xyzw = np.array(
                        [
                            act_quat_wxyz_r_b[0, 1],
                            act_quat_wxyz_r_b[0, 2],
                            act_quat_wxyz_r_b[0, 3],
                            act_quat_wxyz_r_b[0, 0],
                        ],
                        dtype=np.float32,
                    )
                    quat_urdf_r_xyzw = np.array(
                        [quat_urdf_r[1], quat_urdf_r[2], quat_urdf_r[3], quat_urdf_r[0]],
                        dtype=np.float32,
                    )
                    if vis_config.show_eef:
                        visualizer.draw_frame(pos_r, quat_urdf_r_xyzw, COLOR_AXIS_X, COLOR_AXIS_Y, COLOR_AXIS_Z)
                        visualizer.draw_frame(act_pos_r_b[0], act_quat_r_xyzw, cx_f, cy_f, cz_f)
                    if vis_config.show_offset:
                        visualizer.draw_frame(
                            pos_r + viz_offset,
                            quat_urdf_r_xyzw,
                            COLOR_AXIS_X,
                            COLOR_AXIS_Y,
                            COLOR_AXIS_Z,
                            length=ORIENT_LENGTH * 0.5,
                            width=FRAME_LINE_SIZE * 2,
                        )
                        visualizer.draw_frame(act_pos_r_b[0] + viz_offset, act_quat_r_xyzw, cx_f, cy_f, cz_f)

                if enable_left:
                    act_pos_l_b, act_quat_wxyz_l_b = eef_prim_l.get_world_poses()
                    act_quat_l_xyzw = np.array(
                        [
                            act_quat_wxyz_l_b[0, 1],
                            act_quat_wxyz_l_b[0, 2],
                            act_quat_wxyz_l_b[0, 3],
                            act_quat_wxyz_l_b[0, 0],
                        ],
                        dtype=np.float32,
                    )
                    quat_urdf_l_xyzw = np.array(
                        [quat_urdf_l[1], quat_urdf_l[2], quat_urdf_l[3], quat_urdf_l[0]],
                        dtype=np.float32,
                    )
                    if vis_config.show_eef:
                        visualizer.draw_frame(pos_l, quat_urdf_l_xyzw, COLOR_AXIS_X, COLOR_AXIS_Y, COLOR_AXIS_Z)
                        visualizer.draw_frame(act_pos_l_b[0], act_quat_l_xyzw, cx_f, cy_f, cz_f)
                    if vis_config.show_offset:
                        visualizer.draw_frame(
                            pos_l + viz_offset,
                            quat_urdf_l_xyzw,
                            COLOR_AXIS_X,
                            COLOR_AXIS_Y,
                            COLOR_AXIS_Z,
                            length=ORIENT_LENGTH * 0.5,
                            width=FRAME_LINE_SIZE * 2,
                        )
                        visualizer.draw_frame(act_pos_l_b[0] + viz_offset, act_quat_l_xyzw, cx_f, cy_f, cz_f)

            if frame % 100 == 0:
                print(f"\nframe {frame}/{n_frames}")
                if enable_right:
                    print("  pos_r         :", pos_r)
                    print("  quat_urdf_r   :", quat_urdf_r)
                if enable_left:
                    print("  pos_l         :", pos_l)
                    print("  quat_urdf_l   :", quat_urdf_l)
                print(f"  IK fails      : right={ik_fail_r}, left={ik_fail_l}")

            world.step(render=True)
            time.sleep(dt)
            frame += 1

        print("\nReplay finished.")
        print(f"IK failures right: {ik_fail_r}/{n_frames}")
        print(f"IK failures left : {ik_fail_l}/{n_frames}")

        while self.app.is_running():
            world.step(render=True)
