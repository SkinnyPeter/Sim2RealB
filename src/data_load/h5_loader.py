from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from src.simulator.quat_utils import detect_quaternion_order


DATA_PATH = Path(__file__).resolve().parents[2] / "data"
H5_DIR = DATA_PATH / "h5"

# H5 file structure when only one arm
# IMPORTANT REMARK: For this structure, qpos will be given to both left_arm and right_arm
# we assume one of the two is disabled on the simulation
STRUCTURE_1_REQUIRED = {
    "actions_arm",
    "actions_hand",
    "observations/images/aria_rgb_cam/color",
    "observations/qpos_arm",
    "observations/qpos_hand",
}

# H5 file structure when two arms
STRUCTURE_2_REQUIRED = {
    "actions_arm_left",
    "actions_arm_right",
    "actions_hand_left",
    "actions_hand_right",
    "observations/images/aria_rgb_cam/color",
    "observations/images/oakd_front_view/color",
    "observations/qpos_arm_left",
    "observations/qpos_arm_right",
    "observations/qpos_hand_left",
    "observations/qpos_hand_right",
}

@dataclass
class ReplayData:
    right_arm: np.ndarray
    left_arm: np.ndarray
    right_hand: np.ndarray | None
    left_hand: np.ndarray | None
    n_frames: int
    structure: str

def resolve_h5_path(sample_id_or_path: str | Path) -> Path:
    """
    Resolve a valid HDF5 file path from either a sample ID or a path.

    Accepted inputs:
    - Sample ID (e.g. "20250804_111125") → maps to DATA_PATH/h5/<id>.h5
    - Filename (e.g. "20250804_111125.h5") → searched inside DATA_PATH/h5/
    - Absolute or relative path to an existing .h5 file → used directly

    Behavior:
    - If the input already points to an existing file, return it as-is
    - Otherwise, construct the path inside the H5 directory
    - Raise FileNotFoundError if the file cannot be found

    Returns:
        Path: absolute path to the resolved .h5 file
    """
    value = Path(sample_id_or_path)

    # Case 1: direct existing path
    if value.exists():
        return value

    # Case 2: given "20250804_111125.h5"
    if value.suffix == ".h5":
        h5_path = H5_DIR / value.name

    # Case 3: given "20250804_111125"
    else:
        h5_path = H5_DIR / f"{value.name}.h5"

    if not h5_path.exists():
        raise FileNotFoundError(f"H5 file not found: {h5_path}")

    return h5_path

def list_h5_datasets(h5_path: Path) -> set[str]:
    """
    Return all dataset paths contained in an HDF5 file.

    Traverses the full HDF5 hierarchy and collects only dataset nodes
    (i.e. actual data arrays), ignoring groups.

    Each returned entry is the full internal path of a dataset
    (e.g. "observations/qpos_arm", "observations/images/.../color").

    This is used to:
    - inspect the file structure
    - validate against predefined structures
    - debug unknown HDF5 formats

    Args:
        h5_path (Path): path to the .h5 file

    Returns:
        set[str]: set of dataset paths present in the file
    """
    datasets = set()

    with h5py.File(h5_path, "r") as f:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets.add(name)

        f.visititems(visitor)

    return datasets

def detect_h5_structure(h5_path: Path) -> str:
    """
    Detect which predefined HDF5 structure matches a given file.

    The function lists all dataset paths in the HDF5 file and compares them
    against the required dataset paths defined in STRUCTURE_1_REQUIRED and
    STRUCTURE_2_REQUIRED.

    Behavior:
    - Returns "structure_1" if all structure 1 required datasets are present
    - Returns "structure_2" if all structure 2 required datasets are present
    - Raises RuntimeError if the file does not match any known structure

    Args:
        h5_path (Path): path to the .h5 file

    Returns:
        str: detected structure name

    Raises:
        RuntimeError: if required datasets are missing for all known structures
    """
    datasets = list_h5_datasets(h5_path)

    missing_1 = STRUCTURE_1_REQUIRED - datasets
    missing_2 = STRUCTURE_2_REQUIRED - datasets

    if not missing_1:
        return "structure_1"

    if not missing_2:
        return "structure_2"

    error_msg = (
        f"H5 file does not match any known structure: {h5_path}\n\n"
        f"Missing for structure_1:\n"
        + "\n".join(f"  - {k}" for k in sorted(missing_1))
        + "\n\n"
        f"Missing for structure_2:\n"
        + "\n".join(f"  - {k}" for k in sorted(missing_2))
        + "\n\n"
        f"Available datasets:\n"
        + "\n".join(f"  - {k}" for k in sorted(datasets))
    )

    raise RuntimeError(error_msg)

def load_replay_h5(sample_id_or_path: str | Path) -> ReplayData:
    """
    Load replay data from an HDF5 file.

    The input can be either a sample ID or a path to an HDF5 file. The function
    first resolves the H5 path, detects the file structure, then loads the
    corresponding arm and hand joint trajectories.

    For structure_1:
    - Uses observations/qpos_arm for both right_arm and left_arm
    - Uses observations/qpos_hand for both right_hand and left_hand

    For structure_2:
    - Loads separate right/left arm and hand trajectories

    Quaternion ordering is checked and converted if needed. The number of replay
    frames is computed as the minimum length across all loaded trajectories.

    Args:
        sample_id_or_path (str | Path): sample ID, filename, or path to .h5 file

    Returns:
        ReplayData: parsed replay arrays, frame count, and detected structure
    """
    h5_path = resolve_h5_path(sample_id_or_path)
    structure = detect_h5_structure(h5_path)

    with h5py.File(h5_path, "r") as f:
        if structure == "structure_1":
            arm = np.array(f["observations/qpos_arm"])
            hand = np.array(f["observations/qpos_hand"])

            right_arm = arm
            left_arm = arm
            right_hand = hand
            left_hand = hand

        elif structure == "structure_2":
            right_arm = np.array(f["observations/qpos_arm_right"])
            left_arm = np.array(f["observations/qpos_arm_left"])
            right_hand = np.array(f["observations/qpos_hand_right"])
            left_hand = np.array(f["observations/qpos_hand_left"])

        else:
            raise RuntimeError(f"Unsupported H5 structure: {structure}")

    right_arm = detect_quaternion_order(right_arm, "right")
    left_arm = detect_quaternion_order(left_arm, "left")

    n_frames = min(len(right_arm), len(left_arm))

    if right_hand is not None:
        n_frames = min(n_frames, len(right_hand))
    if left_hand is not None:
        n_frames = min(n_frames, len(left_hand))

    print("\n===== H5 DATA =====")
    print("file              :", h5_path)
    print("structure         :", structure)
    print("right_arm shape   :", right_arm.shape)
    print("left_arm shape    :", left_arm.shape)
    if right_hand is not None:
        print("right_hand shape  :", right_hand.shape)
    if left_hand is not None:
        print("left_hand shape   :", left_hand.shape)
    print("n_frames          :", n_frames)

    return ReplayData(
        right_arm=right_arm,
        left_arm=left_arm,
        right_hand=right_hand,
        left_hand=left_hand,
        n_frames=n_frames,
        structure=structure,
    )