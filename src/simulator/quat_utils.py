"""
quat_utils.py

Quaternion utility functions.

Used by simulator.py
"""
import numpy as np

Q_TOOL_TO_URDF_WXYZ = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)  # Rx(180°)

def normalize_quat_wxyz(quat):
    quat = np.asarray(quat, dtype=np.float32).reshape(4)
    n = np.linalg.norm(quat)
    if n < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return quat / n

def quat_multiply_wxyz(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float32)


def tool_quat_to_urdf(q_tool_wxyz):
    q_tool_wxyz = normalize_quat_wxyz(q_tool_wxyz)
    return normalize_quat_wxyz(quat_multiply_wxyz(Q_TOOL_TO_URDF_WXYZ, q_tool_wxyz))


def wxyz_to_rotation_matrix(q):
    w, x, y, z = normalize_quat_wxyz(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)

def detect_quaternion_order(arm_data, label):
    w_if_wxyz = float(np.mean(np.abs(arm_data[:, 3])))
    w_if_xyzw = float(np.mean(np.abs(arm_data[:, 6])))
    print(f"[quat] {label}: mean|col3|={w_if_wxyz:.4f} mean|col6|={w_if_xyzw:.4f}")
    if w_if_xyzw > w_if_wxyz:
        print(f"[quat] {label}: detected xyzw ordering. Reordering to wxyz.")
        reordered = arm_data.copy()
        reordered[:, 3] = arm_data[:, 6]
        reordered[:, 4] = arm_data[:, 3]
        reordered[:, 5] = arm_data[:, 4]
        reordered[:, 6] = arm_data[:, 5]
        return reordered
    print(f"[quat] {label}: appears to be wxyz ordering. Using as-is.")
    return arm_data
