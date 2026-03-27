import h5py
import numpy as np

path = "/home/teamb/Desktop/Sim2RealB_2/data/20250827_151212.h5"

with h5py.File(path, "r") as f:
    q_r = np.array(f["observations/qpos_arm_right"])
    q_l = np.array(f["observations/qpos_arm_left"])

for idx in [0, 500, 1000, 1500]:
    print("\nFRAME", idx)
    print("right pos:", q_r[idx, :3])
    print("left  pos:", q_l[idx, :3])