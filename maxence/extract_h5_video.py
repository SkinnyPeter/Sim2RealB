import h5py
import matplotlib.pyplot as plt

H5_PATH = "/home/teamb/Desktop/Sim2RealB_2/data/20250827_151212.h5"
FRAME_IDX = 0  # change ici

with h5py.File(H5_PATH, "r") as f:
    frame_aria = f["observations/images/aria_rgb_cam/color"][FRAME_IDX]
    frame_oakd = f["observations/images/oakd_front_view/color"][FRAME_IDX]

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.imshow(frame_aria)
plt.title(f"aria_rgb_cam - frame {FRAME_IDX}")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(frame_oakd)
plt.title(f"oakd_front_view - frame {FRAME_IDX}")
plt.axis("off")

plt.tight_layout()
plt.show()