import h5py
import numpy as np

H5_PATH = r"C:\Users\pardo\Desktop\3DV\simulation\data\20250804_104715.h5"

# Inspect h5 file
with h5py.File(H5_PATH, "r") as f:
    print("Top-level keys:", list(f.keys()))
    
    # Print detailed info for all datasets
    def print_item(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"DATASET: {name}")
            print(f"  shape: {obj.shape}")
            print(f"  dtype: {obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"GROUP:   {name}")

    f.visititems(print_item)

    # Load likely joint datasets
    qpos_arm = np.array(f["observations/qpos_arm"])
    qpos_hand = np.array(f["observations/qpos_hand"])

    print("\nqpos_arm:")
    print(" shape:", qpos_arm.shape)
    print(" first row:", qpos_arm[0])

    print("\nqpos_hand:")
    print(" shape:", qpos_hand.shape)
    print(" first row:", qpos_hand[0])
