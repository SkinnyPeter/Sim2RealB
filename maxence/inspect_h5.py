import h5py
import numpy as np

H5_PATH = "/home/teamb/Desktop/Sim2RealB_2/data/20250827_151212.h5"


def preview_dataset(name, obj, max_rows=3):
    if not isinstance(obj, h5py.Dataset):
        return

    print(f"\nDATASET: {name}")
    print(f"  shape: {obj.shape}")
    print(f"  dtype: {obj.dtype}")

    # Try to print attributes if any
    if len(obj.attrs) > 0:
        print("  attrs:")
        for k, v in obj.attrs.items():
            print(f"    {k}: {v}")

    # Preview small part of content
    try:
        data = obj[()]
        arr = np.array(data)

        if arr.ndim == 0:
            print(f"  value: {arr}")
        elif arr.ndim == 1:
            print(f"  first values: {arr[:10]}")
            print(f"  min/max: {arr.min()} / {arr.max()}")
        else:
            print(f"  first {max_rows} rows:")
            print(arr[:max_rows])

            # Numeric min/max if possible
            if np.issubdtype(arr.dtype, np.number):
                try:
                    print("  min per column:")
                    print(arr.min(axis=0))
                    print("  max per column:")
                    print(arr.max(axis=0))
                except Exception:
                    pass

                # Special check for 7D pose-like arrays
                if arr.ndim == 2 and arr.shape[1] == 7:
                    quat = arr[:, 3:7]
                    norms = np.linalg.norm(quat, axis=1)
                    print(
                        f"  quaternion norm stats (cols 3:7): "
                        f"mean={norms.mean():.6f}, std={norms.std():.6f}, "
                        f"min={norms.min():.6f}, max={norms.max():.6f}"
                    )
    except Exception as e:
        print(f"  preview failed: {e}")


def preview_group(name, obj):
    if isinstance(obj, h5py.Group):
        print(f"\nGROUP: {name if name else '/'}")
        if len(obj.attrs) > 0:
            print("  attrs:")
            for k, v in obj.attrs.items():
                print(f"    {k}: {v}")


def main():
    with h5py.File(H5_PATH, "r") as f:
        print("=" * 80)
        print("HDF5 FILE STRUCTURE")
        print("=" * 80)

        # Print groups first
        f.visititems(preview_group)

        print("\n" + "=" * 80)
        print("DATASET DETAILS")
        print("=" * 80)

        # Print datasets with previews
        f.visititems(preview_dataset)

        print("\n" + "=" * 80)
        print("TOP-LEVEL KEYS")
        print("=" * 80)
        print(list(f.keys()))


if __name__ == "__main__":
    main()