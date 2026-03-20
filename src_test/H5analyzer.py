import h5py
import numpy as np

H5_DEFAULT_PATH = "data/20250827_151212.h5"


class H5Analyzer:
    """
    Utility class for inspecting and visualizing data stored in an HDF5 file.
    """

    def __init__(self, file_path=H5_DEFAULT_PATH):
        self.file_path = file_path

    def inspect(self):
        """
        Print the full structure of the HDF5 file and preview common datasets.
        """
        with h5py.File(self.file_path, "r") as f:
            print("Top-level keys:", list(f.keys()))
            print()

            def print_item(name, obj):
                if isinstance(obj, h5py.Group):
                    print(f"GROUP:   {name}")
                elif isinstance(obj, h5py.Dataset):
                    print(f"DATASET: {name}")
                    print(f"  shape: {obj.shape}")
                    print(f"  dtype: {obj.dtype}")

            f.visititems(print_item)

            print("\n===== COMMON DATASETS =====")

            possible_keys = [
                "observations/qpos_arm_left",
                "observations/qpos_arm_right",
                "observations/qpos_hand_left",
                "observations/qpos_hand_right",
                "observations/images/aria_rgb_cam/color",
                "observations/images/oakd_front_view/color",
                "actions_hand_right",
            ]

            for key in possible_keys:
                if key in f:
                    data = np.array(f[key])
                    print(f"\n{key}:")
                    print("  shape:", data.shape)
                    print("  dtype:", data.dtype)
                    if data.ndim >= 1 and len(data) > 0:
                        print("  first element preview:", data[0])
                else:
                    print(f"\n{key}: not found")

    def play_video(self, video_key="observations/images/aria_rgb_cam/color", delay_ms=30):
        """
        Display an image sequence from the HDF5 file as a video.

        Parameters
        ----------
        video_key : str
            Dataset path of the image sequence inside the HDF5 file.
        delay_ms : int
            Delay between frames in milliseconds.
        """
        try:
            import cv2
        except ImportError:
            print("ERROR: OpenCV not installed.")
            print("Install it with: pip install opencv-python")
            return

        with h5py.File(self.file_path, "r") as f:
            if video_key not in f:
                print(f"Dataset '{video_key}' not found in file.")
                return

            images = f[video_key]
            print(f"Playing video from: {video_key}")
            print(f"Video shape: {images.shape}")

            for i in range(images.shape[0]):
                frame = images[i]

                # Convert RGB -> BGR for OpenCV display
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow("H5 Video", frame)

                # ESC to quit
                if cv2.waitKey(delay_ms) & 0xFF == 27:
                    break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    analyzer = H5Analyzer()
    analyzer.inspect()