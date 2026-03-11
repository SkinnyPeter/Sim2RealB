import h5py
import numpy as np

H5_DEFAULT_PATH = r"C:\Users\pardo\Desktop\3DV\simulation\data\20250804_104715.h5"

class H5Analyzer():
    """
    Utility class for inspecting and visualizing data stored in an HDF5 file.

    The class provides tools to:
    - Print dataset names, shapes, and data types
    - Play image sequences stored in the file as a video

    Parameters
    ----------
    file_path : str, optional
        Path to the HDF5 file to analyze. Defaults to `H5_DEFAULT_PATH`.
    """
    def __init__(self, file_path=H5_DEFAULT_PATH):
        """
        Initialize the analyzer with the path to an HDF5 file.

        Parameters
        ----------
        file_path : str
            Path to the HDF5 file.
        """
        self.file_path = file_path

    # Inspect h5 file, return name, shape and dtype
    def inspect(self):
        """
        Inspect the HDF5 file structure.

        This method prints:
        - Top-level keys in the file
        - All groups and datasets
        - Dataset shapes and data types

        It also attempts to load and display information for common
        joint position datasets (`qpos_arm` and `qpos_hand`) if they exist.
        """
        with h5py.File(self.file_path, "r") as f:
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

    def play_video(self):
        """
        Display an image sequence from the HDF5 file as a video.

        The method reads frames from the dataset:
        `observations/images/aria_rgb_cam/color`.

        Frames are displayed using OpenCV until:
        - The sequence ends
        - The user presses the ESC key

        Notes
        -----
        Requires the `opencv-python` package. If OpenCV is not installed,
        the function will print an error message and exit.
        """
        try:
            import cv2
        except ImportError:
            print("ERROR: OpenCV not installed\nThis methode requires OpenCV to play the video. Please install it.")
            return

        with h5py.File(self.file_path, "r") as f:
            images = f["observations/images/aria_rgb_cam/color"]

            for i in range(images.shape[0]):
                frame = images[i]

                # convert RGB -> BGR for OpenCV
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                cv2.imshow("Aria RGB Cam", frame)

                if cv2.waitKey(30) & 0xFF == 27:  # press ESC to stop
                    break

        cv2.destroyAllWindows()