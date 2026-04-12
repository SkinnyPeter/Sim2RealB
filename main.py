from src.h5_analyzer import H5Analyzer
from isaacsim import SimulationApp

from pathlib import Path

# Directory where main.py is located
BASE_DIR = Path(__file__).resolve().parent

data_path = BASE_DIR / "data" / "20250827_151212.h5"
scene_path = BASE_DIR / "scenes" / "scene_v4.usd"

VISUALIZE_EEF = True   # draw EEF spheres (red = right, dark blue = left)
SET_JOINTS    = True   # apply joint positions to arms and hands each frame
ENABLE_RIGHT  = False   # enable motion + visualization for right arm
ENABLE_LEFT   = True   # enable motion + visualization for left arm

data_path = str(data_path)
scene_path = str(scene_path)

def _check_paths():
    missing = []
    for label, path in [("Scene", scene_path), ("Data ", data_path)]:
        if not Path(path).exists():
            missing.append(f"  {label}: {path}")
    if missing:
        raise FileNotFoundError(
            "Missing required files — fix before Isaac Sim starts:\n" + "\n".join(missing)
        )


def main():
    _check_paths()

    # # Create an analyzer with the default file path
    # analyzer = H5Analyzer(data_path)

    # # Use the inspect() method
    # print("Inspecting HDF5 file...")
    # analyzer.inspect()

    # # Use the play_video() method
    # print("\nPlaying video from HDF5 file...") 
    # analyzer.play_video()

    simulation_app = SimulationApp({"headless": False})

    # Simulator needs to be import after simulation_app is created
    from src.simulator import Simulator

    simulator = Simulator(simulation_app, scene_path, data_path)
    simulator.play(visualize_eef=VISUALIZE_EEF, set_joints=SET_JOINTS, enable_right=ENABLE_RIGHT, enable_left=ENABLE_LEFT)

    simulation_app.close()

if __name__ == "__main__":
    main()