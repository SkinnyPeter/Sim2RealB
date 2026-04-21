from isaacsim import SimulationApp
# Simulation app config
config = {
    "headless": False,
    "extension_to_exclude": [
        "isaacsim.sensors.rtx",
        "omni.sensors.nv.lidar",
        "omni.sensors.nv.radar"
    ]
}
simulation_app = SimulationApp(config)

from pathlib import Path
from src.h5_analyzer import H5Analyzer
# Directory where main.py is located
VIDEO = False # TODO: I had a bug with cv2 on my laptop (Mathieu talking)
BASE_DIR = Path(__file__).resolve().parent


# /object_in_bowl_processed_50hz.20250804_175047.h5
data_path = BASE_DIR / "data" / "20250827_151212.h5" # This one is from object_in_bowl_processed_50hz
scene_path = BASE_DIR / "scenes" / "scene.usd"

data_path = str(data_path)
scene_path = str(scene_path)

def main():
    if VIDEO:
        # Create an analyzer with the default file path
        analyzer = H5Analyzer(data_path)

        # Use the inspect() method
        #print("Inspecting HDF5 file...")
        #analyzer.inspect()

        # Use the play_video() method
        print("\nPlaying video from HDF5 file...")
        analyzer.play_video()

    

    # Simulator needs to be import after simulation_app is created
    from src.simulator import Simulator

    simulator = Simulator(simulation_app, scene_path,data_path)
    simulator.play()

    simulation_app.close()

if __name__ == "__main__":
    main()