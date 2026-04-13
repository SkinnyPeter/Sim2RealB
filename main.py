from dataclasses import dataclass
from src.h5_analyzer import H5Analyzer
from src.visualization import VisConfig
from isaacsim import SimulationApp

from pathlib import Path


@dataclass
class SimConfig:
    set_joints:       bool  = True
    enable_right:     bool  = True
    enable_left:      bool  = True
    ik_solver:        str   = "curobo"  # "lula" or "curobo"
    num_seeds:        int   = 32        # curobo IK seeds (more = more stable, higher GPU cost)
    camera_eye:    tuple = (1.97035, 0.00915, 1.58108)  # viewport camera position
    camera_target: tuple = (0.51, 0.0, 1.23)           # look-at point; set both to None to skip

# Directory where main.py is located
BASE_DIR = Path(__file__).resolve().parent

data_path = BASE_DIR / "data" / "20250827_151212.h5"
scene_path = BASE_DIR / "scenes" / "scene_v4.usd"

SIM = SimConfig(
    set_joints   = True,
    enable_right = True,
    enable_left  = True,
    ik_solver    = "curobo",  # "lula" or "curobo"
    num_seeds    = 128,
)

VIS = VisConfig(
    enabled    = True,
    show_eef   = False,   # draw frames at EEF / target position
    show_offset= True,   # draw frames lifted 1 m above
    video_mode = False,  # boost opacity of faded frames for screen recording
)

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
    simulator.play(sim_config=SIM, vis_config=VIS)

    simulation_app.close()

if __name__ == "__main__":
    main()