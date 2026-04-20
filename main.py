from dataclasses import dataclass
from pathlib import Path
import os

from src.visualization import VisConfig
from isaacsim import SimulationApp


@dataclass
class SimConfig:
    set_joints: bool = True
    enable_right: bool = True
    enable_left: bool = True
    camera_eye: tuple = (1.97035, 0.00915, 1.58108)
    camera_target: tuple = (0.51, 0.0, 1.23)


BASE_DIR = Path(__file__).resolve().parent
DESCRIPTION_ROOT = Path(
    os.environ.get(
        "PANDAORCA_ROOT",
        "/local/home/teamb/Desktop/Real2Sim/assets/pandaorca_description-main",
    )
)

data_path = Path(os.environ.get("SIM2REAL_H5", "/local/home/teamb/Desktop/data/20250827_151212.h5"))
scene_path = BASE_DIR / "scenes" / "scene.usd"

SIM = SimConfig(
    set_joints=True,
    enable_right=True,
    enable_left=True,
)

VIS = VisConfig(
    enabled=True,
    show_eef=False,
    show_offset=True,
    video_mode=False,
)


def _check_paths():
    missing = []
    for label, path in [
        ("Scene", scene_path),
        ("Data", data_path),
        ("pandaorca_description", DESCRIPTION_ROOT),
    ]:
        if not Path(path).exists():
            missing.append(f"  {label}: {path}")
    if missing:
        raise FileNotFoundError(
            "Missing required files — fix before Isaac Sim starts:\n" + "\n".join(missing)
        )


def main():
    _check_paths()

    simulation_app = SimulationApp({"headless": False})

    from src.simulator import Simulator

    simulator = Simulator(simulation_app, str(scene_path), str(data_path), str(DESCRIPTION_ROOT))
    simulator.play(sim_config=SIM, vis_config=VIS)

    simulation_app.close()


if __name__ == "__main__":
    main()
