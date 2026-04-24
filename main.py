from dataclasses import dataclass, field
from pathlib import Path

from src.visualization import VisConfig
from isaacsim import SimulationApp


@dataclass
class ObjectConfig:
    usd_path: Path
    trajectory_npy: Path
    prim_path: str = None  # auto: /World/<usd_stem> if None


@dataclass
class SimConfig:
    set_joints: bool = True
    enable_right: bool = True
    enable_left: bool = True
    camera_eye: tuple = (1.97035, 0.00915, 1.58108)
    camera_target: tuple = (0.51, 0.0, 1.23)
    object_cam: str = "right"    # TODO: confirm which camera trajectories were recorded from
    object_scale: float = 0.001  # uniform mm→m applied to all objects
    objects: list = field(default_factory=list)  # list[ObjectConfig]


BASE_DIR = Path(__file__).resolve().parent

data_path = BASE_DIR / "data" / "h5" / "20250804_105355.h5"
scene_path = BASE_DIR / "scenes" / "scene.usd"

SIM = SimConfig(
    set_joints=True,
    enable_right=True,
    enable_left=True,
    objects=[
        ObjectConfig(
            usd_path=BASE_DIR / "assets" / "objects" / "rubber_duck.usd",
            trajectory_npy=BASE_DIR / "data" / "trajectory.npy",
        ),
    ],
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

    from src.simulator.simulator import Simulator

    simulator = Simulator(simulation_app, str(scene_path), str(data_path))
    simulator.play(sim_config=SIM, vis_config=VIS)

    simulation_app.close()


if __name__ == "__main__":
    main()
