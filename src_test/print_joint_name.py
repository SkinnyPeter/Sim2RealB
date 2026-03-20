from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.utils.stage import open_stage
from isaacsim.core.prims import SingleArticulation

SCENE_PATH = "/home/teamb/Desktop/Sim2RealB/scenes/scene.usd"
H5_PATH = "/home/teamb/Desktop/Sim2RealB/data/20250827_151212.h5"

ROBOT_PATH = "/World/Franka"


def main():
    open_stage(SCENE_PATH)

    world = World()
    robot = SingleArticulation(ROBOT_PATH, name="franka")

    world.reset()
    robot.initialize()

    print("===== JOINT / DOF NAMES =====")
    for i, name in enumerate(robot.dof_names):
        print(f"{i}: {name}")

    print(f"\nTotal number of DOF: {len(robot.dof_names)}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()