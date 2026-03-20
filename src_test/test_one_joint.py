import numpy as np

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.utils.stage import open_stage
from isaacsim.core.prims import SingleArticulation

SCENE_PATH = "/home/teamb/Desktop/Sim2RealB/scenes/scene.usd"
ROBOT_PATH = "/World/Franka"


def main():
    open_stage(SCENE_PATH)

    world = World()
    robot = SingleArticulation(ROBOT_PATH, name="franka")

    world.reset()
    robot.initialize()

    n = len(robot.dof_names)
    q = np.zeros(n)
    q_target = np.zeros(n)

    q_target[0] = 0.4
    q_target[1] = -0.3
    q_target[2] = 0.2
    q_target[3] = -1.0
    q_target[4] = 0.1
    q_target[5] = 1.0
    q_target[6] = 0.2

    if n >= 9:
        q[7] = 0.04
        q[8] = 0.04
        q_target[7] = 0.04
        q_target[8] = 0.04

    Kp = 0.02

    while simulation_app.is_running():
        world.step(render=True)

        if not world.is_playing():
            continue

        error = q_target - q
        q = q + Kp * error

        robot.set_joint_positions(q)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()