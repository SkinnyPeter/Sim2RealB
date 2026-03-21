import h5py
import numpy as np

from isaacsim import SimulationApp

# Start Isaac Sim
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.utils.stage import open_stage
from isaacsim.core.prims import SingleArticulation

SCENE_PATH = "/home/teamb/Desktop/Sim2RealB/scenes/scene.usd"
H5_PATH = "/home/teamb/Desktop/Sim2RealB/data/20250827_151212.h5"

ROBOT_PATH = "/World/Franka"


def main():
    # Open scene
    open_stage(SCENE_PATH)

    # Create world and robot wrapper
    world = World()
    robot = SingleArticulation(ROBOT_PATH, name="franka")

    # Initialize simulation
    world.reset()
    robot.initialize()

    print("DOF names:", robot.dof_names)
    print("Number of DOF:", len(robot.dof_names))

    # Load arm trajectory
    with h5py.File(H5_PATH, "r") as f:
        if "observations/qpos_arm_left" not in f:
            raise KeyError("Dataset 'observations/qpos_arm_left' not found in H5 file.")

        q_arm = np.array(f["observations/qpos_arm_left"])

    print("Trajectory shape:", q_arm.shape)

    # Safety checks
    if q_arm.ndim != 2 or q_arm.shape[1] != 7:
        raise ValueError(
            f"Expected arm trajectory of shape (N, 7), got {q_arm.shape}"
        )

    if len(robot.dof_names) < 9:
        raise ValueError(
            f"Expected at least 9 DOF for Franka + gripper, got {len(robot.dof_names)}"
        )

    frame = 0

    while simulation_app.is_running():
        world.step(render=True)

        if not world.is_playing():
            continue

        if frame >= len(q_arm):
            print("Replay finished.")
            break

        # 7 arm joints + 2 gripper joints
        q = np.zeros(len(robot.dof_names))
        q[:7] = q_arm[frame]

        # Keep gripper open if available
        if len(robot.dof_names) >= 9:
            q[7] = 0.04
            q[8] = 0.04

        robot.set_joint_positions(q)
        frame += 1


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()