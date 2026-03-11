import h5py
import numpy as np

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.utils.stage import open_stage
from isaacsim.core.prims import SingleArticulation

SCENE_PATH = r"C:\Users\pardo\Desktop\3DV\simulation\scenes\scene.usd"
H5_PATH = r"C:\Users\pardo\Desktop\3DV\simulation\data\20250804_104715.h5"

open_stage(SCENE_PATH)

world = World()
robot = SingleArticulation("/World/Franka", name="franka")

world.reset()
robot.initialize()

print(robot.dof_names)

# -------------------
# load dataset
# -------------------

with h5py.File(H5_PATH, "r") as f:
    #q_arm = np.array(f["actions_arm"])
    q_arm = np.array(f["observations/qpos_arm"])

print("trajectory:", q_arm.shape)

# -------------------
# replay
# -------------------

frame = 0

while simulation_app.is_running():

    world.step(render=True)

    if not world.is_playing():
        continue

    if frame >= len(q_arm):
        break

    q = np.zeros(9)

    q[:7] = q_arm[frame]     # arm
    q[7:] = 0.04             # open gripper

    robot.set_joint_positions(q)

    frame += 1

simulation_app.close()