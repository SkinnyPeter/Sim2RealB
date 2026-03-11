from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.utils.stage import open_stage
from isaacsim.core.prims import SingleArticulation

# Load prebuilt environment
scene_path = r"C:\Users\pardo\Desktop\3DV\simulation\scenes\scene.usd"
open_stage(scene_path)

world = World()
world.step(render=True)

robot = SingleArticulation(
    prim_path="/World/Franka",
    name="franka"
)

world.reset()
robot.initialize()

print(robot.dof_names)