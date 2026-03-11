"""
Print the content of C:.../scene.usd
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.utils.stage import open_stage
import omni.usd

scene_path = r"C:\Users\pardo\Desktop\3DV\simulation\scenes\scene.usd"
open_stage(scene_path)

stage = omni.usd.get_context().get_stage()

print("===== STAGE CONTENT =====")
for prim in stage.Traverse():
    print(prim.GetPath())

while simulation_app.is_running():
    pass

simulation_app.close()