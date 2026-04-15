from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})  # headless so it's fast

from isaacsim.core.utils.stage import open_stage
import omni.usd
from pxr import UsdGeom, Gf

SCENE_PATH = r"/home/teamb/Desktop/Sim2RealB/scenes/scene.usd"
open_stage(SCENE_PATH)
stage = omni.usd.get_context().get_stage()

for path in ["/World/Franka_left", "/World/Franka_right"]:
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        print(path, "— prim not found")
        continue
    xf = UsdGeom.XformCache()
    transform = xf.GetLocalToWorldTransform(prim)
    print(f"\n{path}")
    print(f"  translation : {transform.ExtractTranslation()}")
    print(f"  rotation    : {Gf.Rotation(transform.ExtractRotationMatrix())}")

simulation_app.close()