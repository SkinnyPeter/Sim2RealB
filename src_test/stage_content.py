from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.utils.stage import open_stage
import omni.usd

SCENE_PATH = "/home/teamb/Desktop/Sim2RealB/scenes/scene.usd"
H5_PATH = "/home/teamb/Desktop/Sim2RealB/data/20250827_151212.h5"



def main():
    open_stage(SCENE_PATH)

    stage = omni.usd.get_context().get_stage()

    if stage is None:
        print("ERROR: Could not access the USD stage.")
        return

    print("===== STAGE CONTENT =====")
    for prim in stage.Traverse():
        print(prim.GetPath())


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()