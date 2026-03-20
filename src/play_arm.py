import h5py
import numpy as np
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.utils.stage import open_stage
from isaacsim.core.prims import SingleArticulation

SCENE_PATH = r"/home/teamb/Desktop/Sim2RealB/scenes/scene.usd"
H5_PATH    = r"/home/teamb/Desktop/Sim2RealB/data/20250827_151212.h5"

with h5py.File(H5_PATH, "r") as f:
    print(f.attrs.keys())          # check for a freq/hz attribute
    print(dict(f.attrs))           # print all top-level metadata
    if "timestamps" in f:
        ts = np.array(f["timestamps"])
        dt = np.mean(np.diff(ts))
        print(f"Recorded at ~{1/dt:.1f} Hz")

CHECK = False
if CHECK:
        
    # -------------------
    # setup
    # -------------------
    open_stage(SCENE_PATH)
    world = World()
    robot = SingleArticulation("/World/Franka", name="franka")

    world.reset()
    robot.initialize()

    world.play()  # start simulation

    print("DOF names:", robot.dof_names)

    # -------------------
    # load dataset
    # -------------------
    with h5py.File(H5_PATH, "r") as f:
        q_arm = np.array(f["observations/qpos_arm_left"])

    print("Trajectory shape:", q_arm.shape)  # expect (2331, 7)

    # -------------------
    # replay
    # -------------------
    frame = 0
    while simulation_app.is_running():

        if not world.is_playing():
            world.step(render=True)
            continue

        if frame >= len(q_arm):
            print("Replay finished.")
            break

        # set joint positions before stepping
        q = np.zeros(9)
        q[:7] = q_arm[frame]  # 7 arm joints
        q[7:] = 0.04          # both fingers open

        robot.set_joint_positions(q)
        frame += 1

        world.step(render=True)

    simulation_app.close()