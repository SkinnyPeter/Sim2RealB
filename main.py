from src.h5_analyzer import H5Analyzer
from isaacsim import SimulationApp

# CONFIGURATION
data_path = "C:/Users/pardo/Desktop/3DV/quat/Sim2RealB/data/20250826_111157.h5"
# data_path = ""data/20250827_151212.h5""

scence_path = "C:/Users/pardo/Desktop/3DV/quat/Sim2RealB/scenes/scene.usd"
def main():
    # Create an analyzer with the default file path
    analyzer = H5Analyzer(data_path)

    # Use the inspect() method
    #print("Inspecting HDF5 file...")
    #analyzer.inspect()

    # Use the play_video() method
    print("\nPlaying video from HDF5 file...")
    #analyzer.play_video()

    simulation_app = SimulationApp({"headless": True})

    # Simulator needs to be import after simulation_app is created
    from src.simulator import Simulator

    simulator = Simulator(simulation_app, scence_path,data_path)
    simulator.play()

    simulation_app.close()

if __name__ == "__main__":
    main()