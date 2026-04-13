from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from pathlib import Path
import omni.usd
import time


BASE_DIR = Path(__file__).resolve().parent.parent

LEFT_IN = BASE_DIR / "assets" / "connectors" / "connector_left.usd"
RIGHT_IN = BASE_DIR / "assets" / "connectors" / "connector_right.usd"

LEFT_OUT = BASE_DIR / "assets" / "connectors" / "connector_left_flat.usd"
RIGHT_OUT = BASE_DIR / "assets" / "connectors" / "connector_right_flat.usd"


def flatten_one(input_usd: Path, output_usd: Path):
    ctx = omni.usd.get_context()

    print(f"Opening: {input_usd}")
    ctx.open_stage(str(input_usd))

    for _ in range(40):
        simulation_app.update()
        time.sleep(0.05)

    stage = ctx.get_stage()
    if stage is None:
        raise RuntimeError(f"Could not open stage: {input_usd}")

    print("Flattening stage...")
    flat_stage = stage.Flatten()

    output_usd.parent.mkdir(parents=True, exist_ok=True)
    flat_stage.Export(str(output_usd))

    print(f"Saved flattened USD to: {output_usd}")


def main():
    if LEFT_IN.exists():
        flatten_one(LEFT_IN, LEFT_OUT)
    else:
        print(f"Missing: {LEFT_IN}")

    if RIGHT_IN.exists():
        flatten_one(RIGHT_IN, RIGHT_OUT)
    else:
        print(f"Missing: {RIGHT_IN}")

    print("Done.")
    simulation_app.close()


if __name__ == "__main__":
    main()