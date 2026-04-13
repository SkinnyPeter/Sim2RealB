from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from pathlib import Path
import omni.usd
import time


BASE_DIR = Path(__file__).resolve().parent.parent

LEFT_IN = BASE_DIR / "assets" / "orca" / "scene_left" / "scene_left.usd"
RIGHT_IN = BASE_DIR / "assets" / "orca" / "scene_right" / "scene_right.usd"

OUT_DIR = BASE_DIR / "assets" / "orca_flat"
LEFT_OUT = OUT_DIR / "scene_left_flat.usd"
RIGHT_OUT = OUT_DIR / "scene_right_flat.usd"


def flatten_one(input_usd: Path, output_usd: Path):
    ctx = omni.usd.get_context()

    print(f"\nOpening: {input_usd}")
    ctx.open_stage(str(input_usd))

    # Let Isaac load everything
    for _ in range(50):
        simulation_app.update()
        time.sleep(0.05)

    stage = ctx.get_stage()
    if stage is None:
        raise RuntimeError(f"Could not open stage: {input_usd}")

    print("Flattening stage...")
    flat_stage = stage.Flatten()

    output_usd.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving flattened USD to: {output_usd}")
    flat_stage.Export(str(output_usd))


def main():
    flatten_one(LEFT_IN, LEFT_OUT)
    flatten_one(RIGHT_IN, RIGHT_OUT)

    print("\n✅ ORCA assets flattened successfully.")
    simulation_app.close()


if __name__ == "__main__":
    main()