from pathlib import Path

from src.experiments.hpa_debug_experiment import HPADebugExperiment
from src.loaders.map_loader import load_moving_ai_map


def main() -> None:
    grid_map = load_moving_ai_map(
        Path("Data/bg512-map/AR0204SR.map")
    )

    experiment = HPADebugExperiment(
        grid_map=grid_map,
        cluster_size=32,
    )

    experiment.run()


if __name__ == "__main__":
    main()