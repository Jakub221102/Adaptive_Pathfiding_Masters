from pathlib import Path

from code.src.experiments.experiment_config import ExperimentConfig
from code.src.experiments.failed_scenario_analysis_experiment import (
    FailedScenarioAnalysisExperiment,
)


def main() -> None:
    config = ExperimentConfig(
        map_path=Path("../../Data/bg512-map/AR0204SR.map"),
        scen_path=Path("../../Data/bg512-scen/AR0204SR.map.scen"),
        min_optimal_length=100,
        cluster_size=128,
        benchmark_scenarios=100,
    )

    experiment = FailedScenarioAnalysisExperiment(config)
    experiment.run()


if __name__ == "__main__":
    main()
