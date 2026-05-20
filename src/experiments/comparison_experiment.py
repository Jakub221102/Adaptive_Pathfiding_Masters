from src.experiments.algorithm_registry import create_algorithm
from src.experiments.base_experiment import BaseExperiment
from src.experiments.experiment_config import AlgorithmName, ExperimentConfig
from src.visualization.overlays.performance_overlay import PerformanceOverlay
from src.visualization.pygame_models import AlgorithmVisualization, RGBColor


class ComparisonExperiment(BaseExperiment):
    def __init__(
        self,
        config: ExperimentConfig,
        algorithms: list[AlgorithmName],
    ) -> None:
        super().__init__(config)

        if not algorithms:
            raise ValueError("At least one algorithm is required.")

        if len(algorithms) > 9:
            raise ValueError("Comparison supports up to 9 algorithms.")

        self.algorithms = algorithms

    def run(self) -> None:
        grid_map, scenario = self._load_map_and_scenario()

        viewer = self._create_viewer(
            grid_map=grid_map,
            title="Algorithm comparison",
        )

        visualizations: list[AlgorithmVisualization] = []

        colors = self._get_default_colors()

        for index, algorithm_name in enumerate(self.algorithms):
            algorithm = create_algorithm(
                name=algorithm_name,
                cluster_size=self.config.cluster_size,
            )

            result = algorithm.find_path(
                grid_map=grid_map,
                start=scenario.start,
                goal=scenario.goal,
            )

            visualizations.append(
                AlgorithmVisualization(
                    name=result.algorithm_name,
                    path=result.path,
                    color=colors[index],
                    result=result,
                )
            )

        viewer.run_grid_comparison_view(
            start=scenario.start,
            goal=scenario.goal,
            visualizations=visualizations,
        )

    @staticmethod
    def _get_default_colors() -> list[RGBColor]:
        return [
            RGBColor(r=255, g=220, b=0),
            RGBColor(r=0, g=160, b=255),
            RGBColor(r=0, g=220, b=120),
            RGBColor(r=255, g=120, b=0),
            RGBColor(r=200, g=0, b=255),
            RGBColor(r=255, g=80, b=120),
            RGBColor(r=120, g=255, b=255),
            RGBColor(r=180, g=180, b=180),
            RGBColor(r=255, g=255, b=255),
        ]