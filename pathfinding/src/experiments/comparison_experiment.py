from pathfinding.src.experiments.algorithm_registry import create_algorithm
from pathfinding.src.experiments.base_experiment import BaseExperiment
from pathfinding.src.experiments.experiment_config import AlgorithmName, ExperimentConfig
from pathfinding.src.visualization.pygame_models import (
    AlgorithmAnimationVisualization,
    AlgorithmVisualization,
    RGBColor,
)


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

        if self.config.animated_comparison:
            animation_visualizations: list[AlgorithmAnimationVisualization] = []
            colors = self._get_default_colors()

            for index, algorithm_name in enumerate(self.algorithms):
                algorithm = create_algorithm(
                    name=algorithm_name,
                    cluster_size=self.config.cluster_size,
                )

                result, steps = algorithm.find_path_with_steps(
                    grid_map=grid_map,
                    start=scenario.start,
                    goal=scenario.goal,
                    step_record_interval=self.config.step_record_interval,
                )

                animation_visualizations.append(
                    AlgorithmAnimationVisualization(
                        name=result.algorithm_name,
                        result=result,
                        steps=steps,
                        color=colors[index],
                    )
                )

            viewer.run_grid_animation_comparison_view(
                start=scenario.start,
                goal=scenario.goal,
                visualizations=animation_visualizations,
                total_animation_time_ms=self.config.comparison_animation_time_ms,
            )

            return

        visualizations: list[AlgorithmVisualization] = []
        colors = self._get_default_colors()

        for index, algorithm_name in enumerate(self.algorithms):
            algorithm = create_algorithm(
                name=algorithm_name,
                cluster_size=self.config.cluster_size,
                max_entrances_per_cluster_pair=self.config.max_entrances_per_cluster_pair,
                min_entrance_width=self.config.min_entrance_width,
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
