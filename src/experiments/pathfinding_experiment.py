from src.experiments.algorithm_registry import create_algorithm
from src.experiments.experiment_config import ExperimentConfig
from src.loaders.map_loader import load_moving_ai_map
from src.loaders.scen_loader import load_moving_ai_scenarios
from src.visualization.pygame_models import PygameViewerConfig, ViewerMode
from src.visualization.pygame_viewer import PygameGridViewer
from src.visualization.overlays.cluster_overlay import ClusterOverlay
from src.visualization.overlays.performance_overlay import PerformanceOverlay


class PathfindingExperiment:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def run(self) -> None:
        grid_map = load_moving_ai_map(self.config.map_path)
        scenarios = load_moving_ai_scenarios(self.config.scen_path)

        filtered_scenarios = [
            scenario for scenario in scenarios
            if scenario.optimal_length is not None
            and scenario.optimal_length > self.config.min_optimal_length
        ]

        if not filtered_scenarios:
            raise ValueError("No scenarios match the selected filter.")

        if self.config.scenario_index >= len(filtered_scenarios):
            raise IndexError(
                f"scenario_index={self.config.scenario_index} is out of range. "
                f"Available: {len(filtered_scenarios)}"
            )

        scenario = filtered_scenarios[self.config.scenario_index]
        algorithm = create_algorithm(self.config.algorithm)

        viewer = PygameGridViewer(
            grid_map=grid_map,
            config=PygameViewerConfig(
                fps=self.config.fps,
                draw_grid=False,
                window_title=f"{self.config.algorithm.value} on {grid_map.name}",
            ),
        )
        if self.config.show_cluster_overlay:
            viewer.add_overlay(
                ClusterOverlay(
                    grid_map=grid_map,
                    cell_size=viewer.cell_size,
                    cluster_size=self.config.cluster_size,
                    window_width=viewer.window_width,
                    window_height=viewer.window_height,
                )
            )

        if self.config.viewer_mode == ViewerMode.ANIMATED:
            result, steps = algorithm.find_path_with_steps(
                grid_map=grid_map,
                start=scenario.start,
                goal=scenario.goal,
                step_record_interval=self.config.step_record_interval,
            )

            viewer.add_overlay(
                PerformanceOverlay(result=result)
            )

            viewer.run_algorithm_animation(
                start=scenario.start,
                goal=scenario.goal,
                steps=steps,
                step_delay_ms=self.config.step_delay_ms,
                steps_per_frame=self.config.steps_per_frame,
            )

        elif self.config.viewer_mode == ViewerMode.STATIC:
            result = algorithm.find_path(
                grid_map=grid_map,
                start=scenario.start,
                goal=scenario.goal,
            )

            viewer.add_overlay(
                PerformanceOverlay(result=result)
            )

            viewer.run_static_path_view(
                start=scenario.start,
                goal=scenario.goal,
                path=result.path,
            )