from src.experiments.algorithm_registry import create_algorithm
from src.experiments.base_experiment import BaseExperiment
from src.visualization.overlays.cluster_overlay import ClusterOverlay
from src.visualization.overlays.performance_overlay import PerformanceOverlay
from src.visualization.pygame_models import ViewerMode


class PathfindingExperiment(BaseExperiment):
    def run(self) -> None:
        grid_map, scenario = self._load_map_and_scenario()

        algorithm = create_algorithm(
            name=self.config.algorithm,
            cluster_size=self.config.cluster_size,
        )

        viewer = self._create_viewer(
            grid_map=grid_map,
            title=f"{self.config.algorithm.value} on {grid_map.name}",
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

            viewer.add_overlay(PerformanceOverlay(result=result))

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

            viewer.add_overlay(PerformanceOverlay(result=result))

            viewer.run_static_path_view(
                start=scenario.start,
                goal=scenario.goal,
                path=result.path,
            )
        else:
            raise ValueError(f"Unsupported viewer mode: {self.config.viewer_mode}")