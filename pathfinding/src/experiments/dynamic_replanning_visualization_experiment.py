from pathfinding.src.core.dynamic_models import (
    DynamicGridMap,
    DynamicObstacleEvent,
    MovingObstacle,
)
from pathfinding.src.core.models import GridMap, Scenario
from pathfinding.src.experiments.algorithm_registry import create_algorithm
from pathfinding.src.experiments.base_experiment import BaseExperiment
from pathfinding.src.experiments.dynamic_replanning_experiment import (
    DynamicReplanningExperiment,
)
from pathfinding.src.experiments.dynamic_simulation import run_dynamic_simulation
from pathfinding.src.experiments.experiment_config import ExperimentConfig
from pathfinding.src.visualization.dynamic_frame_presenter import DynamicFramePresenter


class DynamicReplanningVisualizationExperiment(BaseExperiment):
    def __init__(
            self,
            config: ExperimentConfig,
            events: list[DynamicObstacleEvent] | None = None,
            *,
            moving_obstacles: list[MovingObstacle] | None = None,
            grid_map: GridMap | None = None,
            scenario: Scenario | None = None,
    ) -> None:
        super().__init__(config)
        self.events = sorted(events or [], key=lambda event: event.step_index)
        self.moving_obstacles = list(moving_obstacles or [])
        self._grid_map_override = grid_map
        self._scenario_override = scenario

    def run(self) -> None:
        if self._grid_map_override is not None and self._scenario_override is not None:
            grid_map = self._grid_map_override
            scenario = self._scenario_override
        else:
            grid_map, scenario = self._load_map_and_scenario()

        dynamic_map = DynamicGridMap(base_map=grid_map)

        algorithm = create_algorithm(
            name=self.config.algorithm,
            cluster_size=self.config.cluster_size,
            max_entrances_per_cluster_pair=self.config.max_entrances_per_cluster_pair,
            min_entrance_width=self.config.min_entrance_width,
        )

        viewer = self._create_viewer(
            grid_map=grid_map,
            title=(
                f"Dynamic replanning: {self.config.algorithm.value} "
                f"on {grid_map.name}"
            ),
        )
        viewer.initialize_for_dynamic_animation()

        presenter = DynamicFramePresenter(
            viewer=viewer,
            start=scenario.start,
            goal=scenario.goal,
            config=self.config,
        )

        stats, _ = run_dynamic_simulation(
            algorithm=algorithm,
            dynamic_map=dynamic_map,
            scenario=scenario,
            events=self.events,
            moving_obstacles=self.moving_obstacles,
            frame_callback=presenter.present_keyframe,
            step_record_interval=self.config.step_record_interval,
            path_block_lookahead=self.config.path_block_lookahead,
            wait_when_no_path=self.config.wait_when_no_path,
            max_wait_steps=self.config.max_wait_steps,
        )

        presenter.finalize()
        viewer.finalize_dynamic_animation()
        DynamicReplanningExperiment._print_stats(stats)
