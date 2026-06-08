from pathfinding.src.core.dynamic_models import (
    DynamicGridMap,
    DynamicObstacleEvent,
    DynamicReplanningStats,
)
from pathfinding.src.core.models import GridMap, Scenario
from pathfinding.src.experiments.algorithm_registry import create_algorithm
from pathfinding.src.experiments.base_experiment import BaseExperiment
from pathfinding.src.experiments.dynamic_simulation import run_dynamic_simulation
from pathfinding.src.experiments.experiment_config import ExperimentConfig


class DynamicReplanningExperiment(BaseExperiment):
    def __init__(
            self,
            config: ExperimentConfig,
            events: list[DynamicObstacleEvent],
            *,
            grid_map: GridMap | None = None,
            scenario: Scenario | None = None,
    ) -> None:
        super().__init__(config)
        self.events = sorted(events, key=lambda event: event.step_index)
        self._grid_map_override = grid_map
        self._scenario_override = scenario

    def run(self) -> DynamicReplanningStats:
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

        stats, _ = run_dynamic_simulation(
            algorithm=algorithm,
            dynamic_map=dynamic_map,
            scenario=scenario,
            events=self.events,
            step_record_interval=self.config.step_record_interval,
            path_block_lookahead=self.config.path_block_lookahead,
            wait_when_no_path=self.config.wait_when_no_path,
            max_wait_steps=self.config.max_wait_steps,
            moving_obstacle_collision_policy=(
                self.config.moving_obstacle_collision_policy
            ),
            moving_obstacle_prediction_steps=(
                self.config.moving_obstacle_prediction_steps
            ),
        )
        self._print_stats(stats)
        return stats

    @staticmethod
    def _print_stats(stats: DynamicReplanningStats) -> None:
        print(f"Algorithm: {stats.algorithm_name}")
        print(f"Initial path found: {stats.initial_path_found}")
        print(f"Final goal reached: {stats.final_goal_reached}")
        print(f"Replanning count: {stats.replanning_count}")
        print(f"Dynamic events applied: {stats.dynamic_events_applied}")
        print(f"Waiting steps: {stats.waiting_steps}")
        print(f"Travelled steps: {stats.travelled_steps}")
        print(f"Total path cost: {stats.total_path_cost:.3f}")
        print(f"Total execution time: {stats.total_execution_time_ms:.3f} ms")
