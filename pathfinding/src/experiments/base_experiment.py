from abc import ABC, abstractmethod

from pathfinding.src.core.models import GridMap, Scenario
from pathfinding.src.experiments.experiment_config import ExperimentConfig
from pathfinding.src.loaders.map_loader import load_moving_ai_map
from pathfinding.src.loaders.scen_loader import load_moving_ai_scenarios
from pathfinding.src.visualization.pygame_models import PygameViewerConfig
from pathfinding.src.visualization.pygame_viewer import PygameGridViewer


class BaseExperiment(ABC):
    def __init__(
            self,
            config: ExperimentConfig,
    ) -> None:
        self.config = config

    def _load_map_and_scenario(self) -> tuple[GridMap, Scenario]:
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

        return grid_map, filtered_scenarios[self.config.scenario_index]

    def _load_filtered_scenarios(self) -> tuple[GridMap, list[Scenario]]:
        grid_map = load_moving_ai_map(self.config.map_path)
        scenarios = load_moving_ai_scenarios(self.config.scen_path)

        filtered_scenarios = [
            scenario for scenario in scenarios
            if scenario.optimal_length is not None
               and scenario.optimal_length > self.config.min_optimal_length
        ]

        if not filtered_scenarios:
            raise ValueError("No scenarios match the selected filter.")

        return grid_map, filtered_scenarios

    # simpler version
    # def _load_map_and_scenario(self) -> tuple[GridMap, Scenario]:
    #     grid_map, filtered_scenarios = self._load_filtered_scenarios()
    #
    #     if self.config.scenario_index >= len(filtered_scenarios):
    #         raise IndexError(
    #             f"scenario_index={self.config.scenario_index} is out of range. "
    #             f"Available: {len(filtered_scenarios)}"
    #         )
    #
    #     return grid_map, filtered_scenarios[self.config.scenario_index]

    def _create_viewer(
            self,
            grid_map: GridMap,
            title: str,
    ) -> PygameGridViewer:
        return PygameGridViewer(
            grid_map=grid_map,
            config=PygameViewerConfig(
                fps=self.config.fps,
                draw_grid=False,
                window_title=title,
            ),
        )

    @abstractmethod
    def run(self) -> None:
        pass
