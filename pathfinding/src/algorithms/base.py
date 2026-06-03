from abc import ABC, abstractmethod

from pathfinding.src.core.models import GridMap, PathfindingResult, Position
from pathfinding.src.core.trace import RawAlgorithmStep


class PathfindingAlgorithm(ABC):
    name: str

    @abstractmethod
    def find_path(
            self,
            grid_map: GridMap,
            start: Position,
            goal: Position,
    ) -> PathfindingResult:
        pass

    @abstractmethod
    def find_path_with_steps(
            self,
            grid_map: GridMap,
            start: Position,
            goal: Position,
            step_record_interval: int = 10,
    ) -> tuple[PathfindingResult, list[RawAlgorithmStep]]:
        pass
