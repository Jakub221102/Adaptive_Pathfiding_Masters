from pathfinding.src.algorithms.astar import AStar
from pathfinding.src.core.models import GridMap, Position

LocalPathCacheKey = tuple[tuple[int, int], tuple[int, int], int | None]


class LocalPathCache:
    def __init__(self) -> None:
        self._astar = AStar()
        self._cache: dict[LocalPathCacheKey, list[Position]] = {}
        self.hits: int = 0
        self.misses: int = 0

    def clear(self) -> None:
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    def size(self) -> int:
        return len(self._cache)

    def find_path(
            self,
            grid_map: GridMap,
            start: Position,
            goal: Position,
            cluster_id: int | None,
            allowed_positions: set[tuple[int, int]] | None,
    ) -> list[Position]:
        cache_key = self._build_key(
            start=start,
            goal=goal,
            cluster_id=cluster_id,
        )

        if cache_key in self._cache:
            self.hits += 1
            return self._cache[cache_key]

        self.misses += 1

        result = self._astar.find_path(
            grid_map=grid_map,
            start=start,
            goal=goal,
            allowed_positions=allowed_positions,
        )

        if not result.found:
            self._cache[cache_key] = []
            return []

        self._cache[cache_key] = result.path
        return result.path

    @staticmethod
    def _build_key(
            start: Position,
            goal: Position,
            cluster_id: int | None,
    ) -> LocalPathCacheKey:
        return (
            (start.row, start.col),
            (goal.row, goal.col),
            cluster_id,
        )
