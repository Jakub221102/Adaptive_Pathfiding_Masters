from src.algorithms.hpa.hpa_star import HPAStar
from src.core.models import GridMap, Position


class HPADebugExperiment:
    def __init__(
        self,
        grid_map: GridMap,
        cluster_size: int = 32,
        start: Position | None = None,
        goal: Position | None = None,
    ) -> None:
        self.grid_map = grid_map
        self.cluster_size = cluster_size
        self.start = start
        self.goal = goal

    def run(self) -> None:
        hpa = HPAStar(cluster_size=self.cluster_size, max_entrances_per_cluster_pair=2)

        hpa.preprocess_map(self.grid_map)

        print("Preprocessing stats:")
        print(hpa.get_preprocessing_stats().model_dump_json(indent=2))

        if self.start is None or self.goal is None:
            return

        result = hpa.find_path(
            grid_map=self.grid_map,
            start=self.start,
            goal=self.goal,
        )

        print("First query stats:")
        print(hpa.get_last_query_stats().model_dump_json(indent=2))

        result = hpa.find_path(
            grid_map=self.grid_map,
            start=self.start,
            goal=self.goal,
        )

        print("Second query stats:")
        print(hpa.get_last_query_stats().model_dump_json(indent=2))