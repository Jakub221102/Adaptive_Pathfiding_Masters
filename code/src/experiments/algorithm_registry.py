from code.src.algorithms.astar import AStar
from code.src.algorithms.base import PathfindingAlgorithm
from code.src.algorithms.hpa.hpa_star import HPAStar
from code.src.experiments.experiment_config import AlgorithmName


def create_algorithm(
        name: AlgorithmName,
        cluster_size: int = 32,
        max_entrances_per_cluster_pair: int = 2,
        min_entrance_width: int = 1,
) -> PathfindingAlgorithm:
    match name:
        case AlgorithmName.ASTAR:
            return AStar()
        case AlgorithmName.HPA_STAR:
            return HPAStar(
                cluster_size=cluster_size,
                max_entrances_per_cluster_pair=max_entrances_per_cluster_pair,
                min_entrance_width=min_entrance_width,
            )
        case _:
            raise ValueError(f"Unsupported algorithm: {name}")
