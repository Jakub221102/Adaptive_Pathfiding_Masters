from src.algorithms.astar import AStar
from src.algorithms.base import PathfindingAlgorithm
from src.algorithms.hpa.hpa_star import HPAStar
from src.experiments.experiment_config import AlgorithmName


def create_algorithm(
    name: AlgorithmName,
    cluster_size: int = 32,
) -> PathfindingAlgorithm:
    match name:
        case AlgorithmName.ASTAR:
            return AStar()
        case AlgorithmName.HPA_STAR:
            return HPAStar(cluster_size=cluster_size, max_entrances_per_cluster_pair=2)
        case _:
            raise ValueError(f"Unsupported algorithm: {name}")