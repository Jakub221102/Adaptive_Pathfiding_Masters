from pathfinding.src.algorithms.astar import AStar
from pathfinding.src.algorithms.base import PathfindingAlgorithm
from pathfinding.src.algorithms.dstar_lite import DStarLite
from pathfinding.src.algorithms.hpa.hpa_star import HPAStar
from pathfinding.src.algorithms.jps import JumpPointSearch
from pathfinding.src.experiments.experiment_config import AlgorithmName


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
        case AlgorithmName.JPS:
            return JumpPointSearch()
        case AlgorithmName.DSTAR_LITE:
            return DStarLite()
        case _:
            raise ValueError(f"Unsupported algorithm: {name}")
