from src.algorithms.astar import AStar
from src.algorithms.hpa_star import HPAStar
from src.experiments.experiment_config import AlgorithmName


def create_algorithm(name: AlgorithmName):
    match name:
        case AlgorithmName.ASTAR:
            return AStar()
        case AlgorithmName.HPA_STAR:
            return HPAStar(cluster_size=32)
        case _:
            raise ValueError(f"Unsupported algorithm: {name}")