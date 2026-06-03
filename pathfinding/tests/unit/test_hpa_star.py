from pathfinding.src.algorithms.astar import AStar
from pathfinding.src.algorithms.hpa.hpa_star import HPAStar
from pathfinding.src.core.models import Position
from pathfinding.tests.helpers import build_grid_map


def test_hpa_star_finds_path_on_open_map() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    start = Position(row=0, col=0)
    goal = Position(row=7, col=7)

    astar_result = AStar().find_path(grid_map, start, goal)
    hpa_result = HPAStar(
        cluster_size=4,
        max_entrances_per_cluster_pair=2,
    ).find_path(grid_map, start, goal)

    assert hpa_result.found is True
    assert hpa_result.path_cost >= astar_result.path_cost


def test_hpa_star_preprocessing_stats_are_collected() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    hpa = HPAStar(
        cluster_size=4,
        max_entrances_per_cluster_pair=2,
    )

    hpa.preprocess_map(grid_map)
    stats = hpa.get_preprocessing_stats()

    assert stats.cluster_count > 0
    assert stats.entrance_count > 0
    assert stats.abstract_node_count > 0
    assert stats.abstract_edge_count > 0
    assert stats.preprocessing_time_ms >= 0
