from pathlib import Path

import pytest

from pathfinding.src.algorithms.astar import AStar
from pathfinding.src.algorithms.hpa.hpa_star import HPAStar
from pathfinding.src.algorithms.jps import JumpPointSearch
from pathfinding.src.loaders.map_loader import load_moving_ai_map
from pathfinding.src.loaders.scen_loader import load_moving_ai_scenarios

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


def test_static_algorithms_on_movingai_sample() -> None:
    grid_map = load_moving_ai_map(
        FIXTURES_DIR / "AR0204SR_sample.map"
    )

    scenarios = load_moving_ai_scenarios(
        FIXTURES_DIR / "AR0204SR_sample.map.scen"
    )

    scenario = scenarios[0]

    astar_result = AStar().find_path(
        grid_map=grid_map,
        start=scenario.start,
        goal=scenario.goal,
    )

    jps_result = JumpPointSearch().find_path(
        grid_map=grid_map,
        start=scenario.start,
        goal=scenario.goal,
    )

    hpa_result = HPAStar(
        cluster_size=16,
        max_entrances_per_cluster_pair=2,
    ).find_path(
        grid_map=grid_map,
        start=scenario.start,
        goal=scenario.goal,
    )

    assert astar_result.found is True
    assert jps_result.found is True
    assert hpa_result.found is True

    assert astar_result.path_cost == pytest.approx(
        scenario.optimal_length,
        abs=1.0,
    )

    assert jps_result.path_cost == pytest.approx(
        astar_result.path_cost,
        abs=1e-6,
    )

    assert hpa_result.path_cost >= astar_result.path_cost
    assert hpa_result.path_cost <= astar_result.path_cost * 1.30


def test_static_algorithms_quality_does_not_regress() -> None:
    grid_map = load_moving_ai_map(
        FIXTURES_DIR / "AR0204SR_sample.map"
    )

    scenarios = load_moving_ai_scenarios(
        FIXTURES_DIR / "AR0204SR_sample.map.scen"
    )

    scenario = scenarios[0]

    astar_result = AStar().find_path(
        grid_map=grid_map,
        start=scenario.start,
        goal=scenario.goal,
    )

    jps_result = JumpPointSearch().find_path(
        grid_map=grid_map,
        start=scenario.start,
        goal=scenario.goal,
    )

    hpa_result = HPAStar(
        cluster_size=16,
        max_entrances_per_cluster_pair=2,
    ).find_path(
        grid_map=grid_map,
        start=scenario.start,
        goal=scenario.goal,
    )

    assert astar_result.found is True
    assert jps_result.found is True
    assert hpa_result.found is True

    assert astar_result.path_cost <= scenario.optimal_length + 1.0
    assert jps_result.path_cost <= astar_result.path_cost + 1e-6
    assert hpa_result.path_cost <= astar_result.path_cost * 1.30

    assert astar_result.visited_nodes <= 2_000
    assert jps_result.visited_nodes <= 100
    assert hpa_result.visited_nodes <= 100
