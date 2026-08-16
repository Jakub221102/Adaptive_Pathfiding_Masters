from pathfinding.src.algorithms.mapf.models import AgentPath, TimedState
from pathfinding.src.algorithms.mapf.smoke_scenario import build_mapf_scenario
from pathfinding.src.core.models import Position, Scenario
from pathfinding.src.visualization.mapf_viewer_helpers import (
    animation_max_timestep,
    compute_mapf_viewport,
    position_at_timestep,
    position_at_timestep_as_position,
)
from pathfinding.tests.helpers import build_grid_map


def _path(agent_id: int, coordinates: list[tuple[int, int, int]]) -> AgentPath:
    return AgentPath(
        agent_id=agent_id,
        states=tuple(
            TimedState(row=row, col=col, timestep=timestep)
            for row, col, timestep in coordinates
        ),
    )


def _scenario(
    start_row: int,
    start_col: int,
    goal_row: int,
    goal_col: int,
) -> Scenario:
    return Scenario(
        map_name="test.map",
        width=10,
        height=10,
        start=Position(row=start_row, col=start_col),
        goal=Position(row=goal_row, col=goal_col),
        optimal_length=1.0,
    )


def test_position_at_timestep_returns_explicit_state() -> None:
    path = _path(0, [(2, 3, 0), (2, 4, 1), (2, 5, 2)])

    assert position_at_timestep(path, 1) == (2, 4)
    assert position_at_timestep_as_position(path, 1) == Position(row=2, col=4)


def test_position_at_timestep_returns_final_position_after_path_end() -> None:
    path = _path(0, [(1, 1, 0), (1, 2, 1), (1, 3, 2)])

    assert position_at_timestep(path, 5) == (1, 3)
    assert position_at_timestep(path, 2) == (1, 3)


def test_position_at_timestep_supports_wait() -> None:
    path = _path(0, [(4, 4, 0), (4, 4, 1), (4, 5, 2)])

    assert position_at_timestep(path, 0) == (4, 4)
    assert position_at_timestep(path, 1) == (4, 4)
    assert position_at_timestep(path, 2) == (4, 5)


def test_animation_max_timestep_equals_makespan() -> None:
    paths = (
        _path(0, [(0, 0, 0), (0, 1, 1), (0, 2, 2)]),
        _path(1, [(1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 0, 3), (1, 1, 4)]),
    )

    assert animation_max_timestep(paths) == 4


def test_compute_mapf_viewport_includes_paths_starts_and_goals() -> None:
    grid_map = build_grid_map([[0 for _ in range(20)] for _ in range(20)])
    scenarios = [
        _scenario(0, 0, 0, 9),
        _scenario(5, 0, 5, 9),
    ]
    mapf_scenario = build_mapf_scenario(
        scenarios=scenarios,
        grid_map=grid_map,
        agent_count=2,
    )
    paths = (
        _path(0, [(0, 0, 0), (0, 5, 1), (0, 9, 2)]),
        _path(1, [(5, 0, 0), (5, 9, 1)]),
    )

    viewport = compute_mapf_viewport(
        grid_map=grid_map,
        scenario=mapf_scenario,
        paths=paths,
        padding=0,
    )

    assert viewport.min_row == 0
    assert viewport.max_row == 5
    assert viewport.min_col == 0
    assert viewport.max_col == 9


def test_compute_mapf_viewport_applies_padding() -> None:
    grid_map = build_grid_map([[0 for _ in range(20)] for _ in range(20)])
    scenarios = [_scenario(5, 5, 5, 6)]
    mapf_scenario = build_mapf_scenario(
        scenarios=scenarios,
        grid_map=grid_map,
        agent_count=1,
    )
    paths = (_path(0, [(5, 5, 0), (5, 6, 1)]),)

    viewport = compute_mapf_viewport(
        grid_map=grid_map,
        scenario=mapf_scenario,
        paths=paths,
        padding=2,
    )

    assert viewport.min_row == 3
    assert viewport.max_row == 7
    assert viewport.min_col == 3
    assert viewport.max_col == 8


def test_compute_mapf_viewport_is_clamped_to_map_bounds() -> None:
    grid_map = build_grid_map([[0 for _ in range(6)] for _ in range(6)])
    scenarios = [_scenario(0, 0, 0, 5)]
    mapf_scenario = build_mapf_scenario(
        scenarios=scenarios,
        grid_map=grid_map,
        agent_count=1,
    )
    paths = (_path(0, [(0, 0, 0), (0, 5, 1)]),)

    viewport = compute_mapf_viewport(
        grid_map=grid_map,
        scenario=mapf_scenario,
        paths=paths,
        padding=10,
    )

    assert viewport.min_row == 0
    assert viewport.min_col == 0
    assert viewport.max_row == 5
    assert viewport.max_col == 5


def test_compute_mapf_viewport_is_deterministic() -> None:
    grid_map = build_grid_map([[0 for _ in range(10)] for _ in range(10)])
    scenarios = [_scenario(1, 1, 2, 8)]
    mapf_scenario = build_mapf_scenario(
        scenarios=scenarios,
        grid_map=grid_map,
        agent_count=1,
    )
    paths = (_path(0, [(1, 1, 0), (2, 8, 1)]),)

    first = compute_mapf_viewport(
        grid_map=grid_map,
        scenario=mapf_scenario,
        paths=paths,
        padding=1,
    )
    second = compute_mapf_viewport(
        grid_map=grid_map,
        scenario=mapf_scenario,
        paths=paths,
        padding=1,
    )

    assert first == second
