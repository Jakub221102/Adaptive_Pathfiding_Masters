from pathfinding.src.algorithms.mapf.metrics import makespan, sum_of_costs
from pathfinding.src.algorithms.mapf.models import AgentPath, MAPFAgent, MAPFScenario, TimedState
from pathfinding.src.algorithms.mapf.prioritized_planning import plan_prioritized
from pathfinding.src.core.models import Position
from pathfinding.tests.helpers import build_grid_map


def _path(agent_id: int, coordinates: list[tuple[int, int, int]]) -> AgentPath:
    return AgentPath(
        agent_id=agent_id,
        states=tuple(
            TimedState(row=row, col=col, timestep=timestep)
            for row, col, timestep in coordinates
        ),
    )


def test_one_state_path_contributes_zero_to_sum_of_costs() -> None:
    path = _path(0, [(0, 0, 0)])

    assert sum_of_costs((path,)) == 0


def test_three_state_path_contributes_two_to_sum_of_costs() -> None:
    path = _path(0, [(0, 0, 0), (0, 1, 1), (0, 2, 2)])

    assert sum_of_costs((path,)) == 2


def test_sum_of_costs_across_multiple_paths() -> None:
    paths = (
        _path(0, [(0, 0, 0), (0, 1, 1), (0, 2, 2)]),
        _path(1, [(1, 0, 0), (1, 1, 1), (1, 2, 2), (1, 3, 3), (1, 4, 4), (1, 5, 5)]),
        _path(2, [(2, 0, 0), (2, 1, 1), (2, 2, 2), (2, 3, 3)]),
    )

    assert sum_of_costs(paths) == 10


def test_wait_counts_toward_sum_of_costs() -> None:
    path = _path(0, [(0, 0, 0), (0, 0, 1), (0, 1, 2)])

    assert sum_of_costs((path,)) == 2


def test_coordinates_do_not_affect_sum_of_costs() -> None:
    short_path = _path(0, [(0, 0, 0), (0, 0, 1), (0, 1, 2)])
    long_spatial_path = _path(1, [(0, 0, 0), (5, 9, 1), (2, 3, 2)])

    assert sum_of_costs((short_path,)) == sum_of_costs((long_spatial_path,))


def test_non_zero_start_timestep_uses_transition_count_for_sum_of_costs() -> None:
    path = _path(0, [(0, 0, 5), (0, 1, 6), (0, 2, 7)])

    assert sum_of_costs((path,)) == 2
    assert path.states[-1].timestep == 7


def test_one_state_path_produces_makespan_zero() -> None:
    path = _path(0, [(0, 0, 0)])

    assert makespan((path,)) == 0


def test_makespan_returns_largest_individual_path_cost() -> None:
    paths = (
        _path(0, [(0, 0, 0), (0, 1, 1), (0, 2, 2)]),
        _path(1, [(1, 0, 0), (1, 1, 1), (1, 2, 2), (1, 3, 3), (1, 4, 4), (1, 5, 5)]),
        _path(2, [(2, 0, 0), (2, 1, 1), (2, 2, 2), (2, 3, 3)]),
    )

    assert makespan(paths) == 5


def test_wait_counts_toward_makespan() -> None:
    move_path = _path(0, [(0, 0, 0), (0, 1, 1)])
    wait_path = _path(1, [(1, 0, 0), (1, 0, 1), (1, 1, 2)])

    assert makespan((move_path,)) == 1
    assert makespan((wait_path,)) == 2
    assert makespan((move_path, wait_path)) == 2


def test_non_zero_start_timestep_does_not_inflate_makespan() -> None:
    path = _path(0, [(0, 0, 5), (0, 1, 6), (0, 2, 7)])

    assert makespan((path,)) == 2
    assert path.states[-1].timestep == 7


def test_empty_paths_have_zero_sum_of_costs() -> None:
    assert sum_of_costs(()) == 0


def test_empty_paths_have_zero_makespan() -> None:
    assert makespan(()) == 0


def test_makespan_is_at_most_sum_of_costs_for_multiple_paths() -> None:
    paths = (
        _path(0, [(0, 0, 0), (0, 1, 1), (0, 2, 2)]),
        _path(1, [(1, 0, 0), (1, 1, 1), (1, 2, 2), (1, 3, 3), (1, 4, 4), (1, 5, 5)]),
        _path(2, [(2, 0, 0), (2, 1, 1), (2, 2, 2), (2, 3, 3)]),
    )

    assert makespan(paths) <= sum_of_costs(paths)


def test_prioritized_planning_paths_produce_expected_metrics() -> None:
    grid_map = build_grid_map([[0, 0, 0]])
    scenario = MAPFScenario(
        agents=(
            MAPFAgent(
                agent_id=0,
                start=Position(row=0, col=0),
                goal=Position(row=0, col=2),
            ),
        )
    )

    result = plan_prioritized(grid_map=grid_map, scenario=scenario, max_timestep=5)

    assert result.success is True
    assert sum_of_costs(result.paths) == sum(len(path.states) - 1 for path in result.paths)
    assert makespan(result.paths) == max(len(path.states) - 1 for path in result.paths)
