from pathfinding.src.core.hpa_models import AbstractEdge, Cluster
from pathfinding.src.core.models import GridMap, Position
from pathfinding.src.core.trace import RawAlgorithmStep


def build_hpa_steps_from_edges(
        grid_map: GridMap,
        abstract_edges: list[AbstractEdge],
        cluster_lookup: dict[tuple[int, int], Cluster],
        corridor_radius: int = 2,
        include_cluster_cells: bool = True,
) -> list[RawAlgorithmStep]:
    steps: list[RawAlgorithmStep] = []

    visited_nodes: set[tuple[int, int]] = set()
    accumulated_path: list[tuple[int, int]] = []

    for edge in abstract_edges:
        if not edge.path:
            continue

        for position in edge.path:
            node = (position.row, position.col)

            if include_cluster_cells:
                cluster = cluster_lookup.get(node)

                if cluster is not None:
                    add_cluster_cells_to_visited(
                        grid_map=grid_map,
                        cluster=cluster,
                        visited_nodes=visited_nodes,
                    )

            add_corridor_cells_to_visited(
                grid_map=grid_map,
                position=position,
                radius=corridor_radius,
                visited_nodes=visited_nodes,
            )

            if not accumulated_path or accumulated_path[-1] != node:
                accumulated_path.append(node)

        current_position = edge.path[-1]

        steps.append(
            RawAlgorithmStep(
                current=(current_position.row, current_position.col),
                open_nodes=frozenset(),
                closed_nodes=frozenset(visited_nodes),
                path=tuple(accumulated_path),
            )
        )

    return steps


def add_corridor_cells_to_visited(
        grid_map: GridMap,
        position: Position,
        radius: int,
        visited_nodes: set[tuple[int, int]],
) -> None:
    for row_offset in range(-radius, radius + 1):
        for col_offset in range(-radius, radius + 1):
            row = position.row + row_offset
            col = position.col + col_offset

            candidate = Position(row=row, col=col)

            if grid_map.in_bounds(candidate) and grid_map.is_walkable(candidate):
                visited_nodes.add((row, col))


def add_cluster_cells_to_visited(
        grid_map: GridMap,
        cluster: Cluster,
        visited_nodes: set[tuple[int, int]],
) -> None:
    for row in range(cluster.row_start, cluster.row_end):
        for col in range(cluster.col_start, cluster.col_end):
            position = Position(row=row, col=col)

            if grid_map.is_walkable(position):
                visited_nodes.add((row, col))
