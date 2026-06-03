from pathfinding.src.core.models import GridMap


def build_grid_map(
        cells: list[list[int]],
        name: str = "test_map",
) -> GridMap:
    return GridMap(
        name=name,
        width=len(cells[0]),
        height=len(cells),
        cells=cells,
    )
