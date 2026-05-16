import matplotlib.pyplot as plt
import numpy as np

from src.core.models import GridMap, Position


def plot_path(
    grid_map: GridMap,
    path: list[Position],
    start: Position,
    goal: Position,
) -> None:
    grid_array = np.array(grid_map.cells)

    plt.figure(figsize=(8, 8))
    plt.imshow(grid_array, cmap="gray_r")

    if path:
        rows = [position.row for position in path]
        cols = [position.col for position in path]
        plt.plot(cols, rows, linewidth=2)

    plt.scatter(start.col, start.row, marker="o", s=80, label="Start")
    plt.scatter(goal.col, goal.row, marker="x", s=80, label="Goal")

    plt.title(f"Pathfinding result: {grid_map.name}")
    plt.legend()
    plt.show()