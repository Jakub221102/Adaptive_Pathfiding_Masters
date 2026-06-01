from pathlib import Path

from code.src.core.models import GridMap


def load_moving_ai_map(path: Path) -> GridMap:
    lines = path.read_text().splitlines()

    height = None
    width = None
    map_start_index = None

    for index, line in enumerate(lines):
        if line.startswith("height"):
            height = int(line.split()[1])
        elif line.startswith("width"):
            width = int(line.split()[1])
        elif line.strip() == "map":
            map_start_index = index + 1
            break

    if height is None or width is None or map_start_index is None:
        raise ValueError(f"Invalid MovingAI map format: {path}")

    raw_map = lines[map_start_index:map_start_index + height]

    cells: list[list[int]] = []

    for row in raw_map:
        cells.append([
            0 if char in {".", "G", "S"} else 1
            for char in row
        ])

    return GridMap(
        name=path.name,
        width=width,
        height=height,
        cells=cells,
    )
