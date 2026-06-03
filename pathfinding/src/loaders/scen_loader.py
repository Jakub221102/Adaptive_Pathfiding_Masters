from pathlib import Path

from pathfinding.src.core.models import Position, Scenario


def load_moving_ai_scenarios(path: Path, limit: int | None = None) -> list[Scenario]:
    lines = path.read_text().splitlines()

    scenarios: list[Scenario] = []

    for line in lines[1:]:
        parts = line.split()

        if len(parts) < 9:
            continue

        scenario = Scenario(
            map_name=parts[1],
            width=int(parts[2]),
            height=int(parts[3]),
            start=Position(row=int(parts[5]), col=int(parts[4])),
            goal=Position(row=int(parts[7]), col=int(parts[6])),
            optimal_length=float(parts[8]),
        )

        scenarios.append(scenario)

        if limit is not None and len(scenarios) >= limit:
            break

    return scenarios
