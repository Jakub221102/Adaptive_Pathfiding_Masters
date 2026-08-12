from dataclasses import dataclass

from pathfinding.src.core.models import Position


@dataclass(frozen=True, slots=True)
class MAPFAgent:
    agent_id: int
    start: Position
    goal: Position


@dataclass(frozen=True, slots=True, eq=False)
class TimedState:
    position: Position
    timestep: int

    def __post_init__(self) -> None:
        if self.timestep < 0:
            raise ValueError("timestep must be non-negative")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TimedState):
            return NotImplemented

        return (
            self.position.row == other.position.row
            and self.position.col == other.position.col
            and self.timestep == other.timestep
        )

    def __hash__(self) -> int:
        return hash((self.position.row, self.position.col, self.timestep))
