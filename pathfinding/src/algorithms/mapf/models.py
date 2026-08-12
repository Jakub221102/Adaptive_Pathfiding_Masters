from __future__ import annotations

from dataclasses import dataclass

from pathfinding.src.core.models import Position


@dataclass(frozen=True, slots=True)
class MAPFAgent:
    agent_id: int
    start: Position
    goal: Position


@dataclass(frozen=True, slots=True)
class TimedState:
    row: int
    col: int
    timestep: int

    def __post_init__(self) -> None:
        if self.row < 0:
            raise ValueError("row must be non-negative")
        if self.col < 0:
            raise ValueError("col must be non-negative")
        if self.timestep < 0:
            raise ValueError("timestep must be non-negative")

    @classmethod
    def from_position(cls, position: Position, timestep: int) -> TimedState:
        return cls(row=position.row, col=position.col, timestep=timestep)
