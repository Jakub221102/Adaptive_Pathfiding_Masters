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


@dataclass(frozen=True, slots=True)
class MAPFScenario:
    agents: tuple[MAPFAgent, ...]

    def __post_init__(self) -> None:
        if not self.agents:
            raise ValueError("scenario must contain at least one agent")

        agent_ids = {agent.agent_id for agent in self.agents}
        if len(agent_ids) != len(self.agents):
            raise ValueError("agent_id must be unique")


@dataclass(frozen=True, slots=True)
class AgentPath:
    agent_id: int
    states: tuple[TimedState, ...]

    def __post_init__(self) -> None:
        if not self.states:
            raise ValueError("states must not be empty")

        for index in range(len(self.states) - 1):
            current = self.states[index]
            next_state = self.states[index + 1]

            if next_state.timestep != current.timestep + 1:
                raise ValueError("timesteps must be strictly consecutive")


@dataclass(frozen=True, slots=True)
class MAPFResult:
    success: bool
    paths: tuple[AgentPath, ...]
