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


@dataclass(frozen=True, slots=True)
class VertexConflict:
    agent1_id: int
    agent2_id: int
    row: int
    col: int
    timestep: int

    def __post_init__(self) -> None:
        if self.agent1_id == self.agent2_id:
            raise ValueError("agent1_id and agent2_id must differ")
        if self.row < 0:
            raise ValueError("row must be non-negative")
        if self.col < 0:
            raise ValueError("col must be non-negative")
        if self.timestep < 0:
            raise ValueError("timestep must be non-negative")


@dataclass(frozen=True, slots=True)
class EdgeConflict:
    """Swap conflict during one timestep transition.

    ``timestep`` is the arrival timestep of the transition, i.e. the conflict
    between states at ``timestep - 1`` and ``timestep``.
    """

    agent1_id: int
    agent2_id: int

    agent1_from_row: int
    agent1_from_col: int
    agent1_to_row: int
    agent1_to_col: int

    agent2_from_row: int
    agent2_from_col: int
    agent2_to_row: int
    agent2_to_col: int

    timestep: int

    def __post_init__(self) -> None:
        if self.agent1_id == self.agent2_id:
            raise ValueError("agent1_id and agent2_id must differ")
        if self.agent1_from_row < 0:
            raise ValueError("agent1_from_row must be non-negative")
        if self.agent1_from_col < 0:
            raise ValueError("agent1_from_col must be non-negative")
        if self.agent1_to_row < 0:
            raise ValueError("agent1_to_row must be non-negative")
        if self.agent1_to_col < 0:
            raise ValueError("agent1_to_col must be non-negative")
        if self.agent2_from_row < 0:
            raise ValueError("agent2_from_row must be non-negative")
        if self.agent2_from_col < 0:
            raise ValueError("agent2_from_col must be non-negative")
        if self.agent2_to_row < 0:
            raise ValueError("agent2_to_row must be non-negative")
        if self.agent2_to_col < 0:
            raise ValueError("agent2_to_col must be non-negative")
        if self.timestep <= 0:
            raise ValueError("timestep must be positive")


@dataclass(frozen=True, slots=True)
class VertexConstraint:
    agent_id: int
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


@dataclass(frozen=True, slots=True)
class EdgeConstraint:
    """Directed edge constraint forbidding a specific transition.

    ``timestep`` is the arrival timestep: forbids ``(from_row, from_col, t-1)``
    to ``(to_row, to_col, t)``.
    """

    agent_id: int

    from_row: int
    from_col: int
    to_row: int
    to_col: int

    timestep: int

    def __post_init__(self) -> None:
        if self.from_row < 0:
            raise ValueError("from_row must be non-negative")
        if self.from_col < 0:
            raise ValueError("from_col must be non-negative")
        if self.to_row < 0:
            raise ValueError("to_row must be non-negative")
        if self.to_col < 0:
            raise ValueError("to_col must be non-negative")
        if self.timestep <= 0:
            raise ValueError("timestep must be positive")


Conflict = VertexConflict | EdgeConflict
Constraint = VertexConstraint | EdgeConstraint
