from pydantic import BaseModel, Field


class Position(BaseModel):
    row: int = Field(ge=0)
    col: int = Field(ge=0)


class Scenario(BaseModel):
    map_name: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    start: Position
    goal: Position
    optimal_length: float | None = None


class GridMap(BaseModel):
    name: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    cells: list[list[int]]

    def is_walkable(self, position: Position) -> bool:
        return self.cells[position.row][position.col] == 0

    def in_bounds(self, position: Position) -> bool:
        return 0 <= position.row < self.height and 0 <= position.col < self.width


class PathfindingResult(BaseModel):
    algorithm_name: str
    found: bool

    path: list[Position]

    path_length: int
    path_cost: float

    visited_nodes: int
    execution_time_ms: float
    scanned_nodes: int | None = None


class FailedScenarioResult(BaseModel):
    scenario_index: int
    start: Position
    goal: Position
    optimal_length: float | None

    astar_found: bool
    hpa_found: bool

    astar_path_length: int
    hpa_path_length: int

    hpa_abstract_nodes_visited: int
    hpa_abstract_path_edge_count: int
