from pydantic import BaseModel, Field

from src.core.models import Position


class Cluster(BaseModel):
    id: int
    row_start: int = Field(ge=0)
    row_end: int = Field(ge=0)
    col_start: int = Field(ge=0)
    col_end: int = Field(ge=0)

    def contains(self, position: Position) -> bool:
        return (
            self.row_start <= position.row < self.row_end
            and self.col_start <= position.col < self.col_end
        )


class Entrance(BaseModel):
    cluster_a_id: int
    cluster_b_id: int
    position_a: Position
    position_b: Position


class AbstractNode(BaseModel):
    id: int
    cluster_id: int
    position: Position


class AbstractEdge(BaseModel):
    from_node_id: int
    to_node_id: int
    cost: float


class AbstractGraph(BaseModel):
    nodes: list[AbstractNode]
    edges: list[AbstractEdge]


class HPAPathResult(BaseModel):
    abstract_path: list[AbstractNode]
    refined_path: list[Position]