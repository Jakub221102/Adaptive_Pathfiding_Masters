from enum import Enum


from pydantic import BaseModel, Field

from src.core.models import PathfindingResult, Position


class CellState(str, Enum):
    EMPTY = "empty"
    WALL = "wall"
    START = "start"
    GOAL = "goal"
    OPEN = "open"
    CLOSED = "closed"
    PATH = "path"
    CURRENT = "current"


class RGBColor(BaseModel):
    r: int = Field(ge=0, le=255)
    g: int = Field(ge=0, le=255)
    b: int = Field(ge=0, le=255)

    def as_tuple(self) -> tuple[int, int, int]:
        return self.r, self.g, self.b


class ViewerColors(BaseModel):
    empty: RGBColor = RGBColor(r=245, g=245, b=245)
    wall: RGBColor = RGBColor(r=35, g=35, b=35)
    start: RGBColor = RGBColor(r=0, g=200, b=0)
    goal: RGBColor = RGBColor(r=220, g=0, b=0)
    open: RGBColor = RGBColor(r=80, g=160, b=255)
    closed: RGBColor = RGBColor(r=120, g=120, b=120)
    path: RGBColor = RGBColor(r=255, g=210, b=0)
    current: RGBColor = RGBColor(r=180, g=0, b=255)
    grid_line: RGBColor = RGBColor(r=210, g=210, b=210)
    background: RGBColor = RGBColor(r=20, g=20, b=20)


class PygameViewerConfig(BaseModel):
    cell_size: int | None = None
    max_window_width: int = 1400
    max_window_height: int = 900
    max_comparison_window_width: int = 1000
    max_comparison_window_height: int = 600
    fps: int = Field(default=60, gt=0)
    draw_grid: bool = False
    window_title: str = "Pathfinding Viewer"


class AlgorithmStep(BaseModel):
    current: Position | None = None
    open_nodes: list[Position] = []
    closed_nodes: list[Position] = []
    path: list[Position] = []

class ViewerMode(str, Enum):
    STATIC = "static"
    ANIMATED = "animated"

class NamedPath(BaseModel):
    name: str
    path: list[Position]
    color: RGBColor

class Viewport(BaseModel):
    x_offset: int
    width: int

class AlgorithmVisualization(BaseModel):
    name: str
    path: list[Position]
    color: RGBColor
    result: PathfindingResult | None = None