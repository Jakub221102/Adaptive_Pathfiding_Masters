import pygame

from src.core.models import GridMap, Position
from src.core.trace import RawAlgorithmStep
from src.visualization.overlays.base_overlay import BaseOverlay
from src.visualization.pygame_models import (
    AlgorithmVisualization,
    CellState,
    NamedPath,
    PygameViewerConfig,
    ViewerColors,
)


GridNodeCollection = (
    frozenset[tuple[int, int]]
    | set[tuple[int, int]]
    | tuple[tuple[int, int], ...]
)


class PygameGridViewer:
    def __init__(
        self,
        grid_map: GridMap,
        config: PygameViewerConfig | None = None,
        colors: ViewerColors | None = None,
    ) -> None:
        self.grid_map = grid_map
        self.config = config or PygameViewerConfig()
        self.colors = colors or ViewerColors()

        self.overlays: list[BaseOverlay] = []

        self.cell_size = self._calculate_cell_size()
        self.window_width = self.grid_map.width * self.cell_size
        self.window_height = self.grid_map.height * self.cell_size

        self.screen: pygame.Surface | None = None
        self.clock: pygame.time.Clock | None = None

    def run_static_path_view(
        self,
        start: Position,
        goal: Position,
        path: list[Position],
    ) -> None:
        self._initialize_pygame()

        running = True

        while running:
            running = self._handle_events()

            self._draw_base_map()
            self._draw_path(path)
            self._draw_special_marker(start, CellState.START)
            self._draw_special_marker(goal, CellState.GOAL)
            self._draw_overlays()

            self._finalize_frame()

        pygame.quit()

    def run_algorithm_animation(
        self,
        start: Position,
        goal: Position,
        steps: list[RawAlgorithmStep],
        step_delay_ms: int = 1,
        steps_per_frame: int = 50,
    ) -> None:
        if not steps:
            self.run_static_path_view(start=start, goal=goal, path=[])
            return

        self._initialize_pygame()

        running = True
        step_index = 0
        last_step_time = pygame.time.get_ticks()

        while running:
            running = self._handle_events()

            now = pygame.time.get_ticks()

            if step_index < len(steps) - 1 and now - last_step_time >= step_delay_ms:
                step_index = min(step_index + steps_per_frame, len(steps) - 1)
                last_step_time = now

            current_step = steps[step_index]

            self._draw_base_map()
            self._draw_nodes(current_step.closed_nodes, CellState.CLOSED)
            self._draw_nodes(current_step.open_nodes, CellState.OPEN)

            if current_step.current is not None:
                row, col = current_step.current
                self._draw_cell_by_coordinates(row, col, CellState.CURRENT)

            if current_step.path:
                self._draw_path(current_step.path)

            self._draw_overlays()
            self._draw_special_marker(start, CellState.START)
            self._draw_special_marker(goal, CellState.GOAL)

            self._finalize_frame()

        pygame.quit()

    def run_multi_path_view(
        self,
        start: Position,
        goal: Position,
        named_paths: list[NamedPath],
    ) -> None:
        self._initialize_pygame()

        running = True

        while running:
            running = self._handle_events()

            self._draw_base_map()
            self._draw_overlays()

            for named_path in named_paths:
                self._draw_path(
                    path=named_path.path,
                    color=named_path.color.as_tuple(),
                )

            self._draw_special_marker(start, CellState.START)
            self._draw_special_marker(goal, CellState.GOAL)

            self._finalize_frame()

        pygame.quit()

    def run_side_by_side_view(
        self,
        start: Position,
        goal: Position,
        left_visualization: AlgorithmVisualization,
        right_visualization: AlgorithmVisualization,
    ) -> None:
        comparison_width = self.window_width * 2
        self._initialize_pygame(
            window_width=comparison_width,
            window_height=self.window_height,
        )

        running = True
        right_offset = self.window_width

        while running:
            running = self._handle_events()

            if self.screen is None:
                return

            self.screen.fill(self.colors.background.as_tuple())

            self._draw_base_map_with_offset(x_offset=0)
            self._draw_base_map_with_offset(x_offset=right_offset)

            self._draw_path(
                path=left_visualization.path,
                color=left_visualization.color.as_tuple(),
                x_offset=0,
            )

            self._draw_path(
                path=right_visualization.path,
                color=right_visualization.color.as_tuple(),
                x_offset=right_offset,
            )

            self._draw_special_marker(start, CellState.START, x_offset=0)
            self._draw_special_marker(goal, CellState.GOAL, x_offset=0)

            self._draw_special_marker(start, CellState.START, x_offset=right_offset)
            self._draw_special_marker(goal, CellState.GOAL, x_offset=right_offset)

            self._finalize_frame()

        pygame.quit()

    def add_overlay(
        self,
        overlay: BaseOverlay,
    ) -> None:
        self.overlays.append(overlay)

    def _initialize_pygame(
        self,
        window_width: int | None = None,
        window_height: int | None = None,
    ) -> None:
        pygame.init()

        self.screen = pygame.display.set_mode(
            (
                window_width or self.window_width,
                window_height or self.window_height,
            )
        )
        self.clock = pygame.time.Clock()

        pygame.display.set_caption(self.config.window_title)

    @staticmethod
    def _handle_events() -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        return True

    def _finalize_frame(self) -> None:
        pygame.display.flip()

        if self.clock is not None:
            self.clock.tick(self.config.fps)

    def _calculate_cell_size(self) -> int:
        if self.config.cell_size is not None:
            return self.config.cell_size

        width_scale = self.config.max_window_width // self.grid_map.width
        height_scale = self.config.max_window_height // self.grid_map.height

        return max(min(width_scale, height_scale), 1)

    def _grid_to_screen(
        self,
        row: int,
        col: int,
        x_offset: int = 0,
    ) -> tuple[int, int]:
        x = col * self.cell_size + self.cell_size // 2 + x_offset
        y = row * self.cell_size + self.cell_size // 2

        return x, y

    def _draw_overlays(self) -> None:
        if self.screen is None:
            return

        for overlay in self.overlays:
            overlay.draw(self.screen)

    def _draw_base_map(self) -> None:
        if self.screen is None:
            return

        self.screen.fill(self.colors.background.as_tuple())
        self._draw_base_map_with_offset(x_offset=0)

    def _draw_base_map_with_offset(
        self,
        x_offset: int,
    ) -> None:
        for row in range(self.grid_map.height):
            for col in range(self.grid_map.width):
                position = Position(row=row, col=col)

                state = (
                    CellState.EMPTY
                    if self.grid_map.is_walkable(position)
                    else CellState.WALL
                )

                self._draw_cell(
                    position=position,
                    state=state,
                    x_offset=x_offset,
                )

    def _draw_positions(
        self,
        positions: list[Position],
        state: CellState,
        x_offset: int = 0,
    ) -> None:
        for position in positions:
            self._draw_position(
                position=position,
                state=state,
                x_offset=x_offset,
            )

    def _draw_position(
        self,
        position: Position,
        state: CellState,
        x_offset: int = 0,
    ) -> None:
        self._draw_cell(
            position=position,
            state=state,
            x_offset=x_offset,
        )

    def _draw_cell(
        self,
        position: Position,
        state: CellState,
        x_offset: int = 0,
    ) -> None:
        self._draw_cell_by_coordinates(
            row=position.row,
            col=position.col,
            state=state,
            x_offset=x_offset,
        )

    def _draw_nodes(
        self,
        nodes: GridNodeCollection,
        state: CellState,
        x_offset: int = 0,
    ) -> None:
        for row, col in nodes:
            self._draw_cell_by_coordinates(
                row=row,
                col=col,
                state=state,
                x_offset=x_offset,
            )

    def _draw_cell_by_coordinates(
        self,
        row: int,
        col: int,
        state: CellState,
        x_offset: int = 0,
    ) -> None:
        if self.screen is None:
            return

        color = self._get_color(state)

        rect = pygame.Rect(
            col * self.cell_size + x_offset,
            row * self.cell_size,
            self.cell_size,
            self.cell_size,
        )

        pygame.draw.rect(self.screen, color, rect)

        if self.config.draw_grid:
            pygame.draw.rect(
                self.screen,
                self.colors.grid_line.as_tuple(),
                rect,
                width=1,
            )

    def _draw_special_marker(
        self,
        position: Position,
        state: CellState,
        x_offset: int = 0,
    ) -> None:
        if self.screen is None:
            return

        color = self._get_color(state)
        center_x, center_y = self._grid_to_screen(
            row=position.row,
            col=position.col,
            x_offset=x_offset,
        )

        radius = max(self.cell_size * 2, 4)

        pygame.draw.circle(
            self.screen,
            color,
            (center_x, center_y),
            radius,
        )

    def _draw_path(
        self,
        path: tuple[tuple[int, int], ...] | list[Position],
        color: tuple[int, int, int] | None = None,
        x_offset: int = 0,
    ) -> None:
        if self.screen is None or len(path) < 2:
            return

        path_color = color or self.colors.path.as_tuple()

        points: list[tuple[int, int]] = []

        for node in path:
            if isinstance(node, Position):
                row, col = node.row, node.col
            else:
                row, col = node

            points.append(
                self._grid_to_screen(
                    row=row,
                    col=col,
                    x_offset=x_offset,
                )
            )

        pygame.draw.lines(
            self.screen,
            path_color,
            False,
            points,
            width=max(self.cell_size * 2, 2),
        )

    def _get_color(self, state: CellState) -> tuple[int, int, int]:
        match state:
            case CellState.EMPTY:
                return self.colors.empty.as_tuple()
            case CellState.WALL:
                return self.colors.wall.as_tuple()
            case CellState.START:
                return self.colors.start.as_tuple()
            case CellState.GOAL:
                return self.colors.goal.as_tuple()
            case CellState.OPEN:
                return self.colors.open.as_tuple()
            case CellState.CLOSED:
                return self.colors.closed.as_tuple()
            case CellState.PATH:
                return self.colors.path.as_tuple()
            case CellState.CURRENT:
                return self.colors.current.as_tuple()
            case _:
                return self.colors.empty.as_tuple()