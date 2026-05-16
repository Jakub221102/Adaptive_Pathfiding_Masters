import pygame

from src.core.models import GridMap, Position
from src.core.trace import RawAlgorithmStep
from src.visualization.pygame_models import (
    CellState,
    PygameViewerConfig,
    ViewerColors,
    NamedPath
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
        pygame.init()

        self.screen = pygame.display.set_mode(
            (self.window_width, self.window_height)
        )
        self.clock = pygame.time.Clock()

        pygame.display.set_caption(self.config.window_title)

        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self._draw_base_map()
            self._draw_positions(path, CellState.PATH)
            self._draw_position(start, CellState.START)
            self._draw_position(goal, CellState.GOAL)

            if self.config.show_cluster_overlay:
                self._draw_cluster_overlay(
                    cluster_size=self.config.cluster_size,
                )

            pygame.display.flip()
            self.clock.tick(self.config.fps)

        pygame.quit()

    def run_algorithm_animation(
            self,
            start: Position,
            goal: Position,
            steps: list[RawAlgorithmStep],
            step_delay_ms: int = 1,
            steps_per_frame: int = 50,
    ) -> None:
        pygame.init()

        self.screen = pygame.display.set_mode(
            (self.window_width, self.window_height)
        )
        self.clock = pygame.time.Clock()

        pygame.display.set_caption(self.config.window_title)

        running = True
        step_index = 0
        last_step_time = pygame.time.get_ticks()

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

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

            self._draw_special_marker(start, CellState.START)
            self._draw_special_marker(goal, CellState.GOAL)

            pygame.display.flip()
            self.clock.tick(self.config.fps)

        pygame.quit()

    def run_multi_path_view(
            self,
            start: Position,
            goal: Position,
            named_paths: list[NamedPath],
    ) -> None:
        pygame.init()

        self.screen = pygame.display.set_mode(
            (self.window_width, self.window_height)
        )
        self.clock = pygame.time.Clock()

        pygame.display.set_caption(self.config.window_title)

        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self._draw_base_map()

            if self.config.show_cluster_overlay:
                self._draw_cluster_overlay(
                    cluster_size=self.config.cluster_size,
                )

            for named_path in named_paths:
                self._draw_path_with_color(
                    path=named_path.path,
                    color=named_path.color.as_tuple(),
                )

            self._draw_special_marker(start, CellState.START)
            self._draw_special_marker(goal, CellState.GOAL)

            pygame.display.flip()
            self.clock.tick(self.config.fps)

        pygame.quit()

    def _draw_base_map(self) -> None:
        if self.screen is None:
            return

        self.screen.fill(self.colors.background.as_tuple())

        for row in range(self.grid_map.height):
            for col in range(self.grid_map.width):
                position = Position(row=row, col=col)

                if self.grid_map.is_walkable(position):
                    state = CellState.EMPTY
                else:
                    state = CellState.WALL

                self._draw_cell(position, state)

    def _calculate_cell_size(self) -> int:
        if self.config.cell_size is not None:
            return self.config.cell_size

        width_scale = self.config.max_window_width // self.grid_map.width
        height_scale = self.config.max_window_height // self.grid_map.height

        cell_size = min(width_scale, height_scale)

        return max(cell_size, 1)

    def _draw_positions(
        self,
        positions: list[Position],
        state: CellState,
    ) -> None:
        for position in positions:
            self._draw_position(position, state)

    def _draw_position(
        self,
        position: Position,
        state: CellState,
    ) -> None:
        self._draw_cell(position, state)

    def _draw_cell(
        self,
        position: Position,
        state: CellState,
    ) -> None:
        if self.screen is None:
            return

        color = self._get_color(state)

        rect = pygame.Rect(
            position.col * self.cell_size,
            position.row * self.cell_size,
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

    def _draw_nodes(
            self,
            nodes: frozenset[tuple[int, int]] | set[tuple[int, int]] | tuple[tuple[int, int], ...],
            state: CellState,
    ) -> None:
        for row, col in nodes:
            self._draw_cell_by_coordinates(row, col, state)

    def _draw_cell_by_coordinates(
            self,
            row: int,
            col: int,
            state: CellState,
    ) -> None:
        if self.screen is None:
            return

        color = self._get_color(state)

        rect = pygame.Rect(
            col * self.cell_size,
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
    ) -> None:
        if self.screen is None:
            return

        color = self._get_color(state)

        center_x = position.col * self.cell_size + self.cell_size // 2
        center_y = position.row * self.cell_size + self.cell_size // 2

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
    ) -> None:
        if self.screen is None or len(path) < 2:
            return

        color = self.colors.path.as_tuple()

        points = []

        for node in path:
            if isinstance(node, Position):
                row, col = node.row, node.col
            else:
                row, col = node

            x = col * self.cell_size + self.cell_size // 2
            y = row * self.cell_size + self.cell_size // 2

            points.append((x, y))

        pygame.draw.lines(
            self.screen,
            color,
            False,
            points,
            width=max(self.cell_size * 2, 2),
        )

    def _draw_path_with_color(
            self,
            path: list[Position],
            color: tuple[int, int, int],
    ) -> None:
        if self.screen is None or len(path) < 2:
            return

        points = []

        for position in path:
            x = position.col * self.cell_size + self.cell_size // 2
            y = position.row * self.cell_size + self.cell_size // 2
            points.append((x, y))

        pygame.draw.lines(
            self.screen,
            color,
            False,
            points,
            width=max(self.cell_size * 2, 2),
        )

    def _draw_cluster_overlay(
            self,
            cluster_size: int,
    ) -> None:
        if self.screen is None:
            return

        color = (255, 0, 255)

        for row in range(0, self.grid_map.height, cluster_size):
            y = row * self.cell_size
            pygame.draw.line(
                self.screen,
                color,
                (0, y),
                (self.window_width, y),
                width=1,
            )

        for col in range(0, self.grid_map.width, cluster_size):
            x = col * self.cell_size
            pygame.draw.line(
                self.screen,
                color,
                (x, 0),
                (x, self.window_height),
                width=1,
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