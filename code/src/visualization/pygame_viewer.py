from typing import TypeAlias

import pygame

from code.src.core.models import GridMap, Position
from code.src.core.trace import RawAlgorithmStep
from code.src.visualization.overlays.base_overlay import BaseOverlay
from code.src.visualization.pygame_models import (
    AlgorithmVisualization,
    CellState,
    NamedPath,
    PygameViewerConfig,
    ViewerColors,
    AlgorithmAnimationVisualization,
    ViewportRenderContext,
)

GridNode: TypeAlias = tuple[int, int]

GridNodeCollection: TypeAlias = (
        frozenset[GridNode]
        | set[GridNode]
        | tuple[GridNode, ...]
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

            if self.screen is None:
                return

            self._draw_base_map(self.screen)
            self._draw_path(self.screen, path)
            self._draw_special_marker(self.screen, start, CellState.START)
            self._draw_special_marker(self.screen, goal, CellState.GOAL)
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

            if self.screen is None:
                return

            now = pygame.time.get_ticks()

            if step_index < len(steps) - 1 and now - last_step_time >= step_delay_ms:
                step_index = min(step_index + steps_per_frame, len(steps) - 1)
                last_step_time = now

            current_step = steps[step_index]

            self._draw_base_map(self.screen)

            self._draw_nodes(self.screen, current_step.closed_nodes, CellState.CLOSED)
            self._draw_nodes(self.screen, current_step.open_nodes, CellState.OPEN)

            if current_step.current is not None:
                row, col = current_step.current
                self._draw_cell_by_coordinates(
                    self.screen,
                    row,
                    col,
                    CellState.CURRENT,
                )

            self._draw_overlays()

            if current_step.path:
                self._draw_path(self.screen, current_step.path)

            self._draw_special_marker(self.screen, start, CellState.START)
            self._draw_special_marker(self.screen, goal, CellState.GOAL)

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

            if self.screen is None:
                return

            self._draw_base_map(self.screen)
            self._draw_overlays()

            for named_path in named_paths:
                self._draw_path(
                    self.screen,
                    path=named_path.path,
                    color=named_path.color.as_tuple(),
                )

            self._draw_special_marker(self.screen, start, CellState.START)
            self._draw_special_marker(self.screen, goal, CellState.GOAL)

            self._finalize_frame()

        pygame.quit()

    def run_grid_comparison_view(
            self,
            start: Position,
            goal: Position,
            visualizations: list[AlgorithmVisualization],
    ) -> None:
        rows, cols = self._calculate_comparison_grid(len(visualizations))

        viewport_width = self.config.max_comparison_window_width // cols
        viewport_height = self.config.max_comparison_window_height // rows

        total_width = viewport_width * cols
        total_height = viewport_height * rows

        self._initialize_pygame(
            window_width=total_width,
            window_height=total_height,
        )

        running = True

        while running:
            running = self._handle_events()

            if self.screen is None:
                return

            self.screen.fill(self.colors.background.as_tuple())

            for index, visualization in enumerate(visualizations):
                row_index = index // cols
                col_index = index % cols

                x_offset = col_index * viewport_width
                y_offset = row_index * viewport_height

                context = ViewportRenderContext(
                    x=x_offset,
                    y=y_offset,
                    width=viewport_width,
                    height=viewport_height,
                    cell_size=1,
                )

                viewport_surface = self._render_visualization_to_surface(
                    start=start,
                    goal=goal,
                    visualization=visualization,
                )

                scaled_surface = pygame.transform.scale(
                    viewport_surface,
                    (viewport_width, viewport_height),
                )

                self.screen.blit(
                    scaled_surface,
                    (x_offset, y_offset),
                )

                self._draw_performance_box(
                    visualization=visualization,
                    x=x_offset + 10,
                    y=y_offset + 10,
                )

            self._finalize_frame()

        pygame.quit()

    def run_grid_animation_comparison_view(
            self,
            start: Position,
            goal: Position,
            visualizations: list[AlgorithmAnimationVisualization],
            total_animation_time_ms: int = 8000,
    ) -> None:
        rows, cols = self._calculate_comparison_grid(len(visualizations))

        viewport_width = self.config.max_comparison_window_width // cols
        viewport_height = self.config.max_comparison_window_height // rows

        total_width = viewport_width * cols
        total_height = viewport_height * rows

        self._initialize_pygame(
            window_width=total_width,
            window_height=total_height,
        )

        animation_start_time = pygame.time.get_ticks()
        running = True

        while running:
            running = self._handle_events()

            if self.screen is None:
                return

            elapsed_ms = pygame.time.get_ticks() - animation_start_time
            progress = min(elapsed_ms / total_animation_time_ms, 1.0)

            self.screen.fill(self.colors.background.as_tuple())

            for index, visualization in enumerate(visualizations):
                row_index = index // cols
                col_index = index % cols

                x_offset = col_index * viewport_width
                y_offset = row_index * viewport_height

                context = ViewportRenderContext(
                    x=x_offset,
                    y=y_offset,
                    width=viewport_width,
                    height=viewport_height,
                    cell_size=1,
                )

                current_step = self._get_step_by_progress(
                    steps=visualization.steps,
                    progress=progress,
                )

                viewport_surface = self._render_animation_step_to_surface(
                    start=start,
                    goal=goal,
                    visualization=visualization,
                    current_step=current_step,
                )

                scaled_surface = pygame.transform.scale(
                    viewport_surface,
                    (viewport_width, viewport_height),
                )

                self.screen.blit(
                    scaled_surface,
                    (x_offset, y_offset),
                )

                self._draw_animation_performance_box(
                    visualization=visualization,
                    step=current_step,
                    context=context,
                    progress=progress,
                )

            self._finalize_frame()

        pygame.quit()

    def _render_animation_step_to_surface(
            self,
            start: Position,
            goal: Position,
            visualization: AlgorithmAnimationVisualization,
            current_step: RawAlgorithmStep | None,
    ) -> pygame.Surface:
        surface = pygame.Surface(
            (
                self.grid_map.width,
                self.grid_map.height,
            )
        )

        original_cell_size = self.cell_size
        self.cell_size = 1

        self._draw_base_map(surface)

        if current_step is not None:
            self._draw_nodes(surface, current_step.closed_nodes, CellState.CLOSED)
            self._draw_nodes(surface, current_step.open_nodes, CellState.OPEN)

            if current_step.current is not None:
                row, col = current_step.current
                self._draw_cell_by_coordinates(
                    surface,
                    row,
                    col,
                    CellState.CURRENT,
                )

            if current_step.path:
                self._draw_path(
                    surface,
                    current_step.path,
                    color=visualization.color.as_tuple(),
                )

        elif visualization.result.path:
            self._draw_path(
                surface,
                visualization.result.path,
                color=visualization.color.as_tuple(),
            )

        self._draw_special_marker(surface, start, CellState.START)
        self._draw_special_marker(surface, goal, CellState.GOAL)

        self.cell_size = original_cell_size

        return surface

    def _draw_animation_performance_box(
            self,
            visualization: AlgorithmAnimationVisualization,
            step: RawAlgorithmStep | None,
            context: ViewportRenderContext,
            progress: float,
    ) -> None:
        if self.screen is None:
            return

        font = pygame.font.SysFont("Arial", 16)

        step_path_length = len(step.path) if step is not None else 0
        closed_count = len(step.closed_nodes) if step is not None else 0

        lines = [
            f"Algorithm: {visualization.result.algorithm_name}",
            f"Progress: {progress * 100:.1f}%",
            f"Path length: {visualization.result.path_length}",
            f"Current path: {step_path_length}",
            f"Closed nodes: {closed_count}",
            f"Time: {visualization.result.execution_time_ms:.3f} ms",
        ]

        padding = 8
        line_height = 20
        box_width = 190
        box_height = padding * 2 + len(lines) * line_height

        x = context.x + 10
        y = context.y + 10

        background = pygame.Surface((box_width, box_height))
        background.set_alpha(180)
        background.fill((0, 0, 0))

        self.screen.blit(background, (x, y))

        for index, line in enumerate(lines):
            text_surface = font.render(line, True, (255, 255, 255))
            self.screen.blit(
                text_surface,
                (
                    x + padding,
                    y + padding + index * line_height,
                ),
            )

    @staticmethod
    def _get_step_by_progress(
            steps: list[RawAlgorithmStep],
            progress: float,
    ) -> RawAlgorithmStep | None:
        if not steps:
            return None

        if len(steps) == 1:
            return steps[0]

        step_index = int(progress * (len(steps) - 1))
        return steps[step_index]

    def add_overlay(
            self,
            overlay: BaseOverlay,
    ) -> None:
        self.overlays.append(overlay)

    def _render_visualization_to_surface(
            self,
            start: Position,
            goal: Position,
            visualization: AlgorithmVisualization,
    ) -> pygame.Surface:
        surface = pygame.Surface(
            (
                self.grid_map.width,
                self.grid_map.height,
            )
        )

        surface.fill(self.colors.background.as_tuple())

        original_cell_size = self.cell_size
        self.cell_size = 1

        self._draw_base_map(surface)
        self._draw_path(
            surface,
            path=visualization.path,
            color=visualization.color.as_tuple(),
        )
        self._draw_special_marker(surface, start, CellState.START)
        self._draw_special_marker(surface, goal, CellState.GOAL)

        self.cell_size = original_cell_size

        return surface

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

    @staticmethod
    def _calculate_comparison_grid(
            item_count: int,
    ) -> tuple[int, int]:
        if item_count <= 0:
            raise ValueError("At least one visualization is required.")

        if item_count == 1:
            return 1, 1

        if item_count == 2:
            return 1, 2

        if item_count == 3:
            return 1, 3

        if item_count == 4:
            return 2, 2

        if item_count <= 6:
            return 2, 3

        if item_count <= 9:
            return 3, 3

        raise ValueError("Grid comparison supports up to 9 visualizations.")

    def _draw_performance_box(
            self,
            visualization: AlgorithmVisualization,
            x: int,
            y: int,
    ) -> None:
        if self.screen is None or visualization.result is None:
            return

        font = pygame.font.SysFont("Arial", 16)

        result = visualization.result

        lines = [
            f"Algorithm: {result.algorithm_name}",
            f"Found: {result.found}",
            f"Path length: {result.path_length}",
            f"Visited nodes: {result.visited_nodes}",
            f"Time: {result.execution_time_ms:.3f} ms",
        ]

        padding = 8
        line_height = 20
        box_width = 160
        box_height = padding * 2 + len(lines) * line_height

        background = pygame.Surface((box_width, box_height))
        background.set_alpha(180)
        background.fill((0, 0, 0))

        self.screen.blit(background, (x, y))

        for index, line in enumerate(lines):
            text_surface = font.render(line, True, (255, 255, 255))
            self.screen.blit(
                text_surface,
                (
                    x + padding,
                    y + padding + index * line_height,
                ),
            )

    def _grid_to_screen(
            self,
            row: int,
            col: int,
    ) -> tuple[int, int]:
        x = col * self.cell_size + self.cell_size // 2
        y = row * self.cell_size + self.cell_size // 2

        return x, y

    def _draw_overlays(
            self,
            context: ViewportRenderContext | None = None,
    ) -> None:
        if self.screen is None:
            return

        for overlay in self.overlays:
            overlay.draw(self.screen, context)

    def _draw_base_map(
            self,
            surface: pygame.Surface,
    ) -> None:
        surface.fill(self.colors.background.as_tuple())

        for row in range(self.grid_map.height):
            for col in range(self.grid_map.width):
                position = Position(row=row, col=col)

                state = (
                    CellState.EMPTY
                    if self.grid_map.is_walkable(position)
                    else CellState.WALL
                )

                self._draw_cell(
                    surface=surface,
                    position=position,
                    state=state,
                )

    def _draw_positions(
            self,
            surface: pygame.Surface,
            positions: list[Position],
            state: CellState,
    ) -> None:
        for position in positions:
            self._draw_position(
                surface=surface,
                position=position,
                state=state,
            )

    def _draw_position(
            self,
            surface: pygame.Surface,
            position: Position,
            state: CellState,
    ) -> None:
        self._draw_cell(
            surface=surface,
            position=position,
            state=state,
        )

    def _draw_cell(
            self,
            surface: pygame.Surface,
            position: Position,
            state: CellState,
    ) -> None:
        self._draw_cell_by_coordinates(
            surface=surface,
            row=position.row,
            col=position.col,
            state=state,
        )

    def _draw_nodes(
            self,
            surface: pygame.Surface,
            nodes: GridNodeCollection,
            state: CellState,
    ) -> None:
        for row, col in nodes:
            self._draw_cell_by_coordinates(
                surface=surface,
                row=row,
                col=col,
                state=state,
            )

    def _draw_cell_by_coordinates(
            self,
            surface: pygame.Surface,
            row: int,
            col: int,
            state: CellState,
    ) -> None:
        color = self._get_color(state)

        rect = pygame.Rect(
            col * self.cell_size,
            row * self.cell_size,
            self.cell_size,
            self.cell_size,
        )

        pygame.draw.rect(surface, color, rect)

        if self.config.draw_grid:
            pygame.draw.rect(
                surface,
                self.colors.grid_line.as_tuple(),
                rect,
                width=1,
            )

    def _draw_special_marker(
            self,
            surface: pygame.Surface,
            position: Position,
            state: CellState,
    ) -> None:
        color = self._get_color(state)
        center_x, center_y = self._grid_to_screen(
            row=position.row,
            col=position.col,
        )

        radius = max(self.cell_size * 2, 4)

        pygame.draw.circle(
            surface,
            color,
            (center_x, center_y),
            radius,
        )

    def _draw_path(
            self,
            surface: pygame.Surface,
            path: tuple[tuple[int, int], ...] | list[Position],
            color: tuple[int, int, int] | None = None,
    ) -> None:
        if len(path) < 2:
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
                )
            )

        pygame.draw.lines(
            surface,
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
