from __future__ import annotations

from collections.abc import Sequence

import pygame
from pydantic import BaseModel, Field

from pathfinding.src.algorithms.mapf.conflicts import detect_conflicts
from pathfinding.src.algorithms.mapf.metrics import makespan, sum_of_costs
from pathfinding.src.algorithms.mapf.models import AgentPath, MAPFScenario
from pathfinding.src.core.models import GridMap, Position
from pathfinding.src.visualization.mapf_viewer_helpers import (
    animation_max_timestep,
    calculate_viewport_cell_size,
    compute_mapf_viewport,
    position_at_timestep,
)
from pathfinding.src.visualization.pygame_models import CellState, ViewerColors

MAPF_ALGORITHM_LABEL = "Fixed-Priority Prioritized Planning"

_AGENT_COLORS: tuple[tuple[int, int, int], ...] = (
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 190),
)


class MapfViewerConfig(BaseModel):
    max_window_width: int = 1400
    max_window_height: int = 900
    viewport_padding: int = Field(default=2, ge=0)
    timesteps_per_second: float = Field(default=8.0, gt=0)
    render_fps: int = Field(default=60, gt=0)
    draw_grid: bool = False
    window_title: str = "MAPF Solution Viewer"
    show_starts: bool = False
    minimum_marker_radius: int = Field(default=4, ge=1)


def visualize_mapf_solution(
    grid_map: GridMap,
    scenario: MAPFScenario,
    paths: Sequence[AgentPath],
    config: MapfViewerConfig | None = None,
) -> None:
    viewer = MapfSolutionViewer(
        grid_map=grid_map,
        scenario=scenario,
        paths=tuple(paths),
        config=config,
    )
    viewer.run()


class MapfSolutionViewer:
    def __init__(
        self,
        grid_map: GridMap,
        scenario: MAPFScenario,
        paths: tuple[AgentPath, ...],
        config: MapfViewerConfig | None = None,
        colors: ViewerColors | None = None,
    ) -> None:
        if len(paths) != len(scenario.agents):
            raise ValueError("paths must contain one entry per scenario agent")

        self.grid_map = grid_map
        self.scenario = scenario
        self.paths = paths
        self.config = config or MapfViewerConfig()
        self.colors = colors or ViewerColors()

        self.viewport = compute_mapf_viewport(
            grid_map=grid_map,
            scenario=scenario,
            paths=paths,
            padding=self.config.viewport_padding,
        )
        self.cell_size = calculate_viewport_cell_size(
            viewport=self.viewport,
            max_window_width=self.config.max_window_width,
            max_window_height=self.config.max_window_height,
        )
        self.max_timestep = animation_max_timestep(paths)
        self.soc = sum_of_costs(paths)
        self.makespan_value = makespan(paths)
        self.conflict_count = len(detect_conflicts(paths))

        self.window_width = self.viewport.width_cells * self.cell_size
        self.window_height = self.viewport.height_cells * self.cell_size

        self.screen: pygame.Surface | None = None
        self.clock: pygame.time.Clock | None = None
        self.current_timestep = 0
        self.paused = False
        self.running = True
        self._last_advance_ms = 0

    def run(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        self.clock = pygame.time.Clock()
        pygame.display.set_caption(self.config.window_title)
        self._last_advance_ms = pygame.time.get_ticks()

        while self.running:
            self._handle_events()
            self._maybe_advance_timestep()
            self._draw_frame()
            if self.clock is not None:
                self.clock.tick(self.config.render_fps)

        pygame.quit()

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                continue

            if event.type != pygame.KEYDOWN:
                continue

            if event.key == pygame.K_ESCAPE:
                self.running = False
            elif event.key == pygame.K_SPACE:
                self.paused = not self.paused
            elif event.key == pygame.K_r:
                self.current_timestep = 0
                self.paused = False
                self._last_advance_ms = pygame.time.get_ticks()
            elif event.key == pygame.K_RIGHT:
                self.paused = True
                self.current_timestep = min(
                    self.current_timestep + 1,
                    self.max_timestep,
                )
                if self.current_timestep == self.max_timestep:
                    self.paused = True
            elif event.key == pygame.K_LEFT:
                self.paused = True
                self.current_timestep = max(self.current_timestep - 1, 0)

    def _maybe_advance_timestep(self) -> None:
        if self.paused or self.current_timestep >= self.max_timestep:
            return

        now = pygame.time.get_ticks()
        step_interval_ms = 1000.0 / self.config.timesteps_per_second

        if now - self._last_advance_ms < step_interval_ms:
            return

        self.current_timestep += 1
        self._last_advance_ms = now

        if self.current_timestep >= self.max_timestep:
            self.current_timestep = self.max_timestep
            self.paused = True

    def _draw_frame(self) -> None:
        if self.screen is None:
            return

        self.screen.fill(self.colors.background.as_tuple())
        self._draw_base_map()
        self._draw_goals()
        if self.config.show_starts:
            self._draw_starts()
        self._draw_agents()
        self._draw_overlay()
        pygame.display.flip()

    def _draw_base_map(self) -> None:
        if self.screen is None:
            return

        for row in range(self.viewport.min_row, self.viewport.max_row + 1):
            for col in range(self.viewport.min_col, self.viewport.max_col + 1):
                position = Position(row=row, col=col)
                state = (
                    CellState.EMPTY
                    if self.grid_map.is_walkable(position)
                    else CellState.WALL
                )
                self._draw_cell(row, col, self._cell_color(state))

    def _draw_goals(self) -> None:
        for agent in self.scenario.agents:
            color = self._agent_color(agent.agent_id)
            self._draw_hollow_marker(
                row=agent.goal.row,
                col=agent.goal.col,
                color=color,
            )

    def _draw_starts(self) -> None:
        for agent in self.scenario.agents:
            center_x, center_y = self._grid_to_screen(agent.start.row, agent.start.col)
            radius = max(self.cell_size // 4, 2)
            pygame.draw.circle(
                self.screen,
                (180, 180, 180),
                (center_x, center_y),
                radius,
                width=1,
            )

    def _draw_agents(self) -> None:
        path_by_agent = {path.agent_id: path for path in self.paths}

        for agent in self.scenario.agents:
            path = path_by_agent[agent.agent_id]
            row, col = position_at_timestep(path, self.current_timestep)
            color = self._agent_color(agent.agent_id)
            self._draw_filled_marker(row=row, col=col, color=color)
            self._draw_agent_label(
                row=row,
                col=col,
                label=str(agent.agent_id),
                color=color,
            )

    def _draw_overlay(self) -> None:
        if self.screen is None:
            return

        font = pygame.font.SysFont("Arial", 16)
        lines = [
            MAPF_ALGORITHM_LABEL,
            f"Agents: {len(self.scenario.agents)}",
            f"SoC: {self.soc}",
            f"Makespan: {self.makespan_value}",
            f"Conflicts: {self.conflict_count}",
            f"timestep: {self.current_timestep} / {self.max_timestep}",
            "SPACE pause/resume | R reset | LEFT/RIGHT step | ESC exit",
        ]

        if self.paused:
            lines.insert(-1, "Status: paused")

        padding = 8
        line_height = 20
        box_width = max(font.size(line)[0] for line in lines) + padding * 2
        box_height = padding * 2 + len(lines) * line_height

        background = pygame.Surface((box_width, box_height))
        background.set_alpha(200)
        background.fill((0, 0, 0))
        self.screen.blit(background, (10, 10))

        for index, line in enumerate(lines):
            text_surface = font.render(line, True, (255, 255, 255))
            self.screen.blit(
                text_surface,
                (10 + padding, 10 + padding + index * line_height),
            )

    def _grid_to_screen(self, row: int, col: int) -> tuple[int, int]:
        local_row = row - self.viewport.min_row
        local_col = col - self.viewport.min_col
        x = local_col * self.cell_size + self.cell_size // 2
        y = local_row * self.cell_size + self.cell_size // 2
        return x, y

    def _draw_cell(
        self,
        row: int,
        col: int,
        color: tuple[int, int, int],
    ) -> None:
        if self.screen is None:
            return

        local_row = row - self.viewport.min_row
        local_col = col - self.viewport.min_col
        rect = pygame.Rect(
            local_col * self.cell_size,
            local_row * self.cell_size,
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

    def _draw_filled_marker(
        self,
        row: int,
        col: int,
        color: tuple[int, int, int],
    ) -> None:
        if self.screen is None:
            return

        center_x, center_y = self._grid_to_screen(row, col)
        radius = max(self.cell_size // 2, self.config.minimum_marker_radius)
        pygame.draw.circle(self.screen, color, (center_x, center_y), radius)

    def _draw_hollow_marker(
        self,
        row: int,
        col: int,
        color: tuple[int, int, int],
    ) -> None:
        if self.screen is None:
            return

        center_x, center_y = self._grid_to_screen(row, col)
        radius = max(self.cell_size // 2, self.config.minimum_marker_radius)
        pygame.draw.circle(
            self.screen,
            color,
            (center_x, center_y),
            radius,
            width=max(2, radius // 3),
        )

    def _draw_agent_label(
        self,
        row: int,
        col: int,
        label: str,
        color: tuple[int, int, int],
    ) -> None:
        if self.screen is None:
            return

        font = pygame.font.SysFont("Arial", max(10, self.cell_size // 2), bold=True)
        text_surface = font.render(label, True, (255, 255, 255))
        text_rect = text_surface.get_rect()
        center_x, center_y = self._grid_to_screen(row, col)
        text_rect.center = (center_x, center_y)
        outline_color = tuple(max(0, channel - 80) for channel in color)
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            shadow = font.render(label, True, outline_color)
            shadow_rect = shadow.get_rect()
            shadow_rect.center = (center_x + dx, center_y + dy)
            self.screen.blit(shadow, shadow_rect)
        self.screen.blit(text_surface, text_rect)

    def _agent_color(self, agent_id: int) -> tuple[int, int, int]:
        return _AGENT_COLORS[agent_id % len(_AGENT_COLORS)]

    def _cell_color(self, state: CellState) -> tuple[int, int, int]:
        if state == CellState.EMPTY:
            return self.colors.empty.as_tuple()
        if state == CellState.WALL:
            return self.colors.wall.as_tuple()
        return self.colors.empty.as_tuple()
