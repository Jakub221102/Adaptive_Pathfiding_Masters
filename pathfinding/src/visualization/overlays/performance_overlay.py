import pygame

from pathfinding.src.core.models import PathfindingResult
from pathfinding.src.visualization.overlays.base_overlay import BaseOverlay
from pathfinding.src.visualization.pygame_models import ViewportRenderContext


class PerformanceOverlay(BaseOverlay):
    def __init__(
            self,
            result: PathfindingResult,
            x: int = 10,
            y: int = 10,
    ) -> None:
        self.result = result
        self.x = x
        self.y = y

    def draw(
            self,
            screen: pygame.Surface,
            context: ViewportRenderContext | None = None,
    ) -> None:
        font = pygame.font.SysFont("Arial", 16)

        x = self.x
        y = self.y

        if context is not None:
            x = context.x + self.x
            y = context.y + self.y

        lines = [
            f"Algorithm: {self.result.algorithm_name}",
            f"Found: {self.result.found}",
            f"Path length: {self.result.path_length}",
            f"Visited nodes: {self.result.visited_nodes}",
            f"Time: {self.result.execution_time_ms:.3f} ms",
        ]

        padding = 8
        line_height = 20
        box_width = 150
        box_height = padding * 2 + len(lines) * line_height

        background = pygame.Surface((box_width, box_height))
        background.set_alpha(180)
        background.fill((0, 0, 0))

        screen.blit(background, (x, y))

        for index, line in enumerate(lines):
            text_surface = font.render(line, True, (255, 255, 255))
            screen.blit(
                text_surface,
                (
                    x + padding,
                    y + padding + index * line_height,
                ),
            )
