import pygame

from src.core.models import PathfindingResult
from src.visualization.overlays.base_overlay import BaseOverlay


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
    ) -> None:
        font = pygame.font.SysFont("Arial", 16)

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

        screen.blit(background, (self.x, self.y))

        for index, line in enumerate(lines):
            text_surface = font.render(line, True, (255, 255, 255))
            screen.blit(
                text_surface,
                (
                    self.x + padding,
                    self.y + padding + index * line_height,
                ),
            )