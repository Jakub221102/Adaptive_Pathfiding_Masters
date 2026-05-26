import pygame

from src.core.trace import RawAlgorithmStep
from src.visualization.overlays.base_overlay import BaseOverlay


class HeatmapOverlay(BaseOverlay):
    def __init__(
        self,
        steps: list[RawAlgorithmStep],
        cell_size: int,
        alpha: int = 120,
    ) -> None:
        self.steps = steps
        self.cell_size = cell_size
        self.alpha = alpha
        self.visit_counts = self._build_visit_counts()

    def draw(
            self,
            screen: pygame.Surface,
    ) -> None:
        if not self.visit_counts:
            return

        heatmap_surface = pygame.Surface(
            screen.get_size(),
            pygame.SRCALPHA,
        )

        max_visits = max(self.visit_counts.values())

        for (row, col), count in self.visit_counts.items():
            intensity = count / max_visits
            color = self._get_heatmap_color(intensity)

            rect = pygame.Rect(
                col * self.cell_size,
                row * self.cell_size,
                self.cell_size,
                self.cell_size,
            )

            pygame.draw.rect(
                heatmap_surface,
                (*color, self.alpha),
                rect,
            )

        screen.blit(heatmap_surface, (0, 0))

    def _build_visit_counts(
        self,
    ) -> dict[tuple[int, int], int]:
        visit_counts: dict[tuple[int, int], int] = {}

        for step in self.steps:
            for node in step.closed_nodes:
                visit_counts[node] = visit_counts.get(node, 0) + 1

        return visit_counts

    @staticmethod
    def _get_heatmap_color(
        intensity: float,
    ) -> tuple[int, int, int]:
        red = int(255 * intensity)
        green = int(120 * (1 - intensity))
        blue = 0

        return red, green, blue