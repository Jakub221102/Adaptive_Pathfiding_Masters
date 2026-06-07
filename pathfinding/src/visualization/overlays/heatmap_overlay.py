import pygame

from pathfinding.src.core.trace import RawAlgorithmStep
from pathfinding.src.visualization.overlays.base_overlay import BaseOverlay
from pathfinding.src.visualization.pygame_models import ViewportRenderContext


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
        self._step_index = 0

    def set_animation_step_index(self, step_index: int) -> None:
        self._step_index = max(0, min(step_index, len(self.steps) - 1))

    def draw(
            self,
            screen: pygame.Surface,
            context: ViewportRenderContext | None = None,
    ) -> None:
        visit_counts = self._build_visit_counts_up_to(self._step_index)

        if not visit_counts:
            return

        heatmap_surface = pygame.Surface(
            screen.get_size(),
            pygame.SRCALPHA,
        )

        max_visits = max(visit_counts.values())

        x_offset = context.x if context is not None else 0
        y_offset = context.y if context is not None else 0
        cell_size = context.cell_size if context is not None else self.cell_size

        for (row, col), count in visit_counts.items():
            intensity = count / max_visits
            color = self._get_heatmap_color(intensity)

            rect = pygame.Rect(
                x_offset + col * cell_size,
                y_offset + row * cell_size,
                cell_size,
                cell_size,
            )

            pygame.draw.rect(
                heatmap_surface,
                (*color, self.alpha),
                rect,
            )

        screen.blit(heatmap_surface, (0, 0))

    def _build_visit_counts_up_to(
            self,
            step_index: int,
    ) -> dict[tuple[int, int], int]:
        visit_counts: dict[tuple[int, int], int] = {}

        for step in self.steps[: step_index + 1]:
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
