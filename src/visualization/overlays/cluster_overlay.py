import pygame

from src.core.models import GridMap
from src.visualization.overlays.base_overlay import BaseOverlay


class ClusterOverlay(BaseOverlay):

    def __init__(
        self,
        grid_map: GridMap,
        cell_size: int,
        cluster_size: int,
        window_width: int,
        window_height: int,
    ) -> None:
        self.grid_map = grid_map
        self.cell_size = cell_size
        self.cluster_size = cluster_size
        self.window_width = window_width
        self.window_height = window_height

    def draw(
        self,
        screen: pygame.Surface,
    ) -> None:
        color = (255, 0, 255)

        for row in range(0, self.grid_map.height, self.cluster_size):
            y = row * self.cell_size

            pygame.draw.line(
                screen,
                color,
                (0, y),
                (self.window_width, y),
                width=1,
            )

        for col in range(0, self.grid_map.width, self.cluster_size):
            x = col * self.cell_size

            pygame.draw.line(
                screen,
                color,
                (x, 0),
                (x, self.window_height),
                width=1,
            )