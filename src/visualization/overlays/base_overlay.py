from abc import ABC, abstractmethod
from src.visualization.pygame_models import ViewportRenderContext

import pygame


class BaseOverlay(ABC):

    @abstractmethod
    def draw(
        self,
        screen: pygame.Surface,
        context: ViewportRenderContext | None = None,
    ) -> None:
        pass