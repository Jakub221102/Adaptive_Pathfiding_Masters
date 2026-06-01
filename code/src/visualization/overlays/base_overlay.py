from abc import ABC, abstractmethod

import pygame

from code.src.visualization.pygame_models import ViewportRenderContext


class BaseOverlay(ABC):

    @abstractmethod
    def draw(
            self,
            screen: pygame.Surface,
            context: ViewportRenderContext | None = None,
    ) -> None:
        pass
