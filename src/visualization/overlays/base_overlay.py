from abc import ABC, abstractmethod

import pygame


class BaseOverlay(ABC):

    @abstractmethod
    def draw(
        self,
        screen: pygame.Surface,
    ) -> None:
        pass