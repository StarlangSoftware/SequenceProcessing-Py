from abc import ABC, abstractmethod
from Math.Matrix import Matrix


class Initializer(ABC):
    @abstractmethod
    def initialize(self, row: int, col: int, random: int) -> Matrix:
        pass
