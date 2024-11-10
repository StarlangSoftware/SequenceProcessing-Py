from abc import ABC, abstractmethod
import numpy as np

class Initializer(ABC):
    @abstractmethod
    def initialize(self, rows, cols, random):
        pass
