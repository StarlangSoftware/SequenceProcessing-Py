import numpy as np

from SequenceProcessing.Initializer.Initializer import Initializer


class RandomInitializer(Initializer):
    def initialize(self, rows: int, cols: int, random):
        return np.random.default_rng(random).uniform(-0.01, 0.01, (rows, cols))
