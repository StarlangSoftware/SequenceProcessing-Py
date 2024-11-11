import numpy as np

from SequenceProcessing.Initializer.Initializer import Initializer


class UniformXavier(Initializer):
    def initialize(self, rows: int, cols: int, random):
        limit = np.sqrt(6.0 / (rows + cols))
        return np.random.default_rng(random).uniform(-limit, limit, (rows, cols))
