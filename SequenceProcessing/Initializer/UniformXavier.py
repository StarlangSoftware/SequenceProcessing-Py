import numpy as np

class UniformXavier(Initializer):
    def initialize(self, rows, cols, random):
        limit = np.sqrt(6.0 / (rows + cols))
        return np.random.default_rng(random).uniform(-limit, limit, (rows, cols))
