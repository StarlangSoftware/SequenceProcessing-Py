import numpy as np

class RandomInitializer(Initializer):
    def initialize(self, rows, cols, random):
        return np.random.default_rng(random).uniform(-0.01, 0.01, (rows, cols))
