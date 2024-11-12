from Initializer import Initializer
from Math.Matrix import Matrix
import numpy as np
from typing import Optional


class UniformXavier(Initializer):
    def initialize(self, row: int, col: int, randomSeed: Optional[int] = None) -> Matrix:
        if randomSeed is not None:
            np.random.seed(randomSeed)

        limit = np.sqrt(6.0 / (row + col))

        # Create an empty matrix and populate it with Xavier-initialized values
        matrix = Matrix(row, col)
        for i in range(row):
            for j in range(col):
                matrix.setValue(i, j, np.random.uniform(-limit, limit))

        return matrix
