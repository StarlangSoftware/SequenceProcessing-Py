from Initializer import Initializer
from Math.Matrix import Matrix
import numpy as np
from typing import Optional


class RandomInitializer(Initializer):
    def initialize(self, row: int, col: int, randomSeed: Optional[int] = None) -> Matrix:
        if randomSeed is not None:
            np.random.seed(randomSeed)

        # Initialize the matrix with values between -0.01 and 0.01
        return Matrix(row, col, -0.01, 0.01, randomSeed)
