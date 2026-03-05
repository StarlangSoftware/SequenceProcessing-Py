# SequenceProcessing/Functions/AdditionByConstant.py

from ComputationalGraph.Function import Function
from Math.Tensor import Tensor


class AdditionByConstant(Function):
    def __init__(self, constant: float):
        self.constant = float(constant)

    def calculate(self, tensor: Tensor) -> Tensor:
        values = [v + self.constant for v in tensor.getData()]
        return Tensor(values, tensor.getShape())

    def derivative(self, tensor: Tensor, tensor1: Tensor) -> Tensor:
        return tensor1