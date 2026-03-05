from ComputationalGraph.Function import Function
from Math.Tensor import Tensor


class MultiplyByConstant(Function):

    def __init__(self, constant: float):
        self.constant = float(constant)

    def calculate(self, tensor: Tensor) -> Tensor:
        values = [self.constant * v for v in tensor.getData()]
        return Tensor(values, tensor.getShape())

    def derivative(self, tensor: Tensor, backward: Tensor) -> Tensor:
        shape = tensor.getShape()

        # Create tensor filled with constant
        values = [self.constant] * len(tensor.getData())
        constant_tensor = Tensor(values, shape)

        # Hadamard product with incoming gradient
        return backward.hadamardProduct(constant_tensor)