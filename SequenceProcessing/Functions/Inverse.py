from ComputationalGraph.Function import Function
from Math.Tensor import Tensor


class Inverse(Function):

    def calculate(self, tensor: Tensor) -> Tensor:
        values = []
        shape = tensor.getShape()

        for i in range(shape[0]):
            for j in range(shape[1]):
                values.append(1.0 / tensor.getValue((i, j)))

        return Tensor(values, shape)

    def derivative(self, tensor: Tensor, backward: Tensor) -> Tensor:
        values = []
        shape = tensor.getShape()

        for i in range(shape[0]):
            for j in range(shape[1]):
                values.append(-(tensor.getValue((i, j)) ** 2))

        return backward.hadamardProduct(Tensor(values, shape))