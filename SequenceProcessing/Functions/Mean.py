from ComputationalGraph.Function import Function
from Math.Tensor import Tensor


class Mean(Function):

    def calculate(self, tensor: Tensor) -> Tensor:
        values = []
        means = []
        shape = tensor.getShape()

        for i in range(shape[0]):
            total = 0.0
            for j in range(shape[1]):
                total += tensor.getValue((i, j))
            means.append(total / shape[1])

        for i in range(shape[0]):
            for j in range(shape[1]):
                values.append(means[i])

        return Tensor(values, shape)

    def derivative(self, tensor: Tensor, backward: Tensor) -> Tensor:
        shape = tensor.getShape()
        values = []

        for i in range(shape[0]):
            for j in range(shape[1]):
                values.append(1.0 / shape[1])

        return backward.hadamardProduct(Tensor(values, shape))